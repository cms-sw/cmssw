//
// Original Author: Felice Pantaleo, CERN
//

// #define GPU_DEBUG
// #define DUMP_GPU_TK_TUPLES

#include <array>
#include <cassert>
#include <functional>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousDevice.h"

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoAHost.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHitSoADevice.h"

#include "CAHitNtupletGeneratorOnGPU.h"

namespace {

  using namespace caHitNtupletGenerator;
  using namespace gpuPixelDoublets;
  using namespace pixelTopology;
  using namespace pixelTrack;

  template <typename T>
  T sqr(T x) {
    return x * x;
  }

  //Common Params
  AlgoParams makeCommonParams(edm::ParameterSet const& cfg) {
    return AlgoParams({cfg.getParameter<bool>("onGPU"),
                       cfg.getParameter<unsigned int>("minHitsForSharingCut"),
                       cfg.getParameter<bool>("useRiemannFit"),
                       cfg.getParameter<bool>("fitNas4"),
                       cfg.getParameter<bool>("includeJumpingForwardDoublets"),
                       cfg.getParameter<bool>("earlyFishbone"),
                       cfg.getParameter<bool>("lateFishbone"),
                       cfg.getParameter<bool>("fillStatistics"),
                       cfg.getParameter<bool>("doSharedHitCut"),
                       cfg.getParameter<bool>("dupPassThrough"),
                       cfg.getParameter<bool>("useSimpleTripletCleaner")});
  }

  //This is needed to have the partial specialization for  isPhase1Topology/isPhase2Topology
  template <typename TrackerTraits, typename Enable = void>
  struct topologyCuts {};

  template <typename TrackerTraits>
  struct topologyCuts<TrackerTraits, isPhase1Topology<TrackerTraits>> {
    static constexpr CAParamsT<TrackerTraits> makeCACuts(edm::ParameterSet const& cfg) {
      return CAParamsT<TrackerTraits>{{cfg.getParameter<unsigned int>("minHitsPerNtuplet"),
                                       (float)cfg.getParameter<double>("ptmin"),
                                       (float)cfg.getParameter<double>("CAThetaCutBarrel"),
                                       (float)cfg.getParameter<double>("CAThetaCutForward"),
                                       (float)cfg.getParameter<double>("hardCurvCut"),
                                       (float)cfg.getParameter<double>("dcaCutInnerTriplet"),
                                       (float)cfg.getParameter<double>("dcaCutOuterTriplet")}};
    };

    static constexpr pixelTrack::QualityCutsT<TrackerTraits> makeQualityCuts(edm::ParameterSet const& pset) {
      auto coeff = pset.getParameter<std::array<double, 2>>("chi2Coeff");
      auto ptMax = pset.getParameter<double>("chi2MaxPt");

      coeff[1] = (coeff[1] - coeff[0]) / log2(ptMax);
      return pixelTrack::QualityCutsT<TrackerTraits>{// polynomial coefficients for the pT-dependent chi2 cut
                                                     {(float)coeff[0], (float)coeff[1], 0.f, 0.f},
                                                     // max pT used to determine the chi2 cut
                                                     (float)ptMax,
                                                     // chi2 scale factor: 8 for broken line fit, ?? for Riemann fit
                                                     (float)pset.getParameter<double>("chi2Scale"),
                                                     // regional cuts for triplets
                                                     {(float)pset.getParameter<double>("tripletMaxTip"),
                                                      (float)pset.getParameter<double>("tripletMinPt"),
                                                      (float)pset.getParameter<double>("tripletMaxZip")},
                                                     // regional cuts for quadruplets
                                                     {(float)pset.getParameter<double>("quadrupletMaxTip"),
                                                      (float)pset.getParameter<double>("quadrupletMinPt"),
                                                      (float)pset.getParameter<double>("quadrupletMaxZip")}};
    }
  };

  template <typename TrackerTraits>
  struct topologyCuts<TrackerTraits, isPhase2Topology<TrackerTraits>> {
    static constexpr CAParamsT<TrackerTraits> makeCACuts(edm::ParameterSet const& cfg) {
      return CAParamsT<TrackerTraits>{{cfg.getParameter<unsigned int>("minHitsPerNtuplet"),
                                       (float)cfg.getParameter<double>("ptmin"),
                                       (float)cfg.getParameter<double>("CAThetaCutBarrel"),
                                       (float)cfg.getParameter<double>("CAThetaCutForward"),
                                       (float)cfg.getParameter<double>("hardCurvCut"),
                                       (float)cfg.getParameter<double>("dcaCutInnerTriplet"),
                                       (float)cfg.getParameter<double>("dcaCutOuterTriplet")},
                                      {(bool)cfg.getParameter<bool>("includeFarForwards")}};
    }

    static constexpr pixelTrack::QualityCutsT<TrackerTraits> makeQualityCuts(edm::ParameterSet const& pset) {
      return pixelTrack::QualityCutsT<TrackerTraits>{
          (float)pset.getParameter<double>("maxChi2"),
          (float)pset.getParameter<double>("minPt"),
          (float)pset.getParameter<double>("maxTip"),
          (float)pset.getParameter<double>("maxZip"),
      };
    }
  };

  //Cell Cuts, as they are the cuts have the same logic for Phase2 and Phase1
  //keeping them separate would allow further differentiation in the future
  //moving them to topologyCuts and using the same syntax
  template <typename TrakterTraits>
  CellCutsT<TrakterTraits> makeCellCuts(edm::ParameterSet const& cfg) {
    return CellCutsT<TrakterTraits>{cfg.getParameter<unsigned int>("maxNumberOfDoublets"),
                                    cfg.getParameter<bool>("doClusterCut"),
                                    cfg.getParameter<bool>("doZ0Cut"),
                                    cfg.getParameter<bool>("doPtCut"),
                                    cfg.getParameter<bool>("idealConditions")};
  }

}  // namespace

using namespace std;

template <typename TrackerTraits>
CAHitNtupletGeneratorOnGPU<TrackerTraits>::CAHitNtupletGeneratorOnGPU(const edm::ParameterSet& cfg,
                                                                      edm::ConsumesCollector& iC)
    : m_params(makeCommonParams(cfg),
               makeCellCuts<TrackerTraits>(cfg),
               topologyCuts<TrackerTraits>::makeQualityCuts(cfg.getParameterSet("trackQualityCuts")),
               topologyCuts<TrackerTraits>::makeCACuts(cfg)) {
#ifdef DUMP_GPU_TK_TUPLES
  printf("TK: %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n",
         "tid",
         "qual",
         "nh",
         "nl",
         "charge",
         "pt",
         "eta",
         "phi",
         "tip",
         "zip",
         "chi2",
         "h1",
         "h2",
         "h3",
         "h4",
         "h5",
         "hn");
#endif
}

template <typename TrackerTraits>
void CAHitNtupletGeneratorOnGPU<TrackerTraits>::fillDescriptions(edm::ParameterSetDescription& desc) {
  fillDescriptionsCommon(desc);
  edm::LogWarning("CAHitNtupletGeneratorOnGPU::fillDescriptions")
      << "Note: this fillDescriptions is a dummy one. Most probably you are missing some parameters. \n"
         "please implement your TrackerTraits descriptions in CAHitNtupletGeneratorOnGPU. \n";
}

template <>
void CAHitNtupletGeneratorOnGPU<pixelTopology::Phase1>::fillDescriptions(edm::ParameterSetDescription& desc) {
  fillDescriptionsCommon(desc);

  desc.add<bool>("idealConditions", true);
  desc.add<bool>("includeJumpingForwardDoublets", false);

  edm::ParameterSetDescription trackQualityCuts;
  trackQualityCuts.add<double>("chi2MaxPt", 10.)->setComment("max pT used to determine the pT-dependent chi2 cut");
  trackQualityCuts.add<std::vector<double>>("chi2Coeff", {0.9, 1.8})->setComment("chi2 at 1GeV and at ptMax above");
  trackQualityCuts.add<double>("chi2Scale", 8.)
      ->setComment(
          "Factor to multiply the pT-dependent chi2 cut (currently: 8 for the broken line fit, ?? for the Riemann "
          "fit)");
  trackQualityCuts.add<double>("tripletMinPt", 0.5)->setComment("Min pT for triplets, in GeV");
  trackQualityCuts.add<double>("tripletMaxTip", 0.3)->setComment("Max |Tip| for triplets, in cm");
  trackQualityCuts.add<double>("tripletMaxZip", 12.)->setComment("Max |Zip| for triplets, in cm");
  trackQualityCuts.add<double>("quadrupletMinPt", 0.3)->setComment("Min pT for quadruplets, in GeV");
  trackQualityCuts.add<double>("quadrupletMaxTip", 0.5)->setComment("Max |Tip| for quadruplets, in cm");
  trackQualityCuts.add<double>("quadrupletMaxZip", 12.)->setComment("Max |Zip| for quadruplets, in cm");
  desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
      ->setComment(
          "Quality cuts based on the results of the track fit:\n  - apply a pT-dependent chi2 cut;\n  - apply \"region "
          "cuts\" based on the fit results (pT, Tip, Zip).");
}

template <>
void CAHitNtupletGeneratorOnGPU<pixelTopology::Phase2>::fillDescriptions(edm::ParameterSetDescription& desc) {
  fillDescriptionsCommon(desc);

  desc.add<bool>("idealConditions", false);
  desc.add<bool>("includeFarForwards", true);
  desc.add<bool>("includeJumpingForwardDoublets", true);

  edm::ParameterSetDescription trackQualityCuts;
  trackQualityCuts.add<double>("maxChi2", 5.)->setComment("Max normalized chi2");
  trackQualityCuts.add<double>("minPt", 0.5)->setComment("Min pT in GeV");
  trackQualityCuts.add<double>("maxTip", 0.3)->setComment("Max |Tip| in cm");
  trackQualityCuts.add<double>("maxZip", 12.)->setComment("Max |Zip|, in cm");
  desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
      ->setComment(
          "Quality cuts based on the results of the track fit:\n  - apply cuts based on the fit results (pT, Tip, "
          "Zip).");
}

template <typename TrackerTraits>
void CAHitNtupletGeneratorOnGPU<TrackerTraits>::fillDescriptionsCommon(edm::ParameterSetDescription& desc) {
  // 87 cm/GeV = 1/(3.8T * 0.3)
  // take less than radius given by the hardPtCut and reject everything below
  // auto hardCurvCut = 1.f/(0.35 * 87.f);
  desc.add<double>("ptmin", 0.9f)->setComment("Cut on minimum pt");
  desc.add<double>("CAThetaCutBarrel", 0.002f)->setComment("Cut on RZ alignement for Barrel");
  desc.add<double>("CAThetaCutForward", 0.003f)->setComment("Cut on RZ alignment for Forward");
  desc.add<double>("hardCurvCut", 1.f / (0.35 * 87.f))->setComment("Cut on minimum curvature");
  desc.add<double>("dcaCutInnerTriplet", 0.15f)->setComment("Cut on origin radius when the inner hit is on BPix1");
  desc.add<double>("dcaCutOuterTriplet", 0.25f)->setComment("Cut on origin radius when the outer hit is on BPix1");
  desc.add<bool>("earlyFishbone", true);
  desc.add<bool>("lateFishbone", false);
  desc.add<bool>("fillStatistics", false);
  desc.add<unsigned int>("minHitsPerNtuplet", 4);
  desc.add<unsigned int>("maxNumberOfDoublets", TrackerTraits::maxNumberOfDoublets);
  desc.add<unsigned int>("minHitsForSharingCut", 10)
      ->setComment("Maximum number of hits in a tuple to clean also if the shared hit is on bpx1");

  desc.add<bool>("fitNas4", false)->setComment("fit only 4 hits out of N");
  desc.add<bool>("doClusterCut", true);
  desc.add<bool>("doZ0Cut", true);
  desc.add<bool>("doPtCut", true);
  desc.add<bool>("useRiemannFit", false)->setComment("true for Riemann, false for BrokenLine");
  desc.add<bool>("doSharedHitCut", true)->setComment("Sharing hit nTuples cleaning");
  desc.add<bool>("dupPassThrough", false)->setComment("Do not reject duplicate");
  desc.add<bool>("useSimpleTripletCleaner", true)->setComment("use alternate implementation");
}

template <typename TrackerTraits>
void CAHitNtupletGeneratorOnGPU<TrackerTraits>::beginJob() {
  if (m_params.onGPU_) {
    // allocate pinned host memory only if CUDA is available
    edm::Service<CUDAService> cs;
    if (cs and cs->enabled()) {
      cudaCheck(cudaMalloc(&m_counters, sizeof(Counters)));
      cudaCheck(cudaMemset(m_counters, 0, sizeof(Counters)));
    }
  } else {
    m_counters = new Counters();
    memset(m_counters, 0, sizeof(Counters));
  }
}

template <typename TrackerTraits>
void CAHitNtupletGeneratorOnGPU<TrackerTraits>::endJob() {
  if (m_params.onGPU_) {
    // print the gpu statistics and free pinned host memory only if CUDA is available
    edm::Service<CUDAService> cs;
    if (cs and cs->enabled()) {
      if (m_params.doStats_) {
        // crash on multi-gpu processes
        CAHitNtupletGeneratorKernelsGPU<TrackerTraits>::printCounters(m_counters);
      }
      cudaFree(m_counters);
    }
  } else {
    if (m_params.doStats_) {
      CAHitNtupletGeneratorKernelsCPU<TrackerTraits>::printCounters(m_counters);
    }
    delete m_counters;
  }
}

template <typename TrackerTraits>
TrackSoAHeterogeneousDevice<TrackerTraits> CAHitNtupletGeneratorOnGPU<TrackerTraits>::makeTuplesAsync(
    HitsOnGPU const& hits_d, float bfield, cudaStream_t stream) const {
  using HelixFitOnGPU = HelixFitOnGPU<TrackerTraits>;
  using TrackSoA = TrackSoAHeterogeneousDevice<TrackerTraits>;
  using GPUKernels = CAHitNtupletGeneratorKernelsGPU<TrackerTraits>;

  TrackSoA tracks(stream);

  GPUKernels kernels(m_params);
  kernels.setCounters(m_counters);
  kernels.allocateOnGPU(hits_d.nHits(), stream);

  kernels.buildDoublets(hits_d.view(), hits_d.offsetBPIX2(), stream);

  kernels.launchKernels(hits_d.view(), tracks.view(), stream);

  HelixFitOnGPU fitter(bfield, m_params.fitNas4_);
  fitter.allocateOnGPU(kernels.tupleMultiplicity(), tracks.view());
  if (m_params.useRiemannFit_) {
    fitter.launchRiemannKernels(hits_d.view(), hits_d.nHits(), TrackerTraits::maxNumberOfQuadruplets, stream);
  } else {
    fitter.launchBrokenLineKernels(hits_d.view(), hits_d.nHits(), TrackerTraits::maxNumberOfQuadruplets, stream);
  }
  kernels.classifyTuples(hits_d.view(), tracks.view(), stream);
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
  std::cout << "finished building pixel tracks on GPU" << std::endl;
#endif

  return tracks;
}

template <typename TrackerTraits>
TrackSoAHeterogeneousHost<TrackerTraits> CAHitNtupletGeneratorOnGPU<TrackerTraits>::makeTuples(HitsOnCPU const& hits_h,
                                                                                               float bfield) const {
  using HelixFitOnGPU = HelixFitOnGPU<TrackerTraits>;
  using TrackSoA = TrackSoAHeterogeneousHost<TrackerTraits>;
  using CPUKernels = CAHitNtupletGeneratorKernelsCPU<TrackerTraits>;

  TrackSoA tracks;

  CPUKernels kernels(m_params);
  kernels.setCounters(m_counters);
  kernels.allocateOnGPU(hits_h.nHits(), nullptr);

  kernels.buildDoublets(hits_h.view(), hits_h.offsetBPIX2(), nullptr);
  kernels.launchKernels(hits_h.view(), tracks.view(), nullptr);

  if (0 == hits_h.nHits())
    return tracks;

  // now fit
  HelixFitOnGPU fitter(bfield, m_params.fitNas4_);
  fitter.allocateOnGPU(kernels.tupleMultiplicity(), tracks.view());

  if (m_params.useRiemannFit_) {
    fitter.launchRiemannKernelsOnCPU(hits_h.view(), hits_h.nHits(), TrackerTraits::maxNumberOfQuadruplets);
  } else {
    fitter.launchBrokenLineKernelsOnCPU(hits_h.view(), hits_h.nHits(), TrackerTraits::maxNumberOfQuadruplets);
  }

  kernels.classifyTuples(hits_h.view(), tracks.view(), nullptr);

#ifdef GPU_DEBUG
  std::cout << "finished building pixel tracks on CPU" << std::endl;
#endif

  // check that the fixed-size SoA does not overflow
  auto maxTracks = tracks.view().metadata().size();
  auto nTracks = tracks.view().nTracks();
  assert(nTracks < maxTracks);
  if (nTracks == maxTracks - 1) {
    edm::LogWarning("PixelTracks") << "Unsorted reconstructed pixel tracks truncated to " << maxTracks - 1
                                   << " candidates";
  }

  return tracks;
}

template class CAHitNtupletGeneratorOnGPU<pixelTopology::Phase1>;
template class CAHitNtupletGeneratorOnGPU<pixelTopology::Phase2>;
