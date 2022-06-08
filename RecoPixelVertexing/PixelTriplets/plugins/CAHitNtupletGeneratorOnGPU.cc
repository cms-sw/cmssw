//
// Original Author: Felice Pantaleo, CERN
//

// #define GPU_DEBUG

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

#include "CAHitNtupletGeneratorOnGPU.h"

namespace {

  template <typename T>
  T sqr(T x) {
    return x * x;
  }

  cAHitNtupletGenerator::QualityCuts makeQualityCuts(edm::ParameterSet const& pset) {
    auto coeff = pset.getParameter<std::vector<double>>("chi2Coeff");
    auto ptMax = pset.getParameter<double>("chi2MaxPt");
    if (coeff.size() != 2) {
      throw edm::Exception(edm::errors::Configuration,
                           "CAHitNtupletGeneratorOnGPU.trackQualityCuts.chi2Coeff must have 2 elements");
    }
    coeff[1] = (coeff[1] - coeff[0]) / log2(ptMax);
    return cAHitNtupletGenerator::QualityCuts{// polynomial coefficients for the pT-dependent chi2 cut
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

}  // namespace

using namespace std;

CAHitNtupletGeneratorOnGPU::CAHitNtupletGeneratorOnGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC)
    : m_params(cfg.getParameter<bool>("onGPU"),
               cfg.getParameter<unsigned int>("minHitsPerNtuplet"),
               cfg.getParameter<unsigned int>("maxNumberOfDoublets"),
               cfg.getParameter<unsigned int>("minHitsForSharingCut"),
               cfg.getParameter<bool>("useRiemannFit"),
               cfg.getParameter<bool>("fitNas4"),
               cfg.getParameter<bool>("includeJumpingForwardDoublets"),
               cfg.getParameter<bool>("earlyFishbone"),
               cfg.getParameter<bool>("lateFishbone"),
               cfg.getParameter<bool>("idealConditions"),
               cfg.getParameter<bool>("fillStatistics"),
               cfg.getParameter<bool>("doClusterCut"),
               cfg.getParameter<bool>("doZ0Cut"),
               cfg.getParameter<bool>("doPtCut"),
               cfg.getParameter<bool>("doSharedHitCut"),
               cfg.getParameter<bool>("dupPassThrough"),
               cfg.getParameter<bool>("useSimpleTripletCleaner"),
               cfg.getParameter<double>("ptmin"),
               cfg.getParameter<double>("CAThetaCutBarrel"),
               cfg.getParameter<double>("CAThetaCutForward"),
               cfg.getParameter<double>("hardCurvCut"),
               cfg.getParameter<double>("dcaCutInnerTriplet"),
               cfg.getParameter<double>("dcaCutOuterTriplet"),
               makeQualityCuts(cfg.getParameterSet("trackQualityCuts"))) {
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

void CAHitNtupletGeneratorOnGPU::fillDescriptions(edm::ParameterSetDescription& desc) {
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
  desc.add<bool>("idealConditions", true);
  desc.add<bool>("fillStatistics", false);
  desc.add<unsigned int>("minHitsPerNtuplet", 4);
  desc.add<unsigned int>("maxNumberOfDoublets", caConstants::maxNumberOfDoublets);
  desc.add<unsigned int>("minHitsForSharingCut", 10)
      ->setComment("Maximum number of hits in a tuple to clean also if the shared hit is on bpx1");
  desc.add<bool>("includeJumpingForwardDoublets", false);
  desc.add<bool>("fitNas4", false)->setComment("fit only 4 hits out of N");
  desc.add<bool>("doClusterCut", true);
  desc.add<bool>("doZ0Cut", true);
  desc.add<bool>("doPtCut", true);
  desc.add<bool>("useRiemannFit", false)->setComment("true for Riemann, false for BrokenLine");
  desc.add<bool>("doSharedHitCut", true)->setComment("Sharing hit nTuples cleaning");
  desc.add<bool>("dupPassThrough", false)->setComment("Do not reject duplicate");
  desc.add<bool>("useSimpleTripletCleaner", true)->setComment("use alternate implementation");

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

void CAHitNtupletGeneratorOnGPU::beginJob() {
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

void CAHitNtupletGeneratorOnGPU::endJob() {
  if (m_params.onGPU_) {
    // print the gpu statistics and free pinned host memory only if CUDA is available
    edm::Service<CUDAService> cs;
    if (cs and cs->enabled()) {
      if (m_params.doStats_) {
        // crash on multi-gpu processes
        CAHitNtupletGeneratorKernelsGPU::printCounters(m_counters);
      }
      cudaFree(m_counters);
    }
  } else {
    if (m_params.doStats_) {
      CAHitNtupletGeneratorKernelsCPU::printCounters(m_counters);
    }
    delete m_counters;
  }
}

PixelTrackHeterogeneous CAHitNtupletGeneratorOnGPU::makeTuplesAsync(TrackingRecHit2DGPU const& hits_d,
                                                                    float bfield,
                                                                    cudaStream_t stream) const {
  PixelTrackHeterogeneous tracks(cms::cuda::make_device_unique<pixelTrack::TrackSoA>(stream));

  auto* soa = tracks.get();
  assert(soa);

  CAHitNtupletGeneratorKernelsGPU kernels(m_params);
  kernels.setCounters(m_counters);
  kernels.allocateOnGPU(hits_d.nHits(), stream);

  kernels.buildDoublets(hits_d, stream);
  kernels.launchKernels(hits_d, soa, stream);

  HelixFitOnGPU fitter(bfield, m_params.fitNas4_);
  fitter.allocateOnGPU(&(soa->hitIndices), kernels.tupleMultiplicity(), soa);
  if (m_params.useRiemannFit_) {
    fitter.launchRiemannKernels(hits_d.view(), hits_d.nHits(), caConstants::maxNumberOfQuadruplets, stream);
  } else {
    fitter.launchBrokenLineKernels(hits_d.view(), hits_d.nHits(), caConstants::maxNumberOfQuadruplets, stream);
  }
  kernels.classifyTuples(hits_d, soa, stream);

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
  std::cout << "finished building pixel tracks on GPU" << std::endl;
#endif

  return tracks;
}

PixelTrackHeterogeneous CAHitNtupletGeneratorOnGPU::makeTuples(TrackingRecHit2DCPU const& hits_d, float bfield) const {
  PixelTrackHeterogeneous tracks(std::make_unique<pixelTrack::TrackSoA>());

  auto* soa = tracks.get();
  assert(soa);

  CAHitNtupletGeneratorKernelsCPU kernels(m_params);
  kernels.setCounters(m_counters);
  kernels.allocateOnGPU(hits_d.nHits(), nullptr);

  kernels.buildDoublets(hits_d, nullptr);
  kernels.launchKernels(hits_d, soa, nullptr);

  if (0 == hits_d.nHits())
    return tracks;

  // now fit
  HelixFitOnGPU fitter(bfield, m_params.fitNas4_);
  fitter.allocateOnGPU(&(soa->hitIndices), kernels.tupleMultiplicity(), soa);

  if (m_params.useRiemannFit_) {
    fitter.launchRiemannKernelsOnCPU(hits_d.view(), hits_d.nHits(), caConstants::maxNumberOfQuadruplets);
  } else {
    fitter.launchBrokenLineKernelsOnCPU(hits_d.view(), hits_d.nHits(), caConstants::maxNumberOfQuadruplets);
  }

  kernels.classifyTuples(hits_d, soa, nullptr);

#ifdef GPU_DEBUG
  std::cout << "finished building pixel tracks on CPU" << std::endl;
#endif

  return tracks;
}
