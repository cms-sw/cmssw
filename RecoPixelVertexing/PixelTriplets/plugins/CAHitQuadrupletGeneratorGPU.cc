//
// Author: Felice Pantaleo, CERN
//

#include <array>
#include <cassert>
#include <functional>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "CAHitQuadrupletGeneratorGPU.h"

namespace {

  template <typename T>
  T sqr(T x) {
    return x * x;
  }

  CAHitQuadrupletGeneratorKernels::QualityCuts makeQualityCuts(edm::ParameterSet const& pset) {
    auto coeff = pset.getParameter<std::vector<double>>("chi2Coeff");
    if (coeff.size() != 4) {
      throw edm::Exception(edm::errors::Configuration, "CAHitQuadrupletGeneratorGPU.trackQualityCuts.chi2Coeff must have 4 elements");
    }
    return CAHitQuadrupletGeneratorKernels::QualityCuts {
      // polynomial coefficients for the pT-dependent chi2 cut
      { (float) coeff[0], (float) coeff[1], (float) coeff[2], (float) coeff[3] },
      // max pT used to determine the chi2 cut
      (float) pset.getParameter<double>("chi2MaxPt"),
      // chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
      (float) pset.getParameter<double>("chi2Scale"),
      // regional cuts for triplets
      {
        (float) pset.getParameter<double>("tripletMaxTip"),
        (float) pset.getParameter<double>("tripletMinPt"),
        (float) pset.getParameter<double>("tripletMaxZip")
      },
      // regional cuts for quadruplets
      {
        (float) pset.getParameter<double>("quadrupletMaxTip"),
        (float) pset.getParameter<double>("quadrupletMinPt"),
        (float) pset.getParameter<double>("quadrupletMaxZip")
      }
    };
  }

}  // namespace

using namespace std;

constexpr unsigned int CAHitQuadrupletGeneratorGPU::minLayers;

CAHitQuadrupletGeneratorGPU::CAHitQuadrupletGeneratorGPU(const edm::ParameterSet &cfg, edm::ConsumesCollector &iC)
    : kernels(cfg.getParameter<unsigned int>("minHitsPerNtuplet"),
              cfg.getParameter<bool>("earlyFishbone"),
              cfg.getParameter<bool>("lateFishbone"),
              cfg.getParameter<bool>("idealConditions"),
              cfg.getParameter<bool>("fillStatistics"),
              cfg.getParameter<bool>("doClusterCut"),
              cfg.getParameter<bool>("doZCut"),
              cfg.getParameter<bool>("doPhiCut"),
              cfg.getParameter<double>("ptmin"),
              cfg.getParameter<double>("CAThetaCutBarrel"),
              cfg.getParameter<double>("CAThetaCutForward"),
              cfg.getParameter<double>("hardCurvCut"),
              cfg.getParameter<double>("dcaCutInnerTriplet"),
              cfg.getParameter<double>("dcaCutOuterTriplet"),
              makeQualityCuts(cfg.getParameterSet("trackQualityCuts"))),
      fitter(cfg.getParameter<bool>("fit5as4")),
      caThetaCut(cfg.getParameter<double>("CAThetaCut")),
      caPhiCut(cfg.getParameter<double>("CAPhiCut")),
      caHardPtCut(cfg.getParameter<double>("CAHardPtCut")) {}

void CAHitQuadrupletGeneratorGPU::fillDescriptions(edm::ParameterSetDescription &desc) {
  desc.add<double>("CAThetaCut", 0.00125);
  desc.add<double>("CAPhiCut", 10);
  desc.add<double>("CAHardPtCut", 0);
  // 87 cm/GeV = 1/(3.8T * 0.3)
  // take less than radius given by the hardPtCut and reject everything below
  // auto hardCurvCut = 1.f/(0.35 * 87.f);
  desc.add<double>("ptmin", 0.9f)->setComment("Cut on minimum pt");
  desc.add<double>("CAThetaCutBarrel", 0.002f)->setComment("Cut on RZ alignement for Barrel");
  desc.add<double>("CAThetaCutForward", 0.003f)->setComment("Cut on RZ alignment for Forward");
  desc.add<double>("hardCurvCut", 1.f / (0.35 * 87.f))->setComment("Cut on minimum curvature");
  desc.add<double>("dcaCutInnerTriplet", 0.15f)->setComment("Cut on origin radius when the inner hit is on BPix1");
  desc.add<double>("dcaCutOuterTriplet", 0.25f)->setComment("Cut on origin radius when the outer hit is on BPix1");
  desc.add<bool>("earlyFishbone", false);
  desc.add<bool>("lateFishbone", true);
  desc.add<bool>("idealConditions", true);
  desc.add<bool>("fillStatistics", false);
  desc.add<unsigned int>("minHitsPerNtuplet", 4);
  desc.add<bool>("fit5as4", true);
  desc.add<bool>("doClusterCut", true);
  desc.add<bool>("doZCut", true);
  desc.add<bool>("doPhiCut", true);

  edm::ParameterSetDescription trackQualityCuts;
  trackQualityCuts.add<double>("chi2MaxPt", 10.)->setComment("max pT used to determine the pT-dependent chi2 cut");
  trackQualityCuts.add<std::vector<double>>("chi2Coeff", { 0.68177776, 0.74609577, -0.08035491, 0.00315399 })
    ->setComment("Polynomial coefficients to derive the pT-dependent chi2 cut");
  trackQualityCuts.add<double>("chi2Scale", 30.)->setComment("Factor to multiply the pT-dependent chi2 cut (currently: 30 for the broken line fit, 45 for the Riemann fit)");
  trackQualityCuts.add<double>("tripletMinPt", 0.5)->setComment("Min pT for triplets, in GeV");
  trackQualityCuts.add<double>("tripletMaxTip", 0.3)->setComment("Max |Tip| for triplets, in cm");
  trackQualityCuts.add<double>("tripletMaxZip", 12.)->setComment("Max |Zip| for triplets, in cm");
  trackQualityCuts.add<double>("quadrupletMinPt", 0.3)->setComment("Min pT for quadruplets, in GeV");
  trackQualityCuts.add<double>("quadrupletMaxTip", 0.5)->setComment("Max |Tip| for quadruplets, in cm");
  trackQualityCuts.add<double>("quadrupletMaxZip", 12.)->setComment("Max |Zip| for quadruplets, in cm");
  desc.add<edm::ParameterSetDescription>("trackQualityCuts", trackQualityCuts)
    ->setComment("Quality cuts based on the results of the track fit:\n  - apply a pT-dependent chi2 cut;\n  - apply \"region cuts\" based on the fit results (pT, Tip, Zip).");
}

void CAHitQuadrupletGeneratorGPU::initEvent(edm::Event const &ev, edm::EventSetup const &es) {
  fitter.setBField(1 / PixelRecoUtilities::fieldInInvGev(es));
}

CAHitQuadrupletGeneratorGPU::~CAHitQuadrupletGeneratorGPU() { deallocateOnGPU(); }

void CAHitQuadrupletGeneratorGPU::hitNtuplets(HitsOnCPU const &hh,
                                              edm::EventSetup const &es,
                                              bool useRiemannFit,
                                              bool transferToCPU,
                                              cuda::stream_t<> &cudaStream) {
  hitsOnCPU = &hh;
  launchKernels(hh, useRiemannFit, transferToCPU, cudaStream);
}

void CAHitQuadrupletGeneratorGPU::fillResults(const TrackingRegion &region,
                                              SiPixelRecHitCollectionNew const &rechits,
                                              std::vector<OrderedHitSeeds> &result,
                                              const edm::EventSetup &es) {
  assert(hitsOnCPU);
  auto nhits = hitsOnCPU->nHits();

  uint32_t hitsModuleStart[gpuClustering::MaxNumModules + 1];
  // to be understood where to locate
  cudaCheck(cudaMemcpy(hitsModuleStart,
                       hitsOnCPU->hitsModuleStart(),
                       (gpuClustering::MaxNumModules + 1) * sizeof(uint32_t),
                       cudaMemcpyDefault));

  auto fc = hitsModuleStart;

  hitmap_.clear();
  auto const &rcs = rechits.data();
  hitmap_.resize(nhits);
  for (auto const &h : rcs) {
    auto const &thit = static_cast<BaseTrackerRecHit const &>(h);
    auto detI = thit.det()->index();
    auto const &clus = thit.firstClusterRef();
    assert(clus.isPixel());
    auto i = fc[detI] + clus.pixelCluster().originalId();
    assert(i < nhits);
    hitmap_[i] = &h;
  }

  int index = 0;

  auto const &foundQuads = fetchKernelResult(index);
  unsigned int numberOfFoundQuadruplets = foundQuads.size();

  std::array<BaseTrackerRecHit const *, 4> phits;

  indToEdm.clear();
  indToEdm.resize(numberOfFoundQuadruplets, 64000);

  int nbad = 0;
  // loop over quadruplets
  for (unsigned int quadId = 0; quadId < numberOfFoundQuadruplets; ++quadId) {
    bool bad = pixelTuplesHeterogeneousProduct::bad == quality_[quadId];
    for (unsigned int i = 0; i < 4; ++i) {
      auto k = foundQuads[quadId][i];
      if (k < 0) {
        phits[i] = nullptr;
        continue;
      }  // (actually break...)
      assert(k < int(nhits));
      auto hp = hitmap_[k];
      if (hp == nullptr) {
        edm::LogWarning("CAHitQuadrupletGeneratorGPU") << "hit not found????  " << k << std::endl;
        bad = true;
        break;
      }
      phits[i] = static_cast<BaseTrackerRecHit const *>(hp);
    }
    if (bad) {
      nbad++;
      quality_[quadId] = pixelTuplesHeterogeneousProduct::bad;
      continue;
    }
    if (quality_[quadId] != pixelTuplesHeterogeneousProduct::loose)
      continue;  // FIXME remove dup

    result[index].emplace_back(phits[0], phits[1], phits[2], phits[3]);
    indToEdm[quadId] = result[index].size() - 1;
  }  // end loop over quads

#ifdef GPU_DEBUG
  std::cout << "Q Final quads " << result[index].size() << ' ' << nbad << std::endl;
#endif
}

void CAHitQuadrupletGeneratorGPU::deallocateOnGPU() {
  //product
  cudaFree(gpu_.tuples_d);
  cudaFree(gpu_.helix_fit_results_d);
  cudaFree(gpu_.quality_d);
  cudaFree(gpu_.apc_d);
  cudaFree(gpu_d);
  cudaFreeHost(tuples_);
  cudaFreeHost(hitDetIndices_);
  cudaFreeHost(helix_fit_results_);
  cudaFreeHost(quality_);
}

void CAHitQuadrupletGeneratorGPU::allocateOnGPU() {
  constexpr auto maxNumberOfQuadruplets_ = CAConstants::maxNumberOfQuadruplets();

  // allocate and initialise the GPU memory
  cudaCheck(cudaMalloc(&gpu_.tuples_d, sizeof(TuplesOnGPU::Container)));
  cudaCheck(cudaMemset(gpu_.tuples_d, 0x00, sizeof(TuplesOnGPU::Container)));
  cudaCheck(cudaMalloc(&gpu_.apc_d, sizeof(AtomicPairCounter)));
  cudaCheck(cudaMemset(gpu_.apc_d, 0x00, sizeof(AtomicPairCounter)));
  cudaCheck(cudaMalloc(&gpu_.helix_fit_results_d, sizeof(Rfit::helix_fit) * maxNumberOfQuadruplets_));
  cudaCheck(cudaMemset(gpu_.helix_fit_results_d, 0x00, sizeof(Rfit::helix_fit) * maxNumberOfQuadruplets_));
  cudaCheck(cudaMalloc(&gpu_.quality_d, sizeof(Quality) * maxNumberOfQuadruplets_));
  cudaCheck(cudaMemset(gpu_.quality_d, 0x00, sizeof(Quality) * maxNumberOfQuadruplets_));

  cudaCheck(cudaMalloc(&gpu_d, sizeof(TuplesOnGPU)));
  gpu_.me_d = gpu_d;
  cudaCheck(cudaMemcpy(gpu_d, &gpu_, sizeof(TuplesOnGPU), cudaMemcpyDefault));

  cudaCheck(cudaMallocHost(&tuples_, sizeof(TuplesOnGPU::Container)));
  cudaCheck(cudaMallocHost(&hitDetIndices_, sizeof(TuplesOnGPU::Container)));
  cudaCheck(cudaMallocHost(&helix_fit_results_, sizeof(Rfit::helix_fit) * maxNumberOfQuadruplets_));
  cudaCheck(cudaMallocHost(&quality_, sizeof(Quality) * maxNumberOfQuadruplets_));

  kernels.allocateOnGPU();
  fitter.allocateOnGPU(gpu_.tuples_d, kernels.tupleMultiplicity(), gpu_.helix_fit_results_d);
}

void CAHitQuadrupletGeneratorGPU::launchKernels(HitsOnCPU const &hh,
                                                bool useRiemannFit,
                                                bool transferToCPU,
                                                cuda::stream_t<> &cudaStream) {
  kernels.launchKernels(hh, gpu_, cudaStream.id());
  if (useRiemannFit) {
    fitter.launchRiemannKernels(hh, hh.nHits(), CAConstants::maxNumberOfQuadruplets(), cudaStream);
  } else {
    fitter.launchBrokenLineKernels(hh, hh.nHits(), CAConstants::maxNumberOfQuadruplets(), cudaStream);
  }
  kernels.classifyTuples(hh, gpu_, cudaStream.id());

  if (transferToCPU) {
    cudaCheck(cudaMemcpyAsync(
        tuples_, gpu_.tuples_d, sizeof(TuplesOnGPU::Container), cudaMemcpyDeviceToHost, cudaStream.id()));

    kernels.fillHitDetIndices(hh, gpu_, hitDetIndices_, cudaStream);

    cudaCheck(cudaMemcpyAsync(helix_fit_results_,
                              gpu_.helix_fit_results_d,
                              sizeof(Rfit::helix_fit) * CAConstants::maxNumberOfQuadruplets(),
                              cudaMemcpyDeviceToHost,
                              cudaStream.id()));

    cudaCheck(cudaMemcpyAsync(quality_,
                              gpu_.quality_d,
                              sizeof(Quality) * CAConstants::maxNumberOfQuadruplets(),
                              cudaMemcpyDeviceToHost,
                              cudaStream.id()));
  }
}

void CAHitQuadrupletGeneratorGPU::cleanup(cudaStream_t cudaStream) { kernels.cleanup(cudaStream); }

std::vector<std::array<int, 4>> CAHitQuadrupletGeneratorGPU::fetchKernelResult(int) {
  assert(tuples_);
  auto const &tuples = *tuples_;

  uint32_t sizes[7] = {0};
  std::vector<int> ntk(10000);
  auto add = [&](uint32_t hi) {
    if (hi >= ntk.size())
      ntk.resize(hi + 1);
    ++ntk[hi];
  };

  std::vector<std::array<int, 4>> quadsInterface;
  quadsInterface.reserve(10000);

  nTuples_ = 0;
  for (auto i = 0U; i < tuples.nbins(); ++i) {
    auto sz = tuples.size(i);
    if (sz == 0)
      break;  // we know cannot be less then 3
    ++nTuples_;
    ++sizes[sz];
    for (auto j = tuples.begin(i); j != tuples.end(i); ++j)
      add(*j);
    quadsInterface.emplace_back(std::array<int, 4>());
    quadsInterface.back()[0] = tuples.begin(i)[0];
    quadsInterface.back()[1] = tuples.begin(i)[1];
    quadsInterface.back()[2] = tuples.begin(i)[2];                // [sz-2];
    quadsInterface.back()[3] = sz > 3 ? tuples.begin(i)[3] : -1;  // [sz-1];
  }

#ifdef GPU_DEBUG
  long long ave = 0;
  int nn = 0;
  for (auto k : ntk)
    if (k > 0) {
      ave += k;
      ++nn;
    }
  std::cout << "Q Produced " << quadsInterface.size() << " quadruplets: ";
  for (auto i = 3; i < 7; ++i)
    std::cout << sizes[i] << ' ';
  std::cout << "max/ave " << *std::max_element(ntk.begin(), ntk.end()) << '/' << float(ave) / float(nn) << std::endl;
#endif
  return quadsInterface;
}

void CAHitQuadrupletGeneratorGPU::buildDoublets(HitsOnCPU const &hh, cuda::stream_t<> &stream) {
  kernels.buildDoublets(hh, stream);
}
