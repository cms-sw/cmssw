#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorGPU_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorGPU_h

#include <cuda_runtime.h>

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUSimpleVector.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RZLine.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/RecHitsMap.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkSeedGenerator/interface/FastCircleFit.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"

#include "CAHitQuadrupletGeneratorKernels.h"
#include "HelixFitOnGPU.h"

// FIXME  (split header???)
#include "GPUCACell.h"

class TrackingRegion;

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSetDescription;
}  // namespace edm

class CAHitQuadrupletGeneratorGPU {
public:
  using HitsOnGPU = TrackingRecHit2DSOAView;
  using HitsOnCPU = TrackingRecHit2DCUDA;
  using hindex_type = TrackingRecHit2DSOAView::hindex_type;

  using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;
  using TuplesOnCPU = pixelTuplesHeterogeneousProduct::TuplesOnCPU;
  using Quality = pixelTuplesHeterogeneousProduct::Quality;
  using Output = pixelTuplesHeterogeneousProduct::HeterogeneousPixelTuples;

  static constexpr unsigned int minLayers = 4;
  using ResultType = OrderedHitSeeds;

public:
  CAHitQuadrupletGeneratorGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
      : CAHitQuadrupletGeneratorGPU(cfg, iC) {}
  CAHitQuadrupletGeneratorGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  ~CAHitQuadrupletGeneratorGPU();

  static void fillDescriptions(edm::ParameterSetDescription& desc);
  static const char* fillDescriptionsLabel() { return "caHitQuadrupletGPU"; }

  void initEvent(const edm::Event& ev, const edm::EventSetup& es);

  void buildDoublets(HitsOnCPU const& hh, cuda::stream_t<>& stream);

  void hitNtuplets(HitsOnCPU const& hh,
                   const edm::EventSetup& es,
                   bool useRiemannFit,
                   bool transferToCPU,
                   cuda::stream_t<>& cudaStream);

  TuplesOnCPU getOutput() const {
    return TuplesOnCPU{std::move(indToEdm), hitsOnCPU->view(), tuples_, hitDetIndices_, helix_fit_results_, quality_, gpu_d, nTuples_};
  }

  void cleanup(cudaStream_t stream);
  void fillResults(const TrackingRegion& region,
                   SiPixelRecHitCollectionNew const& rechits,
                   std::vector<OrderedHitSeeds>& result,
                   const edm::EventSetup& es);

  void allocateOnGPU();
  void deallocateOnGPU();

private:
  void launchKernels(HitsOnCPU const& hh, bool useRiemannFit, bool transferToCPU, cuda::stream_t<>& cudaStream);

  std::vector<std::array<int, 4>> fetchKernelResult(int);

  CAHitQuadrupletGeneratorKernels kernels;
  HelixFitOnGPU fitter;

  // not really used at the moment
  const float caThetaCut = 0.00125f;
  const float caPhiCut = 0.1f;
  const float caHardPtCut = 0.f;

  // products
  std::vector<uint32_t> indToEdm;  // index of tuple in reco tracks....
  TuplesOnGPU* gpu_d = nullptr;    // copy of the structure on the gpu itself: this is the "Product"
  TuplesOnGPU::Container* tuples_ = nullptr;
  TuplesOnGPU::Container* hitDetIndices_ = nullptr;
  Rfit::helix_fit* helix_fit_results_ = nullptr;
  Quality* quality_ = nullptr;
  uint32_t nTuples_ = 0;
  TuplesOnGPU gpu_;

  // input
  HitsOnCPU const* hitsOnCPU = nullptr;

  std::vector<TrackingRecHit const*> hitmap_;
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorGPU_h
