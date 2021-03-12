#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h

#include <cuda_runtime.h>
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"

#include "CAHitNtupletGeneratorKernels.h"
#include "HelixFitOnGPU.h"

// FIXME  (split header???)
#include "GPUCACell.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSetDescription;
}  // namespace edm

class CAHitNtupletGeneratorOnGPU {
public:
  using HitsOnGPU = TrackingRecHit2DSOAView;
  using HitsOnCPU = TrackingRecHit2DCUDA;
  using hindex_type = TrackingRecHit2DSOAView::hindex_type;

  using Quality = pixelTrack::Quality;
  using OutputSoA = pixelTrack::TrackSoA;
  using HitContainer = pixelTrack::HitContainer;
  using Tuple = HitContainer;

  using QualityCuts = cAHitNtupletGenerator::QualityCuts;
  using Params = cAHitNtupletGenerator::Params;
  using Counters = cAHitNtupletGenerator::Counters;

public:
  CAHitNtupletGeneratorOnGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
      : CAHitNtupletGeneratorOnGPU(cfg, iC) {}
  CAHitNtupletGeneratorOnGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  ~CAHitNtupletGeneratorOnGPU();

  static void fillDescriptions(edm::ParameterSetDescription& desc);
  static const char* fillDescriptionsLabel() { return "caHitNtupletOnGPU"; }

  PixelTrackHeterogeneous makeTuplesAsync(TrackingRecHit2DGPU const& hits_d, float bfield, cudaStream_t stream) const;

  PixelTrackHeterogeneous makeTuples(TrackingRecHit2DCPU const& hits_d, float bfield) const;

private:
  void buildDoublets(HitsOnCPU const& hh, cudaStream_t stream) const;

  void hitNtuplets(HitsOnCPU const& hh, const edm::EventSetup& es, bool useRiemannFit, cudaStream_t cudaStream);

  void launchKernels(HitsOnCPU const& hh, bool useRiemannFit, cudaStream_t cudaStream) const;

  Params m_params;

  Counters* m_counters = nullptr;
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
