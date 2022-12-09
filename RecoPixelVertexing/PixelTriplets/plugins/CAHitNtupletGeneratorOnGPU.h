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

#include "GPUCACell.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSetDescription;
}  // namespace edm

template <typename TrackerTraits>
class CAHitNtupletGeneratorOnGPU {
public:
  using PixelTrackHeterogeneous = PixelTrackHeterogeneousT<TrackerTraits>;

  using HitsView = TrackingRecHit2DSOAViewT<TrackerTraits>;
  using HitsOnGPU = TrackingRecHit2DGPUT<TrackerTraits>;
  using HitsOnCPU = TrackingRecHit2DCPUT<TrackerTraits>;
  using hindex_type = typename HitsView::hindex_type;

  using HitToTuple = caStructures::HitToTupleT<TrackerTraits>;
  using TupleMultiplicity = caStructures::TupleMultiplicityT<TrackerTraits>;
  using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;

  using GPUCACell = GPUCACellT<TrackerTraits>;
  using OutputSoA = pixelTrack::TrackSoAT<TrackerTraits>;
  using HitContainer = typename OutputSoA::HitContainer;
  using Tuple = HitContainer;

  using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
  using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;

  using Quality = pixelTrack::Quality;

  using QualityCuts = pixelTrack::QualityCutsT<TrackerTraits>;
  using Params = caHitNtupletGenerator::ParamsT<TrackerTraits>;
  using Counters = caHitNtupletGenerator::Counters;

public:
  CAHitNtupletGeneratorOnGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC)
      : CAHitNtupletGeneratorOnGPU(cfg, iC) {}
  CAHitNtupletGeneratorOnGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

  static void fillDescriptions(edm::ParameterSetDescription& desc);
  static void fillDescriptionsCommon(edm::ParameterSetDescription& desc);
  //static const char* fillDescriptionsLabel() { return "caHitNtupletOnGPU"; }

  void beginJob();
  void endJob();

  PixelTrackHeterogeneous makeTuplesAsync(HitsOnGPU const& hits_d, float bfield, cudaStream_t stream) const;

  PixelTrackHeterogeneous makeTuples(HitsOnCPU const& hits_d, float bfield) const;

private:
  void buildDoublets(HitsOnGPU const& hh, cudaStream_t stream) const;

  void hitNtuplets(HitsOnGPU const& hh, const edm::EventSetup& es, bool useRiemannFit, cudaStream_t cudaStream);

  void launchKernels(HitsOnGPU const& hh, bool useRiemannFit, cudaStream_t cudaStream) const;

  Params m_params;

  Counters* m_counters = nullptr;
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorOnGPU_h
