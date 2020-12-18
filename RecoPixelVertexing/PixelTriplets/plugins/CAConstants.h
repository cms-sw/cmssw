#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

//#define ONLY_PHICUT

namespace CAConstants {

  // constants
#ifdef ONLY_PHICUT
  constexpr uint32_t maxNumberOfTuples() { return 48 * 1024; }
  constexpr uint32_t maxNumberOfDoublets() { return 2 * 1024 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 8 * 128; }
#else
#ifdef GPU_SMALL_EVENTS
  // kept for testing and debugging
  constexpr uint32_t maxNumberOfTuples() { return 3 * 1024; }
  constexpr uint32_t maxNumberOfDoublets() { return 128 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 128 / 2; }
#else
  // tested on MC events with 55-75 pileup events
  constexpr uint32_t maxNumberOfTuples() { return 24 * 1024; }
  constexpr uint32_t maxNumberOfDoublets() { return 512 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 128; }
#endif
#endif  // ONLY_PHICUT
  constexpr uint32_t maxNumOfActiveDoublets() { return maxNumberOfDoublets() / 8; }
  constexpr uint32_t maxNumberOfQuadruplets() { return maxNumberOfTuples(); }

  constexpr uint32_t maxNumberOfLayerPairs() { return 20; }
  constexpr uint32_t maxNumberOfLayers() { return 10; }
  constexpr uint32_t maxTuples() { return maxNumberOfTuples(); }

  // types
  using hindex_type = uint32_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
  using tindex_type = uint16_t;  // for tuples

#ifdef ONLY_PHICUT
  using CellNeighbors = cms::cuda::VecArray<uint32_t, 64>;
  using CellTracks = cms::cuda::VecArray<tindex_type, 64>;
#else
  using CellNeighbors = cms::cuda::VecArray<uint32_t, 36>;
  using CellTracks = cms::cuda::VecArray<tindex_type, 48>;
#endif

  using CellNeighborsVector = cms::cuda::SimpleVector<CellNeighbors>;
  using CellTracksVector = cms::cuda::SimpleVector<CellTracks>;

  using OuterHitOfCell = cms::cuda::VecArray<uint32_t, maxCellsPerHit()>;
  using TuplesContainer = cms::cuda::OneToManyAssoc<hindex_type, maxTuples(), 5 * maxTuples()>;
  using HitToTuple =
      cms::cuda::OneToManyAssoc<tindex_type, pixelGPUConstants::maxNumberOfHits, 4 * maxTuples()>;  // 3.5 should be enough
  using TupleMultiplicity = cms::cuda::OneToManyAssoc<tindex_type, 8, maxTuples()>;

}  // namespace CAConstants

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h
