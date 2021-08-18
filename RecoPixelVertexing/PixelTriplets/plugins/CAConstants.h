#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

//#define ONLY_PHICUT

// Cellular automaton constants
namespace caConstants {

  // constants
#ifdef ONLY_PHICUT
  constexpr uint32_t maxCellNeighbors = 64;
  constexpr uint32_t maxCellTracks = 64;
  constexpr uint32_t maxNumberOfTuples = 48 * 1024;
  constexpr uint32_t maxNumberOfDoublets = 2 * 1024 * 1024;
  constexpr uint32_t maxCellsPerHit = 8 * 128;
#else  // ONLY_PHICUT
  constexpr uint32_t maxCellNeighbors = 36;
  constexpr uint32_t maxCellTracks = 48;
#ifdef GPU_SMALL_EVENTS
  // kept for testing and debugging
  constexpr uint32_t maxNumberOfTuples = 3 * 1024;
  constexpr uint32_t maxNumberOfDoublets = 128 * 1024;
  constexpr uint32_t maxCellsPerHit = 128 / 2;
#else   // GPU_SMALL_EVENTS
  // tested on MC events with 55-75 pileup events
  constexpr uint32_t maxNumberOfTuples = 24 * 1024;
  constexpr uint32_t maxNumberOfDoublets = 512 * 1024;
  constexpr uint32_t maxCellsPerHit = 128;
#endif  // GPU_SMALL_EVENTS
#endif  // ONLY_PHICUT
  constexpr uint32_t maxNumOfActiveDoublets = maxNumberOfDoublets / 8;
  constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;

  constexpr uint32_t maxNumberOfLayerPairs = 20;
  constexpr uint32_t maxNumberOfLayers = 10;
  constexpr uint32_t maxTuples = maxNumberOfTuples;

  // Modules constants
  constexpr uint32_t max_ladder_bpx0 = 12;
  constexpr uint32_t first_ladder_bpx0 = 0;
  constexpr float module_length_bpx0 = 6.7f;
  constexpr float module_tolerance_bpx0 = 0.4f;  // projection to cylinder is inaccurate on BPIX1
  constexpr uint32_t max_ladder_bpx4 = 64;
  constexpr uint32_t first_ladder_bpx4 = 84;
  constexpr float radius_even_ladder = 15.815f;
  constexpr float radius_odd_ladder = 16.146f;
  constexpr float module_length_bpx4 = 6.7f;
  constexpr float module_tolerance_bpx4 = 0.2f;
  constexpr float barrel_z_length = 26.f;
  constexpr float forward_z_begin = 32.f;

  // Last indexes
  constexpr uint32_t last_bpix1_detIndex = 96;
  constexpr uint32_t last_barrel_detIndex = 1184;

  // types
  using hindex_type = uint32_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
  using tindex_type = uint16_t;  // for tuples

  using CellNeighbors = cms::cuda::VecArray<uint32_t, maxCellNeighbors>;
  using CellTracks = cms::cuda::VecArray<tindex_type, maxCellTracks>;

  using CellNeighborsVector = cms::cuda::SimpleVector<CellNeighbors>;
  using CellTracksVector = cms::cuda::SimpleVector<CellTracks>;

  using OuterHitOfCell = cms::cuda::VecArray<uint32_t, maxCellsPerHit>;
  using TuplesContainer = cms::cuda::OneToManyAssoc<hindex_type, maxTuples, 5 * maxTuples>;
  using HitToTuple = cms::cuda::OneToManyAssoc<tindex_type, -1, 4 * maxTuples>;  // 3.5 should be enough
  using TupleMultiplicity = cms::cuda::OneToManyAssoc<tindex_type, 8, maxTuples>;

}  // namespace caConstants

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h
