#ifndef RecoPixelVertexing_PixelTriplets_plugins_gpuPixelDoublets_h
#define RecoPixelVertexing_PixelTriplets_plugins_gpuPixelDoublets_h

#include "RecoPixelVertexing/PixelTriplets/plugins/gpuPixelDoubletsAlgos.h"

#define CONSTANT_VAR __constant__

namespace gpuPixelDoublets {

  template <typename TrackerTraits>
  using CellNeighbors = caStructures::CellNeighborsT<TrackerTraits>;
  template <typename TrackerTraits>
  using CellTracks = caStructures::CellTracksT<TrackerTraits>;
  template <typename TrackerTraits>
  using CellNeighborsVector = caStructures::CellNeighborsVectorT<TrackerTraits>;
  template <typename TrackerTraits>
  using CellTracksVector = caStructures::CellTracksVectorT<TrackerTraits>;
  template <typename TrackerTraits>
  using OuterHitOfCell = caStructures::OuterHitOfCellT<TrackerTraits>;
  template <typename TrackerTraits>
  using Hits = typename GPUCACellT<TrackerTraits>::Hits;

  // end constants
  // clang-format on

  template <typename TrackerTraits>
  __global__ void initDoublets(OuterHitOfCell<TrackerTraits> isOuterHitOfCell,
                               int nHits,
                               CellNeighborsVector<TrackerTraits>* cellNeighbors,
                               CellNeighbors<TrackerTraits>* cellNeighborsContainer,
                               CellTracksVector<TrackerTraits>* cellTracks,
                               CellTracks<TrackerTraits>* cellTracksContainer) {
    assert(isOuterHitOfCell.container);
    int first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = first; i < nHits - isOuterHitOfCell.offset; i += gridDim.x * blockDim.x)
      isOuterHitOfCell.container[i].reset();

    if (0 == first) {
      cellNeighbors->construct(TrackerTraits::maxNumOfActiveDoublets, cellNeighborsContainer);
      cellTracks->construct(TrackerTraits::maxNumOfActiveDoublets, cellTracksContainer);
      auto i = cellNeighbors->extend();
      assert(0 == i);
      (*cellNeighbors)[0].reset();
      i = cellTracks->extend();
      assert(0 == i);
      (*cellTracks)[0].reset();
    }
  }

  constexpr auto getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
  constexpr auto getDoubletsFromHistoMinBlocksPerMP = 16;

  template <typename TrackerTraits>
  __global__
#ifdef __CUDACC__
  __launch_bounds__(getDoubletsFromHistoMaxBlockSize, getDoubletsFromHistoMinBlocksPerMP)
#endif
      void getDoubletsFromHisto(GPUCACellT<TrackerTraits>* cells,
                                uint32_t* nCells,
                                CellNeighborsVector<TrackerTraits>* cellNeighbors,
                                CellTracksVector<TrackerTraits>* cellTracks,
                                TrackingRecHit2DSOAViewT<TrackerTraits> const* __restrict__ hhp,
                                OuterHitOfCell<TrackerTraits> isOuterHitOfCell,
                                int nActualPairs,
                                CellCutsT<TrackerTraits> cuts) {
    auto const& __restrict__ hh = *hhp;

    doubletsFromHisto<TrackerTraits>(
        nActualPairs, cells, nCells, cellNeighbors, cellTracks, hh, isOuterHitOfCell, cuts);
  }

}  // namespace gpuPixelDoublets

#endif  // RecoPixelVertexing_PixelTriplets_plugins_gpuPixelDoublets_h
