#ifndef RecoPixelVertexing_PixelTriplets_plugins_gpuPixelDoublets_h
#define RecoPixelVertexing_PixelTriplets_plugins_gpuPixelDoublets_h

#include "RecoPixelVertexing/PixelTriplets/plugins/gpuPixelDoubletsAlgos.h"

#define CONSTANT_VAR __constant__

namespace gpuPixelDoublets {

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
                                HitsConstView<TrackerTraits> hh,
                                OuterHitOfCell<TrackerTraits> isOuterHitOfCell,
                                int nActualPairs,
                                CellCutsT<TrackerTraits> cuts) {

    doubletsFromHisto<TrackerTraits>(
        nActualPairs, cells, nCells, cellNeighbors, cellTracks, hh, isOuterHitOfCell, cuts);
  }

}  // namespace gpuPixelDoublets

#endif  // RecoPixelVertexing_PixelTriplets_plugins_gpuPixelDoublets_h
