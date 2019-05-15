#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoublets_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h

#include "RecoPixelVertexing/PixelTriplets/plugins/gpuPixelDoubletsAlgos.h"

#define CONSTANT_VAR __constant__

namespace gpuPixelDoublets {

  using namespace gpuPixelDoubletsAlgos;

  constexpr int nPairs = 13;
  CONSTANT_VAR const uint8_t layerPairs[2 * nPairs] = {
      0,
      1,
      1,
      2,
      2,
      3,
      // 0, 4,  1, 4,  2, 4,  4, 5,  5, 6,
      0,
      7,
      1,
      7,
      2,
      7,
      7,
      8,
      8,
      9,  // neg
      0,
      4,
      1,
      4,
      2,
      4,
      4,
      5,
      5,
      6,  // pos
  };

  constexpr int16_t phi0p05 = 522;  // round(521.52189...) = phi2short(0.05);
  constexpr int16_t phi0p06 = 626;  // round(625.82270...) = phi2short(0.06);
  constexpr int16_t phi0p07 = 730;  // round(730.12648...) = phi2short(0.07);

  CONSTANT_VAR const int16_t phicuts[nPairs]{phi0p05,
                                             phi0p05,
                                             phi0p06,
                                             phi0p07,
                                             phi0p06,
                                             phi0p06,
                                             phi0p05,
                                             phi0p05,
                                             phi0p07,
                                             phi0p06,
                                             phi0p06,
                                             phi0p05,
                                             phi0p05};

  CONSTANT_VAR float const minz[nPairs] = {-20., -22., -22., -30., -30., -30., -70., -70., 0., 10., 15., -70., -70.};

  CONSTANT_VAR float const maxz[nPairs] = {20., 22., 22., 0., -10., -15., 70., 70., 30., 30., 30., 70., 70.};

  CONSTANT_VAR float const maxr[nPairs] = {20., 20., 20., 9., 7., 6., 5., 5., 9., 7., 6., 5., 5.};

  constexpr uint32_t MaxNumOfDoublets = CAConstants::maxNumberOfDoublets();  // not really relevant

  constexpr uint32_t MaxNumOfActiveDoublets = CAConstants::maxNumOfActiveDoublets();

  using CellNeighbors = CAConstants::CellNeighbors;
  using CellTracks = CAConstants::CellTracks;
  using CellNeighborsVector = CAConstants::CellNeighborsVector;
  using CellTracksVector = CAConstants::CellTracksVector;

  __global__ void initDoublets(GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                               int nHits,
                               CellNeighborsVector* cellNeighbors,
                               CellNeighbors* cellNeighborsContainer,
                               CellTracksVector* cellTracks,
                               CellTracks* cellTracksContainer) {
    int first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = first; i < nHits; i += gridDim.x * blockDim.x)
      isOuterHitOfCell[i].reset();
  }

  constexpr auto getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
  constexpr auto getDoubletsFromHistoMinBlocksPerMP = 16;

  __global__ __launch_bounds__(
      getDoubletsFromHistoMaxBlockSize,
      getDoubletsFromHistoMinBlocksPerMP) void getDoubletsFromHisto(GPUCACell* cells,
                                                                    uint32_t* nCells,
                                                                    CellNeighborsVector* cellNeighbors,
                                                                    CellTracksVector* cellTracks,
                                                                    TrackingRecHit2DSOAView const* __restrict__ hhp,
                                                                    GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                                                                    bool ideal_cond,
                                                                    bool doClusterCut,
                                                                    bool doZCut,
                                                                    bool doPhiCut) {
    auto const& __restrict__ hh = *hhp;
    doubletsFromHisto(layerPairs,
                      nPairs,
                      cells,
                      nCells,
                      cellNeighbors,
                      cellTracks,
                      hh,
                      isOuterHitOfCell,
                      phicuts,
                      minz,
                      maxz,
                      maxr,
                      ideal_cond,
                      doClusterCut,
                      doZCut,
                      doPhiCut);
  }

}  // namespace gpuPixelDoublets

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h
