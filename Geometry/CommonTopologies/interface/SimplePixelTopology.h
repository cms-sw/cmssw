#ifndef Geometry_CommonTopologies_SimplePixelTopology_h
#define Geometry_CommonTopologies_SimplePixelTopology_h

#include <array>
#include <cstdint>
#include <type_traits>
#include "FWCore/Utilities/interface/HostDeviceConstant.h"

namespace pixelTopology {

  constexpr auto maxNumberOfLadders = 160;
  constexpr uint32_t maxLayers = 28;

  template <typename TrackerTraits>
  struct AverageGeometryT {
    //
    float ladderZ[TrackerTraits::numberOfLaddersInBarrel];
    float ladderX[TrackerTraits::numberOfLaddersInBarrel];
    float ladderY[TrackerTraits::numberOfLaddersInBarrel];
    float ladderR[TrackerTraits::numberOfLaddersInBarrel];
    float ladderMinZ[TrackerTraits::numberOfLaddersInBarrel];
    float ladderMaxZ[TrackerTraits::numberOfLaddersInBarrel];
    float endCapZ[2];  // just for pos and neg Layer1
  };

  constexpr int16_t phi0p05 = 522;  // round(521.52189...) = phi2short(0.05);
  constexpr int16_t phi0p06 = 626;  // round(625.82270...) = phi2short(0.06);
  constexpr int16_t phi0p07 = 730;  // round(730.12648...) = phi2short(0.07);
  constexpr int16_t phi0p09 = 900;

  constexpr uint16_t last_barrel_layer = 3;  // this is common between all the topologies

  template <class Function, std::size_t... Indices>
  constexpr auto map_to_array_helper(Function f, std::index_sequence<Indices...>)
      -> std::array<std::invoke_result_t<Function, std::size_t>, sizeof...(Indices)> {
    return {{f(Indices)...}};
  }

  template <int N, class Function>
  constexpr auto map_to_array(Function f) -> std::array<std::invoke_result_t<Function, std::size_t>, N> {
    return map_to_array_helper(f, std::make_index_sequence<N>{});
  }

  template <typename TrackerTraits>
  constexpr uint16_t findMaxModuleStride() {
    bool go = true;
    int n = 2;
    while (go) {
      for (uint8_t i = 1; i < TrackerTraits::numberOfLayers + 1; ++i) {
        if (TrackerTraits::layerStart[i] % n != 0) {
          go = false;
          break;
        }
      }
      if (!go)
        break;
      n *= 2;
    }
    return n / 2;
  }

  template <typename TrackerTraits>
  constexpr uint16_t maxModuleStride = findMaxModuleStride<TrackerTraits>();

  template <typename TrackerTraits>
  constexpr uint8_t findLayer(uint32_t detId, uint8_t sl = 0) {
    for (uint8_t i = sl; i < TrackerTraits::numberOfLayers + 1; ++i)
      if (detId < TrackerTraits::layerStart[i + 1])
        return i;
    return TrackerTraits::numberOfLayers + 1;
  }

  template <typename TrackerTraits>
  constexpr uint8_t findLayerFromCompact(uint32_t detId) {
    detId *= maxModuleStride<TrackerTraits>;
    for (uint8_t i = 0; i < TrackerTraits::numberOfLayers + 1; ++i)
      if (detId < TrackerTraits::layerStart[i + 1])
        return i;
    return TrackerTraits::numberOfLayers + 1;
  }

  template <typename TrackerTraits>
  constexpr uint32_t layerIndexSize = TrackerTraits::numberOfModules / maxModuleStride<TrackerTraits>;

  template <typename TrackerTraits>
#ifdef __CUDA_ARCH__
  __device__
#endif
      constexpr std::array<uint8_t, layerIndexSize<TrackerTraits>>
          layer = map_to_array<layerIndexSize<TrackerTraits>>(findLayerFromCompact<TrackerTraits>);

  template <typename TrackerTraits>
  constexpr uint8_t getLayer(uint32_t detId) {
    return layer<TrackerTraits>[detId / maxModuleStride<TrackerTraits>];
  }

  template <typename TrackerTraits>
  constexpr bool validateLayerIndex() {
    bool res = true;
    for (auto i = 0U; i < TrackerTraits::numberOfModules; ++i) {
      auto j = i / maxModuleStride<TrackerTraits>;
      res &= (layer<TrackerTraits>[j] < TrackerTraits::numberOfLayers);
      res &= (i >= TrackerTraits::layerStart[layer<TrackerTraits>[j]]);
      res &= (i < TrackerTraits::layerStart[layer<TrackerTraits>[j] + 1]);
    }
    return res;
  }

  template <typename TrackerTraits>
#ifdef __CUDA_ARCH__
  __device__
#endif
      constexpr inline uint32_t
      layerStart(uint32_t i) {
    return TrackerTraits::layerStart[i];
  }

  constexpr inline uint16_t divu52(uint16_t n) {
    n = n >> 2;
    uint16_t q = (n >> 1) + (n >> 4);
    q = q + (q >> 4) + (q >> 5);
    q = q >> 3;
    uint16_t r = n - q * 13;
    return q + ((r + 3) >> 4);
  }
}  // namespace pixelTopology

namespace phase1PixelTopology {

  using pixelTopology::phi0p05;
  using pixelTopology::phi0p06;
  using pixelTopology::phi0p07;

  constexpr uint32_t numberOfLayers = 28;
  constexpr int nPairs = 13 + 2 + 4;
  constexpr uint16_t numberOfModules = 1856;

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

  HOST_DEVICE_CONSTANT uint8_t layerPairs[2 * nPairs] = {
      0, 1, 0, 4, 0, 7,              // BPIX1 (3)
      1, 2, 1, 4, 1, 7,              // BPIX2 (6)
      4, 5, 7, 8,                    // FPIX1 (8)
      2, 3, 2, 4, 2, 7, 5, 6, 8, 9,  // BPIX3 & FPIX2 (13)
      0, 2, 1, 3,                    // Jumping Barrel (15)
      0, 5, 0, 8,                    // Jumping Forward (BPIX1,FPIX2)
      4, 6, 7, 9                     // Jumping Forward (19)
  };

  HOST_DEVICE_CONSTANT int16_t phicuts[nPairs]{phi0p05,
                                               phi0p07,
                                               phi0p07,
                                               phi0p05,
                                               phi0p06,
                                               phi0p06,
                                               phi0p05,
                                               phi0p05,
                                               phi0p06,
                                               phi0p06,
                                               phi0p06,
                                               phi0p05,
                                               phi0p05,
                                               phi0p05,
                                               phi0p05,
                                               phi0p05,
                                               phi0p05,
                                               phi0p05,
                                               phi0p05};
  HOST_DEVICE_CONSTANT float minz[nPairs] = {
      -20., 0., -30., -22., 10., -30., -70., -70., -22., 15., -30, -70., -70., -20., -22., 0, -30., -70., -70.};
  HOST_DEVICE_CONSTANT float maxz[nPairs] = {
      20., 30., 0., 22., 30., -10., 70., 70., 22., 30., -15., 70., 70., 20., 22., 30., 0., 70., 70.};
  HOST_DEVICE_CONSTANT float maxr[nPairs] = {
      20., 9., 9., 20., 7., 7., 5., 5., 20., 6., 6., 5., 5., 20., 20., 9., 9., 9., 9.};

  static constexpr uint32_t layerStart[numberOfLayers + 1] = {0,
                                                              96,
                                                              320,
                                                              672,  // barrel
                                                              1184,
                                                              1296,
                                                              1408,  // positive endcap
                                                              1520,
                                                              1632,
                                                              1744,  // negative endcap
                                                              numberOfModules};
}  // namespace phase1PixelTopology

namespace phase2PixelTopology {

  using pixelTopology::phi0p05;
  using pixelTopology::phi0p06;
  using pixelTopology::phi0p07;
  using pixelTopology::phi0p09;

  constexpr uint32_t numberOfLayers = 28;
  constexpr int nPairs = 23 + 6 + 14 + 8 + 4;  // include far forward layer pairs
  constexpr uint16_t numberOfModules = 3892;

  HOST_DEVICE_CONSTANT uint8_t layerPairs[2 * nPairs] = {

      0,  1,  0,  4,  0,  16,  //BPIX1 (3)
      1,  2,  1,  4,  1,  16,  //BPIX2 (6)
      2,  3,  2,  4,  2,  16,  //BPIX3 & Forward (9)

      4,  5,  5,  6,  6,  7,  7,  8,  8,  9,  9,  10, 10, 11,  //POS (16)
      16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23,  //NEG (23)

      0,  2,  0,  5,  0,  17, 0,  6,  0,  18,  // BPIX1 Jump (28)
      1,  3,  1,  5,  1,  17, 1,  6,  1,  18,  // BPIX2 Jump (33)

      11, 12, 12, 13, 13, 14, 14, 15,  //Late POS (37)
      23, 24, 24, 25, 25, 26, 26, 27,  //Late NEG (41)

      4,  6,  5,  7,  6,  8,  7,  9,  8,  10, 9,  11, 10, 12,  //POS Jump (48)
      16, 18, 17, 19, 18, 20, 19, 21, 20, 22, 21, 23, 22, 24,  //NEG Jump (55)
  };
  HOST_DEVICE_CONSTANT uint32_t layerStart[numberOfLayers + 1] = {0,
                                                                  108,
                                                                  324,
                                                                  504,  //Barrel
                                                                  756,
                                                                  864,
                                                                  972,
                                                                  1080,
                                                                  1188,
                                                                  1296,
                                                                  1404,
                                                                  1512,
                                                                  1620,
                                                                  1796,
                                                                  1972,
                                                                  2148,  //Fp
                                                                  2324,
                                                                  2432,
                                                                  2540,
                                                                  2648,
                                                                  2756,
                                                                  2864,
                                                                  2972,
                                                                  3080,
                                                                  3188,
                                                                  3364,
                                                                  3540,
                                                                  3716,  //Np
                                                                  numberOfModules};

  HOST_DEVICE_CONSTANT int16_t phicuts[nPairs]{
      phi0p05, phi0p05, phi0p05, phi0p06, phi0p07, phi0p07, phi0p06, phi0p07, phi0p07, phi0p05, phi0p05,
      phi0p05, phi0p05, phi0p05, phi0p05, phi0p05, phi0p05, phi0p05, phi0p05, phi0p05, phi0p05, phi0p05,
      phi0p05, phi0p05, phi0p05, phi0p05, phi0p05, phi0p05, phi0p05, phi0p07, phi0p07, phi0p07, phi0p07,
      phi0p07, phi0p07, phi0p07, phi0p07, phi0p07, phi0p07, phi0p07, phi0p07, phi0p07, phi0p07, phi0p07,
      phi0p07, phi0p07, phi0p07, phi0p05, phi0p05, phi0p05, phi0p05, phi0p05, phi0p05, phi0p05, phi0p05};

  HOST_DEVICE_CONSTANT float minz[nPairs] = {
      -16.0, 4.0,   -22.0, -17.0, 6.0,   -22.0, -18.0, 11.0,  -22.0,  23.0,   30.0,   39.0,   50.0,   65.0,
      82.0,  109.0, -28.0, -35.0, -44.0, -55.0, -70.0, -87.0, -113.0, -16.,   7.0,    -22.0,  11.0,   -22.0,
      -17.0, 9.0,   -22.0, 13.0,  -22.0, 137.0, 173.0, 199.0, 229.0,  -142.0, -177.0, -203.0, -233.0, 23.0,
      30.0,  39.0,  50.0,  65.0,  82.0,  109.0, -28.0, -35.0, -44.0,  -55.0,  -70.0,  -87.0,  -113.0};

  HOST_DEVICE_CONSTANT float maxz[nPairs] = {

      17.0, 22.0,  -4.0,  17.0,  22.0,  -6.0,  18.0,  22.0,  -11.0,  28.0,   35.0,   44.0,   55.0,   70.0,
      87.0, 113.0, -23.0, -30.0, -39.0, -50.0, -65.0, -82.0, -109.0, 17.0,   22.0,   -7.0,   22.0,   -10.0,
      17.0, 22.0,  -9.0,  22.0,  -13.0, 142.0, 177.0, 203.0, 233.0,  -137.0, -173.0, -199.0, -229.0, 28.0,
      35.0, 44.0,  55.0,  70.0,  87.0,  113.0, -23.0, -30.0, -39.0,  -50.0,  -65.0,  -82.0,  -109.0};

  HOST_DEVICE_CONSTANT float maxr[nPairs] = {5.0, 5.0, 5.0, 7.0, 8.0, 8.0,  7.0, 7.0, 7.0, 6.0, 6.0, 6.0, 6.0, 5.0,
                                             6.0, 5.0, 6.0, 6.0, 6.0, 6.0,  5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                             5.0, 8.0, 8.0, 8.0, 8.0, 6.0,  5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 5.0, 9.0,
                                             9.0, 9.0, 8.0, 8.0, 8.0, 11.0, 9.0, 9.0, 9.0, 8.0, 8.0, 8.0, 11.0};
}  // namespace phase2PixelTopology

namespace pixelTopology {

  struct Phase2 {
    // types
    using hindex_type = uint32_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
    using tindex_type = uint32_t;  // for tuples
    using cindex_type = uint32_t;  // for cells

    static constexpr uint32_t maxCellNeighbors = 64;
    static constexpr uint32_t maxCellTracks = 302;
    static constexpr uint32_t maxHitsOnTrack = 15;
    static constexpr uint32_t maxHitsOnTrackForFullFit = 6;
    static constexpr uint32_t avgHitsPerTrack = 7;
    static constexpr uint32_t maxCellsPerHit = 256;
    static constexpr uint32_t avgTracksPerHit = 10;
    static constexpr uint32_t maxNumberOfTuples = 256 * 1024;
    //this is well above thanks to maxNumberOfTuples
    static constexpr uint32_t maxHitsForContainers = avgHitsPerTrack * maxNumberOfTuples;
    static constexpr uint32_t maxNumberOfDoublets = 5 * 512 * 1024;
    static constexpr uint32_t maxNumOfActiveDoublets = maxNumberOfDoublets / 8;
    static constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;
    static constexpr uint32_t maxDepth = 12;
    static constexpr uint32_t numberOfLayers = 28;

    static constexpr uint32_t maxSizeCluster = 2047;

    static constexpr uint32_t getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
    static constexpr uint32_t getDoubletsFromHistoMinBlocksPerMP = 16;

    static constexpr uint16_t last_bpix1_detIndex = 108;
    static constexpr uint16_t last_bpix2_detIndex = 324;
    static constexpr uint16_t last_barrel_detIndex = 504;

    static constexpr uint32_t maxPixInModule = 6000;

    static constexpr float moduleLength = 4.345f;
    static constexpr float endcapCorrection = 0.0f;

    static constexpr float xerr_barrel_l1_def = 0.00035f;
    static constexpr float yerr_barrel_l1_def = 0.00125f;
    static constexpr float xerr_barrel_ln_def = 0.00035f;
    static constexpr float yerr_barrel_ln_def = 0.00125f;
    static constexpr float xerr_endcap_def = 0.00060f;
    static constexpr float yerr_endcap_def = 0.00180f;

    static constexpr float bigPixXCorrection = 0.0f;
    static constexpr float bigPixYCorrection = 0.0f;

    static constexpr float dzdrFact = 8 * 0.0285 / 0.015;  // from dz/dr to "DY"
    static constexpr float z0Cut = 7.5f;
    static constexpr float doubletHardPt = 0.8f;

    static constexpr int minYsizeB1 = 25;
    static constexpr int minYsizeB2 = 15;

    static constexpr int nPairsMinimal = 33;
    static constexpr int nPairsFarForwards = nPairsMinimal + 8;  // include barrel "jumping" layer pairs
    static constexpr int nPairs = phase2PixelTopology::nPairs;   // include far forward layer pairs

    static constexpr int maxDYsize12 = 12;
    static constexpr int maxDYsize = 10;
    static constexpr int maxDYPred = 20;

    static constexpr uint16_t numberOfModules = 3892;

    static constexpr uint16_t clusterBinning = 1024;
    static constexpr uint16_t clusterBits = 10;

    static constexpr uint16_t numberOfModulesInBarrel = 756;
    static constexpr uint16_t numberOfModulesInLadder = 9;
    static constexpr uint16_t numberOfLaddersInBarrel = numberOfModulesInBarrel / numberOfModulesInLadder;

    static constexpr uint16_t firstEndcapPos = 4;
    static constexpr uint16_t firstEndcapNeg = 16;

    static constexpr int16_t xOffset = -1e4;  //not used actually, to suppress static analyzer warnings

    static constexpr char const *nameModifier = "Phase2";

    static constexpr uint32_t const *layerStart = phase2PixelTopology::layerStart;
    static constexpr float const *minz = phase2PixelTopology::minz;
    static constexpr float const *maxz = phase2PixelTopology::maxz;
    static constexpr float const *maxr = phase2PixelTopology::maxr;

    static constexpr uint8_t const *layerPairs = phase2PixelTopology::layerPairs;
    static constexpr int16_t const *phicuts = phase2PixelTopology::phicuts;

    static constexpr inline bool isBigPixX(uint16_t px) { return false; }
    static constexpr inline bool isBigPixY(uint16_t py) { return false; }

    static constexpr inline uint16_t localX(uint16_t px) { return px; }
    static constexpr inline uint16_t localY(uint16_t py) { return py; }
  };

  struct Phase1 {
    // types
    using hindex_type = uint32_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
    using tindex_type = uint16_t;  // for tuples
    using cindex_type = uint32_t;  // for cells

    static constexpr uint32_t maxCellNeighbors = 36;
    static constexpr uint32_t maxCellTracks = 48;
    static constexpr uint32_t maxHitsOnTrack = 10;
    static constexpr uint32_t maxHitsOnTrackForFullFit = 6;
    static constexpr uint32_t avgHitsPerTrack = 5;
    static constexpr uint32_t maxCellsPerHit = 256;
    static constexpr uint32_t avgTracksPerHit = 6;
    static constexpr uint32_t maxNumberOfTuples = 32 * 1024;
    static constexpr uint32_t maxHitsForContainers = avgHitsPerTrack * maxNumberOfTuples;
    static constexpr uint32_t maxNumberOfDoublets = 512 * 1024;
    static constexpr uint32_t maxNumOfActiveDoublets = maxNumberOfDoublets / 8;
    static constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;
    static constexpr uint32_t maxDepth = 6;
    static constexpr uint32_t numberOfLayers = 10;

    static constexpr uint32_t maxSizeCluster = 1023;

    static constexpr uint32_t getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
    static constexpr uint32_t getDoubletsFromHistoMinBlocksPerMP = 16;

    static constexpr uint16_t last_bpix1_detIndex = 96;
    static constexpr uint16_t last_bpix2_detIndex = 320;
    static constexpr uint16_t last_barrel_detIndex = 1184;

    static constexpr uint32_t maxPixInModule = 6000;

    static constexpr float moduleLength = 6.7f;
    static constexpr float endcapCorrection = 1.5f;

    static constexpr float xerr_barrel_l1_def = 0.00200f;
    static constexpr float yerr_barrel_l1_def = 0.00210f;
    static constexpr float xerr_barrel_ln_def = 0.00200f;
    static constexpr float yerr_barrel_ln_def = 0.00210f;
    static constexpr float xerr_endcap_def = 0.0020f;
    static constexpr float yerr_endcap_def = 0.00210f;

    static constexpr float bigPixXCorrection = 1.0f;
    static constexpr float bigPixYCorrection = 8.0f;

    static constexpr float dzdrFact = 8 * 0.0285 / 0.015;  // from dz/dr to "DY"
    static constexpr float z0Cut = 12.f;
    static constexpr float doubletHardPt = 0.5f;

    static constexpr int minYsizeB1 = 36;
    static constexpr int minYsizeB2 = 28;

    static constexpr int nPairsForQuadruplets = 13;                     // quadruplets require hits in all layers
    static constexpr int nPairsForTriplets = nPairsForQuadruplets + 2;  // include barrel "jumping" layer pairs
    static constexpr int nPairs = nPairsForTriplets + 4;                // include forward "jumping" layer pairs

    static constexpr int maxDYsize12 = 28;
    static constexpr int maxDYsize = 20;
    static constexpr int maxDYPred = 20;

    static constexpr uint16_t numberOfModules = 1856;

    static constexpr uint16_t numRowsInRoc = 80;
    static constexpr uint16_t numColsInRoc = 52;
    static constexpr uint16_t lastRowInRoc = numRowsInRoc - 1;
    static constexpr uint16_t lastColInRoc = numColsInRoc - 1;

    static constexpr uint16_t numRowsInModule = 2 * numRowsInRoc;
    static constexpr uint16_t numColsInModule = 8 * numColsInRoc;
    static constexpr uint16_t lastRowInModule = numRowsInModule - 1;
    static constexpr uint16_t lastColInModule = numColsInModule - 1;

    static constexpr uint16_t clusterBinning = numColsInModule + 2;
    static constexpr uint16_t clusterBits = 9;

    static constexpr uint16_t numberOfModulesInBarrel = 1184;
    static constexpr uint16_t numberOfModulesInLadder = 8;
    static constexpr uint16_t numberOfLaddersInBarrel = numberOfModulesInBarrel / numberOfModulesInLadder;

    static constexpr uint16_t firstEndcapPos = 4;
    static constexpr uint16_t firstEndcapNeg = 7;

    static constexpr int16_t xOffset = -81;

    static constexpr char const *nameModifier = "";

    static constexpr uint32_t const *layerStart = phase1PixelTopology::layerStart;
    static constexpr float const *minz = phase1PixelTopology::minz;
    static constexpr float const *maxz = phase1PixelTopology::maxz;
    static constexpr float const *maxr = phase1PixelTopology::maxr;

    static constexpr uint8_t const *layerPairs = phase1PixelTopology::layerPairs;
    static constexpr int16_t const *phicuts = phase1PixelTopology::phicuts;

    static constexpr inline bool isEdgeX(uint16_t px) { return (px == 0) | (px == lastRowInModule); }

    static constexpr inline bool isEdgeY(uint16_t py) { return (py == 0) | (py == lastColInModule); }

    static constexpr inline uint16_t toRocX(uint16_t px) { return (px < numRowsInRoc) ? px : px - numRowsInRoc; }

    static constexpr inline uint16_t toRocY(uint16_t py) {
      auto roc = divu52(py);
      return py - 52 * roc;
    }

    static constexpr inline bool isBigPixX(uint16_t px) { return (px == 79) | (px == 80); }
    static constexpr inline bool isBigPixY(uint16_t py) {
      auto ly = toRocY(py);
      return (ly == 0) | (ly == lastColInRoc);
    }

    static constexpr inline uint16_t localX(uint16_t px) {
      auto shift = 0;
      if (px > lastRowInRoc)
        shift += 1;
      if (px > numRowsInRoc)
        shift += 1;
      return px + shift;
    }

    static constexpr inline uint16_t localY(uint16_t py) {
      auto roc = divu52(py);
      auto shift = 2 * roc;
      auto yInRoc = py - 52 * roc;
      if (yInRoc > 0)
        shift += 1;
      return py + shift;
    }
  };

  template <typename T>
  using isPhase1Topology = typename std::enable_if<std::is_base_of<Phase1, T>::value>::type;

  template <typename T>
  using isPhase2Topology = typename std::enable_if<std::is_base_of<Phase2, T>::value>::type;

  // struct HIonPhase1 : public Phase1 {
  //     static constexpr uint32_t maxNumberOfDoublets=3*1024*1024;};

}  // namespace pixelTopology

#endif  // Geometry_CommonTopologies_SimplePixelTopology_h
