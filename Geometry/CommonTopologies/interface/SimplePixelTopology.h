#ifndef Geometry_CommonTopologies_SimplePixelTopology_h
#define Geometry_CommonTopologies_SimplePixelTopology_h

#include <array>
#include <cstdint>
#include <type_traits>
#include "FWCore/Utilities/interface/HostDeviceConstant.h"

namespace pixelTopology {

  // TODO
  // Once CUDA is dropped this could be wrapped in #ifdef CA_TRIPLETS_HOLE
  // see DataFormats/TrackingRecHitSoa/interface/TrackingRecHitSoA.h

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
      constexpr inline uint32_t layerStart(uint32_t i) {
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

  constexpr uint32_t numberOfLayers = 10;
  constexpr int nPairs = 13 + 2 + 4;
  constexpr uint16_t numberOfModules = 1856;
  constexpr int nStartingPairs = 3;  // number of layer pairs to start Ntuplet-building from

  constexpr uint32_t maxNumClustersPerModules = 1024;

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

  HOST_DEVICE_CONSTANT uint8_t startingPairs[nPairs] = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

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

  HOST_DEVICE_CONSTANT int16_t maxDPhi[nPairs]{phi0p05,
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
  HOST_DEVICE_CONSTANT float minInner[nPairs] = {
      -20., 0., -30., -22., 10., -30., -70., -70., -22., 15., -30, -70., -70., -20., -22., 0, -30., -70., -70.};
  HOST_DEVICE_CONSTANT float maxInner[nPairs] = {
      20., 30., 0., 22., 30., -10., 70., 70., 22., 30., -15., 70., 70., 20., 22., 30., 0., 70., 70.};
  HOST_DEVICE_CONSTANT float minOuter[nPairs] = {
      -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100};
  HOST_DEVICE_CONSTANT float maxOuter[nPairs] = {
      100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
  HOST_DEVICE_CONSTANT float minDZ[nPairs] = {
      -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100};
  HOST_DEVICE_CONSTANT float maxDZ[nPairs] = {
      100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
  HOST_DEVICE_CONSTANT float minPt[nPairs] = {
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  HOST_DEVICE_CONSTANT float maxZ0[nPairs] = {
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5};
  HOST_DEVICE_CONSTANT float maxDR[nPairs] = {
      20., 9., 9., 20., 7., 7., 5., 5., 20., 6., 6., 5., 5., 20., 20., 9., 9., 9., 9.};

  // Beam-spot compatibility cut (transverse impact parameter, cm), one entry per layer pair.
  // Each pair carries the value belonging to its inner layer: pairs whose inner hit is on BPIX1
  // get the tight 0.15 cm, every other pair 0.25 cm.
  HOST_DEVICE_CONSTANT float maxDCA[nPairs] = {
      0.15,
      0.15,
      0.15,  // BPIX1
      0.25,
      0.25,
      0.25,  // BPIX2
      0.25,
      0.25,  // FPIX1
      0.25,
      0.25,
      0.25,
      0.25,
      0.25,  // BPIX3 & FPIX2
      0.15,
      0.25,  // Jumping Barrel
      0.15,
      0.15,  // Jumping Forward (BPIX1,FPIX2)
      0.25,
      0.25  // Jumping Forward
  };

  // Tolerance of the RZ-alignment test applied to triplets, one entry per layer pair.
  // Each pair carries the value belonging to its inner layer: barrel layers 0.002, forward layers 0.003.
  HOST_DEVICE_CONSTANT float maxRZTolerance[nPairs] = {
      0.002,
      0.002,
      0.002,  // BPIX1
      0.002,
      0.002,
      0.002,  // BPIX2
      0.003,
      0.003,  // FPIX1
      0.002,
      0.002,
      0.002,
      0.003,
      0.003,  // BPIX3 & FPIX2
      0.002,
      0.002,  // Jumping Barrel
      0.002,
      0.002,  // Jumping Forward (BPIX1,FPIX2)
      0.003,
      0.003  // Jumping Forward
  };

  // Per-LAYER form of the two triplet cuts, kept because it is the form of the `geometry` PSet
  // (caDCACuts / caThetaCuts). The per-layer-pair tables above are the internal form the CA SoA
  // consumes; the producer broadcasts these onto them (value of the pair's own inner layer).
  HOST_DEVICE_CONSTANT float dcaCuts[numberOfLayers] = {0.15, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};

  HOST_DEVICE_CONSTANT float thetaCuts[numberOfLayers] = {
      0.002, 0.002, 0.002, 0.002, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003};

  // -------------------------------------------------------------------------------------------------------
  // Deprecated arrays only used in the CUDA version (values have no meaning in alpaka):

  // The layerStart array is only used in the CUDA version (which supports only the non-extended CA).
  // In the alpaka version of the CA the array is built in globalBeginRun from the geometry directly
  // and the values here become irrelevant.
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
  HOST_DEVICE_CONSTANT float minz[nPairs] = {
      -20., 0., -30., -22., 10., -30., -70., -70., -22., 15., -30, -70., -70., -20., -22., 0, -30., -70., -70.};
  HOST_DEVICE_CONSTANT float maxz[nPairs] = {
      20., 30., 0., 22., 30., -10., 70., 70., 22., 30., -15., 70., 70., 20., 22., 30., 0., 70., 70.};
  HOST_DEVICE_CONSTANT float maxr[nPairs] = {
      20., 9., 9., 20., 7., 7., 5., 5., 20., 6., 6., 5., 5., 20., 20., 9., 9., 9., 9.};
}  // namespace phase1PixelTopology

namespace phase2PixelTopology {

  // The parameters set here include the extension of the CA to the first 3 barrel layers of the OT.
  // This incorporates therefore the general set for Phase-2 and builds the basis for both CA configurations (with or without OT extension).
  // The layer pairs are ordered in such a way that the OT extended pairs are at the end of the arrays. So one can get the non-extended config by
  // chopping off the last elements.
  // The actual implementation of the splitting in these two configs is done below by having two separate structs in the pixelTopology namespace:
  //   - pixelTopology::Phase2    -> no OT extension
  //   - pixelTopology::Phase2OT  -> with OT extension

  constexpr uint32_t nLayersPix = 28;                      // pixel layers
  constexpr uint32_t nLayersOT = 3;                        // considered OT layers
  constexpr uint32_t nLayersTot = nLayersPix + nLayersOT;  // total number of layers for extended CA

  constexpr int nPairsPix = 57;                    // pixel only layer pairs
  constexpr int nPairsOT = 16;                     // layer pairs with OT layers
  constexpr int nPairsTot = nPairsPix + nPairsOT;  // total number of layer pairs for extended CA

  constexpr uint16_t nModulesPix = 4000;                      // pixel modules
  constexpr uint16_t nModulesOT = 2872;                       // considered OT modules (barrel only)
  constexpr uint16_t nModulesTot = nModulesPix + nModulesOT;  // total number of modules for extended CA

  // Module counts for Phase2OTStubs (with barrel + both disk sides)
  // OT barrel: 7288 modules, OT backward disks: 2956, OT forward disks: 2956
  constexpr uint32_t nModulesOTBarrel = 7288;
  constexpr uint32_t nModulesOTBackward = 2956;
  constexpr uint32_t nModulesOTForward = 2956;
  constexpr uint32_t nModulesOTStubs = nModulesOTBarrel + nModulesOTBackward + nModulesOTForward;  // 13200
  constexpr uint32_t nModulesTotStubs = nModulesPix + nModulesOTStubs;                             // 17200

  constexpr int nStartingPairs = 24;  // number of layer pairs to start Ntuplet-building from

  constexpr uint16_t numberOfModules = nModulesPix;
  constexpr uint32_t maxNumClustersPerModules = 1024;

  HOST_DEVICE_CONSTANT uint8_t layerPairs[2 * nPairsTot] = {
      0,  1,  0,  2,  0,  4,  0,  5,  0,  16, 0,  17,          // starting on BPIX1 (6)
      1,  2,  1,  3,  1,  4,  1,  5,  1,  16, 1,  17,          // starting on BPIX2 (12)
      2,  3,  2,  4,  2,  16,                                  // starting on BPIX3 (15)
      4,  5,  4,  6,  5,  6,  5,  7,  6,  7,  6,  8,  7,  8,   // forward endcap (22)
      7,  9,  8,  9,  8,  10, 9,  10, 9,  11, 10, 11, 10, 12,  // forward endcap (29)
      11, 12, 11, 13, 11, 14, 11, 15, 12, 13, 13, 14, 14, 15,  // forward endcap (36)
      16, 17, 16, 18, 17, 18, 17, 19, 18, 19, 18, 20, 19, 20,  // backward endcap (43)
      19, 21, 20, 21, 20, 22, 21, 22, 21, 23, 22, 23, 22, 24,  // backward endcap (50)
      23, 24, 23, 25, 23, 26, 23, 27, 24, 25, 25, 26, 26, 27,  // backward endcap (57)

      2,  28, 2,  28, 2,  28, 3,  28,          // barrel to OT (61)
      4,  28, 5,  28, 6,  28, 7,  28, 8,  28,  // forward endcap to OT (66)
      16, 28, 17, 28, 18, 28, 19, 28, 20, 28,  // backward endcap to OT (71)
      28, 29, 29, 30                           // OT to OT (73)
  };

  HOST_DEVICE_CONSTANT uint8_t startingPairs[nPairsTot] = {
      1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  HOST_DEVICE_CONSTANT int16_t phicuts[nPairsTot]{
      350,  600,  450,  522,  450,  522,       // BPIX1
      400,  650,  500,  730,  500,  730,       // BPIX2
      350,  400,  400,                         // BPIX3
      300,  522,  300,  522,  250,  522, 250,  // forward endcap
      522,  250,  522,  300,  522,  240, 650,  // forward endcap
      300,  200,  220,  250,  250,  250, 250,  // forward endcap
      300,  522,  300,  522,  250,  522, 250,  // backward endcap
      522,  250,  522,  300,  522,  240, 650,  // backward endcap
      300,  200,  220,  250,  250,  250, 250,  // backward endcap

      1200, 1200, 1200, 1000,        // barrel to OT
      1000, 1000, 1000, 1000, 850,   // forward endcap to OT
      1000, 1000, 1000, 1000, 1000,  // backward endcap to OT
      1100, 1250                     // OT to OT
  };

  HOST_DEVICE_CONSTANT int16_t maxDPhi[nPairsTot]{
      350,  600,  450,  522,  450,  522,       // BPIX1
      400,  650,  500,  730,  500,  730,       // BPIX2
      350,  400,  400,                         // BPIX3
      300,  522,  300,  522,  250,  522, 250,  // forward endcap
      522,  250,  522,  300,  522,  240, 650,  // forward endcap
      300,  200,  220,  250,  250,  250, 250,  // forward endcap
      300,  522,  300,  522,  250,  522, 250,  // backward endcap
      522,  250,  522,  300,  522,  240, 650,  // backward endcap
      300,  200,  220,  250,  250,  250, 250,  // backward endcap

      1200, 1200, 1200, 1000,        // barrel to OT
      1000, 1000, 1000, 1000, 850,   // forward endcap to OT
      1000, 1000, 1000, 1000, 1000,  // backward endcap to OT
      1100, 1250                     // OT to OT
  };

  HOST_DEVICE_CONSTANT float minInner[nPairsTot] = {
      -17,   -14,  4,      7,   -10000, -10000,      // BPIX1
      -17,   -15,  6,      9,   -10000, -10000,      // BPIX2
      -18,   11,   -10000,                           // BPIX3
      0,     0,    0,      0,   0,      0,      0,   // forward endcap
      0,     0,    0,      0,   0,      0,      12,  // forward endcap
      0,     0,    0,      0,   0,      0,      0,   // forward endcap
      0,     0,    0,      0,   0,      0,      0,   // backward endcap
      0,     0,    0,      0,   0,      0,      12,  // backward endcap
      0,     0,    0,      0,   0,      0,      0,   // backward endcap

      -10,   -20,  10,     -20,     // barrel to O
      11,    11,   11,     11,  0,  // forward end
      11,    11,   11,     11,  0,  // backward en
      -1200, -1200                  // OT to OT
  };

  HOST_DEVICE_CONSTANT float maxInner[nPairsTot] = {
      17,    14,    10000, 10000, -4,    -7,      // BPIX1
      17,    15,    10000, 10000, -6,    -9,      // BPIX2
      18,    10000, -11,                          // BPIX3
      14,    14,    13,    13,    13,    13, 13,  // forward endcap
      13,    13,    13,    13,    13,    13, 16,  // forward endcap
      16,    6,     4,     6,     22,    22, 22,  // forward endcap
      14,    14,    13,    13,    13,    13, 13,  // backward endcap
      13,    13,    13,    13,    13,    13, 16,  // backward endcap
      16,    6,     4,     6,     22,    22, 22,  // backward endcap

      10,    -10,   20,    20,            // barrel to OT
      10000, 10000, 10000, 10000, 10000,  // forward endcap to OT
      10000, 10000, 10000, 10000, 10000,  // backward endcap to OT
      1200,  1200                         // OT to OT
  };

  HOST_DEVICE_CONSTANT float minOuter[nPairsTot] = {
      -10000, -10000, 0,   0,    0,      0,      // BPIX1
      -10000, -10000, 6,   6,    6,      6,      // BPIX2
      -10000, 11,     11,                        // BPIX3
      3,      3,      3,   3,    3,      3, 3,   // forward endcap
      3,      3,      3,   4,    4,      3, 20,  // forward endcap
      6,      0,      0,   0,    7,      7, 7,   // forward endcap
      3,      3,      3,   3,    3,      3, 3,   // backward endcap
      3,      3,      3,   4,    4,      3, 20,  // backward endcap
      6,      0,      0,   0,    7,      7, 7,   // backward endcap

      -30,    -50,    25,  -45,           // barrel to OT
      30,     40,     55,  70,   80,      // forward endcap to OT
      -57,    -70,    -95, -110, -10000,  // backward endcap to OT
      -10000, -10000                      // OT to OT
  };

  HOST_DEVICE_CONSTANT float maxOuter[nPairsTot] = {
      10000, 10000, 10,    10000, 10,    10000,         // BPIX1
      10000, 10000, 10000, 10000, 10000, 10000,         // BPIX2
      10000, 10000, 10000,                              // BPIX3
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      21,    7,     7,     10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap
      21,    7,     7,     10000, 10000, 10000, 10000,  // backward endcap

      30,    -25,   50,    45,            // barrel to OT
      57,    80,    95,    110,   10000,  // forward endcap to OT
      -30,   -40,   -55,   -70,   -80,    // backward endcap to OT
      10000, 10000                        // OT to OT
  };

  HOST_DEVICE_CONSTANT float maxDR[nPairsTot] = {
      5.0,     10.0,    8.0,     5.0,     8.0,  5.0,         // BPIX1
      7.0,     10.0,    8.0,     10.0,    8.0,  10.0,        // BPIX2
      7.0,     7.0,     7.0,                                 // BPIX3
      4.5,     9.0,     4.5,     9.0,     4.5,  9.0,  4.5,   // forward endcap
      8.0,     4.0,     8.0,     4.5,     8.0,  4.0,  10.0,  // forward endcap
      5.0,     3.0,     3.0,     4.0,     4.0,  4.0,  3.5,   // forward endcap
      4.5,     9.0,     4.5,     9.0,     4.5,  9.0,  4.5,   // backward endcap
      8.0,     4.0,     8.0,     4.5,     8.0,  4.0,  10.0,  // backward endcap
      5.0,     3.0,     3.0,     4.0,     4.0,  4.0,  3.5,   // backward endcap

      10000.0, 10000.0, 10000.0, 10000.0,        // barrel to OT
      16.0,    16.0,    16.0,    16.0,    14.0,  // forward endcap to OT
      16.0,    16.0,    16.0,    16.0,    14.0,  // backward endcap to OT
      10000.0, 10000.0                           // OT to OT
  };

  HOST_DEVICE_CONSTANT float minDZ[nPairsTot] = {
      -16.0,  -16.0,  0.0,    0.0,    -25.0,  -25.0,           // BPIX1
      -13.0,  -15.0,  0.0,    0.0,    -19.0,  -21.0,           // BPIX2
      -9.0,   0.0,    -13.0,                                   // BPIX3
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // forward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // forward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // forward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // backward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // backward endcap
      -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // backward endcap

      -15.0,  -35.0,  10.0,   -22.0,          // barrel to OT
      5.0,    -10.0,  5.0,    15.0,   25.0,   // forward endcap to OT
      -32.5,  -50.0,  -50.0,  -70.0,  -70.0,  // backward endcap to OT
      -50.0,  -40.0                           // OT to OT
  };

  HOST_DEVICE_CONSTANT float maxDZ[nPairsTot] = {
      16.0,  16.0,  25.0,  25.0,  0.0,   0.0,           // BPIX1
      13.0,  15.0,  19.0,  21.0,  0.0,   0.0,           // BPIX2
      9.0,   13.0,  0.0,                                // BPIX3
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // forward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap
      10000, 10000, 10000, 10000, 10000, 10000, 10000,  // backward endcap

      15.0,  -10.0, 35.0,  22.0,          // barrel to OT
      32.5,  50.0,  50.0,  70.0,  70.0,   // forward endcap to OT
      -5.0,  -10.0, -5.0,  -15.0, -25.0,  // backward endcap to OT
      50.0,  40.0                         // OT to OT
  };

  HOST_DEVICE_CONSTANT float minPt[nPairsTot] = {
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85,        // BPIX1
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85,        // BPIX2
      0.85, 0.85, 0.85,                          // BPIX3
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // forward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // forward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // forward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // backward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // backward endcap
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // backward endcap

      2.00, 0.85, 0.85, 0.85,        // barrel to OT
      0.85, 0.85, 0.85, 0.85, 0.85,  // forward endcap to OT
      0.85, 0.85, 0.85, 0.85, 0.85,  // backward endcap to OT
      0.85, 0.85                     // OT to OT
  };

  HOST_DEVICE_CONSTANT float maxZ0[nPairsTot] = {
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5,        // BPIX1
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5,        // BPIX2
      12.5, 12.5, 12.5,                          // BPIX3
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,  // forward endcap
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,  // forward endcap
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,  // forward endcap
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,  // backward endcap
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,  // backward endcap
      12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,  // backward endcap

      12.5, 12.5, 12.5, 12.5,        // barrel to OT
      12.5, 12.5, 12.5, 12.5, 12.5,  // forward endcap to OT
      12.5, 12.5, 12.5, 12.5, 12.5,  // backward endcap to OT
      12.5, 12.5                     // OT to OT
  };

  // Beam-spot compatibility cut (transverse impact parameter, cm), one entry per layer pair.
  // Each pair carries the value belonging to its inner layer: BPIX1 0.15, BPIX2 0.25, BPIX3/BPIX4 0.20,
  // all remaining pixel layers 0.25, OT layers 0.10.
  HOST_DEVICE_CONSTANT float maxDCA[nPairsTot] = {
      0.15, 0.15, 0.15, 0.15, 0.15, 0.15,        // starting on BPIX1
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25,        // starting on BPIX2
      0.20, 0.20, 0.20,                          // starting on BPIX3
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  // forward endcap
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  // forward endcap
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  // forward endcap
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  // backward endcap
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  // backward endcap
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  // backward endcap

      0.20, 0.20, 0.20, 0.20,        // barrel to OT
      0.25, 0.25, 0.25, 0.25, 0.25,  // forward endcap to OT
      0.25, 0.25, 0.25, 0.25, 0.25,  // backward endcap to OT
      0.10, 0.10                     // OT to OT
  };

  // Tolerance of the RZ-alignment test applied to triplets, one entry per layer pair.
  // Each pair carries the value belonging to its inner layer: pixel barrel layers 0.002,
  // all remaining pixel layers and the OT layers 0.003.
  HOST_DEVICE_CONSTANT float maxRZTolerance[nPairsTot] = {
      0.002, 0.002, 0.002, 0.002, 0.002, 0.002,         // starting on BPIX1
      0.002, 0.002, 0.002, 0.002, 0.002, 0.002,         // starting on BPIX2
      0.002, 0.002, 0.002,                              // starting on BPIX3
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,  // forward endcap
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,  // forward endcap
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,  // forward endcap
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,  // backward endcap
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,  // backward endcap
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,  // backward endcap

      0.002, 0.002, 0.002, 0.002,         // barrel to OT
      0.003, 0.003, 0.003, 0.003, 0.003,  // forward endcap to OT
      0.003, 0.003, 0.003, 0.003, 0.003,  // backward endcap to OT
      0.003, 0.003                        // OT to OT
  };

  // Layer count of the Phase2OTStubs topology: 28 pixel + 6 OT barrel + 20 OT disk layers (10 per side)
  constexpr uint32_t nLayersPhase2OTStubs = 54;

  // =====================================================================================================
  // Default CA graph and cuts of the stub-seeded topology (Phase2OTStubs).
  //
  // The tables below mirror, entry for entry and in the same pair order, the deployed configuration
  // HLTrigger/Configuration/python/HLT_75e33/modules/hltPhase2PixelTracksSoAWithStubs_cfi.py
  // (its `layerPairs` table, with layersToExclude = []). CAHitNtupletGenerator::fillDescriptions derives
  // the defaults of the `geometry` PSet (pairGraph, startingPairs, phiCuts, ptCuts, minInner, maxInner,
  // minOuter, maxOuter, maxDR, minDZ, maxDZ) from them, so the compiled-in defaults describe the same
  // graph as the HLT menu. Regenerate them from the cfi rather than editing them by hand.
  //
  // Layer numbering (54 CA layers, inside-out):
  //   0-27:  pixel layers, as in Phase2 (0-3 barrel, 4-15 forward disks, 16-27 backward disks)
  //   28-33: OT barrel layers 1-6 (28-30 PS, 31-33 2S)
  //   34-43: OT disks 1-5 at z > 0, each disk as a PS layer (even id) followed by a 2S layer (odd id)
  //   44-53: OT disks 1-5 at z < 0, with the same PS/2S alternation
  //
  // The 112 pairs, in table order (the same index ranges label every per-pair table below):
  //   0-14     pixel-only pairs, as in Phase2OT
  //   15-17    pixel barrel L3 -> OT barrel L1, three z windows (central, backward, forward)
  //   18       pixel barrel L4 -> OT barrel L1
  //   19-23    pixel forward disks 1-5 -> OT barrel L1
  //   24-28    pixel backward disks 1-5 -> OT barrel L1
  //   29-49    pixel forward disk chain, as in Phase2OT
  //   50-70    pixel backward disk chain, as in Phase2OT
  //   71-75    OT barrel chain L1 -> L6
  //   76-78    OT barrel PS L1-L3 -> forward disk 1, PS layer (34)
  //   79-80    OT barrel PS L3 -> disk 1 2S layer, forward (35) and backward (45)
  //   81-82    OT barrel 2S L4-L5 -> forward disk 1, 2S layer (35)
  //   83-85    OT barrel PS L1-L3 -> backward disk 1, PS layer (44)
  //   86-87    OT barrel 2S L4-L5 -> backward disk 1, 2S layer (45)
  //   88-95    forward disks: PS chain and PS -> 2S links, alternating
  //   96-99    forward disks: 2S chain
  //   100-107  backward disks: PS chain and PS -> 2S links, alternating
  //   108-111  backward disks: 2S chain
  //
  // nPairsPhase2OTStubs is the compile-time upper bound on the number of layer pairs: it sizes the
  // shared-memory array innerLayerCumulativeSize[TrackerTraits::nPairs] (CAPixelDoubletsAlgos.h) and
  // the per-pair default arrays below, and it must stay >= the pair count supplied at run time, since
  // nothing asserts on that indexing. nDefaultPairsPhase2OTStubs is the number of pairs the tables
  // spell out (Phase2OTStubs::nPairsForQuadruplets, the length of the fillDescriptions defaults); the
  // entries from there up to the bound are zero-initialized and never read.
  // =====================================================================================================
  constexpr int nPairsPhase2OTStubs = 128;
  constexpr int nDefaultPairsPhase2OTStubs = 112;
  static_assert(nDefaultPairsPhase2OTStubs <= nPairsPhase2OTStubs,
                "the Phase2OTStubs default graph must fit in the per-pair tables");
  constexpr int nStartingPairsPhase2OTStubs = 23;  // pairs flagged in startingPairsPhase2OTStubs

  // Per-pair form of the two triplet cuts, geometry.caDCACutsPerPair (beam-spot compatibility) and
  // geometry.caThetaCutsPerPair (r-z alignment tolerance): the form the CA SoA consumes and the deployed
  // configuration sets. Both are optional parameters without a default, so fillDescriptions does not
  // read these tables; the per-layer defaults it does read are dcaCutsPhase2OTStubs / thetaCutsPhase2OTStubs.
  HOST_DEVICE_CONSTANT float maxDCAPhase2OTStubs[nPairsPhase2OTStubs] = {
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2,   // 0-7 pixel-only, as Phase2OT
      0.25, 0.25, 0.25, 0.25, 0.2,  0.25, 0.25,        // 8-14
      0.5,  0.5,  0.5,                                 // 15-17 PXB3 -> OTB1 (cen/bwd/fwd)
      0.5,                                             // 18 PXB4 -> OTB1
      0.5,  0.5,  0.5,  0.5,  0.5,                     // 19-23 fwd PXD1-5 -> OTB1
      0.5,  0.5,  0.5,  0.5,  0.5,                     // 24-28 bwd PXD1-5 -> OTB1
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  // 29-36 fwd pixel disks
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.4,   // 37-44
      0.4,  0.25, 0.4,  0.4,  0.4,                     // 45-49
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  // 50-57 bwd pixel disks
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.4,   // 58-65
      0.4,  0.25, 0.4,  0.4,  0.4,                     // 66-70
      0.5,  2,    2,    2,    0.5,                     // 71-75 OT barrel chain
      3,    3,    3,                                   // 76-78 OTB PS -> fwd D1 PS
      5,    5,                                         // 79-80 OTB3 -> D1 2S (fwd, bwd)
      5,    5,                                         // 81-82 OTB 2S -> fwd D1 2S
      3,    3,    3,                                   // 83-85 OTB PS -> bwd D1 PS
      5,    5,                                         // 86-87 OTB 2S -> bwd D1 2S
      3,    5,    3,    5,    3,    5,    0.65, 3,     // 88-95 fwd disks PS chain, PS->2S
      5,    5,    5,    5,                             // 96-99 fwd disks 2S chain
      3,    5,    3,    5,    3,    5,    0.65, 3,     // 100-107 bwd disks PS chain, PS->2S
      5,    5,    5,    5                              // 108-111 bwd disks 2S chain
  };

  HOST_DEVICE_CONSTANT float maxRZTolerancePhase2OTStubs[nPairsPhase2OTStubs] = {
      0.002,   0.002,   0.002,  0.002,   0.002,  0.002,   0.002, 0.002,    // 0-7 pixel-only, as Phase2OT
      0.002,   0.002,   0.002,  0.002,   0.002,  0.002,   0.002,           // 8-14
      0.002,   0.002,   0.002,                                             // 15-17 PXB3 -> OTB1 (cen/bwd/fwd)
      0.002,                                                               // 18 PXB4 -> OTB1
      0.003,   0.003,   0.003,  0.003,   0.003,                            // 19-23 fwd PXD1-5 -> OTB1
      0.003,   0.003,   0.003,  0.003,   0.003,                            // 24-28 bwd PXD1-5 -> OTB1
      0.003,   0.003,   0.003,  0.003,   0.003,  0.003,   0.003, 0.003,    // 29-36 fwd pixel disks
      0.003,   0.003,   0.003,  0.003,   0.003,  0.003,   0.003, 0.003,    // 37-44
      0.003,   0.003,   0.003,  0.003,   0.003,                            // 45-49
      0.003,   0.003,   0.003,  0.003,   0.003,  0.003,   0.003, 0.003,    // 50-57 bwd pixel disks
      0.003,   0.003,   0.003,  0.003,   0.003,  0.003,   0.003, 0.003,    // 58-65
      0.003,   0.003,   0.003,  0.003,   0.003,                            // 66-70
      0.005,   0.005,   0.0365, 0.0886,  0.0902,                           // 71-75 OT barrel chain
      0.005,   0.005,   0.005,                                             // 76-78 OTB PS -> fwd D1 PS
      0.03712, 0.03712,                                                    // 79-80 OTB3 -> D1 2S (fwd, bwd)
      0.06984, 0.096,                                                      // 81-82 OTB 2S -> fwd D1 2S
      0.005,   0.005,   0.005,                                             // 83-85 OTB PS -> bwd D1 PS
      0.06984, 0.06984,                                                    // 86-87 OTB 2S -> bwd D1 2S
      0.02,    0.1312,  0.02,   0.06984, 0.02,   0.05096, 0.02,  0.05096,  // 88-95 fwd disks PS chain, PS->2S
      0.1312,  0.1312,  0.1312, 0.3392,                                    // 96-99 fwd disks 2S chain
      0.02,    0.05096, 0.02,   0.06984, 0.02,   0.05096, 0.02,  0.06984,  // 100-107 bwd disks PS chain, PS->2S
      0.096,   0.1312,  0.1312, 0.18                                       // 108-111 bwd disks 2S chain
  };

  // The CA graph (geometry.pairGraph): (inner, outer) layer of every pair, in cfi order.
  HOST_DEVICE_CONSTANT uint8_t layerPairsPhase2OTStubs[2 * nPairsPhase2OTStubs] = {
      0,  1,  0,  2,  0,  4,  0,  5,  0,  16, 0,  17, 1,  2,  1,  3,   // pairs 0-7: pixel-only, as Phase2OT
      1,  4,  1,  5,  1,  16, 1,  17, 2,  3,  2,  4,  2,  16,          // pairs 8-14
      2,  28, 2,  28, 2,  28,                                          // pairs 15-17: PXB3 -> OTB1 (cen/bwd/fwd)
      3,  28,                                                          // pair 18: PXB4 -> OTB1
      4,  28, 5,  28, 6,  28, 7,  28, 8,  28,                          // pairs 19-23: fwd PXD1-5 -> OTB1
      16, 28, 17, 28, 18, 28, 19, 28, 20, 28,                          // pairs 24-28: bwd PXD1-5 -> OTB1
      4,  5,  4,  6,  5,  6,  5,  7,  6,  7,  6,  8,  7,  8,  7,  9,   // pairs 29-36: fwd pixel disks
      8,  9,  8,  10, 9,  10, 9,  11, 10, 11, 10, 12, 11, 12, 11, 13,  // pairs 37-44
      11, 14, 11, 15, 12, 13, 13, 14, 14, 15,                          // pairs 45-49
      16, 17, 16, 18, 17, 18, 17, 19, 18, 19, 18, 20, 19, 20, 19, 21,  // pairs 50-57: bwd pixel disks
      20, 21, 20, 22, 21, 22, 21, 23, 22, 23, 22, 24, 23, 24, 23, 25,  // pairs 58-65
      23, 26, 23, 27, 24, 25, 25, 26, 26, 27,                          // pairs 66-70
      28, 29, 29, 30, 30, 31, 31, 32, 32, 33,                          // pairs 71-75: OT barrel chain
      28, 34, 29, 34, 30, 34,                                          // pairs 76-78: OTB PS -> fwd D1 PS
      30, 35, 30, 45,                                                  // pairs 79-80: OTB3 -> D1 2S (fwd, bwd)
      31, 35, 32, 35,                                                  // pairs 81-82: OTB 2S -> fwd D1 2S
      28, 44, 29, 44, 30, 44,                                          // pairs 83-85: OTB PS -> bwd D1 PS
      31, 45, 32, 45,                                                  // pairs 86-87: OTB 2S -> bwd D1 2S
      34, 36, 34, 37, 36, 38, 36, 39, 38, 40, 38, 41, 40, 42, 40, 43,  // pairs 88-95: fwd disks PS chain, PS->2S
      35, 37, 37, 39, 39, 41, 41, 43,                                  // pairs 96-99: fwd disks 2S chain
      44, 46, 44, 47, 46, 48, 46, 49, 48, 50, 48, 51, 50, 52, 50, 53,  // pairs 100-107: bwd disks PS chain, PS->2S
      45, 47, 47, 49, 49, 51, 51, 53                                   // pairs 108-111: bwd disks 2S chain
  };

  // Starting-pair flags (geometry.startingPairs lists the ids of the flagged pairs), sized to this
  // topology's own pair bound so that fillDescriptions can scan nPairsForQuadruplets entries in bounds.
  HOST_DEVICE_CONSTANT uint8_t startingPairsPhase2OTStubs[nPairsPhase2OTStubs] = {
      1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0,                    // 0-14 pixel-only, as Phase2OT
      0, 0, 0,                                                        // 15-17 PXB3 -> OTB1 (cen/bwd/fwd)
      0,                                                              // 18 PXB4 -> OTB1
      0, 0, 0, 0, 0,                                                  // 19-23 fwd PXD1-5 -> OTB1
      0, 0, 0, 0, 0,                                                  // 24-28 bwd PXD1-5 -> OTB1
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,  // 29-49 fwd pixel disks
      1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,  // 50-70 bwd pixel disks
      0, 0, 0, 0, 0,                                                  // 71-75 OT barrel chain
      0, 0, 0,                                                        // 76-78 OTB PS -> fwd D1 PS
      0, 0,                                                           // 79-80 OTB3 -> D1 2S (fwd, bwd)
      0, 0,                                                           // 81-82 OTB 2S -> fwd D1 2S
      0, 0, 0,                                                        // 83-85 OTB PS -> bwd D1 PS
      0, 0,                                                           // 86-87 OTB 2S -> bwd D1 2S
      0, 0, 0, 0, 0, 0, 0, 0,                                         // 88-95 fwd disks PS chain, PS->2S
      0, 0, 0, 0,                                                     // 96-99 fwd disks 2S chain
      0, 0, 0, 0, 0, 0, 0, 0,                                         // 100-107 bwd disks PS chain, PS->2S
      0, 0, 0, 0                                                      // 108-111 bwd disks 2S chain
  };

  // Phi cuts for the layer pairs (geometry.phiCuts)
  HOST_DEVICE_CONSTANT int16_t maxDPhiPhase2OTStubs[nPairsPhase2OTStubs] = {
      350,  600,  450,  522,  450,  522,  400,  650,   // 0-7 pixel-only, as Phase2OT
      500,  730,  500,  730,  350,  400,  400,         // 8-14
      1200, 1200, 1200,                                // 15-17 PXB3 -> OTB1 (cen/bwd/fwd)
      1000,                                            // 18 PXB4 -> OTB1
      1000, 1000, 1000, 1000, 1000,                    // 19-23 fwd PXD1-5 -> OTB1
      1000, 1000, 1000, 1000, 1000,                    // 24-28 bwd PXD1-5 -> OTB1
      300,  522,  300,  522,  250,  522,  250,  522,   // 29-36 fwd pixel disks
      250,  522,  300,  522,  240,  650,  300,  200,   // 37-44
      220,  250,  250,  250,  250,                     // 45-49
      300,  522,  300,  522,  250,  522,  250,  522,   // 50-57 bwd pixel disks
      250,  522,  300,  522,  240,  650,  300,  200,   // 58-65
      220,  250,  250,  250,  250,                     // 66-70
      1100, 1250, 1250, 2000, 2000,                    // 71-75 OT barrel chain
      1600, 1700, 2000,                                // 76-78 OTB PS -> fwd D1 PS
      1000, 1000,                                      // 79-80 OTB3 -> D1 2S (fwd, bwd)
      1000, 1000,                                      // 81-82 OTB 2S -> fwd D1 2S
      2500, 2500, 2000,                                // 83-85 OTB PS -> bwd D1 PS
      1000, 1000,                                      // 86-87 OTB 2S -> bwd D1 2S
      2500, 1000, 2500, 1000, 2500, 1000, 2500, 1000,  // 88-95 fwd disks PS chain, PS->2S
      1000, 1000, 1000, 1000,                          // 96-99 fwd disks 2S chain
      2500, 1000, 2500, 1000, 2500, 1000, 2500, 1000,  // 100-107 bwd disks PS chain, PS->2S
      1000, 1000, 1000, 1000                           // 108-111 bwd disks 2S chain
  };

  // Min inner-hit coordinate, z (barrel) or r (disk), per pair (geometry.minInner)
  HOST_DEVICE_CONSTANT float minInnerPhase2OTStubs[nPairsPhase2OTStubs] = {
      -17.0, -14.0, 3.0,    7.0,    -10000, -10000, -17.0,  -15.0,  // 0-7 pixel-only, as Phase2OT
      6.0,   9.0,   -10000, -10000, -18.0,  11.0,   -10000,         // 8-14
      -10,   -20,   10,                                             // 15-17 PXB3 -> OTB1 (cen/bwd/fwd)
      -20,                                                          // 18 PXB4 -> OTB1
      11.6,  11.6,  11.6,   11.8,   0,                              // 19-23 fwd PXD1-5 -> OTB1
      11.6,  11.6,  11.6,   11.8,   0,                              // 24-28 bwd PXD1-5 -> OTB1
      0,     0,     0,      0,      0,      0,      0,      0,      // 29-36 fwd pixel disks
      0,     0,     0,      0,      0,      12.5,   0,      0,      // 37-44
      0,     0,     0,      0,      0,                              // 45-49
      0,     0,     0,      0,      0,      0,      0,      0,      // 50-57 bwd pixel disks
      0,     0,     0,      0,      0,      12.5,   0,      0,      // 58-65
      0,     0,     0,      0,      0,                              // 66-70
      -1200, -1200, -10000, -10000, -10000,                         // 71-75 OT barrel chain
      40,    80,    100,                                            // 76-78 OTB PS -> fwd D1 PS
      80,    -108,                                                  // 79-80 OTB3 -> D1 2S (fwd, bwd)
      90,    101,                                                   // 81-82 OTB 2S -> fwd D1 2S
      -130,  -130,  -130,                                           // 83-85 OTB PS -> bwd D1 PS
      -116,  -116,                                                  // 86-87 OTB 2S -> bwd D1 2S
      20,    45,    20,     45,     20,     20,     20,     20,     // 88-95 fwd disks PS chain, PS->2S
      55,    55,    55,     55,                                     // 96-99 fwd disks 2S chain
      20,    45,    20,     45,     20,     45,     20,     45,     // 100-107 bwd disks PS chain, PS->2S
      55,    55,    55,     55                                      // 108-111 bwd disks 2S chain
  };

  // Max inner-hit coordinate, z (barrel) or r (disk), per pair (geometry.maxInner)
  HOST_DEVICE_CONSTANT float maxInnerPhase2OTStubs[nPairsPhase2OTStubs] = {
      17.0,  14.0,  10000, 10000, -3.0,  -7.0,  17.0,  15.0,  // 0-7 pixel-only, as Phase2OT
      10000, 10000, -6.0,  -9.0,  18.0,  10000, -11.0,        // 8-14
      10,    -10,   20,                                       // 15-17 PXB3 -> OTB1 (cen/bwd/fwd)
      20,                                                     // 18 PXB4 -> OTB1
      10000, 10000, 10000, 10000, 10000,                      // 19-23 fwd PXD1-5 -> OTB1
      10000, 10000, 10000, 10000, 10000,                      // 24-28 bwd PXD1-5 -> OTB1
      14.0,  14.0,  13.0,  13.0,  13.0,  13.0,  13.0,  13.0,  // 29-36 fwd pixel disks
      13.0,  13.0,  13.0,  13.0,  13.0,  16.5,  16.5,  6.0,   // 37-44
      4.6,   6.0,   22.5,  22.5,  22.5,                       // 45-49
      14.0,  14.0,  13.0,  13.0,  13.0,  13.0,  13.0,  13.0,  // 50-57 bwd pixel disks
      13.0,  13.0,  13.0,  13.0,  13.0,  16.5,  16.5,  6.0,   // 58-65
      4.6,   6.0,   22.5,  22.5,  22.5,                       // 66-70
      1200,  1200,  10000, 10000, 10000,                      // 71-75 OT barrel chain
      130,   130,   130,                                      // 76-78 OTB PS -> fwd D1 PS
      108,   -80,                                             // 79-80 OTB3 -> D1 2S (fwd, bwd)
      116,   116,                                             // 81-82 OTB 2S -> fwd D1 2S
      -40,   -80,   -100,                                     // 83-85 OTB PS -> bwd D1 PS
      -90,   -99,                                             // 86-87 OTB 2S -> bwd D1 2S
      115,   72,    115,   72,    115,   115,   115,   115,   // 88-95 fwd disks PS chain, PS->2S
      105,   105,   105,   105,                               // 96-99 fwd disks 2S chain
      115,   72,    115,   72,    115,   72,    115,   72,    // 100-107 bwd disks PS chain, PS->2S
      105,   105,   105,   105                                // 108-111 bwd disks 2S chain
  };

  // Min outer-hit coordinate, z (barrel) or r (disk), per pair (geometry.minOuter)
  HOST_DEVICE_CONSTANT float minOuterPhase2OTStubs[nPairsPhase2OTStubs] = {
      -10000, -10000, 0,      0,      0,      0,    -10000, -10000,  // 0-7 pixel-only, as Phase2OT
      6.5,    6.5,    6.5,    6.5,    -10000, 11.7, 11.7,            // 8-14
      -30.0,  -50.0,  25.0,                                          // 15-17 PXB3 -> OTB1 (cen/bwd/fwd)
      -45.0,                                                         // 18 PXB4 -> OTB1
      30.0,   40.0,   55.0,   70.0,   80.0,                          // 19-23 fwd PXD1-5 -> OTB1
      -57.5,  -80.0,  -95.0,  -110.0, -10000,                        // 24-28 bwd PXD1-5 -> OTB1
      3.5,    3.5,    3.5,    3.5,    3.5,    3.5,  3.5,    3.5,     // 29-36 fwd pixel disks
      3.5,    3.5,    4.0,    4.0,    3.5,    20.0, 6.0,    0,       // 37-44
      0,      0,      7.0,    7.0,    7.0,                           // 45-49
      3.5,    3.5,    3.5,    3.5,    3.5,    3.5,  3.5,    3.5,     // 50-57 bwd pixel disks
      3.5,    3.5,    4.0,    4.0,    3.5,    20.0, 6.0,    0,       // 58-65
      0,      0,      7.0,    7.0,    7.0,                           // 66-70
      -10000, -10000, -10000, -10000, -10000,                        // 71-75 OT barrel chain
      20,     30,     50,                                            // 76-78 OTB PS -> fwd D1 PS
      50,     50,                                                    // 79-80 OTB3 -> D1 2S (fwd, bwd)
      60,     80,                                                    // 81-82 OTB 2S -> fwd D1 2S
      20,     30,     50,                                            // 83-85 OTB PS -> bwd D1 PS
      60,     80,                                                    // 86-87 OTB 2S -> bwd D1 2S
      20,     55,     20,     55,     20,     20,   20,     20,      // 88-95 fwd disks PS chain, PS->2S
      55,     55,     55,     55,                                    // 96-99 fwd disks 2S chain
      20,     55,     20,     55,     20,     55,   20,     55,      // 100-107 bwd disks PS chain, PS->2S
      55,     55,     55,     55                                     // 108-111 bwd disks 2S chain
  };

  // Max outer-hit coordinate, z (barrel) or r (disk), per pair (geometry.maxOuter)
  HOST_DEVICE_CONSTANT float maxOuterPhase2OTStubs[nPairsPhase2OTStubs] = {
      10000, 10000, 12.0,  10000, 12.0,  10000, 10000, 10000,  // 0-7 pixel-only, as Phase2OT
      10000, 10000, 10000, 10000, 10000, 10000, 10000,         // 8-14
      30.0,  -25.0, 50.0,                                      // 15-17 PXB3 -> OTB1 (cen/bwd/fwd)
      45.0,                                                    // 18 PXB4 -> OTB1
      57.5,  70.0,  95.0,  110.0, 10000,                       // 19-23 fwd PXD1-5 -> OTB1
      -30.0, -40.0, -55.0, -70.0, -80.0,                       // 24-28 bwd PXD1-5 -> OTB1
      10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,  // 29-36 fwd pixel disks
      10000, 10000, 10000, 10000, 10000, 10000, 21.0,  7.5,    // 37-44
      7.5,   10000, 10000, 10000, 10000,                       // 45-49
      10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,  // 50-57 bwd pixel disks
      10000, 10000, 10000, 10000, 10000, 10000, 21.0,  7.5,    // 58-65
      7.5,   10000, 10000, 10000, 10000,                       // 66-70
      10000, 10000, 10000, 10000, 10000,                       // 71-75 OT barrel chain
      40,    60,    80,                                        // 76-78 OTB PS -> fwd D1 PS
      80,    80,                                               // 79-80 OTB3 -> D1 2S (fwd, bwd)
      110,   110,                                              // 81-82 OTB 2S -> fwd D1 2S
      40,    60,    80,                                        // 83-85 OTB PS -> bwd D1 PS
      110,   110,                                              // 86-87 OTB 2S -> bwd D1 2S
      115,   105,   115,   105,   115,   115,   115,   115,    // 88-95 fwd disks PS chain, PS->2S
      105,   105,   105,   105,                                // 96-99 fwd disks 2S chain
      115,   105,   115,   105,   115,   105,   115,   105,    // 100-107 bwd disks PS chain, PS->2S
      105,   105,   105,   105                                 // 108-111 bwd disks 2S chain
  };

  // Max dr between the two hits, per pair (geometry.maxDR)
  HOST_DEVICE_CONSTANT float maxDRPhase2OTStubs[nPairsPhase2OTStubs] = {
      5.0,   10.0,  8.5,   5.0,   8.5,   5.0,   7.0,  10.0,   // 0-7 pixel-only, as Phase2OT
      8.5,   10.0,  8.5,   10.0,  7.0,   7.0,   7.0,          // 8-14
      10000, 10000, 10000,                                    // 15-17 PXB3 -> OTB1 (cen/bwd/fwd)
      10000,                                                  // 18 PXB4 -> OTB1
      16.0,  16.0,  16.0,  16.0,  14.0,                       // 19-23 fwd PXD1-5 -> OTB1
      16.0,  16.0,  16.0,  16.0,  14.0,                       // 24-28 bwd PXD1-5 -> OTB1
      4.5,   9.0,   4.5,   9.0,   4.5,   9.0,   4.5,  8.0,    // 29-36 fwd pixel disks
      4.0,   8.0,   4.5,   8.0,   4.0,   10.0,  5.0,  3.0,    // 37-44
      3.0,   4.0,   4.0,   4.0,   3.5,                        // 45-49
      4.5,   9.0,   4.5,   9.0,   4.5,   9.0,   4.5,  8.0,    // 50-57 bwd pixel disks
      4.0,   8.0,   4.5,   8.0,   4.0,   10.0,  5.0,  3.0,    // 58-65
      3.0,   4.0,   4.0,   4.0,   3.5,                        // 66-70
      10000, 10000, 10000, 10000, 10000,                      // 71-75 OT barrel chain
      10000, 10000, 10000,                                    // 76-78 OTB PS -> fwd D1 PS
      10000, 10000,                                           // 79-80 OTB3 -> D1 2S (fwd, bwd)
      10000, 10000,                                           // 81-82 OTB 2S -> fwd D1 2S
      10000, 10000, 10000,                                    // 83-85 OTB PS -> bwd D1 PS
      10000, 10000,                                           // 86-87 OTB 2S -> bwd D1 2S
      60.0,  10000, 60.0,  10000, 60.0,  60.0,  60.0, 60.0,   // 88-95 fwd disks PS chain, PS->2S
      10000, 10000, 10000, 10000,                             // 96-99 fwd disks 2S chain
      60.0,  10000, 60.0,  10000, 60.0,  10000, 60.0, 10000,  // 100-107 bwd disks PS chain, PS->2S
      10000, 10000, 10000, 10000                              // 108-111 bwd disks 2S chain
  };

  // Min dz between the two hits, per pair (geometry.minDZ)
  HOST_DEVICE_CONSTANT float minDZPhase2OTStubs[nPairsPhase2OTStubs] = {
      -16.0,  -16.0,  0.0,    0.0,    -25.0,  -25.0,  -13.0,  -17.0,   // 0-7 pixel-only, as Phase2OT
      0.0,    0.0,    -19.0,  -21.0,  -9.0,   0.0,    -13.0,           // 8-14
      -15.0,  -35.0,  10.0,                                            // 15-17 PXB3 -> OTB1 (cen/bwd/fwd)
      -22.0,                                                           // 18 PXB4 -> OTB1
      5.0,    5.0,    5.0,    15.0,   25.0,                            // 19-23 fwd PXD1-5 -> OTB1
      -32.5,  -50.0,  -50.0,  -70.0,  -70.0,                           // 24-28 bwd PXD1-5 -> OTB1
      -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // 29-36 fwd pixel disks
      -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // 37-44
      -10000, -10000, -10000, -10000, -10000,                          // 45-49
      -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // 50-57 bwd pixel disks
      -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000,  // 58-65
      -10000, -10000, -10000, -10000, -10000,                          // 66-70
      -50.0,  -40.0,  -30.0,  -30.0,  -25.0,                           // 71-75 OT barrel chain
      -10000, -10000, -10000,                                          // 76-78 OTB PS -> fwd D1 PS
      -10000, -10000,                                                  // 79-80 OTB3 -> D1 2S (fwd, bwd)
      -10000, -10000,                                                  // 81-82 OTB 2S -> fwd D1 2S
      -10000, -10000, -10000,                                          // 83-85 OTB PS -> bwd D1 PS
      -10000, -10000,                                                  // 86-87 OTB 2S -> bwd D1 2S
      15.0,   -10000, 15.0,   -10000, 15.0,   15.0,   15.0,   15.0,    // 88-95 fwd disks PS chain, PS->2S
      -10000, -10000, -10000, -10000,                                  // 96-99 fwd disks 2S chain
      -50.0,  -10000, -50.0,  -10000, -50.0,  -10000, -50.0,  -10000,  // 100-107 bwd disks PS chain, PS->2S
      -10000, -10000, -10000, -10000                                   // 108-111 bwd disks 2S chain
  };

  // Max dz between the two hits, per pair (geometry.maxDZ)
  HOST_DEVICE_CONSTANT float maxDZPhase2OTStubs[nPairsPhase2OTStubs] = {
      16.0,  16.0,  25.0,  25.0,  0.0,   0.0,   13.0,  17.0,   // 0-7 pixel-only, as Phase2OT
      19.0,  21.0,  0.0,   0.0,   9.0,   13.0,  0.0,           // 8-14
      15.0,  -10.0, 35.0,                                      // 15-17 PXB3 -> OTB1 (cen/bwd/fwd)
      22.0,                                                    // 18 PXB4 -> OTB1
      32.5,  50.0,  50.0,  70.0,  70.0,                        // 19-23 fwd PXD1-5 -> OTB1
      -5.0,  -5.0,  -5.0,  -15.0, -25.0,                       // 24-28 bwd PXD1-5 -> OTB1
      10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,  // 29-36 fwd pixel disks
      10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,  // 37-44
      10000, 10000, 10000, 10000, 10000,                       // 45-49
      10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,  // 50-57 bwd pixel disks
      10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000,  // 58-65
      10000, 10000, 10000, 10000, 10000,                       // 66-70
      50.0,  40.0,  30.0,  30.0,  25.0,                        // 71-75 OT barrel chain
      10000, 10000, 10000,                                     // 76-78 OTB PS -> fwd D1 PS
      10000, 10000,                                            // 79-80 OTB3 -> D1 2S (fwd, bwd)
      10000, 10000,                                            // 81-82 OTB 2S -> fwd D1 2S
      10000, 10000, 10000,                                     // 83-85 OTB PS -> bwd D1 PS
      10000, 10000,                                            // 86-87 OTB 2S -> bwd D1 2S
      50.0,  10000, 50.0,  10000, 50.0,  50.0,  50.0,  50.0,   // 88-95 fwd disks PS chain, PS->2S
      10000, 10000, 10000, 10000,                              // 96-99 fwd disks 2S chain
      -15.0, 10000, -15.0, 10000, -15.0, 10000, -15.0, 10000,  // 100-107 bwd disks PS chain, PS->2S
      10000, 10000, 10000, 10000                               // 108-111 bwd disks 2S chain
  };

  // pT cuts for the layer pairs (geometry.ptCuts)
  HOST_DEVICE_CONSTANT float minPtPhase2OTStubs[nPairsPhase2OTStubs] = {
      0.7,  0.8,  0.6,  0.85, 0.6,  0.85, 0.85, 0.85,  // 0-7 pixel-only, as Phase2OT
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,        // 8-14
      2.0,  0.85, 0.85,                                // 15-17 PXB3 -> OTB1 (cen/bwd/fwd)
      0.85,                                            // 18 PXB4 -> OTB1
      0.85, 0.85, 0.85, 0.85, 0.85,                    // 19-23 fwd PXD1-5 -> OTB1
      0.85, 0.85, 0.85, 0.85, 0.85,                    // 24-28 bwd PXD1-5 -> OTB1
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // 29-36 fwd pixel disks
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // 37-44
      0.85, 0.85, 0.85, 0.85, 0.85,                    // 45-49
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // 50-57 bwd pixel disks
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // 58-65
      0.85, 0.85, 0.85, 0.85, 0.85,                    // 66-70
      0.85, 0.85, 0.85, 0.85, 0.85,                    // 71-75 OT barrel chain
      0.85, 0.85, 0.85,                                // 76-78 OTB PS -> fwd D1 PS
      0.85, 0.85,                                      // 79-80 OTB3 -> D1 2S (fwd, bwd)
      0.85, 0.85,                                      // 81-82 OTB 2S -> fwd D1 2S
      0.85, 0.85, 0.85,                                // 83-85 OTB PS -> bwd D1 PS
      0.85, 0.85,                                      // 86-87 OTB 2S -> bwd D1 2S
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // 88-95 fwd disks PS chain, PS->2S
      0.85, 0.85, 0.85, 0.85,                          // 96-99 fwd disks 2S chain
      0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,  // 100-107 bwd disks PS chain, PS->2S
      0.85, 0.85, 0.85, 0.85                           // 108-111 bwd disks 2S chain
  };

  // z0 cuts for the layer pairs: the per-pair form of the scalar cellZ0Cut (geometry.cellZ0CutPerPair,
  // an optional parameter without a default, so this table is not read by fillDescriptions)
  HOST_DEVICE_CONSTANT float maxZ0Phase2OTStubs[nPairsPhase2OTStubs] = {
      13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0,  // 0-7 pixel-only, as Phase2OT
      13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0,        // 8-14
      13.0, 13.0, 13.0,                                // 15-17 PXB3 -> OTB1 (cen/bwd/fwd)
      13.0,                                            // 18 PXB4 -> OTB1
      13.0, 13.0, 13.0, 13.0, 13.0,                    // 19-23 fwd PXD1-5 -> OTB1
      13.0, 13.0, 13.0, 13.0, 13.0,                    // 24-28 bwd PXD1-5 -> OTB1
      13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0,  // 29-36 fwd pixel disks
      13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0,  // 37-44
      13.0, 13.0, 13.0, 13.0, 13.0,                    // 45-49
      13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0,  // 50-57 bwd pixel disks
      13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0, 13.0,  // 58-65
      13.0, 13.0, 13.0, 13.0, 13.0,                    // 66-70
      13.0, 13.0, 24.0, 30.0, 30.0,                    // 71-75 OT barrel chain
      13.0, 13.0, 13.0,                                // 76-78 OTB PS -> fwd D1 PS
      22.0, 22.0,                                      // 79-80 OTB3 -> D1 2S (fwd, bwd)
      34.0, 46.0,                                      // 81-82 OTB 2S -> fwd D1 2S
      13.0, 13.0, 13.0,                                // 83-85 OTB PS -> bwd D1 PS
      35.0, 40.0,                                      // 86-87 OTB 2S -> bwd D1 2S
      18.0, 42.0, 18.0, 47.0, 18.0, 47.0, 16.0, 48.0,  // 88-95 fwd disks PS chain, PS->2S
      47,   47,   45,   47,                            // 96-99 fwd disks 2S chain
      18.0, 41.0, 18.0, 47.0, 18.0, 46.0, 15.0, 47.0,  // 100-107 bwd disks PS chain, PS->2S
      47,   47,   45,   47                             // 108-111 bwd disks 2S chain
  };

  // Per-LAYER form of the two triplet cuts, kept because it is the form of the `geometry` PSet
  // (caDCACuts / caThetaCuts). The per-layer-pair tables above are the internal form the CA SoA
  // consumes; the producer broadcasts these onto them (value of the pair's own inner layer).
  HOST_DEVICE_CONSTANT float dcaCuts[nLayersTot] = {
      0.15,  //BPix1
      0.25, 0.20, 0.20, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  // Pixel layers
      0.10, 0.10, 0.10                                                               // OT layers
  };

  HOST_DEVICE_CONSTANT float thetaCuts[nLayersTot] = {
      0.002, 0.002, 0.002, 0.002,  // BPix
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,  // Pixel layers
      0.003, 0.003, 0.003                                                                  // OT layers
  };

  // Same, for the 54 CA layers of the stub-seeded topology (28 pixel + 6 OT barrel + 10 forward
  // + 10 backward OT disks). The pixel block repeats the values above; the OT layers carry the
  // looser OT settings. Sized exactly nLayersPhase2OTStubs so fillDescriptions never reads past
  // the topology's own layer count. The deployed configuration leaves geometry.caDCACuts and
  // geometry.caThetaCuts at these defaults and overrides both on every pair through the
  // "...PerPair" vectors (maxDCAPhase2OTStubs / maxRZTolerancePhase2OTStubs above).
  HOST_DEVICE_CONSTANT float dcaCutsPhase2OTStubs[nLayersPhase2OTStubs] = {
      0.15,  //BPix1
      0.25, 0.20, 0.20, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
      0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  // Pixel layers (28)
      0.10, 0.10, 0.10, 0.10, 0.10, 0.10,                                            // OT barrel layers (6)
      0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10,                    // OT forward disks (10)
      0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10                     // OT backward disks (10)
  };

  HOST_DEVICE_CONSTANT float thetaCutsPhase2OTStubs[nLayersPhase2OTStubs] = {
      0.002, 0.002, 0.002, 0.002,  // BPix
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,  // Pixel layers (28)
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003,                                            // OT barrel layers (6)
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003,                // OT forward disks (10)
      0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003                 // OT backward disks (10)
  };

  // -------------------------------------------------------------------------------------------------------
  // Deprecated arrays only used in the CUDA version (values have no meaning in alpaka or anywhere else):

  // The layerStart array is only used in the CUDA version (which supports only the non-extended CA).
  // In the alpaka version of the CA the array is built in globalBeginRun from the geometry directly
  // and the values here become irrelevant.
  HOST_DEVICE_CONSTANT uint32_t layerStart[nLayersPix + 1] = {
      0,    216,  432,  612,                                                                 // Barrel
      864,  972,  1080, 1188, 1296, 1404, 1512, 1620, 1728, 1904, 2080, 2256,                // Fp
      2432, 2540, 2648, 2756, 2864, 2972, 3080, 3188, 3296, 3472, 3648, 3824, nModulesPix};  // Np
  HOST_DEVICE_CONSTANT float minz[nPairsTot] = {
      -16.0, 4.0,   -22.0, -17.0, 6.0,   -22.0, -18.0, 11.0,  -22.0,  23.0,   30.0,   39.0,   50.0,   65.0,
      82.0,  109.0, -28.0, -35.0, -44.0, -55.0, -70.0, -87.0, -113.0, -16.,   7.0,    -22.0,  11.0,   -22.0,
      -17.0, 9.0,   -22.0, 13.0,  -22.0, 137.0, 173.0, 199.0, 229.0,  -142.0, -177.0, -203.0, -233.0, 23.0,
      30.0,  39.0,  50.0,  65.0,  82.0,  109.0, -28.0, -35.0, -44.0,  -55.0,  -70.0,  -87.0,  -113.0};

  HOST_DEVICE_CONSTANT float maxz[nPairsTot] = {

      17.0, 22.0,  -4.0,  17.0,  22.0,  -6.0,  18.0,  22.0,  -11.0,  28.0,   35.0,   44.0,   55.0,   70.0,
      87.0, 113.0, -23.0, -30.0, -39.0, -50.0, -65.0, -82.0, -109.0, 17.0,   22.0,   -7.0,   22.0,   -10.0,
      17.0, 22.0,  -9.0,  22.0,  -13.0, 142.0, 177.0, 203.0, 233.0,  -137.0, -173.0, -199.0, -229.0, 28.0,
      35.0, 44.0,  55.0,  70.0,  87.0,  113.0, -23.0, -30.0, -39.0,  -50.0,  -65.0,  -82.0,  -109.0};

  HOST_DEVICE_CONSTANT float maxr[nPairsTot] = {5.0, 5.0, 5.0, 7.0, 8.0, 8.0,  7.0, 7.0, 7.0, 6.0, 6.0, 6.0, 6.0, 5.0,
                                                6.0, 5.0, 6.0, 6.0, 6.0, 6.0,  5.0, 6.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                                5.0, 8.0, 8.0, 8.0, 8.0, 6.0,  5.0, 5.0, 5.0, 6.0, 5.0, 5.0, 5.0, 9.0,
                                                9.0, 9.0, 8.0, 8.0, 8.0, 11.0, 9.0, 9.0, 9.0, 8.0, 8.0, 8.0, 11.0};
}  // namespace phase2PixelTopology

namespace phase1HIonPixelTopology {
  // Storing here the needed constants different w.r.t. pp Phase1 topology.
  // All the other defined by inheritance in the HIon topology struct.
  using pixelTopology::phi0p09;

  constexpr uint32_t maxNumClustersPerModules = 2048;

  HOST_DEVICE_CONSTANT int16_t maxDPhi[phase1PixelTopology::nPairs]{phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09,
                                                                    phi0p09};

  // Per-LAYER form of the two triplet cuts (see the pp Phase-1 namespace). Kept for the `geometry`
  // PSet form; note that, as upstream, the HIonPhase1 traits alias the pp Phase-1 tables.
  HOST_DEVICE_CONSTANT float dcaCuts[phase1PixelTopology::numberOfLayers] = {
      0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};

  HOST_DEVICE_CONSTANT float thetaCuts[phase1PixelTopology::numberOfLayers] = {
      0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002};

}  // namespace phase1HIonPixelTopology

namespace pixelTopology {

  struct Phase2 {
    // Cluster-size window defaults of the SimDoublets validation (upstream values; the CA reads its
    // own configuration parameters instead).
    static constexpr int maxDYsize12 = 12;
    static constexpr int maxDYsize = 10;
    static constexpr int maxDYPred = 24;
    // types
    using hindex_type = uint32_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
    using tindex_type = uint32_t;  // for tuples
    using cindex_type = uint32_t;  // for cells

    static constexpr uint32_t maxCellNeighbors = 64;
    static constexpr uint32_t maxCellTracks = 302;
    static constexpr uint32_t maxLayersPerTrack = 13;
    static constexpr uint32_t maxFishboneHitsPerTrack = 8;
    static constexpr uint32_t maxHitsOnTrack = maxLayersPerTrack + maxFishboneHitsPerTrack;
    static constexpr uint32_t maxHitsOnTrackForFullFit = 6;
    static constexpr uint32_t avgHitsPerTrack = 7;
    static constexpr uint32_t maxCellsPerHit = 256;
    static constexpr uint32_t avgTracksPerHit = 10;
    static constexpr uint32_t maxNumberOfTuples = 60 * 1024;
    // this is well above thanks to maxNumberOfTuples
    static constexpr uint32_t maxHitsForContainers = avgHitsPerTrack * maxNumberOfTuples;
    static constexpr uint32_t maxNumberOfDoublets = 6 * 512 * 1024;
    static constexpr uint32_t maxNumOfActiveDoublets = maxNumberOfDoublets / 8;
    static constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;
    static constexpr uint32_t numberOfLayers = phase2PixelTopology::nLayersPix;
    static constexpr float avgCellsPerHit = 12.;
    static constexpr float avgCellsPerCell = 0.151;
    static constexpr float avgTracksPerCell = 0.04;

    static constexpr uint32_t maxSizeCluster = 2047;

    static constexpr uint32_t getDoubletsFromHistoMaxBlockSize = 128;  // for both x and y
    static constexpr uint32_t getDoubletsFromHistoMinBlocksPerMP = 16;

    static constexpr uint16_t last_bpix1_detIndex = 216;
    static constexpr uint16_t last_bpix2_detIndex = 432;
    static constexpr uint16_t last_barrel_detIndex = 864;

    static constexpr uint32_t maxPixInModule = 6000;
    static constexpr uint32_t maxPixInModuleForMorphing = 0;
    static constexpr uint32_t maxIterClustering = 16;

    static constexpr uint32_t maxNumClustersPerModules = phase2PixelTopology::maxNumClustersPerModules;
    static constexpr uint32_t maxHitsInModule = phase2PixelTopology::maxNumClustersPerModules;

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

    static constexpr int nPairsMinimal = 33;
    static constexpr int nPairsFarForwards = nPairsMinimal + 8;    // include barrel "jumping" layer pairs
    static constexpr int nPairs = phase2PixelTopology::nPairsPix;  // include far forward layer pairs
    static constexpr int nPairsForQuadruplets = nPairs;
    static constexpr int nStartingPairs = phase2PixelTopology::nStartingPairs;

    static constexpr uint16_t numberOfModules = phase2PixelTopology::nModulesPix;

    // 1000 bins < 1024 bins (10 bits) must be:
    // - < 32*32 (warpSize*warpSize for block prefix scan for CUDA)
    // - > number of columns (y) in any module. This is due to the fact
    //     that in pixel clustering we give for granted that in each
    //     bin we only have the pixel belonging to the same column.
    //     See RecoLocalTracker/SiPixelClusterizer/plugins/alpaka/PixelClustering.h#L325-L347
    static constexpr uint16_t clusterBinning = 1000;
    static constexpr uint16_t clusterBits = 10;

    static constexpr uint16_t numberOfModulesInBarrel = 756;
    static constexpr uint16_t numberOfModulesInLadder = 9;
    static constexpr uint16_t numberOfLaddersInBarrel = numberOfModulesInBarrel / numberOfModulesInLadder;

    static constexpr uint16_t firstEndcapPos = 4;
    static constexpr uint16_t firstEndcapNeg = 16;

    static constexpr int16_t xOffset = -1e4;  // not used actually, to suppress static analyzer warnings

    static constexpr char const *nameModifier = "Phase2";

    static constexpr uint32_t const *layerStart = phase2PixelTopology::layerStart;  // only for CUDA

    static constexpr inline bool isBigPixX(uint16_t px) { return false; }
    static constexpr inline bool isBigPixY(uint16_t py) { return false; }

    static constexpr inline uint16_t localX(uint16_t px) { return px; }
    static constexpr inline uint16_t localY(uint16_t py) { return py; }

    // ----------------------------------------
    //  CA cut / geometry parameters
    // ----------------------------------------
    static constexpr uint8_t const *layerPairs = phase2PixelTopology::layerPairs;
    static constexpr uint8_t const *startingPairs = phase2PixelTopology::startingPairs;
    // scalar parameters (doublet building)
    static constexpr int minInnerSizeB1 = 20;
    static constexpr int minInnerSizeB2 = 18;
    static constexpr int maxDSizeB1 = 12;
    static constexpr int maxDSize = 10;
    static constexpr int maxDSizePred = 24;
    static constexpr float cellZ0Cut = 12.5;
    // vector parameters (doublet building)
    static constexpr float const *minInner = phase2PixelTopology::minInner;
    static constexpr float const *maxInner = phase2PixelTopology::maxInner;
    static constexpr float const *minOuter = phase2PixelTopology::minOuter;
    static constexpr float const *maxOuter = phase2PixelTopology::maxOuter;
    static constexpr float const *maxDR = phase2PixelTopology::maxDR;
    static constexpr float const *minDZ = phase2PixelTopology::minDZ;
    static constexpr float const *maxDZ = phase2PixelTopology::maxDZ;
    static constexpr int16_t const *maxDPhi = phase2PixelTopology::maxDPhi;
    static constexpr float const *minPt = phase2PixelTopology::minPt;
    static constexpr float const *maxZ0 = phase2PixelTopology::maxZ0;
    // scalar parameters (doublet linking)
    // p [GeV/c] = B [T] * R [m] * 0.3 (factor from conversion from J to GeV and q = e = 1.6 * 10e-19 C)
    // 87 cm/GeV = 1/(3.8T * 0.3)
    static constexpr float maxCurv = 0.01425;  // corresponds to 800 MeV in 3.8T.
    // vector parameters (doublet linking)
    static constexpr float const *maxRZTolerance = phase2PixelTopology::maxRZTolerance;
    static constexpr float const *maxDCA = phase2PixelTopology::maxDCA;
    // per-LAYER form of the two cuts above (the `geometry` PSet form)
    static constexpr float const *thetaCuts = phase2PixelTopology::thetaCuts;
    static constexpr float const *dcaCuts = phase2PixelTopology::dcaCuts;
    // Deprecated arrays only used in the CUDA version
    static constexpr float const *minz = phase2PixelTopology::minz;
    static constexpr float const *maxz = phase2PixelTopology::maxz;
    static constexpr float const *maxr = phase2PixelTopology::maxr;
    // ----------------------------------------
  };

  struct Phase2OT : public Phase2 {
    static constexpr int nPairs = phase2PixelTopology::nPairsTot;
    static constexpr int nPairsForQuadruplets = nPairs;
    static constexpr uint32_t numberOfLayers = phase2PixelTopology::nLayersTot;    // pixel layers  + OT barrel layers
    static constexpr uint16_t numberOfModules = phase2PixelTopology::nModulesTot;  // pixel modules + OT barrel modules

    static constexpr uint32_t maxNumberOfTuples = 2 * 60 * 1024;
    // this is well above thanks to maxNumberOfTuples
    static constexpr uint32_t maxHitsForContainers = avgHitsPerTrack * maxNumberOfTuples;
    static constexpr uint32_t maxNumberOfDoublets = 12 * 512 * 1024;
    static constexpr uint32_t maxNumOfActiveDoublets = maxNumberOfDoublets / 8;
    static constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;
    static constexpr float avgCellsPerHit = 17.;
    static constexpr float avgCellsPerCell = 0.5;
    static constexpr float avgTracksPerCell = 0.09;
  };

  struct Phase2OTStubs : public Phase2OT {
    // Suffix appended to the default name of the modules templated on this topology, so that the pixel
    // cluster parameter estimator of this chain gets its own event setup product.
    static constexpr char const *nameModifier = "Phase2OTStubs";

    // Extended Phase-2 configuration using OT stubs (barrel + disks) instead of P-side hits
    // CA layers: 28 pixel + 6 OT barrel + 20 OT disk layers (10 per side) = 54 total
    static constexpr uint32_t numberOfLayers = phase2PixelTopology::nLayersPhase2OTStubs;
    // Total modules: 4000 pixel + 13200 OT (7288 barrel + 2956 backward + 2956 forward)
    static constexpr uint32_t numberOfModules = phase2PixelTopology::nModulesTotStubs;

    // Layer pairs: nPairs is the compile-time bound (it sizes the per-pair tables and the doublet
    // kernel's shared memory), nPairsForQuadruplets the number of pairs the default tables spell out,
    // i.e. the length of the fillDescriptions defaults; see nPairsPhase2OTStubs for the details.
    static constexpr int nPairs = phase2PixelTopology::nPairsPhase2OTStubs;
    static constexpr int nPairsForQuadruplets = phase2PixelTopology::nDefaultPairsPhase2OTStubs;
    static constexpr int nStartingPairs = phase2PixelTopology::nStartingPairsPhase2OTStubs;

    // Override layer pair arrays to use extended versions
    static constexpr uint8_t const *layerPairs = phase2PixelTopology::layerPairsPhase2OTStubs;

    // Override cut arrays to use extended versions (per-pair triplet cuts)
    static constexpr float const *maxRZTolerance = phase2PixelTopology::maxRZTolerancePhase2OTStubs;
    static constexpr float const *maxDCA = phase2PixelTopology::maxDCAPhase2OTStubs;
    // per-LAYER form of the two cuts above (the `geometry` PSet form), sized to this topology's
    // own layer count
    static constexpr float const *thetaCuts = phase2PixelTopology::thetaCutsPhase2OTStubs;
    static constexpr float const *dcaCuts = phase2PixelTopology::dcaCutsPhase2OTStubs;
    // Starting-pair flags sized to THIS topology's pair bound: nPairs exceeds nPairsTot, so the
    // inherited Phase2 array would be read past its end when fillDescriptions scans the flags.
    static constexpr uint8_t const *startingPairs = phase2PixelTopology::startingPairsPhase2OTStubs;

    // Override cut arrays to use extended versions (per-pair cuts)
    static constexpr int16_t const *maxDPhi = phase2PixelTopology::maxDPhiPhase2OTStubs;
    static constexpr float const *minInner = phase2PixelTopology::minInnerPhase2OTStubs;
    static constexpr float const *maxInner = phase2PixelTopology::maxInnerPhase2OTStubs;
    static constexpr float const *minOuter = phase2PixelTopology::minOuterPhase2OTStubs;
    static constexpr float const *maxOuter = phase2PixelTopology::maxOuterPhase2OTStubs;
    static constexpr float const *maxDR = phase2PixelTopology::maxDRPhase2OTStubs;
    static constexpr float const *minDZ = phase2PixelTopology::minDZPhase2OTStubs;
    static constexpr float const *maxDZ = phase2PixelTopology::maxDZPhase2OTStubs;
    static constexpr float const *minPt = phase2PixelTopology::minPtPhase2OTStubs;
    static constexpr float const *maxZ0 = phase2PixelTopology::maxZ0Phase2OTStubs;

    // Capacities for pixel+OT tracking, sized with a 10-50% margin above the peak occupancy of
    // ttbar PU200 events.
    static constexpr uint32_t maxNumberOfDoublets = 8 * 1024 * 1024;
    static constexpr uint32_t maxNumOfActiveDoublets = maxNumberOfDoublets / 8;
    static constexpr uint32_t maxNumberOfTuples = 512 * 1024;
    static constexpr uint32_t avgHitsPerTrack = 10;  // pixel+OT tracks have ~7 hits on average, up to ~14
    static constexpr uint32_t maxHitsForContainers = avgHitsPerTrack * maxNumberOfTuples;
    static constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;
    static constexpr float avgCellsPerHit = 23.;    // ~12% margin over peak ratio 20.6
    static constexpr float avgCellsPerCell = 0.3;   // ~32% margin over peak ratio 0.23
    static constexpr float avgTracksPerCell = 0.2;  // ~46% margin over peak ratio 0.14
  };

  struct Phase1 {
    // Cluster-size window defaults of the SimDoublets validation (upstream values; the CA reads its
    // own configuration parameters instead).
    static constexpr int maxDYsize12 = 28;
    static constexpr int maxDYsize = 20;
    static constexpr int maxDYPred = 20;
    // types
    using hindex_type = uint32_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
    using tindex_type = uint16_t;  // for tuples
    using cindex_type = uint32_t;  // for cells

    static constexpr uint32_t maxCellNeighbors = 36;
    static constexpr uint32_t maxCellTracks = 48;
    static constexpr uint32_t maxLayersPerTrack = 7;
    static constexpr uint32_t maxFishboneHitsPerTrack = 2;
    static constexpr uint32_t maxHitsOnTrack = maxLayersPerTrack + maxFishboneHitsPerTrack;
    static constexpr uint32_t maxHitsOnTrackForFullFit = 6;
    static constexpr uint32_t avgHitsPerTrack = 5;
    static constexpr uint32_t maxCellsPerHit = 256;
    static constexpr uint32_t avgTracksPerHit = 6;
    static constexpr uint32_t maxNumberOfTuples = 32 * 1024;
    static constexpr uint32_t maxHitsForContainers = avgHitsPerTrack * maxNumberOfTuples;
    static constexpr uint32_t maxNumberOfDoublets = 512 * 1024;
    static constexpr uint32_t maxNumOfActiveDoublets = maxNumberOfDoublets / 8;
    static constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;
    static constexpr uint32_t numberOfLayers = 10;
    static constexpr float avgCellsPerHit = 25.;
    static constexpr float avgCellsPerCell = 2.;
    static constexpr float avgTracksPerCell = 1.;

    static constexpr uint32_t maxSizeCluster = 1023;

    static constexpr uint32_t getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
    static constexpr uint32_t getDoubletsFromHistoMinBlocksPerMP = 16;

    static constexpr uint16_t last_bpix1_detIndex = 96;
    static constexpr uint16_t last_bpix2_detIndex = 320;
    static constexpr uint16_t last_barrel_detIndex = 1184;

    static constexpr uint32_t maxPixInModule = 6000;
    static constexpr uint32_t maxPixInModuleForMorphing = maxPixInModule * 2 / 5;
    static constexpr uint32_t maxIterClustering = 24;

    static constexpr uint32_t maxNumClustersPerModules = phase1PixelTopology::maxNumClustersPerModules;
    static constexpr uint32_t maxHitsInModule = phase1PixelTopology::maxNumClustersPerModules;

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

    static constexpr int nPairsForQuadruplets = 13;                     // quadruplets require hits in all layers
    static constexpr int nPairsForTriplets = nPairsForQuadruplets + 2;  // include barrel "jumping" layer pairs
    static constexpr int nPairs = nPairsForTriplets + 4;                // include forward "jumping" layer pairs
    static constexpr int nStartingPairs = phase1PixelTopology::nStartingPairs;

    static constexpr uint16_t numberOfModules = phase1PixelTopology::numberOfModules;

    static constexpr uint16_t numRowsInRoc = 80;
    static constexpr uint16_t numColsInRoc = 52;
    static constexpr uint16_t lastRowInRoc = numRowsInRoc - 1;
    static constexpr uint16_t lastColInRoc = numColsInRoc - 1;

    static constexpr uint16_t numRowsInModule = 2 * numRowsInRoc;
    static constexpr uint16_t numColsInModule = 8 * numColsInRoc;
    static constexpr uint16_t lastRowInModule = numRowsInModule - 1;
    static constexpr uint16_t lastColInModule = numColsInModule - 1;

    // 418 bins < 512, 9 bits are enough
    static constexpr uint16_t clusterBinning = numColsInModule + 2;
    static constexpr uint16_t clusterBits = 9;

    static constexpr uint16_t numberOfModulesInBarrel = 1184;
    static constexpr uint16_t numberOfModulesInLadder = 8;
    static constexpr uint16_t numberOfLaddersInBarrel = numberOfModulesInBarrel / numberOfModulesInLadder;

    static constexpr uint16_t firstEndcapPos = 4;
    static constexpr uint16_t firstEndcapNeg = 7;

    static constexpr int16_t xOffset = -81;

    static constexpr char const *nameModifier = "";

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

    static constexpr uint32_t const *layerStart = phase1PixelTopology::layerStart;

    // ----------------------------------------
    //  CA cut / geometry parameters
    // ----------------------------------------
    static constexpr uint8_t const *layerPairs = phase1PixelTopology::layerPairs;
    static constexpr uint8_t const *startingPairs = phase1PixelTopology::startingPairs;
    // scalar parameters (doublet building)
    static constexpr int minInnerSizeB1 = 1;
    static constexpr int minInnerSizeB2 = 1;
    static constexpr int maxDSizeB1 = 28;
    static constexpr int maxDSize = 20;
    static constexpr int maxDSizePred = 20;
    static constexpr float cellZ0Cut = 12.5;
    // vector parameters (doublet building)
    static constexpr float const *minInner = phase1PixelTopology::minInner;
    static constexpr float const *maxInner = phase1PixelTopology::maxInner;
    static constexpr float const *minOuter = phase1PixelTopology::minOuter;
    static constexpr float const *maxOuter = phase1PixelTopology::maxOuter;
    static constexpr float const *maxDR = phase1PixelTopology::maxDR;
    static constexpr float const *minDZ = phase1PixelTopology::minDZ;
    static constexpr float const *maxDZ = phase1PixelTopology::maxDZ;
    static constexpr int16_t const *maxDPhi = phase1PixelTopology::maxDPhi;
    static constexpr float const *minPt = phase1PixelTopology::minPt;
    static constexpr float const *maxZ0 = phase1PixelTopology::maxZ0;
    // scalar parameters (doublet linking)
    // p [GeV/c] = B [T] * R [m] * 0.3 (factor from conversion from J to GeV and q = e = 1.6 * 10e-19 C)
    // 87 cm/GeV = 1/(3.8T * 0.3)
    static constexpr float maxCurv = 1.f / (0.35 * 87.f);  // corresponds to 350 MeV in 3.8T.
    // vector parameters (doublet linking)
    static constexpr float const *maxRZTolerance = phase1PixelTopology::maxRZTolerance;
    static constexpr float const *maxDCA = phase1PixelTopology::maxDCA;
    // per-LAYER form of the two cuts above (the `geometry` PSet form)
    static constexpr float const *thetaCuts = phase1PixelTopology::thetaCuts;
    static constexpr float const *dcaCuts = phase1PixelTopology::dcaCuts;
    // Deprecated arrays only used in the CUDA version
    static constexpr float const *minz = phase1PixelTopology::minz;
    static constexpr float const *maxz = phase1PixelTopology::maxz;
    static constexpr float const *maxr = phase1PixelTopology::maxr;
    // ----------------------------------------
  };

  struct HIonPhase1 : public Phase1 {
    // Storing here the needed constants different w.r.t. pp Phase1 topology.
    // All the other defined by inheritance in the HIon topology struct.

    using tindex_type = uint32_t;  // for tuples

    static constexpr uint32_t maxCellNeighbors = 90;
    static constexpr uint32_t maxCellTracks = 90;
    static constexpr uint32_t maxNumberOfTuples = 256 * 1024;
    static constexpr uint32_t maxNumberOfDoublets = 6 * 512 * 1024;
    static constexpr uint32_t maxHitsForContainers = avgHitsPerTrack * maxNumberOfTuples;
    static constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;

    static constexpr uint32_t maxPixInModule = 10000;
    static constexpr uint32_t maxPixInModuleForMorphing = maxPixInModule * 1 / 10;
    static constexpr uint32_t maxIterClustering = 32;

    static constexpr uint32_t maxNumOfActiveDoublets =
        maxNumberOfDoublets / 4;  // TODO need to think a better way to avoid this duplication
    static constexpr uint32_t maxCellsPerHit = 256;

    static constexpr uint32_t maxNumClustersPerModules = phase1HIonPixelTopology::maxNumClustersPerModules;
    static constexpr uint32_t maxHitsInModule = phase1HIonPixelTopology::maxNumClustersPerModules;

    static constexpr char const *nameModifier = "HIonPhase1";

    // specified vector cuts for HIon
    static constexpr int16_t const *maxDPhi = phase1PixelTopology::maxDPhi;
    static constexpr float const *maxRZTolerance = phase1PixelTopology::maxRZTolerance;
    static constexpr float const *maxDCA = phase1PixelTopology::maxDCA;
    // per-LAYER form of the two cuts above (the `geometry` PSet form). As upstream, HIon uses the
    // pp Phase-1 tables here.
    static constexpr float const *thetaCuts = phase1PixelTopology::thetaCuts;
    static constexpr float const *dcaCuts = phase1PixelTopology::dcaCuts;
  };

  template <typename T>
  using isPhase1Topology = typename std::enable_if<std::is_base_of<Phase1, T>::value>::type;

  template <typename T>
  using isPhase2Topology = typename std::enable_if<std::is_base_of<Phase2, T>::value>::type;

}  // namespace pixelTopology

#endif  // Geometry_CommonTopologies_SimplePixelTopology_h
