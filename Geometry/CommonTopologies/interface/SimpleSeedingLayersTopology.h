#ifndef Geometry_CommonTopologies_SimpleSeedingLayersTopology_h
#define Geometry_CommonTopologies_SimpleSeedingLayersTopology_h
#include <iostream>
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
namespace phase1PixelStripTopology {

  struct LayerData {
    uint32_t start;
    uint32_t end;
    bool isStrip2D = false;  // if true then map every two module indices to the same sequential ID
  };

  enum Layer : uint8_t;

  struct LayerPairData {
    Layer inner;
    Layer outer;
    int16_t phicut;
    float minz;
    float maxz;
    float maxr;
  };

  enum Layer : uint8_t {
    BPIX1 = 0,
    BPIX2,
    BPIX3,
    BPIX4,
    FPIX1Pos,
    FPIX2Pos,
    FPIX3Pos,
    FPIX1Neg,
    FPIX2Neg,
    FPIX3Neg,
    TIB1,
    TIB2,
    TID1Pos2D,
    TID2Pos2D,
    TID3Pos2D,
    TID1Neg2D,
    TID2Neg2D,
    TID3Neg2D,
    nLayers
  };

  constexpr LayerData layerData[nLayers] = {
      {0, 96},             // BPIX1
      {96, 320},           // BPIX2
      {320, 672},          // BPIX3
      {672, 1184},         // BPIX4
      {1184, 1296},        // FPIX1Pos
      {1296, 1408},        // FPIX2Pos
      {1408, 1520},        // FPIX3Pos
      {1520, 1632},        // FPIX1Neg
      {1632, 1744},        // FPIX2Neg
      {1744, 1856},        // FPIX3Neg
      {1856, 2528, true},  // TIB1
      {2528, 3392, true},  // TIB2
      {4580, 4676, true},  // TID1Pos2D
      {4716, 4812, true},  // TID2Pos2D
      {4852, 4948, true},  // TID3Pos2D
      {4988, 5084, true},  // TID1Neg2D
      {5124, 5220, true},  // TID2Neg2D
      {5260, 5356, true},  // TID3Neg2D
  };

  using pixelTopology::phi0p05;
  using pixelTopology::phi0p06;
  using pixelTopology::phi0p07;
  using pixelTopology::phi0p09;
  using pixelTopology::phi5deg;
  //constexpr int16_t phi0p05 = 522;  // round(521.52189...) = phi2short(0.05);
  //constexpr int16_t phi0p06 = 626;  // round(625.82270...) = phi2short(0.06);
  //constexpr int16_t phi0p07 = 730;  // round(730.12648...) = phi2short(0.07);
  //constexpr int16_t phi0p09 = 900;
  //constexpr int16_t phi5deg = 1820;
  constexpr LayerPairData layerPairData[] = {
      {BPIX1, BPIX2, phi0p05, -20., 20., 20.},       // 0
      {BPIX1, FPIX1Pos, phi0p07, 0., 30., 9.},       // 1
      {BPIX1, FPIX1Neg, phi0p07, -30., 0., 9.},      // 2
      {BPIX2, BPIX3, phi0p05, -22., 22., 20.},       // 3
      {BPIX2, FPIX1Pos, phi0p07, 10., 30., 7.},      // 4
      {BPIX2, FPIX1Neg, phi0p06, -30., -10., 7.},    // 5
      {FPIX1Pos, FPIX2Pos, phi0p06, -70., 70., 5.},  // 6
      {FPIX1Neg, FPIX2Neg, phi0p05, -70., 70., 5.},  // 7
      {BPIX1, BPIX3, phi0p05, -20., 20., 20.},       // 8
      {BPIX2, BPIX4, phi0p05, -22., 22., 20.},       // 9
      {BPIX1, FPIX2Pos, phi0p06, 0., 30., 9.},       // 10
      {BPIX1, FPIX2Neg, phi0p05, -30., 0., 9.},      // 11
      {FPIX1Pos, TIB1, 1200, -70., 70., 1000.},      // 12
      {FPIX1Neg, TIB1, 1200, -70., 70., 1000.},      // 13
      {BPIX3, BPIX4, phi0p06, -22., 22., 20.},       // 14
      {BPIX3, FPIX1Pos, phi0p07, 15., 30., 6.},      // 15
      {BPIX3, FPIX1Neg, phi0p06, -30, -15., 6.},     // 16
      {FPIX2Pos, FPIX3Pos, phi0p06, -70., 70., 5.},  // 17
      {FPIX2Neg, FPIX3Neg, phi0p05, -70., 70., 5.},  // 18
      {BPIX3, TIB1, phi5deg, -22., 22., 1000.},      // 19
      {BPIX4, TIB1, phi5deg, -22., 22., 1000.},      // 20
      {BPIX4, TIB2, phi5deg, -22., 22., 1000.},      // 21
      {TIB1, TIB2, phi5deg, -55., 55., 1000.},       // 22
      {FPIX2Neg, TIB1, phi5deg, -70., 70., 1000.},   // 23
      {FPIX3Neg, TIB1, phi5deg, -70., 70., 1000.},   // 24
      {TIB1, TID1Neg2D, phi5deg, -55., 0., 1000.},   // 25
      {TIB2, TID1Neg2D, phi5deg, -55., 0., 1000.},   // 26
      {BPIX2, TIB1, phi5deg, -22., 0., 1000.},       // 27
      {BPIX2, TIB2, phi5deg, -22., 0., 1000.},       // 28
      {BPIX1, TIB1, phi5deg, -22., 0., 1000.},       // 29
      {BPIX3, TIB2, phi5deg, -22., 22., 1000.},      // 30
      {BPIX4, TID1Neg2D, phi5deg, -55., 0., 1000.},  // 31
      {FPIX1Pos, FPIX3Pos, phi0p06, -70., 70., 9.},  // 32
      {FPIX1Neg, FPIX3Neg, phi0p05, -70., 70., 9.},  // 33
      {FPIX1Neg, TIB2, phi5deg, -70., 70., 1000.},   // 34
      {FPIX2Neg, TIB2, phi5deg, -70., 70., 1000.},   // 35
      {FPIX3Neg, TIB2, phi5deg, -70., 70., 1000.},   // 36
      {BPIX2, FPIX2Neg, phi5deg, -30., 0., 1000.}    //  37
  };

  constexpr uint32_t maxNumClustersPerModules = 1024;
  constexpr auto numberOfLayers = nLayers;
  constexpr auto nPairs = std::size(layerPairData);

  constexpr auto makeLayerStart() {
    std::array<uint32_t, numberOfLayers + 1> layerStart = {{0}};
    for (auto i = 0u; i < numberOfLayers; ++i)
      if (layerData[i].isStrip2D)
        layerStart[i + 1] = layerStart[i] + (layerData[i].end - layerData[i].start) / 2;
      else
        layerStart[i + 1] = layerStart[i] + layerData[i].end - layerData[i].start;
    return layerStart;
  }

  HOST_DEVICE_CONSTANT std::array<uint32_t, numberOfLayers + 1> layerStart = makeLayerStart();

  constexpr uint16_t numberOfModules = layerStart[numberOfLayers];

  constexpr auto makeLayerPairs() {
    std::array<uint8_t, 2 * nPairs> layerPairs = {{0}};
    for (auto i = 0u; i < nPairs; ++i) {
      layerPairs[2 * i] = layerPairData[i].inner;
      layerPairs[2 * i + 1] = layerPairData[i].outer;
    }
    return layerPairs;
  }

  HOST_DEVICE_CONSTANT std::array<uint8_t, 2 * nPairs> layerPairs = makeLayerPairs();

  template <class T, class F>
  constexpr auto makePairCutsArray(F&& f) {
    std::array<T, nPairs> result{};
    for (auto i = 0u; i < nPairs; ++i)
      result[i] = f(layerPairData[i]);
    return result;
  }

  HOST_DEVICE_CONSTANT std::array<int16_t, nPairs> phicuts =
      makePairCutsArray<int16_t>([](const auto& x) { return x.phicut; });
  HOST_DEVICE_CONSTANT std::array<float, nPairs> minz = makePairCutsArray<float>([](const auto& x) { return x.minz; });
  HOST_DEVICE_CONSTANT std::array<float, nPairs> maxz = makePairCutsArray<float>([](const auto& x) { return x.maxz; });
  HOST_DEVICE_CONSTANT std::array<float, nPairs> maxr = makePairCutsArray<float>([](const auto& x) { return x.maxr; });

  using IndexMap = std::array<int, layerData[numberOfLayers - 1].end>;

  constexpr IndexMap makeIndexMap() {
    IndexMap indexMap = {{0}};
    for (auto& i : indexMap)
      i = numberOfModules;
    int newIndex = 0;
    for (const auto& layer : layerData) {
      for (auto i = layer.start; i < layer.end; ++i) {
        indexMap[i] = newIndex;
        if (!layer.isStrip2D || i % 2 == 1)
          ++newIndex;
      }
    }
    return indexMap;
  }

  constexpr IndexMap indexMap = makeIndexMap();

  // constexpr uint32_t numberOfLayers = 12;
  // constexpr int nPairs = 21 + 4 + 10 + 1; // without jump + jumping barrel + jumping forward
  // constexpr uint16_t numberOfModules = 3392;

  // HOST_DEVICE_CONSTANT uint8_t layerPairs[2 * nPairs] = {
  //     0, 1, 0, 4, 0, 7,              // BPIX1 (3)
  //     1, 2, 1, 4, 1, 7,              // BPIX2 (6)
  //     4, 5, 7, 8,                    // FPIX1 (8)
  //     2, 3, 2, 4, 2, 7, 5, 6, 8, 9,  // BPIX3 & FPIX2 (13)
  //     0, 2, 1, 3,                    // Jumping Barrel (15)
  //     0, 5, 0, 8,                    // Jumping Forward (BPIX1,FPIX2)
  //     4, 6, 7, 9,                     // Jumping Forward (19)
  //     3, 10,                          // BPIX4 (20)
  //     4, 10, 5, 10, 6, 10,            // Pixel Positive Endcap (23)
  //     7, 10, 8, 10, 9, 10,            // Pixel Negative Endcap (26)
  //     10, 11,                         // TIB1 (27)
  //     1, 10, 2, 10, 3, 11,            // Jumping from Pixel Barrel (30)
  //     4, 11, 5, 11, 6, 11,            // Jumping from Pixel Positive Endcap (33)
  //     7, 11, 8, 11, 9, 11             // Jumping from Pixel Negative Endcap (36)
  // };

  // HOST_DEVICE_CONSTANT int16_t phicuts[nPairs]{phi0p05, phi0p07, phi0p07, phi0p05, phi0p06,
  //                                              phi0p06, phi0p05, phi0p05, phi0p06, phi0p06,
  //                                              phi0p06, phi0p05, phi0p05, phi0p05, phi0p05,

  //                                              phi0p05, phi0p05, phi0p05, phi0p05, phi5deg,
  //                                              phi5deg, phi5deg, phi5deg, phi0p09, phi0p09,
  //                                              phi0p09, phi0p09, phi0p09, phi0p09, phi0p09,

  //                                              phi0p09, phi0p09, phi0p09, phi0p09, phi0p09,
  //                                              phi0p09
  //                                              };

  // HOST_DEVICE_CONSTANT float minz[nPairs] = {
  //     -20., 0., -30., -22., 10., -30., -70., -70., -22., 15., -30, -70., -70., -20., -22., 0, -30., -70., -70.,
  //     -22.,-70.,-70.,-70.,-70.,-70.,-70.,-80.,-22.,-22.,-70.,-70.,-70.,-70.,-70.,-70.,-70.};
  // HOST_DEVICE_CONSTANT float maxz[nPairs] = {
  //     20., 30., 0., 22., 30., -10., 70., 70., 22., 30., -15., 70., 70., 20., 22., 30., 0., 70., 70.,
  //     22.,70.,70.,70.,70.,70.,70.,80.,22.,22.,70.,70., 70.,70.,70.,70.,70.};
  // HOST_DEVICE_CONSTANT float maxr[nPairs] = {
  //     20., 9., 9., 20., 7., 7., 5., 5., 20., 6., 6., 5., 5., 20., 20., 9., 9., 9., 9.,
  //     10000.,10000.,10000.,10000.,10000.,10000.,10000.,10000.,10000.,10000.,10000.,10000., 10000.,10000.,10000.,10000.,10000.};

  // static constexpr uint32_t layerStart[numberOfLayers + 1] = {0,
  //                                                             96,
  //                                                             320,
  //                                                             672,  // barrel
  //                                                             1184,
  //                                                             1296,
  //                                                             1408,  // positive endcap
  //                                                             1520,
  //                                                             1632,
  //                                                             1744,  // negative endcap
  //                                                             1856,
  //                                                             2528,
  //                                                             numberOfModules};

}  // namespace phase1PixelStripTopology
namespace pixelTopology {

  struct Phase1Strip : public Phase1 {
    typedef Phase1 PixelBase;  //Could be removed using based class
    static constexpr uint32_t maxNumClustersPerModules = phase1PixelStripTopology::maxNumClustersPerModules;
    static constexpr uint32_t maxHitsInModule = phase1PixelStripTopology::maxNumClustersPerModules;
    static constexpr uint32_t maxCellNeighbors = 64;
    static constexpr uint32_t maxCellTracks = 90;
    static constexpr uint32_t maxHitsOnTrack = 15;
    static constexpr uint32_t maxHitsOnTrackForFullFit = 6;
    static constexpr uint32_t avgHitsPerTrack = 7;
    static constexpr uint32_t maxCellsPerHit = 256;
    static constexpr uint32_t avgTracksPerHit = 10;
    static constexpr uint32_t maxNumberOfTuples = 32 * 1024 * 4;
    //this is well above thanks to maxNumberOfTuples
    static constexpr uint32_t maxHitsForContainers = avgHitsPerTrack * maxNumberOfTuples;
    static constexpr uint32_t maxNumberOfDoublets = 10 * 256 * 1024;
    static constexpr uint32_t maxNumOfActiveDoublets = maxNumberOfDoublets / 8;
    static constexpr uint32_t maxNumberOfQuadruplets = maxNumberOfTuples;
    static constexpr uint32_t maxDepth = 12;
    static constexpr int minYsizeB1 = 1;
    static constexpr int minYsizeB2 = 1;
    static constexpr uint32_t const* layerStart = phase1PixelStripTopology::layerStart.data();

    static constexpr float const* minz = phase1PixelStripTopology::minz.data();
    static constexpr float const* maxz = phase1PixelStripTopology::maxz.data();
    static constexpr float const* maxr = phase1PixelStripTopology::maxr.data();

    static constexpr uint8_t const* layerPairs = phase1PixelStripTopology::layerPairs.data();
    static constexpr int16_t const* phicuts = phase1PixelStripTopology::phicuts.data();

    static constexpr uint32_t numberOfLayers = phase1PixelStripTopology::numberOfLayers;
    static constexpr uint32_t numberOfStripLayers = numberOfLayers - numberOfPixelLayers;

    static constexpr uint16_t numberOfModules = phase1PixelStripTopology::numberOfModules;
    static constexpr uint16_t numberOfPixelModules = phase1PixelStripTopology::layerStart[numberOfPixelLayers];
    static constexpr uint16_t numberOfStripModules = numberOfModules - numberOfPixelModules;

    static constexpr int nPairsForQuadruplets =
        phase1PixelStripTopology::nPairs;                           // quadruplets require hits in all layers
    static constexpr int nPairsForTriplets = nPairsForQuadruplets;  // include barrel "jumping" layer pairs
    static constexpr int nPairs = nPairsForTriplets;                // include forward "jumping" layer pairs

    static constexpr char const* nameModifier = "Phase1Strip";

    static constexpr int mapIndex(int i) {
      if (i > 0 && i < (int)phase1PixelStripTopology::indexMap.size())
        return phase1PixelStripTopology::indexMap[i];
      else
        return i;
    }
  };

}  // namespace pixelTopology

#endif  // Geometry_CommonTopologies_SimpleSeedingLayersTopology_h
