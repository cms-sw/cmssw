#ifndef Geometry_CommonTopologies_SimplePixelTopology_h
#define Geometry_CommonTopologies_SimplePixelTopology_h

#include <array>
#include <cstdint>
#include <type_traits>

namespace pixelTopology {
  template <class Function, std::size_t... Indices>
  constexpr auto map_to_array_helper(Function f, std::index_sequence<Indices...>)
      -> std::array<std::invoke_result_t<Function, std::size_t>, sizeof...(Indices)> {
    return {{f(Indices)...}};
  }

  template <int N, class Function>
  constexpr auto map_to_array(Function f) -> std::array<std::invoke_result_t<Function, std::size_t>, N> {
    return map_to_array_helper(f, std::make_index_sequence<N>{});
  }

  constexpr auto maxNumberOfLadders = 160;
  constexpr uint32_t maxLayers = 28;

  struct AverageGeometry {
    //
    float ladderZ[maxNumberOfLadders];
    float ladderX[maxNumberOfLadders];
    float ladderY[maxNumberOfLadders];
    float ladderR[maxNumberOfLadders];
    float ladderMinZ[maxNumberOfLadders];
    float ladderMaxZ[maxNumberOfLadders];
    float endCapZ[2];  // just for pos and neg Layer1
  };

  constexpr inline uint16_t localY(uint16_t py, uint16_t n) {
    auto roc = py / n;
    auto shift = 2 * roc;
    auto yInRoc = py - n * roc;
    if (yInRoc > 0)
      shift += 1;
    return py + shift;
  }

}  // namespace pixelTopology

namespace phase1PixelTopology {

  constexpr uint16_t numberOfModulesInBarrel = 1184;
  constexpr uint16_t numberOfModulesInLadder = 8;
  constexpr uint16_t numberOfLaddersInBarrel = numberOfModulesInBarrel / numberOfModulesInLadder;

  constexpr uint16_t numRowsInRoc = 80;
  constexpr uint16_t numColsInRoc = 52;
  constexpr uint16_t lastRowInRoc = numRowsInRoc - 1;
  constexpr uint16_t lastColInRoc = numColsInRoc - 1;

  constexpr uint16_t numRowsInModule = 2 * numRowsInRoc;
  constexpr uint16_t numColsInModule = 8 * numColsInRoc;
  constexpr uint16_t lastRowInModule = numRowsInModule - 1;
  constexpr uint16_t lastColInModule = numColsInModule - 1;

  constexpr int16_t xOffset = -81;
  constexpr int16_t yOffset = -54 * 4;

  constexpr uint16_t pixelThickness = 285;
  constexpr uint16_t pixelPitchX = 100;
  constexpr uint16_t pixelPitchY = 150;

  constexpr uint32_t numPixsInModule = uint32_t(numRowsInModule) * uint32_t(numColsInModule);

  constexpr uint32_t numberOfModules = 1856;
  constexpr uint32_t numberOfLayers = 10;
#ifdef __CUDA_ARCH__
  __device__
#endif
      constexpr uint32_t layerStart[numberOfLayers + 1] = {0,
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
  constexpr char const* layerName[numberOfLayers] = {
      "BL1",
      "BL2",
      "BL3",
      "BL4",  // barrel
      "E+1",
      "E+2",
      "E+3",  // positive endcap
      "E-1",
      "E-2",
      "E-3"  // negative endcap
  };

  constexpr uint16_t findMaxModuleStride() {
    bool go = true;
    int n = 2;
    while (go) {
      for (uint8_t i = 1; i < std::size(layerStart); ++i) {
        if (layerStart[i] % n != 0) {
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

  constexpr uint16_t maxModuleStride = findMaxModuleStride();

  constexpr uint8_t findLayer(uint32_t detId, uint8_t sl = 0) {
    for (uint8_t i = sl; i < std::size(layerStart); ++i)
      if (detId < layerStart[i + 1])
        return i;
    return std::size(layerStart);
  }

  constexpr uint8_t findLayerFromCompact(uint32_t detId) {
    detId *= maxModuleStride;
    for (uint8_t i = 0; i < std::size(layerStart); ++i)
      if (detId < layerStart[i + 1])
        return i;
    return std::size(layerStart);
  }

  constexpr uint32_t layerIndexSize = numberOfModules / maxModuleStride;
#ifdef __CUDA_ARCH__
  __device__
#endif
      constexpr std::array<uint8_t, layerIndexSize>
          layer = pixelTopology::map_to_array<layerIndexSize>(findLayerFromCompact);

  constexpr uint8_t getLayer(uint32_t detId) {
    return phase1PixelTopology::layer[detId / phase1PixelTopology::maxModuleStride];
  }

  constexpr bool validateLayerIndex() {
    bool res = true;
    for (auto i = 0U; i < numberOfModules; ++i) {
      auto j = i / maxModuleStride;
      res &= (layer[j] < numberOfLayers);
      res &= (i >= layerStart[layer[j]]);
      res &= (i < layerStart[layer[j] + 1]);
    }
    return res;
  }

  static_assert(validateLayerIndex(), "layer from detIndex algo is buggy");

  // this is for the ROC n<512 (upgrade 1024)
  constexpr inline uint16_t divu52(uint16_t n) {
    n = n >> 2;
    uint16_t q = (n >> 1) + (n >> 4);
    q = q + (q >> 4) + (q >> 5);
    q = q >> 3;
    uint16_t r = n - q * 13;
    return q + ((r + 3) >> 4);
  }

  constexpr inline bool isEdgeX(uint16_t px) { return (px == 0) | (px == lastRowInModule); }

  constexpr inline bool isEdgeY(uint16_t py) { return (py == 0) | (py == lastColInModule); }

  constexpr inline uint16_t toRocX(uint16_t px) { return (px < numRowsInRoc) ? px : px - numRowsInRoc; }

  constexpr inline uint16_t toRocY(uint16_t py) {
    auto roc = divu52(py);
    return py - 52 * roc;
  }

  constexpr inline bool isBigPixX(uint16_t px) { return (px == 79) | (px == 80); }

  constexpr inline bool isBigPixY(uint16_t py) {
    auto ly = toRocY(py);
    return (ly == 0) | (ly == lastColInRoc);
  }

  constexpr inline uint16_t localX(uint16_t px) {
    auto shift = 0;
    if (px > lastRowInRoc)
      shift += 1;
    if (px > numRowsInRoc)
      shift += 1;
    return px + shift;
  }

  constexpr inline uint16_t localY(uint16_t py) {
    auto roc = divu52(py);
    auto shift = 2 * roc;
    auto yInRoc = py - 52 * roc;
    if (yInRoc > 0)
      shift += 1;
    return py + shift;
  }

}  // namespace phase1PixelTopology

namespace phase2PixelTopology {

  constexpr uint32_t numberOfModulesInBarrel = 756;
  constexpr uint32_t numberOfModulesInLadder = 9;
  constexpr uint32_t numberOfLaddersInBarrel = numberOfModulesInBarrel / numberOfModulesInLadder;

  constexpr uint32_t numberOfModules = 3892;
  constexpr uint8_t numberOfLayers = 28;

  constexpr uint32_t layerStart[numberOfLayers + 1] = {0,
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

  constexpr uint16_t findMaxModuleStride() {
    bool go = true;
    int n = 2;
    while (go) {
      for (uint8_t i = 1; i < numberOfLayers + 1; ++i) {
        if (layerStart[i] % n != 0) {
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

  constexpr uint16_t maxModuleStride = findMaxModuleStride();

  constexpr uint8_t findLayerFromCompact(uint32_t detId) {
    detId *= maxModuleStride;
    for (uint8_t i = 0; i < numberOfLayers + 1; ++i)
      if (detId < layerStart[i + 1])
        return i;
    return numberOfLayers + 1;
  }

  constexpr uint16_t layerIndexSize = numberOfModules / maxModuleStride;
  constexpr std::array<uint8_t, layerIndexSize> layer =
      pixelTopology::map_to_array<layerIndexSize>(findLayerFromCompact);

  constexpr bool validateLayerIndex() {
    bool res = true;
    for (auto i = 0U; i < numberOfModules; ++i) {
      auto j = i / maxModuleStride;
      res &= (layer[j] < numberOfLayers);
      res &= (i >= layerStart[layer[j]]);
      res &= (i < layerStart[layer[j] + 1]);
    }
    return res;
  }

  static_assert(validateLayerIndex(), "phase2 layer from detIndex algo is buggy");

}  // namespace phase2PixelTopology

#endif  // Geometry_CommonTopologies_SimplePixelTopology_h
