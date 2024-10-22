#include <cassert>
#include <iostream>
#include <tuple>

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

namespace {

  // original code from CMSSW_4_4
  using namespace pixelTopology;

  std::tuple<int, bool> localXori(int mpx) {
    const float m_pitchx = 1.f;
    int binoffx = int(mpx);         // truncate to int
    float local_pitchx = m_pitchx;  // defaultpitch

    if (binoffx > 80) {  // ROC 1 - handles x on edge cluster
      binoffx = binoffx + 2;
    } else if (binoffx == 80) {  // ROC 1
      binoffx = binoffx + 1;
      local_pitchx = 2 * m_pitchx;

    } else if (binoffx == 79) {  // ROC 0
      binoffx = binoffx + 0;
      local_pitchx = 2 * m_pitchx;
    } else if (binoffx >= 0) {  // ROC 0
      binoffx = binoffx + 0;

    } else {  // too small
      assert("binoffx too small" == 0);
    }

    return std::make_tuple(binoffx, local_pitchx > m_pitchx);
  }

  std::tuple<int, bool> localYori(int mpy) {
    const float m_pitchy = 1.f;
    int binoffy = int(mpy);         // truncate to int
    float local_pitchy = m_pitchy;  // defaultpitch

    if (binoffy > 416) {  // ROC 8, not real ROC
      binoffy = binoffy + 17;
    } else if (binoffy == 416) {  // ROC 8
      binoffy = binoffy + 16;
      local_pitchy = 2 * m_pitchy;

    } else if (binoffy == 415) {  // ROC 7, last big pixel
      binoffy = binoffy + 15;
      local_pitchy = 2 * m_pitchy;
    } else if (binoffy > 364) {  // ROC 7
      binoffy = binoffy + 15;
    } else if (binoffy == 364) {  // ROC 7
      binoffy = binoffy + 14;
      local_pitchy = 2 * m_pitchy;

    } else if (binoffy == 363) {  // ROC 6
      binoffy = binoffy + 13;
      local_pitchy = 2 * m_pitchy;
    } else if (binoffy > 312) {  // ROC 6
      binoffy = binoffy + 13;
    } else if (binoffy == 312) {  // ROC 6
      binoffy = binoffy + 12;
      local_pitchy = 2 * m_pitchy;

    } else if (binoffy == 311) {  // ROC 5
      binoffy = binoffy + 11;
      local_pitchy = 2 * m_pitchy;
    } else if (binoffy > 260) {  // ROC 5
      binoffy = binoffy + 11;
    } else if (binoffy == 260) {  // ROC 5
      binoffy = binoffy + 10;
      local_pitchy = 2 * m_pitchy;

    } else if (binoffy == 259) {  // ROC 4
      binoffy = binoffy + 9;
      local_pitchy = 2 * m_pitchy;
    } else if (binoffy > 208) {  // ROC 4
      binoffy = binoffy + 9;
    } else if (binoffy == 208) {  // ROC 4
      binoffy = binoffy + 8;
      local_pitchy = 2 * m_pitchy;

    } else if (binoffy == 207) {  // ROC 3
      binoffy = binoffy + 7;
      local_pitchy = 2 * m_pitchy;
    } else if (binoffy > 156) {  // ROC 3
      binoffy = binoffy + 7;
    } else if (binoffy == 156) {  // ROC 3
      binoffy = binoffy + 6;
      local_pitchy = 2 * m_pitchy;

    } else if (binoffy == 155) {  // ROC 2
      binoffy = binoffy + 5;
      local_pitchy = 2 * m_pitchy;
    } else if (binoffy > 104) {  // ROC 2
      binoffy = binoffy + 5;
    } else if (binoffy == 104) {  // ROC 2
      binoffy = binoffy + 4;
      local_pitchy = 2 * m_pitchy;

    } else if (binoffy == 103) {  // ROC 1
      binoffy = binoffy + 3;
      local_pitchy = 2 * m_pitchy;
    } else if (binoffy > 52) {  // ROC 1
      binoffy = binoffy + 3;
    } else if (binoffy == 52) {  // ROC 1
      binoffy = binoffy + 2;
      local_pitchy = 2 * m_pitchy;

    } else if (binoffy == 51) {  // ROC 0
      binoffy = binoffy + 1;
      local_pitchy = 2 * m_pitchy;
    } else if (binoffy > 0) {  // ROC 0
      binoffy = binoffy + 1;
    } else if (binoffy == 0) {  // ROC 0
      binoffy = binoffy + 0;
      local_pitchy = 2 * m_pitchy;
    } else {
      assert("binoffy too small" == 0);
    }

    return std::make_tuple(binoffy, local_pitchy > m_pitchy);
  }

}  // namespace

constexpr void testLayer() {
  for (auto i = 0U; i < Phase1::numberOfModules; ++i) {
    uint32_t layer = getLayer<Phase1>(i);
    uint32_t tLayer = findLayer<Phase1>(i);
    assert(tLayer == layer);

    assert(layer < Phase1::numberOfLayers);
    assert(i >= Phase1::layerStart[layer]);
    assert(i < Phase1::layerStart[layer + 1]);
  }
}

__global__ void kernel_testLayer() { testLayer(); }

int main() {
  cms::cudatest::requireDevices();

  for (uint16_t ix = 0; ix < 80 * 2; ++ix) {
    auto ori = localXori(ix);
    auto xl = Phase1::localX(ix);
    auto bp = Phase1::isBigPixX(ix);
    if (std::get<0>(ori) != xl)
      std::cout << "Error " << std::get<0>(ori) << "!=" << xl << std::endl;
    assert(std::get<1>(ori) == bp);
  }

  for (uint16_t iy = 0; iy < 52 * 8; ++iy) {
    auto ori = localYori(iy);
    auto yl = Phase1::localY(iy);
    auto bp = Phase1::isBigPixY(iy);
    if (std::get<0>(ori) != yl)
      std::cout << "Error " << std::get<0>(ori) << "!=" << yl << std::endl;
    assert(std::get<1>(ori) == bp);
  }

  for (auto i = 0U; i < Phase1::numberOfLayers; ++i) {
    std::cout << "layer " << i << "\", [" << Phase1::layerStart[i] << ", " << Phase1::layerStart[i + 1] << ") "
              << Phase1::layerStart[i + 1] - Phase1::layerStart[i] << std::endl;
  }

  std::cout << "maxModuleStide layerIndexSize " << maxModuleStride<Phase1> << ' '
            << layerIndexSize<Phase1> << std::endl;

  testLayer();

  kernel_testLayer<<<1, 1>>>();
  cudaCheck(cudaDeviceSynchronize());

  return 0;
}
