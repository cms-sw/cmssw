// Authors: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 03/2019

#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalLayerTiles
#define RecoLocalCalo_HGCalRecAlgos_HGCalLayerTiles

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalTilesConstants.h"

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cassert>

class HGCalLayerTiles {
public:

  void fill(const std::vector<float>& x, const std::vector<float>& y) {
    auto cellsSize = x.size();
    for (unsigned int i = 0; i < cellsSize; ++i) {
      tiles_[getGlobalBin(x[i], y[i])].push_back(i);
    }
  }

  int getXBin(float x) const {
    constexpr float xRange = hgcaltilesconstants::maxX - hgcaltilesconstants::minX;
    static_assert(xRange >= 0.);
    constexpr float r = hgcaltilesconstants::nColumns / xRange;
    int xBin = (x - hgcaltilesconstants::minX) * r;
    xBin = std::clamp(xBin, 0, hgcaltilesconstants::nColumns);
    return xBin;
  }

  int getYBin(float y) const {
    constexpr float yRange = hgcaltilesconstants::maxY - hgcaltilesconstants::minY;
    static_assert(yRange >= 0.);
    constexpr float r = hgcaltilesconstants::nRows / yRange;
    int yBin = (y - hgcaltilesconstants::minY) * r;
    yBin = std::clamp(yBin, 0, hgcaltilesconstants::nRows);
    return yBin;
  }

  int getGlobalBin(float x, float y) const { return getXBin(x) + getYBin(y) * hgcaltilesconstants::nColumns; }

  int getGlobalBinByBin(int xBin, int yBin) const { return xBin + yBin * hgcaltilesconstants::nColumns; }

  std::array<int, 4> searchBox(float xMin, float xMax, float yMin, float yMax) const {
    int xBinMin = getXBin(xMin);
    int xBinMax = getXBin(xMax);
    int yBinMin = getYBin(yMin);
    int yBinMax = getYBin(yMax);
    return std::array<int, 4>({{xBinMin, xBinMax, yBinMin, yBinMax}});
  }

  void clear() {
    for (auto& t : tiles_)
      t.clear();
  }

  const std::vector<int>& operator[](int globalBinId) const { return tiles_[globalBinId]; }

private:
  std::array<std::vector<int>, hgcaltilesconstants::nColumns * hgcaltilesconstants::nRows  > tiles_;
};

#endif
