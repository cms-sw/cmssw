// Authors: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 03/2019
// Copyright CERN

#ifndef RecoLocalCalo_HGCalRecAlgos_HGCalLayerTiles
#define RecoLocalCalo_HGCalRecAlgos_HGCalLayerTiles

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalTilesConstants.h"

#include <vector>
#include <array>
#include <cmath>

class HGCalLayerTiles {
public:
  HGCalLayerTiles() { tiles_.resize(hgcalTilesConstants::nColumns * hgcalTilesConstants::nRows); }

  void fill(const std::vector<float>& x, const std::vector<float>& y) {
    auto cellsSize = x.size();
    for (unsigned int i = 0; i < cellsSize; ++i) {
      tiles_[getGlobalBin(x[i], y[i])].push_back(i);
    }
  }

  int getXBin(float x) const {
    constexpr float xRange = hgcalTilesConstants::maxX - hgcalTilesConstants::minX;
    static_assert(xRange >= 0.);
    constexpr float r = hgcalTilesConstants::nColumns / xRange;
    int xBin = (x - hgcalTilesConstants::minX) * r;
    xBin = std::min(xBin, hgcalTilesConstants::nColumns);
    xBin = std::max(xBin, 0);
    return xBin;
  }

  int getYBin(float y) const {
    constexpr float yRange = hgcalTilesConstants::maxY - hgcalTilesConstants::minY;
    static_assert(yRange >= 0.);
    constexpr float r = hgcalTilesConstants::nRows / yRange;
    int yBin = (y - hgcalTilesConstants::minY) * r;
    yBin = std::min(yBin, hgcalTilesConstants::nRows);
    yBin = std::max(yBin, 0);
    return yBin;
  }

  int getGlobalBin(float x, float y) const { return getXBin(x) + getYBin(y) * hgcalTilesConstants::nColumns; }

  int getGlobalBinByBin(int xBin, int yBin) const { return xBin + yBin * hgcalTilesConstants::nColumns; }

  std::array<int, 4> searchBox(float xMin, float xMax, float yMin, float yMax) {
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

  std::vector<int>& operator[](int globalBinId) { return tiles_[globalBinId]; }

private:
  std::vector<std::vector<int> > tiles_;
};

#endif