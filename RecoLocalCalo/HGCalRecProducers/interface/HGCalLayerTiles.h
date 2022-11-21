// Authors: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 03/2019

#ifndef RecoLocalCalo_HGCalRecProducers_HGCalLayerTiles_h
#define RecoLocalCalo_HGCalRecProducers_HGCalLayerTiles_h

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalTilesConstants.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HFNoseTilesConstants.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cassert>

template <typename T>
class HGCalLayerTilesT {
public:
  typedef T type;
  void fill(const std::vector<float>& x,
            const std::vector<float>& y,
            const std::vector<float>& eta,
            const std::vector<float>& phi,
            const std::vector<bool>& isSi) {
    auto cellsSize = x.size();
    for (unsigned int i = 0; i < cellsSize; ++i) {
      tiles_[getGlobalBin(x[i], y[i])].push_back(i);
      if (!isSi[i]) {
        tiles_[getGlobalBinEtaPhi(eta[i], phi[i])].push_back(i);
      }
    }
  }

  int getXBin(float x) const {
    constexpr float xRange = T::maxX - T::minX;
    static_assert(xRange >= 0.);
    constexpr float r = T::nColumns / xRange;
    int xBin = (x - T::minX) * r;
    xBin = std::clamp(xBin, 0, T::nColumns - 1);
    return xBin;
  }

  int getYBin(float y) const {
    constexpr float yRange = T::maxY - T::minY;
    static_assert(yRange >= 0.);
    constexpr float r = T::nRows / yRange;
    int yBin = (y - T::minY) * r;
    yBin = std::clamp(yBin, 0, T::nRows - 1);
    return yBin;
  }

  int getEtaBin(float eta) const {
    constexpr float etaRange = T::maxEta - T::minEta;
    static_assert(etaRange >= 0.);
    constexpr float r = T::nColumnsEta / etaRange;
    int etaBin = (eta - T::minEta) * r;
    etaBin = std::clamp(etaBin, 0, T::nColumnsEta - 1);
    return etaBin;
  }

  int getPhiBin(float phi) const {
    auto normPhi = normalizedPhi(phi);
    constexpr float r = T::nRowsPhi * M_1_PI * 0.5f;
    int phiBin = (normPhi + M_PI) * r;
    return phiBin;
  }

  int mPiPhiBin = getPhiBin(-M_PI);
  int pPiPhiBin = getPhiBin(M_PI);

  int getGlobalBin(float x, float y) const { return getXBin(x) + getYBin(y) * T::nColumns; }

  int getGlobalBinByBin(int xBin, int yBin) const { return xBin + yBin * T::nColumns; }

  int getGlobalBinEtaPhi(float eta, float phi) const {
    return T::nColumns * T::nRows + getEtaBin(eta) + getPhiBin(phi) * T::nColumnsEta;
  }

  int getGlobalBinByBinEtaPhi(int etaBin, int phiBin) const {
    return T::nColumns * T::nRows + etaBin + phiBin * T::nColumnsEta;
  }

  std::array<int, 4> searchBox(float xMin, float xMax, float yMin, float yMax) const {
    int xBinMin = getXBin(xMin);
    int xBinMax = getXBin(xMax);
    int yBinMin = getYBin(yMin);
    int yBinMax = getYBin(yMax);
    return std::array<int, 4>({{xBinMin, xBinMax, yBinMin, yBinMax}});
  }

  std::array<int, 4> searchBoxEtaPhi(float etaMin, float etaMax, float phiMin, float phiMax) const {
    if (etaMax - etaMin < 0) {
      return std::array<int, 4>({{0, 0, 0, 0}});
    }
    int etaBinMin = getEtaBin(etaMin);
    int etaBinMax = getEtaBin(etaMax);
    int phiBinMin = getPhiBin(phiMin);
    int phiBinMax = getPhiBin(phiMax);
    // If the search window cross the phi-bin boundary, add T::nPhiBins to the
    // MAx value. This guarantees that the caller can perform a valid doule
    // loop on eta and phi. It is the caller responsibility to perform a module
    // operation on the phiBin values returned by this function, to explore the
    // correct bins.
    if (phiBinMax < phiBinMin) {
      phiBinMax += T::nRowsPhi;
    }

    return std::array<int, 4>({{etaBinMin, etaBinMax, phiBinMin, phiBinMax}});
  }

  void clear() {
    for (auto& t : tiles_)
      t.clear();
  }

  const std::vector<int>& operator[](int globalBinId) const { return tiles_[globalBinId]; }

private:
  std::array<std::vector<int>, T::nTiles> tiles_;
};

using HGCalLayerTiles = HGCalLayerTilesT<HGCalTilesConstants>;
using HFNoseLayerTiles = HGCalLayerTilesT<HFNoseTilesConstants>;
#endif
