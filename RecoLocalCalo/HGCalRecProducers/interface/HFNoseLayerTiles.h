#ifndef RecoLocalCalo_HGCalRecAlgos_HFNoseLayerTiles
#define RecoLocalCalo_HGCalRecAlgos_HFNoseLayerTiles

#include "RecoLocalCalo/HGCalRecProducers/interface/HFNoseTilesConstants.h"

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cassert>

class HFNoseLayerTiles {
public:
  void fill(const std::vector<float>& x,
            const std::vector<float>& y,
            const std::vector<float>& eta,
            const std::vector<float>& phi,
            const std::vector<bool>& isSi) {
    auto cellsSize = x.size();
    for (unsigned int i = 0; i < cellsSize; ++i) {
      hfntiles_[getGlobalBin(x[i], y[i])].push_back(i);
      if (!isSi[i]) {
        hfntiles_[getGlobalBinEtaPhi(eta[i], phi[i])].push_back(i);
        // Copy cells in phi=[-3.15,-3.] to the last bin
        if (getPhiBin(phi[i]) == mPiPhiBin) {
          hfntiles_[getGlobalBinEtaPhi(eta[i], phi[i] + 2 * M_PI)].push_back(i);
        }
        // Copy cells in phi=[3.,3.15] to the first bin
        if (getPhiBin(phi[i]) == pPiPhiBin) {
          hfntiles_[getGlobalBinEtaPhi(eta[i], phi[i] - 2 * M_PI)].push_back(i);
        }
      }
    }
  }

  int getXBin(float x) const {
    constexpr float xRange = hfnosetilesconstants::maxX - hfnosetilesconstants::minX;
    static_assert(xRange >= 0.);
    constexpr float r = hfnosetilesconstants::nColumns / xRange;
    int xBin = (x - hfnosetilesconstants::minX) * r;
    xBin = std::clamp(xBin, 0, hfnosetilesconstants::nColumns - 1);
    return xBin;
  }

  int getYBin(float y) const {
    constexpr float yRange = hfnosetilesconstants::maxY - hfnosetilesconstants::minY;
    static_assert(yRange >= 0.);
    constexpr float r = hfnosetilesconstants::nRows / yRange;
    int yBin = (y - hfnosetilesconstants::minY) * r;
    yBin = std::clamp(yBin, 0, hfnosetilesconstants::nRows - 1);
    return yBin;
  }

  int getEtaBin(float eta) const {
    constexpr float etaRange = hfnosetilesconstants::maxEta - hfnosetilesconstants::minEta;
    static_assert(etaRange >= 0.);
    constexpr float r = hfnosetilesconstants::nColumnsEta / etaRange;
    int etaBin = (eta - hfnosetilesconstants::minEta) * r;
    etaBin = std::clamp(etaBin, 0, hfnosetilesconstants::nColumnsEta - 1);
    return etaBin;
  }

  int getPhiBin(float phi) const {
    constexpr float phiRange = hfnosetilesconstants::maxPhi - hfnosetilesconstants::minPhi;
    static_assert(phiRange >= 0.);
    constexpr float r = hfnosetilesconstants::nRowsPhi / phiRange;
    int phiBin = (phi - hfnosetilesconstants::minPhi) * r;
    phiBin = std::clamp(phiBin, 0, hfnosetilesconstants::nRowsPhi - 1);
    return phiBin;
  }

  int mPiPhiBin = getPhiBin(-M_PI);
  int pPiPhiBin = getPhiBin(M_PI);

  int getGlobalBin(float x, float y) const { return getXBin(x) + getYBin(y) * hfnosetilesconstants::nColumns; }

  int getGlobalBinByBin(int xBin, int yBin) const { return xBin + yBin * hfnosetilesconstants::nColumns; }

  int getGlobalBinEtaPhi(float eta, float phi) const {
    return hfnosetilesconstants::nColumns * hfnosetilesconstants::nRows + getEtaBin(eta) +
           getPhiBin(phi) * hfnosetilesconstants::nColumnsEta;
  }

  int getGlobalBinByBinEtaPhi(int etaBin, int phiBin) const {
    return hfnosetilesconstants::nColumns * hfnosetilesconstants::nRows + etaBin +
           phiBin * hfnosetilesconstants::nColumnsEta;
  }

  std::array<int, 4> searchBox(float xMin, float xMax, float yMin, float yMax) const {
    int xBinMin = getXBin(xMin);
    int xBinMax = getXBin(xMax);
    int yBinMin = getYBin(yMin);
    int yBinMax = getYBin(yMax);
    return std::array<int, 4>({{xBinMin, xBinMax, yBinMin, yBinMax}});
  }

  std::array<int, 4> searchBoxEtaPhi(float etaMin, float etaMax, float phiMin, float phiMax) const {
    int etaBinMin = getEtaBin(etaMin);
    int etaBinMax = getEtaBin(etaMax);
    int phiBinMin = getPhiBin(phiMin);
    int phiBinMax = getPhiBin(phiMax);
    return std::array<int, 4>({{etaBinMin, etaBinMax, phiBinMin, phiBinMax}});
  }

  void clear() {
    for (auto& t : hfntiles_)
      t.clear();
  }

  const std::vector<int>& operator[](int globalBinId) const { return hfntiles_[globalBinId]; }

private:
  std::array<std::vector<int>, hfnosetilesconstants::nTiles> hfntiles_;
};

#endif
