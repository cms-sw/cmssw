// Authors: Marco Rovere, Felice Pantaleo - marco.rovere@cern.ch, felice.pantaleo@cern.ch
// Date: 05/2019

#ifndef DataFormats_HGCalReco_TICLLayerTile_h
#define DataFormats_HGCalReco_TICLLayerTile_h

#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

class TICLLayerTile {
public:
  void fill(double eta, double phi, unsigned int layerClusterId) {
    tile_[globalBin(eta, phi)].push_back(layerClusterId);
  }

  int etaBin(float eta) const {
    constexpr float etaRange = ticl::constants::maxEta - ticl::constants::minEta;
    static_assert(etaRange >= 0.f);
    float r = ticl::constants::nEtaBins / etaRange;
    int etaBin = (std::abs(eta) - ticl::constants::minEta) * r;
    etaBin = std::clamp(etaBin, 0, ticl::constants::nEtaBins - 1);
    return etaBin;
  }

  int phiBin(float phi) const {
    auto normPhi = normalizedPhi(phi);
    float r = ticl::constants::nPhiBins * M_1_PI * 0.5f;
    int phiBin = (normPhi + M_PI) * r;

    return phiBin;
  }

  int globalBin(int etaBin, int phiBin) const { return phiBin + etaBin * ticl::constants::nPhiBins; }

  int globalBin(double eta, double phi) const { return phiBin(phi) + etaBin(eta) * ticl::constants::nPhiBins; }

  void clear() {
    auto nBins = ticl::constants::nEtaBins * ticl::constants::nPhiBins;
    for (int j = 0; j < nBins; ++j)
      tile_[j].clear();
  }

  const std::vector<unsigned int>& operator[](int globalBinId) const { return tile_[globalBinId]; }

private:
  ticl::Tile tile_;
};

class TICLLayerTiles {
public:
  // This class represents a collection of Tiles, one for each layer in
  // HGCAL. The layer numbering should account for both sides of HGCAL and is
  // not handled internally. It is the user's responsibility to properly
  // number the layers and consistently access them here.
  const TICLLayerTile& operator[](int layer) const { return tiles_[layer]; }
  void fill(int layer, double eta, double phi, unsigned int layerClusterId) {
    tiles_[layer].fill(eta, phi, layerClusterId);
  }

private:
  ticl::Tiles tiles_;
};

#endif
