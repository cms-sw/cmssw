#ifndef DataFormats_HGCalReco_TICLHFNoseLayerTile_h
#define DataFormats_HGCalReco_TICLHFNoseLayerTile_h

#include "DataFormats/HGCalReco/interface/Common.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

class TICLHFNoseLayerTile {
public:
  void fill(double eta, double phi, unsigned int layerClusterId) {
    hfntile_[globalBin(eta, phi)].push_back(layerClusterId);
  }

  int etaBin(float eta) const {
    constexpr float etaRange = ticlHFNose::constants::maxEta - ticlHFNose::constants::minEta;
    static_assert(etaRange >= 0.f);
    float r = ticlHFNose::constants::nEtaBins / etaRange;
    int etaBin = (std::abs(eta) - ticlHFNose::constants::minEta) * r;
    etaBin = std::clamp(etaBin, 0, ticlHFNose::constants::nEtaBins - 1);
    return etaBin;
  }

  int phiBin(float phi) const {
    auto normPhi = normalizedPhi(phi);
    float r = ticlHFNose::constants::nPhiBins * M_1_PI * 0.5f;
    int phiBin = (normPhi + M_PI) * r;

    return phiBin;
  }

  int globalBin(int etaBin, int phiBin) const { return phiBin + etaBin * ticlHFNose::constants::nPhiBins; }

  int globalBin(double eta, double phi) const { return phiBin(phi) + etaBin(eta) * ticlHFNose::constants::nPhiBins; }

  void clear() {
    auto nBins = ticlHFNose::constants::nEtaBins * ticlHFNose::constants::nPhiBins;
    for (int j = 0; j < nBins; ++j)
      hfntile_[j].clear();
  }

  const std::vector<unsigned int>& operator[](int globalBinId) const { return hfntile_[globalBinId]; }

private:
  ticlHFNose::Tile hfntile_;
};

class TICLHFNoseLayerTiles {
public:
  // This class represents a collection of Tiles, one for each layer in
  // HGCAL. The layer numbering should account for both sides of HGCAL and is
  // not handled internally. It is the user's responsibility to properly
  // number the layers and consistently access them here.
  const TICLHFNoseLayerTile& operator[](int layer) const { return hfntiles_[layer]; }
  void fill(int layer, double eta, double phi, unsigned int layerClusterId) {
    hfntiles_[layer].fill(eta, phi, layerClusterId);
  }

private:
  ticlHFNose::Tiles hfntiles_;
};

#endif
