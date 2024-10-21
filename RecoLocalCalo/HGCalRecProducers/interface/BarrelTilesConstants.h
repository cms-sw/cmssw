// Authors: Alessandro Brusamolino

#ifndef RecoLocalCalo_HGCalRecProducer_BarrelTilesConstants_h
#define RecoLocalCalo_HGCalRecProducer_BarrelTilesConstants_h

#include "DataFormats/Math/interface/constexpr_cmath.h"
#include <cmath>
#include <cstdint>
#include <array>

struct EBTilesConstants {
  static constexpr float cellWidthEta = 0.0175f;
  static constexpr float cellWidthPhi = cellWidthEta;
  static constexpr float tileSizeEtaPhi = cellWidthEta;
  static constexpr float minDim1 = -1.5f;
  static constexpr float maxDim1 = 1.5f;
  static constexpr float minDim2 = -M_PI;
  static constexpr float maxDim2 = M_PI;
  static constexpr int nColumns = reco::ceil((maxDim1 - minDim1) / tileSizeEtaPhi);
  static constexpr int nRows = reco::ceil(2. * M_PI / tileSizeEtaPhi);
  static constexpr int nTiles = nColumns * nRows;
  static constexpr float showerSigma = 0.5f;  // in unit of xtals
};

struct HBTilesConstants {
  static constexpr float cellWidthEta = 0.087f;
  static constexpr float cellWidthPhi = cellWidthEta;
  static constexpr float tileSizeEtaPhi = 5 * cellWidthEta;
  static constexpr float minDim1 = -1.5f;
  static constexpr float maxDim1 = 1.5f;
  static constexpr float minDim2 = -M_PI;
  static constexpr float maxDim2 = M_PI;
  static constexpr int nColumns = reco::ceil((maxDim1 - minDim1) / tileSizeEtaPhi);
  static constexpr int nRows = reco::ceil(2. * M_PI / tileSizeEtaPhi);
  static constexpr int nTiles = nColumns * nRows;
  static constexpr float showerSigma = 0.5f;  // in unit of xtals
};

#endif
