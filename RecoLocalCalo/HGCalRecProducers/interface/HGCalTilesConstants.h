
// Authors: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 03/2019

#ifndef RecoLocalCalo_HGCalRecProducer_HGCalTilesConstants_h
#define RecoLocalCalo_HGCalRecProducer_HGCalTilesConstants_h

#include "DataFormats/Math/interface/constexpr_cmath.h"
#include <cmath>
#include <cstdint>
#include <array>

struct HGCalSiliconTilesConstants {
  static constexpr float tileSize = 5.f;
  static constexpr float minDim = -285.f;
  static constexpr float maxDim = 285.f;
  static constexpr int nColumns = reco::ceil((maxDim - minDim) / tileSize);
  static constexpr int nTiles = nColumns * nColumns;
  static constexpr int nRowsPhi = 0;
};

struct HGCalScintillatorTilesConstants {
  static constexpr float tileSize = 0.15f;
  static constexpr float minDim = -3.f;
  static constexpr float maxDim = 3.f;
  static constexpr int nColumns = reco::ceil((maxDim - minDim) / tileSize);
  static constexpr int nRowsPhi = reco::ceil(2. * M_PI / tileSize);
  static constexpr int nTiles = nColumns * nRowsPhi;
};

#endif