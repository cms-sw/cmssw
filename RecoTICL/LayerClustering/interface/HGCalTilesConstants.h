
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
  static constexpr float minDim1 = -285.f;
  static constexpr float maxDim1 = 285.f;
  static constexpr float minDim2 = -285.f;
  static constexpr float maxDim2 = 285.f;
  static constexpr int nColumns = reco::ceil((maxDim1 - minDim1) / tileSize);
  static constexpr int nRows = reco::ceil((maxDim2 - minDim2) / tileSize);
  static constexpr int nTiles = nColumns * nRows;
  static constexpr float invDim1BinSize = nColumns / (maxDim1 - minDim1);
  static constexpr float invDim2BinSize = nRows / (maxDim2 - minDim2);
  static constexpr int maxTileDepth = 64;  // For accelerators.
};

struct HGCalScintillatorTilesConstants {
  static constexpr float tileSize = 0.15f;
  static constexpr float minDim1 = -3.f;
  static constexpr float maxDim1 = 3.f;
  static constexpr float minDim2 = -3.f;
  static constexpr float maxDim2 = 3.f;
  static constexpr int nColumns = reco::ceil((maxDim1 - minDim1) / tileSize);
  static constexpr int nRows = reco::ceil(2. * M_PI / tileSize);
  static constexpr int nTiles = nColumns * nRows;
  static constexpr float invDim1BinSize = nColumns / (maxDim1 - minDim1);
  static constexpr float invDim2BinSize = nRows / (maxDim2 - minDim2);
  static constexpr int maxTileDepth = 32;  // For accelerators.
};

#endif
