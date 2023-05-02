
#ifndef RecoLocalCalo_HGCalRecProducer_HFNoseTilesConstants_h
#define RecoLocalCalo_HGCalRecProducer_HFNoseTilesConstants_h

#include "DataFormats/Math/interface/constexpr_cmath.h"

#include <cstdint>
#include <array>
struct HFNoseTilesConstants {
  static constexpr float tileSize = 5.f;
  static constexpr float minDim1 = -110.f;
  static constexpr float maxDim1 = 110.f;
  static constexpr float minDim2 = -110.f;
  static constexpr float maxDim2 = 110.f;
  static constexpr int nColumns = reco::ceil((maxDim1 - minDim1) / tileSize);
  static constexpr int nRows = reco::ceil((maxDim2 - minDim2) / tileSize);
  static constexpr int nTiles = nColumns * nRows;
};

#endif