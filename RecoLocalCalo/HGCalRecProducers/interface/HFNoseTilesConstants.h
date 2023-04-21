
#ifndef RecoLocalCalo_HGCalRecProducer_HFNoseTilesConstants_h
#define RecoLocalCalo_HGCalRecProducer_HFNoseTilesConstants_h

#include "DataFormats/Math/interface/constexpr_cmath.h"

#include <cstdint>
#include <array>
struct HFNoseTilesConstants {
  static constexpr float tileSize = 5.f;
  static constexpr float minDim = -110.f;
  static constexpr float maxDim = 110.f;
  static constexpr int nColumns = reco::ceil((maxDim - minDim) / tileSize);
  static constexpr int nRowsPhi = 0;
  static constexpr int nTiles = nColumns * nColumns;
};

#endif