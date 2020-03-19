#ifndef RecoLocalCalo_HGCalRecProducer_HFNoseTilesConstants_h
#define RecoLocalCalo_HGCalRecProducer_HFNoseTilesConstants_h

#include "DataFormats/Math/interface/constexpr_cmath.h"

#include <cstdint>
#include <array>

struct HFNoseTilesConstants {
  // inner radius ~ 3cm , other radius ~ 110
  static constexpr float tileSize = 5.f;
  static constexpr float minX = -110.f;
  static constexpr float maxX = 110.f;
  static constexpr float minY = -110.f;
  static constexpr float maxY = 110.f;
  static constexpr int nColumns = reco::ceil((maxX - minX) / tileSize);
  static constexpr int nRows = reco::ceil((maxY - minY) / tileSize);
  static constexpr float tileSizeEtaPhi = 0.15f;
  static constexpr float minEta = -4.2f;
  static constexpr float maxEta = 4.2f;
  //To properly construct search box for cells in phi=[-3.15,-3.] and [3.,3.15], cells in phi=[3.,3.15]
  //are copied to the first bin and cells in phi=[-3.15,-3.] are copied to the last bin
  static constexpr float minPhi = -3.3f;
  static constexpr float maxPhi = 3.3f;
  static constexpr int nColumnsEta = reco::ceil((maxEta - minEta) / tileSizeEtaPhi);
  static constexpr int nRowsPhi = reco::ceil((maxPhi - minPhi) / tileSizeEtaPhi);
  static constexpr int nTiles = nColumns * nRows + nColumnsEta * nRowsPhi;
};

#endif
