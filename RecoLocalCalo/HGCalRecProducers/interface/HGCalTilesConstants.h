// Authors: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 03/2019

#ifndef RecoLocalCalo_HGCalRecProducer_HGCalTilesConstants_h
#define RecoLocalCalo_HGCalRecProducer_HGCalTilesConstants_h

#include "DataFormats/Math/interface/constexpr_cmath.h"
#include <cstdint>
#include <array>

struct HGCalTilesConstants {
  static constexpr float tileSize = 5.f;
  static constexpr float minX = -285.f;
  static constexpr float maxX = 285.f;
  static constexpr float minY = -285.f;
  static constexpr float maxY = 285.f;
  static constexpr int nColumns = reco::ceil((maxX - minX) / tileSize);
  static constexpr int nRows = reco::ceil((maxY - minY) / tileSize);
  static constexpr float tileSizeEtaPhi = 0.15f;
  static constexpr float minEta = -3.f;
  static constexpr float maxEta = 3.f;
  //To properly construct search box for cells in phi=[-3.15,-3.] and [3.,3.15], cells in phi=[3.,3.15]
  //are copied to the first bin and cells in phi=[-3.15,-3.] are copied to the last bin
  static constexpr float minPhi = -3.3f;
  static constexpr float maxPhi = 3.3f;
  static constexpr int nColumnsEta = reco::ceil((maxEta - minEta) / tileSizeEtaPhi);
  static constexpr int nRowsPhi = reco::ceil((maxPhi - minPhi) / tileSizeEtaPhi);
  static constexpr int nTiles = nColumns * nRows + nColumnsEta * nRowsPhi;
};

#endif
