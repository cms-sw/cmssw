// Authors: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 03/2019

#ifndef __RecoLocalCalo_HGCalRecAlgos_HGCalTilesConstants_h__
#define __RecoLocalCalo_HGCalRecAlgos_HGCalTilesConstants_h__

#include <cstdint>
#include <array>

struct HGCalTilesConstants {
  static constexpr float tileSize = 5.f;
  static constexpr float minX = -285.f;
  static constexpr float maxX = 285.f;
  static constexpr float minY = -285.f;
  static constexpr float maxY = 285.f;
  static constexpr int nColumns = static_cast<int>(std::ceil((maxX - minX) / tileSize));
  static constexpr int nRows = static_cast<int>(std::ceil((maxY - minY) / tileSize));
  static constexpr float tileSizeEtaPhi = 0.15f;
  static constexpr float minEta = -3.f;
  static constexpr float maxEta = 3.f;
  //To properly construct search box for cells in phi=[-3.15,-3.] and [3.,3.15], cells in phi=[3.,3.15] are copied to the first bin and cells in phi=[-3.15,-3.] are copied to the last bin
  static constexpr float minPhi = -3.3f;
  static constexpr float maxPhi = 3.3f;
  static constexpr int nColumnsEta = static_cast<int>(std::ceil((maxEta - minEta) / tileSizeEtaPhi));
  static constexpr int nRowsPhi = static_cast<int>(std::ceil((maxPhi - minPhi) / tileSizeEtaPhi));
  static constexpr int nTiles = nColumns * nRows + nColumnsEta * nRowsPhi;
};

#endif
