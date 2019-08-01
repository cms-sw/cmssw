// Authors: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 03/2019

#ifndef RecoLocalCalo_HGCalRecAlgos_interface_HGCalTilesConstants_h
#define RecoLocalCalo_HGCalRecAlgos_interface_HGCalTilesConstants_h

#include <cstdint>
#include <array>

namespace hgcaltilesconstants {

  constexpr int32_t ceil(float num) {
    return (static_cast<float>(static_cast<int32_t>(num)) == num) ? static_cast<int32_t>(num)
                                                                  : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
  }

  constexpr float tileSize = 5.f;
  constexpr float minX = -285.f;
  constexpr float maxX = 285.f;
  constexpr float minY = -285.f;
  constexpr float maxY = 285.f;
  constexpr int nColumns = hgcaltilesconstants::ceil((maxX - minX) / tileSize);
  constexpr int nRows = hgcaltilesconstants::ceil((maxY - minY) / tileSize);
  constexpr float tileSizeEtaPhi = 0.15f;
  constexpr float minEta = -3.f;
  constexpr float maxEta = 3.f;
  //To properly construct search box for cells in phi=[-3.15,-3.] and [3.,3.15], cells in phi=[3.,3.15] are copied to the first bin and cells in phi=[-3.15,-3.] are copied to the last bin
  constexpr float minPhi = -3.3f;
  constexpr float maxPhi = 3.3f;
  constexpr int nColumnsEta = hgcaltilesconstants::ceil((maxEta - minEta) / tileSizeEtaPhi);
  constexpr int nRowsPhi = hgcaltilesconstants::ceil((maxPhi - minPhi) / tileSizeEtaPhi);
  constexpr int nTiles = nColumns * nRows + nColumnsEta * nRowsPhi;

}  // namespace hgcaltilesconstants

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_HGCalTilesConstants_h
