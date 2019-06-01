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

  constexpr float minX = -285.f;
  constexpr float maxX = 285.f;
  constexpr float minY = -285.f;
  constexpr float maxY = 285.f;
  constexpr float tileSize = 5.f;
  constexpr int nColumns = hgcaltilesconstants::ceil((maxX - minX) / tileSize);
  constexpr int nRows = hgcaltilesconstants::ceil((maxY - minY) / tileSize);

}  // namespace hgcaltilesconstants

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_HGCalTilesConstants_h