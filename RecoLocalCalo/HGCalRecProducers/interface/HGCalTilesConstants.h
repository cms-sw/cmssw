// Authors: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 03/2019
// Copyright CERN

#ifndef RecoLocalCalo_HGCalRecAlgos_interface_HGCalTilesConstants_h
#define RecoLocalCalo_HGCalRecAlgos_interface_HGCalTilesConstants_h

#include <cstdint>
#include <array>

namespace hgcalTilesConstants {

  constexpr int32_t ceil(float num) {
    return (static_cast<float>(static_cast<int32_t>(num)) == num) ? static_cast<int32_t>(num)
                                                                  : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
  }

  // first is for CE-E, second for CE-H in cm
  constexpr float minX = -265.f;
  constexpr float maxX = 265.f;
  constexpr float minY = -265.f;
  constexpr float maxY = 265.f;
  constexpr float tileSize = 5.f;
  constexpr int nColumns = hgcalTilesConstants::ceil(maxX - minX / tileSize);
  constexpr int nRows = hgcalTilesConstants::ceil(maxY - minY / tileSize);

}  // namespace hgcalTilesConstants

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_HGCalTilesConstants_h