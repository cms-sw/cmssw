// Authors: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 03/2019

#ifndef RecoLocalCalo_HGCalRecAlgos_interface_HFNoseTilesConstants_h
#define RecoLocalCalo_HGCalRecAlgos_interface_HFNoseTilesConstants_h

#include <cstdint>
#include <array>

namespace hfnosetilesconstants {

  constexpr int32_t ceil(float num) {
    return (static_cast<float>(static_cast<int32_t>(num)) == num) ? static_cast<int32_t>(num)
                                                                  : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
  }

  struct TileConstants {
    // inner radius ~ 3cm , other radius ~ 110
    static constexpr float tileSize = 5.f;
    static constexpr float minX = -110.f;
    static constexpr float maxX = 110.f;
    static constexpr float minY = -110.f;
    static constexpr float maxY = 110.f;
    static constexpr int nColumns = hfnosetilesconstants::ceil((maxX - minX) / tileSize);
    static constexpr int nRows = hfnosetilesconstants::ceil((maxY - minY) / tileSize);
    static constexpr float tileSizeEtaPhi = 0.15f;
    static constexpr float minEta = -4.2f;
    static constexpr float maxEta = 4.2f;
    //To properly construct search box for cells in phi=[-3.15,-3.] and [3.,3.15], cells in phi=[3.,3.15] are copied to the first bin and cells in phi=[-3.15,-3.] are copied to the last bin
    static constexpr float minPhi = -3.3f;
    static constexpr float maxPhi = 3.3f;
    static constexpr int nColumnsEta = hfnosetilesconstants::ceil((maxEta - minEta) / tileSizeEtaPhi);
    static constexpr int nRowsPhi = hfnosetilesconstants::ceil((maxPhi - minPhi) / tileSizeEtaPhi);
    static constexpr int nTiles = nColumns * nRows + nColumnsEta * nRowsPhi;
  };
}  // namespace hfnosetilesconstants

#endif  // RecoLocalCalo_HGCalRecAlgos_interface_HFNoseTilesConstants_h
