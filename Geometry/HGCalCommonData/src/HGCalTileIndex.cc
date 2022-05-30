#include "Geometry/HGCalCommonData/interface/HGCalProperty.h"
#include "Geometry/HGCalCommonData/interface/HGCalTileIndex.h"

int32_t HGCalTileIndex::tileIndex(int32_t layer, int32_t ring, int32_t phi) {
  int32_t id(0);
  id |= (((phi & HGCalProperty::kHGCalPhiMask) << HGCalProperty::kHGCalPhiOffset) |
         ((ring & HGCalProperty::kHGCalRingMask) << HGCalProperty::kHGCalRingOffset) |
         ((layer & HGCalProperty::kHGCalLayerMask) << HGCalProperty::kHGCalLayerOffset));
  return id;
}

int32_t HGCalTileIndex::tileLayer(int32_t id) {
  return ((id >> HGCalProperty::kHGCalLayerOffset) & HGCalProperty::kHGCalLayerMask);
}

int32_t HGCalTileIndex::tileRing(int32_t id) {
  return ((id >> HGCalProperty::kHGCalRingOffset) & HGCalProperty::kHGCalRingMask);
}

int32_t HGCalTileIndex::tilePhi(int32_t id) {
  return ((id >> HGCalProperty::kHGCalPhiOffset) & HGCalProperty::kHGCalPhiMask);
}

int32_t HGCalTileIndex::tileCassette(int32_t iphi, int32_t phiOffset, int32_t nphiCassette, int32_t cassettes) {
  int32_t cassette(0);
  if (nphiCassette > 1) {
    cassette = (iphi - phiOffset) / nphiCassette;
    if (cassette < 0)
      cassette += cassettes;
    else if (cassette >= cassettes)
      cassette = cassettes - 1;
  }
  return (cassette + 1);
}

int32_t HGCalTileIndex::tileProperty(int32_t type, int32_t sipm) {
  return (((type % HGCalProperty::kHGCalFactor) * HGCalProperty::kHGCalOffsetType) +
          ((sipm % HGCalProperty::kHGCalFactor) * HGCalProperty::kHGCalOffsetSiPM));
}

int32_t HGCalTileIndex::tileType(int32_t property) {
  return ((property / HGCalProperty::kHGCalOffsetType) % HGCalProperty::kHGCalFactor);
}

int32_t HGCalTileIndex::tileSiPM(int32_t property) {
  return ((property / HGCalProperty::kHGCalOffsetSiPM) % HGCalProperty::kHGCalFactor);
}

int32_t HGCalTileIndex::tilePack(int32_t ly, int32_t k1, int32_t k2) {
  return (
      ((ly % HGCalProperty::kHGCalTilePack) * HGCalProperty::kHGCalTilePack + (k1 % HGCalProperty::kHGCalTilePack)) *
          HGCalProperty::kHGCalTilePack +
      (k2 % HGCalProperty::kHGCalTilePack));
}

std::tuple<int32_t, int32_t, int32_t> HGCalTileIndex::tileUnpack(int32_t index) {
  int32_t ly =
      (index / (HGCalProperty::kHGCalTilePack * HGCalProperty::kHGCalTilePack)) % HGCalProperty::kHGCalTilePack;
  int32_t k1 = (index / HGCalProperty::kHGCalTilePack) % HGCalProperty::kHGCalTilePack;
  int32_t k2 = (index % HGCalProperty::kHGCalTilePack);
  return std::make_tuple(ly, k1, k2);
}

bool HGCalTileIndex::tileExist(const int32_t* hex, int32_t zside, int32_t iphi) {
  int32_t phi(iphi - 1);
  if (zside > 0) {
    phi += HGCalProperty::kHGCalTilePhisBy2;
    if (phi >= HGCalProperty::kHGCalTilePhis)
      phi -= HGCalProperty::kHGCalTilePhis;
  }
  int32_t jj = phi % HGCalProperty::kHGCalTilePhisBy3;
  int32_t iw = jj / HGCalProperty::kHGCalTilePhisBy12;
  int32_t ibit = HGCalProperty::kHGCalTilePhisBy12 - (jj % HGCalProperty::kHGCalTilePhisBy12) - 1;
  bool ok = (hex[iw] & (1 << ibit));
  return ok;
}
