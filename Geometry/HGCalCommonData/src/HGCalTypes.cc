#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"

int32_t HGCalTypes::packTypeUV(int type, int u, int v) {
  int32_t iu = std::abs(u);
  int32_t iv = std::abs(v);
  int32_t copy = type * factype_ + iv * facv_ + iu;
  if (u < 0)
    copy += signu_;
  if (v < 0)
    copy += signv_;
  return copy;
}

int32_t HGCalTypes::getUnpackedType(int copy) { return ((copy / factype_) % maxtype_); }

int32_t HGCalTypes::getUnpackedU(int copy) {
  int32_t iu = (copy % maxuv_);
  int32_t u = (((copy / signu_) % maxsign_) > 0) ? -iu : iu;
  return u;
}

int32_t HGCalTypes::getUnpackedV(int copy) {
  int32_t iv = ((copy / facv_) % maxuv_);
  int32_t v = (((copy / signv_) % maxsign_) > 0) ? -iv : iv;
  return v;
}

int32_t HGCalTypes::packCellTypeUV(int type, int u, int v) { return (type * faccelltype_ + v * faccell_ + u); }

int32_t HGCalTypes::getUnpackedCellType(int copy) { return ((copy / faccelltype_) % faccell_); }

int32_t HGCalTypes::getUnpackedCellU(int copy) { return (copy % faccell_); }

int32_t HGCalTypes::getUnpackedCellV(int copy) { return ((copy / faccell_) % faccell_); }

int32_t HGCalTypes::packCellType6(int type, int cell) { return (type * faccell6_ + cell); }

int32_t HGCalTypes::getUnpackedCellType6(int id) { return (id / faccell6_); }

int32_t HGCalTypes::getUnpackedCell6(int id) { return (id % faccell6_); }

int32_t HGCalTypes::layerType(int type) {
  static constexpr int32_t types[5] = {HGCalTypes::WaferCenter,
                                       HGCalTypes::WaferCenterB,
                                       HGCalTypes::CornerCenterYp,
                                       HGCalTypes::CornerCenterYm,
                                       HGCalTypes::WaferCenterR};
  return ((type >= 0 && type < 5) ? types[type] : HGCalTypes::WaferCenter);
}
