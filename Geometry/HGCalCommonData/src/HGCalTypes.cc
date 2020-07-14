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

int32_t HGCalTypes::getUnpackedType(int copy) { return (copy / factype_); }

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
