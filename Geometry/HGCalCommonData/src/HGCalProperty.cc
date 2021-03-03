#include "Geometry/HGCalCommonData/interface/HGCalProperty.h"

int32_t HGCalProperty::waferProperty(const int32_t thick, const int32_t part, const int32_t orient) {
  return (((thick % HGCalProperty::kHGCalFactor) * HGCalProperty::kHGCalOffsetThick) +
          ((part % HGCalProperty::kHGCalFactor) * HGCalProperty::kHGCalOffsetPartial) +
          ((orient % HGCalProperty::kHGCalFactor) * HGCalProperty::kHGCalOffsetOrient));
}

int32_t HGCalProperty::waferThick(const int32_t property) {
  return ((property / HGCalProperty::kHGCalOffsetThick) % HGCalProperty::kHGCalFactor);
}

int32_t HGCalProperty::waferPartial(const int32_t property) {
  return ((property / HGCalProperty::kHGCalOffsetPartial) % HGCalProperty::kHGCalFactor);
}

int32_t HGCalProperty::waferOrient(const int32_t property) {
  return ((property / HGCalProperty::kHGCalOffsetOrient) % HGCalProperty::kHGCalFactor);
}

int32_t HGCalProperty::tileProperty(const int32_t type, const int32_t sipm) {
  return (((type % HGCalProperty::kHGCalFactor) * HGCalProperty::kHGCalOffsetType) +
          ((sipm % HGCalProperty::kHGCalFactor) * HGCalProperty::kHGCalOffsetSiPM));
}

int32_t HGCalProperty::tileType(const int32_t property) {
  return ((property / HGCalProperty::kHGCalOffsetType) % HGCalProperty::kHGCalFactor);
}

int32_t HGCalProperty::tileSiPM(const int32_t property) {
  return ((property / HGCalProperty::kHGCalOffsetSiPM) % HGCalProperty::kHGCalFactor);
}
