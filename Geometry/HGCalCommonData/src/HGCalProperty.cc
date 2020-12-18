#include "Geometry/HGCalCommonData/interface/HGCalProperty.h"

const int32_t kHGCalFactor = 10;
const int32_t kHGCalOffsetThick = 1;
const int32_t kHGCalOffsetPartial = 10;
const int32_t kHGCalOffsetOrient = 100;
const int32_t kHGCalOffsetType = 1;
const int32_t kHGCalOffsetSiPM = 10;

int32_t HGCalProperty::waferProperty(const int32_t thick, const int32_t part, const int32_t orient) {
  return (((thick % kHGCalFactor) * kHGCalOffsetThick) + ((part % kHGCalFactor) * kHGCalOffsetPartial) +
          ((orient % kHGCalFactor) * kHGCalOffsetOrient));
}

int32_t HGCalProperty::waferThick(const int32_t property) { return ((property / kHGCalOffsetThick) % kHGCalFactor); }

int32_t HGCalProperty::waferPartial(const int32_t property) {
  return ((property / kHGCalOffsetPartial) % kHGCalFactor);
}

int32_t HGCalProperty::waferOrient(const int32_t property) { return ((property / kHGCalOffsetOrient) % kHGCalFactor); }

int32_t HGCalProperty::tileProperty(const int32_t type, const int32_t sipm) {
  return (((type % kHGCalFactor) * kHGCalOffsetType) + ((sipm % kHGCalFactor) * kHGCalOffsetSiPM));
}

int32_t HGCalProperty::tileType(const int32_t property) { return ((property / kHGCalOffsetType) % kHGCalFactor); }

int32_t HGCalProperty::tileSiPM(const int32_t property) { return ((property / kHGCalOffsetSiPM) % kHGCalFactor); }
