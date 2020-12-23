#ifndef Geometry_HGCalCommonData_HGCalProperty_h
#define Geometry_HGCalCommonData_HGCalProperty_h

#include <cmath>
#include <cstdint>

namespace HGCalProperty {
  int32_t waferProperty(const int32_t thick, const int32_t partial, const int32_t orient);
  int32_t waferThick(const int32_t property);
  int32_t waferPartial(const int32_t property);
  int32_t waferOrient(const int32_t property);
  int32_t tileProperty(const int32_t type, const int32_t sipm);
  int32_t tileType(const int32_t property);
  int32_t tileSiPM(const int32_t property);

  const int32_t kHGCalWaferUOffset = 0;
  const int32_t kHGCalWaferUMask = 0x1F;
  const int32_t kHGCalWaferUSignOffset = 5;
  const int32_t kHGCalWaferUSignMask = 0x1;
  const int32_t kHGCalWaferVOffset = 6;
  const int32_t kHGCalWaferVMask = 0x1F;
  const int32_t kHGCalWaferVSignOffset = 11;
  const int32_t kHGCalWaferVSignMask = 0x1;
  const int32_t kHGCalWaferCopyOffset = 0;
  const int32_t kHGCalWaferCopyMask = 0x7FFFF;
  const int32_t kHGCalLayerOldMask = 0x1000000;

  const int32_t kHGCalLayerOffset = 18;
  const int32_t kHGCalLayerMask = 0x1F;

  const int32_t kHGCalPhiOffset = 0;
  const int32_t kHGCalPhiMask = 0x1FF;
  const int32_t kHGCalRingOffset = 9;
  const int32_t kHGCalRingMask = 0x1FF;

  const int32_t kHGCalFactor = 10;
  const int32_t kHGCalOffsetThick = 1;
  const int32_t kHGCalOffsetPartial = 10;
  const int32_t kHGCalOffsetOrient = 100;
  const int32_t kHGCalOffsetType = 1;
  const int32_t kHGCalOffsetSiPM = 10;
};  // namespace HGCalProperty

#endif
