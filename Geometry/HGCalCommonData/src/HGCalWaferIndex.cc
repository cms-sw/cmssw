#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

const int kHGCalWaferUOffset = 0;
const int kHGCalWaferUMask = 0x1F;
const int kHGCalWaferUSignOffset = 5;
const int kHGCalWaferUSignMask = 0x1;
const int kHGCalWaferVOffset = 6;
const int kHGCalWaferVMask = 0x1F;
const int kHGCalWaferVSignOffset = 11;
const int kHGCalWaferVSignMask = 0x1;
const int kHGCalLayerOffset = 18;
const int kHGCalLayerMask = 0x1F;
const int kHGCalWaferCopyOffset = 0;
const int kHGCalWaferCopyMask = 0x7FFFF;
const int kHGCalLayerOldMask = 0x1000000;

int32_t HGCalWaferIndex::waferIndex(int32_t layer, int32_t waferU, int32_t waferV, bool old) {
  int32_t id(0);
  if (old) {
    id |= (((waferU & kHGCalWaferCopyMask) << kHGCalWaferCopyOffset) |
           ((layer & kHGCalLayerMask) << kHGCalLayerOffset) | kHGCalLayerOldMask);
  } else {
    int waferUabs(std::abs(waferU)), waferVabs(std::abs(waferV));
    int waferUsign = (waferU >= 0) ? 0 : 1;
    int waferVsign = (waferV >= 0) ? 0 : 1;
    id |= (((waferUabs & kHGCalWaferUMask) << kHGCalWaferUOffset) |
           ((waferUsign & kHGCalWaferUSignMask) << kHGCalWaferUSignOffset) |
           ((waferVabs & kHGCalWaferVMask) << kHGCalWaferVOffset) |
           ((waferVsign & kHGCalWaferVSignMask) << kHGCalWaferVSignOffset) |
           ((layer & kHGCalLayerMask) << kHGCalLayerOffset));
  }
  return id;
}

int HGCalWaferIndex::waferLayer(const int32_t id) { return (id >> kHGCalLayerOffset) & kHGCalLayerMask; }

int HGCalWaferIndex::waferU(const int32_t id) {
  int32_t iu = (id >> kHGCalWaferUOffset) & kHGCalWaferUMask;
  return (((id >> kHGCalWaferUSignOffset) & kHGCalWaferUSignMask) ? -iu : iu);
}

int HGCalWaferIndex::waferV(const int32_t id) {
  int32_t iv = (id >> kHGCalWaferVOffset) & kHGCalWaferVMask;
  return (((id >> kHGCalWaferVSignOffset) & kHGCalWaferVSignMask) ? -iv : iv);
}

int HGCalWaferIndex::waferCopy(const int32_t id) { return (id >> kHGCalWaferCopyOffset) & kHGCalWaferCopyMask; }

bool HGCalWaferIndex::waferFormat(const int32_t id) { return ((id & kHGCalLayerOldMask) == 0); }
