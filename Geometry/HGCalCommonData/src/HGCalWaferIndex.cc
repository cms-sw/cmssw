#include "Geometry/HGCalCommonData/interface/HGCalProperty.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

int32_t HGCalWaferIndex::waferIndex(int32_t layer, int32_t waferU, int32_t waferV, bool old) {
  int32_t id(0);
  if (old) {
    id |= (((waferU & HGCalProperty::kHGCalWaferCopyMask) << HGCalProperty::kHGCalWaferCopyOffset) |
           ((layer & HGCalProperty::kHGCalLayerMask) << HGCalProperty::kHGCalLayerOffset) |
           HGCalProperty::kHGCalLayerOldMask);
  } else {
    int waferUabs(std::abs(waferU)), waferVabs(std::abs(waferV));
    int waferUsign = (waferU >= 0) ? 0 : 1;
    int waferVsign = (waferV >= 0) ? 0 : 1;
    id |= (((waferUabs & HGCalProperty::kHGCalWaferUMask) << HGCalProperty::kHGCalWaferUOffset) |
           ((waferUsign & HGCalProperty::kHGCalWaferUSignMask) << HGCalProperty::kHGCalWaferUSignOffset) |
           ((waferVabs & HGCalProperty::kHGCalWaferVMask) << HGCalProperty::kHGCalWaferVOffset) |
           ((waferVsign & HGCalProperty::kHGCalWaferVSignMask) << HGCalProperty::kHGCalWaferVSignOffset) |
           ((layer & HGCalProperty::kHGCalLayerMask) << HGCalProperty::kHGCalLayerOffset));
  }
  return id;
}

int32_t HGCalWaferIndex::waferLayer(const int32_t id) {
  return (id >> HGCalProperty::kHGCalLayerOffset) & HGCalProperty::kHGCalLayerMask;
}

int32_t HGCalWaferIndex::waferU(const int32_t id) {
  int32_t iu = (id >> HGCalProperty::kHGCalWaferUOffset) & HGCalProperty::kHGCalWaferUMask;
  return (((id >> HGCalProperty::kHGCalWaferUSignOffset) & HGCalProperty::kHGCalWaferUSignMask) ? -iu : iu);
}

int32_t HGCalWaferIndex::waferV(const int32_t id) {
  int32_t iv = (id >> HGCalProperty::kHGCalWaferVOffset) & HGCalProperty::kHGCalWaferVMask;
  return (((id >> HGCalProperty::kHGCalWaferVSignOffset) & HGCalProperty::kHGCalWaferVSignMask) ? -iv : iv);
}

int32_t HGCalWaferIndex::waferCopy(const int32_t id) {
  return (id >> HGCalProperty::kHGCalWaferCopyOffset) & HGCalProperty::kHGCalWaferCopyMask;
}

bool HGCalWaferIndex::waferFormat(const int32_t id) { return ((id & HGCalProperty::kHGCalLayerOldMask) == 0); }
