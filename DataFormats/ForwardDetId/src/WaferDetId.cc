#include "DataFormats/ForwardDetId/interface/WaferDetId.h"

WaferDetId::WaferDetId() : DetId() { }

WaferDetId::WaferDetId(uint32_t rawid) : DetId(rawid) { }

WaferDetId::WaferDetId(DetId::Detector det, int subdet) : DetId(det,subdet) { }
