#ifndef HcalDetIdRelationship_h
#define HcalDetIdRelationship_h

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include <vector>

template<class Item> bool hcalEqualDetId(Item* cell, const DetId& fId) {
  return hcalEqualDetId(cell->rawId(),fId);
}

bool hcalEqualDetId(uint32_t id, const DetId& fId);

DetId hcalTransformedId(const DetId& aid);

#endif
