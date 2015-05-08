#ifndef HcalDetIdRelationship_h
#define HcalDetIdRelationship_h

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include <vector>

template<class Item> bool hcalEqualDetId(Item* cell, DetId fId) {
  return ((fId.det()==DetId::Hcal && HcalDetId(cell->rawId()) == HcalDetId(fId)) ||
	  (fId.det()==DetId::Calo && fId.subdetId()==HcalZDCDetId::SubdetectorId && HcalZDCDetId(cell->rawId()) == HcalZDCDetId(fId)) ||
	  (fId.det()!=DetId::Hcal && (fId.det()==DetId::Calo && fId.subdetId()!=HcalZDCDetId::SubdetectorId) && (cell->rawId() == fId)));
}

DetId hcalTransformedId(DetId aid);

#endif
