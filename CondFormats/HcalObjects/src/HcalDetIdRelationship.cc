#include "CondFormats/HcalObjects/interface/HcalDetIdRelationship.h"

bool hcalEqualDetId(uint32_t id, const DetId& fId) {
  return ((fId.det()==DetId::Hcal && HcalDetId(id) == HcalDetId(fId)) ||
	  (fId.det()==DetId::Calo && fId.subdetId()==HcalZDCDetId::SubdetectorId && HcalZDCDetId(id) == HcalZDCDetId(fId)) ||
	  (fId.det()!=DetId::Hcal && (fId.det()==DetId::Calo && fId.subdetId()!=HcalZDCDetId::SubdetectorId) && (id == fId.rawId())));
}

DetId hcalTransformedId(const DetId& aid) {
  DetId id;
  if (aid.det()==DetId::Hcal) {
    HcalDetId hcid(aid);
    id   = HcalDetId(hcid.subdet(),hcid.ieta(),hcid.iphi(),hcid.depth());
  } else if (aid.det()==DetId::Calo && aid.subdetId()==HcalZDCDetId::SubdetectorId) {
    HcalZDCDetId hcid(aid);
    id   = HcalZDCDetId(hcid.section(),(hcid.zside()>0),hcid.channel());
  } else {
    id   = aid;
  }
  return id;
}

