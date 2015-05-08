#include "CondFormats/HcalObjects/interface/HcalDetIdRelationship.h"

DetId hcalTransformedId(DetId aid) {
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

