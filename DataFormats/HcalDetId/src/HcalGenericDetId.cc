/** \class HcalGenericDetId
    \author F.Ratnikov, UMd
   Generic HCAL detector ID suitable for all Hcal subdetectors
*/

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include <iostream>
#include <cstdlib>

HcalOtherSubdetector HcalGenericDetId::otherSubdet () const {
  if (HcalSubdetector(subdetId()) != HcalOther) return HcalOtherEmpty;
  return HcalOtherSubdetector ((rawId()>>20)&0x1F);
}

HcalGenericDetId::HcalGenericSubdetector HcalGenericDetId::genericSubdet () const {
  if (null()) return HcalGenEmpty;
  switch (det()) {
  case Calo : 
    switch (subdetId()) {
    case HcalZDCDetId::SubdetectorId : return HcalGenZDC;
    case HcalCastorDetId::SubdetectorId : return HcalGenCastor;
    default: return HcalGenUnknown;
    } 
  case Hcal :
    switch (HcalSubdetector(subdetId())) {
    case 0: return HcalGenEmpty;
    case HcalBarrel: return HcalGenBarrel;
    case HcalEndcap: return HcalGenEndcap;
    case HcalOuter: return HcalGenOuter;
    case HcalForward: return HcalGenForward;
    case HcalTriggerTower: return HcalGenTriggerTower;
    case HcalOther:
      switch (otherSubdet ()) {
      case HcalCalibration: return HcalGenCalibration;
      default: return HcalGenUnknown;
    }
    default: return HcalGenUnknown;
    }
    default: return HcalGenUnknown;
  }
  return HcalGenUnknown;
}
  
bool HcalGenericDetId::isHcalDetId () const {
  HcalGenericSubdetector subdet = genericSubdet ();
  return subdet == HcalGenBarrel || subdet == HcalGenEndcap || subdet == HcalGenOuter || subdet == HcalGenForward; 
}

bool HcalGenericDetId::isHcalCalibDetId () const {
  HcalGenericSubdetector subdet = genericSubdet ();
  return subdet == HcalGenCalibration;
}

bool HcalGenericDetId::isHcalTrigTowerDetId () const {
  HcalGenericSubdetector subdet = genericSubdet ();
  return subdet == HcalGenTriggerTower;
}

bool HcalGenericDetId::isHcalZDCDetId () const {
  HcalGenericSubdetector subdet = genericSubdet ();
  return subdet == HcalGenZDC;
}

bool HcalGenericDetId::isHcalCastorDetId () const {
  HcalGenericSubdetector subdet = genericSubdet ();
  return subdet == HcalGenCastor;
}

std::ostream& operator<<(std::ostream& s,const HcalGenericDetId& id) {
  if (id.null()) s << "(Null Id)";
  else 
    switch (id.genericSubdet()) {
    case HcalGenericDetId::HcalGenBarrel: 
    case HcalGenericDetId::HcalGenEndcap: 
    case HcalGenericDetId::HcalGenOuter: 
    case HcalGenericDetId::HcalGenForward: s << HcalDetId(id); break;
    case HcalGenericDetId::HcalGenTriggerTower: s << HcalTrigTowerDetId(id); break;
    case HcalGenericDetId::HcalGenZDC: s << HcalZDCDetId(id); break;
    case HcalGenericDetId::HcalGenCastor: s << HcalCastorDetId(id); break;
    case HcalGenericDetId::HcalGenCalibration: s << HcalCalibDetId(id); break;
    default: s << "(Hcal Unknown Id: 0x" << std::hex << id.rawId() << std::dec << ')';
    }
  return s;
}

