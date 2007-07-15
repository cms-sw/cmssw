/** \class HcalGenericDetId
    \author F.Ratnikov, UMd
   Generic HCAL detector ID suitable for all Hcal subdetectors
   $Id: HcalGenericDetId.cc,v 1.1 2006/09/08 21:39:22 mansj Exp $
*/

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"

HcalSubdetector HcalGenericDetId::subdet() const {
  if (det () != Hcal) return HcalEmpty;
  return (HcalSubdetector)(subdetId()); 
}

HcalOtherSubdetector HcalGenericDetId::otherSubdet () const {
  if (subdet() != HcalOther) return HcalOtherEmpty;
  return HcalOtherSubdetector ((rawId()>>20)&0x1F);
}

HcalGenericDetId::HcalGenericSubdetector HcalGenericDetId::genericSubdet () const {
  switch (subdet()) {
  case HcalBarrel: return HcalGenBarrel;
  case HcalEndcap: return HcalGenEndcap;
  case HcalOuter: return HcalGenOuter;
  case HcalForward: return HcalGenForward;
  case HcalTriggerTower: return HcalGenTriggerTower;
  case HcalOther:
    switch (otherSubdet ()) {
    case HcalZDC: return HcalGenZDC;
    case HcalCalibration: return HcalGenCalibration;
    default: return HcalGenEmpty;
    }
  default: return HcalGenEmpty;
  }
  return HcalGenEmpty;
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

std::ostream& operator<<(std::ostream& s,const HcalGenericDetId& id) {
  if (id.null()) s << "(Null Id)";
  else 
    switch (id.subdet()) {
    case HcalBarrel: 
    case HcalEndcap: 
    case HcalOuter: 
    case HcalForward: s << HcalDetId(id); break;
    case HcalTriggerTower: s << HcalTrigTowerDetId(id); break;
    case HcalOther:
      switch (id.otherSubdet ()) {
      case HcalZDC: s << HcalZDCDetId(id); break;
      case HcalCalibration: s << HcalCalibDetId(id); break;
      default: s << "(Hcal Unknown Other Id: 0x" << std::hex << id.rawId() << std::dec << ')';
      } break;
    default: s << "(Hcal Unknown Id: 0x" << std::hex << id.rawId() << std::dec << ')';
    }
  return s;
}
