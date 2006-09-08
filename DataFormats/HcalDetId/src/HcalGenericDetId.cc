/** \class HcalGenericDetId
    \author F.Ratnikov, UMd
   Generic HCAL detector ID suitable for all Hcal subdetectors
   $Id: HcalGenericDetId.cc,v 1.1 2006/07/31 18:31:03 fedor Exp $
*/

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"

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
    }
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

