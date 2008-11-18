/** \class HcalGenericDetId
    \author F.Ratnikov, UMd
   Generic HCAL detector ID suitable for all Hcal subdetectors
   $Id: HcalGenericDetId.cc,v 1.4 2007/10/03 01:39:14 mansj Exp $
*/

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include <iostream>

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

int HcalGenericDetId::hashedId() const {
  int index = -1;

  int HBhalf = 1296;
  int HEhalf = 1296;
  int HOhalf = 1080;
  int HFhalf = 864;
  int HThalf = 2088;
  int ZDChalf = 11;

  int zside=0, ietaAbs=0, iphi=0, depth=0;
  int sector=0, rbx=0, channel=0;

  // HB valid DetIds: phi=1-72,eta=1-14,depth=1; phi=1-72,eta=15-16,depth=1-2
  if (genericSubdet() == HcalGenericDetId::HcalGenBarrel )
    {
      HcalDetId tid(rawId() ); 
      zside = tid.zside();
      ietaAbs = tid.ietaAbs();
      iphi = tid.iphi();
      depth = tid.depth();

      if (ietaAbs < 16)   index = (iphi - 1)*18 + (ietaAbs - 1) + (depth - 1);
      if (ietaAbs == 16)  index = (iphi - 1)*18 + ietaAbs + (depth - 1);
      
      if (zside == -1) index += HBhalf;
    }
  
  // HE valid DetIds: phi=1-72,eta=16-17,depth=1; phi=1-72,eta=18-20,depth=1-2; 
  //                  phi=1-71(in steps of 2),eta=21-26,depth=1-2; phi=1-71(in steps of 2),eta=27-28,depth=1-3
  //                  phi=1-71(in steps of 2),eta=29,depth=1-2
  if (genericSubdet() ==  HcalGenericDetId::HcalGenEndcap )
    {
      HcalDetId tid(rawId() ); 
      zside = tid.zside();
      ietaAbs = tid.ietaAbs();
      iphi = tid.iphi();
      depth = tid.depth();
      
      if (ietaAbs == 16 || ietaAbs == 17)  index = (iphi - 1)*8 + (iphi/2)*20 + (ietaAbs - 16);
      if (ietaAbs >= 18 && ietaAbs <= 20)  index = (iphi - 1)*8 + (iphi/2)*20 + 2  + 2*(ietaAbs-18) + (depth - 1);
      if (ietaAbs >= 21 && ietaAbs <= 26)  index = (iphi - 1)*8 + (iphi/2)*20 + 8  + 2*(ietaAbs-21) + (depth - 1);
      if (ietaAbs >= 27 && ietaAbs <= 28)  index = (iphi - 1)*8 + (iphi/2)*20 + 20 + 3*(ietaAbs-27) + (depth - 1);
      if (ietaAbs == 29)                     index = (iphi - 1)*8 + (iphi/2)*20 + 26 + 2*(ietaAbs-29) + (depth - 1);
      
      if (zside == -1) index += HEhalf;
    }

  // HO valid DetIds: phi=1-72,eta=1-15,depth=4!
  if (genericSubdet() == HcalGenericDetId::HcalGenOuter )
    {
      HcalDetId tid(rawId() ); 
      zside = tid.zside();
      ietaAbs = tid.ietaAbs();
      iphi = tid.iphi();
      depth = tid.depth();
      
      index = (iphi - 1)*15 + (ietaAbs - 1);

      if (zside == -1) index += HOhalf;
  }

  // HF valid DetIds: phi=1-71(in steps of 2),eta=29-39,depth=1-2; phi=3-71(in steps of 4),eta=40-41,depth=1-2
  if (genericSubdet() == HcalGenericDetId::HcalGenForward )
    {
      HcalDetId tid(rawId() ); 
      zside = tid.zside();
      ietaAbs = tid.ietaAbs();
      iphi = tid.iphi();
      depth = tid.depth();

      index = ((iphi-1)/4)*4 + ((iphi-1)/2)*22 + 2*(ietaAbs-29) + (depth - 1);

      if (zside == -1)  index += HFhalf;
    }

  // TriggerTower:
  if (genericSubdet() == HcalGenericDetId::HcalGenTriggerTower )
    {
      HcalTrigTowerDetId tid(rawId() ); 
      zside = tid.zside();
      ietaAbs = tid.ietaAbs();
      iphi = tid.iphi();

      int HTphi1_18 = 576;
      if (iphi < 19) index = (iphi-1)*32 + (ietaAbs-1);
      else index = HTphi1_18 + (iphi-19)*28 + (ietaAbs-1);

      if (zside == -1) index += HThalf;
    }

  // ZDC: ZDC_EM: 1 to 5, ZDC_HAD: 1 to 4, ZDC_LUM: 1,2, eta = +1, -1
  if (genericSubdet() == HcalGenericDetId::HcalGenZDC )
    {
      HcalZDCDetId tid(rawId() ); 
      zside = tid.zside();
      depth = tid.depth();

      switch (tid.section() ) {
      case HcalZDCDetId::EM:   index = (depth-1); break;
      case HcalZDCDetId::HAD:  index = 5 + (depth-1); break;
      case HcalZDCDetId::LUM:  index = 9 + (depth-1); break;
      default: break;
      }
      if (zside == -1) index += ZDChalf;
    }

  // Castor: ???
  if (genericSubdet() == HcalGenericDetId::HcalGenCastor )
    {
      HcalCastorDetId tid(rawId() ); 

    }

  // Calibration channels: no zside=-1 ! with current naming convention
  if (genericSubdet() == HcalGenericDetId::HcalGenCalibration )
    {
      HcalCalibDetId tid(rawId() ); 
      sector = tid.sector();
      rbx = tid.rbx();
      channel = tid.cboxChannel();

      index = (sector-1)*18*7 + (rbx-1)*7 + (channel-1);

    }

  //std::cout << "eta=" << ietaAbs << " side=" << zside << " phi=" << iphi << " depth=" << depth << " index=" << index << std::endl;

  return index;
}
