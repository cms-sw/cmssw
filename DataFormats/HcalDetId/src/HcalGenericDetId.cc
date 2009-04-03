/** \class HcalGenericDetId
    \author F.Ratnikov, UMd
   Generic HCAL detector ID suitable for all Hcal subdetectors
   $Id: HcalGenericDetId.cc,v 1.12 2009/03/24 16:05:59 rofierzy Exp $
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

int HcalGenericDetId::hashedId(bool h2mode_) const {
  int index = -1;

  int zside=0, ietaAbs=0, ieta=0, iphi=0, depth=0, channel=0, sector=0, module=0;

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
      
      if (!h2mode_)
	{
	  if (ietaAbs == 16 || ietaAbs == 17)  index = (iphi - 1)*8 + (iphi/2)*20 + (ietaAbs - 16);
	  if (ietaAbs >= 18 && ietaAbs <= 20)  index = (iphi - 1)*8 + (iphi/2)*20 + 2  + 2*(ietaAbs-18) + (depth - 1);
	  if (ietaAbs >= 21 && ietaAbs <= 26)  index = (iphi - 1)*8 + (iphi/2)*20 + 8  + 2*(ietaAbs-21) + (depth - 1);
	  if (ietaAbs >= 27 && ietaAbs <= 28)  index = (iphi - 1)*8 + (iphi/2)*20 + 20 + 3*(ietaAbs-27) + (depth - 1);
	  if (ietaAbs == 29)                   index = (iphi - 1)*8 + (iphi/2)*20 + 26 + 2*(ietaAbs-29) + (depth - 1);
	}
      else
	{
	  // make as general as possible, don't care about tight packing for the moment
	  index = (iphi-1)*4*14 + (ietaAbs - 16)*4 + (depth - 1);

//	  if (ietaAbs == 16)                   index = (iphi - 1)*11 + (iphi/2)*20 + (ietaAbs - 16);
//	  if (ietaAbs == 17)                   index = (iphi - 1)*11 + (iphi/2)*20 + 1 + (ietaAbs - 17) + (depth - 1);
//	  if (ietaAbs >= 18 && ietaAbs <= 20)  index = (iphi - 1)*11 + (iphi/2)*20 + 5  + 2*(ietaAbs-18) + (depth - 1);
//	  if (ietaAbs >= 21 && ietaAbs <= 26)  index = (iphi - 1)*11 + (iphi/2)*20 + 11 + 2*(ietaAbs-21) + (depth - 1);
//	  if (ietaAbs >= 27 && ietaAbs <= 28)  index = (iphi - 1)*11 + (iphi/2)*20 + 23 + 3*(ietaAbs-27) + (depth - 1);
//	  if (ietaAbs == 29)                   index = (iphi - 1)*11 + (iphi/2)*20 + 29 + 2*(ietaAbs-29) + (depth - 1);
	}
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

      if ((iphi-1)%4==0) index = (iphi-1)*32 + (ietaAbs-1) - (12*((iphi-1)/4));
      else               index = (iphi-1)*28 + (ietaAbs-1) + (4*(((iphi-1)/4)+1));

      if (zside == -1) index += HThalf;
      ieta = ietaAbs*zside;      
    }

  // ZDC: ZDC_EM: 1 to 5, ZDC_HAD: 1 to 4, ZDC_LUM: 1,2, eta = +1, -1
  if (genericSubdet() == HcalGenericDetId::HcalGenZDC )
    {
      HcalZDCDetId tid(rawId() ); 
      zside   = tid.zside();
      channel = tid.channel();
      //depth   = tid.depth();//depth is not unique, channel is

      switch (tid.section() ) {
      case HcalZDCDetId::EM:   index = (channel-1); break;
      case HcalZDCDetId::HAD:  index = 5 + (channel-1); break;
      case HcalZDCDetId::LUM:  index = 9 + (channel-1); break;
      default: break;
      }
      if (zside == -1) index += ZDChalf;
    }

  // Castor: zside +-1, sector (phi-segmentation) 1..16, module (z segm.) 1..14
  // total: 14*16=224 per zside
  if (genericSubdet() == HcalGenericDetId::HcalGenCastor )
    {
      HcalCastorDetId tid(rawId() ); 
      zside = tid.zside();
      sector = tid.sector();
      module = tid.module();

      index = 14*(sector-1) + (module-1);
      if (zside == -1) index += CASTORhalf;

    }

  // Calibration channels: no zside=-1 ! with current naming convention
  if (genericSubdet() == HcalGenericDetId::HcalGenCalibration )
    {
      HcalCalibDetId tid(rawId() );
      channel = tid.cboxChannel();
      ieta = tid.ieta();
      iphi = tid.iphi();
      zside = tid.zside();


      if (tid.calibFlavor()==HcalCalibDetId::CalibrationBox) {
	
	HcalSubdetector subDet = tid.hcalSubdet();
	
	if (subDet==HcalBarrel) {
	  //std::cout<<"CALIB_HB:  ";
	  //dphi = 4 (18 phi values), 3 channel types (0,1,2), eta = -1 or 1
	  //total of 18*3*2=108 channels
	  index = ((iphi+1)/4-1) + 18*channel + 27*(ieta+1);
	}
	else if (subDet==HcalEndcap) {
	  //std::cout<<"CALIB_HE:  ";
	  //dphi = 4 (18 phi values), 6 channel types (0,1,3,4,5,6), eta = -1 or 1
	  //total of 18*6*2=216 channels
	  if (channel>2) channel-=1;
	  index = ((iphi+1)/4-1) + 18*channel + 54*(ieta+1) + 108;
	} 
	else if (subDet==HcalForward) {
	  //std::cout<<"CALIB_HF:  ";
	  //dphi = 18 (4 phi values), 3 channel types (0,1,8), eta = -1 or 1
	  if (channel==8) channel = 2;
	  //total channels 4*3*2=24
	  index = (iphi-1)/18 + 4*channel + 6*(ieta+1) + 324;
	}
	else if (subDet==HcalOuter) {
	  //std::cout<<"CALIB_HO:  ";
	  //there are 5 special calib crosstalk channels, one in each ring
	  if (channel==7) {
	    channel = 2;
	    index = (ieta+2) + 420;
	  }
	  //for HOM/HOP dphi = 6 (12 phi values),  2 channel types (0,1), eta = -2,-1 or 1,2
	  //for HO0/YB0 dphi = 12 (6 phi values),  2 channel types (0,1), eta = 0
	  else{
	    if (ieta<0) index      = ((iphi+1)/12-1) + 36*channel + 6*(ieta+2) + 348;
	    else if (ieta>0) index = ((iphi+1)/12-1) + 36*channel + 6*(ieta+2) + 6 + 348;
	    else index             = ((iphi+1)/6-1)  + 36*channel + 6*(ieta+2) + 348;
	  }
	} 
	else {
	  std::cout << "HCAL Det Id not valid!" << std::endl;
	  index = 0;
	}
	
      }
      else if (tid.calibFlavor()==HcalCalibDetId::HOCrosstalk) {
	//std::cout<<"HX:  ";
	//for YB0/HO0 phi is grouped in 6 groups of 6 with dphi=2 but the transitions are 1 or 3
	// in such a way that the %36 operation yeilds unique values for every iphi
	if (abs(ieta)==4)  index = ((iphi-1)%36) + (((zside+1)*36)/2) + 72 + 425;   //ieta = 1 YB0/HO0;
	else               index = (iphi-1) + (36*(zside+1)*2) + 425;  //ieta = 0 for HO2M/HO1M ieta=2 for HO1P/HO2P;
      }
      //std::cout << "  " << ieta << "  " << zside << "  " << iphi << "  " << depth << "  " << index << std::endl;
    }
  //std::cout << "eta:  " << ieta << "  side:  " << zside << "  phi:  " << iphi << "  depth:  " << depth << "  index:  " << index << std::endl;
 
  return index;
}
