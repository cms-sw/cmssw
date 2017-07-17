#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h" 
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

HcalCalibDetId::HcalCalibDetId() : HcalOtherDetId() {
}


HcalCalibDetId::HcalCalibDetId(uint32_t rawid) : HcalOtherDetId(rawid) {
}

HcalCalibDetId::HcalCalibDetId(HcalSubdetector subdet, int ieta, int iphi, int ctype) : HcalOtherDetId(HcalCalibration) {

  id_|=(CalibrationBox<<17); // Calibration Category, bits [17:19] (= "1" for CalibrationBox)
  id_|=(ctype&0xF);           // calibration channel type, bits [0:3]
  id_|=(((ieta+2)&0x7)<<11);     // eta index, bits [11:13]
  id_|=((subdet&0x7)<<14);   // subdetector, bits [14:16]
  if (subdet==4) id_|=((((((((iphi-1)&0x7E)+1)/18)*18)+1)&0x7F)<<4);      // phi index, bits [4:10] dphi = 18 for HF, values 1,19,37,55 (lower edge)
  //if (subdet==4) id_|=(((((((((iphi-1)>>1)<<1)+1)/18)*18)+1)&0x7F)<<4);      // phi index, bits [4:10] dphi = 18 for HF, values 1,19,37,55 (lower edge)
  //else if (subdet==1||subdet==2||subdet==3) id_|=((((((iphi+1)&0x7C)+71)%72)&0x7F)<<4);      // phi index, bits [4:10] dphi=4 for HBHEHO, values 3,7,...,71, (lower edge)
  else if (subdet==1||subdet==2) id_|=(((((((iphi+1)>>2)&0x1F)<<2)+71)%72)<<4);      // phi index, bits [4:10] dphi=4 for HBHE, values 3,7,...,71, (lower edge)
  else if (subdet==3&&ieta==0) id_|=( ((((((iphi+1)/6)*6)+71)%72)&0x7F) <<4);      // phi index, bits [4:10] dphi=6 for HO0, values 5,11,...,71, (lower edge)
  else if (subdet==3&&ieta!=0) id_|=( ((((((iphi+1)/12)*12)+71)%72)&0x7F) <<4);      // phi index, bits [4:10] dphi=12 for HOP and HOM, values 11,23,,...,71, (lower edge)
  else id_|=((iphi&0x7F)<<4);      // phi index, bits [4:10], simply allow all values from 0-127, shouldn't be any
}

HcalCalibDetId::HcalCalibDetId(int ieta, int iphi) : HcalOtherDetId(HcalCalibration) {
  id_|=(HOCrosstalk<<17); // Calibration Category, bits [17:19] (= "2" for HOX)
  id_|=(iphi&0x7F)               // phi index, bits [0:6]
      |((abs(ieta)&0xF)<<7)     // eta index, bits [7:10]
      |(((ieta > 0)?(1):(0))<<11); // z side, bit [11]
}

HcalCalibDetId::HcalCalibDetId(CalibDetType dt, int value) : HcalOtherDetId(HcalCalibration) {
  id_|=(dt<<17);
  id_|=value&0xFF;
}

HcalCalibDetId::HcalCalibDetId(CalibDetType dt, int value1, int value2, int value3)  : HcalOtherDetId(HcalCalibration) {
  id_|=(dt<<17);
  id_|=(value1&0x3F)<<10;
  id_|=(value2&0x1F)<<5;
  id_|=(value3&0x1F);
}

HcalCalibDetId::HcalCalibDetId(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Hcal || gen.subdetId()!=HcalOther)) {
    throw cms::Exception("Invalid DetId") << "Cannot initialize HcalCalibDetId from " << std::hex << gen.rawId() << std::dec; 
  }
  id_=gen.rawId();
  if (subdet()!=HcalCalibration) {
    throw cms::Exception("Invalid DetId") << "Cannot initialize HcalCalibDetId from " << std::hex << gen.rawId() << std::dec; 
  }
}

HcalCalibDetId& HcalCalibDetId::operator=(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Hcal || gen.subdetId()!=HcalOther)) {
    throw cms::Exception("Invalid DetId") << "Cannot assign HcalCalibDetId from " << std::hex << gen.rawId() << std::dec; 
  }
  id_=gen.rawId();
  if (subdet()!=HcalCalibration) {
    throw cms::Exception("Invalid DetId") << "Cannot assign HcalCalibDetId from " << std::hex << gen.rawId() << std::dec; 
  }
  return *this;
}

int HcalCalibDetId::cboxChannel() const {
  return (calibFlavor()==CalibrationBox)?(id_&0xF):(0);
}

HcalSubdetector HcalCalibDetId::hcalSubdet() const {  
  return (HcalSubdetector)((calibFlavor()==CalibrationBox)?((id_>>14)&0x7):(0));
}
    
int HcalCalibDetId::ieta() const {
  return (calibFlavor()==CalibrationBox)?(((id_>>11)&0x7)-2):((calibFlavor()==HOCrosstalk)?(((id_>>7)&0xF)*zside()):(0));
}

int HcalCalibDetId::iphi() const {
  return (calibFlavor()==CalibrationBox)?((id_>>4)&0x7F):((calibFlavor()==HOCrosstalk)?(id_&0x7F):(0));
}

int HcalCalibDetId::zside() const {
  return (calibFlavor()==HOCrosstalk)?(((id_>>11)&0x1)?(1):(-1)):(0);
}

std::string HcalCalibDetId::cboxChannelString() const {
  switch (cboxChannel()) {
  case(cbox_MixerHigh): return "Mixer-High";
  case(cbox_MixerLow): return "Mixer-Low";
  case(cbox_LaserMegatile): return "Megatile";
  case(cbox_RadDam_Layer0_RM4): return "RadDam-L0-RM4";
  case(cbox_RadDam_Layer7_RM4): return "RadDam-L7-RM4";
  case(cbox_RadDam_Layer0_RM1): return "RadDam-L0-RM1";
  case(cbox_RadDam_Layer7_RM1): return "RadDam-L7-RM1";
  case(cbox_HOCrosstalkPIN): return "HO-Crosstalk-PIN";
  case(cbox_HF_ScintillatorPIN): return "HF-Scint-PIN";
  default : return "";
  }
}

int HcalCalibDetId::channel() const { return (calibFlavor()==uMNqie)?(id_&0xFF):(id_&0x1F); }

int HcalCalibDetId::fiber() const { return (calibFlavor()==CastorRadFacility)?((id_>>5)&0x1F):(0); }

int HcalCalibDetId::rm() const { return (calibFlavor()==CastorRadFacility)?((id_>>10)&0x3F):(0); }


std::ostream& operator<<(std::ostream& s,const HcalCalibDetId& id) {
  std::string sd;
  switch (id.hcalSubdet()) {
    case(HcalBarrel) : sd="HB"; break;
    case(HcalEndcap) : sd="HE"; break;
    case(HcalOuter) : sd="HO"; break;
    case(HcalForward) : sd="HF"; break;
    default: break;
  }
  switch (id.calibFlavor()) {
  case(HcalCalibDetId::CalibrationBox):
    return s << "(HcalCalibBox " << sd << ' ' << id.ieta() << "," << id.iphi()
	     << ' ' << id.cboxChannelString() << ')';
  case(HcalCalibDetId::HOCrosstalk):
    return s << "(HOCrosstalk "  << id.ieta() << "," << id.iphi() 
	     << ')';
  case (HcalCalibDetId::uMNqie):
    return s << "(uMNqie " << id.channel() << ')';
  case (HcalCalibDetId::CastorRadFacility):
    return s << "(CastorRadFacility " << id.rm() << " / " << id.fiber() << " / " << id.channel() << ')';
  default: return s;
  };
}
