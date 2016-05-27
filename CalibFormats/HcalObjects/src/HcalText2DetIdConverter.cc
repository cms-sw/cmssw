/** \class HcalText2DetIdConverter
    \author F.Ratnikov, UMd
*/
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <cstdio>

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

#include "CalibFormats/HcalObjects/interface/HcalText2DetIdConverter.h"

namespace {
  std::string strip (const std::string& fString) {
    if (fString.empty ()) return fString;
    int startIndex = fString.find_first_not_of(" \t\n");
    int endIndex = fString.find_last_not_of(" \t\n");
    return fString.substr(startIndex, (endIndex-startIndex)+1);
  }

  int calibChannel (const std::string& fName) {
    return 
      fName == "Mixer-High" ? 1 :
      fName == "Mixer-Low" ? 2 :
      fName == "Megatile" ? 3 :
      fName == "Mixer-Scintillator" ? 4 :
      fName == "RadDam1" ? 5 :
      fName == "RadDam2" ? 6 :
      fName == "RadDam3" ? 7 :
      atoi(fName.c_str());
    //      0;
  }
}

HcalText2DetIdConverter::HcalText2DetIdConverter (const std::string& fFlavor, const std::string& fField1,
						  const std::string& fField2, const std::string& fField3) {
  if (!init (fFlavor, fField1, fField2, fField3)) {
    std::cerr << "HcalText2DetIdConverter::HcalText2DetIdConverter-> Can not initiate detId from items: "
	      << fFlavor << '/' << fField1 << '/' << fField2 << '/' << fField3 << std::endl;
    throw cms::Exception("HcalGenDetId initialization error") 
      << " Can not initiate detId from items: "
      << fFlavor << '/' << fField1 << '/' << fField2 << '/' << fField3 << std::endl;
  }
}

HcalText2DetIdConverter::HcalText2DetIdConverter (DetId fId) {
  init (fId);
}

bool HcalText2DetIdConverter::isHcalDetId () const {
  return HcalGenericDetId (mId).isHcalDetId ();
}

bool HcalText2DetIdConverter::isHcalCalibDetId () const {
  return HcalGenericDetId (mId).isHcalCalibDetId ();
}

bool HcalText2DetIdConverter::isHcalTrigTowerDetId () const {
  return HcalGenericDetId (mId).isHcalTrigTowerDetId ();
}

bool HcalText2DetIdConverter::isHcalZDCDetId () const {
  return HcalGenericDetId (mId).isHcalZDCDetId ();
}

bool HcalText2DetIdConverter::init (DetId fId) {
  bool result = true;
  flavorName = "UNKNOWN";
  mId = fId;
  HcalGenericDetId genId (mId);
  if (fId == HcalDetId::Undefined) {
    flavorName = "NA";
  }
  else if (genId.isHcalDetId ()) {
    HcalDetId hcalId (mId);
    flavorName = genId.genericSubdet () == HcalGenericDetId::HcalGenBarrel ? "HB" :
      genId.genericSubdet () == HcalGenericDetId::HcalGenEndcap ? "HE" :
      genId.genericSubdet () == HcalGenericDetId::HcalGenOuter ? "HO" :
      genId.genericSubdet () == HcalGenericDetId::HcalGenForward ? "HF" : "H_UNKNOWN";
    setField (1, hcalId.ieta());
    setField (2, hcalId.iphi());
    setField (3, hcalId.depth());
  }
  else if (genId.isHcalTrigTowerDetId ()) {
    HcalTrigTowerDetId triggerId (mId);
    if (triggerId == HcalTrigTowerDetId::Undefined) {
      flavorName = "NT";
    }
    else {
      flavorName = "HT";
      setField (1, triggerId.ieta());
      setField (2, triggerId.iphi());
      setField (3, triggerId.version()*10 + triggerId.depth());
/*      
      if (triggerId.version() == 0) {
        setField (1, triggerId.ieta());
        setField (2, triggerId.iphi());
        setField (3, 1);
      } else if (triggerId.version() == 1) {
        setField (1, triggerId.ieta());
        setField (2, triggerId.iphi());
        setField (3, 10);  // We use the tens digit to indicate version
      } else {
        // Unknown version
      }
*/
    }
  }
  else if (genId.isHcalZDCDetId ()) {
    HcalZDCDetId zdcId (mId);
    switch (zdcId.section()) {
    case HcalZDCDetId::EM: flavorName = "ZDC_EM"; break;
    case HcalZDCDetId::HAD: flavorName = "ZDC_HAD"; break;
    case HcalZDCDetId::LUM: flavorName = "ZDC_LUM"; break;
    default: result = false;
    }
    setField (1, zdcId.zside());
    setField (2, zdcId.channel());
    setField (3, -99);
  }
  else if (genId.isHcalCalibDetId ()) {
    HcalCalibDetId calibId (mId);
    if (calibId.calibFlavor()==HcalCalibDetId::CalibrationBox) {
      switch (calibId.hcalSubdet()) {
      case HcalBarrel:  flavorName = "CALIB_HB"; break;
      case HcalEndcap:  flavorName = "CALIB_HE"; break;
      case HcalOuter:  flavorName = "CALIB_HO"; break;
      case HcalForward:  flavorName = "CALIB_HF"; break;
      default: result = false;
      }
      setField (1, calibId.ieta());
      setField (2, calibId.iphi());
      setField (3, calibId.cboxChannel() );
    } else if (calibId.calibFlavor()==HcalCalibDetId::HOCrosstalk) {
      flavorName="HOX";
      setField (1, calibId.ieta());
      setField (2, calibId.iphi());
      setField (3, -999);
    } else if (calibId.calibFlavor()==HcalCalibDetId::uMNqie) {
      flavorName="UMNQIE";
      setField (1, calibId.channel());
      setField (2, -999);
      setField (3, -999);
    } else if (calibId.calibFlavor()==HcalCalibDetId::CastorRadFacility) {
      flavorName="CRF";
      setField (1, calibId.rm());
      setField (2, calibId.fiber());
      setField (3, calibId.channel());
    }
  }
  else {
    flavorName = "UNKNOWN_FLAVOR";
    std::cerr << "HcalText2DetIdConverter::init-> Unknown detId: " << std::hex << std::showbase << mId.rawId() << std::endl;
    result = false;
  }
  return result;
}


bool HcalText2DetIdConverter::init (const std::string& fFlavor, const std::string& fField1,
				   const std::string& fField2, const std::string& fField3) {
  bool result = true;
  flavorName = strip (fFlavor);
  field1 = strip (fField1);
  field2 = strip (fField2);
  field3 = strip (fField3);
  if (flavorName == "HB" ||
      flavorName == "HE" ||
      flavorName == "HF" ||
      flavorName == "HO") {
    HcalSubdetector sub = flavorName == "HB" ? HcalBarrel :
      flavorName == "HE" ? HcalEndcap :
      flavorName == "HO" ? HcalOuter :
      HcalForward;
    mId = HcalDetId (sub, getField (1), getField (2), getField (3));
  }
  else if (flavorName == "HT") {
    // We use the depth to signal the "version" being used (RCT or 1x1 HF). RCT
    // has a 0 in the 10s digit, whereas 1x1 has a 1. The ones digit is still
    // used to indicate depth, although in the 1x1 case this must be 0, so we
    // set it as such.
    mId = HcalTrigTowerDetId (getField (1), getField (2), getField (3));
/*
    const int depth_field = getField(3);
    const int ones = depth_field % 10;
    const int tens = (depth_field - ones) / 10;
    if (tens == 0) {
      const int depth = ones;
      const int version = 0;
      mId = HcalTrigTowerDetId (getField (1), getField (2), depth, version);
    } else if (tens == 1) {
      const int depth = 0;
      const int version = 1;
      mId = HcalTrigTowerDetId(getField(1), getField(2), depth, version);
    } else {
      // Undefined version!
    }
*/
  }
  else if (flavorName.find ("ZDC_") == 0) {
    HcalZDCDetId::Section section = flavorName == "ZDC_EM" ? HcalZDCDetId::EM :
      flavorName == "ZDC_HAD" ? HcalZDCDetId::HAD : 
      flavorName == "ZDC_LUM" ? HcalZDCDetId::LUM : HcalZDCDetId::Unknown;
    mId = HcalZDCDetId (section, getField (1)>0, getField (2));
  }
  else if (flavorName.find ("CALIB_") == 0) {
    HcalSubdetector sd = HcalOther;
    if (flavorName.find("HB")!=std::string::npos) sd=HcalBarrel;
    else if (flavorName.find("HE")!=std::string::npos) sd=HcalEndcap;
    else if (flavorName.find("HO")!=std::string::npos) sd=HcalOuter;
    else if (flavorName.find("HF")!=std::string::npos) sd=HcalForward;
    
    int ieta=getField(1);
    int iphi=getField(2);
    int channel = calibChannel (field3);
    mId = HcalCalibDetId (sd, ieta,iphi,channel);
  }
  else if (flavorName=="HOX") {
    int ieta=getField(1);
    int iphi=getField(2);
    mId = HcalCalibDetId (ieta,iphi);
  }
  else if (flavorName=="UMNQIE") {
    int channel=getField(1);
    mId = HcalCalibDetId (HcalCalibDetId::uMNqie,channel);
  }
  else if (flavorName=="CRF") {
    int rm=getField(1);
    int fiber=getField(2);
    int channel=getField(3);
    mId = HcalCalibDetId (HcalCalibDetId::CastorRadFacility,rm,fiber,channel);
  }
  else if (flavorName == "NA") {
    mId = HcalDetId::Undefined;
  } 
  else {
    std::cerr << "HcalText2DetIdConverter::init-> Unknown HcalDetId flavor: " << flavorName << std::endl;
    result = false;
  }
  return result;
}


int HcalText2DetIdConverter::getField (int i) const{
  char* endptr;
  const char* nptr = i == 1 ? field1.c_str() :
    i == 2 ? field2.c_str() : field3.c_str();
  long result = strtol (nptr, &endptr, 0);
    if (*nptr != '\0' && *endptr == '\0') {
      return result;
    }
    if (i == 2 && isHcalCalibDetId ()) {
      int result = calibChannel (field2);
      if (i) return result;
    }
    if (*nptr != '\0') {
      std::cerr << "HcalText2DetIdConverter::getField-> Can not convert string "<< nptr << " to int. Bad symbol: " << *endptr << std::endl;
    }
    return 0;
  }

void HcalText2DetIdConverter::setField (int i, int fValue) {
  char buffer [128];
  sprintf (buffer, "%d", fValue);
  if (i == 1) field1 = buffer;
  else if (i == 2) field2 = buffer;
  else  field3 = buffer;
}

std::string HcalText2DetIdConverter::toString () const {
  return flavorName + " " + field1 + " " + field2 + " " + field3;
}
