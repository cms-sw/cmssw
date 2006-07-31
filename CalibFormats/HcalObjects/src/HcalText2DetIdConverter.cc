/** \class HcalText2DetIdConverter
    \author F.Ratnikov, UMd
    $Id$
*/
#include <stdlib.h>
#include <iostream>

#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/HcalObjects/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

#include "CalibFormats/HcalObjects/interface/HcalText2DetIdConverter.h"

namespace {
  std::string strip (const std::string& fString) {
    if (fString.empty ()) return fString;
    int startIndex = fString.find_first_not_of(" \t\n");
    int endIndex = fString.find_last_not_of(" \t\n");
    return fString.substr(startIndex, (endIndex-startIndex)+1);
  }

  HcalCalibDetId::SectorId calibSector (const std::string& fName) {
    return
      fName == "CALIB_HBP" ? HcalCalibDetId::HBplus :
      fName == "CALIB_HBM" ? HcalCalibDetId::HBminus :
      fName == "CALIB_HEP" ? HcalCalibDetId::HEplus :
      fName == "CALIB_HEM" ? HcalCalibDetId::HEminus :
      fName == "CALIB_HFP" ? HcalCalibDetId::HFplus :
      fName == "CALIB_HFM" ? HcalCalibDetId::HFminus :
      fName == "CALIB_HO2P" ? HcalCalibDetId::HO2plus :
      fName == "CALIB_HO1P" ? HcalCalibDetId::HO1plus :
      fName == "CALIB_HO0" ? HcalCalibDetId::HOzero :
      fName == "CALIB_HO1M" ? HcalCalibDetId::HO1minus :
      fName == "CALIB_HO2M" ? HcalCalibDetId::HO2minus : 
      HcalCalibDetId::SectorId (0);
  }

  int calibChannel (const std::string& fName) {
    return fName == "Mixer-High" ? 1 :
      fName == "Mixer-Low" ? 2 :
      fName == "Megatile" ? 3 :
      fName == "Mixer-Scintillator" ? 4 :
      fName == "RadDam1" ? 5 :
      fName == "RadDam2" ? 6 :
      fName == "RadDam3" ? 7 :
      0;
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
  mId = fId;
  HcalGenericDetId genId (mId);
  if (genId.isHcalDetId ()) {
    HcalDetId hcalId (mId);
    flavorName = genId.genericSubdet () == HcalGenericDetId::HcalGenBarrel ? "HB" :
      genId.genericSubdet () == HcalGenericDetId::HcalGenEndcap ? "HE" :
      genId.genericSubdet () == HcalGenericDetId::HcalGenOuter ? "HO" :
      genId.genericSubdet () == HcalGenericDetId::HcalGenForward ? "HF" : "UNKNOWN";
    setField (1, hcalId.ieta());
    setField (2, hcalId.iphi());
    setField (3, hcalId.depth());
  }
  else if (genId.isHcalTrigTowerDetId ()) {
    HcalTrigTowerDetId triggerId (mId);
    flavorName = "HT";
    setField (1, triggerId.ieta());
    setField (2, triggerId.iphi());
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
  }
  else if (genId.isHcalCalibDetId ()) {
    HcalCalibDetId calibId (mId);
    switch (calibId.sector ()) {
    case HcalCalibDetId::HBplus:  flavorName = "CALIB_HBP"; break;
    case HcalCalibDetId::HBminus: flavorName = "CALIB_HBM"; break;
    case HcalCalibDetId::HEplus:  flavorName = "CALIB_HEP"; break;
    case HcalCalibDetId::HEminus: flavorName = "CALIB_HEM"; break;
    case HcalCalibDetId::HFplus:  flavorName = "CALIB_HFP"; break;
    case HcalCalibDetId::HFminus: flavorName = "CALIB_HFM"; break;
    case HcalCalibDetId::HO2plus: flavorName = "CALIB_HO2P"; break;
    case HcalCalibDetId::HO1plus: flavorName = "CALIB_HO1P"; break;
    case HcalCalibDetId::HOzero:  flavorName = "CALIB_HO0"; break;
    case HcalCalibDetId::HO1minus:flavorName = "CALIB_HO1M"; break;
    case HcalCalibDetId::HO2minus:flavorName = "CALIB_HO2M"; break;
    default: result = false;
    }
    setField (1, calibId.rbx());
    switch (calibId.cboxChannel ()) {
    case 1: field2 = "Mixer-High"; break;
    case 2: field2 = "Mixer-Low"; break;
    case 3: field2 = "Megatile"; break;
    case 4: field2 = "Mixer-Scintillator"; break;
    case 5: field2 = "RadDam1"; break;
    case 6: field2 = "RadDam2"; break;
    case 7: field2 = "RadDam3"; break;
    default: result = false;
    }
  }
  else {
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
    mId = HcalTrigTowerDetId (getField (1), getField (2));
  }
  else if (flavorName.find ("ZDC_") == 0) {
    HcalZDCDetId::Section section = flavorName == "ZDC_EM" ? HcalZDCDetId::EM :
      flavorName == "ZDC_HAD" ? HcalZDCDetId::HAD : 
      flavorName == "ZDC_HAD" ? HcalZDCDetId::LUM : HcalZDCDetId::Unknown;
    mId = HcalZDCDetId (section, getField (1)>0, getField (2));
  }
  else if (flavorName.find ("CALIB_") == 0) {
    HcalCalibDetId::SectorId sector = calibSector (flavorName);
    int channel = calibChannel (field2);
    mId = HcalCalibDetId (sector, getField (1), channel);
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
    std::cerr << "HcalText2DetIdConverter::getField-> Can not convert string to int: " << nptr << std::endl;
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
