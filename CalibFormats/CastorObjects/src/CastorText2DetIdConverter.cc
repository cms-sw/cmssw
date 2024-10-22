/** \class CastorText2DetIdConverter
*/
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cstdio>

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"

#include "CalibFormats/CastorObjects/interface/CastorText2DetIdConverter.h"

namespace {
  std::string strip(const std::string& fString) {
    if (fString.empty())
      return fString;
    int startIndex = fString.find_first_not_of(" \t\n");
    int endIndex = fString.find_last_not_of(" \t\n");
    return fString.substr(startIndex, (endIndex - startIndex) + 1);
  }
}  // namespace

CastorText2DetIdConverter::CastorText2DetIdConverter(const std::string& fFlavor,
                                                     const std::string& fField1,
                                                     const std::string& fField2,
                                                     const std::string& fField3) {
  if (!init(fFlavor, fField1, fField2, fField3)) {
    std::cerr << "CastorText2DetIdConverter::CastorText2DetIdConverter-> Can not initiate detId from items: " << fFlavor
              << '/' << fField1 << '/' << fField2 << '/' << fField3 << std::endl;
    throw cms::Exception("HcalGenDetId initialization error")
        << " Can not initiate detId from items: " << fFlavor << '/' << fField1 << '/' << fField2 << '/' << fField3
        << std::endl;
  }
}

CastorText2DetIdConverter::CastorText2DetIdConverter(DetId fId) { init(fId); }

bool CastorText2DetIdConverter::isHcalCastorDetId() const { return HcalGenericDetId(mId).isHcalCastorDetId(); }

bool CastorText2DetIdConverter::init(DetId fId) {
  bool result = true;
  flavorName = "UNKNOWN";
  mId = fId;
  HcalGenericDetId genId(mId);
  if (fId == HcalDetId::Undefined) {
    flavorName = "NA";
  }

  else if (genId.isHcalCastorDetId()) {
    HcalCastorDetId castorId(mId);
    switch (castorId.section()) {
      case HcalCastorDetId::EM:
        flavorName = "CASTOR_EM";
        break;
      case HcalCastorDetId::HAD:
        flavorName = "CASTOR_HAD";
        break;
      default:
        result = false;
    }
    setField(1, castorId.zside());
    setField(2, castorId.sector());
    setField(3, castorId.module());
  }

  else {
    flavorName = "UNKNOWN_FLAVOR";
    std::cerr << "CastorText2DetIdConverter::init-> Unknown detId: " << std::hex << std::showbase << mId.rawId()
              << std::endl;
    result = false;
  }
  return result;
}

bool CastorText2DetIdConverter::init(const std::string& fFlavor,
                                     const std::string& fField1,
                                     const std::string& fField2,
                                     const std::string& fField3) {
  bool result = true;
  flavorName = strip(fFlavor);
  field1 = strip(fField1);
  field2 = strip(fField2);
  field3 = strip(fField3);
  if (flavorName.find("CASTOR_") == 0) {
    HcalCastorDetId::Section section = flavorName == "CASTOR_EM"    ? HcalCastorDetId::EM
                                       : flavorName == "CASTOR_HAD" ? HcalCastorDetId::HAD
                                                                    : HcalCastorDetId::Unknown;
    mId = HcalCastorDetId(section, getField(1) > 0, getField(2), getField(3));
  }

  else {
    std::cerr << "CastorText2DetIdConverter::init-> Unknown DetId flavor: " << flavorName << std::endl;
    result = false;
  }
  return result;
}

int CastorText2DetIdConverter::getField(int i) const {
  char* endptr;
  const char* nptr = i == 1 ? field1.c_str() : i == 2 ? field2.c_str() : field3.c_str();
  long result = strtol(nptr, &endptr, 0);
  if (*nptr != '\0' && *endptr == '\0') {
    return result;
  }
  if (*nptr != '\0') {
    std::cerr << "CastorText2DetIdConverter::getField-> Can not convert string " << nptr
              << " to int. Bad symbol: " << *endptr << std::endl;
  }
  return 0;
}

void CastorText2DetIdConverter::setField(int i, int fValue) {
  char buffer[128];
  sprintf(buffer, "%d", fValue);
  if (i == 1)
    field1 = buffer;
  else if (i == 2)
    field2 = buffer;
  else
    field3 = buffer;
}

std::string CastorText2DetIdConverter::toString() const {
  return flavorName + " " + field1 + " " + field2 + " " + field3;
}
