///////////////////////////////////////////////////////////////////////////////
// File: EcalBarrelNumberingScheme.cc
// Description: Numbering scheme for barrel electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include <sstream>

//#define EDM_ML_DEBUG

EcalBarrelNumberingScheme::EcalBarrelNumberingScheme() : EcalNumberingScheme() {
  edm::LogVerbatim("EcalGeom") << "Creating EcalBarrelNumberingScheme";
}

EcalBarrelNumberingScheme::~EcalBarrelNumberingScheme() {
  edm::LogVerbatim("EcalGeom") << "Deleting EcalBarrelNumberingScheme";
}

uint32_t EcalBarrelNumberingScheme::getUnitID(const EcalBaseNumber& baseNumber) const {
  const uint32_t nLevels(baseNumber.getLevels());

#ifdef EDM_ML_DEBUG
  std::ostringstream st1;
  for (uint32_t k = 0; k < nLevels; ++k)
    st1 << ", " << baseNumber.getLevelName(k) << ":" << baseNumber.getCopyNumber(k);
  edm::LogVerbatim("EcalGeom") << "ECalBarrelNumberingScheme geometry levels = " << nLevels << st1.str();
#endif
  if (11 > nLevels) {
    edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID(): "
                                << "Not enough levels found in EcalBaseNumber ( " << nLevels << ") Returning 0";
    return 0;
  }

  const std::string& cryName(baseNumber.getLevelName(0).substr(0, 7));  // name of crystal volume

  const int cryType(::atoi(cryName.c_str() + 5));

  uint32_t wallCopy(0), hawCopy(0), fawCopy(0), supmCopy(0);
  const int off(13 < nLevels ? 3 : 0);

  if ((nLevels != 11) && (nLevels != 14)) {
    wallCopy = baseNumber.getCopyNumber(3 + off);
    hawCopy = baseNumber.getCopyNumber(4 + off);
    fawCopy = baseNumber.getCopyNumber(5 + off);
    supmCopy = baseNumber.getCopyNumber(6 + off);
  } else {
    auto num1 = numbers(baseNumber.getLevelName(3 + off));
    wallCopy = num1.second;
    hawCopy = num1.first;
    auto num2 = numbers(baseNumber.getLevelName(4 + off));
    fawCopy = num2.second;
    supmCopy = num2.first;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << nLevels << " off: " << off << " cryType: " << cryType << " wallCopy: " << wallCopy
                               << " hawCopy: " << hawCopy << " fawCopy: " << fawCopy << " supmCopy: " << supmCopy;
#endif
  // error checking

  if (1 > cryType || 17 < cryType) {
    edm::LogWarning("EdalGeom") << "ECalBarrelNumberingScheme::getUnitID(): "
                                << "****************** Bad crystal name = " << cryName
                                << ", Volume Name = " << baseNumber.getLevelName(0);
    return 0;
  }

  if (1 > wallCopy || 5 < wallCopy) {
    edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID(): "
                                << "****************** Bad wall copy = " << wallCopy
                                << ", Volume Name = " << baseNumber.getLevelName(3);
    return 0;
  }

  if (1 > hawCopy || 2 < hawCopy) {
    edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID(): "
                                << "****************** Bad haw copy = " << hawCopy
                                << ", Volume Name = " << baseNumber.getLevelName(4);
    return 0;
  }

  if (1 > fawCopy || 10 < fawCopy) {
    edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID(): "
                                << "****************** Bad faw copy = " << fawCopy
                                << ", Volume Name = " << baseNumber.getLevelName(5);
    return 0;
  }

  if (1 > supmCopy || 36 < supmCopy) {
    edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID(): "
                                << "****************** Bad supermodule copy = " << supmCopy
                                << ", Volume Name = " << baseNumber.getLevelName(6);
    return 0;
  }

  // all inputs are fine. Go ahead and decode

  const int32_t zsign(18 < supmCopy ? -1 : 1);

  const int32_t eta(5 * (cryType - 1) + wallCopy);

  const int32_t phi(18 < supmCopy ? 20 * (supmCopy - 19) + 2 * (10 - fawCopy) + 3 - hawCopy
                                  : 20 * (supmCopy - 1) + 2 * (fawCopy - 1) + hawCopy);

  const int32_t intindex(EBDetId(zsign * eta, phi).rawId());

  /*
  static int count ( 1 ) ;
  if( 0==count%1000 )
  {
     std::cout<<"************************** NLEVELS="<<nLevels
	      <<", eta="<<eta<<", phi="<<phi<<", zsign="<<zsign<<std::endl;
  }
  ++count;
*/
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << "EcalBarrelNumberingScheme: "
                               << "supmCopy = " << supmCopy << ", fawCopy = " << fawCopy << ", hawCopy = " << hawCopy
                               << ", wallCopy = " << wallCopy << ", cryType = " << cryType
                               << "\n           zsign = " << zsign << ", eta = " << eta << ", phi = " << phi
                               << ", packed index = 0x" << std::hex << intindex << std::dec;
#endif
  return intindex;
}

std::pair<int, int> EcalBarrelNumberingScheme::numbers(const std::string& name) const {
  int num1(-1), num2(-1);
  if (name.find('#') != std::string::npos) {
    uint32_t ip1 = name.find('#');
    if (name.find('!') != std::string::npos) {
      uint32_t ip2 = name.find('!');
      num1 = ::atoi(name.substr(ip1 + 1, ip2 - ip1 - 1).c_str());
      if (name.find('#', ip2) != std::string::npos) {
        uint32_t ip3 = name.find('#', ip2);
        num2 = ::atoi(name.substr(ip3 + 1, name.size() - ip3 - 1).c_str());
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << "EcalBarrelNumberingScheme::Numbers from " << name << " are " << num1 << " and "
                               << num2;
#endif
  return std::make_pair(num1, num2);
}
