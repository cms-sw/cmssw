///////////////////////////////////////////////////////////////////////////////
// File: EcalBarrelNumberingScheme.cc
// Description: Numbering scheme for barrel electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

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
  edm::LogVerbatim("EcalGeom") << "ECalBarrelNumberingScheme geometry levels = " << nLevels;
#endif
  if (12 > nLevels) {
    edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID(): "
                                << "Not enough levels found in EcalBaseNumber ( " << nLevels << ") Returning 0";
    return 0;
  }

  const std::string& cryName(baseNumber.getLevelName(0).substr(0, 7));  // name of crystal volume

  const int cryType(::atoi(cryName.c_str() + 5));

  const int off(13 < nLevels ? 3 : 0);

  const uint32_t wallCopy(baseNumber.getCopyNumber(3 + off));
  const uint32_t hawCopy(baseNumber.getCopyNumber(4 + off));
  const uint32_t fawCopy(baseNumber.getCopyNumber(5 + off));
  const uint32_t supmCopy(baseNumber.getCopyNumber(6 + off));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << nLevels << ", " << off << ", " << cryType << ", " << baseNumber.getLevelName(0) << ":"
                               << baseNumber.getCopyNumber(0) << ", " << baseNumber.getLevelName(1) << ":"
                               << baseNumber.getCopyNumber(1) << ", " << baseNumber.getLevelName(2) << ":"
                               << baseNumber.getCopyNumber(2) << ", " << baseNumber.getLevelName(3) << ":"
                               << baseNumber.getCopyNumber(3) << ", " << baseNumber.getLevelName(4) << ":"
                               << baseNumber.getCopyNumber(4) << ", " << baseNumber.getLevelName(5) << ":"
                               << baseNumber.getCopyNumber(5) << ", " << baseNumber.getLevelName(6) << ":"
                               << baseNumber.getCopyNumber(6) << ", " << baseNumber.getLevelName(7) << ":"
                               << baseNumber.getCopyNumber(7) << ", " << baseNumber.getLevelName(8) << ":"
                               << baseNumber.getCopyNumber(8) << ", " << baseNumber.getLevelName(9) << ":"
                               << baseNumber.getCopyNumber(9) << ", " << baseNumber.getLevelName(10) << ":"
                               << baseNumber.getCopyNumber(10);
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
