///////////////////////////////////////////////////////////////////////////////
// File: EcalBarrelNumberingScheme.cc
// Description: Numbering scheme for barrel electromagnetic calorimeter
///////////////////////////////////////////////////////////////////////////////
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include <iostream>

EcalBarrelNumberingScheme::EcalBarrelNumberingScheme() : 
  EcalNumberingScheme() {
  edm::LogInfo("EcalGeom") << "Creating EcalBarrelNumberingScheme";
}

EcalBarrelNumberingScheme::~EcalBarrelNumberingScheme() {
  edm::LogInfo("EcalGeom") << "Deleting EcalBarrelNumberingScheme";
}

uint32_t EcalBarrelNumberingScheme::getUnitID(const EcalBaseNumber& baseNumber) const {
  if (baseNumber.getLevels()<1) {
    edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID: No "
				<< "level found in EcalBaseNumber Returning 0";
    return 0;
  }

  int PVid  = baseNumber.getCopyNumber(0);
  int MVid  = 1; 
  int MMVid = 1;

  if (baseNumber.getLevels() > 1) {
    MVid = baseNumber.getCopyNumber(1);
  } else { 
    edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID: Null"
				<< " pointer to alveole ! Use default id=1";  }
  if (baseNumber.getLevels() > 2) { 
    MMVid = baseNumber.getCopyNumber(2);
  } else { 
    edm::LogWarning("EcalGeom") << "ECalBarrelNumberingScheme::getUnitID: Null"
				<< " pointer to module ! Use default id=1";
  }

  // z side 
  int zside   = baseNumber.getCopyNumber("EREG");
  if ( zside == 1 || zside == 2 ) {
    zside=2*(1-zside)+1;
  }
  else if ( zside == 0 ) {
    // MTCC geometry
    int zMTCC = baseNumber.getCopyNumber("EREG_P");
    if ( zMTCC == 1 ) {
      zside = 1;
    }
  }

  // eta index of in Lyon geometry
  int ieta = PVid%5;
  if( ieta == 0) {ieta = 5;}
  int eta = 5 * (int) ((float)(PVid - 1)/10.) + ieta;

  // phi index in Lyon geometry
  int isubm = 1 + (int) ((float)(PVid - 1)/5.);
  int iphi  = (isubm%2) == 0 ? 2: 1;
  int phi=-1;

  if (zside == 1)
    phi = (20*(18-MMVid) + 2*(10-MVid) + iphi + 20) % 360  ;
  else if (zside == -1)
    phi = (541 - (20*(18-MMVid) + 2*(10-MVid) + iphi) ) % 360  ;

  if (phi == 0) 
    phi = 360;

  //pack it into an integer
  // to be consistent with EBDetId convention
  //  zside=2*(1-zside)+1;
  uint32_t intindex = EBDetId(zside*eta,phi).rawId();

  LogDebug("EcalGeom") << "EcalBarrelNumberingScheme zside = "  << zside 
		       << " eta = " << eta << " phi = " << phi 
		       << " packed index = 0x" << std::hex << intindex 
		       << std::dec;
  return intindex;

}

