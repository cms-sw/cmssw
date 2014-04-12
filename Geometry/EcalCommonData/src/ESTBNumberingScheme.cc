#include "Geometry/EcalCommonData/interface/ESTBNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

#include <iostream>

ESTBNumberingScheme::ESTBNumberingScheme() : 
  EcalNumberingScheme() {

  int ix[30] = {3, 2, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 1,
		1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
  int iy[30] = {4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1,
		2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};

  int i;
  for(i=0; i<30; ++i) {
    iX[i] = ix[i];
    iY[i] = iy[i];
  }
    
  edm::LogInfo("EcalGeom") << "Creating ESTBNumberingScheme";
}

ESTBNumberingScheme::~ESTBNumberingScheme() {
  edm::LogInfo("EcalGeom") << "Deleting ESTBNumberingScheme";
}

uint32_t ESTBNumberingScheme::getUnitID(const EcalBaseNumber& baseNumber) const  {
  int level = baseNumber.getLevels();
  uint32_t intIndex = 0; 
  if (level > 0) {

    // depth index - silicon layer 1-st or 2-nd
    int layer = 0;
    if(baseNumber.getLevelName(0) == "SFSX") {
      layer = 1;
    } else if (baseNumber.getLevelName(0) == "SFSY") {
      layer = 2;
    } else {
      edm::LogWarning("EcalGeom") << "ESTBNumberingScheme: Wrong name"
				 << " of Presh. Si. Strip : " 
				 << baseNumber.getLevelName(0);
    }

    // Z index +Z = 1 ; -Z = 2
    int zside   = baseNumber.getCopyNumber("EREG");
    zside=2*(1-zside)+1;
    // wafer number
    int wafer = baseNumber.getCopyNumber(3);
    int x=0,y=0;
    // strip number inside wafer
    int strip = baseNumber.getCopyNumber(0);

    x = iX[wafer];
    y = iY[wafer];
    
    intIndex =  ESDetId(strip, x, y, layer, zside).rawId(); 
    
    LogDebug("EcalGeom") << "ESTBNumberingScheme : zside " << zside 
			<< " layer " << layer << " wafer " << wafer << " X " 
			<< x << " Y "<< y << " strip " << strip 
			<< " UnitID 0x" << std::hex << intIndex << std::dec;
    for (int ich = 0; ich < level; ich++) {
      LogDebug("EcalGeom") << "Name = " << baseNumber.getLevelName(ich) 
			  << " copy = " << baseNumber.getCopyNumber(ich);
    }
  }

  return intIndex;
}
