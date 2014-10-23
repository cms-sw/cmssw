#include "Geometry/MuonNumbering/interface/ME0NumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include <iostream>

//#define LOCAL_DEBUG

ME0NumberingScheme::ME0NumberingScheme( const MuonDDDConstants& muonConstants ) {
  initMe(muonConstants);
}

ME0NumberingScheme::ME0NumberingScheme( const DDCompactView& cpv ){
  MuonDDDConstants muonConstants(cpv);
  initMe(muonConstants);
}

void ME0NumberingScheme::initMe ( const MuonDDDConstants& muonConstants ) {
  int theLevelPart= muonConstants.getValue("level");
  theRegionLevel  = muonConstants.getValue("m0_region")/theLevelPart;
  theLayerLevel   = muonConstants.getValue("m0_layer")/theLevelPart;
  theSectorLevel  = muonConstants.getValue("m0_sector")/theLevelPart;
  theRollLevel    = muonConstants.getValue("m0_roll")/theLevelPart;
#ifdef LOCAL_DEBUG
  std::cout << "Initialize ME0NumberingScheme" <<std::endl;
  std::cout << "theRegionLevel " << theRegionLevel <<std::endl;
  std::cout << "theLayerLevel "  << theLayerLevel   <<std::endl;
  std::cout << "theSectorLevel " << theSectorLevel <<std::endl;
  std::cout << "theRollLevel "   << theRollLevel   <<std::endl;
#endif
}

int ME0NumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber num) {

#ifdef LOCAL_DEBUG
  std::cout << "ME0Numbering "<<num.getLevels()<<std::endl;
  for (int level=1;level<=num.getLevels();level++) {
    std::cout << "level "<<level << " " << num.getSuperNo(level)
	      << " " << num.getBaseNo(level) << std::endl;
  }
#endif

  int maxLevel = theLayerLevel;
  if (num.getLevels()!=maxLevel) {
    std::cout << "MuonME0NS::BNToUN "
	      << "BaseNumber has " << num.getLevels() << " levels,"
	      << "need "<<maxLevel<<std::endl;
    return 0;
  }

  int region(0), layer(0), chamber(0), roll(0);

  //decode significant ME0 levels
  
  if (num.getBaseNo(theRegionLevel) == 0) 
    region = 1;
  else                                    
    region =-1;
  layer   = num.getBaseNo(theLayerLevel)+1;
  roll=1;
  chamber = num.getBaseNo(theSectorLevel) + 1;
  // collect all info
  
#ifdef LOCAL_DEBUG
  std::cout << "ME0NumberingScheme: Region " << region 
	    << " Layer " << layer
	    << " Chamber " << chamber << " Roll " << roll << std::endl;
#endif

  // Build the actual numbering
  ME0DetId id(region,layer,chamber,roll);
  
  
#ifdef LOCAL_DEBUG
  std::cout << " DetId " << id << std::endl;
#endif
      
  return id.rawId();
}




