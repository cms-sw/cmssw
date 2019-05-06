#include "Geometry/MuonNumbering/interface/ME0NumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

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
  theNEtaPart     = muonConstants.getValue("m0_nroll");

  // Debug using LOCAL_DEBUG
#ifdef LOCAL_DEBUG
  edm::LogVerbatim("ME0NumberingScheme") 
    << "Initialize ME0NumberingScheme" 
    << "\ntheRegionLevel " << theRegionLevel
    << "\ntheLayerLevel "  << theLayerLevel 
    << "\ntheSectorLevel " << theSectorLevel
    << "\ntheRollLevel "   << theRollLevel
    << "\ntheNEtaPart  "   << theNEtaPart;
#endif
  // -----------------------
}

int ME0NumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber& num) {

    edm::LogVerbatim("ME0NumberingScheme")
      << "ME0NumberingScheme::baseNumberToUnitNumber BEGIN ";
  // Debug using LOCAL_DEBUG
#ifdef LOCAL_DEBUG
    edm::LogVerbatim("ME0NumberingScheme") << "ME0Numbering "<< num.getLevels();
    for (int level=1;level<=num.getLevels();level++) {
      edm::LogVerbatim("ME0NumberingScheme") 
	<< "level "<<level << " " << num.getSuperNo(level)
	<< " " << num.getBaseNo(level);
    }
#endif
  // -----------------------

  int maxLevel = theRollLevel;
  if (num.getLevels()!=maxLevel) {
    throw cms::Exception("MuonNumbering") << "MuonME0NS::BNToUN "
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
  chamber = num.getBaseNo(theSectorLevel) + 1;
  roll = num.getBaseNo(theRollLevel)+1;

  // collect all info
  
  // Debug using LOCAL_DEBUG
#ifdef LOCAL_DEBUG
  edm::LogVerbatim("ME0NumberingScheme") 
    << "ME0NumberingScheme: Region " << region << " Layer " << layer
    << " Chamber " << chamber << " Roll " << roll;
#endif
  // -----------------------

  // Build the actual numbering
  ME0DetId id(region,layer,chamber,roll);
  
  // Debug using LOCAL_DEBUG  
#ifdef LOCAL_DEBUG
  edm::LogVerbatim("ME0NumberingScheme") << " DetId " << id;
#endif
  // ---------------------
    
  return id.rawId();
}
