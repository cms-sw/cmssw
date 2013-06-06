#include "Geometry/MuonNumbering/interface/GEMNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;

GEMNumberingScheme::GEMNumberingScheme( const MuonDDDConstants& muonConstants ) {
  initMe(muonConstants);
}

GEMNumberingScheme::GEMNumberingScheme( const DDCompactView& cpv ){
  MuonDDDConstants muonConstants(cpv);
  initMe(muonConstants);
}

void GEMNumberingScheme::initMe ( const MuonDDDConstants& muonConstants ) {
  int theLevelPart= muonConstants.getValue("level");
  theRegionLevel  = muonConstants.getValue("mg_region")/theLevelPart;
  theStationLevel = muonConstants.getValue("mg_station")/theLevelPart;
  theRingLevel    = muonConstants.getValue("mg_ring")/theLevelPart;
  theSectorLevel  = muonConstants.getValue("mg_sector")/theLevelPart;
  theRollLevel    = muonConstants.getValue("mg_roll")/theLevelPart;

  LogDebug( "GEMNumbering" )
      << "Initialize GEMNumberingScheme"
      << "\ntheRegionLevel " << theRegionLevel
      << "\ntheStationLevel "<< theStationLevel
      << "\ntheRingLevel "   << theRingLevel
      << "\ntheSectorLevel " << theSectorLevel
      << "\ntheRollLevel "   << theRollLevel;
}

int GEMNumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber num) {

  LogDebug( "GEMNumbering" ) << num.getLevels();
  for (int level=1;level<=num.getLevels();level++) {
    LogDebug( "GEMNumbering" ) << level << " " << num.getSuperNo(level)
			       << " " << num.getBaseNo(level);
  }

  int maxLevel = theRollLevel;
  if (num.getLevels()!=maxLevel) {
    LogDebug( "GEMNumbering" ) << "MuonGEMNS::BNToUN "
			       << "BaseNumber has " << num.getLevels() << " levels,"
			       << "need "<<maxLevel;
    return 0;
  }

  int region(0), ring(0), station(0), layer(0), chamber(0), roll(0);

  //decode significant GEM levels
  
  if (num.getBaseNo(theRegionLevel) == 0) region = 1;
  else                                    region =-1;
  station = num.getBaseNo(theStationLevel)+1;
  ring    = num.getBaseNo(theRingLevel)+1;
  roll    = num.getBaseNo(theRollLevel)+1;
  const int copyno = num.getBaseNo(theSectorLevel) + 1;
  if (copyno < 50) {
    if (copyno%2 == 0) {
      layer   = 2;
      chamber = copyno-1;
    } else {
      layer   = 1;
      chamber = copyno;
    }
  } else {
    int copynp = copyno - 50;
    if (copynp%2 != 0) {
      layer   = 2;
      chamber = copynp-1;
    } else {
      layer   = 1;
      chamber = copynp;
    }
  }

  // collect all info

  LogDebug( "GEMNumbering" ) << "GEMNumberingScheme: Region " << region << " Ring "
			     << ring << " Station " << station << " Layer " << layer
			     << " Chamber " << chamber << " Roll " << roll;

  // Build the actual numbering
  GEMDetId id(region,ring,station,layer,chamber, roll);
  
  LogDebug( "GEMNumbering" ) << " DetId " << id;
      
  return id.rawId();
}




