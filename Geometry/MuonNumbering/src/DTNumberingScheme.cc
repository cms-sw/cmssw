#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;

DTNumberingScheme::DTNumberingScheme( const MuonDDDConstants& muonConstants ) {
  initMe(muonConstants);
}

DTNumberingScheme::DTNumberingScheme( const DDCompactView& cpv ) {
  MuonDDDConstants muonConstants(cpv);
  initMe(muonConstants);
}

void DTNumberingScheme::initMe ( const MuonDDDConstants& muonConstants ) {
  int theLevelPart=muonConstants.getValue("level");
  theRegionLevel=muonConstants.getValue("mb_region")/theLevelPart;
  theWheelLevel=muonConstants.getValue("mb_wheel")/theLevelPart;
  theStationLevel=muonConstants.getValue("mb_station")/theLevelPart;
  theSuperLayerLevel=muonConstants.getValue("mb_superlayer")/theLevelPart;
  theLayerLevel=muonConstants.getValue("mb_layer")/theLevelPart;
  theWireLevel=muonConstants.getValue("mb_wire")/theLevelPart;

  LogDebug( "DTNumbering" )
      << "Initialize DTNumberingScheme"
      << "\ntheRegionLevel " << theRegionLevel
      << "\ntheWheelLevel " << theWheelLevel
      << "\ntheStationLevel " << theStationLevel
      << "\ntheSuperLayerLevel " << theSuperLayerLevel
      << "\ntheLayerLevel " << theLayerLevel
      << "\ntheWireLevel " << theWireLevel;
}

int DTNumberingScheme::baseNumberToUnitNumber(const MuonBaseNumber num){
  
  LogDebug( "DTNumbering" ) <<num.getLevels();
  for (int level=1;level<=num.getLevels();level++) {
    LogDebug( "DTNumbering" ) << level << " " << num.getSuperNo(level)
			      << " " << num.getBaseNo(level);
  }

  if (num.getLevels()!=theWireLevel) {
    LogDebug( "DTNumbering" ) << "DTNS::BNToUN "
			      << "BaseNumber has " << num.getLevels() << " levels,"
			      << "need "<<theWireLevel;
    return 0;
  }

  return getDetId(num);
}

int DTNumberingScheme::getDetId(const MuonBaseNumber num) const {
  
  int wire_id=0;
  int layer_id=0;
  int superlayer_id=0;
  int sector_id=0;
  int station_id=0;
  int wheel_id=0;

  //decode significant barrel levels
  decode(num,
         wire_id,
         layer_id,
         superlayer_id,
         sector_id,
         station_id,
         wheel_id);
  
  DTWireId id(wheel_id,station_id,sector_id,superlayer_id,layer_id,wire_id);
  
  LogDebug( "DTNumbering" ) << "DTNumberingScheme: " << id;
  
  return id.rawId();
}

void DTNumberingScheme::decode(const MuonBaseNumber& num,
			       int& wire_id,
			       int& layer_id,
			       int& superlayer_id,
			       int& sector_id,
			       int& station_id,
			       int& wheel_id) const {
  for (int level=1;level<=num.getLevels();level++) {

    //decode
    if (level==theWheelLevel) {
      const int copyno=num.getBaseNo(level);      
      wheel_id=copyno-2;

    } else if (level==theStationLevel) {
      const int station_tag = num.getSuperNo(level);
      const int copyno = num.getBaseNo(level);
      station_id=station_tag;
      sector_id=copyno+1;   

    } else if (level==theSuperLayerLevel) {
      const int copyno = num.getBaseNo(level);
      superlayer_id = copyno + 1;

    } else if (level==theLayerLevel) {
      const int copyno = num.getBaseNo(level);
      layer_id=copyno+1;

    } else if (level==theWireLevel) {
      const int copyno = num.getBaseNo(level);
      wire_id = copyno+1;
    }
  }
}
