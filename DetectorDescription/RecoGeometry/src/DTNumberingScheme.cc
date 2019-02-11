#include "DetectorDescription/RecoGeometry/interface/DTNumberingScheme.h"
#include "DetectorDescription/DDCMS/interface/MuonNumbering.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"

using namespace cms;

DTNumberingScheme::DTNumberingScheme(MuonConstants& muonConstants) {
  initMe(muonConstants);
}

void
DTNumberingScheme::initMe(MuonConstants& muonConstants) {
  int levelPart = muonConstants["level"];
  theRegionLevel = muonConstants["mb_region"]/levelPart;
  theWheelLevel = muonConstants["mb_wheel"]/levelPart;
  theStationLevel = muonConstants["mb_station"]/levelPart;
  theSuperLayerLevel = muonConstants["mb_superlayer"]/levelPart;
  theLayerLevel = muonConstants["mb_layer"]/levelPart;
  theWireLevel = muonConstants["mb_wire"]/levelPart;
}

int
DTNumberingScheme::getDetId(const MuonBaseNumber& num) const {

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

  return id.rawId();
}

void
DTNumberingScheme::decode(const MuonBaseNumber& num,
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
