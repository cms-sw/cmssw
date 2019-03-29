#include "Geometry/MuonNumbering/interface/DD4hep_DTNumberingScheme.h"
#include "Geometry/MuonNumbering/interface/DD4hep_MuonNumbering.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include <cassert>

using namespace cms;

DTNumberingScheme::DTNumberingScheme(const MuonConstants& muonConstants) {
  initMe(muonConstants);
}

void
DTNumberingScheme::initMe(const MuonConstants& muonConstants) {
  int levelPart = get("level", muonConstants);
  assert(levelPart != 0);
  theRegionLevel = get("mb_region", muonConstants)/levelPart;
  theWheelLevel = get("mb_wheel", muonConstants)/levelPart;
  theStationLevel = get("mb_station", muonConstants)/levelPart;
  theSuperLayerLevel = get("mb_superlayer", muonConstants)/levelPart;
  theLayerLevel = get("mb_layer", muonConstants)/levelPart;
  theWireLevel = get("mb_wire", muonConstants)/levelPart;
}

int
DTNumberingScheme::getDetId(const MuonBaseNumber& num) const {

  int wire_id(0);
  int layer_id(0);
  int superlayer_id(0);
  int sector_id(0);
  int station_id(0);
  int wheel_id(0);

  //decode significant barrel levels
  decode(num,
         wire_id,
         layer_id,
         superlayer_id,
         sector_id,
         station_id,
         wheel_id);

  DTWireId id(wheel_id, station_id, sector_id, superlayer_id, layer_id, wire_id);

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
  for(int level = 1; level <= num.getLevels(); ++level) {

    //decode
    if(level == theWheelLevel) {
      const int copyno = num.getBaseNo(level);      
      wheel_id = copyno-2;

    } else if (level == theStationLevel) {
      const int station_tag = num.getSuperNo(level);
      const int copyno = num.getBaseNo(level);
      station_id = station_tag;
      sector_id = copyno + 1;   
    } else if(level == theSuperLayerLevel) {
      const int copyno = num.getBaseNo(level);
      superlayer_id = copyno + 1;
    } else if (level == theLayerLevel) {
      const int copyno = num.getBaseNo(level);
      layer_id = copyno+1;
    } else if(level == theWireLevel) {
      const int copyno = num.getBaseNo(level);
      wire_id = copyno + 1;
    }
  }
}

const int
DTNumberingScheme::get(const char* key,
		       const MuonConstants& muonConstants) const {
  int result(0);
  auto const& it = muonConstants.find(key);
  if(it != end(muonConstants)) 
    result = it->second;
  return result;
}
