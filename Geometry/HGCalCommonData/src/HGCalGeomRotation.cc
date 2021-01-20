#include "Geometry/HGCalCommonData/interface/HGCalGeomRotation.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void HGCalGeomRotation::uvMappingFromSector0(WaferCentring waferCentring,
                                             int& moduleU,
                                             int& moduleV,
                                             unsigned sector) const {

  if (sector == 0) {
    return;
  }

  int offset;

  if (waferCentring == WaferCentred) {
    offset = 0;
  } else if (waferCentring == CornerCentredY) {
    offset = -1;
  } else if (waferCentring == CornerCentredMercedes) {
    offset = 1;
  } else {
    throw cms::Exception("RotationException")
        << "HGCalGeomRotation: WaferCentring must be one of: WaferCentred, CornerCentredY or CornerCentredMercedes";
  }

  if (_sectorType == Sector60Degrees && waferCentring != WaferCentred){
    edm::LogError("HGCalGeomRotation") << "HGCalGeomRotation: 60 degree sector defintion selected, but not WaferCentred centring. This is incompatible, switching to WaferCentred centring";
    waferCentring = WaferCentred;
  }

  int uPrime, vPrime;

  if ( _sectorType == Sector120Degrees ){
    if (sector == 1) {
      uPrime = -moduleV + offset;
      vPrime = moduleU - moduleV + offset;
    } else if (sector == 2) {
      uPrime = moduleV - moduleU;
      vPrime = -moduleU + offset;
    } else {
      throw cms::Exception("RotationException") << "HGCalGeomRotation: desired sector must be either 0, 1 or 2";
    }

    moduleU = uPrime;
    moduleV = vPrime;

  }
  else if( _sectorType == Sector60Degrees){
    if ( sector > 5 ){
      throw cms::Exception("RotationException") << "HGCalGeomRotation: desired sector must be either 0, 1, 2, 3, 4, 5 or 6";
    }
    for (unsigned rot = 0; rot < sector; rot++){
      RotateModule60DegreesAnticlockwise( moduleU, moduleV);
    }
  }

}

unsigned HGCalGeomRotation::uvMappingToSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const {
  unsigned sector = 0;

  if (_sectorType == Sector60Degrees){

    sector = uvMappingTo60DegreeSector0( waferCentring, moduleU, moduleV );

  } else if (_sectorType == Sector120Degrees){

    sector = uvMappingTo120DegreeSector0( waferCentring, moduleU, moduleV );

  }

  return sector;

}

unsigned HGCalGeomRotation::uvMappingTo120DegreeSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const {

  unsigned sector = 0;
  int offset;

  if (waferCentring == WaferCentred) {
    if (moduleU > 0 && moduleV >= 0)
      return sector;
    
    offset = 0;
    if (moduleU >= moduleV && moduleV < 0)
      sector = 2;
    else
      sector = 1;
    
  } else if (waferCentring == CornerCentredY) {
    if (moduleU >= 0 && moduleV >= 0)
      return sector;
    
    offset = -1;
    if (moduleU > moduleV && moduleV < 0)
      sector = 2;
    else
      sector = 1;
    
  } else if (waferCentring == CornerCentredMercedes) {
    if (moduleU >= 1 && moduleV >= 1)
      return sector;
    
    offset = 1;
      if (moduleU >= moduleV && moduleV < 1)
	sector = 2;
      else
	sector = 1;
  } else {
    throw cms::Exception("RotationException")
      << "HGCalGeomRotation: WaferCentring must be one of: WaferCentred, CornerCentredY or CornerCentredMercedes";
  }

  int moduleURotated, moduleVRotated;
    
  if (sector == 1) {
    moduleURotated = moduleV - moduleU;
    moduleVRotated = -moduleU + offset;
    
  } else if (sector == 2) {
    moduleURotated = -moduleV + offset;
    moduleVRotated = moduleU - moduleV + offset;
  } else {
    throw cms::Exception("RotationException") << "HGCalGeomRotation: desired sector must be either 0, 1 or 2";
  }

  moduleU = moduleURotated;
  moduleV = moduleVRotated;

  return sector;
}



unsigned HGCalGeomRotation::uvMappingTo60DegreeSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const {

  unsigned sector = 0;

  if ( waferCentring != WaferCentred ){

    edm::LogError("HGCalGeomRotation") << "HGCalGeomRotation: 60 degree sector defintion selected, but not WaferCentred centring. This is incompatible, switching to WaferCentred centring";
    waferCentring = WaferCentred;

  }

  if (moduleU > 0 && moduleV >= 0){
    if (moduleV <= moduleU){
      return sector;
    }
    else{
      sector = 1;
    }
  }
  if (moduleU >= moduleV && moduleV < 0){
    if ( moduleU >= 0 ){
      sector = 5;
    }
    else{
      sector = 4;
    }
  }
  else{
    if ( moduleV > 0 ){
      sector = 2;
    }
    else{
      sector = 3;
    }
  }

  for (unsigned rot = 0; rot < sector; rot++){
    RotateModule60DegreesClockwise( moduleU, moduleV);
  }

  return sector;

}

void HGCalGeomRotation::RotateModule60DegreesAnticlockwise(int& moduleU, int& moduleV) const {
  
  int moduleURotated, moduleVRotated;
  moduleURotated = moduleU - moduleV;
  moduleVRotated = moduleU;

  moduleU = moduleURotated;
  moduleV = moduleVRotated;

}

void HGCalGeomRotation::RotateModule60DegreesClockwise(int& moduleU, int& moduleV) const {
  
  int moduleURotated, moduleVRotated;
  moduleURotated = moduleV;
  moduleVRotated = moduleV - moduleU;

  moduleU = moduleURotated;
  moduleV = moduleVRotated;

}
