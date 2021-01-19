#include "Geometry/HGCalCommonData/interface/HGCalGeomRotation.h"
#include "FWCore/Utilities/interface/Exception.h"

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

  int uPrime, vPrime;

  if (sector == 1) {
    uPrime = -moduleV + offset;
    vPrime = moduleU - moduleV + offset;
  } else if (sector == 2) {
    uPrime = moduleV - moduleU;
    vPrime = -moduleU + offset;
  } else {
    throw cms::Exception("RotationException") << "HGCalGeomRotation: desired sector must be eother 0, 1 or 2";
  }
  moduleU = uPrime;
  moduleV = vPrime;
}

unsigned HGCalGeomRotation::uvMappingToSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const {
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
    throw cms::Exception("RotationException") << "HGCalGeomRotation: desired sector must be eother 0, 1 or 2";
  }

  moduleU = moduleURotated;
  moduleV = moduleVRotated;

  return sector;
}
