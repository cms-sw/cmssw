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

  if (sectorType_ == SectorType::Sector60Degrees) {
    uvMappingFrom60DegreeSector0(waferCentring, moduleU, moduleV, sector);
  } else if (sectorType_ == SectorType::Sector120Degrees) {
    uvMappingFrom120DegreeSector0(waferCentring, moduleU, moduleV, sector);
  }
}

unsigned HGCalGeomRotation::uvMappingToSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const {
  unsigned sector = 0;

  if (sectorType_ == SectorType::Sector60Degrees) {
    sector = uvMappingTo60DegreeSector0(waferCentring, moduleU, moduleV);
  } else if (sectorType_ == SectorType::Sector120Degrees) {
    sector = uvMappingTo120DegreeSector0(waferCentring, moduleU, moduleV);
  }

  return sector;
}

void HGCalGeomRotation::uvMappingFrom60DegreeSector0(WaferCentring waferCentring,
                                                     int& moduleU,
                                                     int& moduleV,
                                                     unsigned sector) const {
  if (waferCentring != WaferCentring::WaferCentred) {
    edm::LogError("HGCalGeomRotation")
        << "HGCalGeomRotation: 60 degree sector defintion selected, but not WaferCentred centring. This is "
           "incompatible, assuming WaferCentred centring";
  }

  if (sector > 5) {
    throw cms::Exception("RotationException") << "HGCalGeomRotation: desired sector must be either 0, 1, 2, 3, 4, or 5";
  }
  for (unsigned rot = 0; rot < sector; rot++) {
    RotateModule60DegreesAnticlockwise(moduleU, moduleV);
  }
}

void HGCalGeomRotation::uvMappingFrom120DegreeSector0(WaferCentring waferCentring,
                                                      int& moduleU,
                                                      int& moduleV,
                                                      unsigned sector) const {
  int offset;

  if (waferCentring == WaferCentring::WaferCentred) {
    offset = 0;
  } else if (waferCentring == WaferCentring::CornerCentredY) {
    offset = -1;
  } else if (waferCentring == WaferCentring::CornerCentredMercedes) {
    offset = 1;
  } else {
    throw cms::Exception("RotationException")
        << "HGCalGeomRotation: WaferCentring must be one of: WaferCentred, CornerCentredY or CornerCentredMercedes";
  }

  if (sector > 2) {
    edm::LogError("RotationException") << "HGCalGeomRotation: desired sector must be either 0, 1 or 2";
    return;
  }
  for (unsigned rot = 0; rot < sector; rot++) {
    RotateModule120DegreesAnticlockwise(moduleU, moduleV, offset);
  }
}

unsigned HGCalGeomRotation::uvMappingTo120DegreeSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const {
  unsigned sector = 0;
  int offset;

  if (waferCentring == WaferCentring::WaferCentred) {
    if (moduleU > 0 && moduleV >= 0)
      return sector;

    offset = 0;
    if (moduleU >= moduleV && moduleV < 0)
      sector = 2;
    else
      sector = 1;

  } else if (waferCentring == WaferCentring::CornerCentredY) {
    if (moduleU >= 0 && moduleV >= 0)
      return sector;

    offset = -1;
    if (moduleU > moduleV && moduleV < 0)
      sector = 2;
    else
      sector = 1;

  } else if (waferCentring == WaferCentring::CornerCentredMercedes) {
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

  for (unsigned rot = 0; rot < sector; rot++) {
    RotateModule120DegreesClockwise(moduleU, moduleV, offset);
  }

  return sector;
}

unsigned HGCalGeomRotation::uvMappingTo60DegreeSector0(WaferCentring waferCentring, int& moduleU, int& moduleV) const {
  unsigned sector = 0;

  if (waferCentring != WaferCentring::WaferCentred) {
    edm::LogError("HGCalGeomRotation")
        << "HGCalGeomRotation: 60 degree sector defintion selected, but not WaferCentred centring. This is "
           "incompatible, assuming WaferCentred centring";
  }

  if (moduleU > 0 && moduleV >= 0) {
    if (moduleV < moduleU) {
      return sector;
    } else {
      sector = 1;
    }
  } else if (moduleU >= moduleV && moduleV < 0) {
    if (moduleU >= 0) {
      sector = 5;
    } else {
      sector = 4;
    }
  } else {
    if (moduleV > 0) {
      sector = 2;
    } else {
      sector = 3;
    }
  }

  for (unsigned rot = 0; rot < sector; rot++) {
    RotateModule60DegreesClockwise(moduleU, moduleV);
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

void HGCalGeomRotation::RotateModule120DegreesAnticlockwise(int& moduleU, int& moduleV, int offset) const {
  int moduleURotated, moduleVRotated;

  moduleURotated = -moduleV + offset;
  moduleVRotated = moduleU - moduleV + offset;

  moduleU = moduleURotated;
  moduleV = moduleVRotated;
}

void HGCalGeomRotation::RotateModule120DegreesClockwise(int& moduleU, int& moduleV, int offset) const {
  int moduleURotated, moduleVRotated;

  moduleURotated = moduleV - moduleU;
  moduleVRotated = -moduleU + offset;

  moduleU = moduleURotated;
  moduleV = moduleVRotated;
}
