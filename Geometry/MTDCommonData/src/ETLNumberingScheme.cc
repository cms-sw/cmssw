#define EDM_ML_DEBUG

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include <iostream>

ETLNumberingScheme::ETLNumberingScheme() : MTDNumberingScheme() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << "Creating ETLNumberingScheme";
#endif
}

ETLNumberingScheme::~ETLNumberingScheme() {
#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << "Deleting ETLNumberingScheme";
#endif
}

uint32_t ETLNumberingScheme::getUnitID(const MTDBaseNumber& baseNumber) const {
  const uint32_t nLevels(baseNumber.getLevels());

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << "ETLNumberingScheme geometry levels = " << nLevels;
#endif

  if (11 > nLevels) {
    edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): "
                               << "Not enough levels found in MTDBaseNumber ( " << nLevels << ") Returning 0";
    return 0;
  }

  const bool prev8(baseNumber.getLevelName(2).find("Sensor") != std::string::npos);
  const bool prev9(baseNumber.getLevelName(2).find("Half_") == std::string::npos);

  std::stringstream ss;
  auto dump_levels = [&]() {
    for (size_t ii = 0; ii < nLevels; ii++) {
      ss << ii << ": " << baseNumber.getLevelName(ii) << "  ";
    }
    ss << "\nReturning 0";
    return ss.str();
  };

  uint32_t modCopy(baseNumber.getCopyNumber(2));
  if (!prev9) {
    modCopy = baseNumber.getCopyNumber(3);
  }
  uint32_t modtyp(0);
  if (prev9) {
    modtyp = (baseNumber.getLevelName(2).find("_Left") != std::string::npos) ? 1 : 2;
  } else {
    modtyp = baseNumber.getCopyNumber(2);
  }
  uint32_t sensor(0);
  if (!prev8) {
    sensor = baseNumber.getCopyNumber(1);
  }
  // for v9 keep the same sensor order inside a module as in v8
  if (!prev9 && modtyp == 2) {
    sensor = (sensor == 1) ? 2 : 1;
  }

  uint32_t discN, sectorS, sectorN;
  uint32_t offset = prev9 ? 3 : 4;
  discN = (baseNumber.getLevelName(offset + 1).find("Disc1") != std::string::npos) ? 0 : 1;
  sectorS = (baseNumber.getLevelName(offset).find("Front") != std::string::npos) ? 0 : 1;
  sectorN = baseNumber.getCopyNumber(offset);

  ETLDetId tmpId;
  uint32_t ringCopy = static_cast<int>(tmpId.encodeSector(discN, sectorS, sectorN));

  uint32_t nSide(999);
  if (baseNumber.getLevelName(7).find("CALOECTSFront") != std::string::npos) {
    nSide = 8;
  } else if (baseNumber.getLevelName(8).find("CALOECTSFront") != std::string::npos) {
    nSide = 9;
  } else {
    edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): incorrect volume stack: \n" << dump_levels();
    return 0;
  }
  const uint32_t sideCopy(baseNumber.getCopyNumber(nSide));
  const uint32_t zside(sideCopy == 1 ? 1 : 0);

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << dump_levels();
#endif

  // error checking

  if (modtyp == 0) {
    edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): "
                               << "****************** Bad module name = " << modtyp
                               << ", Volume Name = " << baseNumber.getLevelName(4);
    return 0;
  }

  if (1 > modCopy || static_cast<unsigned>(std::max(ETLDetId::kETLv4maxModule, ETLDetId::kETLv5maxModule)) < modCopy) {
    edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): "
                               << "****************** Bad module copy = " << modCopy
                               << ", Volume Number = " << baseNumber.getCopyNumber(4);
    return 0;
  }

  if (1 > ringCopy || ETLDetId::kETLv4maxRing < ringCopy) {
    edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): "
                               << "****************** Bad ring copy = " << ringCopy
                               << ", Volume Number = " << baseNumber.getCopyNumber(3);
    return 0;
  }

  // all inputs are fine. Go ahead and decode

  // Different for v8 and pre v8 ETL geometries
  uint32_t intindex = 0;
  uint32_t altintindex = 0;
  if (prev8) {
    ETLDetId thisETLdetid(zside, ringCopy, modCopy, modtyp);
    intindex = thisETLdetid.rawId();
#ifdef EDM_ML_DEBUG
    edm::LogInfo("MTDGeom") << "ETL Numbering scheme: "
                            << " ring = " << ringCopy << " zside = " << zside << " module = " << modCopy
                            << " modtyp = " << modtyp << " Raw Id = " << intindex;
#endif

    ETLDetId altETLdetid(zside, discN, sectorS, sectorN, modCopy, modtyp);
    altintindex = altETLdetid.rawId();

  } else {
    ETLDetId thisETLdetid(zside, ringCopy, modCopy, modtyp, sensor);
    intindex = thisETLdetid.rawId();
#ifdef EDM_ML_DEBUG
    edm::LogInfo("MTDGeom") << "ETL Numbering scheme: "
                            << " ring = " << ringCopy << " zside = " << zside << " module = " << modCopy
                            << " modtyp = " << modtyp << " sensor = " << sensor << " Raw Id = " << intindex;
#endif

    ETLDetId altETLdetid(zside, discN, sectorS, sectorN, modCopy, modtyp, sensor);
    altintindex = altETLdetid.rawId();
  }

  if (intindex != altintindex) {
    edm::LogWarning("MTDGeom") << "Incorrect alternative construction \n"
                               << "disc = " << discN << " disc side = " << sectorS << " sector = " << sectorN << "\n";
  }

  return intindex;
}
