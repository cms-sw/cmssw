//#define EDM_ML_DEBUG

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

  // Discriminate pre-TDR and TDR scenarios

  const bool preTDR = (baseNumber.getLevelName(3).find("Ring") != std::string::npos);

  const uint32_t modCopy(baseNumber.getCopyNumber(2));

  const std::string_view& ringName(baseNumber.getLevelName(3));  // name of ring volume
  int modtyp(0);
  std::string_view baseName = ringName.substr(ringName.find(':') + 1);
  int ringCopy(::atoi(baseName.data() + 4));

  uint32_t discN, sectorS, sectorN;
  if (!preTDR) {
    discN = (baseNumber.getLevelName(4).find("Disc1") != std::string::npos) ? 0 : 1;
    sectorS = (baseNumber.getLevelName(3).find("Front") != std::string::npos) ? 0 : 1;
    sectorN = baseNumber.getCopyNumber(3);

    ETLDetId tmpId;
    ringCopy = static_cast<int>(tmpId.encodeSector(discN, sectorS, sectorN));

    modtyp = (baseNumber.getLevelName(2).find("_Left") != std::string::npos) ? 1 : 2;
  }

  // Side choice: up to scenario D38 is given by level 7 (HGCal v9)
  int nSide(7);
  const std::string_view& sideName(baseNumber.getLevelName(nSide));
  // Side choice: from scenario D41 is given by level 8 (HGCal v10)
  if (sideName.find("CALOECTSFront") != std::string::npos) {
    nSide = 8;
  }
  const uint32_t sideCopy(baseNumber.getCopyNumber(nSide));
  const uint32_t zside(sideCopy == 1 ? 1 : 0);

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << baseNumber.getLevelName(0) << ", " << baseNumber.getLevelName(1) << ", "
                          << baseNumber.getLevelName(2) << ", " << baseNumber.getLevelName(3) << ", "
                          << baseNumber.getLevelName(4) << ", " << baseNumber.getLevelName(5) << ", "
                          << baseNumber.getLevelName(6) << ", " << baseNumber.getLevelName(7) << ", "
                          << baseNumber.getLevelName(8) << ", " << baseNumber.getLevelName(9) << ", "
                          << baseNumber.getLevelName(10) << ", " << baseNumber.getLevelName(11);
#endif

  // error checking

  if ((modtyp != 0 && preTDR) || (modtyp == 0 && !preTDR)) {
    edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): "
                               << "****************** Bad module name = " << modtyp
                               << ", Volume Name = " << baseNumber.getLevelName(4);
    return 0;
  }

  if ((preTDR && (1 > modCopy || ETLDetId::kETLv1maxModule < modCopy)) ||
      (!preTDR && (1 > modCopy ||
                   static_cast<unsigned>(std::max(ETLDetId::kETLv4maxModule, ETLDetId::kETLv5maxModule)) < modCopy))) {
    edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): "
                               << "****************** Bad module copy = " << modCopy
                               << ", Volume Number = " << baseNumber.getCopyNumber(4);
    return 0;
  }

  if ((preTDR && (1 > ringCopy || ETLDetId::kETLv1maxRing < ringCopy)) ||
      (!preTDR && (1 > ringCopy || ETLDetId::kETLv4maxRing < ringCopy))) {
    edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): "
                               << "****************** Bad ring copy = " << ringCopy
                               << ", Volume Number = " << baseNumber.getCopyNumber(3);
    return 0;
  }

  // all inputs are fine. Go ahead and decode

  ETLDetId thisETLdetid(zside, ringCopy, modCopy, modtyp);
  const uint32_t intindex = thisETLdetid.rawId();

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << "ETL Numbering scheme: "
                          << " ring = " << ringCopy << " zside = " << zside << " module = " << modCopy
                          << " modtyp = " << modtyp << " Raw Id = " << intindex << thisETLdetid;
#endif
  if (!preTDR) {
    ETLDetId altETLdetid(zside, discN, sectorS, sectorN, modCopy, modtyp);
    const uint32_t altintindex = altETLdetid.rawId();
    if (intindex != altintindex) {
      edm::LogWarning("MTDGeom") << "Incorrect alternative construction \n"
                                 << "disc = " << discN << " disc side = " << sectorS << " sector = " << sectorN << "\n"
                                 << altETLdetid;
    }
  }

  return intindex;
}
