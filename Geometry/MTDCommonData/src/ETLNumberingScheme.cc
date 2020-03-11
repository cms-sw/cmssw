#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include <iostream>

#define EDM_ML_DEBUG

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

  const std::string& ringName(baseNumber.getLevelName(3));  // name of ring volume
  int modtyp(0);
  std::string baseName = ringName.substr(ringName.find(":") + 1);
  int ringCopy(::atoi(baseName.c_str() + 4));

  if (!preTDR) {
    uint32_t discN = (baseNumber.getLevelName(4).find("Disk1") != std::string::npos) ? 0 : 1;
    uint32_t quarterS = (baseNumber.getLevelName(3).find("Front") != std::string::npos) ? 0 : 1;
    uint32_t quarterN = baseNumber.getCopyNumber(3);
    const uint32_t quarterOffset = 4;

    ringCopy = quarterN + quarterS * quarterOffset + 2 * quarterOffset * discN;

    modtyp = (baseNumber.getLevelName(2).find("_2") != std::string::npos) ? 2 : 1;
  }

  // Side choice: up to scenario D38 is given by level 7 (HGCal v9)
  int nSide(7);
  const std::string& sideName(baseNumber.getLevelName(nSide));
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
      (!preTDR && (1 > modCopy || ETLDetId::kETLv2maxModule < modCopy))) {
    edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): "
                               << "****************** Bad module copy = " << modCopy
                               << ", Volume Number = " << baseNumber.getCopyNumber(4);
    return 0;
  }

  if ((preTDR && (1 > ringCopy || ETLDetId::kETLv1maxRing < ringCopy)) ||
      (!preTDR && (1 > ringCopy || ETLDetId::kETLv2maxRing < ringCopy))) {
    edm::LogWarning("MTDGeom") << "ETLNumberingScheme::getUnitID(): "
                               << "****************** Bad ring copy = " << ringCopy
                               << ", Volume Number = " << baseNumber.getCopyNumber(3);
    return 0;
  }

  // all inputs are fine. Go ahead and decode

  ETLDetId thisETLdetid(zside, ringCopy, modCopy, modtyp);
  const int32_t intindex = thisETLdetid.rawId();

#ifdef EDM_ML_DEBUG
  edm::LogInfo("MTDGeom") << "ETL Numbering scheme: "
                          << " ring = " << ringCopy << " zside = " << zside << " module = " << modCopy
                          << " modtyp = " << modtyp << " Raw Id = " << intindex << thisETLdetid;
#endif

  return intindex;
}
