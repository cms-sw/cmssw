//#define EDM_ML_DEBUG

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"

#include <iostream>
#include <cstring>

BTLNumberingScheme::BTLNumberingScheme() : MTDNumberingScheme() {
  LogDebug("MTDGeom") << "Creating BTLNumberingScheme";
}

BTLNumberingScheme::~BTLNumberingScheme() { LogDebug("MTDGeom") << "Deleting BTLNumberingScheme"; }

uint32_t BTLNumberingScheme::getUnitID(const MTDBaseNumber& baseNumber) const {
  uint32_t intindex(0);
  const uint32_t nLevels(baseNumber.getLevels());

  LogDebug("MTDGeom") << "BTLNumberingScheme geometry levels = " << nLevels;

  uint32_t zside(999), rodCopy(0), runitCopy(0), modCopy(0), modtyp(0), crystal(0);

  bool isDD4hepOK(false);
  if (nLevels == kBTLcrystalLevel + 1) {
    if (baseNumber.getLevelName(9) == "world_volume_1") {
      isDD4hepOK = true;
    }
  }

  auto bareBaseName = [&](std::string_view name) {
    size_t ipos = name.rfind('_');
    return (isDD4hepOK) ? name.substr(0, ipos) : name;
  };

  if (nLevels == kBTLcrystalLevel || isDD4hepOK) {
    LogDebug("MTDGeom") << bareBaseName(baseNumber.getLevelName(0)) << ", " << bareBaseName(baseNumber.getLevelName(1))
                        << ", " << bareBaseName(baseNumber.getLevelName(2)) << ", "
                        << bareBaseName(baseNumber.getLevelName(3)) << ", " << bareBaseName(baseNumber.getLevelName(4))
                        << ", " << bareBaseName(baseNumber.getLevelName(5)) << ", "
                        << bareBaseName(baseNumber.getLevelName(6)) << ", " << bareBaseName(baseNumber.getLevelName(7))
                        << ", " << bareBaseName(baseNumber.getLevelName(8));

    // barphiflat scenario

    if (baseNumber.getLevelName(0).find("Timingactive") != std::string_view::npos) {
      crystal = baseNumber.getCopyNumber(0);

      modCopy = baseNumber.getCopyNumber(2);
      rodCopy = baseNumber.getCopyNumber(3);

      const std::string_view& modName(baseNumber.getLevelName(2));  // name of module volume
      uint32_t pos = modName.find("Positive");

      zside = (pos <= modName.size() ? 1 : 0);
      std::string_view baseName = modName.substr(modName.find(':') + 1);

      modtyp = ::atoi(&baseName.at(7));
      if (modtyp == 17) {
        modtyp = 2;
      } else if (modtyp == 33) {
        modtyp = 3;
      }

      // error checking

      if (1 > crystal || 64 < crystal) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad crystal number = " << crystal
                                   << ", Volume Number = " << baseNumber.getCopyNumber(0);
        return 0;
      }

      if (1 > modtyp || 3 < modtyp) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad module name = " << modName
                                   << ", Volume Name = " << baseNumber.getLevelName(2);
        return 0;
      }

      if (1 > modCopy || 54 < modCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad module copy = " << modCopy
                                   << ", Volume Number = " << baseNumber.getCopyNumber(2);
        return 0;
      }

      if (1 > rodCopy || 36 < rodCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad rod copy = " << rodCopy
                                   << ", Volume Number = " << baseNumber.getCopyNumber(4);
        return 0;
      }

      if (1 < zside) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad side = " << zside
                                   << ", Volume Name = " << baseNumber.getLevelName(2);
        return 0;
      }
    } else if (baseNumber.getLevelName(0).find("BTLCrystal") != std::string_view::npos) {
      // v2 scenario

      crystal = baseNumber.getCopyNumber(0);
      modCopy = baseNumber.getCopyNumber(1);
      runitCopy = baseNumber.getCopyNumber(2);
      rodCopy = baseNumber.getCopyNumber(3);

      const std::string_view& rodName(baseNumber.getLevelName(3));  // name of module volume
      uint32_t pos = rodName.find("Zpos");
      zside = (pos <= rodName.size() ? 1 : 0);

      // for negative side swap module numbers betwee sides of the tray, so as to keep the same number for the same phi angle
      // in the existing model. This introduces a misalignemtn between module number and volume copy for the negative side.
      if (zside == 0) {
        modCopy = negModCopy[modCopy - 1];
      }

      modtyp = ::atoi(&bareBaseName(baseNumber.getLevelName(2)).back());

      // error checking

      if (1 > crystal || BTLDetId::kCrystalsPerModuleV2 < crystal) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad crystal number = " << crystal
                                   << ", Volume Number = " << baseNumber.getCopyNumber(0);
        return 0;
      }

      if (1 > modtyp || 3 < modtyp) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad RU name, Volume Name = "
                                   << bareBaseName(baseNumber.getLevelName(2));
        return 0;
      }

      if (1 > modCopy || BTLDetId::kModulesPerRUV2 < modCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad module copy = " << modCopy
                                   << ", Volume Number = " << baseNumber.getCopyNumber(1);
        return 0;
      }

      if (1 > runitCopy || BTLDetId::kRUPerTypeV2 < runitCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad readout unit copy = " << runitCopy
                                   << ", Volume Number = " << baseNumber.getCopyNumber(2);
        return 0;
      }

      if (1 > rodCopy || BTLDetId::HALF_ROD < rodCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad rod copy = " << rodCopy
                                   << ", Volume Number = " << baseNumber.getCopyNumber(3);
        return 0;
      }

      if (1 < zside) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad side = " << zside
                                   << ", Volume Name = " << baseNumber.getLevelName(3);
        return 0;
      }
    }

    // all inputs are fine. Go ahead and decode

    BTLDetId thisBTLdetid(zside, rodCopy, runitCopy, modCopy, modtyp, crystal);
    intindex = thisBTLdetid.rawId();

  } else if (nLevels == kBTLmoduleLevel && baseNumber.getLevelName(0).find("BTLModule") != std::string_view::npos) {
    // v2 scenario, geographicalId per module
    // for tracking navigation geometry
    LogDebug("MTDGeom") << bareBaseName(baseNumber.getLevelName(0)) << ", " << bareBaseName(baseNumber.getLevelName(1))
                        << ", " << bareBaseName(baseNumber.getLevelName(2)) << ", "
                        << bareBaseName(baseNumber.getLevelName(3)) << ", " << bareBaseName(baseNumber.getLevelName(4))
                        << ", " << bareBaseName(baseNumber.getLevelName(5)) << ", "
                        << bareBaseName(baseNumber.getLevelName(6)) << ", " << bareBaseName(baseNumber.getLevelName(7));

    modCopy = baseNumber.getCopyNumber(0);
    runitCopy = baseNumber.getCopyNumber(1);
    rodCopy = baseNumber.getCopyNumber(2);

    const std::string_view& rodName(baseNumber.getLevelName(2));  // name of module volume
    uint32_t pos = rodName.find("Zpos");
    zside = (pos <= rodName.size() ? 1 : 0);

    // for negative side swap module numbers betwee sides of the tray, so as to keep the same number for the same phi angle
    // in the existing model. This introduces a misalignemtn between module number and volume copy for the negative side.
    if (zside == 0) {
      modCopy = negModCopy[modCopy - 1];
    }

    modtyp = ::atoi(&bareBaseName(baseNumber.getLevelName(1)).back());

    // error checking

    if (1 > modtyp || 3 < modtyp) {
      edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                 << "****************** Bad RU name, Volume Name = "
                                 << bareBaseName(baseNumber.getLevelName(1));
      return 0;
    }

    if (1 > modCopy || BTLDetId::kModulesPerRUV2 < modCopy) {
      edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                 << "****************** Bad module copy = " << modCopy
                                 << ", Volume Number = " << baseNumber.getCopyNumber(0);
      return 0;
    }

    if (1 > runitCopy || BTLDetId::kRUPerTypeV2 < runitCopy) {
      edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                 << "****************** Bad readout unit copy = " << runitCopy
                                 << ", Volume Number = " << baseNumber.getCopyNumber(1);
      return 0;
    }

    if (1 > rodCopy || BTLDetId::HALF_ROD < rodCopy) {
      edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                 << "****************** Bad rod copy = " << rodCopy
                                 << ", Volume Number = " << baseNumber.getCopyNumber(2);
      return 0;
    }

    if (1 < zside) {
      edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                 << "****************** Bad side = " << zside
                                 << ", Volume Name = " << baseNumber.getLevelName(2);
      return 0;
    }

    // all inputs are fine. Go ahead and decode

    BTLDetId thisBTLdetid(zside, rodCopy, runitCopy, modCopy, modtyp, 0);
    intindex = thisBTLdetid.geographicalId(BTLDetId::CrysLayout::v2).rawId();

  } else {
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                               << "Not enough levels found in MTDBaseNumber ( " << nLevels
                               << ") or not correct path. Returning 0";
    return 0;
  }

  LogDebug("MTDGeom") << "BTL Numbering scheme: "
                      << " zside = " << zside << " rod = " << rodCopy << " modtyp = " << modtyp << " RU = " << runitCopy
                      << " module = " << modCopy << " crystal = " << crystal << " Raw Id = " << intindex << "\n"
                      << BTLDetId(intindex);

  return intindex;
}
