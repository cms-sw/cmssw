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

  uint32_t zside(999), rodCopy(0), runitCopy(0), modCopy(0), modType(0), crystal(0), dmodCopy(0), smodCopy(0);

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
    LogDebug("MTDGeom") << bareBaseName(baseNumber.getLevelName(0)) << "[" << baseNumber.getCopyNumber(0) << "], "
                        << bareBaseName(baseNumber.getLevelName(1)) << "[" << baseNumber.getCopyNumber(1) << "], "
                        << bareBaseName(baseNumber.getLevelName(2)) << "[" << baseNumber.getCopyNumber(2) << "], "
                        << bareBaseName(baseNumber.getLevelName(3)) << "[" << baseNumber.getCopyNumber(3) << "], "
                        << bareBaseName(baseNumber.getLevelName(4)) << "[" << baseNumber.getCopyNumber(4) << "], "
                        << bareBaseName(baseNumber.getLevelName(5)) << "[" << baseNumber.getCopyNumber(5) << "], "
                        << bareBaseName(baseNumber.getLevelName(6)) << "[" << baseNumber.getCopyNumber(6) << "], "
                        << bareBaseName(baseNumber.getLevelName(7)) << "[" << baseNumber.getCopyNumber(7) << "], "
                        << bareBaseName(baseNumber.getLevelName(8)) << "[" << baseNumber.getCopyNumber(8) << "]";
    // barphiflat (v1) scenario - not supported
    if (baseNumber.getLevelName(4).find("Timingactive") != std::string_view::npos) {
      edm::LogError("MTDGeom") << "Geometry v1 of BTL not supported, run on a Geometry configuration D95 or latest ";
      throw cms::Exception("Configuration") << "Invalid BTL Geometry configuration (v1)";
    } else if (baseNumber.getLevelName(0).find("BTLCrystal") != std::string_view::npos &&
               baseNumber.getLevelName(4).find("LayerTiming") != std::string_view::npos) {
      // v4 scenario
      // zside copy number
      const std::string_view& rodName(baseNumber.getLevelName(3));  // name of module volume
      uint32_t pos = rodName.find("Zpos");
      zside = (pos <= rodName.size() ? 1 : 0);

      // rod (tray) copy number
      rodCopy = baseNumber.getCopyNumber(3) - 1;

      // RU, global module and crystal copy numbers
      // (everything start from 0)
      runitCopy = baseNumber.getCopyNumber(2) - 1;

      modCopy = int(baseNumber.getCopyNumber(1)) - 1;
      crystal = int(baseNumber.getCopyNumber(0)) - 1;

      // Detector and sensor module numbers from global module number 0-23
      dmodCopy = int((modCopy / BTLDetId::kDModulesInRUCol) / BTLDetId::kSModulesInDM) +
                 (modCopy % BTLDetId::kDModulesInRUCol) * BTLDetId::kDModulesInRURow;
      smodCopy = int(modCopy / BTLDetId::kDModulesInRUCol) % BTLDetId::kSModulesInDM;

      // error checking
      if (0 > int(crystal) || BTLDetId::kCrystalsPerModuleV2 - 1 < crystal) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad crystal number = " << int(crystal)
                                   << ", Volume Number (counting from 0)= " << baseNumber.getCopyNumber(0) - 1;
        return 0;
      }

      if (0 > int(modCopy) || BTLDetId::kModulesPerRUV2 - 1 < modCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad module copy = " << int(modCopy)
                                   << ", Volume Number (counting from 0)= " << baseNumber.getCopyNumber(1) - 1;
        return 0;
      }

      if (0 > int(smodCopy) || BTLDetId::kSModulesPerDM - 1 < smodCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad detector module copy = " << int(smodCopy)
                                   << ", Volume Number (counting from 0)= " << baseNumber.getCopyNumber(1) - 1;
        return 0;
      }

      if (0 > int(dmodCopy) || BTLDetId::kDModulesPerRU - 1 < dmodCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad detector module copy = " << int(dmodCopy)
                                   << ", Volume Number (counting from 0)= " << baseNumber.getCopyNumber(1) - 1;
        return 0;
      }

      if (0 > int(runitCopy) || BTLDetId::kRUPerRod - 1 < runitCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad readout unit copy = " << int(runitCopy)
                                   << ", Volume Name= " << baseNumber.getLevelName(2)
                                   << ", Volume Number (counting from 0)= " << baseNumber.getCopyNumber(2) - 1;
        return 0;
      }

      if (0 > int(rodCopy) || BTLDetId::HALF_ROD - 1 < rodCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad rod copy = " << int(rodCopy)
                                   << ", Volume Number (counting from 0)= " << baseNumber.getCopyNumber(3);
        return 0;
      }

      if (1 < zside) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad side = " << zside
                                   << ", Volume Name = " << baseNumber.getLevelName(3);
        return 0;
      }

      // all inputs are fine. Go ahead and decode
      BTLDetId thisBTLdetid(zside, rodCopy, runitCopy, dmodCopy, smodCopy, crystal);
      intindex = thisBTLdetid.rawId();
    } else if (baseNumber.getLevelName(0).find("BTLCrystal") != std::string_view::npos &&
               baseNumber.getLevelName(4).find("Layer1Timing") != std::string_view::npos) {
      // v2 or v3 scenario

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

      bool isV2(bareBaseName(baseNumber.getLevelName(0)).back() != 'l');

#ifdef EDM_ML_DEBUG
      LogDebug("MTDGeom") << "BTLNumberingScheme::getUnitID(): isV2 " << isV2;
#endif

      if (isV2) {
        // V2: the type is embedded in crystal name
        modType = ::atoi(&bareBaseName(baseNumber.getLevelName(2)).back());
      } else {
        // V3: build type and RU number per type from global RU number
        modType = globalru2type[runitCopy - 1];
        runitCopy = globalru2ru[runitCopy - 1];
      }

      // error checking

      if (1 > crystal || BTLDetId::kCrystalsPerModuleV2 < crystal) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad crystal number = " << crystal
                                   << ", Volume Number = " << baseNumber.getCopyNumber(0);
        return 0;
      }

      if (1 > modType || 3 < modType) {
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

      // all inputs are fine. Go ahead and decode
      BTLDetId thisBTLdetid(zside, rodCopy, runitCopy, modCopy, modType, crystal, isV2);
      intindex = thisBTLdetid.rawId();
    }
  } else if (nLevels == kBTLmoduleLevel && baseNumber.getLevelName(0).find("BTLModule") != std::string_view::npos) {
    // v4 scenario, geographicalId per module
    // for tracking navigation geometry
    LogDebug("MTDGeom") << bareBaseName(baseNumber.getLevelName(0)) << "[" << baseNumber.getCopyNumber(0) << "], "
                        << bareBaseName(baseNumber.getLevelName(1)) << "[" << baseNumber.getCopyNumber(1) << "], "
                        << bareBaseName(baseNumber.getLevelName(2)) << "[" << baseNumber.getCopyNumber(2) << "], "
                        << bareBaseName(baseNumber.getLevelName(3)) << "[" << baseNumber.getCopyNumber(3) << "], "
                        << bareBaseName(baseNumber.getLevelName(4)) << "[" << baseNumber.getCopyNumber(4) << "], "
                        << bareBaseName(baseNumber.getLevelName(5)) << "[" << baseNumber.getCopyNumber(5) << "], "
                        << bareBaseName(baseNumber.getLevelName(6)) << "[" << baseNumber.getCopyNumber(6) << "], "
                        << bareBaseName(baseNumber.getLevelName(7)) << "[" << baseNumber.getCopyNumber(7) << "]";
    if (baseNumber.getLevelName(3).find("LayerTiming") != std::string_view::npos) {
      const std::string_view& rodName(baseNumber.getLevelName(2));  // name of module volume
      uint32_t pos = rodName.find("Zpos");
      zside = (pos <= rodName.size() ? 1 : 0);

      // rod (tray), RU and module copy numbers
      // (everything start from 0)
      rodCopy = baseNumber.getCopyNumber(2) - 1;
      runitCopy = baseNumber.getCopyNumber(1) - 1;
      modCopy = baseNumber.getCopyNumber(0) - 1;

      // eval detector and sensor module numbers from global module number 1-24
      dmodCopy = int((modCopy / BTLDetId::kDModulesInRUCol) / BTLDetId::kSModulesInDM) +
                 (modCopy % BTLDetId::kDModulesInRUCol) * BTLDetId::kDModulesInRURow;
      smodCopy = int(modCopy / BTLDetId::kDModulesInRUCol) % BTLDetId::kSModulesInDM;

      if (0 > int(modCopy) || BTLDetId::kModulesPerRUV2 - 1 < modCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad module copy = " << int(modCopy)
                                   << ", Volume Number (counting from 0)= " << baseNumber.getCopyNumber(0) - 1;
        return 0;
      }

      if (0 > int(smodCopy) || BTLDetId::kSModulesPerDM - 1 < smodCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad detector module copy = " << int(smodCopy)
                                   << ", Volume Number (counting from 0)= " << baseNumber.getCopyNumber(0) - 1;
        return 0;
      }

      if (0 > int(dmodCopy) || BTLDetId::kDModulesPerRU - 1 < dmodCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad detector module copy = " << int(dmodCopy)
                                   << ", Volume Number (counting from 0)= " << baseNumber.getCopyNumber(0) - 1;
        return 0;
      }

      if (0 > int(runitCopy) || BTLDetId::kRUPerRod - 1 < runitCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad readout unit copy = " << int(runitCopy)
                                   << ", Volume Number (counting from 0)= " << baseNumber.getCopyNumber(1) - 1;
        return 0;
      }

      if (0 > int(rodCopy) || BTLDetId::HALF_ROD - 1 < rodCopy) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad rod copy = " << int(rodCopy)
                                   << ", Volume Number (counting from 0)= " << baseNumber.getCopyNumber(2);
        return 0;
      }

      if (1 < zside) {
        edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                                   << "****************** Bad side = " << zside
                                   << ", Volume Name = " << baseNumber.getLevelName(2);
        return 0;
      }

      // all inputs are fine. Go ahead and decode
      BTLDetId thisBTLdetid(zside, rodCopy, runitCopy, dmodCopy, smodCopy, 0);
      intindex = thisBTLdetid.geographicalId(BTLDetId::CrysLayout::v4).rawId();
    }

    else if (baseNumber.getLevelName(3).find("Layer1Timing") != std::string_view::npos) {
      // v2 or v3 scenario
      // zside copy number
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

      bool isV2(bareBaseName(baseNumber.getLevelName(0)).back() != 'e');

#ifdef EDM_ML_DEBUG
      LogDebug("MTDGeom") << "BTLNumberingScheme::getUnitID(): isV2 " << isV2;
#endif

      if (isV2) {
        // V2: the type is embedded in crystal name
        modType = ::atoi(&bareBaseName(baseNumber.getLevelName(1)).back());
      } else {
        // V3: build type and RU number per type from global RU number
        modType = globalru2type[runitCopy - 1];
        runitCopy = globalru2ru[runitCopy - 1];
      }

      // error checking

      if (1 > modType || 3 < modType) {
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
      BTLDetId thisBTLdetid(zside, rodCopy, runitCopy, modCopy, modType, 0, isV2);
      intindex = thisBTLdetid.geographicalId(BTLDetId::CrysLayout::v2).rawId();
    }
  } else {
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                               << "Not enough levels found in MTDBaseNumber ( " << nLevels
                               << ") or not correct path. Returning 0";
    return 0;
  }

  LogDebug("MTDGeom") << "BTL Numbering scheme: "
                      << " Raw Id = " << intindex << " zside = " << zside << " rod = " << rodCopy
                      << " runit = " << runitCopy << " dmodule = " << dmodCopy << " smodule = " << smodCopy
                      << " module = " << modCopy + 1 << " crystal = " << crystal << "\n"
                      << BTLDetId(intindex);

  return intindex;
}
