#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/MTDCommonData/interface/BTLElectronicsMapping.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <ostream>
#include <algorithm>
#include <vector>
BTLElectronicsMapping::BTLElectronicsMapping() {}

// Get SiPM Channel from crystal ID

int BTLElectronicsMapping::SiPMCh(uint32_t smodCopy, uint32_t crystal, uint32_t SiPMSide) {
  if (0 > int(crystal) || crystal > BTLDetId::kCrystalsPerModuleV2) {
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::BTLElectronicsMapping(): "
                               << "****************** Bad crystal number = " << int(crystal);
    return 0;
  }

  if (0 > int(smodCopy) || smodCopy > BTLDetId::kSModulesPerDM) {
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                               << "****************** Bad detector module copy = " << int(smodCopy);
    return 0;
  }

  if (smodCopy == 0)
    return BTLElectronicsMapping::SiPMChannelMapFW[crystal + SiPMSide * BTLDetId::kCrystalsPerModuleV2];
  else
    return BTLElectronicsMapping::SiPMChannelMapBW[crystal + SiPMSide * BTLDetId::kCrystalsPerModuleV2];
}

int BTLElectronicsMapping::SiPMCh(BTLDetId det, uint32_t SiPMSide) {
  uint32_t smodCopy = det.smodule();
  uint32_t crystal = det.crystal();

  return BTLElectronicsMapping::SiPMCh(smodCopy, crystal, SiPMSide);
}

int BTLElectronicsMapping::SiPMCh(uint32_t rawId, uint32_t SiPMSide) {
  BTLDetId theId(rawId);
  return BTLElectronicsMapping::SiPMCh(theId, SiPMSide);
}

BTLElectronicsMapping::SiPMChPair BTLElectronicsMapping::GetSiPMChPair(uint32_t smodCopy, uint32_t crystal) {
  BTLElectronicsMapping::SiPMChPair SiPMChs;
  SiPMChs.Minus = BTLElectronicsMapping::SiPMCh(smodCopy, crystal, 0);
  SiPMChs.Plus = BTLElectronicsMapping::SiPMCh(smodCopy, crystal, 1);
  return SiPMChs;
}

BTLElectronicsMapping::SiPMChPair BTLElectronicsMapping::GetSiPMChPair(BTLDetId det) {
  BTLElectronicsMapping::SiPMChPair SiPMChs;
  SiPMChs.Minus = BTLElectronicsMapping::SiPMCh(det, 0);
  SiPMChs.Plus = BTLElectronicsMapping::SiPMCh(det, 1);
  return SiPMChs;
}
BTLElectronicsMapping::SiPMChPair BTLElectronicsMapping::GetSiPMChPair(uint32_t rawID) {
  BTLElectronicsMapping::SiPMChPair SiPMChs;
  SiPMChs.Minus = BTLElectronicsMapping::SiPMCh(rawID, 0);
  SiPMChs.Plus = BTLElectronicsMapping::SiPMCh(rawID, 1);
  return SiPMChs;
}

// Get TOFHIR Channel from crystal ID

int BTLElectronicsMapping::TOFHIRCh(uint32_t smodCopy, uint32_t crystal, uint32_t SiPMSide) {
  int SiPMCh_ = BTLElectronicsMapping::SiPMCh(smodCopy, crystal, SiPMSide);
  return BTLElectronicsMapping::THChannelMap[SiPMCh_];
}

int BTLElectronicsMapping::TOFHIRCh(BTLDetId det, uint32_t SiPMSide) {
  uint32_t smodCopy = det.smodule();
  uint32_t crystal = det.crystal();

  return BTLElectronicsMapping::TOFHIRCh(smodCopy, crystal, SiPMSide);
}

int BTLElectronicsMapping::TOFHIRCh(uint32_t rawId, uint32_t SiPMSide) {
  BTLDetId theId(rawId);
  return BTLElectronicsMapping::TOFHIRCh(theId, SiPMSide);
}

BTLElectronicsMapping::TOFHIRChPair BTLElectronicsMapping::GetTOFHIRChPair(uint32_t smodCopy, uint32_t crystal) {
  BTLElectronicsMapping::TOFHIRChPair TOFHIRChs;
  TOFHIRChs.Minus = BTLElectronicsMapping::TOFHIRCh(smodCopy, crystal, 0);
  TOFHIRChs.Plus = BTLElectronicsMapping::TOFHIRCh(smodCopy, crystal, 1);
  return TOFHIRChs;
}

BTLElectronicsMapping::TOFHIRChPair BTLElectronicsMapping::GetTOFHIRChPair(BTLDetId det) {
  BTLElectronicsMapping::TOFHIRChPair TOFHIRChs;
  TOFHIRChs.Minus = BTLElectronicsMapping::TOFHIRCh(det, 0);
  TOFHIRChs.Plus = BTLElectronicsMapping::TOFHIRCh(det, 1);
  return TOFHIRChs;
}
BTLElectronicsMapping::TOFHIRChPair BTLElectronicsMapping::GetTOFHIRChPair(uint32_t rawID) {
  BTLElectronicsMapping::TOFHIRChPair TOFHIRChs;
  TOFHIRChs.Minus = BTLElectronicsMapping::TOFHIRCh(rawID, 0);
  TOFHIRChs.Plus = BTLElectronicsMapping::TOFHIRCh(rawID, 1);
  return TOFHIRChs;
}

// Get crystal ID from TOFHIR Channel

int BTLElectronicsMapping::THChToXtal(uint32_t smodCopy, uint32_t THCh) {
  if (0 > int(smodCopy) || BTLDetId::kSModulesPerDM < smodCopy) {
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                               << "****************** Bad detector module copy = " << int(smodCopy);
    return 0;
  }

  auto THChPos =
      std::find(BTLElectronicsMapping::THChannelMap.begin(), BTLElectronicsMapping::THChannelMap.end(), THCh);
  int targetSiPMCh = std::distance(BTLElectronicsMapping::THChannelMap.begin(), THChPos);

  std::array<uint32_t, BTLDetId::kCrystalsPerModuleV2 * 2> SiPMChMap;
  if (smodCopy == 0)
    SiPMChMap = BTLElectronicsMapping::SiPMChannelMapFW;
  else
    SiPMChMap = BTLElectronicsMapping::SiPMChannelMapBW;

  auto targetpos = std::find(SiPMChMap.begin(), SiPMChMap.end(), targetSiPMCh);
  return std::distance(SiPMChMap.begin(), targetpos) % BTLDetId::kCrystalsPerModuleV2 + 1;
}

BTLDetId BTLElectronicsMapping::THChToBTLDetId(
    uint32_t zside, uint32_t rod, uint32_t runit, uint32_t dmodule, uint32_t smodCopy, uint32_t THCh) {
  if (0 > int(THCh) || 31 < THCh) {
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                               << "****************** Bad TOFHIR channel = " << int(THCh);
    return 0;
  }

  if (0 > int(smodCopy) || BTLDetId::kSModulesPerDM < smodCopy) {
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                               << "****************** Bad detector module copy = " << int(smodCopy);
    return 0;
  }

  if (0 > int(dmodule) || 12 < dmodule) {
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                               << "****************** Bad module copy = " << int(dmodule);
    return 0;
  }

  if (1 > rod || 36 < rod) {
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                               << "****************** Bad rod copy = " << rod;
    return 0;
  }

  if (1 < zside) {
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::getUnitID(): "
                               << "****************** Bad side = " << zside;
    return 0;
  }

  int crystal = BTLElectronicsMapping::THChToXtal(smodCopy, THCh);

  return BTLDetId(zside, rod, runit, dmodule, smodCopy, crystal);
}

// Get TOFHIR asic number
// if dmodule is odd number (DM range [1-12]) 
//    SM1 --> TOFHIR A0 (simply 0)
//    SM2 --> TOFHIR A1 (simply 1)
// else if dmodule is even number the order is inverted
//    SM1 --> TOFHIR A1 (simply 1)
//    SM2 --> TOFHIR A0 (simply 0)
int BTLElectronicsMapping::TOFHIRASIC(uint32_t dmodule, uint32_t smodCopy) { 
    if (dmodule % BTLDetId::kSModulesInDM == 0) return smodCopy;
    else return BTLDetId::kSModulesInDM - smodCopy - 1;
  }

int BTLElectronicsMapping::TOFHIRASIC(BTLDetId det) {
  uint32_t dmodule = det.dmodule();
  uint32_t smodCopy = det.smodule();
  return BTLElectronicsMapping::TOFHIRASIC(dmodule, smodCopy);
}

int BTLElectronicsMapping::TOFHIRASIC(uint32_t rawID) {
  BTLDetId theId(rawID);
  return BTLElectronicsMapping::TOFHIRASIC(theId);
}


/** Returns FE board number */
int BTLElectronicsMapping::FEBoardFromDM(uint32_t dmodule) { return dmodule; }

int BTLElectronicsMapping::FEBoard(BTLDetId det) {
  uint32_t dmodule = det.dmodule();
  return BTLElectronicsMapping::FEBoardFromDM(dmodule);
}

int BTLElectronicsMapping::FEBoard(uint32_t rawID) {
  BTLDetId theId(rawID);
  return BTLElectronicsMapping::FEBoard(theId);
}

