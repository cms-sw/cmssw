#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/MTDRawToDigi/interface/BTLElectronicsSpecs.h"
#include "EventFilter/MTDRawToDigi/interface/BTLElectronicsMapping.h"
#include <Geometry/MTDCommonData/interface/MTDTopologyMode.h>

#include <stdexcept>

BTLElectronicsMapping::BTLElectronicsMapping() {}

int BTLElectronicsMapping::SiPMCh(uint32_t smodCopy, uint32_t crystal, uint32_t SiPMSide) const {
  if (0 > int(crystal) || crystal > BTLDetId::kCrystalsPerModuleV2) {
    edm::LogWarning("BTLElectronicsMapping") << "BTLElectronicsMapping::SiPMCh "
                                             << "****************** Bad crystal number = " << int(crystal);
    return 0;
  }

  if (0 > int(smodCopy) || smodCopy > BTLDetId::kSModulesPerDM) {
    edm::LogWarning("BTLElectronicsMapping") << "BTLNumberingScheme::getUnitID(): "
                                             << "****************** Bad detector module copy = " << int(smodCopy);
    return 0;
  }

  if (smodCopy == 0)
    return BTLElectronicsSpecs::SiPMChannelMapFW[crystal + SiPMSide * BTLDetId::kCrystalsPerModuleV2];
  else
    return BTLElectronicsSpecs::SiPMChannelMapBW[crystal + SiPMSide * BTLDetId::kCrystalsPerModuleV2];
}

int BTLElectronicsMapping::SiPMCh(BTLDetId det, uint32_t SiPMSide) const {
  uint32_t smodCopy = det.smodule();
  uint32_t crystal = det.crystal();
  return SiPMCh(smodCopy, crystal, SiPMSide);
}

int BTLElectronicsMapping::SiPMCh(uint32_t rawId, uint32_t SiPMSide) const {
  BTLDetId theId(rawId);
  return SiPMCh(theId, SiPMSide);
}

// -- Get TOFHIR Channel Id from crystal Id
int BTLElectronicsMapping::TOFHIRCh(uint32_t smodCopy, uint32_t crystal, uint32_t SiPMSide) const {
  int SiPMCh_ = BTLElectronicsMapping::SiPMCh(smodCopy, crystal, SiPMSide);
  return BTLElectronicsSpecs::THChannelMap[SiPMCh_];
}

int BTLElectronicsMapping::TOFHIRCh(BTLDetId det, uint32_t SiPMSide) const {
  uint32_t smodCopy = det.smodule();
  uint32_t crystal = det.crystal();
  return BTLElectronicsMapping::TOFHIRCh(smodCopy, crystal, SiPMSide);
}

int BTLElectronicsMapping::TOFHIRCh(uint32_t rawId, uint32_t SiPMSide) const {
  BTLDetId theId(rawId);
  return BTLElectronicsMapping::TOFHIRCh(theId, SiPMSide);
}

// -- Get TOFHIR asic number
// if dmodule is odd number (DM range [1-12])
//    SM1 --> TOFHIR A0 (simply 0)
//    SM2 --> TOFHIR A1 (simply 1)
// else if dmodule is even number the order is inverted
//    SM1 --> TOFHIR A1 (simply 1)
//    SM2 --> TOFHIR A0 (simply 0)
int BTLElectronicsMapping::TOFHIRASIC(uint32_t dmodule, uint32_t smodCopy) const {
  if (dmodule % BTLDetId::kSModulesInDM == 0)
    return smodCopy;
  else
    return BTLDetId::kSModulesInDM - smodCopy - 1;
}

int BTLElectronicsMapping::TOFHIRASIC(BTLDetId det) const {
  uint32_t dmodule = det.dmodule();
  uint32_t smodCopy = det.smodule();
  return BTLElectronicsMapping::TOFHIRASIC(dmodule, smodCopy);
}

int BTLElectronicsMapping::TOFHIRASIC(uint32_t rawID) const {
  BTLDetId theId(rawID);
  return BTLElectronicsMapping::TOFHIRASIC(theId);
}

// -- Get e-link from a given DM,SM (TOFHIR)
int BTLElectronicsMapping::elinkFromSM(uint32_t dmodule, uint32_t smodCopy, int lpgbt_id) const {
  if (int(dmodule) < 0 || dmodule > BTLDetId::kDModulesPerRU) {
    edm::LogWarning("BTLElectronicsMapping") << "BTLElectronicsMapping::elinkFromSM: "
                                             << "****************** dmodule = " << dmodule << "  not valid!";
    return -1;
  }

  if (int(smodCopy) < 0 || smodCopy > BTLDetId::kSModulesPerDM) {
    edm::LogWarning("BTLElectronicsMapping") << "BTLElectronicsMapping::elinkFromSM: "
                                             << "****************** smodCopy = " << smodCopy << "  not valid!";
    return -1;
  }

  if (lpgbt_id < 0 || lpgbt_id > 1) {
    edm::LogWarning("BTLElectronicsMapping") << "BTLElectronicsMapping::hslinkFromRU: "
                                             << "****************** lpgbt_id = " << lpgbt_id << "  not valid!";
    return -1;
  }

  int chipId = TOFHIRASIC(dmodule, smodCopy);
  return (lpgbt_id == 0) ? BTLElectronicsSpecs::FE_to_ELINK_mapping_L0[dmodule][chipId]
                         : BTLElectronicsSpecs::FE_to_ELINK_mapping_L1[dmodule][chipId];
}

int BTLElectronicsMapping::elink(BTLDetId det, int lpgbt_id) const {
  uint32_t dmodule = det.dmodule();
  uint32_t smodCopy = det.smodule();
  return BTLElectronicsMapping::elinkFromSM(dmodule, smodCopy, lpgbt_id);
}

int BTLElectronicsMapping::elink(uint32_t rawID, int lpgbt_id) const {
  BTLDetId theId(rawID);
  return BTLElectronicsMapping::elink(theId, lpgbt_id);
}

// -- Get HS-link Id from RU and Tray
// TEMPORARY MAPPING: within a group of 6 trays ( = supertray): tray 0 --> first block of 12 links, tray 1--> second block of 12 links, etc.

int BTLElectronicsMapping::opticalTxPosition(uint32_t tray, int optTxCh) const {
  const bool useN5Mapping = (BTLElectronicsSpecs::kHSLinksOffset == 4) && (tray % 6 == 0);

  return useN5Mapping ? BTLElectronicsSpecs::tx_inv_n5[optTxCh] : BTLElectronicsSpecs::tx_inv_common[optTxCh];
}

int BTLElectronicsMapping::hslinkFromRU(uint32_t runit, uint32_t tray, int lpgbt_id) const {
  if (int(runit) < 0 || runit > BTLDetId::kRUPerRod) {
    edm::LogWarning("BTLElectronicsMapping") << "BTLElectronicsMapping::hslinkFromRU "
                                             << "****************** runit = " << runit << "  not valid!";
    return -1;
  }

  if (lpgbt_id < 0 || lpgbt_id > 1) {
    edm::LogWarning("BTLElectronicsMapping") << "BTLElectronicsMapping::hslinkFromRU: "
                                             << "****************** lpgbt_id = " << lpgbt_id << "  not valid!";
    return -1;
  }

  const int optTxCh = 2 * runit + lpgbt_id;

  // pos is the position in the opt tx array
  const int pos = opticalTxPosition(tray, optTxCh);

  return (BTLElectronicsSpecs::kHSLinksOffset + 12 * (tray % 6) + pos);
}

int BTLElectronicsMapping::hslink(BTLDetId det, int lpgbt_id) const {
  uint32_t ru = det.runit();
  uint32_t tray = det.mtdRR();
  return BTLElectronicsMapping::hslinkFromRU(ru, tray, lpgbt_id);
}

int BTLElectronicsMapping::hslink(uint32_t rawID, int lpgbt_id) const {
  BTLDetId theId(rawID);
  return hslink(theId, lpgbt_id);
}

int BTLElectronicsMapping::slinkFromTray(uint32_t tray, uint32_t zside) const {
  if (int(tray) < 0 || tray >= BTLDetId::HALF_ROD) {
    edm::LogWarning("BTLElectronicsMapping") << "BTLElectronicsMapping::SlinkFromTray: "
                                             << "****************** tray = " << tray << "  not valid!";
    return -1;
  }

  if (int(zside) < 0 || zside > 1) {
    edm::LogWarning("BTLElectronicsMapping")
        << "BTLElectronicsMapping::SlinkFromTray "
        << "****************** zside = " << zside << "  not valid (should be 0 or 1)!";
    return -1;
  }
  // TEMPORARY MAPPING:
  // trays [0-35], Z- --> Slinks [0,5]
  // trays [0-35], Z+ --> Slinks [6,11]
  return (BTLElectronicsSpecs::kFirstFEDId + tray / 6 + 6 * zside);
}

int BTLElectronicsMapping::slink(BTLDetId det) const {
  uint32_t tray = det.mtdRR();
  uint32_t zside = det.mtdSide();
  return BTLElectronicsMapping::slinkFromTray(tray, zside);
}

int BTLElectronicsMapping::slink(uint32_t rawID) const {
  BTLDetId theId(rawID);
  return BTLElectronicsMapping::slink(theId);
}
