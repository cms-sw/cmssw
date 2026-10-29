#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/MTDObjects/interface/BTLElectronicsSpecs.h"
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

int BTLElectronicsMapping::hslinkFromRUandTray(uint32_t runit, uint32_t tray) const {
  if (int(runit) < 0 || runit >= BTLDetId::kRUPerRod) {
    edm::LogWarning("BTLElectronicsMapping") << "BTLElectronicsMapping::hslinkFromRUandTray "
                                             << "****************** runit = " << runit << "  not valid!";
    return -1;
  }

  if (int(tray) < 0 || tray >= BTLDetId::HALF_ROD) {
    edm::LogWarning("BTLElectronicsMapping") << "BTLElectronicsMapping::hslinkFromRUandTray: "
                                             << "****************** tray = " << tray << "  not valid!";
    return -1;
  }

  // Based on the proposal discussed here:
  // https://indico.cern.ch/event/1728068/contributions/7267163/attachments/3340485/5987766/LinkMapping_DAQmeeting_20260910.pdf
  return (runit + 6 * (tray % 6));
}

int BTLElectronicsMapping::hslink(BTLDetId det) const {
  uint32_t ru = det.runit();
  uint32_t tray = det.mtdRR();
  return BTLElectronicsMapping::hslinkFromRUandTray(ru, tray);
}

int BTLElectronicsMapping::hslink(uint32_t rawID) const {
  BTLDetId theId(rawID);
  return hslink(theId);
}

int BTLElectronicsMapping::slinkFromRUandTray(uint32_t runit, uint32_t tray, uint32_t zside) const {
  if (int(runit) < 0 || runit > BTLDetId::kRUPerRod) {
    edm::LogWarning("BTLElectronicsMapping") << "BTLElectronicsMapping::SlinkFromRUandTray "
                                             << "****************** runit = " << runit << "  not valid!";
    return -1;
  }

  if (int(tray) < 0 || tray >= BTLDetId::HALF_ROD) {
    edm::LogWarning("BTLElectronicsMapping") << "BTLElectronicsMapping::SlinkFromRUandTray: "
                                             << "****************** tray = " << tray << "  not valid!";
    return -1;
  }

  if (int(zside) < 0 || zside > 1) {
    edm::LogWarning("BTLElectronicsMapping")
        << "BTLElectronicsMapping::SlinkFromRUandTray "
        << "****************** zside = " << zside << "  not valid (should be 0 or 1)!";
    return -1;
  }

  //S-link id assignment (1 S-link = 1 FED id):
  // - each tray is read out by 2 S-links: RUs 0-2 --> first S-link (halfTray = 0), RUs 3-5 --> second S-link (halfTray = 1),
  // - S-links are numbered consecutively along the trays
  //trays [0-35], Z- --> Slinks [0,71]
  //trays [0-35], Z+ --> Slinks [72,143]
  int halfTray = (runit < 3) ? 0 : 1;
  return (BTLElectronicsSpecs::kNumberOfSlinks / 2 * zside + tray * 2 + halfTray);
}

int BTLElectronicsMapping::slink(BTLDetId det) const {
  uint32_t ru = det.runit();
  uint32_t tray = det.mtdRR();
  uint32_t zside = det.mtdSide();
  return BTLElectronicsMapping::slinkFromRUandTray(ru, tray, zside);
}

int BTLElectronicsMapping::slink(uint32_t rawID) const {
  BTLDetId theId(rawID);
  return BTLElectronicsMapping::slink(theId);
}

int BTLElectronicsMapping::fedFromRUandTray(uint32_t runit, uint32_t tray, uint32_t zside) const {
  return (BTLElectronicsSpecs::kFirstFEDId + BTLElectronicsMapping::slinkFromRUandTray(runit, tray, zside));
}

int BTLElectronicsMapping::fed(BTLDetId det) const {
  uint32_t ru = det.runit();
  uint32_t tray = det.mtdRR();
  uint32_t zside = det.mtdSide();
  return BTLElectronicsMapping::fedFromRUandTray(ru, tray, zside);
}

int BTLElectronicsMapping::fed(uint32_t rawID) const {
  BTLDetId theId(rawID);
  return BTLElectronicsMapping::fed(theId);
}
