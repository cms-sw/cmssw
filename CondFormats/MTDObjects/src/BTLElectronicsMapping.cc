#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "Geometry/MTDCommonData/interface/BTLElectronicsMapping.h"
#include "CondFormats/MTDObjects/interface/BTLElectronicsMapping.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <Geometry/MTDCommonData/interface/MTDTopologyMode.h>

#include <ostream>
#include <algorithm>
#include <vector>
BTLElectronicsMapping::BTLElectronicsMapping(const BTLDetId::CrysLayout lay) {
  if (static_cast<int>(lay) < 7) {
    throw cms::Exception("BTLElectronicsMapping")
        << "MTD Topology mode with layout " << static_cast<int>(lay) << " is not supported\n"
        << "use layout : 7 (v4) or later!" << std::endl;
  }
}

// Get SiPM Channel from crystal ID

int BTLElectronicsMapping::SiPMCh(uint32_t smodCopy, uint32_t crystal, uint32_t SiPMSide) const{
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

int BTLElectronicsMapping::SiPMCh(BTLDetId det, uint32_t SiPMSide) const {
  uint32_t smodCopy = det.smodule();
  uint32_t crystal = det.crystal();

  return BTLElectronicsMapping::SiPMCh(smodCopy, crystal, SiPMSide);
}

int BTLElectronicsMapping::SiPMCh(uint32_t rawId, uint32_t SiPMSide) const {
  BTLDetId theId(rawId);
  return BTLElectronicsMapping::SiPMCh(theId, SiPMSide);
}

BTLElectronicsMapping::SiPMChPair BTLElectronicsMapping::GetSiPMChPair(uint32_t smodCopy, uint32_t crystal) const {
  BTLElectronicsMapping::SiPMChPair SiPMChs;
  SiPMChs.Minus = BTLElectronicsMapping::SiPMCh(smodCopy, crystal, 0);
  SiPMChs.Plus = BTLElectronicsMapping::SiPMCh(smodCopy, crystal, 1);
  return SiPMChs;
}

BTLElectronicsMapping::SiPMChPair BTLElectronicsMapping::GetSiPMChPair(BTLDetId det) const {
  BTLElectronicsMapping::SiPMChPair SiPMChs;
  SiPMChs.Minus = BTLElectronicsMapping::SiPMCh(det, 0);
  SiPMChs.Plus = BTLElectronicsMapping::SiPMCh(det, 1);
  return SiPMChs;
}
BTLElectronicsMapping::SiPMChPair BTLElectronicsMapping::GetSiPMChPair(uint32_t rawID) const {
  BTLElectronicsMapping::SiPMChPair SiPMChs;
  SiPMChs.Minus = BTLElectronicsMapping::SiPMCh(rawID, 0);
  SiPMChs.Plus = BTLElectronicsMapping::SiPMCh(rawID, 1);
  return SiPMChs;
}

// Get TOFHIR Channel from crystal ID

int BTLElectronicsMapping::TOFHIRCh(uint32_t smodCopy, uint32_t crystal, uint32_t SiPMSide) const {
  int SiPMCh_ = BTLElectronicsMapping::SiPMCh(smodCopy, crystal, SiPMSide);
  return BTLElectronicsMapping::THChannelMap[SiPMCh_];
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

BTLElectronicsMapping::TOFHIRChPair BTLElectronicsMapping::GetTOFHIRChPair(uint32_t smodCopy, uint32_t crystal) const {
  BTLElectronicsMapping::TOFHIRChPair TOFHIRChs;
  TOFHIRChs.Minus = BTLElectronicsMapping::TOFHIRCh(smodCopy, crystal, 0);
  TOFHIRChs.Plus = BTLElectronicsMapping::TOFHIRCh(smodCopy, crystal, 1);
  return TOFHIRChs;
}

BTLElectronicsMapping::TOFHIRChPair BTLElectronicsMapping::GetTOFHIRChPair(BTLDetId det) const {
  BTLElectronicsMapping::TOFHIRChPair TOFHIRChs;
  TOFHIRChs.Minus = BTLElectronicsMapping::TOFHIRCh(det, 0);
  TOFHIRChs.Plus = BTLElectronicsMapping::TOFHIRCh(det, 1);
  return TOFHIRChs;
}
BTLElectronicsMapping::TOFHIRChPair BTLElectronicsMapping::GetTOFHIRChPair(uint32_t rawID) const {
  BTLElectronicsMapping::TOFHIRChPair TOFHIRChs;
  TOFHIRChs.Minus = BTLElectronicsMapping::TOFHIRCh(rawID, 0);
  TOFHIRChs.Plus = BTLElectronicsMapping::TOFHIRCh(rawID, 1);
  return TOFHIRChs;
}

// Get crystal ID from TOFHIR Channel

int BTLElectronicsMapping::THChToXtal(uint32_t smodCopy, uint32_t THCh) const {
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
    uint32_t zside, uint32_t rod, uint32_t runit, uint32_t dmodule, uint32_t smodCopy, uint32_t THCh) const {
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

/** Returns FE board number */
int BTLElectronicsMapping::FEBoardFromDM(uint32_t dmodule) const { return dmodule; }

int BTLElectronicsMapping::FEBoard(BTLDetId det) const {
  uint32_t dmodule = det.dmodule();
  return BTLElectronicsMapping::FEBoardFromDM(dmodule);
}

int BTLElectronicsMapping::FEBoard(uint32_t rawID) const {
  BTLDetId theId(rawID);
  return BTLElectronicsMapping::FEBoard(theId);
}

/** Returns CC board number */
int BTLElectronicsMapping::CCBoardFromRU(uint32_t runit) const { return runit; }

int BTLElectronicsMapping::CCBoard(BTLDetId det) const {
  uint32_t runit = det.runit();
  return BTLElectronicsMapping::CCBoardFromRU(runit);
}

int BTLElectronicsMapping::CCBoard(uint32_t rawID) const {
  BTLDetId theId(rawID);
  return BTLElectronicsMapping::CCBoard(theId);
}



/** TOFHIR/SM <-> e-link mapping **/
int BTLElectronicsMapping::elinkFromSM(uint32_t dmodule, uint32_t smodCopy, int lpgbt_id) const {

  if (int(dmodule) <  0 || dmodule > BTLDetId::kDModulesPerRU){
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::elinkFromSM: "
                               << "****************** dmodule = " << dmodule
                               << "  not valid!";
    return -1;
  }
  
  if (int(smodCopy) < 0 || smodCopy > BTLDetId::kSModulesPerDM){
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::elinkFromSM: "
                               << "****************** smodCopy = " << smodCopy
                               << "  not valid!";
    return -1;
  }

  if (lpgbt_id < 0 || lpgbt_id > 1){
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::hslinkFromRU: "
                               << "****************** lpgbt_id = " << lpgbt_id
                               << "  not valid!";
    return -1;
  }

  // from DM and chipId of the SM --> get elink
  int chipId = TOFHIRASIC(dmodule, smodCopy); 
  return (lpgbt_id == 0) ? BTLElectronicsMapping::FE_to_ELINK_mapping_L0[dmodule][chipId] : BTLElectronicsMapping::FE_to_ELINK_mapping_L1[dmodule][chipId];
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

std::pair<int, int> BTLElectronicsMapping::elinkToSM(int elink, int lpgbt_id) const {
  
  if (elink < 0 || elink > int(BTLDetId::kModulesPerRUV2)){
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::elinkToSM: "
                               << "****************** elink = " << elink
                               << "  not valid!";
    return std::make_pair(-1,-1);
  }

  if (lpgbt_id < 0 || lpgbt_id > 1){
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::elinkToSM: "
                               << "****************** lpgbt_id = " << lpgbt_id
                               << "  not valid!";
    return std::make_pair(-1,-1);
  }

  const auto& mapping = (lpgbt_id == 0) ? BTLElectronicsMapping::ELINK_to_FE_mapping_L0 : BTLElectronicsMapping::ELINK_to_FE_mapping_L1;
  int dm     = mapping[elink].first;
  int chipId = mapping[elink].second;
  int smCopy = chipId;
  if (dm % BTLDetId::kSModulesInDM != 0){
    smCopy = BTLDetId::kSModulesInDM - chipId - 1;
  }
  return ( std::make_pair(dm, smCopy));
}



int BTLElectronicsMapping::hslinkToRU(int hslink) const {
  return ( OptTx_map[hslink]/2 );  
}

int BTLElectronicsMapping::hslinkFromRU(uint32_t runit, uint32_t tray, int lpgbt_id) const {

  if (int (runit) < 0 || runit > BTLDetId::kRUPerRod){
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::hslinkFromRU "
                               << "****************** runit = " << runit
                               << "  not valid!";
    return -1;
  }

  if (lpgbt_id < 0 || lpgbt_id > 1){
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::hslinkFromRU: "
                               << "****************** lpgbt_id = " << lpgbt_id
                               << "  not valid!";
    return -1;
  }
  
  const int optTxCh = 2 * runit + lpgbt_id; 
  // TEMPORARY MAPPING: within a group of 6 trays ( = supertray): tray 0 --> first block of 12 links, tray 1--> second block of 12 links, etc.
  // For the first block (N5), channel ids of the optical tx are reversed.
  // pos is the position in the tx array
  const int pos = (tray%6 == 0) ? tx_inv_n5[optTxCh] : tx_inv_common[optTxCh];

  return ( kOffsetHSLinks + 12 * (tray % 6) + pos );
}


int BTLElectronicsMapping::hslink(BTLDetId det, int lpgbt_id) const {
  uint32_t ru = det.runit();
  uint32_t tray = det.mtdRR();
  return BTLElectronicsMapping::hslinkFromRU(ru, tray, lpgbt_id);
}

int BTLElectronicsMapping::hslink(uint32_t rawID, int lpgbt_id) const {
  BTLDetId theId(rawID);
  return BTLElectronicsMapping::hslink(theId, lpgbt_id);
}

int BTLElectronicsMapping::SlinkFromTray(uint32_t tray, uint32_t zside) const {
  
  if ( int(tray) < 0 || tray >= BTLDetId::HALF_ROD){
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::SlinkFromTray: "
                               << "****************** tray = " << tray
                               << "  not valid!";
    return -1;
  }

  if (int(zside) < 0 || zside >= 1){
    edm::LogWarning("MTDGeom") << "BTLNumberingScheme::SlinkFromTray "
                               << "****************** zside = " << zside
                               << "  not valid (should be 0 or 1)!";
    return -1;
  }
  
  // TEMPORARY MAPPING:
  // trays [0-35], Z- --> Slinks [0,5]
  // trays [0-35], Z+ --> Slinks [6,11]
  return (MIN_SLINK_ID + tray/6 + 6 * zside); 
}

int BTLElectronicsMapping::Slink(BTLDetId det) const {
  uint32_t tray = det.mtdRR();
  uint32_t zside = det.mtdSide();
  return BTLElectronicsMapping::SlinkFromTray(tray, zside);
}

int BTLElectronicsMapping::Slink(uint32_t rawID) const {
  BTLDetId theId(rawID);
  return BTLElectronicsMapping::Slink(theId);
}


std::pair<uint32_t, uint32_t> BTLElectronicsMapping::getTrayFromLinks(int slink, int hslink) const{

  int superTray = slink - MIN_SLINK_ID;

  uint32_t zside = (superTray < 6 ) ? 0 : 1;
  
  int hslinkBlock = int(hslink - kOffsetHSLinks)/12;
  
  uint32_t tray = (superTray - 6 * zside) + 6 * hslinkBlock;

  return( std::make_pair(tray, zside));
}
