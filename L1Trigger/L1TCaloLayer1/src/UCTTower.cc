#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#include "UCTTower.hh"
#include "UCTLogging.hh"

bool UCTTower::process() {
  if(region >= l1tcalo::NRegionsInCard) {
    return processHFTower();
  }
  if(ecalET > l1tcalo::etInputMax) ecalET = l1tcalo::etInputMax;
  if(hcalET > l1tcalo::etInputMax) hcalET = l1tcalo::etInputMax;
  uint32_t calibratedECALET = ecalET;
  uint32_t logECALET = (uint32_t) log2((double) ecalET);
  if(logECALET > l1tcalo::erMaxV) logECALET = l1tcalo::erMaxV;
  if(ecalLUT != 0) {
    uint32_t etaAddress = region * l1tcalo::NEtaInRegion + iEta;
    uint32_t fbAddress = 0;
    if(ecalFG) fbAddress = 1;
    uint32_t value = (*ecalLUT)[etaAddress][fbAddress][ecalET];
    calibratedECALET = value & l1tcalo::etInputMax;
    logECALET = (value & 0x7000) >> 12;
  }
  uint32_t calibratedHCALET = hcalET;
  uint32_t logHCALET = (uint32_t) log2((double) hcalET);
  if(logHCALET > l1tcalo::erMaxV) logHCALET = l1tcalo::erMaxV;
  if(hcalLUT != 0) {
    uint32_t etaAddress = region * l1tcalo::NEtaInRegion + iEta;
    uint32_t fbAddress = 0;
    if((hcalFB & 0x1) != 0) fbAddress = 1;
    uint32_t value = (*hcalLUT)[etaAddress][fbAddress][hcalET];
    calibratedHCALET = value & l1tcalo::etInputMax;
    logHCALET = (value & 0x7000) >> 12;
  }
  towerData = calibratedECALET + calibratedHCALET;
  if(towerData > l1tcalo::etMask) towerData = l1tcalo::etMask;
  uint32_t er = 0;
  if(ecalET == 0 || hcalET == 0) {
    er = 0;
    towerData |= l1tcalo::zeroFlagMask;
    if(hcalET == 0 && ecalET != 0)
      towerData |= l1tcalo::eohrFlagMask;
  }
  else if(ecalET == hcalET) {
    er = 0;
    towerData |= l1tcalo::eohrFlagMask;
  }
  else if(ecalET > hcalET) {
    er = logECALET - logHCALET;
    if(er > l1tcalo::erMaxV) er = l1tcalo::erMaxV;
    towerData |= l1tcalo::eohrFlagMask;
  }
  else {
    er = logHCALET - logECALET;
    if(er > l1tcalo::erMaxV) er = l1tcalo::erMaxV;
  }
  towerData |= (er << l1tcalo::erShift);
  // Unfortunately, hcalFlag is presently bogus :(
  // It has never been studied nor used in Run-1
  // The same status persists in Run-2, but it is available usage
  // Currently, summarize all hcalFeatureBits in one flag bit
  if((hcalFB & 0x1) != 0) towerData |= l1tcalo::hcalFlagMask; // FIXME - ignore top bits if(hcalFB != 0)
  if(ecalFG) towerData |= l1tcalo::ecalFlagMask;
  // Store ecal and hcal calibrated ET in unused upper bits
  towerData |= (calibratedECALET << l1tcalo::ecalShift);
  towerData |= (calibratedHCALET << l1tcalo::hcalShift);
  // All done!
  return true;
}

bool UCTTower::processHFTower() {
  if(hcalET > l1tcalo::etInputMax) hcalET = l1tcalo::etInputMax;
  if(hcalFB > 0x3) hcalFB = 0x3;
  uint32_t calibratedET = hcalET;
  if(hfLUT != 0) {
    const std::vector< uint32_t > a = hfLUT->at((region - l1tcalo::NRegionsInCard) * l1tcalo::NHFEtaInRegion + iEta);
    calibratedET = a[hcalET];
  }
  towerData = calibratedET + (hcalFB << l1tcalo::miscShift) + (location() << l1tcalo::ecalShift);
  return true;
}

bool UCTTower::setECALData(bool eFG, uint32_t eET) {
  ecalFG = eFG;
  ecalET = eET;
  if(eET > l1tcalo::etInputMax) {
    LOG_ERROR << "UCTTower::setData - ecalET too high " << eET << "; Pegged to l1tcalo::etInputMax" << std::endl;
    ecalET = l1tcalo::etInputMax;
  }
  return true;
}

bool UCTTower::setHCALData(uint32_t hFB, uint32_t hET) {
  hcalET = hET;
  hcalFB = hFB;
  if(hET > l1tcalo::etInputMax) {
    LOG_ERROR << "UCTTower::setData - ecalET too high " << hET << "; Pegged to l1tcalo::etInputMax" << std::endl;
    hcalET = l1tcalo::etInputMax;
  }
  if(hFB > 0x3F) {
    LOG_ERROR << "UCTTower::setData - too many hcalFeatureBits " << std::hex << hFB 
	      << "; Used only bottom 6 bits" << std::endl;
    hcalFB &= 0x3F;
  }
  return true;
}

bool UCTTower::setHFData(uint32_t fbIn, uint32_t etIn) {
  ecalFG = false; // HF has no separate ecal section
  ecalET = 0;
  hcalET = etIn; // We reuse HCAL place as HF
  hcalFB = fbIn;
  return true;
}

const uint16_t UCTTower::location() const {
  uint16_t l = 0;
  if(negativeEta) l = 0x8000; // Used top bit for +/- eta-side
  l |= iPhi;                  // Max iPhi is 4, so bottom 2 bits for iPhi
  l |= (iEta   << 2);         // Max iEta is 4, so 2 bits needed
  l |= (region << 4);         // Max region number 14, so 4 bits needed
  l |= (card   << 8);         // Max card number is 6, so 3 bits needed
  l |= (crate  << 11);        // Max crate number is 2, so 2 bits needed
  return l;
}

UCTTower::UCTTower(uint16_t location) {
  if((location & 0x8000) != 0) negativeEta = true;
  crate =  (location & 0x1800) >> 11;
  card =   (location & 0x0700) >>  8;
  region = (location & 0x00F0) >>  4;
  iEta =   (location & 0x000C) >>  2;
  iPhi =   (location & 0x0003);
  towerData = 0;
}

const uint64_t UCTTower::extendedData() const {
  uint64_t d = rawData();
  uint64_t l = location();
  uint64_t r = (l << 48) + d;
  return r;
}

std::ostream& operator<<(std::ostream& os, const UCTTower& t) {
  //  if((t.ecalET + t.hcalET) == 0) return os;
  
  os << "Side Crt  Crd  Rgn  iEta iPhi cEta cPhi eET  eFG  hET  hFB  Summary" << std::endl;

  UCTGeometry g;
  std::string side = "+eta ";
  if(t.negativeEta) side = "-eta ";
  os << side
     << std::showbase << std::internal << std::setfill('0') << std::setw(4) << std::hex
     << t.crate << " "
     << std::showbase << std::internal << std::setfill('0') << std::setw(4) << std::hex
     << t.card << " "
     << std::showbase << std::internal << std::setfill('0') << std::setw(4) << std::hex
     << t.region << " "
     << std::showbase << std::internal << std::setfill('0') << std::setw(4) << std::hex
     << t.iEta << " "
     << std::showbase << std::internal << std::setfill('0') << std::setw(4) << std::hex
     << t.iPhi << " "
     << std::setw(4) << std::setfill(' ') << std::dec
     << g.getCaloEtaIndex(t.negativeEta, t.region, t.iEta) << " "
     << std::setw(4) << std::setfill(' ') << std::dec
     << g.getCaloPhiIndex(t.crate, t.card, t.region, t.iPhi) << " "
     << std::showbase << std::internal << std::setfill('0') << std::setw(4) << std::hex
     << t.ecalET << " "
     << std::showbase << std::internal << std::setfill('0') << std::setw(4) << std::hex
     << t.ecalFG << " "
     << std::showbase << std::internal << std::setfill('0') << std::setw(4) << std::hex
     << t.hcalET << " "
     << std::showbase << std::internal << std::setfill('0') << std::setw(4) << std::hex
     << t.hcalFB << " "
     << std::showbase << std::internal << std::setfill('0') << std::setw(10) << std::hex
     << t.towerData
     << std::endl;
  return os;

}
