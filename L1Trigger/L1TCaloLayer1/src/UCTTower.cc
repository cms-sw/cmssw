#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#include "UCTTower.hh"
#include "UCTLogging.hh"

using namespace l1tcalo;

bool UCTTower::process() {
  if(region >= NRegionsInCard) {
    return processHFTower();
  }
  if(ecalET > etInputMax) ecalET = etInputMax;
  if(hcalET > etInputMax) hcalET = etInputMax;
  uint32_t calibratedECALET = ecalET;
  uint32_t logECALET = (uint32_t) log2((double) ecalET);
  if(logECALET > erMaxV) logECALET = erMaxV;
  if(ecalLUT != 0) {
    uint32_t etaAddress = region * NEtaInRegion + iEta;
    uint32_t fbAddress = 0;
    if(ecalFG) fbAddress = 1;
    uint32_t value = (*ecalLUT)[etaAddress][fbAddress][ecalET];
    calibratedECALET = value & etInputMax;
    logECALET = (value & 0x7000) >> 12;
  }
  uint32_t calibratedHCALET = hcalET;
  uint32_t logHCALET = (uint32_t) log2((double) hcalET);
  if(logHCALET > erMaxV) logHCALET = erMaxV;
  if(hcalLUT != 0) {
    uint32_t etaAddress = region * NEtaInRegion + iEta;
    uint32_t fbAddress = 0;
    if((hcalFB & 0x1) != 0) fbAddress = 1;
    uint32_t value = (*hcalLUT)[etaAddress][fbAddress][hcalET];
    calibratedHCALET = value & etInputMax;
    logHCALET = (value & 0x7000) >> 12;
  }
  towerData = calibratedECALET + calibratedHCALET;
  if(towerData > etMask) towerData = etMask;
  uint32_t er = 0;
  if(calibratedECALET == 0 || calibratedHCALET == 0) {
    er = 0;
    towerData |= zeroFlagMask;
    if(calibratedHCALET == 0 && calibratedECALET != 0)
      towerData |= eohrFlagMask;
  }
  else if(calibratedECALET == calibratedHCALET) {
    er = 0;
    towerData |= eohrFlagMask;
  }
  else if(calibratedECALET > calibratedHCALET) {
    er = logECALET - logHCALET;
    if(er > erMaxV) er = erMaxV;
    towerData |= eohrFlagMask;
  }
  else {
    er = logHCALET - logECALET;
    if(er > erMaxV) er = erMaxV;
  }
  towerData |= (er << erShift);
  // Unfortunately, hcalFlag is presently bogus :(
  // It has never been studied nor used in Run-1
  // The same status persists in Run-2, but it is available usage
  // Currently, summarize all hcalFeatureBits in one flag bit
  if((hcalFB & 0x1) != 0) towerData |= hcalFlagMask; // FIXME - ignore top bits if(hcalFB != 0)
  if(ecalFG) towerData |= ecalFlagMask;
  // Store ecal and hcal calibrated ET in unused upper bits
  towerData |= (calibratedECALET << ecalShift);
  towerData |= (calibratedHCALET << hcalShift);
  // All done!
  return true;
}

bool UCTTower::processHFTower() {
  uint32_t calibratedET = hcalET;
  if(hfLUT != 0) {
    const std::vector< uint32_t > a = hfLUT->at((region - NRegionsInCard) * NHFEtaInRegion + iEta);
    calibratedET = a[hcalET] & 0xFF;
  }
  uint32_t absCaloEta = abs(caloEta());
  if(absCaloEta > 29 && absCaloEta < 40) {
    // Divide by two (since two duplicate towers are sent)
    calibratedET /= 2;
  }
  else if(absCaloEta == 40 || absCaloEta == 41) {
    // Divide by four
    calibratedET /= 4;
  }
  towerData = calibratedET | zeroFlagMask;
  if((hcalFB & 0x1) == 0x1) towerData |= ecalFlagMask; // LSB defines short over long fiber ratio
  if((hcalFB & 0x2) == 0x2) towerData |= hcalFlagMask; // MSB defines minbias flag
  return true;
}

bool UCTTower::setECALData(bool eFG, uint32_t eET) {
  ecalFG = eFG;
  ecalET = eET;
  if(eET > etInputMax) {
    LOG_ERROR << "UCTTower::setData - ecalET too high " << eET << "; Pegged to etInputMax" << std::endl;
    ecalET = etInputMax;
  }
  return true;
}

bool UCTTower::setHCALData(uint32_t hFB, uint32_t hET) {
  hcalET = hET;
  hcalFB = hFB;
  if(hET > etInputMax) {
    LOG_ERROR << "UCTTower::setData - hcalET too high " << hET << "; Pegged to etInputMax" << std::endl;
    hcalET = etInputMax;
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
  if(etIn > etInputMax) {
    LOG_ERROR << "UCTTower::setData - HF ET too high " << etIn << "; Pegged to etInputMax" << std::endl;
    hcalET = etInputMax;
  }
  if(fbIn > 0x3) {
    LOG_ERROR << "UCTTower::setData - too many HF FeatureBits " << std::hex << fbIn
	      << "; Used only bottom 2 bits" << std::endl;
    hcalFB &= 0x3;
  }
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
