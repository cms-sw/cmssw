#include "DataFormats/HGCRecHit/interface/HGCRecHit.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <cassert>
#include <cmath>

HGCRecHit::HGCRecHit() : CaloRecHit(), flagBits_(0) {}

HGCRecHit::HGCRecHit(const DetId& id, float energy, float time, uint32_t flags, uint32_t flagBits)
    : CaloRecHit(id, energy, time, flags), flagBits_(flagBits) {}

float HGCRecHit::chi2() const {
  uint32_t rawChi2 = 0x7F & (flags() >> 4);
  return (float)rawChi2 / (float)((1 << 7) - 1) * 64.f;
}

float HGCRecHit::outOfTimeChi2() const {
  uint32_t rawChi2Prob = 0x7F & (flags() >> 24);
  return (float)rawChi2Prob / (float)((1 << 7) - 1) * 64.f;
}

float HGCRecHit::outOfTimeEnergy() const {
  uint32_t rawEnergy = (0x1FFF & flags() >> 11);
  uint16_t exponent = rawEnergy >> 10;
  uint16_t significand = ~(0xE << 9) & rawEnergy;
  return (float)significand * pow(10, exponent - 5);
}

void HGCRecHit::setChi2(float chi2) {
  // bound the max value of the chi2
  if (chi2 > 64)
    chi2 = 64;
  // use 7 bits
  uint32_t rawChi2 = lround(chi2 / 64.f * ((1 << 7) - 1));
  // shift by 4 bits (recoFlag)
  setFlags((~(0x7F << 4) & flags()) | ((rawChi2 & 0x7F) << 4));
}

void HGCRecHit::setOutOfTimeEnergy(float energy) {
  if (energy > 0.001f) {
    uint16_t exponent = lround(floor(log10(energy))) + 3;
    uint16_t significand = lround(energy / pow(10, exponent - 5));
    // use 13 bits (3 exponent, 10 significand)
    uint32_t rawEnergy = exponent << 10 | significand;
    // shift by 11 bits (recoFlag + chi2)
    setFlags((~(0x1FFF << 11) & flags()) | ((rawEnergy & 0x1FFF) << 11));
  }
}

void HGCRecHit::setOutOfTimeChi2(float chi2) {
  // bound the max value of chi2
  if (chi2 > 64)
    chi2 = 64;
  // use 7 bits
  uint32_t rawChi2 = lround(chi2 / 64.f * ((1 << 7) - 1));
  // shift by 24 bits (recoFlag + chi2 + outOfTimeEnergy)
  setFlags((~(0x7F << 24) & flags()) | ((rawChi2 & 0x7F) << 24));
}

void HGCRecHit::setSignalOverSigmaNoise(float sOverNoise) {
  // bound the max value of sOverNoise
  if (sOverNoise > 32.f)
    sOverNoise = 32.f;
  //use 8 bits
  signalOverSigmaNoise_ = lround(sOverNoise / 32.f * ((1 << 8) - 1));
}

float HGCRecHit::signalOverSigmaNoise() const { return (float)signalOverSigmaNoise_ * 0.125f; }

void HGCRecHit::setTimeError(float timeErr) {
  //expected resolution on single cell for that given S/N
  timeError_ = timeErr;
}

float HGCRecHit::timeError() const { return timeError_; }

bool HGCRecHit::isTimeValid() const {
  if (timeError() <= 0)
    return false;
  else
    return true;
}

bool HGCRecHit::isTimeErrorValid() const {
  if (!isTimeValid())
    return false;
  if (timeError() >= 10000)
    return false;

  return true;
}

/// check if one of the flags in a set is true
bool HGCRecHit::checkFlags(const std::vector<int>& flagsvec) const {
  for (std::vector<int>::const_iterator flagPtr = flagsvec.begin(); flagPtr != flagsvec.end();
       ++flagPtr) {  // check if one of the flags is up
    if (checkFlag(*flagPtr))
      return true;
  }
  return false;
}

std::ostream& operator<<(std::ostream& s, const HGCRecHit& hit) {
  if (hit.detid().det() == DetId::Forward && hit.detid().subdetId() == HGCEE)
    return s << HGCalDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else if (hit.detid().det() == DetId::Forward && hit.detid().subdetId() == HGCHEF)
    return s << HGCalDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else if (hit.detid().det() == DetId::Hcal && hit.detid().subdetId() == HcalEndcap)
    return s << HcalDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else if (hit.detid().det() == DetId::HGCalEE || hit.detid().det() == DetId::HGCalHSi)
    return s << HGCSiliconDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else if (hit.detid().det() == DetId::HGCalHSc)
    return s << HGCScintillatorDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else if (hit.detid().det() == DetId::Forward && hit.detid().subdetId() == HFNose)
    return s << HFNoseDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else
    return s << "HGCRecHit undefined subdetector";
}
