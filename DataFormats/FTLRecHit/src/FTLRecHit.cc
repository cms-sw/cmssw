#include "DataFormats/FTLRecHit/interface/FTLRecHit.h"
#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"
#include <cassert>
#include <math.h>

namespace {
  constexpr float chi2_constant = 64.f/((1<<7) - 1);
}

FTLRecHit::FTLRecHit() : CaloRecHit(), flagBits_(0) {
}

FTLRecHit::FTLRecHit(const DetId& id, float energy, float time, uint32_t flags, uint32_t flagBits) :
  CaloRecHit(id,energy,time,flags), flagBits_(flagBits) {
}

float FTLRecHit::chi2() const {
  
  uint32_t rawChi2 = 0x7F & (flags()>>4);
  return (float)rawChi2 * chi2_constant;
}

float FTLRecHit::outOfTimeChi2() const {
  uint32_t rawChi2Prob = 0x7F & (flags()>>24);
  return (float)rawChi2Prob * chi2_constant;
}

float FTLRecHit::outOfTimeEnergy() const {
  uint32_t rawEnergy = (0x1FFF & flags()>>11);
  uint16_t exponent = rawEnergy>>10;
  uint16_t significand = ~(0xE<<9) & rawEnergy;
  return (float) significand*pow(10,exponent-5);
}

void FTLRecHit::setChi2( float chi2 ) {
  // bound the max value of the chi2
  if ( chi2 > 64 ) chi2 = 64;
  // use 7 bits
  uint32_t rawChi2 = lround( chi2 / 64. * ((1<<7)-1) );
  // shift by 4 bits (recoFlag)
  setFlags( (~(0x7F<<4) & flags()) | ((rawChi2 & 0x7F)<<4) );
}

void FTLRecHit::setOutOfTimeEnergy( float energy ) {
  if ( energy > 0.001 ) {
    uint16_t exponent = lround(floor(log10(energy)))+3;
    uint16_t significand = lround(energy/pow(10,exponent-5));
    // use 13 bits (3 exponent, 10 significand)
    uint32_t rawEnergy = exponent<<10 | significand;
    // shift by 11 bits (recoFlag + chi2)
    setFlags( ( ~(0x1FFF<<11) & flags()) | ((rawEnergy & 0x1FFF)<<11) );
  }
}

void FTLRecHit::setOutOfTimeChi2( float chi2 ) {
  // bound the max value of chi2
  if ( chi2 > 64 ) chi2 = 64;
  // use 7 bits
  uint32_t rawChi2 = lround( chi2 / 64. * ((1<<7)-1) );
  // shift by 24 bits (recoFlag + chi2 + outOfTimeEnergy)
  setFlags( (~(0x7F<<24) & flags()) | ((rawChi2 & 0x7F)<<24) );
}

void FTLRecHit::setTimeError( uint8_t timeErrBits ) {
  // take the bits and put them in the right spot
  setAux( (~0xFF & aux()) | timeErrBits );
}

float FTLRecHit::timeError() const {
  uint32_t timeErrorBits = 0xFF & aux();
  // all bits off --> time reco bailed out (return negative value)
  if( (0xFF & timeErrorBits) == 0x00 )
                return -1;
  // all bits on  --> time error over 5 ns (return large value)
  if( (0xFF & timeErrorBits) == 0xFF )
    return 10000;
  
  float LSB = 1.26008;
  uint8_t exponent = timeErrorBits>>5;
  uint8_t significand = timeErrorBits & ~(0x7<<5);
  return pow(2.,exponent)*significand*LSB/1000.;
}

bool FTLRecHit::isTimeValid() const {
  if(timeError() <= 0)
    return false;
  else
    return true;
}

bool FTLRecHit::isTimeErrorValid() const {
  if(!isTimeValid())
    return false;
  if(timeError() >= 10000)
    return false;
  
  return true;
}

/// check if one of the flags in a set is true
bool FTLRecHit::checkFlags(const std::vector<int>&  flagsvec ) const {
  
  for (std::vector<int>::const_iterator flagPtr = flagsvec.begin(); 
       flagPtr!= flagsvec.end(); ++flagPtr) { // check if one of the flags is up
    if (checkFlag(*flagPtr)) return true;    
  }
  return false;
}


std::ostream& operator<<(std::ostream& s, const FTLRecHit& hit) {
  if (hit.detid().det() == DetId::Forward && hit.detid().subdetId() == FastTime) 
    return s << FastTimeDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else
    return s << "FTLRecHit undefined subdetector" ;
}
