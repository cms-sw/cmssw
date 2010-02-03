#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <math.h>

EcalUncalibratedRecHit::EcalUncalibratedRecHit() :
     amplitude_(0.), pedestal_(0.), jitter_(0.), chi2_(10000.), flags_(0) { }

EcalUncalibratedRecHit::EcalUncalibratedRecHit(const DetId& id, const double& ampl, const double& ped,
                          const double& jit, const double& chi2, const uint32_t &flags) :
     amplitude_(ampl), pedestal_(ped), jitter_(jit), chi2_(chi2), flags_(flags), id_(id) { }

EcalUncalibratedRecHit::~EcalUncalibratedRecHit() {
}

bool EcalUncalibratedRecHit::isSaturated() const {
  return ( recoFlag() == kSaturated );
}

float EcalUncalibratedRecHit::outOfTimeEnergy() const
{
        uint32_t rawEnergy = (0x1FFF & flags_>>4);
        uint16_t exponent = rawEnergy>>10;
        uint16_t significand = ~(0xE<<9) & rawEnergy;
        return (float) significand*pow(10,exponent-5);
}

void EcalUncalibratedRecHit::setRecoFlag( uint32_t flag )
{
        flags_ = (~0xF & flags_) | (flag & 0xF);
}

void EcalUncalibratedRecHit::setOutOfTimeEnergy( float energy )
{
        if ( energy > 0 ) {
                uint16_t exponent = lround(floor(log10(energy)))+3;
                uint16_t significand = lround(energy/pow(10,exponent-5));
                uint32_t rawEnergy = exponent<<10 | significand;
                setFlags( ( ~(0x1FFF<<4) & flags_) | ((rawEnergy & 0x1FFF)<<4) );
        }
}

void EcalUncalibratedRecHit::setOutOfTimeChi2Prob( float chi2Prob )
{
        if ( chi2Prob < 0 || chi2Prob > 1 ) {
                edm::LogWarning("EcalUncalibratedRecHit::setOutOfTimeChi2Prob") << "chi2Prob outside limits [0, 1] : " << chi2Prob;
        } else {
                // use 7 bits
                uint32_t rawChi2Prob = lround( chi2Prob * ((1<<7)-1) );
                // shift by 17 bits (recoFlag + outOfTimeEnergy)
                setFlags( (~(0x7F<<17) & flags_) | ((rawChi2Prob & 0x7F)<<17) );
        }
}

float EcalUncalibratedRecHit::outOfTimeChi2Prob() const
{
        uint32_t rawChi2Prob = 0x7F & (flags_>>17);
        return (float)rawChi2Prob / (float)((1<<7)-1);
}

