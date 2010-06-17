#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <math.h>

EcalUncalibratedRecHit::EcalUncalibratedRecHit() :
     amplitude_(0.), pedestal_(0.), jitter_(0.), chi2_(10000.), flags_(0), aux_(0) { }

EcalUncalibratedRecHit::EcalUncalibratedRecHit(const DetId& id, double ampl, double ped,
                          double jit, double chi2, uint32_t flags, uint32_t aux) :
     amplitude_(ampl), pedestal_(ped), jitter_(jit), chi2_(chi2), flags_(flags), aux_(aux), id_(id) { }

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
        if ( energy > 0.001 ) {
                uint16_t exponent = lround(floor(log10(energy)))+3;
                uint16_t significand = lround(energy/pow(10,exponent-5));
                uint32_t rawEnergy = exponent<<10 | significand;
                setFlags( ( ~(0x1FFF<<4) & flags_) | ((rawEnergy & 0x1FFF)<<4) );
        }
}

void EcalUncalibratedRecHit::setOutOfTimeChi2( float chi2 )
{
        // use 7 bits
        if ( chi2 > 64. ) chi2 = 64.;
        uint32_t rawChi2 = lround( chi2 / 64. * ((1<<7)-1) );
        // shift by 17 bits (recoFlag + outOfTimeEnergy)
        setFlags( (~(0x7F<<17) & flags_) | ((rawChi2 & 0x7F)<<17) );
}

float EcalUncalibratedRecHit::outOfTimeChi2() const
{
        uint32_t rawChi2 = 0x7F & (flags_>>17);
        return (float)rawChi2 / (float)((1<<7)-1) * 64.;
}
