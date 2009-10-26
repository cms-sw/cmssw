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
        uint32_t rawEnergy = (flags_>>4);
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
        if ( energy < 0 ) {
                edm::LogWarning("EcalUncalibratedRecHit::setOutOfTimeEnergy") << "Negative energy, cannot set it : " << energy;
        } else {
                uint16_t exponent = lround(floor(log10(energy)))+3;
                uint16_t significand = lround(energy/pow(10,exponent-5));
                uint32_t rawEnergy = exponent<<10 | significand;
                setFlags( ( ~(0xFFFFFFF<<4) & flags_) | ((rawEnergy & 0xFFFFFFF)<<4) );
        }
}

