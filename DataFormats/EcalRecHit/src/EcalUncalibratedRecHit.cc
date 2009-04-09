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
        return (float)rawEnergy * 0.01;
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
                uint32_t rawEnergy = lround( energy / 0.01); // 0.01 ADC count resolution
                setFlags( ( ~(0xFFFFFFF<<4) & flags_) | ((rawEnergy & 0xFFFFFFF)<<4) );
        }
}


