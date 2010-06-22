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

float EcalUncalibratedRecHit::jitterError() const
{
        // stored in ps, but return BXs to match with jitter units
        uint32_t jitterErrorBits = 0xFF & aux_;
        // all bits off --> time reco bailed out (return negative value)
        if( (0xFF & jitterErrorBits) == 0x00)
                return -1;
        // all bits on  --> time error over 5 ns (return large value)
        if( (0xFF & jitterErrorBits) == 0xFF)
                return 10000;

        float LSB = 1.26008;
        uint8_t exponent = jitterErrorBits>>5;
        uint8_t significand = jitterErrorBits & ~(0x7<<5);
        return (float)(pow(2.,exponent)*significand*LSB)/(25.*1000);
}

void EcalUncalibratedRecHit::setJitterError( float jitterErr )
{
        // use 8 bits (3 exp, 5 mant) and store in ps
        // has range of 5 ps - 5000 ps
        // expect input in BX units
        // all bits off --> time reco bailed out
        if(jitterErr < 0)
        {
                aux_ = (~0xFF & aux_);
                return;
        }
        // all bits on  --> time error over 5 ns
        if(25*jitterErr >= 5)
        {
                aux_ = (0xFF | aux_);
                return;
        }

        float LSB = 1.26008;
        float quantityInLSB = (1000*25*jitterErr)/LSB;
        int log2OfQuantity = (int) (log2( quantityInLSB ));
        int exponentTmp = log2OfQuantity - 4;
        uint8_t exponent=0;
        if (exponentTmp>0) exponent = exponentTmp;
        uint8_t significand = (int) ( lround( quantityInLSB / pow(2.,exponent) )   );
        uint32_t jitterErrorBits = exponent<<5 | significand;
  
        if( (0xFF & jitterErrorBits) == 0xFF)
          jitterErrorBits = 0xFE;
        if( (0xFF & jitterErrorBits) == 0x00)
          jitterErrorBits = 0x01;

        aux_ = (~0xFF & aux_) | (jitterErrorBits & 0xFF);

}

bool EcalUncalibratedRecHit::isJitterValid() const
{
        if(jitterError() <= 0)
          return false;
        else
          return true;
}

bool EcalUncalibratedRecHit::isJitterErrorValid() const
{
        if(!isJitterValid())
          return false;
        if(jitterError() >= 10000)
          return false;

        return true;
}

uint8_t EcalUncalibratedRecHit::jitterErrorBits() const
{
        uint8_t jitterErrorBits = 0xFF & aux_;
        return jitterErrorBits;
}
