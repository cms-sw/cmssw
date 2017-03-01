#include "DataFormats/HGCRecHit/interface/HGCUncalibratedRecHit.h"
#include <math.h>

HGCUncalibratedRecHit::HGCUncalibratedRecHit() :
  amplitude_(0.), pedestal_(0.), jitter_(0.), chi2_(10000.), OOTamplitude_(0.), OOTchi2_(10000.), flags_(0), aux_(0) { }

HGCUncalibratedRecHit::HGCUncalibratedRecHit(const DetId& id, float ampl, float ped,
					     float jit, float chi2, uint32_t flags, uint32_t aux) :
  amplitude_(ampl), pedestal_(ped), jitter_(jit), chi2_(chi2), OOTamplitude_(0.), OOTchi2_(10000.), flags_(flags), aux_(aux), id_(id) { }

HGCUncalibratedRecHit::~HGCUncalibratedRecHit() {
}

bool HGCUncalibratedRecHit::isSaturated() const {
  return HGCUncalibratedRecHit::checkFlag(kSaturated);
}


float HGCUncalibratedRecHit::jitterError() const {
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

void HGCUncalibratedRecHit::setJitterError( float jitterErr ) {
  // use 8 bits (3 exp, 5 mant) and store in ps
  // has range of 5 ps - 5000 ps
  // expect input in BX units
  // all bits off --> time reco bailed out
  if(jitterErr < 0) {
    aux_ = (~0xFF & aux_);
    return;
  }
  // all bits on  --> time error over 5 ns
  if(25*jitterErr >= 5) {
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

bool HGCUncalibratedRecHit::isJitterValid() const {
  if(jitterError() <= 0)
    return false;
  else
    return true;
}

bool HGCUncalibratedRecHit::isJitterErrorValid() const {
  if(!isJitterValid())
    return false;
  if(jitterError() >= 10000)
    return false;
  
  return true;
}

uint8_t HGCUncalibratedRecHit::jitterErrorBits() const {
  uint8_t jitterErrorBits = 0xFF & aux_;
  return jitterErrorBits;
}


void HGCUncalibratedRecHit::setFlagBit(HGCUncalibratedRecHit::Flags flag) {

  if  (flag == kGood) {
    //then set all bits to zero;
    flags_  = 0;
    return;
  }
  // else set the flagbit
  flags_|= 0x1 <<  flag;  
}


bool HGCUncalibratedRecHit::checkFlag(HGCUncalibratedRecHit::Flags flag) const {
  if(flag == kGood){ if ( ! flags_ ) return true;else return false;} // if all flags are unset, then hit is good
  return  flags_ & ( 0x1<<flag);
}
