#include "DataFormats/FTLRecHit/interface/FTLUncalibratedRecHit.h"
#include <cmath>

namespace {
  constexpr float FTLURecHitLSB = 1.26008;
}

FTLUncalibratedRecHit::FTLUncalibratedRecHit() :
  amplitude_(0.), time_(0.), flags_(0), aux_(0) { }

FTLUncalibratedRecHit::FTLUncalibratedRecHit(const DetId& id, float ampl, float time, uint32_t flags, uint32_t aux) :
  amplitude_(ampl), time_(time), flags_(flags), aux_(aux), id_(id) { }

FTLUncalibratedRecHit::~FTLUncalibratedRecHit() {
}

bool FTLUncalibratedRecHit::isSaturated() const {
  return FTLUncalibratedRecHit::checkFlag(kSaturated);
}


float FTLUncalibratedRecHit::timeError() const {
  // stored in ps, but return BXs to match with time units
  uint32_t timeErrorBits = 0xFF & aux_;
  // all bits off --> time reco bailed out (return negative value)
  if( (0xFF & timeErrorBits) == 0x00)
    return -1;
  // all bits on  --> time error over 5 ns (return large value)
  if( (0xFF & timeErrorBits) == 0xFF)
    return 10000;

  uint8_t exponent = timeErrorBits>>5;
  uint8_t significand = timeErrorBits & ~(0x7<<5);
  return (float)((1<<exponent)*significand*FTLURecHitLSB)/(25.*1000);
}

void FTLUncalibratedRecHit::setTimeError( float timeErr ) {
  // use 8 bits (3 exp, 5 mant) and store in ps
  // has range of 5 ps - 5000 ps
  // expect input in BX units
  // all bits off --> time reco bailed out
  if(timeErr < 0) {
    aux_ = (~0xFF & aux_);
    return;
  }
  // all bits on  --> time error over 5 ns
  if(25*timeErr >= 5) {
    aux_ = (0xFF | aux_);
    return;
  }

  float quantityInLSB = (1000*25*timeErr)/FTLURecHitLSB;
  int log2OfQuantity = (int) (log2( quantityInLSB ));
  int exponentTmp = log2OfQuantity - 4;
  uint8_t exponent=0;
  if (exponentTmp>0) exponent = exponentTmp;
  uint8_t significand = (int) ( lround( quantityInLSB / (1<<exponent) )   );
  uint32_t timeErrorBits = exponent<<5 | significand;
  
  if( (0xFF & timeErrorBits) == 0xFF)
    timeErrorBits = 0xFE;
  if( (0xFF & timeErrorBits) == 0x00)
    timeErrorBits = 0x01;

  aux_ = (~0xFF & aux_) | (timeErrorBits & 0xFF);

}

bool FTLUncalibratedRecHit::isTimeValid() const {
  if(timeError() <= 0)
    return false;
  else
    return true;
}

bool FTLUncalibratedRecHit::isTimeErrorValid() const {
  if(!isTimeValid())
    return false;
  if(timeError() >= 10000)
    return false;
  
  return true;
}

uint8_t FTLUncalibratedRecHit::timeErrorBits() const {
  uint8_t timeErrorBits = 0xFF & aux_;
  return timeErrorBits;
}


void FTLUncalibratedRecHit::setFlagBit(FTLUncalibratedRecHit::Flags flag) {

  if  (flag == kGood) {
    //then set all bits to zero;
    flags_  = 0;
    return;
  }
  // else set the flagbit
  flags_|= 0x1 <<  flag;  
}


bool FTLUncalibratedRecHit::checkFlag(FTLUncalibratedRecHit::Flags flag) const {
  if(flag == kGood){ if ( ! flags_ ) return true;else return false;} // if all flags are unset, then hit is good
  return  flags_ & ( 0x1<<flag);
}
