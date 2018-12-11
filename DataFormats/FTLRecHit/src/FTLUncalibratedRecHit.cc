#include "DataFormats/FTLRecHit/interface/FTLUncalibratedRecHit.h"
#include <cmath>
#include <limits>

FTLUncalibratedRecHit::FTLUncalibratedRecHit() :
  amplitude_(-1.f,-1.f), time_(-1.f,-1.f), timeError_(-1.f), id_(DetId()),
  row_(0), column_(0), flags_(std::numeric_limits<unsigned char>::max()) { }

FTLUncalibratedRecHit::FTLUncalibratedRecHit(const DetId& id, std::pair <float,float> ampl,
					     std::pair <float,float> time, float timeError, unsigned char flags) :
  amplitude_(ampl), time_(time), timeError_(timeError), id_(id), row_(0), column_(0), flags_(flags) { }

FTLUncalibratedRecHit::FTLUncalibratedRecHit(const DetId& id, uint8_t row, uint8_t column,
					     std::pair <float,float>  ampl, std::pair <float,float> time,
					     float timeError, unsigned char flags) :
  amplitude_(ampl), time_(time), timeError_(timeError), id_(id), row_(row), column_(column), flags_(flags) { }

FTLUncalibratedRecHit::~FTLUncalibratedRecHit() {
}

bool FTLUncalibratedRecHit::isSaturated() const {
  return FTLUncalibratedRecHit::checkFlag(kSaturated);
}

bool FTLUncalibratedRecHit::isTimeValid() const {
  if(timeError() < 0)
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
