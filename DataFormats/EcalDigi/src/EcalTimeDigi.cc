#include "DataFormats/EcalDigi/interface/EcalTimeDigi.h"



EcalTimeDigi::EcalTimeDigi() : id_(0), size_(0), sampleOfInterest_(-1), data_(MAXSAMPLES)  {
}

EcalTimeDigi::EcalTimeDigi(const DetId& id) : id_(id),
					      size_(0), sampleOfInterest_(-1), data_(MAXSAMPLES) {
}

void EcalTimeDigi::setSize(unsigned int size) {
//   if (size<0) size_=0;
//   else 
  if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else size_=size;
  data_.resize(size_);
}

//unsigned int EcalTimeDigi::sampleOfInterest() const
//{
//  if (size_ == 1)
//    return 0;
//  else if (size_ == 5)
//    return 2;
//  else
//    return -1;
//} 


