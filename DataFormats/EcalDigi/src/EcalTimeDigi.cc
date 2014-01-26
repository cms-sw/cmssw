#include "DataFormats/EcalDigi/interface/EcalTimeDigi.h"



EcalTimeDigi::EcalTimeDigi() : size_(0), data_(MAXSAMPLES) {
}
EcalTimeDigi::EcalTimeDigi(const DetId& id) : id_(id),
										   size_(0), data_(MAXSAMPLES) {
}

void EcalTimeDigi::setSize(int size) {
  if (size<0) size_=0;
  else if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else size_=size;
}

int EcalTimeDigi::sampleOfInterest() const
{
  if (size_ == 1)
    return 0;
  else if (size_ == 5)
    return 2;
  else
    return -1;
} 


