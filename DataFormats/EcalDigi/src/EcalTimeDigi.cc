#include "DataFormats/EcalDigi/interface/EcalTimeDigi.h"

namespace {
  constexpr unsigned int MAXSAMPLES = 10;
}

EcalTimeDigi::EcalTimeDigi() : id_(0), size_(0), sampleOfInterest_(-1), data_(MAXSAMPLES)  {
}

EcalTimeDigi::EcalTimeDigi(const DetId& id) : id_(id),
					      size_(0), sampleOfInterest_(-1), data_(MAXSAMPLES) {
}

void EcalTimeDigi::setSize(unsigned int size) {
  if (size>MAXSAMPLES) size_=MAXSAMPLES;
  else size_=size;
  data_.resize(size_);
}




