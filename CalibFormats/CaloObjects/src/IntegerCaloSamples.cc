#include "CalibFormats/CaloObjects/interface/IntegerCaloSamples.h"

IntegerCaloSamples::IntegerCaloSamples() : id_(), size_(0), presamples_(0) {
  for (int i=0; i<MAXSAMPLES; i++) data_[i]=0;
}

IntegerCaloSamples::IntegerCaloSamples(const DetId& id, int size) : id_(id), size_(size), presamples_(0) {
  for (int i=0; i<MAXSAMPLES; i++) data_[i]=0;
}

void IntegerCaloSamples::setPresamples(int pre) {
  presamples_=pre;
}


std::ostream& operator<<(std::ostream& s, const IntegerCaloSamples& samples) {
  s << "DetId=" << samples.id().rawId();
  s << ", "<<  samples.size() << "samples" << std::endl;
  for (int i=0; i<samples.size(); i++)
    s << i << ":" << samples[i] << std::endl;
  return s;
}
