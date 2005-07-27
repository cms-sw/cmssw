#include "CalibFormats/CaloObjects/interface/CaloSamples.h"

CaloSamples::CaloSamples() : id_(), size_(0), presamples_(0) {
  for (int i=0; i<MAXSAMPLES; i++) data_[i]=0;
}

CaloSamples::CaloSamples(const cms::DetId& id, int size) : id_(id), size_(size), presamples_(0) {
  for (int i=0; i<MAXSAMPLES; i++) data_[i]=0;
}

void CaloSamples::setPresamples(int pre) {
}

std::ostream& operator<<(std::ostream& s, const CaloSamples& samples) {
  s << "DetId=" << samples.id().rawId();
  s << ", "<<  samples.size() << "samples" << std::endl;
  for (int i=0; i<samples.size(); i++)
    s << i << ":" << samples[i];
  return s;
}
