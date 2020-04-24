#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"

EcalSamplesCorrelation::EcalSamplesCorrelation() {}
EcalSamplesCorrelation::~EcalSamplesCorrelation() {}

EcalSamplesCorrelation::EcalSamplesCorrelation(
    const EcalSamplesCorrelation& aset) {}

template <typename T>
static inline void print_vector(std::ostream& o, const std::vector<T>& vect) {
  o << "[";
  for (std::vector<double>::const_iterator i = vect.begin(); i != vect.end();
       ++i) {

    std::cout << *i << ", ";
  }
  o << "]";
}

void EcalSamplesCorrelation::print(std::ostream& o) const {
  o << "EB Gain 12 correlation:";
  print_vector<double>(o, this->EBG12SamplesCorrelation);
  o << std::endl;
  o << "EB Gain 6 correlation:";
  print_vector<double>(o, this->EBG6SamplesCorrelation);
  o << std::endl;
  o << "EB Gain 1 correlation:";
  print_vector<double>(o, this->EBG1SamplesCorrelation);
  o << std::endl;

  o << "EE Gain 12 correlation:";
  print_vector<double>(o, this->EEG12SamplesCorrelation);
  o << std::endl;
  o << "EE Gain 6 correlation:";
  print_vector<double>(o, this->EEG6SamplesCorrelation);
  o << std::endl;
  o << "EE Gain 1 correlation:";
  print_vector<double>(o, this->EEG1SamplesCorrelation);
  o << std::endl;
}
