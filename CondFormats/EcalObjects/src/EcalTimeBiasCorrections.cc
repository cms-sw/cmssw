#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"

EcalTimeBiasCorrections::EcalTimeBiasCorrections() {}
EcalTimeBiasCorrections::~EcalTimeBiasCorrections() {}

EcalTimeBiasCorrections::EcalTimeBiasCorrections(const EcalTimeBiasCorrections& aset) {}

template <typename T>
static inline void print_vector(std::ostream& o, const std::vector<T>& vect) {
  o << "[";
  for (std::vector<float>::const_iterator i = vect.begin(); i != vect.end(); ++i) {
    std::cout << *i << ", ";
  }
  o << "]";
}

void EcalTimeBiasCorrections::print(std::ostream& o) const {
  o << "EB Amplitude bins:";
  print_vector<float>(o, this->EBTimeCorrAmplitudeBins);
  o << std::endl;
  o << "EE Amplitude bins:";
  print_vector<float>(o, this->EETimeCorrAmplitudeBins);
  o << std::endl;

  o << "EB Shift bins:";
  print_vector<float>(o, this->EBTimeCorrShiftBins);
  o << std::endl;
  o << "EE Shift bins:";
  print_vector<float>(o, this->EETimeCorrShiftBins);
  o << std::endl;
}
