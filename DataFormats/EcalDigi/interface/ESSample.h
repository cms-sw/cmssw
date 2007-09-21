#ifndef DIGIECAL_ESSAMPLE_H
#define DIGIECAL_ESSAMPLE_H

#include <ostream>
#include <boost/cstdint.hpp>

class ESSample {

 public:

  ESSample() { theSample = 0; }
  ESSample(const uint16_t& data) { theSample = data; }
  ESSample(const int& adc);
    
  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the ADC sample (12 bits)
  int adc() const { return theSample&0xFFF; }
  /// for streaming
  uint16_t operator()() { return theSample; }

 private:

  uint16_t theSample;

};

std::ostream& operator<<(std::ostream&, const ESSample&);

#endif
