#ifndef DIGIECAL_ESSAMPLE_H
#define DIGIECAL_ESSAMPLE_H

#include <ostream>
#include <cstdint>

class ESSample {
public:
  ESSample() { theSample = 0; }
  ESSample(int16_t data) { theSample = data; }
  ESSample(int adc);

  /// get the raw word
  int16_t raw() const { return theSample; }
  /// get the ADC sample (singed 16 bits)
  int adc() const { return theSample; }
  /// for streaming
  int16_t operator()() { return theSample; }

private:
  int16_t theSample;
};

std::ostream& operator<<(std::ostream&, const ESSample&);

#endif
