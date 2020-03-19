#ifndef DIGIECAL_ECALFEMSAMPLE_H
#define DIGIECAL_ECALFEMSAMPLE_H

#include <ostream>
#include <cstdint>

/** \class EcalFEMSample
 *  Simple container packer/unpacker for a single sample from the FEM electronics
 *
 *
 *  $Id: 
 */

class EcalFEMSample {
public:
  EcalFEMSample() { theSample = 0; }
  EcalFEMSample(uint16_t data) { theSample = data; }
  EcalFEMSample(int adc, int gainId);

  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the ADC sample (12 bits)
  int adc() const { return theSample & 0xFFF; }
  /// get the gainId (2 bits)
  int gainId() const { return (theSample >> 12) & 0x3; }
  /// for streaming
  uint16_t operator()() { return theSample; }

private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream&, const EcalFEMSample&);

#endif
