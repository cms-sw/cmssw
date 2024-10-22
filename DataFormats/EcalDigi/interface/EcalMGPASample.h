#ifndef DataFormats_EcalDigi_EcalMGPASample_h
#define DataFormats_EcalDigi_EcalMGPASample_h

#include <iosfwd>
#include <cstdint>
#include "DataFormats/EcalDigi/interface/EcalConstants.h"

namespace ecalMGPA {
  typedef uint16_t sample_type;

  /// get the ADC sample (12 bits)
  constexpr int adc(sample_type sample) { return sample & ecalPh1::kAdcMask; }
  /// get the gainId (2 bits)
  constexpr int gainId(sample_type sample) { return (sample >> ecalPh1::NBITS) & ecalPh1::kGainIdMask; }
  constexpr sample_type pack(int adc, int gainId) {
    return (adc & ecalPh1::kAdcMask) | ((gainId & ecalPh1::kGainIdMask) << ecalPh1::NBITS);
  }
}  // namespace ecalMGPA

/** \class EcalMGPASample
 *  Simple container packer/unpacker for a single sample from the MGPA electronics
 *
 *
 */
class EcalMGPASample {
public:
  EcalMGPASample() { theSample = 0; }
  EcalMGPASample(uint16_t data) { theSample = data; }
  EcalMGPASample(int adc, int gainId);

  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the ADC sample (12 bits)
  int adc() const { return theSample & ecalPh1::kAdcMask; }
  /// get the gainId (2 bits)
  int gainId() const { return (theSample >> ecalPh1::NBITS) & ecalPh1::kGainIdMask; }
  /// for streaming
  uint16_t operator()() const { return theSample; }
  operator uint16_t() const { return theSample; }

private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream&, const EcalMGPASample&);

#endif
