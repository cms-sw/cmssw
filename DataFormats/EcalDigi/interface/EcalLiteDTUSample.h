#ifndef DataFormats_EcalDigi_EcalLiteDTUSample_h
#define DataFormats_EcalDigi_EcalLiteDTUSample_h

#include <iosfwd>
#include <cstdint>
#include "DataFormats/EcalDigi/interface/EcalConstants.h"

namespace ecalLiteDTU {
  typedef uint16_t sample_type;

  /// get the ADC sample (12 bits)
  constexpr int adc(sample_type sample) { return sample & ecalPh2::kAdcMask; }
  /// get the gainId (2 bits)
  constexpr int gainId(sample_type sample) { return (sample >> ecalPh2::NBITS) & ecalPh2::kGainIdMask; }
  constexpr sample_type pack(int adc, int gainId) {
    return (adc & ecalPh2::kAdcMask) | ((gainId & ecalPh2::kGainIdMask) << ecalPh2::NBITS);
  }
}  // namespace ecalLiteDTU

/** \class EcalLiteDTUSample
 *  Simple container packer/unpacker for a single sample from the Lite_CATIA electronics
 *
 *
 */
class EcalLiteDTUSample {
public:
  EcalLiteDTUSample() { theSample = 0; }
  EcalLiteDTUSample(uint16_t data) { theSample = data; }
  EcalLiteDTUSample(int adc, int gainId);

  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the ADC sample (12 bits)
  int adc() const { return theSample & ecalPh2::kAdcMask; }
  /// get the gainId (2 bits)
  int gainId() const { return (theSample >> ecalPh2::NBITS) & ecalPh2::kGainIdMask; }
  /// for streaming
  uint16_t operator()() const { return theSample; }
  operator uint16_t() const { return theSample; }

private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream&, const EcalLiteDTUSample&);

#endif
