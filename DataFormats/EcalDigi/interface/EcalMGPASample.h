#ifndef DIGIECAL_ECALMGPASAMPLE_H
#define DIGIECAL_ECALMGPASAMPLE_H

#include <iosfwd>
#include <boost/cstdint.hpp>

namespace ecalMGPA {
  typedef uint16_t sample_type;
 
  /// get the ADC sample (12 bits)
  inline int adc(sample_type sample) { return sample&0xFFF; }
  /// get the gainId (2 bits)
  inline int gainId(sample_type sample) { return (sample>>12)&0x3; }
  inline sample_type pack(int adc, int gainId) {
    return (adc&0xFFF) | ((gainId&0x3)<<12);
  }
}


/** \class EcalMGPASample
 *  Simple container packer/unpacker for a single sample from teh MGPA electronics
 *
 *
 *  $Id: EcalMGPASample.h,v 1.6 2007/08/07 06:09:39 innocent Exp $
 */
class EcalMGPASample {
 public:
  EcalMGPASample() { theSample=0; }
  EcalMGPASample(uint16_t data) { theSample=data; }
  EcalMGPASample(int adc, int gainId);
    
  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the ADC sample (12 bits)
  int adc() const { return theSample&0xFFF; }
  /// get the gainId (2 bits)
  int gainId() const { return (theSample>>12)&0x3; }
  /// for streaming
  uint16_t operator()() const { return theSample; }
  operator uint16_t () const { return theSample; }

 private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream&, const EcalMGPASample&);
  


#endif
