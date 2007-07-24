#ifndef DIGIECAL_ECALMGPASAMPLE_H
#define DIGIECAL_ECALMGPASAMPLE_H

#include <iosfwd>
#include <boost/cstdint.hpp>

/** \class EcalMGPASample
 *  Simple container packer/unpacker for a single sample from teh MGPA electronics
 *
 *
 *  $Id: EcalMGPASample.h,v 1.4 2007/04/16 12:58:56 meridian Exp $
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
