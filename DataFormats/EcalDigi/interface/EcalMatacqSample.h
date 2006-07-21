#ifndef DIGIECAL_ECALMatacqSAMPLE_H
#define DIGIECAL_ECALMatacqSAMPLE_H

#include <ostream>
#include <boost/cstdint.hpp>

/** \class EcalMatacqSample
 *  Simple container packer/unpacker for a single sample from teh Matacq electronics
 *
 *
 *  $Id: EcalMatacqSample.h,v 1.2 2005/10/06 11:26:58 meridian Exp $
 */

class EcalMatacqSample {
 public:
  EcalMatacqSample() { theSample=0; }
  EcalMatacqSample(uint16_t data) { theSample=data; }
  EcalMatacqSample(int adc);
    
  /// get the raw word
  uint16_t raw() const { return theSample; }
  /// get the ADC sample (16 bits)
  int adc() const { return theSample&0xFFFF; }
  /// for streaming
  uint16_t operator()() { return theSample; }

 private:
  uint16_t theSample;
};

std::ostream& operator<<(std::ostream&, const EcalMatacqSample&);
  


#endif
