#ifndef DIGIHGCAL_HGCSAMPLE_H
#define DIGIHGCAL_HGCSAMPLE_H

#include <ostream>
#include <boost/cstdint.hpp>

/**
   @class HGCSample
   @short wrapper for a data word
 */

class HGCSample {

public:

  enum HGCSampleMasks {ADC_MASK=0xffff };
  enum HGCSamplePos   {ADC_POS=0x0     };

  /**
     @short CTOR
   */
  HGCSample() : value_(0) { }
  HGCSample(uint16_t value) : value_(value) { }

  /**
     @short setters
   */
  void setADC(uint16_t adc) { value_ = ((adc & ADC_MASK) << ADC_POS); }
  
  /**
     @short getters
   */
  uint16_t raw() const { return value_; }
  int adc()      const { return ((value_ >> ADC_POS) & ADC_MASK); }
  uint16_t operator()() { return value_; }
  
private:

  // the word
  uint16_t value_;

};

  
#endif
