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

  enum HGCSampleMasks {ADC_MASK=0x7fff, GAIN_MASK=0x1 };
  enum HGCSamplePos   {ADC_POS=0      , GAIN_POS=15 };

  /**
     @short CTOR
   */
  HGCSample() : value_(0) { }
  HGCSample(uint16_t value) : value_(value) { }

  /**
     @short setters
   */
  void setGain(uint16_t gain)           { setWord(gain,GAIN_MASK,GAIN_POS);               }
  void setADC(uint16_t adc)             { setWord(adc,ADC_MASK,ADC_POS);                  }
  void set(uint16_t gain, uint16_t adc) { setGain(gain);                     setADC(adc); }  

  /**
     @short getters
  */
  uint16_t raw()  const { return value_; }
  uint16_t gain() const { return ((value_ >> GAIN_POS) & GAIN_MASK); }
  uint16_t adc()  const { return ((value_ >> ADC_POS) & ADC_MASK); }
  uint16_t operator()() { return value_; }
  
private:

  /**
     @short wrapper to reset words at a given position
   */
  void setWord(uint16_t word, uint16_t mask, uint16_t pos)
  {
    //clear required bits
    value_ &= ~((word & mask) << pos); 
    //now set the new value
    value_ |= ((word & mask) << pos);
  }

  // the word
  uint16_t value_;

};

  
#endif
