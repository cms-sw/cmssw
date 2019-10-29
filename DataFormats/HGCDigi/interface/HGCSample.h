#ifndef DIGIHGCAL_HGCSAMPLE_H
#define DIGIHGCAL_HGCSAMPLE_H

#include <iostream>
#include <ostream>
#include <cstdint>

/**
   @class HGCSample
   @short wrapper for a data word
 */

class HGCSample {

public:
  enum HGCSampleMasks  { kThreshMask  = 0x1,   kModeMask  = 0x1,   kToAValidMask  = 0x1,
			 kGainMask    = 0x3,   kToAMask   = 0x3ff, kDataMask      = 0x3ff };
  enum HGCSampleShifts { kThreshShift = 31,    kModeShift = 30,    kToAValidShift = 29,
			 kToGainShift = 20,    kToAShift  = 10,    kDataShift     = 0 };

  /**
     @short CTOR
  */
 HGCSample() : value_(0) { }
 HGCSample(uint32_t value) : value_(value) { }
 HGCSample(const HGCSample& o) : value_(o.value_) { }

  /**
     @short setters
  */
  void setThreshold(bool thr)     { setWord(thr, kThreshMask, kThreshShift); }
  void setMode(bool mode)         { setWord(mode, kModeMask, kModeShift); }
  void setGain(uint16_t gain)     { setWord(gain, kGainMask, kToGainShift); }
  void setToA(uint16_t toa)       { setWord(toa, kToAMask, kToAShift); }
  void setData(uint16_t data)     { setWord(data, kDataMask, kDataShift); }
  void setToAValid(bool toaFired) { setWord(toaFired, kToAValidMask, kToAValidShift); }

  void set(bool thr, bool mode, uint16_t gain,
	   uint16_t toa, uint16_t data) {
    setThreshold(thr);
    setMode(mode);
    setGain(gain);
    setToA(toa);
    setData(data);
  }

  void print(std::ostream& out = std::cout) {
    out << "THR: " << threshold() << " Mode: " << mode() 
	<< " ToA: " << toa() << " Data: " << data() << " Raw=0x"
        << std::hex << raw() << std::dec << std::endl;
  }

  /**
     @short getters
  */
  uint32_t raw()         const { return value_; }
  bool     threshold()   const { return getWord(kThreshMask, kThreshShift); }
  bool     mode()        const { return getWord(kModeMask, kModeShift); }
  uint32_t gain()        const { return getWord(kGainMask, kToGainShift); }
  uint32_t toa()         const { return getWord(kToAMask, kToAShift); }
  uint32_t data()        const { return getWord(kDataMask, kDataShift); }
  bool     getToAValid() const { return getWord(kToAValidMask, kToAValidShift); }
  uint32_t operator()() { return value_; }


 private:
  /**
     @short wrapper to reset words at a given position
  */
  void setWord(uint32_t word, uint32_t mask, uint32_t shift) {

    // deal with saturation: set to mask 
    // should we throw ?
    word = ( mask > word  ?  word : mask );

    // mask (not strictly needed) and shift
    const uint32_t masked_word = (word & mask) << shift;

    //clear to 0  bits which will be set by word
    value_ &= ~(mask << shift);

    //now set bits
    value_ |= (masked_word);
  }

  uint32_t getWord(uint32_t mask, uint32_t shift) const {
    return ((value_ >> shift) & mask);
  }

  // a 32-bit word
  uint32_t value_;
};

#endif
