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
  enum HGCSampleMasks  { kThreshMask  = 0x1, kModeMask  = 0x1, kGainMask    = 0x3,   kToAMask  = 0x3ff, kDataMask  = 0x3ff };
  enum HGCSampleShifts { kThreshShift = 31,  kModeShift = 30,  kToGainShift = 20,    kToAShift = 10,    kDataShift = 0 };

  /**
     @short CTOR
   */
 HGCSample() : value_(0), toaFired_(false) { /* std::cout << "GF HGCSample const 1" << std::endl; */ }
 HGCSample(uint32_t value) : value_(value), toaFired_(false) { /* std::cout << "GF HGCSample const 1" << std::endl;  */ }
 HGCSample(const HGCSample& o) : value_(o.value_), toaFired_(o.toaFired_) { /* std::cout << "GF HGCSample const 1" << std::endl;  */ }

  /**
     @short setters
   */

  // GF indentation
  void setThreshold(bool thr) { setWord(thr, kThreshMask, kThreshShift); }
  void setMode(bool mode) { setWord(mode, kModeMask, kModeShift); }
  void setGain(uint16_t gain) { setWord(gain, kGainMask, kToGainShift); }
  void setToA(uint16_t toa) { setWord(toa, kToAMask, kToAShift); }
  void setToAValid(bool toaFired) { toaFired_ = toaFired; }
  void setData(uint16_t data) { setWord(data, kDataMask, kDataShift); }

  // GF: why do we not use setWord for this case ??
  void set(bool thr, bool mode, uint16_t gain, uint16_t toa, uint16_t data) {
    setThreshold(thr);
    setMode(mode);
    setGain(gain);
    setToA(toa);
    setData(data);
  }

  void print(std::ostream& out = std::cout) {
    out << "THR: " << threshold() << " Mode: " << mode() << " ToA: " << toa() << " Data: " << data() << " Raw=0x"
        << std::hex << raw() << std::dec << std::endl;
  }

  /**
     @short getters
  */
  uint32_t raw() const { return value_; }
  bool     threshold() const { return ((value_ >> kThreshShift) & kThreshMask); }
  bool     mode() const { return ((value_ >> kModeShift) & kModeMask); }
  bool     getToAValid() const { return toaFired_; }
  uint32_t gain() const { return ((value_ >> kToGainShift) & kGainMask); }
  uint32_t toa() const { return ((value_ >> kToAShift) & kToAMask); }
  uint32_t data() const { return ((value_ >> kDataShift) & kDataMask); }
  uint32_t operator()() { return value_; }

private:
  /**
     @short wrapper to reset words at a given position
   */
  void setWord(uint32_t word, uint32_t mask, uint32_t pos) {

    // deal with saturation: set to mask 
    // should we throw ?
    word = ( mask > word  ?  word : mask );
    // std::cout <<  "word: " << word << "  mask: " << mask <<  " pos: " << pos << std::endl;

    // mask (not strictly needed) and shift
    const uint32_t masked_word = (word & mask) << pos;
    //std::cout <<  "\t masked_word " << masked_word << std::endl;

    //clear to 0  bits which will be set
    value_ &= ~(mask << pos);
    // std::cout <<  "\t value tmp " << value_ << std::endl;

    //now set bits
    value_ |= (masked_word);
    //std::cout <<  "\t value " << value_ << std::endl;
  }

  // a 32-bit word
  uint32_t value_;
  bool toaFired_;
};

#endif
