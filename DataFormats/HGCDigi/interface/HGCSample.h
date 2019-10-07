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
  enum HGCSampleMasks { kThreshMask = 0x1, kModeMask = 0x1, kToAMask = 0x3ff, kDataMask = 0xfff };
  enum HGCSampleShifts { kThreshShift = 31, kModeShift = 30, kToAShift = 13, kDataShift = 0 };

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
  void setToA(uint16_t toa) { setWord(toa, kToAMask, kToAShift); }
  void setToAValid(bool toaFired) { toaFired_ = toaFired; }
  void setData(uint16_t data) { setWord(data, kDataMask, kDataShift); }

  // GF: why do we not use setWord for this case ??
  void set(bool thr, bool mode, uint16_t toa, uint16_t data) {

    // std::cout << "GF HGCSample - in the set method " << std::endl;
    toa = (toa > (uint16_t)kToAMask ? (uint16_t)kToAMask : toa);
    data = (data > (uint16_t)kDataMask ? (uint16_t)kDataMask : data);

    value_ = (((uint32_t)thr & kThreshMask) << kThreshShift | ((uint32_t)mode & kModeMask) << kModeShift |
              ((uint32_t)toa & kToAMask) << kToAShift | ((uint32_t)data & kDataMask) << kDataShift);
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
  uint32_t toa() const { return ((value_ >> kToAShift) & kToAMask); }
  uint32_t data() const { return ((value_ >> kDataShift) & kDataMask); }
  uint32_t operator()() { return value_; }

private:
  /**
     @short wrapper to reset words at a given position
   */
  void setWord(uint32_t word, uint32_t mask, uint32_t pos) {
    if (word > mask)
      word = mask;  // deal with saturation - should we throw ?
    //clear required bits
    // std::cout <<  "word: " << word << "  mask: " << mask <<  " pos: " << pos << std::endl;
    const uint32_t masked_word = (word & mask) << pos;
    //std::cout <<  "\t masked_word " << masked_word << std::endl;
    value_ &= ~(mask << pos);
    // std::cout <<  "\t value tmp " << value_ << std::endl;
    //now set the new value
    value_ |= (masked_word);
    //std::cout <<  "\t value " << value_ << std::endl;
  }

  // a 32-bit word
  uint32_t value_;
  bool toaFired_;
};

#endif
