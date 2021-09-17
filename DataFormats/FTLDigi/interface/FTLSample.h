#ifndef DIGIFTL_FTLSAMPLE_H
#define DIGIFTL_FTLSAMPLE_H

#include <iostream>
#include <ostream>
#include <cstdint>

/**
   @class FTLSample
   @short wrapper for a data word
 */

class FTLSample {
public:
  enum FTLSampleMasks { kThreshMask = 0x1, kModeMask = 0x1, kToAMask = 0xfff, kDataMask = 0xfff };
  enum FTLSampleShifts { kThreshShift = 31, kModeShift = 30, kToAShift = 13, kDataShift = 0 };

  /**
     @short CTOR
   */
  FTLSample() : value_(0) {}
  FTLSample(uint32_t value) : value_(value) {}
  FTLSample(const FTLSample& o) : value_(o.value_) {}

  /**
     @short setters
   */
  void setThreshold(bool thr) { setWord(thr, kThreshMask, kThreshShift); }
  void setMode(bool mode) { setWord(mode, kModeMask, kModeShift); }
  void setToA(uint16_t toa) { setWord(toa, kToAMask, kToAShift); }
  void setData(uint16_t data) { setWord(data, kDataMask, kDataShift); }
  void set(bool thr, bool mode, uint16_t toa, uint16_t data) {
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
  bool threshold() const { return ((value_ >> kThreshShift) & kThreshMask); }
  bool mode() const { return ((value_ >> kModeShift) & kModeMask); }
  uint32_t toa() const { return ((value_ >> kToAShift) & kToAMask); }
  uint32_t data() const { return ((value_ >> kDataShift) & kDataMask); }
  uint32_t operator()() { return value_; }

private:
  /**
     @short wrapper to reset words at a given position
   */
  void setWord(uint32_t word, uint32_t mask, uint32_t pos) {
    //clear required bits
    const uint32_t masked_word = (word & mask) << pos;
    value_ &= ~(masked_word);
    //now set the new value
    value_ |= (masked_word);
  }

  // a 32-bit word
  uint32_t value_;
};

#endif
