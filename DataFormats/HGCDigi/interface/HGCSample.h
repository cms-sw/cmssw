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
  enum HGCSampleMasks {
    kThreshMask = 0x1,
    kModeMask = 0x1,
    kToAValidMask = 0x1,
    kGainMask = 0xf,
    kToAMask = 0x3ff,
    kDataMask = 0xfff
  };
  enum HGCSampleShifts {
    kThreshShift = 31,
    kModeShift = 30,
    kToAValidShift = 29,
    kToGainShift = 22,
    kToAShift = 12,
    kDataShift = 0
  };

  /**
     @short CTOR
  */
  HGCSample() : value_(0) {}
  HGCSample(uint32_t value) : value_(value) {}
  HGCSample(const HGCSample& o) : value_(o.value_) {}


  /**
     @short Data Model Evolution
  */
static uint32_t convertToNewFormat(uint32_t valueOldForm, bool toaFiredOldForm) { 
  // combine value&toaFired from the dataformat V9-or-earlier
  // from persisted objects
  // to produce a value_ compatible w/ the V10 format
  // i.e.
  // 1) shift the toa 12 bits by 1 bit
  // 2) insert the toaFired into _value
  // nothing can be done for the gain bits: info about gain was not preswent in V9-or-earlier and will be left to 0 in V10
  uint32_t valueNewForm(valueOldForm);
  std::cout << "\n\n function valueNewForm - valueOldForm: " << valueOldForm << " toaFiredOldForm: " << toaFiredOldForm << " valueNewForm: " << valueNewForm << std::endl;

  std::cout << "valueOldForm: " << valueOldForm << std::endl;
  std::cout << "toaFiredOldForm: " << toaFiredOldForm << std::endl;
  std::cout << "valueNewForm: " << valueNewForm << std::endl;
  // set to 0 the 17 bits bits between 13 and 29 (both included)
  valueNewForm &= ~(0x3FFFF << kToAShift);
  std::cout << "\nvalueNewForm (after 0-ing): " << valueNewForm << std::endl;
  // copy toa to start from bit 13
  std::cout << "\t extracting ToA " << ((valueOldForm >> 13) & kToAMask) << std::endl;
  valueNewForm |= ((valueOldForm >> 13) & kToAMask) << kToAShift;
  std::cout << "after setting ToA valueNewForm: " << valueNewForm << std::endl;
  // set 1 bit toaFiredOldForm in position 29
  std::cout << "\t toaFiredOldForm is  " << toaFiredOldForm << std::endl;
  valueNewForm |= (toaFiredOldForm & kToAValidMask ) << kToAValidShift;
  std::cout << "after setting toaFired valueNewForm: " << valueNewForm << std::endl;

  return valueNewForm;
  }
  /**
     @short setters
  */
  void setThreshold(bool thr) { setWord(thr, kThreshMask, kThreshShift); }
  void setMode(bool mode) { setWord(mode, kModeMask, kModeShift); }
  void setGain(uint16_t gain) { setWord(gain, kGainMask, kToGainShift); }
  void setToA(uint16_t toa) { setWord(toa, kToAMask, kToAShift); }
  void setData(uint16_t data) { setWord(data, kDataMask, kDataShift); }
  void setToAValid(bool toaFired) { setWord(toaFired, kToAValidMask, kToAValidShift); }

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
  bool threshold() const { return getWord(kThreshMask, kThreshShift); }
  bool mode() const { return getWord(kModeMask, kModeShift); }
  uint16_t gain() const { return getWord(kGainMask, kToGainShift); }
  uint16_t toa() const { return getWord(kToAMask, kToAShift); }
  uint16_t data() const { return getWord(kDataMask, kDataShift); }
  bool getToAValid() const { return getWord(kToAValidMask, kToAValidShift); }
  uint32_t operator()() { return value_; }

private:
  /**
     @short wrapper to reset words at a given position
  */
  void setWord(uint16_t word, HGCSampleMasks mask, HGCSampleShifts shift) {
    // mask and shift bits
    const uint32_t masked_word = (word & mask) << shift;

    //clear to 0  bits which will be set by word
    value_ &= ~(mask << shift);

    //now set bits
    value_ |= (masked_word);
  }

  uint32_t getWord(HGCSampleMasks mask, HGCSampleShifts shift) const { return ((value_ >> shift) & mask); }

  // a 32-bit word
  uint32_t value_;
};

#endif
