#ifndef DIGIHGCAL_HGCSAMPLE_H
#define DIGIHGCAL_HGCSAMPLE_H

#include <iostream>
#include <ostream>
#include <boost/cstdint.hpp>

/**
   @class HGCSample
   @short wrapper for a data word
 */

class HGCSample {

public:

  enum HGCSampleMasks { kThreshMask = 0x1, kModeMask = 0x1, kToAMask = 0x3ff, kDataMask = 0xfff};
  enum HGCSampleShifts{ kThreshShift = 31, kModeShift = 30, kToAShift = 16,   kDataShift = 0};

  /**
     @short CTOR
   */
 HGCSample() : value_(0) { }
 HGCSample(uint32_t value) : value_(value) { }

  /**
     @short setters
   */
  void setThreshold(bool thr)           { setWord(thr,  kThreshMask, kThreshShift); }
  void setMode(bool mode)               { setWord(mode, kModeMask,   kModeShift);   }
  void setToA(uint16_t toa)             { setWord(toa,  kToAMask,    kToAShift);    }
  void setData(uint16_t data)           { setWord(data, kDataMask,   kDataShift);   }
  void set(bool thr, bool mode,uint16_t toa, uint16_t data) 
  { 
    setThreshold(thr);
    setMode(mode);
    setToA(toa);
    setData(data);
  }  
  void print(std::ostream &out=std::cout)
  {
    out << "THR: " << threshold() 
	<< " Mode: " << mode() 
	<< " ToA: " << toa() 
	<< " Data: " << data() 
	<< " Raw=0x" << std::hex << raw() << std::dec << std::endl;  
  }

  /**
     @short getters
  */
  uint32_t raw()  const      { return value_;                   }
  bool threshold() const     { return ((value_ >> 31) & 0x1 );  }
  bool mode() const          { return ((value_ >> 30) & 0x1 );  }
  uint32_t toa()  const      { return ((value_ >> 16) & 0x3ff); }
  uint32_t data()  const     { return ((value_ >> 0)  & 0xfff); }
  uint32_t operator()()      { return value_;                   }
  
private:

  /**
     @short wrapper to reset words at a given position
   */
  void setWord(uint32_t word, uint32_t mask, uint32_t pos)
  {
    //clear required bits
    value_ &= ~((word & mask) << pos); 
    //now set the new value
    value_ |= ((word & mask) << pos);
  }

  // a 32-bit word
  uint32_t value_;
};

  
#endif
