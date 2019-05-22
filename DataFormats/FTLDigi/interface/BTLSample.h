#ifndef DIGIFTL_BTLSAMPLE_H
#define DIGIFTL_BTLSAMPLE_H

#include <iostream>
#include <ostream>
#include <cstdint>

/**
   @class BTLSample
   @short wrapper for a data word
 */

class BTLSample {

public:

  enum BTLSampleDataMasks { kToA2Mask = 0x3ff, kToAMask = 0x3ff, kDataMask = 0x3ff};
  enum BTLSampleDataShifts{ kToA2Shift = 20, kToAShift = 10, kDataShift = 0};

  enum BTLSampleFlagMasks { kThreshMask = 0x1, kModeMask = 0x1};
  enum BTLSampleFlagShifts{ kThreshShift = 1, kModeShift = 0};

  /**
     @short CTOR
  */
 BTLSample() : value_(0), flag_(0), row_(0), col_(0) { }
 BTLSample(uint32_t value, uint16_t flag, uint8_t row, uint8_t col) : value_(value), flag_(flag), row_(row), col_(col) { }
 BTLSample( const BTLSample& o ) : value_(o.value_), flag_(o.flag_), row_(o.row_), col_(o.col_) { }
  
  /**
     @short setters
  */
  void setThreshold(bool thr)           { setFlagWord(thr,  kThreshMask, kThreshShift); }
  void setMode(bool mode)               { setFlagWord(mode, kModeMask,   kModeShift);   }
  void setToA(uint16_t toa)             { setDataWord(toa,  kToAMask,    kToAShift);    }
  void setToA2(uint16_t toa2)           { setDataWord(toa2, kToA2Mask,   kToA2Shift);   }
  void setData(uint16_t data)           { setDataWord(data, kDataMask,   kDataShift);   }
  void set(bool thr, bool mode, uint16_t toa2, uint16_t toa, uint16_t data, uint8_t row, uint8_t col) 
  { 
    flag_  = ( ( (uint16_t)thr  & kThreshMask ) << kThreshShift | 
               ( (uint16_t)mode & kModeMask   ) << kModeShift     );    

    value_ = ( ( (uint32_t)toa2 & kToA2Mask   ) << kToA2Shift   | 
               ( (uint32_t)toa  & kToAMask    ) << kToAShift    | 
               ( (uint32_t)data & kDataMask   ) << kDataShift     );    
    row_ = row;
    col_ = col;
  }  
  void print(std::ostream &out=std::cout)
  {
    out << "THR: " << threshold() 
	<< " Mode: " << mode() 
	<< " ToA2: " << toa2() 
	<< " ToA: " << toa() 
	<< " Data: " << data() 
	<< " Row: " << (uint32_t)row() 
	<< " Column: " << (uint32_t)column() 
	<< " Raw Flag=0x" << std::hex << raw_flag() << std::dec
	<< " Raw Data=0x" << std::hex << raw_data() << std::dec << std::endl;  
        
  }

  /**
     @short getters
  */
  uint32_t raw_data()  const      { return value_;                   }
  uint16_t raw_flag()  const      { return flag_;                    }
  bool threshold() const     { return ( (flag_  >> kThreshShift) & kThreshMask ); }
  bool mode() const          { return ( (flag_  >> kModeShift)   & kModeMask   ); }
  uint32_t toa()  const      { return ( (value_ >> kToAShift)    & kToAMask    ); }
  uint32_t toa2() const      { return ( (value_ >> kToA2Shift)   & kToA2Mask   ); }
  uint32_t data() const      { return ( (value_ >> kDataShift)   & kDataMask   ); }
  uint8_t row() const { return row_; } 
  uint8_t column() const { return col_; }
  
private:

  /**
     @short wrapper to reset words at a given position
   */
  void setDataWord(uint32_t word, uint32_t mask, uint32_t pos)
  {
    //clear required bits
    value_ &= ~(mask << pos);
    //now set the new value
    value_ |= ( (word & mask) << pos );
  }
  void setFlagWord(uint16_t word, uint16_t mask, uint16_t pos)
  {
    //clear required bits
    flag_ &= ~(mask << pos);
    //now set the new value
    flag_ |= ( (word & mask) << pos );
  }

  // bit-words for data and flags
  uint32_t value_;
  uint16_t flag_;
  uint8_t row_,col_;
};

  
#endif
