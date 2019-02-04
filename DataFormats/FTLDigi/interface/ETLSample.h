#ifndef DIGIFTL_ETLSAMPLE_H
#define DIGIFTL_ETLSAMPLE_H

#include <iostream>
#include <ostream>
#include <cstdint>

/**
   @class ETLSample
   @short wrapper for a data word
 */

class ETLSample {

public:

  enum ETLSampleMasks { kThreshMask = 0x1, kModeMask = 0x1, kColumnMask = 0x1f, 
			kRowMask = 0x3f, kToAMask = 0x7ff, kDataMask = 0xff };
  enum ETLSampleShifts{ kThreshShift = 31, kModeShift = 30, kColumnShift = 25, 
			kRowShift = 19, kToAShift = 8,   kDataShift = 0 };

  /**
     @short CTOR
   */
 ETLSample() : value_(0) { }
 ETLSample(uint32_t value) : value_(value) { }
 ETLSample( const ETLSample& o ) : value_(o.value_) { }

  /**
     @short setters
   */
  void setThreshold(bool thr)           { setWord(thr,  kThreshMask, kThreshShift); }
  void setMode(bool mode)               { setWord(mode, kModeMask,   kModeShift);   }
  void setColumn(uint8_t col)           { setWord(col, kColumnMask, kColumnShift);  }
  void setRow(uint8_t row)              { setWord(row, kRowMask, kRowShift);        }
  void setToA(uint16_t toa)             { setWord(toa,  kToAMask,    kToAShift);    }
  void setData(uint16_t data)           { setWord(data, kDataMask,   kDataShift);   }
  void set(bool thr, bool mode, uint16_t toa, uint16_t data, uint8_t row, uint8_t col) 
  { 
    value_ = ( ( (uint32_t)thr  & kThreshMask ) << kThreshShift | 
               ( (uint32_t)mode & kModeMask   ) << kModeShift   |
	       ( (uint32_t)col  & kColumnMask ) << kColumnShift |
	       ( (uint32_t)row  & kRowMask    ) << kRowShift    |
               ( (uint32_t)toa  & kToAMask    ) << kToAShift    | 
               ( (uint32_t)data & kDataMask   ) << kDataShift     );    
  }  
  void print(std::ostream &out=std::cout)
  {
    out << "(row,col) : (" << row() << ',' << column() << ") "
	<< "THR: " << threshold() 
	<< " Mode: " << mode() 
	<< " ToA: " << toa() 
	<< " Data: " << data() 
	<< " Raw=0x" << std::hex << raw() << std::dec << std::endl;  
  }

  /**
     @short getters
  */
  uint32_t raw()  const      { return value_;                   }
  bool threshold() const     { return ( (value_ >> kThreshShift) & kThreshMask ); }
  bool mode() const          { return ( (value_ >> kModeShift)   & kModeMask   ); }
  uint32_t column() const    { return ( (value_ >> kColumnShift) & kColumnMask ); }
  uint32_t row() const       { return ( (value_ >> kRowShift)    & kRowMask    ); }
  uint32_t toa()  const      { return ( (value_ >> kToAShift)    & kToAMask    ); }
  uint32_t data()  const     { return ( (value_ >> kDataShift)   & kDataMask   ); }
  uint32_t operator()()      { return value_;                   }
  
private:

  /**
     @short wrapper to reset words at a given position
   */
  void setWord(uint32_t word, uint32_t mask, uint32_t pos)
  {
    //clear required bits
    value_ &= ~(mask << pos);
    //now set the new value
    value_ |= ( (word & mask) << pos );
  }

  // a 32-bit word
  uint32_t value_;
};

  
#endif
