// Tools for unpacking and packing EMTF data

#ifndef EMTFUnpackerTools_h
#define EMTFUnpackerTools_h

// Generally useful includes
#include <iostream>
#include <iomanip>  // For things like std::setw

namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      inline int PowInt(int base, int exp) {
	if (exp == 0) return 1;
	if (exp == 1) return base;
	  return base * PowInt(base, exp-1);
      }
      
      inline uint16_t GetHexBits(uint16_t word, uint16_t lowBit, uint16_t highBit) {
	return ( (word >> lowBit) & (PowInt(2, (1 + highBit - lowBit)) - 1) );
      }
      
      inline uint32_t GetHexBits(uint32_t word, uint32_t lowBit, uint32_t highBit) {
	return ( (word >> lowBit) & (PowInt(2, (1 + highBit - lowBit)) - 1) );
      }
      
      inline uint32_t GetHexBits(uint16_t word1, uint16_t lowBit1, uint16_t highBit1, 
				 uint16_t word2, uint16_t lowBit2, uint16_t highBit2) {
	uint16_t word1_sel = (word1 >> lowBit1) & (PowInt(2, (1 + highBit1 - lowBit1)) - 1);
	uint16_t word2_sel = (word2 >> lowBit2) & (PowInt(2, (1 + highBit2 - lowBit2)) - 1);
	return ( (word2_sel << (1 + highBit1 - lowBit1)) | word1_sel );
      }

    } // End namespace emtf
  } // End namespace stage2
} // End namespace l1t

#endif /* define EMTFUnpackerTools_h */
