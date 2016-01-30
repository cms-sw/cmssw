// Tools for unpacking and packing EMTF data

// Generally useful includes
#include <iostream>
#include <iomanip>  // For things like std::setw

namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      class EMTFUnpackerTools { 
      public:

	int powInt(int base, int exp) {
	  if (exp == 0) return 1;
	  if (exp == 1) return base;
	  return base * powInt(base, exp-1);
	}

	int GetHexBits(uint64_t word, int lowBit, int highBit) {
	  return ( (word >> lowBit) & (powInt(2, (1 + highBit - lowBit)) - 1) );
	}

      }; // End class EMTFUnpackerTools

    } // End namespace emtf
  } // End namespace stage2
} // End namespace l1t
