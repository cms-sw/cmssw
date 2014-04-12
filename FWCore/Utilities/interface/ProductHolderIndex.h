#ifndef FWCore_Utilities_ProductHolderIndex_h
#define FWCore_Utilities_ProductHolderIndex_h

#include <limits>

namespace edm {

  typedef unsigned int ProductHolderIndex;

#ifndef __GCCXML__
  enum ProductHolderIndexValues {

    // All values of the ProductHolderIndex in this enumeration should
    // have this bit set to 1, 
    ProductHolderIndexValuesBit = 1U << 30,

    ProductHolderIndexInvalid = std::numeric_limits<unsigned int>::max(),
    ProductHolderIndexInitializing = std::numeric_limits<unsigned int>::max() - 1,
    ProductHolderIndexAmbiguous = std::numeric_limits<unsigned int>::max() - 2
  };
#endif
}
#endif
