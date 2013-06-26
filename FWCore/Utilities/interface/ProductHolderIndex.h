#ifndef FWCore_Utilities_ProductHolderIndex_h
#define FWCore_Utilities_ProductHolderIndex_h

#include <limits>

namespace edm {

  typedef unsigned int ProductHolderIndex;

#ifndef __GCCXML__
  enum ProductHolderIndexValues {
    ProductHolderIndexInvalid = std::numeric_limits<unsigned int>::max(),
    ProductHolderIndexInitializing = std::numeric_limits<unsigned int>::max() - 1,
    ProductHolderIndexAmbiguous = std::numeric_limits<unsigned int>::max() - 2
  };
#endif
}
#endif
