#ifndef FWCore_Utilities_ProductResolverIndex_h
#define FWCore_Utilities_ProductResolverIndex_h

#include <limits>

namespace edm {

  typedef unsigned int ProductResolverIndex;

  enum ProductResolverIndexValues {

    // All values of the ProductResolverIndex in this enumeration should
    // have this bit set to 1,
    ProductResolverIndexValuesBit = 1U << 30,

    ProductResolverIndexInvalid = std::numeric_limits<unsigned int>::max(),
    ProductResolverIndexInitializing = std::numeric_limits<unsigned int>::max() - 1,
    ProductResolverIndexAmbiguous = std::numeric_limits<unsigned int>::max() - 2
  };
}  // namespace edm
#endif
