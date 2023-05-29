#ifndef FWCore_Framework_TypeMatch_h
#define FWCore_Framework_TypeMatch_h

/** \class edm::TypeMatch

This is intended to be used with the class GetterOfProducts.
GetterOfProducts already forces matching on type. This guarantees that
 no duplicates for the same underlying data product are matched. Such
duplication can occur when an EDAlias is involved.

\author C Jones, created 14 March, 2023

*/

#include "DataFormats/Provenance/interface/BranchDescription.h"

namespace edm {
  class TypeMatch {
  public:
    bool operator()(edm::BranchDescription const& branchDescription) const {
      return not branchDescription.isAnyAlias();
    }
  };
}  // namespace edm
#endif
