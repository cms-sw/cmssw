#ifndef DataFormats_Provenance_SelectedProducts_h
#define DataFormats_Provenance_SelectedProducts_h

#include "boost/array.hpp"
#include <vector>

#include "DataFormats/Provenance/interface/BranchType.h"

namespace edm {
  class BranchDescription;
  typedef std::vector<BranchDescription const *> SelectedProducts;
  typedef boost::array<SelectedProducts, NumBranchTypes> SelectedProductsForBranchType;
}

#endif
