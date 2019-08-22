#ifndef DataFormats_Provenance_SelectedProducts_h
#define DataFormats_Provenance_SelectedProducts_h

#include <array>
#include <vector>

#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm {
  class BranchDescription;
  typedef std::vector<std::pair<BranchDescription const *, EDGetToken>> SelectedProducts;
  typedef std::array<SelectedProducts, NumBranchTypes> SelectedProductsForBranchType;
}  // namespace edm

#endif
