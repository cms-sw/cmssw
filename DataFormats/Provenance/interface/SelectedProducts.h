#ifndef DataFormats_Provenance_SelectedProducts_h
#define DataFormats_Provenance_SelectedProducts_h

#include <array>
#include <vector>

#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ProductDescriptionFwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm {
  typedef std::vector<std::pair<ProductDescription const *, EDGetToken>> SelectedProducts;
  typedef std::array<SelectedProducts, NumBranchTypes> SelectedProductsForBranchType;
}  // namespace edm

#endif
