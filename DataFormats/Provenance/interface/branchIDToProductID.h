#ifndef DataFormats_Provenance_branchIDToProductID_h
#define DataFormats_Provenance_branchIDToProductID_h

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include <vector>

namespace edm {
  // Fill in helper map for Branch to ProductID mapping
  std::vector<ProcessIndex> makeBranchListIndexToProcessIndex(BranchListIndexes const& branchListIndexes);

  ProductID branchIDToProductID(BranchID const& bid,
                                BranchIDListHelper const& branchIDListHelper,
                                std::vector<ProcessIndex> const& branchListIndexToProcessIndex);
}  // namespace edm

#endif
