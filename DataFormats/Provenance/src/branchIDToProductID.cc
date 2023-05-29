#include "DataFormats/Provenance/interface/branchIDToProductID.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>

namespace edm {
  std::vector<ProcessIndex> makeBranchListIndexToProcessIndex(BranchListIndexes const& branchListIndexes) {
    ProcessIndex pix = 0;
    auto const nelem = 1 + *std::max_element(branchListIndexes.begin(), branchListIndexes.end());
    std::vector<ProcessIndex> branchListIndexToProcessIndex(nelem, std::numeric_limits<BranchListIndex>::max());
    for (auto const& blindex : branchListIndexes) {
      branchListIndexToProcessIndex[blindex] = pix;
      ++pix;
    }
    return branchListIndexToProcessIndex;
  }

  ProductID branchIDToProductID(BranchID const& bid,
                                BranchIDListHelper const& branchIDListHelper,
                                std::vector<ProcessIndex> const& branchListIndexToProcessIndex) {
    if (not bid.isValid()) {
      throw Exception(errors::NotFound, "InvalidID") << "branchIDToProductID: invalid BranchID supplied\n";
    }

    auto range = branchIDListHelper.branchIDToIndexMap().equal_range(bid);
    for (auto it = range.first; it != range.second; ++it) {
      edm::BranchListIndex blix = it->second.first;
      if (blix < branchListIndexToProcessIndex.size()) {
        auto v = branchListIndexToProcessIndex[blix];
        if (v != std::numeric_limits<edm::BranchListIndex>::max()) {
          edm::ProductIndex productIndex = it->second.second;
          edm::ProcessIndex processIndex = v;
          return edm::ProductID(processIndex + 1, productIndex + 1);
        }
      }
    }
    // cannot throw, because some products may legitimately not have product ID's (e.g. pile-up).
    return edm::ProductID();
  }

}  // namespace edm
