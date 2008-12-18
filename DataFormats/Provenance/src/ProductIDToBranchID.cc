/*----------------------------------------------------------------------
  
ProductIDToBranchID: Free functions to map between ProductIDs and BranchIDs

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ProductIDToBranchID.h"

namespace edm {

  BranchID
  productIDToBranchID(ProductID const& pid, BranchIDLists const& lists, BranchListIndexes const& indexes) {
    if (!pid.isValid()) {
      return BranchID();
    }
    BranchID::value_type bid = 0;
    try {
      BranchListIndex blix = indexes.at(pid.processIndex()-1);
      BranchIDList const& blist = lists.at(blix);
      bid = blist.at(pid.productIndex()-1);
    }
    catch (std::exception) {
      return BranchID();
    }
    return BranchID(bid);
  }
}
