/*----------------------------------------------------------------------
  
ProductIDToBranchID: Free functions to map between ProductIDs and BranchIDs

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ProductIDToBranchID.h"

namespace edm {

  BranchID
  productIDToBranchID(ProductID const& pid, BranchIDLists const& lists, BranchListIndexes const& indexes) {

    if (pid.isValid()) {
      size_t procIndex = pid.processIndex()-1;
      if (procIndex < indexes.size()) {
        BranchListIndex blix = indexes[procIndex];
        if (blix < lists.size()) {
          BranchIDList const& blist = lists[blix];
          size_t prodIndex =pid.productIndex()-1;
          if (prodIndex<blist.size()) {
            BranchID::value_type bid = blist[prodIndex];
            return BranchID(bid);
          }
        }
      }
    }
    return BranchID();
  }
}
