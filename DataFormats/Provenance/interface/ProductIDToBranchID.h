#ifndef DataFormats_Provenance_ProductIDToBranchID_h
#define DataFormats_Provenance_ProductIDToBranchID_h

/*----------------------------------------------------------------------
  
ProductIDToBranchID: Free function to map from ProductID to BranchID

----------------------------------------------------------------------*/
#include <iosfwd>

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/ProductID.h"

namespace edm {
  BranchID
  productIDToBranchID(ProductID const& pid, BranchIDLists const& lists, BranchListIndexes const& indexes);
}
#endif
