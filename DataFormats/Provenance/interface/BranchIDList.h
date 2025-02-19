#ifndef DataFormats_Provenance_BranchIDList_h
#define DataFormats_Provenance_BranchIDList_h

/*----------------------------------------------------------------------
  
BranchIDList: 
BranchIDLists: 
        one table stored per File
	table BranchIDLists keyed by ProcessInfo::branchListIndex_;
	entry BranchIDList keyed by ProductID::productIndex_;

----------------------------------------------------------------------*/

#include <vector>
#include "DataFormats/Provenance/interface/BranchID.h"

namespace edm {
  typedef std::vector<BranchID::value_type> BranchIDList;
  typedef std::vector<BranchIDList> BranchIDLists;
}
#endif
