#ifndef DataFormats_Provenance_BranchListIndex_h
#define DataFormats_Provenance_BranchListIndex_h

/*----------------------------------------------------------------------
  
ProcessInfo: 
ProcessInfoVector: 
	table keyed by ProductID::processIndex_.
	processIndex_ is monotonically increasing with time.
        one table stored per Occurrence(Event. Lumi, Run)

----------------------------------------------------------------------*/

#include <vector>

namespace edm {
  typedef unsigned short BranchListIndex;
  typedef std::vector<BranchListIndex> BranchListIndexes;
}
#endif
