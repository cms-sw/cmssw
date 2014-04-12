#ifndef Framework_BranchActionType_h
#define Framework_BranchActionType_h

/*----------------------------------------------------------------------
  
BranchActionType: BranchAction

----------------------------------------------------------------------*/

namespace edm {
  enum BranchActionType {
    BranchActionGlobalBegin = 0,
    BranchActionStreamBegin = 1,
    BranchActionStreamEnd = 2,
    BranchActionGlobalEnd = 3
  };
}
#endif
