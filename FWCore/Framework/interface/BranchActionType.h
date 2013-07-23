#ifndef Framework_BranchActionType_h
#define Framework_BranchActionType_h

/*----------------------------------------------------------------------
  
BranchActionType: BranchAction

$Id: BranchActionType.h,v 1.2 2007/06/05 04:02:30 wmtan Exp $
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
