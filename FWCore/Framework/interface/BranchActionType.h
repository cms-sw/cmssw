#ifndef Framework_BranchActionType_h
#define Framework_BranchActionType_h

/*----------------------------------------------------------------------
  
BranchActionType: BranchAction

$Id: BranchActionType.h,v 1.3 2008/10/16 23:06:28 wmtan Exp $
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
