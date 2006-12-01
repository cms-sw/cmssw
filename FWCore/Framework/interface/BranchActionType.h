#ifndef Framework_BranchActionType_h
#define Framework_BranchActionType_h

/*----------------------------------------------------------------------
  
BranchActionType: BranchAction

$Id: BranchActionType.h,v 1.3 2006/11/07 00:30:43 wmtan Exp $
----------------------------------------------------------------------*/

namespace edm {
  // Note: These enum values are used as subscripts for a fixed size array, so they must not change.
  enum BranchActionType {
    BranchActionEvent = 0,
    BranchActionBeginLumi = 1,
    BranchActionEndLumi = 2,
    BranchActionBeginRun = 3,
    BranchActionEndRun = 4,
    EndBranchActionType = 5
  };
}
#endif
