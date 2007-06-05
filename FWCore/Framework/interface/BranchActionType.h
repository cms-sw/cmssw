#ifndef Framework_BranchActionType_h
#define Framework_BranchActionType_h

/*----------------------------------------------------------------------
  
BranchActionType: BranchAction

$Id: BranchActionType.h,v 1.1 2006/12/01 03:29:52 wmtan Exp $
----------------------------------------------------------------------*/

namespace edm {
  enum BranchActionType {
    BranchActionEvent = 0,
    BranchActionBegin = 1,
    BranchActionEnd = 2,
    EndBranchActionType = 3
  };
}
#endif
