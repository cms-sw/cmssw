#ifndef Common_BranchType_h
#define Common_BranchType_h

/*----------------------------------------------------------------------
  
BranchType: The type of a Branch (Event, LuminosityBlock, or Run)

$Id: BranchType.h,v 1.21 2006/09/28 20:35:10 wmtan Exp $
----------------------------------------------------------------------*/

namespace edm {
	enum BranchType {
	  InEvent = 0,
	  InLumi,
	  InRun
	};
}
#endif
