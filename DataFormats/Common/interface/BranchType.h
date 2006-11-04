#ifndef Common_BranchType_h
#define Common_BranchType_h

#include <string>
/*----------------------------------------------------------------------
  
BranchType: The type of a Branch (Event, LuminosityBlock, or Run)

$Id: BranchType.h,v 1.1 2006/11/04 00:35:22 wmtan Exp $
----------------------------------------------------------------------*/

namespace edm {
  enum BranchType {
    InEvent = 0,
    InLumi,
    InRun
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, BranchType const& p) {
    std::string const type = ((p == InEvent) ? "Event" : ((p == InRun) ? "Run" : "LuminosityBlock"));
    os << type;
    return os;
  }

}
#endif
