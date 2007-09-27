#ifndef IOPool_Input_Inputfwd_h
#define IOPool_Input_Inputfwd_h

#include <map>

#include "Rtypes.h"
#include "Reflex/Type.h"
class TBranch;

#include "DataFormats/Provenance/interface/ConstBranchDescription.h"

namespace edm {
  class BranchKey;
  class RootFile;
  class RootDelayedReader;
  class RootTree;
  namespace input {
    struct EventBranchInfo {
      EventBranchInfo(ConstBranchDescription const& prod) :
        branchDescription_(prod), provenanceBranch_(0), productBranch_(0) {}
      ConstBranchDescription branchDescription_;
      TBranch * provenanceBranch_;
      TBranch * productBranch_;
    };
    typedef std::map<BranchKey const, EventBranchInfo> BranchMap;
    typedef Long64_t EntryNumber;
  }
}
#endif
