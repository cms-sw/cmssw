#ifndef IOPool_Input_Inputfwd_h
#define IOPool_Input_Inputfwd_h

#include <map>

#include "Rtypes.h"
#include "Reflex/Type.h"
class TBranch;

#include "DataFormats/Provenance/interface/ConstBranchDescription.h"

namespace edm {
  class BranchKey;
  class FileFormatVersion;
  class RootFile;
  class RootDelayedReader;
  class RootTree;
  namespace input {
    struct EventBranchInfo {
      EventBranchInfo(ConstBranchDescription const& prod) :
        branchDescription_(prod),
	productBranch_(0),
	provenanceBranch_(0) {}
      ConstBranchDescription branchDescription_;
      TBranch * productBranch_;
      // The rest are for backward compatibility
      TBranch * provenanceBranch_;
    };
    typedef std::map<BranchKey const, EventBranchInfo> BranchMap;
    typedef Long64_t EntryNumber;
  }
}
#endif
