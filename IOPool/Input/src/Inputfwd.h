#ifndef Input_Inputfwd_h
#define Input_Inputfwd_h

#include <map>

#include "TBranch.h"
#include "Rtypes.h"
#include "Reflex/Type.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"

namespace edm {
  class BranchKey;
  class RootFile;
  class RootDelayedReader;
  class RootTree;
  namespace input {
    struct EventBranchInfo {
      BranchDescription branchDescription_;
      ROOT::Reflex::Type type;
      TBranch * provenanceBranch_;
      TBranch * productBranch_;
    };
    typedef std::map<BranchKey const, EventBranchInfo> BranchMap;
    typedef Long64_t EntryNumber;
  }
}
#endif
