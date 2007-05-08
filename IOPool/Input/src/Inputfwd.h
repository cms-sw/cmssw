#ifndef Input_Inputfwd_h
#define Input_Inputfwd_h

#include <map>

#include "TBranch.h"
#include "Reflex/Type.h"

namespace edm {
  class BranchKey;
  class RootFile;
  class RootDelayedReader;
  namespace input {
    typedef std::map<BranchKey const, std::pair<ROOT::Reflex::Type, TBranch *const> > BranchMap;
    typedef Long64_t EntryNumber;
  }
}
#endif
