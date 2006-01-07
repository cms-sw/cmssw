#ifndef Input_Inputfwd_h
#define Input_Inputfwd_h

#include <string>
#include <map>

#include "TBranch.h"

namespace edm {
  class BranchKey;
  class RootFile;
  class RootDelayedReader;
  namespace input {
    typedef std::map<BranchKey, std::pair<std::string, TBranch *> > BranchMap;
    typedef Long64_t EntryNumber;
  }
}
#endif
