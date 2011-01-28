#ifndef IOPool_Input_Inputfwd_h
#define IOPool_Input_Inputfwd_h

#include "DataFormats/Provenance/interface/ConstBranchDescription.h"

#include "Rtypes.h"

#include <map>

class TBranch;
class TClass;
class TFile;
class TTree;
class TTreeCache;

namespace edm {
  struct BranchKey;
  class FileFormatVersion;
  class RootDelayedReader;
  class RootFile;
  class RootTree;
  namespace input {
    unsigned int const defaultCacheSize = 20U * 1024 * 1024;
    unsigned int const defaultNonEventCacheSize = 1U * 1024 * 1024;
    unsigned int const defaultLearningEntries = 20U;
    unsigned int const defaultNonEventLearningEntries = 1U;
    struct BranchInfo {
      BranchInfo(ConstBranchDescription const& prod) :
        branchDescription_(prod),
        productBranch_(0),
        provenanceBranch_(0),
        classCache_(0),
        offsetToEDProduct_(0) {}
      ConstBranchDescription branchDescription_;
      TBranch* productBranch_;
      TBranch* provenanceBranch_; // For backward compatibility
      mutable TClass* classCache_;
      mutable Int_t offsetToEDProduct_;
    };
    typedef std::map<BranchKey const, BranchInfo> BranchMap;
    typedef Long64_t EntryNumber;
    Int_t getEntry(TBranch* branch, EntryNumber entryNumber);
    Int_t getEntry(TTree* tree, EntryNumber entryNumber);
    Int_t getEntryWithCache(TBranch* branch, EntryNumber entryNumber, TTreeCache* treeCache, TFile* filePtr);
    Int_t getEntryWithCache(TTree* tree, EntryNumber entryNumber, TTreeCache* treeCache, TFile* filePtr);
  }
}
#endif
