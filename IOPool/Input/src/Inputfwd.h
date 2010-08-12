#ifndef IOPool_Input_Inputfwd_h
#define IOPool_Input_Inputfwd_h

#include <map>

#include "Rtypes.h"
class TBranch;
class TFile;
class TTree;
class TTreeCache;
class TClass;

#include "DataFormats/Provenance/interface/ConstBranchDescription.h"

namespace edm {
  class BranchKey;
  class FileFormatVersion;
  class RootFile;
  class RootDelayedReader;
  class RootTree;
  namespace input {
    unsigned int const defaultCacheSize = 20 * 1024 * 1024;
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
      mutable TClass * classCache_;
      mutable Int_t offsetToEDProduct_;
    };
    typedef std::map<BranchKey const, BranchInfo> BranchMap;
    typedef Long64_t EntryNumber;
    Int_t getEntry(TBranch * branch, EntryNumber entryNumber);
    Int_t getEntry(TTree * tree, EntryNumber entryNumber);
    Int_t getEntryWithCache(TBranch * branch, EntryNumber entryNumber, TTreeCache* treeCache, TFile* filePtr);
    Int_t getEntryWithCache(TTree * tree, EntryNumber entryNumber, TTreeCache* treeCache, TFile* filePtr);
  }
}
#endif
