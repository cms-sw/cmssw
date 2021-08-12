#ifndef IOPool_Input_RootTreeCacheManager_h
#define IOPool_Input_RootTreeCacheManager_h
// -*- C++ -*-
//
// Package:     IOPool/Input
// Class  :     roottree::CacheManagerBase
//
/**\class roottree::CacheManagerBase RootTreeCacheManager.h IOPool/Input/src/RootTreeCacheManager.h

 Description: Base class for TTreeCache policy implementations

 Usage:
    <usage>

*/

#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "FWCore/Utilities/interface/BranchType.h"

#include <memory>

class TFileCacheRead;
class TBranch;
class TTree;

namespace edm {
  class InputFile;

  namespace roottree {
    using EntryNumber = IndexIntoFile::EntryNumber_t;

    class CacheManagerBase {
    public:
      CacheManagerBase(std::shared_ptr<InputFile> filePtr) : filePtr_(filePtr) {}
      virtual ~CacheManagerBase() = default;

      virtual void createPrimaryCache(unsigned int cacheSize) = 0;
      virtual void setEntryNumber(EntryNumber theEntryNumber, EntryNumber entryNumber, EntryNumber entries) = 0;

      virtual void resetTraining() {}
      virtual void reset() {}
      virtual void trainCache(char const* branchNames) {}
      virtual void init(TTree* tree, unsigned int treeAutoFlush) { tree_ = tree; }
      virtual void reserve(Int_t branchCount) {}
      virtual void getEntry(TBranch* branch, EntryNumber entryNumber);
      virtual void getAuxEntry(TBranch* auxBranch, EntryNumber entryNumber);
      virtual void getEntryForAllBranches(EntryNumber entryNumber) const = 0;

      static std::unique_ptr<CacheManagerBase> create(const std::string& strategy,
                                                      std::shared_ptr<InputFile> filePtr,
                                                      unsigned int learningEntries,
                                                      bool enablePrefetching,
                                                      BranchType const& branchType);

    protected:
      std::shared_ptr<InputFile> filePtr_;
      TTree* tree_ = nullptr;
    };

  }  // namespace roottree
}  // namespace edm
#endif
