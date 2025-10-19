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

class TTreeCache;
class TBranch;
class TTree;

namespace edm {
  class InputFile;

  namespace roottree {
    using EntryNumber = IndexIntoFile::EntryNumber_t;

    class CacheManagerBase {
    public:
      enum class CacheStrategy {
        kNone,
        kSimple,
        kSimpleWithAuxCache,
        kSparse,
      };

      CacheManagerBase(std::shared_ptr<InputFile> filePtr) : filePtr_(filePtr) {}
      virtual ~CacheManagerBase() = default;

      virtual void createPrimaryCache(unsigned int cacheSize) = 0;
      // set the tree to read at nextEntryNumber; the current entryNumber is used when detecting
      // non-serial reads, especially skipping backwards in the tree
      virtual void setEntryNumber(EntryNumber nextEntryNumber, EntryNumber entryNumber, EntryNumber entries) = 0;

      virtual void resetTraining(bool promptRead = false) {}
      virtual void reset() {}
      virtual void trainCache(char const* branchNames) {}
      virtual void init(TTree* tree, unsigned int treeAutoFlush) { tree_ = tree; }
      virtual void reserve(Int_t branchCount) {}
      virtual void getEntry(TBranch* branch, EntryNumber entryNumber);
      virtual void getAuxEntry(TBranch* auxBranch, EntryNumber entryNumber);
      virtual void getEntryForAllBranches(EntryNumber entryNumber) const;

      static std::unique_ptr<CacheManagerBase> create(CacheStrategy strategy,
                                                      std::shared_ptr<InputFile> filePtr,
                                                      unsigned int learningEntries,
                                                      bool enablePrefetching,
                                                      BranchType const& branchType);

    protected:
      std::shared_ptr<TTreeCache> createCacheWithSize(unsigned int cacheSize);

      std::shared_ptr<InputFile> filePtr_;
      TTree* tree_ = nullptr;

      static constexpr bool cachestats = false;
    };

  }  // namespace roottree
}  // namespace edm
#endif
