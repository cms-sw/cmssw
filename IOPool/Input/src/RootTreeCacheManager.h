#ifndef IOPool_Input_RootTreeCacheManager_h
#define IOPool_Input_RootTreeCacheManager_h

#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "FWCore/Utilities/interface/BranchType.h"

#include <memory>

#include "Rtypes.h"

class TTreeCache;
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

      virtual void setCacheSize(unsigned int cacheSize) = 0;
      virtual void setEntryNumber(EntryNumber theEntryNumber, EntryNumber entryNumber, EntryNumber entries) = 0;

      virtual void resetTraining() {}
      virtual void reset() {}
      virtual void SetCacheRead(TTreeCache* cache = nullptr);
      virtual void trainCache(char const* branchNames) {}
      virtual void init(TTree* tree, unsigned int treeAutoFlush) { tree_ = tree; }
      virtual void reserve(Int_t branchCount) {}
      virtual void getEntry(TBranch* branch, EntryNumber entryNumber);

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
