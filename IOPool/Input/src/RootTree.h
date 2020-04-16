#ifndef IOPool_Input_RootTree_h
#define IOPool_Input_RootTree_h

/*----------------------------------------------------------------------

RootTree.h // used by ROOT input sources

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputType.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "Rtypes.h"
#include "TBranch.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

class TBranch;
class TClass;
class TTree;
class TTreeCache;

namespace edm {
  class BranchKey;
  class RootDelayedReader;
  class InputFile;
  class RootTree;

  class StreamContext;
  class ModuleCallingContext;

  namespace signalslot {
    template <typename T>
    class Signal;
  }

  namespace roottree {
    unsigned int const defaultCacheSize = 20U * 1024 * 1024;
    unsigned int const defaultNonEventCacheSize = 1U * 1024 * 1024;
    unsigned int const defaultLearningEntries = 20U;
    unsigned int const defaultNonEventLearningEntries = 1U;
    typedef IndexIntoFile::EntryNumber_t EntryNumber;
    struct BranchInfo {
      BranchInfo(BranchDescription const& prod)
          : branchDescription_(prod),
            productBranch_(nullptr),
            provenanceBranch_(nullptr),
            classCache_(nullptr),
            offsetToWrapperBase_(0) {}
      BranchDescription const branchDescription_;
      TBranch* productBranch_;
      TBranch* provenanceBranch_;  // For backward compatibility
      //All access to a ROOT file is serialized
      CMS_SA_ALLOW mutable TClass* classCache_;
      CMS_SA_ALLOW mutable Int_t offsetToWrapperBase_;
    };

    class BranchMap {
      enum {
        kKeys,
        kInfos,
      };

    public:
      void reserve(size_t iSize) { map_.reserve(iSize); }
      void insert(edm::BranchID const& iKey, BranchInfo const& iInfo) { map_.emplace(iKey.id(), iInfo); }
      BranchInfo const* find(BranchID const& iKey) const {
        auto itFound = map_.find(iKey.id());
        if (itFound == map_.end()) {
          return nullptr;
        }
        return &itFound->second;
      }
      BranchInfo* find(BranchID const& iKey) {
        auto itFound = map_.find(iKey.id());
        if (itFound == map_.end()) {
          return nullptr;
        }
        return &itFound->second;
      }

    private:
      std::unordered_map<unsigned int, BranchInfo> map_;
    };

    Int_t getEntry(TBranch* branch, EntryNumber entryNumber);
    Int_t getEntry(TTree* tree, EntryNumber entryNumber);
    std::unique_ptr<TTreeCache> trainCache(TTree* tree,
                                           InputFile& file,
                                           unsigned int cacheSize,
                                           char const* branchNames);
  }  // namespace roottree

  class RootTree {
  public:
    typedef roottree::BranchMap BranchMap;
    typedef roottree::EntryNumber EntryNumber;
    RootTree(std::shared_ptr<InputFile> filePtr,
             BranchType const& branchType,
             unsigned int nIndexes,
             unsigned int maxVirtualSize,
             unsigned int cacheSize,
             unsigned int learningEntries,
             bool enablePrefetching,
             InputType inputType);
    ~RootTree();

    RootTree(RootTree const&) = delete;             // Disallow copying and moving
    RootTree& operator=(RootTree const&) = delete;  // Disallow copying and moving

    bool isValid() const;
    void numberOfBranchesToAdd(size_t iSize) { branches_.reserve(iSize); }
    void addBranch(BranchDescription const& prod, std::string const& oldBranchName);
    void dropBranch(std::string const& oldBranchName);
    void getEntry(TBranch* branch, EntryNumber entry) const;
    void setPresence(BranchDescription& prod, std::string const& oldBranchName);

    bool next() { return ++entryNumber_ < entries_; }
    bool nextWithCache();
    bool previous() { return --entryNumber_ >= 0; }
    bool current() const { return entryNumber_ < entries_ && entryNumber_ >= 0; }
    bool current(EntryNumber entry) const { return entry < entries_ && entry >= 0; }
    void rewind() { entryNumber_ = 0; }
    void close();
    bool skipEntries(unsigned int& offset);
    EntryNumber const& entryNumber() const { return entryNumber_; }
    EntryNumber const& entryNumberForIndex(unsigned int index) const;
    EntryNumber const& entries() const { return entries_; }
    void setEntryNumber(EntryNumber theEntryNumber);
    void insertEntryForIndex(unsigned int index);
    std::vector<std::string> const& branchNames() const { return branchNames_; }
    DelayedReader* rootDelayedReader() const;
    DelayedReader* resetAndGetRootDelayedReader() const;
    template <typename T>
    void fillAux(T*& pAux) {
      auxBranch_->SetAddress(&pAux);
      getEntry(auxBranch_, entryNumber_);
    }
    template <typename T>
    void fillBranchEntryMeta(TBranch* branch, T*& pbuf) {
      if (metaTree_ != nullptr) {
        // Metadata was in separate tree.  Not cached.
        branch->SetAddress(&pbuf);
        roottree::getEntry(branch, entryNumber_);
      } else {
        fillBranchEntry<T>(branch, pbuf);
      }
    }

    template <typename T>
    void fillBranchEntry(TBranch* branch, T*& pbuf) {
      branch->SetAddress(&pbuf);
      getEntry(branch, entryNumber_);
    }

    template <typename T>
    void fillBranchEntryMeta(TBranch* branch, EntryNumber entryNumber, T*& pbuf) {
      if (metaTree_ != nullptr) {
        // Metadata was in separate tree.  Not cached.
        branch->SetAddress(&pbuf);
        roottree::getEntry(branch, entryNumber);
      } else {
        fillBranchEntry<T>(branch, entryNumber, pbuf);
      }
    }

    template <typename T>
    void fillBranchEntry(TBranch* branch, EntryNumber entryNumber, T*& pbuf) {
      branch->SetAddress(&pbuf);
      getEntry(branch, entryNumber);
    }

    TTree const* tree() const { return tree_; }
    TTree* tree() { return tree_; }
    TTree const* metaTree() const { return metaTree_; }
    BranchMap const& branches() const;

    //For backwards compatibility
    TBranch* branchEntryInfoBranch() const { return branchEntryInfoBranch_; }

    inline TTreeCache* checkTriggerCache(TBranch* branch, EntryNumber entryNumber) const;
    TTreeCache* checkTriggerCacheImpl(TBranch* branch, EntryNumber entryNumber) const;
    inline TTreeCache* selectCache(TBranch* branch, EntryNumber entryNumber) const;
    void trainCache(char const* branchNames);
    void resetTraining() { trainNow_ = true; }

    BranchType branchType() const { return branchType_; }

    void setSignals(
        signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadSource,
        signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadSource);

  private:
    void setCacheSize(unsigned int cacheSize);
    void setTreeMaxVirtualSize(int treeMaxVirtualSize);
    void startTraining();
    void stopTraining();

    std::shared_ptr<InputFile> filePtr_;
    // We use bare pointers for pointers to some ROOT entities.
    // Root owns them and uses bare pointers internally.
    // Therefore,using smart pointers here will do no good.
    TTree* tree_;
    TTree* metaTree_;
    BranchType branchType_;
    TBranch* auxBranch_;
    // We use a smart pointer to own the TTreeCache.
    // Unfortunately, ROOT owns it when attached to a TFile, but not after it is detached.
    // So, we make sure to it is detached before closing the TFile so there is no double delete.
    std::shared_ptr<TTreeCache> treeCache_;
    std::shared_ptr<TTreeCache> rawTreeCache_;
    //All access to a ROOT file is serialized
    CMS_SA_ALLOW mutable std::shared_ptr<TTreeCache> triggerTreeCache_;
    CMS_SA_ALLOW mutable std::shared_ptr<TTreeCache> rawTriggerTreeCache_;
    CMS_SA_ALLOW mutable std::unordered_set<TBranch*> trainedSet_;
    CMS_SA_ALLOW mutable std::unordered_set<TBranch*> triggerSet_;
    EntryNumber entries_;
    EntryNumber entryNumber_;
    std::unique_ptr<std::vector<EntryNumber> > entryNumberForIndex_;
    std::vector<std::string> branchNames_;
    BranchMap branches_;
    bool trainNow_;
    EntryNumber switchOverEntry_;
    CMS_SA_ALLOW mutable EntryNumber rawTriggerSwitchOverEntry_;
    CMS_SA_ALLOW mutable bool performedSwitchOver_;
    unsigned int learningEntries_;
    unsigned int cacheSize_;
    unsigned long treeAutoFlush_;
    // Enable asynchronous I/O in ROOT (done in a separate thread).  Only takes
    // effect on the primary treeCache_; all other caches have this explicitly disabled.
    bool enablePrefetching_;
    bool enableTriggerCache_;
    std::unique_ptr<RootDelayedReader> rootDelayedReader_;

    TBranch* branchEntryInfoBranch_;  //backwards compatibility
    // below for backward compatibility
    TTree* infoTree_;        // backward compatibility
    TBranch* statusBranch_;  // backward compatibility
  };
}  // namespace edm
#endif
