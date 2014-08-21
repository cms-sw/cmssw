#ifndef IOPool_Input_RootTree_h
#define IOPool_Input_RootTree_h

/*----------------------------------------------------------------------

RootTree.h // used by ROOT input sources

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputType.h"

#include "Rtypes.h"
#include "TBranch.h"

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

class TBranch;
class TClass;
class TTree;
class TTreeCache;

namespace edm {
  class BranchKey;
  class DelayedReader;
  class InputFile;
  class RootTree;

  namespace roottree {
    unsigned int const defaultCacheSize = 20U * 1024 * 1024;
    unsigned int const defaultNonEventCacheSize = 1U * 1024 * 1024;
    unsigned int const defaultLearningEntries = 20U;
    unsigned int const defaultNonEventLearningEntries = 1U;
    typedef IndexIntoFile::EntryNumber_t EntryNumber;
    struct BranchInfo {
      BranchInfo(BranchDescription const& prod) :
        branchDescription_(prod),
        productBranch_(nullptr),
        provenanceBranch_(nullptr),
        classCache_(nullptr),
        offsetToWrapperBase_(0) {}
      BranchDescription const branchDescription_;
      TBranch* productBranch_;
      TBranch* provenanceBranch_; // For backward compatibility
      mutable TClass* classCache_;
      mutable Int_t offsetToWrapperBase_;
    };
    typedef std::map<BranchKey const, BranchInfo> BranchMap;
    Int_t getEntry(TBranch* branch, EntryNumber entryNumber);
    Int_t getEntry(TTree* tree, EntryNumber entryNumber);
    std::unique_ptr<TTreeCache> trainCache(TTree* tree, InputFile& file, unsigned int cacheSize, char const* branchNames);
  }

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

    RootTree(RootTree const&) = delete; // Disallow copying and moving
    RootTree& operator=(RootTree const&) = delete; // Disallow copying and moving

    bool isValid() const;
    void addBranch(BranchKey const& key,
                   BranchDescription const& prod,
                   std::string const& oldBranchName);
    void dropBranch(std::string const& oldBranchName);
    void getEntry(TBranch *branch, EntryNumber entry) const;
    void setPresence(BranchDescription& prod,
                   std::string const& oldBranchName);

    bool next() {return ++entryNumber_ < entries_;}
    bool previous() {return --entryNumber_ >= 0;}
    bool current() const {return entryNumber_ < entries_ && entryNumber_ >= 0;}
    bool current(EntryNumber entry) const {return entry < entries_ && entry >= 0;}
    void rewind() {entryNumber_ = 0;}
    void close();
    EntryNumber const& entryNumber() const {return entryNumber_;}
    EntryNumber const& entryNumberForIndex(unsigned int index) const;
    EntryNumber const& entries() const {return entries_;}
    void setEntryNumber(EntryNumber theEntryNumber);
    void insertEntryForIndex(unsigned int index);
    std::vector<std::string> const& branchNames() const {return branchNames_;}
    DelayedReader* rootDelayedReader() const;
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
    
    TTree const* tree() const {return tree_;}
    TTree* tree() {return tree_;}
    TTree const* metaTree() const {return metaTree_;}
    BranchMap const& branches() const;

    //For backwards compatibility
    TBranch* branchEntryInfoBranch() const {return branchEntryInfoBranch_;}

    inline TTreeCache* checkTriggerCache(TBranch* branch, EntryNumber entryNumber) const;
    TTreeCache* checkTriggerCacheImpl(TBranch* branch, EntryNumber entryNumber) const;
    inline TTreeCache* selectCache(TBranch* branch, EntryNumber entryNumber) const;
    void trainCache(char const* branchNames);
    void resetTraining() {trainNow_ = true;}

    BranchType branchType() const {return branchType_;}
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
    mutable std::shared_ptr<TTreeCache> triggerTreeCache_;
    mutable std::shared_ptr<TTreeCache> rawTriggerTreeCache_;
    mutable std::unordered_set<TBranch*> trainedSet_;
    mutable std::unordered_set<TBranch*> triggerSet_;
    EntryNumber entries_;
    EntryNumber entryNumber_;
    std::unique_ptr<std::vector<EntryNumber> > entryNumberForIndex_;
    std::vector<std::string> branchNames_;
    std::shared_ptr<BranchMap> branches_;
    bool trainNow_;
    EntryNumber switchOverEntry_;
    mutable EntryNumber rawTriggerSwitchOverEntry_;
    mutable bool performedSwitchOver_;
    unsigned int learningEntries_;
    unsigned int cacheSize_;
    unsigned long treeAutoFlush_;
// Enable asynchronous I/O in ROOT (done in a separate thread).  Only takes
// effect on the primary treeCache_; all other caches have this explicitly disabled.
    bool enablePrefetching_;
    bool enableTriggerCache_;
    std::unique_ptr<DelayedReader> rootDelayedReader_;

    TBranch* branchEntryInfoBranch_; //backwards compatibility
    // below for backward compatibility
    TTree* infoTree_; // backward compatibility
    TBranch* statusBranch_; // backward compatibility
  };
}
#endif
