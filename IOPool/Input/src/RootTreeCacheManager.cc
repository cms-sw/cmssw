#include "InputFile.h"
#include "RootTreeCacheManager.h"

#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TBranch.h"
#include "TTree.h"
#include "TTreeCache.h"
#include "TTreeCacheUnzip.h"

#include <unordered_set>
#include <cassert>

#include "oneapi/tbb/task_arena.h"

namespace edm {
  namespace roottree {
    void CacheManagerBase::getEntry(TBranch* branch, EntryNumber entryNumber) { branch->GetEntry(entryNumber); }
    void CacheManagerBase::getAuxEntry(TBranch* branch, EntryNumber entryNumber) { getEntry(branch, entryNumber); }
    void CacheManagerBase::getEntryForAllBranches(EntryNumber entryNumber) const { tree_->GetEntry(entryNumber); }
    std::shared_ptr<TTreeCache> CacheManagerBase::createCacheWithSize(unsigned int cacheSize) {
      return filePtr_->createCacheWithSize(tree_, cacheSize);
    }

    // policy with no TTreeCache
    class NoCache : public CacheManagerBase {
    public:
      NoCache(std::shared_ptr<InputFile> filePtr) : CacheManagerBase(filePtr) {}
      void createPrimaryCache(unsigned int cacheSize) override {}
      void setEntryNumber(EntryNumber nextEntryNumber, EntryNumber entryNumber, EntryNumber entries) override;
    };

    void NoCache::setEntryNumber(EntryNumber nextEntryNumber, EntryNumber entryNumber, EntryNumber entries) {
      if (nextEntryNumber != entryNumber) {
        tree_->LoadTree(nextEntryNumber);
      }
    }

    //
    // SimpleCache implements a policy that uses the TTreeCache
    // in a straightforward way, with no swapping of Caches.
    // It is intended for use primarily in production-like jobs
    // that read nearly all the branches in a file sequentially,
    // with little or no event selection or other event skipping.
    //
    class SimpleCache : public CacheManagerBase {
    public:
      SimpleCache(std::shared_ptr<InputFile> filePtr,
                  unsigned int learningEntries,
                  bool enablePrefetching,
                  BranchType const& branchType);
      void reset() override;
      void createPrimaryCache(unsigned int cacheSize) override;
      void setEntryNumber(EntryNumber nextEntryNumber, EntryNumber entryNumber, EntryNumber entries) override;
      void trainCache(char const* branchNames) override;
      void resetTraining(bool promptRead) override;
      void getEntry(TBranch* branch, EntryNumber entryNumber) override;
      void getEntryForAllBranches(EntryNumber entryNumber) const override;

    protected:
      std::shared_ptr<TTreeCache> treeCache_;
      BranchType branchType_;
      unsigned int learningEntries_;
      bool enablePrefetching_;
    };

    SimpleCache::SimpleCache(std::shared_ptr<InputFile> filePtr,
                             unsigned int learningEntries,
                             bool enablePrefetching,
                             BranchType const& branchType)
        : CacheManagerBase(filePtr),
          branchType_(branchType),
          learningEntries_(learningEntries),
          enablePrefetching_(enablePrefetching) {}

    void SimpleCache::reset() {
      if (cachestats && treeCache_) {
        treeCache_->Print("a cachedbranches");
      }
      // We own the treeCache_.
      // We make sure the treeCache_ is detached from the file,
      // so that ROOT does not also delete it.
      filePtr_->clearCacheRead(tree_);
      treeCache_.reset();
    }

    void SimpleCache::setEntryNumber(EntryNumber nextEntryNumber, EntryNumber entryNumber, EntryNumber entries) {
      if (nextEntryNumber != entryNumber) {
        auto guard = filePtr_->setCacheReadTemporarily(treeCache_.get(), tree_);
        oneapi::tbb::this_task_arena::isolate([&]() { tree_->LoadTree(nextEntryNumber); });
      }
    }

    void SimpleCache::getEntry(TBranch* branch, EntryNumber entryNumber) {
      auto guard = filePtr_->setCacheReadTemporarily(treeCache_.get(), tree_);
      oneapi::tbb::this_task_arena::isolate([&]() { branch->GetEntry(entryNumber); });
    }

    void SimpleCache::getEntryForAllBranches(EntryNumber entryNumber) const {
      auto guard = filePtr_->setCacheReadTemporarily(treeCache_.get(), tree_);
      oneapi::tbb::this_task_arena::isolate([&]() { tree_->GetEntry(entryNumber); });
    }

    void SimpleCache::createPrimaryCache(unsigned int cacheSize) {
      treeCache_ = createCacheWithSize(cacheSize);
      assert(treeCache_);
      treeCache_->SetEnablePrefetching(enablePrefetching_);
    }

    void SimpleCache::resetTraining(bool promptRead) {
      if (cachestats) {
        treeCache_->Print("a cachedbranches");
      }
      const auto addBranches = promptRead ? "*" : nullptr;
      trainCache(addBranches);
    }

    void SimpleCache::trainCache(char const* branchNames) {
      treeCache_->StartLearningPhase();
      treeCache_->SetEntryRange(0, tree_->GetEntries());
      if (branchNames) {
        treeCache_->SetLearnEntries(0);
        treeCache_->AddBranch(branchNames, kTRUE);
        treeCache_->StopLearningPhase();
      } else {
        treeCache_->SetLearnEntries(learningEntries_);
        auto guard = filePtr_->setCacheReadTemporarily(treeCache_.get(), tree_);
        tree_->LoadTree(0);
      }
    }

    class SimpleWithAuxCache : public SimpleCache {
    public:
      SimpleWithAuxCache(std::shared_ptr<InputFile> filePtr,
                         unsigned int learningEntries,
                         bool enablePrefetching,
                         BranchType const& branchType);
      void getAuxEntry(TBranch* branch, EntryNumber entryNumber) override;
      void reset() override;

    private:
      TTreeCache* getAuxCache(TBranch* auxBranch);
      std::shared_ptr<TTreeCache> auxCache_;
    };

    SimpleWithAuxCache::SimpleWithAuxCache(std::shared_ptr<InputFile> filePtr,
                                           unsigned int learningEntries,
                                           bool enablePrefetching,
                                           BranchType const& branchType)
        : SimpleCache(filePtr, learningEntries, enablePrefetching, branchType) {}

    TTreeCache* SimpleWithAuxCache::getAuxCache(TBranch* auxBranch) {
      if (not auxCache_) {
        auxCache_ = createCacheWithSize(1 * 1024 * 1024);
        if (auxCache_) {
          auxCache_->SetEnablePrefetching(enablePrefetching_);
          auxCache_->SetLearnEntries(0);
          auxCache_->StartLearningPhase();
          auxCache_->SetEntryRange(0, tree_->GetEntries());
          auxCache_->AddBranch(auxBranch->GetName(), kTRUE);
          auxCache_->StopLearningPhase();
        }
      }
      return auxCache_.get();
    }

    void SimpleWithAuxCache::reset() {
      if constexpr (cachestats) {
        if (auxCache_)
          auxCache_->Print("a cachedbranches");
      }
      auxCache_.reset();
      SimpleCache::reset();
    }

    void SimpleWithAuxCache::getAuxEntry(TBranch* branch, EntryNumber entryNumber) {
      auto guard = filePtr_->setCacheReadTemporarily(getAuxCache(branch), tree_);
      branch->GetEntry(entryNumber);
    }

    //
    // SparseReadCache implements a policy for jobs with access
    // patterns typical of late-stage analysis jobs, which may
    // read only a sparse selection of branches from selected
    // events.  Handles many special cases by swapping between
    // different specialized caches.  Not thread safe.
    //
    class SparseReadCache : public CacheManagerBase {
    public:
      SparseReadCache(std::shared_ptr<InputFile> filePtr,
                      unsigned int learningEntries,
                      bool enablePrefetching,
                      BranchType const& branchType)
          : CacheManagerBase(filePtr),
            branchType_(branchType),
            learningEntries_(learningEntries),
            enablePrefetching_(enablePrefetching),
            enableTriggerCache_(branchType_ == InEvent) {}
      void createPrimaryCache(unsigned int cacheSize) override;
      void resetTraining(bool promptRead) override;
      void reset() override;
      void setEntryNumber(EntryNumber nextEntryNumber, EntryNumber entryNumber, EntryNumber entries) override;
      void trainCache(char const* branchNames) override;
      void getEntry(TBranch* branch, EntryNumber entryNumber) override;
      void getAuxEntry(TBranch* auxBranch, EntryNumber entryNumber) override;
      void init(TTree* tree, unsigned int treeAutoFlush) override;
      void reserve(Int_t branchCount) override;
      void getEntryForAllBranches(EntryNumber entryNumber) const override;

    private:
      void getEntryUsingCache(TBranch* branch, EntryNumber entryNumber, TTreeCache* cache);
      TTreeCache* getAuxCache(TBranch* auxBranch);
      TTreeCache* selectCache(TBranch* branch, EntryNumber entryNumber);
      TTreeCache* checkTriggerCache(TBranch* branch, EntryNumber entryNumber);
      TTreeCache* checkTriggerCacheImpl(TBranch* branch, EntryNumber entryNumber);
      void startTraining(EntryNumber entryNumber);
      void stopTraining();

      BranchType branchType_;
      bool trainNow_ = false;
      EntryNumber switchOverEntry_ = -1;
      EntryNumber rawTriggerSwitchOverEntry_ = -1;
      bool performedSwitchOver_ = false;
      unsigned int learningEntries_;
      unsigned int cacheSize_ = 0;
      unsigned long treeAutoFlush_ = 0;

      // Enable asynchronous I/O in ROOT (done in a separate thread).  Only takes
      // effect on the primary treeCache_; all other caches have this explicitly disabled.
      bool enablePrefetching_;
      bool enableTriggerCache_;

      // We use a smart pointer to own the TTreeCache.
      // Unfortunately, ROOT owns it when attached to a TFile, but not after it is detached.
      // So, we make sure to it is detached before closing the TFile so there is no double delete.
      std::shared_ptr<TTreeCache> treeCache_;
      std::shared_ptr<TTreeCache> rawTreeCache_;
      std::shared_ptr<TTreeCache> auxCache_;
      std::shared_ptr<TTreeCache> triggerTreeCache_;
      std::shared_ptr<TTreeCache> rawTriggerTreeCache_;

      std::unordered_set<TBranch*> trainedSet_;
      std::unordered_set<TBranch*> triggerSet_;
    };

    void SparseReadCache::createPrimaryCache(unsigned int cacheSize) {
      cacheSize_ = cacheSize;
      treeCache_ = createCacheWithSize(cacheSize);
      if (treeCache_)
        treeCache_->SetEnablePrefetching(enablePrefetching_);
      rawTreeCache_.reset();
    }

    TTreeCache* SparseReadCache::checkTriggerCache(TBranch* branch, EntryNumber entryNumber) {
      if (!treeCache_->IsAsyncReading() && enableTriggerCache_ && (trainedSet_.find(branch) == trainedSet_.end())) {
        return checkTriggerCacheImpl(branch, entryNumber);
      } else {
        return nullptr;
      }
    }

    // If this function is called, we already know
    // the trigger cache is active and it was a cache miss for the regular cache.
    TTreeCache* SparseReadCache::checkTriggerCacheImpl(TBranch* branch, EntryNumber entryNumber) {
      // This branch is not going to be in the cache.
      // Assume this is a "trigger pattern".
      // Always make sure the branch is added to the trigger set.
      if (triggerSet_.find(branch) == triggerSet_.end()) {
        triggerSet_.insert(branch);
        if (triggerTreeCache_.get()) {
          triggerTreeCache_->AddBranch(branch, kTRUE);
        }
      }

      if (rawTriggerSwitchOverEntry_ < 0) {
        // The trigger has never fired before.  Take everything not in the
        // trainedSet and load it from disk

        // Calculate the end of the next cluster; triggers in the next cluster
        // will use the triggerCache, not the rawTriggerCache.
        //
        // Guarantee that rawTriggerSwitchOverEntry_ is positive (non-zero) after completion
        // of this if-block.
        TTree::TClusterIterator clusterIter = tree_->GetClusterIterator(entryNumber);
        while ((rawTriggerSwitchOverEntry_ < entryNumber) || (rawTriggerSwitchOverEntry_ <= 0)) {
          rawTriggerSwitchOverEntry_ = clusterIter();
        }

        // ROOT will automatically expand the cache to fit one cluster; hence, we use
        // 5 MB as the cache size below
        rawTriggerTreeCache_ = createCacheWithSize(5 * 1024 * 1024);
        if (rawTriggerTreeCache_) {
          rawTriggerTreeCache_->SetEnablePrefetching(false);
          TObjArray* branches = tree_->GetListOfBranches();
          int branchCount = branches->GetEntriesFast();

          // Train the rawTriggerCache to have everything not in the regular cache.
          rawTriggerTreeCache_->SetLearnEntries(0);
          rawTriggerTreeCache_->SetEntryRange(entryNumber, rawTriggerSwitchOverEntry_);
          for (int i = 0; i < branchCount; i++) {
            auto tmp_branch = dynamic_cast<TBranch*>(branches->UncheckedAt(i));
            if (trainedSet_.find(tmp_branch) != trainedSet_.end()) {
              continue;
            }
            rawTriggerTreeCache_->AddBranch(tmp_branch, kTRUE);
          }
          rawTriggerTreeCache_->StopLearningPhase();
        }
        performedSwitchOver_ = false;

        return rawTriggerTreeCache_.get();
      } else if (!performedSwitchOver_ and entryNumber < rawTriggerSwitchOverEntry_) {
        // The raw trigger has fired and it contents are valid.
        return rawTriggerTreeCache_.get();
      } else if (rawTriggerSwitchOverEntry_ > 0) {
        // The raw trigger has fired, but we are out of the cache.  Use the
        // triggerCache instead.
        if (!performedSwitchOver_) {
          rawTriggerTreeCache_.reset();
          performedSwitchOver_ = true;

          // Train the triggerCache
          triggerTreeCache_ = createCacheWithSize(5 * 1024 * 1024);
          triggerTreeCache_->SetEnablePrefetching(false);
          triggerTreeCache_->SetLearnEntries(0);
          triggerTreeCache_->SetEntryRange(entryNumber, tree_->GetEntries());
          for (std::unordered_set<TBranch*>::const_iterator it = triggerSet_.begin(), itEnd = triggerSet_.end();
               it != itEnd;
               it++) {
            triggerTreeCache_->AddBranch(*it, kTRUE);
          }
          triggerTreeCache_->StopLearningPhase();
        }
        return triggerTreeCache_.get();
      }

      // By construction, this case should be impossible.
      assert(false);
      return nullptr;
    }

    TTreeCache* SparseReadCache::getAuxCache(TBranch* auxBranch) {
      if (not auxCache_ and cacheSize_ > 0) {
        auxCache_ = createCacheWithSize(1 * 1024 * 1024);
        if (auxCache_) {
          auxCache_->SetEnablePrefetching(enablePrefetching_);
          auxCache_->SetLearnEntries(0);
          auxCache_->StartLearningPhase();
          auxCache_->SetEntryRange(0, tree_->GetEntries());
          auxCache_->AddBranch(auxBranch->GetName(), kTRUE);
          auxCache_->StopLearningPhase();
        }
      }
      return auxCache_.get();
    }

    TTreeCache* SparseReadCache::selectCache(TBranch* branch, EntryNumber entryNumber) {
      TTreeCache* triggerCache = nullptr;
      if (!treeCache_) {
        return nullptr;
      } else if (treeCache_->IsLearning() && rawTreeCache_) {
        treeCache_->AddBranch(branch, kTRUE);
        trainedSet_.insert(branch);
        return rawTreeCache_.get();
      } else if ((triggerCache = checkTriggerCache(branch, entryNumber))) {
        // A NULL return value from checkTriggerCache indicates the trigger cache case
        // does not apply, and we should continue below.
        return triggerCache;
      } else {
        // The "normal" TTreeCache case.
        return treeCache_.get();
      }
    }

    void SparseReadCache::getEntryUsingCache(TBranch* branch, EntryNumber entryNumber, TTreeCache* cache) {
      // We make sure the treeCache_ is detached from the file,
      // so that ROOT does not also delete it.
      try {
        auto guard = filePtr_->setCacheReadTemporarily(cache, tree_);
        branch->GetEntry(entryNumber);
      } catch (cms::Exception const& e) {
        Exception t(errors::FileReadError, "", e);
        t.addContext(std::string("Reading branch ") + branch->GetName());
        throw t;
      } catch (std::exception const& e) {
        Exception t(errors::FileReadError);
        t << e.what();
        t.addContext(std::string("Reading branch ") + branch->GetName());
        throw t;
      } catch (...) {
        Exception t(errors::FileReadError);
        t << "An exception of unknown type was thrown.";
        t.addContext(std::string("Reading branch ") + branch->GetName());
        throw t;
      }
    }

    void SparseReadCache::getEntry(TBranch* branch, EntryNumber entryNumber) {
      auto cache = selectCache(branch, entryNumber);
      getEntryUsingCache(branch, entryNumber, cache);
    }

    void SparseReadCache::getAuxEntry(TBranch* branch, EntryNumber entryNumber) {
      auto cache = getAuxCache(branch);
      getEntryUsingCache(branch, entryNumber, cache);
    }

    void SparseReadCache::getEntryForAllBranches(EntryNumber entryNumber) const {
      auto guard = filePtr_->setCacheReadTemporarily(treeCache_.get(), tree_);
      oneapi::tbb::this_task_arena::isolate([&]() { tree_->GetEntry(entryNumber); });
    }

    void SparseReadCache::startTraining(EntryNumber entryNumber) {
      if (cacheSize_ == 0) {
        return;
      }
      assert(treeCache_);
      assert(branchType_ == InEvent);
      assert(!rawTreeCache_);

      switchOverEntry_ = entryNumber + learningEntries_;
      auto treeStart = switchOverEntry_;

      treeCache_->SetLearnEntries(learningEntries_);
      rawTreeCache_ = createCacheWithSize(cacheSize_);
      if (rawTreeCache_) {
        rawTreeCache_->SetEnablePrefetching(false);
        rawTreeCache_->SetLearnEntries(0);
        auto rawStart = entryNumber;
        auto rawEnd = switchOverEntry_;
        if (switchOverEntry_ >= tree_->GetEntries()) {
          treeStart = switchOverEntry_ - tree_->GetEntries();
          rawEnd = tree_->GetEntries();
        }
        rawTreeCache_->StartLearningPhase();
        rawTreeCache_->SetEntryRange(rawStart, rawEnd);
        rawTreeCache_->AddBranch("*", kTRUE);
        rawTreeCache_->StopLearningPhase();
      }
      treeCache_->StartLearningPhase();
      treeCache_->SetEntryRange(treeStart, tree_->GetEntries());
      // Make sure that 'branchListIndexes' branch exist in input file
      if (filePtr_->Get(poolNames::branchListIndexesBranchName().c_str()) != nullptr) {
        treeCache_->AddBranch(poolNames::branchListIndexesBranchName().c_str(), kTRUE);
      }
      treeCache_->AddBranch(BranchTypeToAuxiliaryBranchName(branchType_).c_str(), kTRUE);
      trainedSet_.clear();
      triggerSet_.clear();
      assert(treeCache_->GetTree() == tree_);
    }

    void SparseReadCache::stopTraining() {
      auto guard = filePtr_->setCacheReadTemporarily(treeCache_.get(), tree_);
      treeCache_->StopLearningPhase();
      rawTreeCache_.reset();
    }

    void SparseReadCache::resetTraining(bool promptRead) { trainNow_ = true; }

    void SparseReadCache::reset() {
      // We own the treeCache_.
      // We make sure the treeCache_ is detached from the file,
      // so that ROOT does not also delete it.
      filePtr_->clearCacheRead(tree_);
      if constexpr (cachestats) {
        if (treeCache_)
          treeCache_->Print("a cachedbranches");
        if (rawTreeCache_)
          rawTreeCache_->Print("a cachedbranches");
        if (triggerTreeCache_)
          triggerTreeCache_->Print("a cachedbranches");
        if (rawTriggerTreeCache_)
          rawTriggerTreeCache_->Print("a cachedbranches");
        if (auxCache_)
          auxCache_->Print("a cachedbranches");
      }
      treeCache_.reset();
      rawTreeCache_.reset();
      triggerTreeCache_.reset();
      rawTriggerTreeCache_.reset();
      auxCache_.reset();
    }

    void SparseReadCache::setEntryNumber(EntryNumber nextEntryNumber, EntryNumber entryNumber, EntryNumber entries) {
      {
        auto guard = filePtr_->setCacheReadTemporarily(treeCache_.get(), tree_);

        // Detect a backward skip.  If the skip is sufficiently large, we roll the dice and reset the treeCache.
        // This will cause some amount of over-reading: we pre-fetch all the events in some prior cluster.
        // However, because reading one event in the cluster is supposed to be equivalent to reading all events in the cluster,
        // we're not incurring additional over-reading - we're just doing it more efficiently.
        // NOTE: Constructor guarantees treeAutoFlush_ is positive, even if TTree->GetAutoFlush() is negative.
        if (nextEntryNumber < entryNumber and nextEntryNumber >= 0) {
          //We started reading the file near the end, now we need to correct for the learning length
          if (switchOverEntry_ > tree_->GetEntries()) {
            switchOverEntry_ = switchOverEntry_ - tree_->GetEntries();
            if (rawTreeCache_) {
              rawTreeCache_->SetEntryRange(nextEntryNumber, switchOverEntry_);
              rawTreeCache_->FillBuffer();
            }
          }
          if (performedSwitchOver_ and triggerTreeCache_) {
            //We are using the triggerTreeCache_ not the rawTriggerTreeCache_.
            //The triggerTreeCache was originally told to start from an entry further in the file.
            triggerTreeCache_->SetEntryRange(nextEntryNumber, tree_->GetEntries());
          } else if (rawTriggerTreeCache_) {
            //move the switch point to the end of the cluster holding nextEntryNumber
            rawTriggerSwitchOverEntry_ = -1;
            TTree::TClusterIterator clusterIter = tree_->GetClusterIterator(nextEntryNumber);
            while ((rawTriggerSwitchOverEntry_ < nextEntryNumber) || (rawTriggerSwitchOverEntry_ <= 0)) {
              rawTriggerSwitchOverEntry_ = clusterIter();
            }
            rawTriggerTreeCache_->SetEntryRange(nextEntryNumber, rawTriggerSwitchOverEntry_);
          }
        }
        if ((nextEntryNumber < static_cast<EntryNumber>(entryNumber - treeAutoFlush_)) && (treeCache_) &&
            (!treeCache_->IsLearning()) && (entries > 0) && (switchOverEntry_ >= 0)) {
          treeCache_->SetEntryRange(nextEntryNumber, entries);
          treeCache_->FillBuffer();
        }

        tree_->LoadTree(nextEntryNumber);
      }
      if (treeCache_ && trainNow_ && nextEntryNumber >= 0) {
        startTraining(nextEntryNumber);
        trainNow_ = false;
        trainedSet_.clear();
        triggerSet_.clear();
        rawTriggerSwitchOverEntry_ = -1;
      }
      if (treeCache_ && treeCache_->IsLearning() && switchOverEntry_ >= 0 && nextEntryNumber >= switchOverEntry_) {
        stopTraining();
      }
    }

    void SparseReadCache::trainCache(char const* branchNames) {
      if (cacheSize_ == 0) {
        return;
      }

      tree_->LoadTree(0);
      assert(treeCache_);
      {
        auto guard = filePtr_->setCacheReadTemporarily(treeCache_.get(), tree_);
        treeCache_->StartLearningPhase();
        treeCache_->SetEntryRange(0, tree_->GetEntries());
        treeCache_->AddBranch(branchNames, kTRUE);
        treeCache_->StopLearningPhase();
        assert(treeCache_->GetTree() == tree_);
        //want guard to end here
      }
      if (branchType_ == InEvent) {
        // Must also manually add things to the trained set.
        TObjArray* branches = tree_->GetListOfBranches();
        int branchCount = branches->GetEntriesFast();
        for (int i = 0; i < branchCount; i++) {
          TBranch* branch = (TBranch*)branches->UncheckedAt(i);
          if ((branchNames[0] == '*') || (strcmp(branchNames, branch->GetName()) == 0)) {
            trainedSet_.insert(branch);
          }
        }
      }
    }

    void SparseReadCache::init(TTree* tree, unsigned int treeAutoFlush) {
      tree_ = tree;
      treeAutoFlush_ = treeAutoFlush;
      if (treeAutoFlush < learningEntries_) {
        learningEntries_ = treeAutoFlush;
      }
    }

    void SparseReadCache::reserve(Int_t branchCount) {
      trainedSet_.reserve(branchCount);
      triggerSet_.reserve(branchCount);
    }

    std::unique_ptr<CacheManagerBase> CacheManagerBase::create(CacheStrategy strategy,
                                                               std::shared_ptr<InputFile> filePtr,
                                                               unsigned int learningEntries,
                                                               bool enablePrefetching,
                                                               BranchType const& branchType) {
      switch (strategy) {
        case CacheStrategy::kNone:
          return std::make_unique<NoCache>(filePtr);
        case CacheStrategy::kSimple:
          return std::make_unique<SimpleCache>(filePtr, learningEntries, enablePrefetching, branchType);
        case CacheStrategy::kSimpleWithAuxCache:
          return std::make_unique<SimpleWithAuxCache>(filePtr, learningEntries, enablePrefetching, branchType);
        case CacheStrategy::kSparse:
          return std::make_unique<SparseReadCache>(filePtr, learningEntries, enablePrefetching, branchType);
      }

      throw cms::Exception("BadConfig") << "CacheManagerBase:  unknown cache strategy requested";
    }
  }  // namespace roottree
}  // namespace edm
