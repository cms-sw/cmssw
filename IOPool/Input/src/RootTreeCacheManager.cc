#include "InputFile.h"
#include "RootTreeCacheManager.h"

#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/scope.h"

#include "TBranch.h"
#include "TTree.h"
#include "TTreeCache.h"
#include "TTreeCacheUnzip.h"

#include <unordered_set>
#include <cassert>
#include <iostream>

#include "oneapi/tbb/task_arena.h"

#define ROOTDBG
#define PARALLELUNZIP

namespace edm {
  namespace roottree {
    void CacheManagerBase::getEntry(TBranch* branch, EntryNumber entryNumber) { branch->GetEntry(entryNumber); }
    void CacheManagerBase::SetCacheRead(TTreeCache* cache) { filePtr_->SetCacheRead(cache, tree_); }

    class SimpleCache : public CacheManagerBase {
    public:
      SimpleCache(std::shared_ptr<InputFile> filePtr,
                  unsigned int learningEntries,
                  bool enablePrefetching,
                  BranchType const& branchType);
      void reset() override;
      void setCacheSize(unsigned int cacheSize) override;
      void setEntryNumber(EntryNumber theEntryNumber, EntryNumber entryNumber, EntryNumber entries) override;
      void trainCache(char const* branchNames) override;
      void resetTraining() override;
      void getEntry(TBranch* branch, EntryNumber entryNumber) override;

    private:
      // SimpleCache does not own the treeCache_
      TTreeCache* treeCache_;
      std::unique_ptr<oneapi::tbb::task_arena> arena_;
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
          enablePrefetching_(enablePrefetching) {
      if (branchType_ == InEvent) {
#ifdef PARALLELUNZIP
        TTreeCacheUnzip::SetParallelUnzip();
        //enablePrefetching_ = true;
#endif
      }
    }

    void SimpleCache::reset() {
      if (treeCache_) {
        treeCache_->Print();
      }
    }

    void SimpleCache::setEntryNumber(EntryNumber theEntryNumber, EntryNumber entryNumber, EntryNumber entries) {
      tree_->LoadTree(theEntryNumber);
      if (theEntryNumber > learningEntries_ && treeCache_->IsLearning()) {
        treeCache_->StopLearningPhase();
        std::cout << "TTreeCache stop learning\n";
      }
      if (arena_ && theEntryNumber > learningEntries_ && !treeCache_->IsLearning()) {
#ifdef ROOTDBG
        gDebug = 1;
#endif
        arena_->execute([&]() { treeCache_->FillBuffer(); });
      }
      gDebug = 0;
      assert(dynamic_cast<TTreeCache*>(filePtr_->GetCacheRead(tree_)) == treeCache_);
    }

    void SimpleCache::getEntry(TBranch* branch, EntryNumber entryNumber) {
      if (entryNumber < learningEntries_ || treeCache_->IsLearning()) {
        treeCache_->AddBranch(branch, kTRUE);
      } else {
#ifdef ROOTDBG
        gDebug = 1;
#endif
      }
      if (arena_) {
        arena_->execute([&]() {
          branch->GetEntry(entryNumber);
          treeCache_->FillBuffer();
        });
      } else {
        branch->GetEntry(entryNumber);
      }
      gDebug = 0;
      assert(dynamic_cast<TTreeCache*>(filePtr_->GetCacheRead(tree_)) == treeCache_);
    }

    void SimpleCache::setCacheSize(unsigned int cacheSize) {
      if (branchType_ == InEvent) {
#ifdef ROOTDBG
        gDebug = 1;
#endif
#ifdef PARALLELUNZIP
        tree_->SetParallelUnzip(true);
        arena_ = std::make_unique<oneapi::tbb::task_arena>(tbb::this_task_arena::max_concurrency());
        assert(TTreeCacheUnzip::IsParallelUnzip());
#endif
        enablePrefetching_ = true;
      } else {
        tree_->SetParallelUnzip(false);
      }

      tree_->SetCacheSize(static_cast<Long64_t>(cacheSize));
      treeCache_ = dynamic_cast<TTreeCache*>(filePtr_->GetCacheRead(tree_));
      assert(treeCache_);

      if (treeCache_) {
        treeCache_->SetEnablePrefetching(enablePrefetching_);
        treeCache_->SetLearnEntries(learningEntries_);
        treeCache_->SetOptimizeMisses(true);
      }

#ifdef PARALLELUNZIP
      if (branchType_ == InEvent) {
        auto treeCacheUnzip = dynamic_cast<TTreeCacheUnzip*>(filePtr_->GetCacheRead(tree_));
        assert(TTreeCacheUnzip::IsParallelUnzip() && treeCacheUnzip);
      }
#endif
      gDebug = 0;
    }

    void SimpleCache::resetTraining() {
      trainCache(nullptr);
      treeCache_->SetLearnEntries(learningEntries_);
    }

    void SimpleCache::trainCache(char const* branchNames) {
      std::cout << "TTreeCache starting training\n";
      if (branchType_ != InEvent) {
        tree_->SetParallelUnzip(false);
      }
      treeCache_->StartLearningPhase();
      tree_->LoadTree(0);
      tree_->SetCacheEntryRange(0, tree_->GetEntries());
      if (branchNames) {
        tree_->AddBranchToCache(branchNames, kTRUE);
      }
    }

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
      ~SparseReadCache() override { reset(); }
      void setCacheSize(unsigned int cacheSize) override;
      void resetTraining() override;
      void reset() override;
      void setEntryNumber(EntryNumber theEntryNumber, EntryNumber entryNumber, EntryNumber entries) override;
      void trainCache(char const* branchNames) override;
      void getEntry(TBranch* branch, EntryNumber entryNumber) override;
      void init(TTree* tree, unsigned int treeAutoFlush) override;
      void reserve(Int_t branchCount) override;

    private:
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
      mutable std::shared_ptr<TTreeCache> treeCache_;
      mutable std::shared_ptr<TTreeCache> rawTreeCache_;
      CMS_SA_ALLOW mutable std::shared_ptr<TTreeCache> auxCache_;
      CMS_SA_ALLOW mutable std::shared_ptr<TTreeCache> triggerTreeCache_;
      CMS_SA_ALLOW mutable std::shared_ptr<TTreeCache> rawTriggerTreeCache_;

      CMS_SA_ALLOW mutable std::shared_ptr<TTreeCache> auxCache_;std::unordered_set<TBranch*> trainedSet_;
      CMS_SA_ALLOW mutable std::shared_ptr<TTreeCache> auxCache_;std::unordered_set<TBranch*> triggerSet_;
    };

    void SparseReadCache::setCacheSize(unsigned int cacheSize) {
      auto const guard = edm::scope_exit{[&] { SetCacheRead(nullptr); }};

      cacheSize_ = cacheSize;
      tree_->SetCacheSize(static_cast<Long64_t>(cacheSize));
      treeCache_.reset(dynamic_cast<TTreeCache*>(filePtr_->GetCacheRead(tree_)));
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
        tree_->SetCacheSize(static_cast<Long64_t>(5 * 1024 * 1024));
        rawTriggerTreeCache_.reset(dynamic_cast<TTreeCache*>(filePtr_->GetCacheRead(tree_)));
        if (rawTriggerTreeCache_)
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
        performedSwitchOver_ = false;
        rawTriggerTreeCache_->StopLearningPhase();
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

    TTreeCache* SparseReadCache::selectCache(TBranch* branch, EntryNumber entryNumber) {
      auto const guard = edm::scope_exit{[&] { SetCacheRead(nullptr); }};
      TTreeCache* triggerCache = nullptr;
      if (promptRead_) { // should move to simple dsr
        return rawTreeCache_.get();
      }
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

    void SparseReadCache::getEntry(TBranch* branch, EntryNumber entryNumber) {
      auto const guard = edm::scope_exit{[&] { SetCacheRead(nullptr); }};
      SetCacheRead(selectCache(branch, entryNumber));
      branch->GetEntry(entryNumber);
    }

    inline void RootTree::getEntryUsingCache(TBranch* branch, EntryNumber entryNumber, TTreeCache* cache) const {
      try {
        auto guard = filePtr_->setCacheReadTemporarily(cache, tree_);
        branch->GetEntry(entryNumber);
      } catch (cms::Exception const& e) {
        // We make sure the treeCache_ is detached from the file,
        // so that ROOT does not also delete it.
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

    void SparseReadCache::startTraining(EntryNumber entryNumber) {
      auto const guard = edm::scope_exit{[&] { SetCacheRead(nullptr); }};
      if (cacheSize_ == 0) {
        return;
      }
      assert(treeCache_);
      assert(branchType_ == InEvent);
      assert(!rawTreeCache_);
      treeCache_->StartLearningPhase();
      treeCache_->SetLearnEntries(learningEntries_);
      tree_->SetCacheSize(static_cast<Long64_t>(cacheSize_));
      auto treeStart = switchOverEntry_;
      rawTreeCache_.reset(dynamic_cast<TTreeCache*>(filePtr_->GetCacheRead(tree_)));
      if (rawTreeCache_) {
        rawTreeCache_->SetEnablePrefetching(false);
        rawTreeCache_->SetLearnEntries(0);
        switchOverEntry_ = entryNumber + learningEntries_;
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
      auto const guard = edm::scope_exit{[&] { SetCacheRead(nullptr); }};
      treeCache_->StopLearningPhase();
      rawTreeCache_.reset();
    }

    void SparseReadCache::resetTraining() {
      treeCache_->StartLearningPhase();
      trainNow_ = true;
    }

    void SparseReadCache::reset() {
      if (treeCache_)
        treeCache_->Print();
      if (rawTreeCache_)
        rawTreeCache_->Print();
      if (triggerTreeCache_)
        triggerTreeCache_->Print();
      if (rawTriggerTreeCache_)
        rawTriggerTreeCache_->Print();
      SetCacheRead(nullptr);
      treeCache_.reset();
      rawTreeCache_.reset();
      triggerTreeCache_.reset();
      rawTriggerTreeCache_.reset();
      auxCache_.reset();
   }

    void SparseReadCache::setEntryNumber(EntryNumber theEntryNumber, EntryNumber entryNumber, EntryNumber entries) {
      auto guard = filePtr_->setCacheReadTemporarily(treeCache_.get(), tree_);

      // Detect a backward skip.  If the skip is sufficiently large, we roll the dice and reset the treeCache.
      // This will cause some amount of over-reading: we pre-fetch all the events in some prior cluster.
      // However, because reading one event in the cluster is supposed to be equivalent to reading all events in the cluster,
      // we're not incurring additional over-reading - we're just doing it more efficiently.
      // NOTE: Constructor guarantees treeAutoFlush_ is positive, even if TTree->GetAutoFlush() is negative.
      if (theEntryNumber < entryNumber and theEntryNumber >= 0) {
        //We started reading the file near the end, now we need to correct for the learning length
        if (switchOverEntry_ > tree_->GetEntries()) {
          switchOverEntry_ = switchOverEntry_ - tree_->GetEntries();
          if (rawTreeCache_) {
            rawTreeCache_->SetEntryRange(theEntryNumber, switchOverEntry_);
            rawTreeCache_->FillBuffer();
          }
        }
        if (performedSwitchOver_ and triggerTreeCache_) {
          //We are using the triggerTreeCache_ not the rawTriggerTreeCache_.
          //The triggerTreeCache was originally told to start from an entry further in the file.
          triggerTreeCache_->SetEntryRange(theEntryNumber, tree_->GetEntries());
        } else if (rawTriggerTreeCache_) {
          //move the switch point to the end of the cluster holding theEntryNumber
          rawTriggerSwitchOverEntry_ = -1;
          TTree::TClusterIterator clusterIter = tree_->GetClusterIterator(theEntryNumber);
          while ((rawTriggerSwitchOverEntry_ < theEntryNumber) || (rawTriggerSwitchOverEntry_ <= 0)) {
            rawTriggerSwitchOverEntry_ = clusterIter();
          }
          rawTriggerTreeCache_->SetEntryRange(theEntryNumber, rawTriggerSwitchOverEntry_);
        }
      }
      if ((theEntryNumber < static_cast<EntryNumber>(entryNumber - treeAutoFlush_)) && (treeCache_) &&
          (!treeCache_->IsLearning()) && (entries > 0) && (switchOverEntry_ >= 0)) {
        treeCache_->SetEntryRange(theEntryNumber, entries);
        treeCache_->FillBuffer();
      }

      tree_->LoadTree(theEntryNumber);
      SetCacheRead(nullptr);
      if (treeCache_ && trainNow_ && theEntryNumber >= 0) {
        startTraining(entryNumber);
        trainNow_ = false;
        trainedSet_.clear();
        triggerSet_.clear();
        rawTriggerSwitchOverEntry_ = -1;
      }
      if (treeCache_ && treeCache_->IsLearning() && switchOverEntry_ >= 0 && theEntryNumber >= switchOverEntry_) {
        stopTraining();
      }
    }

    void SparseReadCache::trainCache(char const* branchNames) {
      // We own the treeCache_.
      // We make sure the treeCache_ is detached from the file,
      // so that ROOT does not also delete it.

      if (cacheSize_ == 0) {
        return;
      }

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

    std::unique_ptr<CacheManagerBase> CacheManagerBase::create(const std::string& strategy,
                                                               std::shared_ptr<InputFile> filePtr,
                                                               unsigned int learningEntries,
                                                               bool enablePrefetching,
                                                               BranchType const& branchType) {
      if (strategy == "Simple") {
        return std::make_unique<SimpleCache>(filePtr, learningEntries, enablePrefetching, branchType);
      } else if (strategy == "Sparse") {
        return std::make_unique<SparseReadCache>(filePtr, learningEntries, enablePrefetching, branchType);
      }
      throw cms::Exception("BadConfig") << "CacheManagerBase:  unknown cache strategy requested: " << strategy;
    }
  }  // namespace roottree
}  // namespace edm
