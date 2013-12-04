#include "RootTree.h"
#include "RootDelayedReader.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "InputFile.h"
#include "TTree.h"
#include "TTreeIndex.h"
#include "TTreeCache.h"

#include <cassert>
#include <iostream>

namespace edm {
  namespace {
    TBranch* getAuxiliaryBranch(TTree* tree, BranchType const& branchType) {
      TBranch* branch = tree->GetBranch(BranchTypeToAuxiliaryBranchName(branchType).c_str());
      if (branch == nullptr) {
        branch = tree->GetBranch(BranchTypeToAuxBranchName(branchType).c_str());
      }
      return branch;
    }
    TBranch* getProductProvenanceBranch(TTree* tree, BranchType const& branchType) {
      TBranch* branch = tree->GetBranch(BranchTypeToBranchEntryInfoBranchName(branchType).c_str());
      return branch;
    }
  }
  RootTree::RootTree(boost::shared_ptr<InputFile> filePtr,
                     BranchType const& branchType,
                     unsigned int nIndexes,
                     unsigned int maxVirtualSize,
                     unsigned int cacheSize,
                     unsigned int learningEntries,
                     bool enablePrefetching,
                     InputType inputType) :
    filePtr_(filePtr),
    tree_(dynamic_cast<TTree*>(filePtr_.get() != nullptr ? filePtr_->Get(BranchTypeToProductTreeName(branchType).c_str()) : nullptr)),
    metaTree_(dynamic_cast<TTree*>(filePtr_.get() != nullptr ? filePtr_->Get(BranchTypeToMetaDataTreeName(branchType).c_str()) : nullptr)),
    branchType_(branchType),
    auxBranch_(tree_ ? getAuxiliaryBranch(tree_, branchType_) : nullptr),
    treeCache_(),
    rawTreeCache_(),
    triggerTreeCache_(),
    rawTriggerTreeCache_(),
    trainedSet_(),
    triggerSet_(),
    entries_(tree_ ? tree_->GetEntries() : 0),
    entryNumber_(-1),
    entryNumberForIndex_(new std::vector<EntryNumber>(nIndexes, IndexIntoFile::invalidEntry)),
    branchNames_(),
    branches_(new BranchMap),
    trainNow_(false),
    switchOverEntry_(-1),
    rawTriggerSwitchOverEntry_(-1),
    learningEntries_(learningEntries),
    cacheSize_(cacheSize),
    treeAutoFlush_(0),
    enablePrefetching_(enablePrefetching),
    enableTriggerCache_(branchType_ == InEvent),
    rootDelayedReader_(new RootDelayedReader(*this, filePtr, inputType)),
    branchEntryInfoBranch_(metaTree_ ? getProductProvenanceBranch(metaTree_, branchType_) : (tree_ ? getProductProvenanceBranch(tree_, branchType_) : 0)),
    infoTree_(dynamic_cast<TTree*>(filePtr_.get() != nullptr ? filePtr->Get(BranchTypeToInfoTreeName(branchType).c_str()) : nullptr)) // backward compatibility
    {
      assert(tree_);
      // On merged files in older releases of ROOT, the autoFlush setting is always negative; we must guess.
      // TODO: On newer merged files, we should be able to get this from the cluster iterator.
      long treeAutoFlush = (tree_ ? tree_->GetAutoFlush() : 0);
      if (treeAutoFlush < 0) {
        // The "+1" is here to avoid divide-by-zero in degenerate cases.
        Long64_t averageEventSizeBytes = tree_->GetZipBytes() / (tree_->GetEntries()+1) + 1;
        treeAutoFlush_ = cacheSize_/averageEventSizeBytes+1;
      } else {
        treeAutoFlush_ = treeAutoFlush;
      }
      if (treeAutoFlush_ < learningEntries_) {
        learningEntries_ = treeAutoFlush_;
      }
      setTreeMaxVirtualSize(maxVirtualSize);
      setCacheSize(cacheSize);
      if (tree_) {
         Int_t branchCount = tree_->GetListOfBranches()->GetEntriesFast();
         trainedSet_.reserve(branchCount);
         triggerSet_.reserve(branchCount);
      }
  }

  RootTree::~RootTree() {
  }

  RootTree::EntryNumber const&
  RootTree::entryNumberForIndex(unsigned int index) const {
    assert(index < entryNumberForIndex_->size());
    return (*entryNumberForIndex_)[index];
  }

  void
  RootTree::insertEntryForIndex(unsigned int index) {
    assert(index < entryNumberForIndex_->size());
    (*entryNumberForIndex_)[index] = entryNumber();
  }

  bool
  RootTree::isValid() const {
    if (metaTree_ == nullptr || metaTree_->GetNbranches() == 0) {
      return tree_ != nullptr && auxBranch_ != nullptr;
    }
    if (tree_ != nullptr && auxBranch_ != nullptr && metaTree_ != nullptr) { // backward compatibility
      if (branchEntryInfoBranch_ != nullptr || infoTree_ != nullptr) return true; // backward compatibility
      return (entries_ == metaTree_->GetEntries() && tree_->GetNbranches() <= metaTree_->GetNbranches() + 1);  // backward compatibility
    } // backward compatibility
    return false;
  }

  DelayedReader*
  RootTree::rootDelayedReader() const {
    rootDelayedReader_->reset();
    return rootDelayedReader_.get();
  }  

  void
  RootTree::setPresence(BranchDescription& prod, std::string const& oldBranchName) {
      assert(isValid());
      if(tree_->GetBranch(oldBranchName.c_str()) == nullptr){
        prod.setDropped(true);
      }
  }

  void
  RootTree::addBranch(BranchKey const& key,
                      BranchDescription const& prod,
                      std::string const& oldBranchName) {
      assert(isValid());
      //use the translated branch name
      TBranch* branch = tree_->GetBranch(oldBranchName.c_str());
      roottree::BranchInfo info = roottree::BranchInfo(BranchDescription(prod));
      info.productBranch_ = nullptr;
      if (prod.present()) {
        info.productBranch_ = branch;
        //we want the new branch name for the JobReport
        branchNames_.push_back(prod.branchName());
      }
      TTree* provTree = (metaTree_ != nullptr ? metaTree_ : tree_);
      info.provenanceBranch_ = provTree->GetBranch(oldBranchName.c_str());
      branches_->insert(std::make_pair(key, info));
  }

  void
  RootTree::dropBranch(std::string const& oldBranchName) {
      //use the translated branch name
      TBranch* branch = tree_->GetBranch(oldBranchName.c_str());
      if (branch != nullptr) {
        TObjArray* leaves = tree_->GetListOfLeaves();
        int entries = leaves->GetEntries();
        for (int i = 0; i < entries; ++i) {
          TLeaf* leaf = (TLeaf*)(*leaves)[i];
          if (leaf == nullptr) continue;
          TBranch* br = leaf->GetBranch();
          if (br == nullptr) continue;
          if (br->GetMother() == branch) {
            leaves->Remove(leaf);
          }
        }
        leaves->Compress();
        tree_->GetListOfBranches()->Remove(branch);
        tree_->GetListOfBranches()->Compress();
        delete branch;
      }
  }

  roottree::BranchMap const&
  RootTree::branches() const {return *branches_;}

  void
  RootTree::setCacheSize(unsigned int cacheSize) {
    cacheSize_ = cacheSize;
    tree_->SetCacheSize(static_cast<Long64_t>(cacheSize));
    treeCache_.reset(dynamic_cast<TTreeCache*>(filePtr_->GetCacheRead()));
    if(treeCache_) treeCache_->SetEnablePrefetching(enablePrefetching_);
    filePtr_->SetCacheRead(0);
    rawTreeCache_.reset();
  }

  void
  RootTree::setTreeMaxVirtualSize(int treeMaxVirtualSize) {
    if (treeMaxVirtualSize >= 0) tree_->SetMaxVirtualSize(static_cast<Long64_t>(treeMaxVirtualSize));
  }

  void
  RootTree::setEntryNumber(EntryNumber theEntryNumber) {
    filePtr_->SetCacheRead(treeCache_.get());

    // Detect a backward skip.  If the skip is sufficiently large, we roll the dice and reset the treeCache.
    // This will cause some amount of over-reading: we pre-fetch all the events in some prior cluster.
    // However, because reading one event in the cluster is supposed to be equivalent to reading all events in the cluster,
    // we're not incurring additional over-reading - we're just doing it more efficiently.
    // NOTE: Constructor guarantees treeAutoFlush_ is positive, even if TTree->GetAutoFlush() is negative.
    if ((theEntryNumber < static_cast<EntryNumber>(entryNumber_-treeAutoFlush_)) &&
        (treeCache_) && (!treeCache_->IsLearning()) && (entries_ > 0) && (switchOverEntry_ >= 0)) {
      treeCache_->SetEntryRange(theEntryNumber, entries_);
      treeCache_->FillBuffer();
    }

    entryNumber_ = theEntryNumber;
    tree_->LoadTree(entryNumber_);
    filePtr_->SetCacheRead(0);
    if(treeCache_ && trainNow_ && entryNumber_ >= 0) {
      startTraining();
      trainNow_ = false;
      trainedSet_.clear();
      triggerSet_.clear();
      rawTriggerSwitchOverEntry_ = -1;
    }
    if (treeCache_ && treeCache_->IsLearning() && switchOverEntry_ >= 0 && entryNumber_ >= switchOverEntry_) {
      stopTraining();
    }
  }

  // The actual implementation is done below; it's split in this strange
  // manner in order to keep a by-definition-rare code path out of the instruction cache.
  inline TTreeCache*
  RootTree::checkTriggerCache(TBranch* branch, EntryNumber entryNumber) const {
    if (!treeCache_->IsAsyncReading() && enableTriggerCache_ && (trainedSet_.find(branch) == trainedSet_.end())) {
      return checkTriggerCacheImpl(branch, entryNumber);
    } else {
      return NULL;
    }
  }

  // See comments in the header.  If this function is called, we already know
  // the trigger cache is active and it was a cache miss for the regular cache.
  TTreeCache*
  RootTree::checkTriggerCacheImpl(TBranch* branch, EntryNumber entryNumber) const {
    // This branch is not going to be in the cache.
    // Assume this is a "trigger pattern".
    // Always make sure the branch is added to the trigger set.
    if (triggerSet_.find(branch) == triggerSet_.end()) {
      triggerSet_.insert(branch);
      if (triggerTreeCache_.get()) { triggerTreeCache_->AddBranch(branch, kTRUE); }
    }

    if (rawTriggerSwitchOverEntry_ < 0) {
      // The trigger has never fired before.  Take everything not in the
      // trainedSet and load it from disk

      // Calculate the end of the next cluster; triggers in the next cluster
      // will use the triggerCache, not the rawTriggerCache.
      TTree::TClusterIterator clusterIter = tree_->GetClusterIterator(entryNumber);
      while (rawTriggerSwitchOverEntry_ < entryNumber) {
        rawTriggerSwitchOverEntry_ = clusterIter();
      }

      // ROOT will automatically expand the cache to fit one cluster; hence, we use
      // 5 MB as the cache size below
      tree_->SetCacheSize(static_cast<Long64_t>(5*1024*1024));
      rawTriggerTreeCache_.reset(dynamic_cast<TTreeCache*>(filePtr_->GetCacheRead()));
      if(rawTriggerTreeCache_) rawTriggerTreeCache_->SetEnablePrefetching(false);
      TObjArray *branches = tree_->GetListOfBranches();
      int branchCount = branches->GetEntriesFast();

      // Train the rawTriggerCache to have everything not in the regular cache.
      rawTriggerTreeCache_->SetLearnEntries(0);
      rawTriggerTreeCache_->SetEntryRange(entryNumber, rawTriggerSwitchOverEntry_);
      for (int i=0;i<branchCount;i++) {
        TBranch *tmp_branch = (TBranch*)branches->UncheckedAt(i);
        if (trainedSet_.find(tmp_branch) != trainedSet_.end()) {
          continue;
        }
        rawTriggerTreeCache_->AddBranch(tmp_branch, kTRUE);
      }
      performedSwitchOver_ = false;
      rawTriggerTreeCache_->StopLearningPhase();
      filePtr_->SetCacheRead(0);

      return rawTriggerTreeCache_.get();
    } else if (entryNumber_ < rawTriggerSwitchOverEntry_) {
      // The raw trigger has fired and it contents are valid.
      return rawTriggerTreeCache_.get();
    } else if (rawTriggerSwitchOverEntry_ > 0) {
      // The raw trigger has fired, but we are out of the cache.  Use the
      // triggerCache instead.
      if (!performedSwitchOver_) {
        rawTriggerTreeCache_.reset();
        performedSwitchOver_ = true;

        // Train the triggerCache
        tree_->SetCacheSize(static_cast<Long64_t>(5*1024*1024));
        triggerTreeCache_.reset(dynamic_cast<TTreeCache*>(filePtr_->GetCacheRead()));
        triggerTreeCache_->SetEnablePrefetching(false);
        triggerTreeCache_->SetLearnEntries(0);
        triggerTreeCache_->SetEntryRange(entryNumber, tree_->GetEntries());
        for(std::unordered_set<TBranch*>::const_iterator it = triggerSet_.begin(), itEnd = triggerSet_.end();
            it != itEnd;
            it++)
        {
          triggerTreeCache_->AddBranch(*it, kTRUE);
        }
        triggerTreeCache_->StopLearningPhase();
        filePtr_->SetCacheRead(0);
      }
      return triggerTreeCache_.get();
    } else if (entryNumber_ < rawTriggerSwitchOverEntry_) {
      // The raw trigger has fired and it contents are valid.
      return rawTriggerTreeCache_.get();
    } else if (rawTriggerSwitchOverEntry_ > 0) {
      // The raw trigger has fired, but we are out of the cache.  Use the
      // triggerCache instead.
      if (!performedSwitchOver_) {
        rawTriggerTreeCache_.reset();
        performedSwitchOver_ = true; 
        
        // Train the triggerCache
        tree_->SetCacheSize(static_cast<Long64_t>(5*1024*1024));
        triggerTreeCache_.reset(dynamic_cast<TTreeCache*>(filePtr_->GetCacheRead()));
        triggerTreeCache_->SetEnablePrefetching(false);
        triggerTreeCache_->SetLearnEntries(0);
        triggerTreeCache_->SetEntryRange(entryNumber, tree_->GetEntries());
        for(std::unordered_set<TBranch*>::const_iterator it = triggerSet_.begin(), itEnd = triggerSet_.end();
              it != itEnd;
              it++)
        { 
          triggerTreeCache_->AddBranch(*it, kTRUE);
        }
        triggerTreeCache_->StopLearningPhase();
        filePtr_->SetCacheRead(0);
      }
      return triggerTreeCache_.get();
    }

    // By construction, this case should be impossible.
    assert (false);
    return NULL;
  }

  inline TTreeCache*
  RootTree::selectCache(TBranch* branch, EntryNumber entryNumber) const {
    TTreeCache *triggerCache = NULL;
    if (!treeCache_) {
      return NULL;
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

  void
  RootTree::getEntry(TBranch* branch, EntryNumber entryNumber) const {
    try {
      TTreeCache * cache = selectCache(branch, entryNumber);
      filePtr_->SetCacheRead(cache);
      branch->GetEntry(entryNumber);
      filePtr_->SetCacheRead(0);
    } catch(cms::Exception const& e) {
      // We make sure the treeCache_ is detached from the file,
      // so that ROOT does not also delete it.
      filePtr_->SetCacheRead(0);
      Exception t(errors::FileReadError, "", e);
      t.addContext(std::string("Reading branch ")+branch->GetName());
      throw t;
    }
  }

  void
  RootTree::startTraining() {
    if (cacheSize_ == 0) {
      return;
    }
    assert(treeCache_);
    assert(branchType_ == InEvent);
    assert(!rawTreeCache_);
    treeCache_->SetLearnEntries(learningEntries_);
    tree_->SetCacheSize(static_cast<Long64_t>(cacheSize_));
    rawTreeCache_.reset(dynamic_cast<TTreeCache *>(filePtr_->GetCacheRead()));
    rawTreeCache_->SetEnablePrefetching(false);
    filePtr_->SetCacheRead(0);
    rawTreeCache_->SetLearnEntries(0);
    switchOverEntry_ = entryNumber_ + learningEntries_;
    rawTreeCache_->StartLearningPhase();
    rawTreeCache_->SetEntryRange(entryNumber_, switchOverEntry_);
    rawTreeCache_->AddBranch("*", kTRUE);
    rawTreeCache_->StopLearningPhase();
    treeCache_->StartLearningPhase();
    treeCache_->SetEntryRange(switchOverEntry_, tree_->GetEntries());
    treeCache_->AddBranch(poolNames::branchListIndexesBranchName().c_str(), kTRUE);
    treeCache_->AddBranch(BranchTypeToAuxiliaryBranchName(branchType_).c_str(), kTRUE);
    trainedSet_.clear();
    triggerSet_.clear();
    assert(treeCache_->GetTree() == tree_);
  }

  void
  RootTree::stopTraining() {
    filePtr_->SetCacheRead(treeCache_.get());
    treeCache_->StopLearningPhase();
    filePtr_->SetCacheRead(0);
    rawTreeCache_.reset();
  }

  void
  RootTree::close () {
    // The TFile is about to be closed, and destructed.
    // Just to play it safe, zero all pointers to quantities that are owned by the TFile.
    auxBranch_  = branchEntryInfoBranch_ = nullptr;
    tree_ = metaTree_ = infoTree_ = nullptr;
    // We own the treeCache_.
    // We make sure the treeCache_ is detached from the file,
    // so that ROOT does not also delete it.
    filePtr_->SetCacheRead(0);
    // We *must* delete the TTreeCache here because the TFilePrefetch object
    // references the TFile.  If TFile is closed, before the TTreeCache is
    // deleted, the TFilePrefetch may continue to do TFile operations, causing
    // deadlocks or exceptions.
    treeCache_.reset();
    rawTreeCache_.reset();
    triggerTreeCache_.reset();
    rawTriggerTreeCache_.reset();
    // We give up our shared ownership of the TFile itself.
    filePtr_.reset();
  }

  void
  RootTree::trainCache(char const* branchNames) {
    if (cacheSize_ == 0) {
      return;
    }
    tree_->LoadTree(0);
    assert(treeCache_);
    filePtr_->SetCacheRead(treeCache_.get());
    treeCache_->StartLearningPhase();
    treeCache_->SetEntryRange(0, tree_->GetEntries());
    treeCache_->AddBranch(branchNames, kTRUE);
    treeCache_->StopLearningPhase();
    assert(treeCache_->GetTree() == tree_);
    // We own the treeCache_.
    // We make sure the treeCache_ is detached from the file,
    // so that ROOT does not also delete it.
    filePtr_->SetCacheRead(0);

    // Must also manually add things to the trained set.
    TObjArray *branches = tree_->GetListOfBranches();
    int branchCount = branches->GetEntriesFast();
    for (int i=0;i<branchCount;i++) {
       TBranch *branch = (TBranch*)branches->UncheckedAt(i);
       if ((branchNames[0] == '*') || (strcmp(branchNames, branch->GetName()) == 0)) {
          trainedSet_.insert(branch);
       } 
    } 
 
  }

  namespace roottree {
    Int_t
    getEntry(TBranch* branch, EntryNumber entryNumber) {
      Int_t n = 0;
      try {
        n = branch->GetEntry(entryNumber);
      }
      catch(cms::Exception const& e) {
        throw Exception(errors::FileReadError, "", e);
      }
      return n;
    }

    Int_t
    getEntry(TTree* tree, EntryNumber entryNumber) {
      Int_t n = 0;
      try {
        n = tree->GetEntry(entryNumber);
      }
      catch(cms::Exception const& e) {
        throw Exception (errors::FileReadError, "", e);
      }
      return n;
    }

    std::unique_ptr<TTreeCache>
    trainCache(TTree* tree, InputFile& file, unsigned int cacheSize, char const* branchNames) {
      tree->LoadTree(0);
      tree->SetCacheSize(cacheSize);
      std::unique_ptr<TTreeCache> treeCache(dynamic_cast<TTreeCache*>(file.GetCacheRead()));
      if (nullptr != treeCache.get()) {
        treeCache->StartLearningPhase();
        treeCache->SetEntryRange(0, tree->GetEntries());
        treeCache->AddBranch(branchNames, kTRUE);
        treeCache->StopLearningPhase();
      }
      // We own the treeCache_.
      // We make sure the treeCache_ is detached from the file,
      // so that ROOT does not also delete it.
      file.SetCacheRead(0);
      return treeCache;
    }
  }
}
