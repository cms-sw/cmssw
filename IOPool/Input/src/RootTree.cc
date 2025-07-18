#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Common/interface/getWrapperBasePtr.h"

#include "InputFile.h"
#include "RootTree.h"
#include "RootDelayedReader.h"
#include "RootPromptReadDelayedReader.h"

#include "TTree.h"
#include "TTreeCache.h"
#include "TLeaf.h"

#include "oneapi/tbb/task_arena.h"
#include <cassert>

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

    std::unique_ptr<RootDelayedReaderBase> makeRootDelayedReader(RootTree const& tree,
                                                                 std::shared_ptr<InputFile> filePtr,
                                                                 InputType inputType,
                                                                 unsigned int nIndexes,
                                                                 bool promptRead) {
      if (promptRead) {
        return std::make_unique<RootPromptReadDelayedReader>(tree, filePtr, inputType, nIndexes);
      }
      return std::make_unique<RootDelayedReader>(tree, filePtr, inputType);
    }
  }  // namespace

  // Used for all RootTrees
  // All the other constructors delegate to this one
  RootTree::RootTree(std::shared_ptr<InputFile> filePtr,
                     BranchType const& branchType,
                     unsigned int nIndexes,
                     unsigned int learningEntries,
                     bool enablePrefetching,
                     bool promptRead,
                     InputType inputType)
      : filePtr_(filePtr),
        branchType_(branchType),
        entryNumberForIndex_(std::make_unique<std::vector<EntryNumber>>(nIndexes, IndexIntoFile::invalidEntry)),
        learningEntries_(learningEntries),
        enablePrefetching_(enablePrefetching),
        enableTriggerCache_(branchType_ == InEvent),
        promptRead_(promptRead),
        rootDelayedReader_(makeRootDelayedReader(*this, filePtr, inputType, nIndexes, promptRead)) {}

  // Used for Event/Lumi/Run RootTrees
  RootTree::RootTree(std::shared_ptr<InputFile> filePtr,
                     BranchType const& branchType,
                     unsigned int nIndexes,
                     Options const& options,
                     unsigned int learningEntries,
                     InputType inputType)
      : RootTree(
            filePtr, branchType, nIndexes, learningEntries, options.enablePrefetching, options.promptReading, inputType) {
    init(BranchTypeToProductTreeName(branchType), options.treeMaxVirtualSize, options.treeCacheSize);
    metaTree_ = dynamic_cast<TTree*>(filePtr_->Get(BranchTypeToMetaDataTreeName(branchType).c_str()));
    auxBranch_ = getAuxiliaryBranch(tree_, branchType_);
    branchEntryInfoBranch_ =
        metaTree_ ? getProductProvenanceBranch(metaTree_, branchType_) : getProductProvenanceBranch(tree_, branchType_);
    infoTree_ =
        dynamic_cast<TTree*>(filePtr->Get(BranchTypeToInfoTreeName(branchType).c_str()));  // backward compatibility
  }

  // Used for ProcessBlock RootTrees
  RootTree::RootTree(std::shared_ptr<InputFile> filePtr,
                     BranchType const& branchType,
                     std::string const& processName,
                     unsigned int nIndexes,
                     Options const& options,
                     unsigned int learningEntries,
                     InputType inputType)
      : RootTree(
            filePtr, branchType, nIndexes, learningEntries, options.enablePrefetching, options.promptReading, inputType) {
    processName_ = processName;
    init(BranchTypeToProductTreeName(branchType, processName), options.treeMaxVirtualSize, options.treeCacheSize);
  }

  void RootTree::init(std::string const& productTreeName, unsigned int maxVirtualSize, unsigned int cacheSize) {
    if (filePtr_.get() != nullptr) {
      tree_ = dynamic_cast<TTree*>(filePtr_->Get(productTreeName.c_str()));
    }
    if (not tree_) {
      throw cms::Exception("WrongFileFormat")
          << "The ROOT file does not contain a TTree named " << productTreeName
          << "\n This is either not an edm ROOT file or is one that has been corrupted.";
    }
    entries_ = tree_->GetEntries();

    // On merged files in older releases of ROOT, the autoFlush setting is always negative; we must guess.
    // TODO: On newer merged files, we should be able to get this from the cluster iterator.
    long treeAutoFlush = tree_->GetAutoFlush();
    if (treeAutoFlush < 0) {
      // The "+1" is here to avoid divide-by-zero in degenerate cases.
      Long64_t averageEventSizeBytes = tree_->GetZipBytes() / (tree_->GetEntries() + 1) + 1;
      treeAutoFlush_ = cacheSize / averageEventSizeBytes + 1;
    } else {
      treeAutoFlush_ = treeAutoFlush;
    }
    if (treeAutoFlush_ < learningEntries_) {
      learningEntries_ = treeAutoFlush_;
    }
    setTreeMaxVirtualSize(maxVirtualSize);
    setCacheSize(cacheSize);
    if (branchType_ == InEvent) {
      Int_t branchCount = tree_->GetListOfBranches()->GetEntriesFast();
      trainedSet_.reserve(branchCount);
      triggerSet_.reserve(branchCount);
    }
  }

  RootTree::~RootTree() {}

  RootTree::EntryNumber const& RootTree::entryNumberForIndex(unsigned int index) const {
    assert(index < entryNumberForIndex_->size());
    return (*entryNumberForIndex_)[index];
  }

  void RootTree::insertEntryForIndex(unsigned int index) {
    assert(index < entryNumberForIndex_->size());
    (*entryNumberForIndex_)[index] = entryNumber();
  }

  bool RootTree::isValid() const {
    // ProcessBlock
    if (branchType_ == InProcess) {
      return tree_ != nullptr;
    }
    // Run/Lumi/Event
    if (metaTree_ == nullptr || metaTree_->GetNbranches() == 0) {
      return tree_ != nullptr && auxBranch_ != nullptr;
    }
    // Backward compatibility for Run/Lumi/Event
    if (tree_ != nullptr && auxBranch_ != nullptr && metaTree_ != nullptr) {  // backward compatibility
      if (branchEntryInfoBranch_ != nullptr || infoTree_ != nullptr)
        return true;  // backward compatibility
      return (entries_ == metaTree_->GetEntries() &&
              tree_->GetNbranches() <= metaTree_->GetNbranches() + 1);  // backward compatibility
    }  // backward compatibility
    return false;
  }

  DelayedReader* RootTree::resetAndGetRootDelayedReader() const {
    rootDelayedReader_->reset();
    return rootDelayedReader_.get();
  }

  RootDelayedReaderBase* RootTree::rootDelayedReader() const { return rootDelayedReader_.get(); }

  void RootTree::setPresence(ProductDescription& prod, std::string const& oldBranchName) {
    assert(isValid());
    if (tree_->GetBranch(oldBranchName.c_str()) == nullptr) {
      prod.setDropped(true);
    }
  }

  void roottree::BranchInfo::setBranch(TBranch* branch, TClass const* wrapperBaseTClass) {
    productBranch_ = branch;
    if (branch) {
      classCache_ = TClass::GetClass(productDescription_.wrappedName().c_str());
      offsetToWrapperBase_ = classCache_->GetBaseClassOffset(wrapperBaseTClass);
    }
  }
  std::unique_ptr<WrapperBase> roottree::BranchInfo::newWrapper() const {
    assert(nullptr != classCache_);
    void* p = classCache_->New();
    return getWrapperBasePtr(p, offsetToWrapperBase_);
  }

  void RootTree::addBranch(ProductDescription const& prod, std::string const& oldBranchName) {
    assert(isValid());
    static TClass const* const wrapperBaseTClass = TClass::GetClass("edm::WrapperBase");
    //use the translated branch name
    TBranch* branch = tree_->GetBranch(oldBranchName.c_str());
    roottree::BranchInfo info = roottree::BranchInfo(prod);
    info.productBranch_ = nullptr;
    if (prod.present()) {
      info.setBranch(branch, wrapperBaseTClass);
      //we want the new branch name for the JobReport
      branchNames_.push_back(prod.branchName());
    }
    branches_.insert(prod.branchID(), info);
  }

  void RootTree::dropBranch(std::string const& oldBranchName) {
    //use the translated branch name
    TBranch* branch = tree_->GetBranch(oldBranchName.c_str());
    if (branch != nullptr) {
      TObjArray* leaves = tree_->GetListOfLeaves();
      int entries = leaves->GetEntries();
      for (int i = 0; i < entries; ++i) {
        TLeaf* leaf = (TLeaf*)(*leaves)[i];
        if (leaf == nullptr)
          continue;
        TBranch* br = leaf->GetBranch();
        if (br == nullptr)
          continue;
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

  roottree::BranchMap const& RootTree::branches() const { return branches_; }

  std::shared_ptr<TTreeCache> RootTree::createCacheWithSize(unsigned int cacheSize) const {
    return filePtr_->createCacheWithSize(*tree_, cacheSize);
  }

  void RootTree::setCacheSize(unsigned int cacheSize) {
    cacheSize_ = cacheSize;
    treeCache_ = createCacheWithSize(cacheSize);
    if (treeCache_)
      treeCache_->SetEnablePrefetching(enablePrefetching_);
    rawTreeCache_.reset();
  }

  void RootTree::setTreeMaxVirtualSize(int treeMaxVirtualSize) {
    if (treeMaxVirtualSize >= 0)
      tree_->SetMaxVirtualSize(static_cast<Long64_t>(treeMaxVirtualSize));
  }

  bool RootTree::nextWithCache() {
    bool returnValue = ++entryNumber_ < entries_;
    if (returnValue) {
      setEntryNumber(entryNumber_);
    }
    return returnValue;
  }

  void RootTree::setEntryNumber(EntryNumber theEntryNumber) {
    {
      auto guard = filePtr_->setCacheReadTemporarily(treeCache_.get(), tree_);

      // Detect a backward skip.  If the skip is sufficiently large, we roll the dice and reset the treeCache.
      // This will cause some amount of over-reading: we pre-fetch all the events in some prior cluster.
      // However, because reading one event in the cluster is supposed to be equivalent to reading all events in the cluster,
      // we're not incurring additional over-reading - we're just doing it more efficiently.
      // NOTE: Constructor guarantees treeAutoFlush_ is positive, even if TTree->GetAutoFlush() is negative.
      if (theEntryNumber < entryNumber_ and theEntryNumber >= 0) {
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
      if ((theEntryNumber < static_cast<EntryNumber>(entryNumber_ - treeAutoFlush_)) && (treeCache_) &&
          (!treeCache_->IsLearning()) && (entries_ > 0) && (switchOverEntry_ >= 0)) {
        treeCache_->SetEntryRange(theEntryNumber, entries_);
        treeCache_->FillBuffer();
      }

      entryNumber_ = theEntryNumber;
      tree_->LoadTree(entryNumber_);
      //want guard to end here
    }
    if (treeCache_ && trainNow_ && entryNumber_ >= 0) {
      startTraining();
      trainNow_ = false;
      trainedSet_.clear();
      triggerSet_.clear();
      rawTriggerSwitchOverEntry_ = -1;
    }
    if (not promptRead_ && treeCache_ && treeCache_->IsLearning() && switchOverEntry_ >= 0 &&
        entryNumber_ >= switchOverEntry_) {
      stopTraining();
    }
  }

  // The actual implementation is done below; it's split in this strange
  // manner in order to keep a by-definition-rare code path out of the instruction cache.
  inline TTreeCache* RootTree::checkTriggerCache(TBranch* branch, EntryNumber entryNumber) const {
    if (!treeCache_->IsAsyncReading() && enableTriggerCache_ && (trainedSet_.find(branch) == trainedSet_.end())) {
      return checkTriggerCacheImpl(branch, entryNumber);
    } else {
      return nullptr;
    }
  }

  // See comments in the header.  If this function is called, we already know
  // the trigger cache is active and it was a cache miss for the regular cache.
  TTreeCache* RootTree::checkTriggerCacheImpl(TBranch* branch, EntryNumber entryNumber) const {
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
      if (rawTriggerTreeCache_)
        rawTriggerTreeCache_->SetEnablePrefetching(false);
      TObjArray* branches = tree_->GetListOfBranches();
      int branchCount = branches->GetEntriesFast();

      // Train the rawTriggerCache to have everything not in the regular cache.
      rawTriggerTreeCache_->SetLearnEntries(0);
      rawTriggerTreeCache_->SetEntryRange(entryNumber, rawTriggerSwitchOverEntry_);
      for (int i = 0; i < branchCount; i++) {
        TBranch* tmp_branch = (TBranch*)branches->UncheckedAt(i);
        if (trainedSet_.find(tmp_branch) != trainedSet_.end()) {
          continue;
        }
        rawTriggerTreeCache_->AddBranch(tmp_branch, kTRUE);
      }
      performedSwitchOver_ = false;
      rawTriggerTreeCache_->StopLearningPhase();

      return rawTriggerTreeCache_.get();
    } else if (!performedSwitchOver_ and entryNumber_ < rawTriggerSwitchOverEntry_) {
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

  inline TTreeCache* RootTree::selectCache(TBranch* branch, EntryNumber entryNumber) const {
    TTreeCache* triggerCache = nullptr;
    if (promptRead_) {
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
  TTreeCache* RootTree::getAuxCache(TBranch* auxBranch) const {
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

  void RootTree::getEntryForAllBranches() const {
    oneapi::tbb::this_task_arena::isolate([&]() {
      auto guard = filePtr_->setCacheReadTemporarily(treeCache_.get(), tree_);
      tree_->GetEntry(entryNumber_);
    });
  }

  void RootTree::getEntry(TBranch* branch, EntryNumber entryNumber) const {
    getEntryUsingCache(branch, entryNumber, selectCache(branch, entryNumber));
  }

  inline void RootTree::getEntryUsingCache(TBranch* branch, EntryNumber entryNumber, TTreeCache* cache) const {
    LogTrace("IOTrace").format(
        "RootTree::getEntryUsingCache() begin for branch {} entry {}", branch->GetName(), entryNumber);
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
    LogTrace("IOTrace").format(
        "RootTree::getEntryUsingCache() end for branch {} entry {}", branch->GetName(), entryNumber);
  }

  bool RootTree::skipEntries(unsigned int& offset) {
    entryNumber_ += offset;
    bool retval = (entryNumber_ < entries_);
    if (retval) {
      offset = 0;
    } else {
      // Not enough entries in the file to skip.
      // The +1 is needed because entryNumber_ is -1 at the initialization of the tree, not 0.
      long long overshoot = entryNumber_ + 1 - entries_;
      entryNumber_ = entries_;
      offset = overshoot;
    }
    return retval;
  }

  void RootTree::startTraining() {
    if (cacheSize_ == 0) {
      return;
    }
    assert(treeCache_);
    assert(branchType_ == InEvent);
    assert(!rawTreeCache_);
    treeCache_->SetLearnEntries(learningEntries_);
    rawTreeCache_ = createCacheWithSize(cacheSize_);
    rawTreeCache_->SetEnablePrefetching(false);
    rawTreeCache_->SetLearnEntries(0);
    if (promptRead_) {
      switchOverEntry_ = entries_;
    } else {
      switchOverEntry_ = entryNumber_ + learningEntries_;
    }
    auto rawStart = entryNumber_;
    auto rawEnd = switchOverEntry_;
    auto treeStart = switchOverEntry_;
    if (switchOverEntry_ >= tree_->GetEntries()) {
      treeStart = switchOverEntry_ - tree_->GetEntries();
      rawEnd = tree_->GetEntries();
    }
    rawTreeCache_->StartLearningPhase();
    rawTreeCache_->SetEntryRange(rawStart, rawEnd);
    rawTreeCache_->AddBranch("*", kTRUE);
    rawTreeCache_->StopLearningPhase();

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

  void RootTree::stopTraining() {
    auto guard = filePtr_->setCacheReadTemporarily(treeCache_.get(), tree_);
    treeCache_->StopLearningPhase();
    rawTreeCache_.reset();
  }

  void RootTree::close() {
    // The TFile is about to be closed, and destructed.
    // Just to play it safe, zero all pointers to quantities that are owned by the TFile.
    auxBranch_ = branchEntryInfoBranch_ = nullptr;
    tree_ = metaTree_ = infoTree_ = nullptr;
    // We own the treeCache_.
    // We make sure the treeCache_ is detached from the file,
    // so that ROOT does not also delete it.
    filePtr_->clearCacheRead(tree_);
    // We *must* delete the TTreeCache here because the TFilePrefetch object
    // references the TFile.  If TFile is closed, before the TTreeCache is
    // deleted, the TFilePrefetch may continue to do TFile operations, causing
    // deadlocks or exceptions.
    treeCache_.reset();
    rawTreeCache_.reset();
    triggerTreeCache_.reset();
    rawTriggerTreeCache_.reset();
    auxCache_.reset();
    // We give up our shared ownership of the TFile itself.
    filePtr_.reset();
  }

  void RootTree::trainCache(char const* branchNames) {
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
      // We own the treeCache_.
      // We make sure the treeCache_ is detached from the file,
      // so that ROOT does not also delete it.

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

  void RootTree::setSignals(
      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadSource,
      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadSource) {
    rootDelayedReader_->setSignals(preEventReadSource, postEventReadSource);
  }

  namespace roottree {
    Int_t getEntry(TBranch* branch, EntryNumber entryNumber) {
      Int_t n = 0;
      try {
        n = branch->GetEntry(entryNumber);
      } catch (cms::Exception const& e) {
        throw Exception(errors::FileReadError, "", e);
      }
      return n;
    }

    Int_t getEntry(TTree* tree, EntryNumber entryNumber) {
      Int_t n = 0;
      try {
        n = tree->GetEntry(entryNumber);
      } catch (cms::Exception const& e) {
        throw Exception(errors::FileReadError, "", e);
      }
      return n;
    }

    std::unique_ptr<TTreeCache> trainCache(TTree* tree,
                                           InputFile& file,
                                           unsigned int cacheSize,
                                           char const* branchNames) {
      tree->LoadTree(0);
      std::unique_ptr<TTreeCache> treeCache = file.createCacheWithSize(*tree, cacheSize);
      if (nullptr != treeCache.get()) {
        treeCache->StartLearningPhase();
        treeCache->SetEntryRange(0, tree->GetEntries());
        treeCache->AddBranch(branchNames, kTRUE);
        treeCache->StopLearningPhase();
      }
      return treeCache;
    }
  }  // namespace roottree
}  // namespace edm
