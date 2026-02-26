#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Common/interface/getWrapperBasePtr.h"

#include "InputFile.h"
#include "RootTree.h"
#include "RootTreeCacheManager.h"
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
  constexpr auto kSimpleCache = roottree::CacheManagerBase::CacheStrategy::kSimple;
  constexpr auto kSparseCache = roottree::CacheManagerBase::CacheStrategy::kSparse;

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
        promptRead_(promptRead),
        rootDelayedReader_(makeRootDelayedReader(*this, filePtr, inputType, nIndexes, promptRead_)),
        treeCacheManager_(
            roottree::CacheManagerBase::create(promptRead_ or branchType_ != InEvent ? kSimpleCache : kSparseCache,
                                               filePtr_,
                                               learningEntries,
                                               enablePrefetching,
                                               branchType_)) {}

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
    treeCacheManager_->init(tree_, treeAutoFlush_);
    setTreeMaxVirtualSize(maxVirtualSize);
    treeCacheManager_->createPrimaryCache(cacheSize);
    if (branchType_ == InEvent) {
      Int_t branchCount = tree_->GetListOfBranches()->GetEntriesFast();
      treeCacheManager_->reserve(branchCount);
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

  void RootTree::setTreeMaxVirtualSize(int treeMaxVirtualSize) {
    if (treeMaxVirtualSize >= 0)
      tree_->SetMaxVirtualSize(static_cast<Long64_t>(treeMaxVirtualSize));
  }

  void RootTree::fillAuxHelper() { treeCacheManager_->getAuxEntry(auxBranch_, entryNumber_); }

  bool RootTree::nextWithCache() {
    bool returnValue = ++entryNumber_ < entries_;
    if (returnValue) {
      setEntryNumber(entryNumber_);
    }
    return returnValue;
  }

  void RootTree::setEntryNumber(EntryNumber newEntryNumber) {
    treeCacheManager_->setEntryNumber(newEntryNumber, entryNumber_, entries_);
    entryNumber_ = newEntryNumber;
  }

  void RootTree::getEntryForAllBranches() const {
    LogTrace("IOTrace").format("RootTree::getEntryForAllBranches() begin for entry {}", entryNumber_);
    treeCacheManager_->getEntryForAllBranches(entryNumber_);
    LogTrace("IOTrace").format("RootTree::getEntryForAllBranches() end for entry {}", entryNumber_);
  }

  void RootTree::getEntry(TBranch* branch, EntryNumber entryNumber) const {
    LogTrace("IOTrace").format("RootTree::getEntry() begin for branch {} entry {}", branch->GetName(), entryNumber);
    treeCacheManager_->getEntry(branch, entryNumber);
    LogTrace("IOTrace").format("RootTree::getEntry() end for branch {} entry {}", branch->GetName(), entryNumber);
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

  void RootTree::resetTraining() { treeCacheManager_->resetTraining(promptRead_); }

  void RootTree::close() {
    // The TFile is about to be closed, and destructed.
    // Just to play it safe, zero all pointers to quantities that are owned by the TFile.
    auxBranch_ = branchEntryInfoBranch_ = nullptr;
    tree_ = metaTree_ = infoTree_ = nullptr;
    // We *must* delete the TTreeCache here because the TFilePrefetch object
    // references the TFile.  If TFile is closed, before the TTreeCache is
    // deleted, the TFilePrefetch may continue to do TFile operations, causing
    // deadlocks or exceptions.
    treeCacheManager_->reset();
    // We give up our shared ownership of the TFile itself.
    filePtr_.reset();
  }

  void RootTree::trainCache(char const* branchNames) { treeCacheManager_->trainCache(branchNames); }

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
      std::unique_ptr<TTreeCache> treeCache = file.createCacheWithSize(tree, cacheSize);
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
