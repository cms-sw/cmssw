#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Common/interface/getWrapperBasePtr.h"

#include "InputFile.h"
#include "RootRNTuple.h"
#include "RootDelayedReader.h"
#include "RootPromptReadDelayedReader.h"

#include "TTree.h"
#include "TLeaf.h"

#include "oneapi/tbb/task_arena.h"
#include <cassert>

namespace edm::rntuple_temp {
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

    std::unique_ptr<RootDelayedReaderBase> makeRootDelayedReader(RootRNTuple const& tree,
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

  // Used for all RootRNTuples
  // All the other constructors delegate to this one
  RootRNTuple::RootRNTuple(std::shared_ptr<InputFile> filePtr,
                           BranchType const& branchType,
                           unsigned int nIndexes,
                           unsigned int learningEntries,
                           bool enablePrefetching,
                           bool promptRead,
                           InputType inputType)
      : filePtr_(filePtr),
        branchType_(branchType),
        entryNumberForIndex_(std::make_unique<std::vector<EntryNumber>>(nIndexes, IndexIntoFile::invalidEntry)),
        enablePrefetching_(enablePrefetching),
        promptRead_(promptRead),
        rootDelayedReader_(makeRootDelayedReader(*this, filePtr, inputType, nIndexes, promptRead)) {}

  // Used for Event/Lumi/Run RootRNTuples
  RootRNTuple::RootRNTuple(std::shared_ptr<InputFile> filePtr,
                           BranchType const& branchType,
                           unsigned int nIndexes,
                           Options const& options,
                           unsigned int learningEntries,
                           InputType inputType)
      : RootRNTuple(
            filePtr, branchType, nIndexes, learningEntries, options.enablePrefetching, options.promptReading, inputType) {
    init(BranchTypeToProductTreeName(branchType), options.treeMaxVirtualSize, options.treeCacheSize);
    auxBranch_ = getAuxiliaryBranch(tree_, branchType_);
  }

  // Used for ProcessBlock RootRNTuples
  RootRNTuple::RootRNTuple(std::shared_ptr<InputFile> filePtr,
                           BranchType const& branchType,
                           std::string const& processName,
                           unsigned int nIndexes,
                           Options const& options,
                           unsigned int learningEntries,
                           InputType inputType)
      : RootRNTuple(
            filePtr, branchType, nIndexes, learningEntries, options.enablePrefetching, options.promptReading, inputType) {
    processName_ = processName;
    init(BranchTypeToProductTreeName(branchType, processName), options.treeMaxVirtualSize, options.treeCacheSize);
  }

  void RootRNTuple::init(std::string const& productTreeName, unsigned int maxVirtualSize, unsigned int cacheSize) {
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
    setTreeMaxVirtualSize(maxVirtualSize);
    setCacheSize(cacheSize);
  }

  RootRNTuple::~RootRNTuple() {}

  RootRNTuple::EntryNumber const& RootRNTuple::entryNumberForIndex(unsigned int index) const {
    assert(index < entryNumberForIndex_->size());
    return (*entryNumberForIndex_)[index];
  }

  void RootRNTuple::insertEntryForIndex(unsigned int index) {
    assert(index < entryNumberForIndex_->size());
    (*entryNumberForIndex_)[index] = entryNumber();
  }

  bool RootRNTuple::isValid() const {
    // ProcessBlock
    if (branchType_ == InProcess) {
      return tree_ != nullptr;
    }
    // Run/Lumi/Event
    return tree_ != nullptr && auxBranch_ != nullptr;
  }

  DelayedReader* RootRNTuple::resetAndGetRootDelayedReader() const {
    rootDelayedReader_->reset();
    return rootDelayedReader_.get();
  }

  RootDelayedReaderBase* RootRNTuple::rootDelayedReader() const { return rootDelayedReader_.get(); }

  void RootRNTuple::setPresence(ProductDescription& prod, std::string const& oldBranchName) {
    assert(isValid());
    if (tree_->GetBranch(oldBranchName.c_str()) == nullptr) {
      prod.setDropped(true);
    }
  }

  void rootrntuple::ProductInfo::setBranch(TBranch* branch, TClass const* wrapperBaseTClass) {
    productBranch_ = branch;
    if (branch) {
      classCache_ = TClass::GetClass(productDescription_.wrappedName().c_str());
      offsetToWrapperBase_ = classCache_->GetBaseClassOffset(wrapperBaseTClass);
    }
  }
  std::unique_ptr<WrapperBase> rootrntuple::ProductInfo::newWrapper() const {
    assert(nullptr != classCache_);
    void* p = classCache_->New();
    return getWrapperBasePtr(p, offsetToWrapperBase_);
  }

  void RootRNTuple::addBranch(ProductDescription const& prod, std::string const& oldBranchName) {
    assert(isValid());
    static TClass const* const wrapperBaseTClass = TClass::GetClass("edm::WrapperBase");
    //use the translated branch name
    TBranch* branch = tree_->GetBranch(oldBranchName.c_str());
    rootrntuple::ProductInfo info = rootrntuple::ProductInfo(prod);
    info.productBranch_ = nullptr;
    if (prod.present()) {
      info.setBranch(branch, wrapperBaseTClass);
      //we want the new branch name for the JobReport
      branchNames_.push_back(prod.branchName());
    }
    branches_.insert(prod.branchID(), info);
  }

  void RootRNTuple::dropBranch(std::string const& oldBranchName) {
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

  rootrntuple::ProductMap const& RootRNTuple::branches() const { return branches_; }

  void RootRNTuple::setCacheSize(unsigned int cacheSize) {}

  void RootRNTuple::setTreeMaxVirtualSize(int treeMaxVirtualSize) {
    if (treeMaxVirtualSize >= 0)
      tree_->SetMaxVirtualSize(static_cast<Long64_t>(treeMaxVirtualSize));
  }

  bool RootRNTuple::nextWithCache() {
    bool returnValue = ++entryNumber_ < entries_;
    if (returnValue) {
      setEntryNumber(entryNumber_);
    }
    return returnValue;
  }

  void RootRNTuple::setEntryNumber(EntryNumber theEntryNumber) {
    {
      entryNumber_ = theEntryNumber;
      tree_->LoadTree(entryNumber_);
      //want guard to end here
    }
  }

  void RootRNTuple::getEntryForAllBranches() const {
    oneapi::tbb::this_task_arena::isolate([&]() { tree_->GetEntry(entryNumber_); });
  }

  void RootRNTuple::getEntry(TBranch* branch, EntryNumber entryNumber) const {
    LogTrace("IOTrace").format(
        "RootRNTuple::getEntryUsingCache() begin for branch {} entry {}", branch->GetName(), entryNumber);
    try {
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
        "RootRNTuple::getEntryUsingCache() end for branch {} entry {}", branch->GetName(), entryNumber);
  }

  bool RootRNTuple::skipEntries(unsigned int& offset) {
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

  void RootRNTuple::close() {
    // The TFile is about to be closed, and destructed.
    // Just to play it safe, zero all pointers to quantities that are owned by the TFile.
    auxBranch_ = nullptr;
    tree_ = nullptr;
    // We give up our shared ownership of the TFile itself.
    filePtr_.reset();
  }

  void RootRNTuple::setSignals(
      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadSource,
      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadSource) {
    rootDelayedReader_->setSignals(preEventReadSource, postEventReadSource);
  }

  namespace rootrntuple {
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
  }  // namespace rootrntuple
}  // namespace edm::rntuple_temp
