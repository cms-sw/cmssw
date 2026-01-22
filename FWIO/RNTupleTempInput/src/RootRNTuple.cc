#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Common/interface/getWrapperBasePtr.h"

#include "InputFile.h"
#include "RootRNTuple.h"
#include "RootDelayedReader.h"
#include "RootPromptReadDelayedReader.h"

#include "oneapi/tbb/task_arena.h"
#include <cassert>

namespace edm::rntuple_temp {
  namespace {
    ROOT::DescriptorId_t getAuxiliaryFieldId(ROOT::RNTupleReader& reader, BranchType const& branchType) {
      return reader.GetDescriptor().FindFieldId(BranchTypeToAuxiliaryBranchName(branchType));
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
                           bool promptRead,
                           InputType inputType)
      : filePtr_(filePtr),
        branchType_(branchType),
        entryNumberForIndex_(std::make_unique<std::vector<EntryNumber>>(nIndexes, IndexIntoFile::invalidEntry)),
        promptRead_(promptRead),
        rootDelayedReader_(makeRootDelayedReader(*this, filePtr, inputType, nIndexes, promptRead)) {}

  // Used for Event/Lumi/Run RootRNTuples
  RootRNTuple::RootRNTuple(std::shared_ptr<InputFile> filePtr,
                           BranchType const& branchType,
                           unsigned int nIndexes,
                           Options const& options,
                           InputType inputType)
      : RootRNTuple(filePtr, branchType, nIndexes, options.promptReading, inputType) {
    init(BranchTypeToProductTreeName(branchType), options);
    auxDesc_ = getAuxiliaryFieldId(*reader_, branchType_);
  }

  // Used for ProcessBlock RootRNTuples
  RootRNTuple::RootRNTuple(std::shared_ptr<InputFile> filePtr,
                           BranchType const& branchType,
                           std::string const& processName,
                           unsigned int nIndexes,
                           Options const& options,
                           InputType inputType)
      : RootRNTuple(filePtr, branchType, nIndexes, options.promptReading, inputType) {
    processName_ = processName;
    init(BranchTypeToProductTreeName(branchType, processName), options);
  }

  void RootRNTuple::init(std::string const& productTreeName, Options const& options) {
    if (filePtr_.get() != nullptr) {
      auto tuple = filePtr_->Get<ROOT::RNTuple>(productTreeName.c_str());
      if (tuple != nullptr) {
        ROOT::RNTupleReadOptions rntupleOptions;
        rntupleOptions.SetClusterCache(options.useClusterCache ? ROOT::RNTupleReadOptions::EClusterCache::kOn
                                                               : ROOT::RNTupleReadOptions::EClusterCache::kOff);
        rntupleOptions.SetUseImplicitMT(options.enableIMT ? ROOT::RNTupleReadOptions::EImplicitMT::kDefault
                                                          : ROOT::RNTupleReadOptions::EImplicitMT::kOff);
        reader_ = ROOT::RNTupleReader::Open(*tuple, rntupleOptions);
      }
    }
    if (not reader_) {
      throw cms::Exception("WrongFileFormat")
          << "The ROOT file does not contain a RNTuple named " << productTreeName
          << "\n This is either not an edm RNTuple ROOT file or is one that has been corrupted.";
    }
    entries_ = reader_->GetNEntries();
  }

  RootRNTuple::~RootRNTuple() {}

  std::optional<ROOT::RNTupleView<void>> RootRNTuple::view(std::string_view iName) {
    auto id = reader_->GetDescriptor().FindFieldId(iName);
    if (id == ROOT::kInvalidDescriptorId) {
      return std::nullopt;
    }
    return reader_->GetView<void>(id, std::shared_ptr<void>());
  }

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
      return bool(reader_);
    }
    // Run/Lumi/Event
    return bool(reader_) && auxDesc_ != ROOT::kInvalidDescriptorId;
  }

  DelayedReader* RootRNTuple::resetAndGetRootDelayedReader() const {
    rootDelayedReader_->reset();
    return rootDelayedReader_.get();
  }

  RootDelayedReaderBase* RootRNTuple::rootDelayedReader() const { return rootDelayedReader_.get(); }

  void RootRNTuple::setPresence(ProductDescription& prod, std::string const& oldBranchName) {
    assert(isValid());
    if (reader_->GetDescriptor().FindFieldId(oldBranchName) == ROOT::kInvalidDescriptorId) {
      prod.setDropped(true);
    }
  }

  void rootrntuple::ProductInfo::setField(ROOT::RFieldToken token,
                                          ROOT::RNTupleView<void> view,
                                          TClass const* wrapperBaseTClass) {
    token_ = token;
    view_ = std::move(view);
    classCache_ = TClass::GetClass(productDescription_.wrappedName().c_str());
    offsetToWrapperBase_ = classCache_->GetBaseClassOffset(wrapperBaseTClass);
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
    auto id = reader_->GetDescriptor().FindFieldId(oldBranchName);
    rootrntuple::ProductInfo info(prod);
    if (prod.present() and id != ROOT::kInvalidDescriptorId) {
      info.setField(reader_->GetModel().GetToken(oldBranchName),
                    reader_->GetView<void>(id, std::shared_ptr<void>()),
                    wrapperBaseTClass);
      //we want the new branch name for the JobReport
      branchNames_.push_back(prod.branchName());
    }
    branches_.insert(prod.branchID(), std::move(info));
  }

  void RootRNTuple::dropBranch(std::string const& oldBranchName) {}

  rootrntuple::ProductMap const& RootRNTuple::branches() const { return branches_; }

  void RootRNTuple::setCacheSize(unsigned int cacheSize) {}

  void RootRNTuple::setTreeMaxVirtualSize(int treeMaxVirtualSize) {}

  bool RootRNTuple::nextWithCache() {
    bool returnValue = ++entryNumber_ < entries_;
    if (returnValue) {
      setEntryNumber(entryNumber_);
    }
    return returnValue;
  }

  void RootRNTuple::setEntryNumber(EntryNumber theEntryNumber) { entryNumber_ = theEntryNumber; }

  void RootRNTuple::getEntryForAllBranches(
      std::unordered_map<unsigned int, std::unique_ptr<edm::WrapperBase>>& iFields) const {
    LogTrace("IOTrace").format("RootRNTuple::getEntryForAllBranches() begin for entry {}", entryNumber_);
    oneapi::tbb::this_task_arena::isolate([&]() {
      auto entry = reader_->GetModel().CreateEntry();
      for (auto& iField : iFields) {
        auto const& prod = branches_.find(iField.first);
        if (prod == nullptr or not prod->valid()) {
          continue;
        }
        iField.second = prod->newWrapper();
        entry->BindRawPtr(prod->token(), reinterpret_cast<void*>(iField.second.get()));
      }
      reader_->LoadEntry(entryNumber_, *entry);
    });
    LogTrace("IOTrace").format("RootRNTuple::getEntryForAllBranches() end for entry {}", entryNumber_);
  }

  void RootRNTuple::getEntry(ROOT::RNTupleView<void>& view, EntryNumber entryNumber) const {
    LogTrace("IOTrace").format(
        "RootRNTuple::getEntryUsingCache() begin for branch {} entry {}", view.GetField().GetFieldName(), entryNumber);
    try {
      view(entryNumber);
    } catch (cms::Exception const& e) {
      // We make sure the treeCache_ is detached from the file,
      // so that ROOT does not also delete it.
      Exception t(errors::FileReadError, "", e);
      t.addContext(std::string("Reading branch ") + view.GetField().GetFieldName());
      throw t;
    } catch (std::exception const& e) {
      Exception t(errors::FileReadError);
      t << e.what();
      t.addContext(std::string("Reading branch ") + view.GetField().GetFieldName());
      throw t;
    } catch (...) {
      Exception t(errors::FileReadError);
      t << "An exception of unknown type was thrown.";
      t.addContext(std::string("Reading branch ") + view.GetField().GetFieldName());
      throw t;
    }
    LogTrace("IOTrace").format(
        "RootRNTuple::getEntryUsingCache() end for branch {} entry {}", view.GetField().GetFieldName(), entryNumber);
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
    auxDesc_ = ROOT::kInvalidDescriptorId;
    //reader_.reset(); //if there are any outstanding views, they will be invalidated
    // We give up our shared ownership of the TFile itself.
    filePtr_.reset();
  }

  void RootRNTuple::setSignals(
      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadSource,
      signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadSource) {
    rootDelayedReader_->setSignals(preEventReadSource, postEventReadSource);
  }

}  // namespace edm::rntuple_temp
