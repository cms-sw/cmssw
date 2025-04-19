#include "RNTupleInputFile.h"

#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/StoredMergeableRunProductMetadata.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/Provenance/interface/ProductDependencies.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ParentageRegistry.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/TimeOfDay.h"
#include "FWCore/Utilities/interface/ExceptionPropagate.h"

#include "ROOT/RNTupleReadOptions.hxx"

#include <cassert>
#include <iostream>
#include <iomanip>

using namespace ROOT::Experimental;
namespace {
  ROOT::Experimental::RNTupleReadOptions options(edm::RNTupleInputFile::Options const& iOpt) {
    ROOT::Experimental::RNTupleReadOptions opt;
    opt.SetMetricsEnabled(iOpt.enableMetrics_);
    opt.SetClusterCache(iOpt.useClusterCache_ ? RNTupleReadOptions::EClusterCache::kOn
                                              : RNTupleReadOptions::EClusterCache::kOff);
    return opt;
  }

  void logFileAction(std::string_view msg, std::string_view fileName) {
    edm::LogAbsolute("fileAction") << std::setprecision(0) << edm::TimeOfDay() << msg << fileName;
    edm::FlushMessageLog();
  }

  std::unique_ptr<TFile> open(std::string const& fileName) {
    logFileAction("  Initiating request to open ", fileName);
    std::unique_ptr<TFile> file;
    {
      // ROOT's context management implicitly assumes that a file is opened and
      // closed on the same thread.  To avoid the problem, we declare a local
      // TContext object; when it goes out of scope, its destructor unregisters
      // the context, guaranteeing the context is unregistered in the same thread
      // it was registered in.  Fixes issue #15524.
      TDirectory::TContext contextEraser;

      file = std::unique_ptr<TFile>(TFile::Open(fileName.c_str()));
    }
    std::exception_ptr e = edm::threadLocalException::getException();
    if (e != std::exception_ptr()) {
      edm::threadLocalException::setException(std::exception_ptr());
      std::rethrow_exception(e);
    }
    if (!file) {
      throw edm::Exception(edm::errors::FileOpenError) << "TFile::Open failed.";
    }
    if (file->IsZombie()) {
      throw edm::Exception(edm::errors::FileOpenError) << "TFile::Open returned zombie.";
    }

    logFileAction("  Successfully opened file ", fileName);
    return file;
  }
}  // namespace
namespace edm {

  RNTupleInputFile::RNTupleInputFile(std::string const& iName, Options const& iOpt)
      : file_(open(iName)),
        runs_(file_.get(), "Runs", "RunAuxiliary", {}),
        lumis_(file_.get(), "LuminosityBlocks", "LuminosityBlockAuxiliary", {}),
        events_(file_.get(), "Events", "EventAuxiliary", options(iOpt)) {}

  std::vector<ParentageID> RNTupleInputFile::readParentage() {
    auto parentageTuple = RNTupleReader::Open(*file_->Get<ROOT::RNTuple>("Parentage"));
    auto entry = parentageTuple->GetModel().CreateBareEntry();

    edm::Parentage parentage;
    entry->BindRawPtr("Description", &parentage);
    std::vector<ParentageID> retValue;

    ParentageRegistry& registry = *ParentageRegistry::instance();

    retValue.reserve(parentageTuple->GetNEntries());

    for (ROOT::Experimental::NTupleSize_t i = 0; i < parentageTuple->GetNEntries(); ++i) {
      parentageTuple->LoadEntry(i, *entry);
      registry.insertMapped(parentage);
      retValue.push_back(parentage.id());
    }

    return retValue;
  }

  void RNTupleInputFile::readParameterSets() {
    auto psets = RNTupleReader::Open(*file_->Get<ROOT::RNTuple>("ParameterSets"));
    assert(psets.get());
    auto entry = psets->GetModel().CreateBareEntry();

    std::pair<ParameterSetID, ParameterSetBlob> idToBlob;
    entry->BindRawPtr("IdToParameterSetsBlobs", &idToBlob);

    // Merge into the parameter set registry.
    pset::Registry& psetRegistry = *pset::Registry::instance();
    for (ROOT::Experimental::NTupleSize_t i = 0; i < psets->GetNEntries(); ++i) {
      psets->LoadEntry(i, *entry);
      ParameterSet pset(idToBlob.second.pset());
      pset.setID(idToBlob.first);
      psetRegistry.insertMapped(pset);
    }
  }

  void RNTupleInputFile::readMeta(edm::ProductRegistry& iReg,
                                  edm::ProcessHistoryRegistry& iHist,
                                  BranchIDLists& iBranchIDLists) {
    auto meta = RNTupleReader::Open(*file_->Get<ROOT::RNTuple>("MetaData"));
    assert(meta.get());

    //BEWARE, if you do not 'BindRawPtr' to all top level Fields,
    // using CreateBareEntry with LoadEntry will seg-fault!
    //auto entry = meta->GetModel().CreateBareEntry();
    auto entry = meta->GetModel().CreateEntry();

    entry->BindRawPtr("IndexIntoFile", &index_);

    edm::FileID id;
    entry->BindRawPtr("FileIdentifier", &id);

    edm::StoredMergeableRunProductMetadata mergeable;
    entry->BindRawPtr("MergeableRunProductMetadata", &mergeable);

    std::vector<edm::ProcessHistory> processHist;
    entry->BindRawPtr("ProcessHistory", &processHist);

    edm::ProductRegistry reg;
    entry->BindRawPtr("ProductRegistry", &reg);

    entry->BindRawPtr("BranchIDLists", &iBranchIDLists);

    edm::ThinnedAssociationsHelper thinned;
    entry->BindRawPtr("ThinnedAssociationsHelper", &thinned);

    edm::ProductDependencies productDependencies;
    entry->BindRawPtr("ProductDependencies", &productDependencies);

    meta->LoadEntry(0, *entry);

    {
      auto& pList = reg.productListUpdator();
      for (auto& product : pList) {
        ProductDescription& prod = product.second;
        prod.initBranchName();
        if (not prod.present())
          continue;
        if (prod.branchType() == InEvent) {
          prod.setDropped(not events_.setupToReadProductIfAvailable(prod));
        } else if (prod.branchType() == InLumi) {
          prod.setDropped(not lumis_.setupToReadProductIfAvailable(prod));
        } else if (prod.branchType() == InRun) {
          prod.setDropped(not runs_.setupToReadProductIfAvailable(prod));
        }
      }
    }
    reg.setFrozen(false);
    iReg.updateFromInput(reg.productList());

    for (auto const& h : processHist) {
      iHist.registerProcessHistory(h);
    }

    std::vector<ProcessHistoryID> orderedHistory;
    index_.fixIndexes(orderedHistory);
    index_.setNumberOfEvents(events_.numberOfEntries());
    //index_.setEventFinder();
    bool needEventNumbers = false;
    bool needEventEntries = false;
    index_.fillEventNumbersOrEntries(needEventNumbers, needEventEntries);

    iter_ = index_.begin(IndexIntoFile::firstAppearanceOrder);
    iterEnd_ = index_.end(IndexIntoFile::firstAppearanceOrder);
  }

  IndexIntoFile::EntryType RNTupleInputFile::getNextItemType() {
    if (*iter_ == *iterEnd_) {
      return IndexIntoFile::kEnd;
    }
    return iter_->getEntryType();
  }

  int RNTupleInputFile::skipEvents(int offset) {
    while (offset > 0 && *iter_ != iterEnd_) {
      int phIndexOfSkippedEvent = IndexIntoFile::invalidIndex;
      RunNumber_t runOfSkippedEvent = IndexIntoFile::invalidRun;
      LuminosityBlockNumber_t lumiOfSkippedEvent = IndexIntoFile::invalidLumi;
      IndexIntoFile::EntryNumber_t skippedEventEntry = IndexIntoFile::invalidEntry;

      iter_->skipEventForward(phIndexOfSkippedEvent, runOfSkippedEvent, lumiOfSkippedEvent, skippedEventEntry);

      // At the end of the file and there were no more events to skip
      if (skippedEventEntry == IndexIntoFile::invalidEntry)
        break;
      /* once we add this feature we will need this code
      if (eventSkipperByID_ && eventSkipperByID_->somethingToSkip()) {
        auto const evtAux = fillEventAuxiliary(skippedEventEntry);
        if (eventSkipperByID_->skipIt(runOfSkippedEvent, lumiOfSkippedEvent, evtAux.id().event())) {
          continue;
        }
      }
      if (duplicateChecker_ && !duplicateChecker_->checkDisabled() && !duplicateChecker_->noDuplicatesInFile()) {
        auto const evtAux = fillEventAuxiliary(skippedEventEntry);
        if (duplicateChecker_->isDuplicateAndCheckActive(
                phIndexOfSkippedEvent, runOfSkippedEvent, lumiOfSkippedEvent, evtAux.id().event(), file_)) {
          continue;
        }
      }
      */
      --offset;
    }
    while (offset < 0) {
      /*
      if (duplicateChecker_) {
        duplicateChecker_->disable();
      }
      */

      int phIndexOfEvent = IndexIntoFile::invalidIndex;
      RunNumber_t runOfEvent = IndexIntoFile::invalidRun;
      LuminosityBlockNumber_t lumiOfEvent = IndexIntoFile::invalidLumi;
      IndexIntoFile::EntryNumber_t eventEntry = IndexIntoFile::invalidEntry;

      iter_->skipEventBackward(phIndexOfEvent, runOfEvent, lumiOfEvent, eventEntry);

      if (eventEntry == IndexIntoFile::invalidEntry)
        break;
      /* once we add this feature we will need this code
      if (eventSkipperByID_ && eventSkipperByID_->somethingToSkip()) {
        auto const evtAux = fillEventAuxiliary(eventEntry);
        if (eventSkipperByID_->skipIt(runOfEvent, lumiOfEvent, evtAux.id().event())) {
          continue;
        }
      }
      */
      ++offset;
    }

    return offset;
  }

  IndexIntoFile::EntryNumber_t RNTupleInputFile::readLuminosityBlock() {
    assert(*iter_ != *iterEnd_);
    assert(iter_->getEntryType() == IndexIntoFile::kLumi);
    auto v = iter_->entry();
    ++(*iter_);
    return v;
  }

  std::shared_ptr<LuminosityBlockAuxiliary> RNTupleInputFile::readLuminosityBlockAuxiliary() {
    auto lumiAux = std::make_shared<LuminosityBlockAuxiliary>();
    assert(*iter_ != *iterEnd_);
    assert(iter_->getEntryType() == IndexIntoFile::kLumi);
    if (!lumiAuxView_) {
      lumiAuxView_ = lumis_.auxView(lumiAux);
    } else {
      lumiAuxView_->Bind(lumiAux);
    }
    (*lumiAuxView_)(iter_->entry());
    return lumiAux;
  }

  IndexIntoFile::EntryNumber_t RNTupleInputFile::readEvent() {
    assert(*iter_ != *iterEnd_);
    assert(iter_->getEntryType() == IndexIntoFile::kEvent);
    auto v = iter_->entry();
    ++(*iter_);
    return v;
  }

  std::shared_ptr<EventAuxiliary> RNTupleInputFile::readEventAuxiliary() {
    auto eventAux = std::make_shared<EventAuxiliary>();
    assert(*iter_ != *iterEnd_);
    assert(iter_->getEntryType() == IndexIntoFile::kEvent);
    if (!eventAuxView_) {
      eventAuxView_ = events_.auxView(eventAux);
    } else {
      eventAuxView_->Bind(eventAux);
    }
    (*eventAuxView_)(iter_->entry());
    return eventAux;
  }

  std::shared_ptr<RunAuxiliary> RNTupleInputFile::readRunAuxiliary() {
    auto runAux = std::make_shared<RunAuxiliary>();
    assert(*iter_ != *iterEnd_);
    assert(iter_->getEntryType() == IndexIntoFile::kRun);

    if (!runAuxView_) {
      runAuxView_ = runs_.auxView(runAux);
    } else {
      runAuxView_->Bind(runAux);
    }
    (*runAuxView_)(iter_->entry());
    return runAux;
  }

  IndexIntoFile::EntryNumber_t RNTupleInputFile::readRun() {
    assert(*iter_ != *iterEnd_);
    assert(iter_->getEntryType() == IndexIntoFile::kRun);
    auto v = iter_->entry();
    ++(*iter_);
    return v;
  }

}  // namespace edm
