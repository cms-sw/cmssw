// -*- C++ -*-
//
// Package:     FWLite
// Class  :     Event
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue May  8 15:07:03 EDT 2007
//

// system include files
#include <iostream>

// user include files
#include "DataFormats/FWLite/interface/Event.h"
#include "TFile.h"
#include "TTree.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Common/interface/EDProductGetter.h"

#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include "FWCore/FWLite/interface/setRefStreamer.h"

#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/interface/ParameterSetConverter.h"
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "DataFormats/FWLite/interface/EventHistoryGetter.h"
#include "DataFormats/FWLite/interface/RunFactory.h"

//used for backwards compatability
#include "DataFormats/Provenance/interface/EventAux.h"

//
// constants, enums and typedefs
//
namespace {
  struct NoDelete {
    void operator()(void*) {}
  };
}  // namespace

namespace fwlite {
  //
  // static data member definitions
  //
  namespace internal {
    class ProductGetter : public edm::EDProductGetter {
    public:
      ProductGetter(Event* iEvent) : event_(iEvent) {}

      edm::WrapperBase const* getIt(edm::ProductID const& iID) const override { return event_->getByProductID(iID); }

      // getThinnedProduct assumes getIt was already called and failed to find
      // the product. The input key is the index of the desired element in the
      // container identified by ProductID (which cannot be found).
      // If the return value is not null, then the desired element was
      // found in a thinned container. If the desired element is not
      // found, then an optional without a value is returned.
      std::optional<std::tuple<edm::WrapperBase const*, unsigned int>> getThinnedProduct(
          edm::ProductID const& pid, unsigned int key) const override {
        return event_->getThinnedProduct(pid, key);
      }

      // getThinnedProducts assumes getIt was already called and failed to find
      // the product. The input keys are the indexes into the container identified
      // by ProductID (which cannot be found). On input the WrapperBase pointers
      // must all be set to nullptr (except when the function calls itself
      // recursively where non-null pointers mark already found elements).
      // Thinned containers derived from the product are searched to see
      // if they contain the desired elements. For each that is
      // found, the corresponding WrapperBase pointer is set and the key
      // is modified to be the key into the container where the element
      // was found. The WrapperBase pointers might or might not all point
      // to the same thinned container.
      void getThinnedProducts(edm::ProductID const& pid,
                              std::vector<edm::WrapperBase const*>& foundContainers,
                              std::vector<unsigned int>& keys) const override {
        event_->getThinnedProducts(pid, foundContainers, keys);
      }

      // This overload is allowed to be called also without getIt()
      // being called first, but the thinned ProductID must come from an
      // existing RefCore. The input key is the index of the desired
      // element in the container identified by the parent ProductID.
      // If the return value is not null, then the desired element was found
      // in a thinned container. If the desired element is not found, then
      // an optional without a value is returned.
      edm::OptionalThinnedKey getThinnedKeyFrom(edm::ProductID const& parent,
                                                unsigned int key,
                                                edm::ProductID const& thinned) const override {
        return event_->getThinnedKeyFrom(parent, key, thinned);
      }

    private:
      unsigned int transitionIndex_() const override { return 0U; }

      Event const* event_;
    };
  }  // namespace internal
     //
     // constructors and destructor
     //
  Event::Event(TFile* iFile, bool useCache, std::function<void(TBranch const&)> baFunc)
      : file_(iFile),
        //  eventTree_(nullptr),
        eventHistoryTree_(nullptr),
        //  eventIndex_(-1),
        branchMap_(iFile),
        pAux_(&aux_),
        pOldAux_(nullptr),
        fileVersion_(-1),
        parameterSetRegistryFilled_(false),
        dataHelper_(branchMap_.getEventTree(),
                    std::make_shared<EventHistoryGetter>(this),
                    std::shared_ptr<BranchMapReader>(&branchMap_, NoDelete()),
                    std::make_shared<internal::ProductGetter>(this),
                    useCache,
                    baFunc) {
    if (nullptr == iFile) {
      throw cms::Exception("NoFile") << "The TFile pointer passed to the constructor was null";
    }

    if (nullptr == branchMap_.getEventTree()) {
      throw cms::Exception("NoEventTree") << "The TFile contains no TTree named " << edm::poolNames::eventTreeName();
    }
    //need to know file version in order to determine how to read the basic event info
    fileVersion_ = branchMap_.getFileVersion(iFile);

    //got this logic from IOPool/Input/src/RootFile.cc

    TTree* eventTree = branchMap_.getEventTree();
    if (fileVersion_ >= 3) {
      auxBranch_ = eventTree->GetBranch(edm::BranchTypeToAuxiliaryBranchName(edm::InEvent).c_str());
      if (nullptr == auxBranch_) {
        throw cms::Exception("NoEventAuxilliary")
            << "The TTree " << edm::poolNames::eventTreeName() << " does not contain a branch named 'EventAuxiliary'";
      }
      auxBranch_->SetAddress(&pAux_);
    } else {
      pOldAux_ = new edm::EventAux();
      auxBranch_ = eventTree->GetBranch(edm::BranchTypeToAuxBranchName(edm::InEvent).c_str());
      if (nullptr == auxBranch_) {
        throw cms::Exception("NoEventAux")
            << "The TTree " << edm::poolNames::eventTreeName() << " does not contain a branch named 'EventAux'";
      }
      auxBranch_->SetAddress(&pOldAux_);
    }
    branchMap_.updateEvent(0);

    if (fileVersion_ >= 7 && fileVersion_ < 17) {
      eventHistoryTree_ = dynamic_cast<TTree*>(iFile->Get(edm::poolNames::eventHistoryTreeName().c_str()));
    }
    runFactory_ = std::make_shared<RunFactory>();
  }

  // Event::Event(Event const& rhs)
  // {
  //    // do actual copying here;
  // }

  Event::~Event() {
    for (auto const& label : labels_) {
      delete[] label;
    }
    delete pOldAux_;
  }

  //
  // assignment operators
  //
  // Event const& Event::operator=(Event const& rhs) {
  //   //An exception safe implementation is
  //   Event temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }

  //
  // member functions
  //

  Event const& Event::operator++() {
    Long_t eventIndex = branchMap_.getEventEntry();
    if (eventIndex < size()) {
      branchMap_.updateEvent(++eventIndex);
    }
    return *this;
  }

  Long64_t Event::indexFromEventId(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, edm::EventNumber_t event) {
    entryFinder_.fillIndex(branchMap_);
    EntryFinder::EntryNumber_t entry = entryFinder_.findEvent(run, lumi, event);
    return (entry == EntryFinder::invalidEntry) ? -1 : entry;
  }

  bool Event::to(Long64_t iEntry) {
    if (iEntry < size()) {
      // this is a valid entry
      return branchMap_.updateEvent(iEntry);
    }
    // if we're here, then iEntry was not valid
    return false;
  }

  bool Event::to(edm::RunNumber_t run, edm::EventNumber_t event) { return to(run, 0U, event); }

  bool Event::to(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, edm::EventNumber_t event) {
    entryFinder_.fillIndex(branchMap_);
    EntryFinder::EntryNumber_t entry = entryFinder_.findEvent(run, lumi, event);
    if (entry == EntryFinder::invalidEntry) {
      return false;
    }
    return branchMap_.updateEvent(entry);
  }

  bool Event::to(const edm::EventID& id) { return to(id.run(), id.luminosityBlock(), id.event()); }

  Event const& Event::toBegin() {
    branchMap_.updateEvent(0);
    return *this;
  }

  //
  // const member functions
  //
  void Event::draw(Option_t* opt) {
    GetterOperate op(dataHelper_.getter());
    branchMap_.getEventTree()->Draw(opt);
  }
  Long64_t Event::draw(
      char const* varexp, const TCut& selection, Option_t* option, Long64_t nentries, Long64_t firstentry) {
    GetterOperate op(dataHelper_.getter());
    return branchMap_.getEventTree()->Draw(varexp, selection, option, nentries, firstentry);
  }
  Long64_t Event::draw(
      char const* varexp, char const* selection, Option_t* option, Long64_t nentries, Long64_t firstentry) {
    GetterOperate op(dataHelper_.getter());
    return branchMap_.getEventTree()->Draw(varexp, selection, option, nentries, firstentry);
  }
  Long64_t Event::scan(
      char const* varexp, char const* selection, Option_t* option, Long64_t nentries, Long64_t firstentry) {
    GetterOperate op(dataHelper_.getter());
    return branchMap_.getEventTree()->Scan(varexp, selection, option, nentries, firstentry);
  }

  Long64_t Event::size() const { return branchMap_.getEventTree()->GetEntries(); }

  bool Event::isValid() const {
    Long_t eventIndex = branchMap_.getEventEntry();
    return eventIndex != -1 and eventIndex < size();
  }

  Event::operator bool() const { return isValid(); }

  bool Event::atEnd() const {
    Long_t eventIndex = branchMap_.getEventEntry();
    return eventIndex == -1 or eventIndex == size();
  }

  std::vector<std::string> const& Event::getProcessHistory() const {
    if (procHistoryNames_.empty()) {
      for (auto const& proc : history()) {
        procHistoryNames_.push_back(proc.processName());
      }
    }
    return procHistoryNames_;
  }

  std::string const Event::getBranchNameFor(std::type_info const& iInfo,
                                            char const* iModuleLabel,
                                            char const* iProductInstanceLabel,
                                            char const* iProcessLabel) const {
    return dataHelper_.getBranchNameFor(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel);
  }

  bool Event::getByLabel(std::type_info const& iInfo,
                         char const* iModuleLabel,
                         char const* iProductInstanceLabel,
                         char const* iProcessLabel,
                         void* oData) const {
    if (atEnd()) {
      throw cms::Exception("OffEnd") << "You have requested data past the last event";
    }
    Long_t eventIndex = branchMap_.getEventEntry();
    return dataHelper_.getByLabel(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel, oData, eventIndex);
  }

  bool Event::getByTokenImp(edm::EDGetToken iToken, edm::WrapperBase const*& oData) const {
    if (atEnd()) {
      throw cms::Exception("OffEnd") << "You have requested data past the last event";
    }
    Long_t eventIndex = branchMap_.getEventEntry();
    oData = dataHelper_.getByBranchID(edm::BranchID(iToken.index()), eventIndex);
    return oData != nullptr;
  }

  edm::EventAuxiliary const& Event::eventAuxiliary() const {
    Long_t eventIndex = branchMap_.getEventEntry();
    updateAux(eventIndex);
    return aux_;
  }

  void Event::updateAux(Long_t eventIndex) const {
    if (auxBranch_->GetEntryNumber() != eventIndex) {
      auxBranch_->GetEntry(eventIndex);
      //handling dealing with old version
      if (nullptr != pOldAux_) {
        conversion(*pOldAux_, aux_);
      }
    }
  }

  const edm::ProcessHistory& Event::history() const {
    edm::ProcessHistoryID processHistoryID;

    bool newFormat = (fileVersion_ >= 5);

    Long_t eventIndex = branchMap_.getEventEntry();
    updateAux(eventIndex);
    if (!newFormat) {
      processHistoryID = aux_.processHistoryID();
    }
    if (historyMap_.empty() || newFormat) {
      procHistoryNames_.clear();
      TTree* meta = dynamic_cast<TTree*>(branchMap_.getFile()->Get(edm::poolNames::metaDataTreeName().c_str()));
      if (nullptr == meta) {
        throw cms::Exception("NoMetaTree")
            << "The TFile does not appear to contain a TTree named " << edm::poolNames::metaDataTreeName();
      }
      if (historyMap_.empty()) {
        if (fileVersion_ < 11) {
          edm::ProcessHistoryMap* pPhm = &historyMap_;
          TBranch* b = meta->GetBranch(edm::poolNames::processHistoryMapBranchName().c_str());
          b->SetAddress(&pPhm);
          b->GetEntry(0);
        } else {
          edm::ProcessHistoryVector historyVector;
          edm::ProcessHistoryVector* pPhv = &historyVector;
          TBranch* b = meta->GetBranch(edm::poolNames::processHistoryBranchName().c_str());
          b->SetAddress(&pPhv);
          b->GetEntry(0);
          for (auto& history : historyVector) {
            historyMap_.insert(std::make_pair(history.setProcessHistoryID(), history));
          }
        }
      }
      if (newFormat) {
        if (fileVersion_ >= 17) {
          processHistoryID = aux_.processHistoryID();
        } else if (fileVersion_ >= 7) {
          edm::History history;
          edm::History* pHistory = &history;
          TBranch* eventHistoryBranch = eventHistoryTree_->GetBranch(edm::poolNames::eventHistoryBranchName().c_str());
          if (!eventHistoryBranch)
            throw edm::Exception(edm::errors::FatalRootError) << "Failed to find history branch in event history tree";
          eventHistoryBranch->SetAddress(&pHistory);
          eventHistoryTree_->GetEntry(eventIndex);
          processHistoryID = history.processHistoryID();
        } else {
          std::vector<edm::EventProcessHistoryID>* pEventProcessHistoryIDs = &eventProcessHistoryIDs_;
          TBranch* b = meta->GetBranch(edm::poolNames::eventHistoryBranchName().c_str());
          b->SetAddress(&pEventProcessHistoryIDs);
          b->GetEntry(0);
          edm::EventProcessHistoryID target(aux_.id(), edm::ProcessHistoryID());
          processHistoryID = std::lower_bound(eventProcessHistoryIDs_.begin(), eventProcessHistoryIDs_.end(), target)
                                 ->processHistoryID();
        }
      }
    }

    return historyMap_[processHistoryID];
  }

  edm::WrapperBase const* Event::getByProductID(edm::ProductID const& iID) const {
    Long_t eventEntry = branchMap_.getEventEntry();
    return dataHelper_.getByProductID(iID, eventEntry);
  }

  std::optional<std::tuple<edm::WrapperBase const*, unsigned int>> Event::getThinnedProduct(edm::ProductID const& pid,
                                                                                            unsigned int key) const {
    Long_t eventEntry = branchMap_.getEventEntry();
    return dataHelper_.getThinnedProduct(pid, key, eventEntry);
  }

  void Event::getThinnedProducts(edm::ProductID const& pid,
                                 std::vector<edm::WrapperBase const*>& foundContainers,
                                 std::vector<unsigned int>& keys) const {
    Long_t eventEntry = branchMap_.getEventEntry();
    return dataHelper_.getThinnedProducts(pid, foundContainers, keys, eventEntry);
  }

  edm::OptionalThinnedKey Event::getThinnedKeyFrom(edm::ProductID const& parent,
                                                   unsigned int key,
                                                   edm::ProductID const& thinned) const {
    Long_t eventEntry = branchMap_.getEventEntry();
    return dataHelper_.getThinnedKeyFrom(parent, key, thinned, eventEntry);
  }

  edm::TriggerNames const& Event::triggerNames(edm::TriggerResults const& triggerResults) const {
    edm::TriggerNames const* names = triggerNames_(triggerResults);
    if (names != nullptr)
      return *names;

    if (!parameterSetRegistryFilled_) {
      fillParameterSetRegistry();
      names = triggerNames_(triggerResults);
    }
    if (names != nullptr)
      return *names;

    throw cms::Exception("TriggerNamesNotFound") << "TriggerNames not found in ParameterSet registry";
    return *names;
  }

  void Event::fillParameterSetRegistry() const {
    if (parameterSetRegistryFilled_)
      return;
    parameterSetRegistryFilled_ = true;

    TTree* meta = dynamic_cast<TTree*>(branchMap_.getFile()->Get(edm::poolNames::metaDataTreeName().c_str()));
    if (nullptr == meta) {
      throw cms::Exception("NoMetaTree") << "The TFile does not contain a TTree named "
                                         << edm::poolNames::metaDataTreeName();
    }

    edm::FileFormatVersion fileFormatVersion;
    edm::FileFormatVersion* fftPtr = &fileFormatVersion;
    if (meta->FindBranch(edm::poolNames::fileFormatVersionBranchName().c_str()) != nullptr) {
      TBranch* fft = meta->GetBranch(edm::poolNames::fileFormatVersionBranchName().c_str());
      fft->SetAddress(&fftPtr);
      fft->GetEntry(0);
    }

    typedef std::map<edm::ParameterSetID, edm::ParameterSetBlob> PsetMap;
    PsetMap psetMap;
    TTree* psetTree(nullptr);
    if (meta->FindBranch(edm::poolNames::parameterSetMapBranchName().c_str()) != nullptr) {
      PsetMap* psetMapPtr = &psetMap;
      TBranch* b = meta->GetBranch(edm::poolNames::parameterSetMapBranchName().c_str());
      b->SetAddress(&psetMapPtr);
      b->GetEntry(0);
    } else if (nullptr == (psetTree = dynamic_cast<TTree*>(
                               branchMap_.getFile()->Get(edm::poolNames::parameterSetsTreeName().c_str())))) {
      throw cms::Exception("NoParameterSetMapTree")
          << "The TTree " << edm::poolNames::parameterSetsTreeName() << " could not be found in the file.";
    } else {
      typedef std::pair<edm::ParameterSetID, edm::ParameterSetBlob> IdToBlobs;
      IdToBlobs idToBlob;
      IdToBlobs* pIdToBlob = &idToBlob;
      psetTree->SetBranchAddress(edm::poolNames::idToParameterSetBlobsBranchName().c_str(), &pIdToBlob);
      for (long long i = 0; i != psetTree->GetEntries(); ++i) {
        psetTree->GetEntry(i);
        psetMap.insert(idToBlob);
      }
    }
    edm::ParameterSetConverter::ParameterSetIdConverter psetIdConverter;
    if (!fileFormatVersion.triggerPathsTracked()) {
      edm::ParameterSetConverter converter(psetMap, psetIdConverter, fileFormatVersion.parameterSetsByReference());
    } else {
      // Merge into the parameter set registry.
      edm::pset::Registry& psetRegistry = *edm::pset::Registry::instance();
      for (auto const& item : psetMap) {
        edm::ParameterSet pset(item.second.pset());
        pset.setID(item.first);
        psetRegistry.insertMapped(pset);
      }
    }
  }

  edm::ParameterSet const* Event::parameterSet(edm::ParameterSetID const& psID) const {
    if (!parameterSetRegistryFilled_) {
      fillParameterSetRegistry();
    }
    return parameterSetForID_(psID);
  }

  edm::TriggerResultsByName Event::triggerResultsByName(edm::TriggerResults const& triggerResults) const {
    edm::TriggerNames const* names = triggerNames_(triggerResults);
    if (names == nullptr && !parameterSetRegistryFilled_) {
      fillParameterSetRegistry();
      names = triggerNames_(triggerResults);
    }
    return edm::TriggerResultsByName(&triggerResults, names);
  }

  //
  // static member functions
  //
  void Event::throwProductNotFoundException(std::type_info const& iType,
                                            char const* iModule,
                                            char const* iProduct,
                                            char const* iProcess) {
    edm::TypeID type(iType);
    throw edm::Exception(edm::errors::ProductNotFound)
        << "A branch was found for \n  type ='" << type.className() << "'\n  module='" << iModule
        << "'\n  productInstance='" << ((nullptr != iProduct) ? iProduct : "") << "'\n  process='"
        << ((nullptr != iProcess) ? iProcess : "")
        << "'\n"
           "but no data is available for this Event";
  }

  fwlite::LuminosityBlock const& Event::getLuminosityBlock() const {
    if (not lumi_) {
      // Branch map pointer not really being shared, owned by event, have to trick Lumi
      lumi_ = std::make_shared<fwlite::LuminosityBlock>(std::shared_ptr<BranchMapReader>(&branchMap_, NoDelete()),
                                                        runFactory_);
    }
    edm::RunNumber_t run = eventAuxiliary().run();
    edm::LuminosityBlockNumber_t lumi = eventAuxiliary().luminosityBlock();
    lumi_->to(run, lumi);
    return *lumi_;
  }

  fwlite::Run const& Event::getRun() const {
    run_ = runFactory_->makeRun(std::shared_ptr<BranchMapReader>(&branchMap_, NoDelete()));
    edm::RunNumber_t run = eventAuxiliary().run();
    run_->to(run);
    return *run_;
  }

}  // namespace fwlite
