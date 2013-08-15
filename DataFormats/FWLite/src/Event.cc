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
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
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
    void operator()(void*){}
  };
}

namespace fwlite {
//
// static data member definitions
//
    namespace internal {
        class ProductGetter : public edm::EDProductGetter {
            public:
                ProductGetter(Event* iEvent) : event_(iEvent) {}

                edm::WrapperHolder
                getIt(edm::ProductID const& iID) const {
                    return event_->getByProductID(iID);
                }
            private:
                Event* event_;
        };
    }
//
// constructors and destructor
//
  Event::Event(TFile* iFile):
  file_(iFile),
//  eventTree_(0),
  eventHistoryTree_(0),
//  eventIndex_(-1),
  branchMap_(iFile),
  pAux_(&aux_),
  pOldAux_(0),
  fileVersion_(-1),
  parameterSetRegistryFilled_(false),
  dataHelper_(branchMap_.getEventTree(),
              boost::shared_ptr<HistoryGetterBase>(new EventHistoryGetter(this)),
              boost::shared_ptr<BranchMapReader>(&branchMap_,NoDelete()),
              boost::shared_ptr<edm::EDProductGetter>(new internal::ProductGetter(this)),
              true) {
    if(0 == iFile) {
      throw cms::Exception("NoFile") << "The TFile pointer passed to the constructor was null";
    }

    if(0 == branchMap_.getEventTree()) {
      throw cms::Exception("NoEventTree") << "The TFile contains no TTree named " << edm::poolNames::eventTreeName();
    }
    //need to know file version in order to determine how to read the basic event info
    fileVersion_ = branchMap_.getFileVersion(iFile);

    //got this logic from IOPool/Input/src/RootFile.cc

    TTree* eventTree = branchMap_.getEventTree();
    if(fileVersion_ >= 3) {
      auxBranch_ = eventTree->GetBranch(edm::BranchTypeToAuxiliaryBranchName(edm::InEvent).c_str());
      if(0 == auxBranch_) {
        throw cms::Exception("NoEventAuxilliary") << "The TTree "
        << edm::poolNames::eventTreeName()
        << " does not contain a branch named 'EventAuxiliary'";
      }
      auxBranch_->SetAddress(&pAux_);
    } else {
      pOldAux_ = new edm::EventAux();
      auxBranch_ = eventTree->GetBranch(edm::BranchTypeToAuxBranchName(edm::InEvent).c_str());
      if(0 == auxBranch_) {
        throw cms::Exception("NoEventAux") << "The TTree "
          << edm::poolNames::eventTreeName()
          << " does not contain a branch named 'EventAux'";
      }
      auxBranch_->SetAddress(&pOldAux_);
    }
    branchMap_.updateEvent(0);

    if(fileVersion_ >= 7 && fileVersion_ < 17) {
      eventHistoryTree_ = dynamic_cast<TTree*>(iFile->Get(edm::poolNames::eventHistoryTreeName().c_str()));
    }
    runFactory_ =  boost::shared_ptr<RunFactory>(new RunFactory());

}

// Event::Event(Event const& rhs)
// {
//    // do actual copying here;
// }

Event::~Event() {
  for(std::vector<char const*>::iterator it = labels_.begin(), itEnd = labels_.end();
      it != itEnd;
      ++it) {
    delete [] *it;
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

Event const&
Event::operator++() {
   Long_t eventIndex = branchMap_.getEventEntry();
   if(eventIndex < size()) {
      branchMap_.updateEvent(++eventIndex);
   }
   return *this;
}

Long64_t
Event::indexFromEventId(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, edm::EventNumber_t event) {
   entryFinder_.fillIndex(branchMap_);
   EntryFinder::EntryNumber_t entry = entryFinder_.findEvent(run, lumi, event);
   return (entry == EntryFinder::invalidEntry) ? -1 : entry;
}

bool
Event::to(Long64_t iEntry) {
   if (iEntry < size()) {
      // this is a valid entry
      return branchMap_.updateEvent(iEntry);
   }
   // if we're here, then iEntry was not valid
   return false;
}

bool
Event::to(edm::RunNumber_t run, edm::EventNumber_t event) {
   return to(run, 0U, event);
}

bool
Event::to(edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi, edm::EventNumber_t event) {
   entryFinder_.fillIndex(branchMap_);
   EntryFinder::EntryNumber_t entry = entryFinder_.findEvent(run, lumi, event);
   if (entry == EntryFinder::invalidEntry) {
      return false;
   }
   return branchMap_.updateEvent(entry);
}

bool
Event::to(const edm::EventID &id) {
   return to(id.run(), id.luminosityBlock(), id.event());
}

Event const&
Event::toBegin() {
   branchMap_.updateEvent(0);
   return *this;
}

//
// const member functions
//
void       Event::draw(Option_t* opt) {
   GetterOperate op(dataHelper_.getter());
   branchMap_.getEventTree()->Draw(opt);
}
Long64_t   Event::draw(char const* varexp, const TCut& selection, Option_t* option, Long64_t nentries, Long64_t firstentry) {
   GetterOperate op(dataHelper_.getter());
   return branchMap_.getEventTree()->Draw(varexp,selection,option,nentries,firstentry);
}
Long64_t   Event::draw(char const* varexp, char const* selection, Option_t* option, Long64_t nentries, Long64_t firstentry) {
   GetterOperate op(dataHelper_.getter());
   return branchMap_.getEventTree()->Draw(varexp,selection,option,nentries,firstentry);
}
Long64_t   Event::scan(char const* varexp, char const* selection, Option_t* option, Long64_t nentries, Long64_t firstentry) {
   GetterOperate op(dataHelper_.getter());
   return branchMap_.getEventTree()->Scan(varexp,selection,option,nentries,firstentry);
}


Long64_t
Event::size() const {
  return branchMap_.getEventTree()->GetEntries();
}

bool
Event::isValid() const {
  Long_t eventIndex = branchMap_.getEventEntry();
  return eventIndex != -1 and eventIndex < size();
}


Event::operator bool() const {
  return isValid();
}

bool
Event::atEnd() const {
  Long_t eventIndex = branchMap_.getEventEntry();
  return eventIndex == -1 or eventIndex == size();
}


std::vector<std::string> const&
Event::getProcessHistory() const {
  if (procHistoryNames_.empty()) {
    const edm::ProcessHistory& h = history();
    for (edm::ProcessHistory::const_iterator iproc = h.begin(), eproc = h.end();
         iproc != eproc; ++iproc) {
      procHistoryNames_.push_back(iproc->processName());
    }
  }
  return procHistoryNames_;
}


std::string const
Event::getBranchNameFor(std::type_info const& iInfo,
                  char const* iModuleLabel,
                  char const* iProductInstanceLabel,
                  char const* iProcessLabel) const {
    return dataHelper_.getBranchNameFor(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel);
}

bool
Event::getByLabel(
                  std::type_info const& iInfo,
                  char const* iModuleLabel,
                  char const* iProductInstanceLabel,
                  char const* iProcessLabel,
                  void* oData) const {
    if(atEnd()) {
        throw cms::Exception("OffEnd") << "You have requested data past the last event";
    }
    Long_t eventIndex = branchMap_.getEventEntry();
    return dataHelper_.getByLabel(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel, oData, eventIndex);
}

bool
Event::getByLabel(std::type_info const& iInfo,
                  char const* iModuleLabel,
                  char const* iProductInstanceLabel,
                  char const* iProcessLabel,
                  edm::WrapperHolder& holder) const {
    if(atEnd()) {
        throw cms::Exception("OffEnd") << "You have requested data past the last event";
    }
    Long_t eventIndex = branchMap_.getEventEntry();
    return dataHelper_.getByLabel(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel, holder, eventIndex);
}

edm::EventAuxiliary const&
Event::eventAuxiliary() const {
   Long_t eventIndex = branchMap_.getEventEntry();
   updateAux(eventIndex);
   return aux_;
}

void
Event::updateAux(Long_t eventIndex) const {
  if(auxBranch_->GetEntryNumber() != eventIndex) {
    auxBranch_->GetEntry(eventIndex);
    //handling dealing with old version
    if(0 != pOldAux_) {
      conversion(*pOldAux_,aux_);
    }
  }
}

const edm::ProcessHistory&
Event::history() const {
  edm::ProcessHistoryID processHistoryID;

  bool newFormat = (fileVersion_ >= 5);

  Long_t eventIndex = branchMap_.getEventEntry();
  updateAux(eventIndex);
  if (!newFormat) {
    processHistoryID = aux_.processHistoryID();
  }
  if(historyMap_.empty() || newFormat) {
    procHistoryNames_.clear();
    TTree *meta = dynamic_cast<TTree*>(branchMap_.getFile()->Get(edm::poolNames::metaDataTreeName().c_str()));
    if(0 == meta) {
      throw cms::Exception("NoMetaTree") << "The TFile does not appear to contain a TTree named "
      << edm::poolNames::metaDataTreeName();
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
          throw edm::Exception(edm::errors::FatalRootError)
            << "Failed to find history branch in event history tree";
        eventHistoryBranch->SetAddress(&pHistory);
        eventHistoryTree_->GetEntry(eventIndex);
        processHistoryID = history.processHistoryID();
      } else {
        std::vector<edm::EventProcessHistoryID> *pEventProcessHistoryIDs = &eventProcessHistoryIDs_;
        TBranch* b = meta->GetBranch(edm::poolNames::eventHistoryBranchName().c_str());
        b->SetAddress(&pEventProcessHistoryIDs);
        b->GetEntry(0);
        edm::EventProcessHistoryID target(aux_.id(), edm::ProcessHistoryID());
        processHistoryID = std::lower_bound(eventProcessHistoryIDs_.begin(), eventProcessHistoryIDs_.end(), target)->processHistoryID();
      }
    }

  }

  return historyMap_[processHistoryID];
}


edm::WrapperHolder
Event::getByProductID(edm::ProductID const& iID) const {
  Long_t eventIndex = branchMap_.getEventEntry();
  return dataHelper_.getByProductID(iID, eventIndex);
}


edm::TriggerNames const&
Event::triggerNames(edm::TriggerResults const& triggerResults) const {
  edm::TriggerNames const* names = triggerNames_(triggerResults);
  if (names != 0) return *names;

  if (!parameterSetRegistryFilled_) {
    fillParameterSetRegistry();
    names = triggerNames_(triggerResults);
  }
  if (names != 0) return *names;

  throw cms::Exception("TriggerNamesNotFound")
    << "TriggerNames not found in ParameterSet registry";
  return *names;
}

void
Event::fillParameterSetRegistry() const {
  if (parameterSetRegistryFilled_) return;
  parameterSetRegistryFilled_ = true;

  TTree* meta = dynamic_cast<TTree*>(branchMap_.getFile()->Get(edm::poolNames::metaDataTreeName().c_str()));
  if (0 == meta) {
    throw cms::Exception("NoMetaTree") << "The TFile does not contain a TTree named "
      << edm::poolNames::metaDataTreeName();
  }

  edm::FileFormatVersion fileFormatVersion;
  edm::FileFormatVersion *fftPtr = &fileFormatVersion;
  if(meta->FindBranch(edm::poolNames::fileFormatVersionBranchName().c_str()) != 0) {
    TBranch *fft = meta->GetBranch(edm::poolNames::fileFormatVersionBranchName().c_str());
    fft->SetAddress(&fftPtr);
    fft->GetEntry(0);
  }

  typedef std::map<edm::ParameterSetID, edm::ParameterSetBlob> PsetMap;
  PsetMap psetMap;
  TTree* psetTree(0);
  if (meta->FindBranch(edm::poolNames::parameterSetMapBranchName().c_str()) != 0) {
    PsetMap *psetMapPtr = &psetMap;
    TBranch* b = meta->GetBranch(edm::poolNames::parameterSetMapBranchName().c_str());
    b->SetAddress(&psetMapPtr);
    b->GetEntry(0);
  } else if(0 == (psetTree = dynamic_cast<TTree *>(branchMap_.getFile()->Get(edm::poolNames::parameterSetsTreeName().c_str())))) {
    throw cms::Exception("NoParameterSetMapTree")
    << "The TTree "
    << edm::poolNames::parameterSetsTreeName() << " could not be found in the file.";
  } else {
    typedef std::pair<edm::ParameterSetID, edm::ParameterSetBlob> IdToBlobs;
    IdToBlobs idToBlob;
    IdToBlobs* pIdToBlob = &idToBlob;
    psetTree->SetBranchAddress(edm::poolNames::idToParameterSetBlobsBranchName().c_str(), &pIdToBlob);
    for(long long i = 0; i != psetTree->GetEntries(); ++i) {
      psetTree->GetEntry(i);
      psetMap.insert(idToBlob);
    }
  }
  edm::ParameterSetConverter::ParameterSetIdConverter psetIdConverter;
  if(!fileFormatVersion.triggerPathsTracked()) {
    edm::ParameterSetConverter converter(psetMap, psetIdConverter, fileFormatVersion.parameterSetsByReference());
  } else {
    // Merge into the parameter set registry.
    edm::pset::Registry& psetRegistry = *edm::pset::Registry::instance();
    for(PsetMap::const_iterator i = psetMap.begin(), iEnd = psetMap.end();
        i != iEnd; ++i) {
      edm::ParameterSet pset(i->second.pset());
      pset.setID(i->first);
      psetRegistry.insertMapped(pset);
    }
  }
}

edm::TriggerResultsByName
Event::triggerResultsByName(std::string const& process) const {

  fwlite::Handle<edm::TriggerResults> hTriggerResults;
  hTriggerResults.getByLabel(*this, "TriggerResults", "", process.c_str());
  if (!hTriggerResults.isValid()) {
    return edm::TriggerResultsByName(0,0);
  }

  edm::TriggerNames const* names = triggerNames_(*hTriggerResults);
  if (names == 0 && !parameterSetRegistryFilled_) {
    fillParameterSetRegistry();
    names = triggerNames_(*hTriggerResults);
  }
  return edm::TriggerResultsByName(hTriggerResults.product(), names);
}

//
// static member functions
//
void
Event::throwProductNotFoundException(std::type_info const& iType, char const* iModule, char const* iProduct, char const* iProcess) {
    edm::TypeID type(iType);
  throw edm::Exception(edm::errors::ProductNotFound) << "A branch was found for \n  type ='" << type.className() << "'\n  module='" << iModule
    << "'\n  productInstance='" << ((0!=iProduct)?iProduct:"") << "'\n  process='" << ((0 != iProcess) ? iProcess : "") << "'\n"
    "but no data is available for this Event";
}


fwlite::LuminosityBlock const& Event::getLuminosityBlock() const {
  if (not lumi_) {
    // Branch map pointer not really being shared, owned by event, have to trick Lumi
    lumi_ = boost::shared_ptr<fwlite::LuminosityBlock> (
             new fwlite::LuminosityBlock(boost::shared_ptr<BranchMapReader>(&branchMap_,NoDelete()),
             runFactory_)
          );
  }
  edm::RunNumber_t             run  = eventAuxiliary().run();
  edm::LuminosityBlockNumber_t lumi = eventAuxiliary().luminosityBlock();
  lumi_->to(run, lumi);
  return *lumi_;
}

fwlite::Run const& Event::getRun() const {
  run_ = runFactory_->makeRun(boost::shared_ptr<BranchMapReader>(&branchMap_,NoDelete()));
  edm::RunNumber_t run = eventAuxiliary().run();
  run_->to(run);
  return *run_;
}

}
