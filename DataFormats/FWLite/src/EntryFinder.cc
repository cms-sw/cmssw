
/*
// -*- C++ -*-
//
// Package:     FWLite/DataFormats
// Class  :     Index
//

   Description: <one line class summary>

   Usage:
   <usage>

*/
//
// Original Author:  Bill Tanenbaum
//

// user include files
#include "DataFormats/FWLite/interface/EntryFinder.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "FWCore/FWLite/interface/BranchMapReader.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TBranch.h"
#include "TFile.h"
#include "TTree.h"

// forward declarations

namespace fwlite {

  // This is a helper class for IndexIntoFile.
  class FWLiteEventFinder : public edm::IndexIntoFile::EventFinder {
  public:
    explicit FWLiteEventFinder(TBranch* auxBranch) : auxBranch_(auxBranch) {}
    virtual ~FWLiteEventFinder() {}
    virtual
    edm::EventNumber_t getEventNumberOfEntry(edm::IndexIntoFile::EntryNumber_t entry) const override {
      void* saveAddress = auxBranch_->GetAddress();
      edm::EventAuxiliary eventAux;
      edm::EventAuxiliary *pEvAux = &eventAux;
      auxBranch_->SetAddress(&pEvAux);
      auxBranch_->GetEntry(entry);
      auxBranch_->SetAddress(saveAddress);
      return eventAux.event();
    }

  private:
     TBranch* auxBranch_;
  };

   EntryFinder::EntryFinder() : indexIntoFile_(), fileIndex_() {}
   EntryFinder::~EntryFinder() {}

   EntryFinder::EntryNumber_t
   EntryFinder::findEvent(edm::RunNumber_t const& run, edm::LuminosityBlockNumber_t const& lumi, edm::EventNumber_t const& event) const {
     EntryFinder::EntryNumber_t ret = invalidEntry;
     if (!indexIntoFile_.empty()) {
       edm::IndexIntoFile::IndexIntoFileItr i = indexIntoFile_.findEventPosition(run, lumi, event);
       if (indexIntoFile_.end(edm::IndexIntoFile::numericalOrder) != i) {
         ret = i.entry();
       }
     } else {
       edm::FileIndex::const_iterator i = fileIndex_.findEventPosition(run, lumi, event);
       if (fileIndex_.end() != i) {
         ret = i->entry_;
       }
     }
     return ret;
   }

   EntryFinder::EntryNumber_t
   EntryFinder::findLumi(edm::RunNumber_t const& run, edm::LuminosityBlockNumber_t const& lumi) const {
     EntryFinder::EntryNumber_t ret = invalidEntry;
     if (!indexIntoFile_.empty()) {
       edm::IndexIntoFile::IndexIntoFileItr i = indexIntoFile_.findLumiPosition(run, lumi);
       if (indexIntoFile_.end(edm::IndexIntoFile::numericalOrder) != i) {
         ret = i.entry();
       }
     } else {
       edm::FileIndex::const_iterator i = fileIndex_.findLumiPosition(run, lumi);
       if (fileIndex_.end() != i) {
         ret = i->entry_;
       }
     }
     return ret;
   }

   EntryFinder::EntryNumber_t
   EntryFinder::findRun(edm::RunNumber_t const& run) const {
     EntryFinder::EntryNumber_t ret = invalidEntry;
     if (!indexIntoFile_.empty()) {
       edm::IndexIntoFile::IndexIntoFileItr i = indexIntoFile_.findRunPosition(run);
       if (indexIntoFile_.end(edm::IndexIntoFile::numericalOrder) != i) {
         ret = i.entry();
       }
     } else {
       edm::FileIndex::const_iterator i = fileIndex_.findRunPosition(run);
       if (fileIndex_.end() != i) {
         ret = i->entry_;
       }
     }
     return ret;
   }

   void
   EntryFinder::fillIndex(BranchMapReader const& branchMap) {
    if (empty()) {
      TTree* meta = dynamic_cast<TTree*>(branchMap.getFile()->Get(edm::poolNames::metaDataTreeName().c_str()));
      if (0 == meta) {
        throw cms::Exception("NoMetaTree") << "The TFile does not contain a TTree named "
          << edm::poolNames::metaDataTreeName();
      }
      if (meta->FindBranch(edm::poolNames::indexIntoFileBranchName().c_str()) != 0) {
        edm::IndexIntoFile* indexPtr = &indexIntoFile_;
        TBranch* b = meta->GetBranch(edm::poolNames::indexIntoFileBranchName().c_str());
        b->SetAddress(&indexPtr);
        b->GetEntry(0);
        TTree* eventTree = branchMap.getEventTree();
        TBranch* auxBranch = eventTree->GetBranch(edm::BranchTypeToAuxiliaryBranchName(edm::InEvent).c_str());
        if(0 == auxBranch) {
          throw cms::Exception("NoEventAuxilliary") << "The TTree "
          << edm::poolNames::eventTreeName()
          << " does not contain a branch named 'EventAuxiliary'";
        }

        indexIntoFile_.setNumberOfEvents(auxBranch->GetEntries());
        indexIntoFile_.setEventFinder(std::shared_ptr<edm::IndexIntoFile::EventFinder>(std::make_shared<FWLiteEventFinder>(auxBranch)));

      } else if (meta->FindBranch(edm::poolNames::fileIndexBranchName().c_str()) != 0) {
        edm::FileIndex* findexPtr = &fileIndex_;
        TBranch* b = meta->GetBranch(edm::poolNames::fileIndexBranchName().c_str());
        b->SetAddress(&findexPtr);
        b->GetEntry(0);
      } else {
        // TBD: fill the FileIndex for old file formats (prior to CMSSW 2_0_0)
        throw cms::Exception("NoIndexBranch") << "The TFile does not contain a TBranch named " <<
          edm::poolNames::indexIntoFileBranchName().c_str() << " or " << edm::poolNames::fileIndexBranchName().c_str();
      }
    }
    assert(!empty());
  }
}
