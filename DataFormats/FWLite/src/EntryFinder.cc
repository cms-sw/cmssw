
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

#include "TBranch.h"
#include "TFile.h"
#include "TTree.h"

// user include files
#include "DataFormats/FWLite/interface/EntryFinder.h"
#include "FWCore/FWLite/interface/BranchMapReader.h"
#include "FWCore/Utilities/interface/Exception.h"

// forward declarations

namespace fwlite {
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

  void
  EntryFinder::fillEventEntriesInIndex(TBranch * auxBranch) {

    if (!indexIntoFile_.eventEntries().empty()) {
      return;
    }
    indexIntoFile_.eventEntries().resize(auxBranch->GetEntries());

    void *saveAddress = auxBranch->GetAddress();
    edm::EventAuxiliary *pEvAux = &this->eventAux_;
    auxBranch->SetAddress(&pEvAux);

    long long offset = 0;
    long long previousBeginEventNumbers = -1;

    for (edm::IndexIntoFile::SortedRunOrLumiItr runOrLumi = indexIntoFile_.beginRunOrLumi(),
                                             endRunOrLumi = indexIntoFile_.endRunOrLumi();
         runOrLumi != endRunOrLumi; ++runOrLumi) {

      if (runOrLumi.isRun()) continue;

      long long beginEventNumbers;
      long long endEventNumbers;
      EntryNumber_t beginEventEntry;
      EntryNumber_t endEventEntry;
      runOrLumi.getRange(beginEventNumbers, endEventNumbers, beginEventEntry, endEventEntry);

      // This is true each time one hits a new lumi section (except if the previous lumi had
      // no events, in which case the offset is still 0 anyway)
      if (beginEventNumbers != previousBeginEventNumbers) offset = 0;

      for (EntryNumber_t entry = beginEventEntry; entry != endEventEntry; ++entry) {
	auxBranch->GetEntry(entry);
        indexIntoFile_.eventEntries().at((entry - beginEventEntry) + offset + beginEventNumbers) =
          edm::IndexIntoFile::EventEntry(eventAux_.event(), entry);
      }

      previousBeginEventNumbers = beginEventNumbers;
      offset += endEventEntry - beginEventEntry;
    }
    indexIntoFile_.sortEventEntries();
    auxBranch->SetAddress(saveAddress);
  }
}
