#ifndef DataFormats_FWLite_EntryFinder_h
#define DataFormats_FWLite_EntryFinder_h
// -*- C++ -*-
//
// Package:     FWLite/DataFormats
// Class  :     EntryFinder 
//
/**\class  DataFormats/FWLite/interface/EntryFinder.h

   Description: <one line class summary>

   Usage:
   <usage>

*/
//
// Original Author:  Bill Tanenbaum
//
#if !defined(__CINT__) && !defined(__MAKECINT__)
// system include files

// user include files
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"

// forward declarations

namespace fwlite {
   class BranchMapReader;
   class EntryFinder {
   public:
     EntryFinder();
     ~EntryFinder();
     typedef edm::IndexIntoFile::EntryNumber_t EntryNumber_t;
     bool empty() const {return indexIntoFile_.empty() && fileIndex_.empty();}
     EntryNumber_t findEvent(edm::RunNumber_t const& run, edm::LuminosityBlockNumber_t const& lumi, edm::EventNumber_t const& event) const;
     EntryNumber_t findLumi(edm::RunNumber_t const& run, edm::LuminosityBlockNumber_t const& lumi) const;
     EntryNumber_t findRun(edm::RunNumber_t const& run) const;
     void fillIndex(BranchMapReader const& branchMap);
     static EntryNumber_t const invalidEntry = -1LL;
   private:
     edm::IndexIntoFile indexIntoFile_;
     edm::FileIndex fileIndex_;
   };
}
#endif /*__CINT__ */
#endif
