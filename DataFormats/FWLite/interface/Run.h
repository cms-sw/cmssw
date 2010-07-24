#ifndef DataFormats_FWLite_Run_h
#define DataFormats_FWLite_Run_h
// -*- C++ -*-
//
// Package:     FWLite/DataFormats
// Class  :     Run
//
/**\class Run Run.h DataFormats/FWLite/interface/Run.h

   Description: <one line class summary>

   Usage:
   <usage>

*/
//
// Original Author:  Eric Vaandering
//         Created:  Wed Jan 13 15:01:20 EDT 2007
//
#if !defined(__CINT__) && !defined(__MAKECINT__)
// system include files
#include <typeinfo>
#include <map>
#include <vector>
#include <memory>
#include <cstring>

#include "Rtypes.h"

// user include files
#include "DataFormats/FWLite/interface/RunBase.h"
#include "DataFormats/FWLite/interface/InternalDataKey.h"
#include "DataFormats/FWLite/interface/EntryFinder.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "FWCore/FWLite/interface/BranchMapReader.h"
#include "DataFormats/FWLite/interface/DataGetterHelper.h"

// forward declarations
namespace edm {
   class EDProduct;
   class ProductRegistry;
   class BranchDescription;
   class EDProductGetter;
   class RunAux;
   class Timestamp;
   class TriggerResults;
   class TriggerNames;
}

namespace fwlite {
   class Event;
   class Run : public RunBase
   {

      public:
         // NOTE: Does NOT take ownership so iFile must remain around
         // at least as long as Run
         Run(TFile* iFile);
         Run(boost::shared_ptr<BranchMapReader> branchMap);
         virtual ~Run();

         const Run& operator++();

         /// Go to event by Run & Run number
         bool to (edm::RunNumber_t run);

         // Go to the very first Event.
         const Run& toBegin();

         // ---------- const member functions ---------------------
         virtual const std::string getBranchNameFor(const std::type_info&,
                                                    const char*,
                                                    const char*,
                                                    const char*) const;

         // This function should only be called by fwlite::Handle<>
         using fwlite::RunBase::getByLabel;
         virtual bool getByLabel(const std::type_info&, const char*, const char*, const char*, void*) const;
         //void getByBranchName(const std::type_info&, const char*, void*&) const;

         bool isValid() const;
         operator bool () const;
         virtual bool atEnd() const;

         Long64_t size() const;

         virtual edm::RunAuxiliary const& runAuxiliary() const;

         const std::vector<edm::BranchDescription>& getBranchDescriptions() const {
            return branchMap_->getBranchDescriptions();
         }

//       void setGetter( //Copy from Event if needed

         edm::EDProduct const* getByProductID(edm::ProductID const&) const;

         // ---------- static member functions --------------------
         static void throwProductNotFoundException(const std::type_info&, const char*, const char*, const char*);

         // ---------- member functions ---------------------------

      private:
         friend class internal::ProductGetter;
         friend class RunHistoryGetter;

         Run(const Run&); // stop default

         const Run& operator=(const Run&); // stop default

         const edm::ProcessHistory& history() const;
         void updateAux(Long_t runIndex) const;

         // ---------- member data --------------------------------
         mutable boost::shared_ptr<BranchMapReader> branchMap_;

         //takes ownership of the strings used by the DataKey keys in data_
         mutable std::vector<const char*> labels_;
         mutable edm::ProcessHistoryMap historyMap_;
         mutable std::vector<std::string> procHistoryNames_;
         mutable edm::RunAuxiliary aux_;
         mutable EntryFinder entryFinder_;
         edm::RunAuxiliary* pAux_;
         edm::RunAux* pOldAux_;
         TBranch* auxBranch_;
         int fileVersion_;

         fwlite::DataGetterHelper dataHelper_;
   };

}
#endif /*__CINT__ */
#endif
