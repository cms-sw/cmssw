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
// $Id: Run.h,v 1.1 2010/02/11 17:21:38 ewv Exp $
//
#if !defined(__CINT__) && !defined(__MAKECINT__)
// system include files
#include <typeinfo>
#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <memory>
#include <cstring>

#include "TBranch.h"
#include "Rtypes.h"
#include "Reflex/Object.h"

// user include files
#include "FWCore/Utilities/interface/TypeID.h"
#include "DataFormats/FWLite/interface/RunBase.h"
#include "DataFormats/FWLite/interface/InternalDataKey.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/EventProcessHistoryID.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "FWCore/FWLite/interface/BranchMapReader.h"
#include "DataFormats/FWLite/interface/HistoryGetterBase.h"
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

//          void setGetter( boost::shared_ptr<edm::EDProductGetter> getter ) { std::cout << "resetting getter" << std::endl; getter_ = getter; }

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
         void fillFileIndex() const;

//          internal::Data& getBranchDataFor(const std::type_info&, const char*, const char*, const char*) const;

         // ---------- member data --------------------------------
         mutable boost::shared_ptr<BranchMapReader> branchMap_;


/*         typedef std::map<internal::DataKey, boost::shared_ptr<internal::Data> > KeyToDataMap;
         mutable KeyToDataMap data_;*/
         //takes ownership of the strings used by the DataKey keys in data_
         mutable std::vector<const char*> labels_;
         mutable edm::ProcessHistoryMap historyMap_;
         mutable std::vector<std::string> procHistoryNames_;
         mutable edm::RunAuxiliary aux_;
         mutable edm::FileIndex fileIndex_;
         edm::RunAuxiliary* pAux_;
         edm::RunAux* pOldAux_;
         TBranch* auxBranch_;
         int fileVersion_;
         mutable bool parameterSetRegistryFilled_;

         //references data in data_;
//          mutable std::map<edm::ProductID,boost::shared_ptr<internal::Data> > idToData_;

         fwlite::DataGetterHelper dataHelper_;
   };

}
#endif /*__CINT__ */
#endif
