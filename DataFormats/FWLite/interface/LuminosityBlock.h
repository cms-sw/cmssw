#ifndef DataFormats_FWLite_LuminosityBlock_h
#define DataFormats_FWLite_LuminosityBlock_h
// -*- C++ -*-
//
// Package:     FWLite/DataFormats
// Class  :     LuminosityBlock
//
/**\class LuminosityBlock LuminosityBlock.h DataFormats/FWLite/interface/LuminosityBlock.h

   Description: <one line class summary>

   Usage:
   <usage>

*/
//
// Original Author:  Eric Vaandering
//         Created:  Wed Jan 13 15:01:20 EDT 2007
// $Id: LuminosityBlock.h,v 1.4 2010/02/11 17:21:38 ewv Exp $
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
#include "DataFormats/FWLite/interface/Run.h"
#include "DataFormats/FWLite/interface/LuminosityBlockBase.h"
#include "DataFormats/FWLite/interface/InternalDataKey.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/EventProcessHistoryID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
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
   class LuminosityBlockAux;
   class Timestamp;
   class TriggerResults;
   class TriggerNames;
}

namespace fwlite {
   class Event;
   class LuminosityBlock : public LuminosityBlockBase
   {

      public:
         // NOTE: Does NOT take ownership so iFile must remain around
         // at least as long as LuminosityBlock
         LuminosityBlock(TFile* iFile);
         LuminosityBlock(boost::shared_ptr<BranchMapReader> branchMap);
         virtual ~LuminosityBlock();

         const LuminosityBlock& operator++();

         /// Go to event by Run & LuminosityBlock number
         bool to (edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi);

         // Go to the very first Event.
         const LuminosityBlock& toBegin();

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

         virtual edm::LuminosityBlockAuxiliary const& luminosityBlockAuxiliary() const;

         const std::vector<edm::BranchDescription>& getBranchDescriptions() const {
            return branchMap_->getBranchDescriptions();
         }

         edm::EDProduct const* getByProductID(edm::ProductID const&) const;

         // ---------- static member functions --------------------
         static void throwProductNotFoundException(const std::type_info&, const char*, const char*, const char*);

         // ---------- member functions ---------------------------
         fwlite::Run const& getRun() const;

      private:
         friend class internal::ProductGetter;
         friend class LumiHistoryGetter;

         LuminosityBlock(const LuminosityBlock&); // stop default

         const LuminosityBlock& operator=(const LuminosityBlock&); // stop default

         const edm::ProcessHistory& history() const;
         void updateAux(Long_t lumiIndex) const;
         void fillFileIndex() const;


         // ---------- member data --------------------------------
         mutable boost::shared_ptr<BranchMapReader> branchMap_;

         mutable boost::shared_ptr<fwlite::Run>  run_;

         //takes ownership of the strings used by the DataKey keys in data_
         mutable std::vector<const char*> labels_;
         mutable edm::ProcessHistoryMap historyMap_;
         mutable std::vector<std::string> procHistoryNames_;
         mutable edm::LuminosityBlockAuxiliary aux_;
         mutable edm::FileIndex fileIndex_;
         edm::LuminosityBlockAuxiliary* pAux_;
         edm::LuminosityBlockAux* pOldAux_;
         TBranch* auxBranch_;
         int fileVersion_;
         mutable bool parameterSetRegistryFilled_;

         fwlite::DataGetterHelper dataHelper_;
   };

}
#endif /*__CINT__ */
#endif
