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
#include "DataFormats/FWLite/interface/Run.h"
#include "DataFormats/FWLite/interface/LuminosityBlockBase.h"
#include "DataFormats/FWLite/interface/InternalDataKey.h"
#include "DataFormats/FWLite/interface/EntryFinder.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "FWCore/Utilities/interface/HideStdSharedPtrFromRoot.h"

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
   class BranchMapReader;
   class HistoryGetterBase;
   class DataGetterHelper;
   class RunFactory;

   class LuminosityBlock : public LuminosityBlockBase
   {

      public:
         // NOTE: Does NOT take ownership so iFile must remain around
         // at least as long as LuminosityBlock
         LuminosityBlock(TFile* iFile);
         LuminosityBlock(std::shared_ptr<BranchMapReader> branchMap,  std::shared_ptr<RunFactory> runFactory);
         virtual ~LuminosityBlock();

         const LuminosityBlock& operator++();

         /// Go to event by Run & LuminosityBlock number
         bool to (edm::RunNumber_t run, edm::LuminosityBlockNumber_t lumi);

         // Go to the very first Event.
         const LuminosityBlock& toBegin();

         // ---------- const member functions ---------------------
         virtual std::string const getBranchNameFor(std::type_info const&,
                                                    char const*,
                                                    char const*,
                                                    char const*) const;

         // This function should only be called by fwlite::Handle<>
         using fwlite::LuminosityBlockBase::getByLabel;
         virtual bool getByLabel(std::type_info const&, char const*, char const*, char const*, void*) const;
         //void getByBranchName(std::type_info const&, char const*, void*&) const;

         bool isValid() const;
         operator bool () const;
         virtual bool atEnd() const;

         Long64_t size() const;

         virtual edm::LuminosityBlockAuxiliary const& luminosityBlockAuxiliary() const;

         std::vector<edm::BranchDescription> const& getBranchDescriptions() const {
            return branchMap_->getBranchDescriptions();
         }

//       void setGetter(//Copy from Event if needed

         edm::EDProduct const* getByProductID(edm::ProductID const&) const;

         // ---------- static member functions --------------------
         static void throwProductNotFoundException(std::type_info const&, char const*, char const*, char const*);

         // ---------- member functions ---------------------------
         fwlite::Run const& getRun() const;

      private:
         friend class internal::ProductGetter;
         friend class LumiHistoryGetter;

         LuminosityBlock(const LuminosityBlock&); // stop default

         const LuminosityBlock& operator=(const LuminosityBlock&); // stop default

         const edm::ProcessHistory& history() const;
         void updateAux(Long_t lumiIndex) const;


         // ---------- member data --------------------------------
         mutable std::shared_ptr<BranchMapReader> branchMap_;

         mutable std::shared_ptr<fwlite::Run>  run_;

         //takes ownership of the strings used by the DataKey keys in data_
         mutable std::vector<char const*> labels_;
         mutable edm::ProcessHistoryMap historyMap_;
         mutable std::vector<std::string> procHistoryNames_;
         mutable edm::LuminosityBlockAuxiliary aux_;
         mutable EntryFinder entryFinder_;
         edm::LuminosityBlockAuxiliary* pAux_;
         edm::LuminosityBlockAux* pOldAux_;
         TBranch* auxBranch_;
         int fileVersion_;

         DataGetterHelper dataHelper_;
         mutable std::shared_ptr<RunFactory> runFactory_;
   };

}
#endif /*__CINT__ */
#endif
