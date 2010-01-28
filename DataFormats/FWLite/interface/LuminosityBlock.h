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
// $Id: Event.h,v 1.25 2009/11/04 17:03:16 elmer Exp $
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
#include "DataFormats/FWLite/interface/LuminosityBlockBase.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/EventProcessHistoryID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "FWCore/FWLite/interface/BranchMapReader.h"

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
   namespace internalLS {
      class DataKey {
         public:
            //NOTE: Do not take ownership of strings.  This is done to avoid
            // doing 'new's and string copies when we just want to lookup the data
            // This means something else is responsible for the pointers remaining
            // valid for the time for which the pointers are still in use
            DataKey(const edm::TypeID& iType,
                    const char* iModule,
                    const char* iProduct,
                    const char* iProcess) :
               type_(iType),
               module_(iModule!=0? iModule:kEmpty),
               product_(iProduct!=0?iProduct:kEmpty),
               process_(iProcess!=0?iProcess:kEmpty) {}

            ~DataKey() {
            }

            bool operator<( const DataKey& iRHS) const {
               if( type_ < iRHS.type_) {
                  return true;
               }
               if( iRHS.type_ < type_ ) {
                  return false;
               }
               int comp = std::strcmp(module_,iRHS.module_);
               if( 0!= comp) {
                  return comp <0;
               }
               comp = std::strcmp(product_,iRHS.product_);
               if( 0!= comp) {
                  return comp <0;
               }
               comp = std::strcmp(process_,iRHS.process_);
               return comp <0;
            }
            static const char* const kEmpty;
            const  char* module() const {return module_;}
            const char* product() const {return product_;}
            const char* process() const {return process_;}
            const edm::TypeID& typeID() const {return type_;}

         private:
            edm::TypeID type_;
            const char* module_;
            const char* product_;
            const char* process_;
      };

      struct Data {
            TBranch* branch_;
            Long64_t lastLuminosityBlock_;
            Reflex::Object obj_;
            void * pObj_; //ROOT requires the address of the pointer be stable
            edm::EDProduct* pProd_;

            ~Data() {
               obj_.Destruct();
            }
      };

      class ProductGetter;
   }

   class LuminosityBlock : public LuminosityBlockBase
   {

      public:
         // NOTE: Does NOT take ownership so iFile must remain around
         // at least as long as LuminosityBlock
         LuminosityBlock(TFile* iFile);
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
            return branchMap_.getBranchDescriptions();
         }

         void setGetter( boost::shared_ptr<edm::EDProductGetter> getter ) { std::cout << "resetting getter" << std::endl; getter_ = getter; }

         edm::EDProduct const* getByProductID(edm::ProductID const&) const;

         // ---------- static member functions --------------------
         static void throwProductNotFoundException(const std::type_info&, const char*, const char*, const char*);

         // ---------- member functions ---------------------------

      private:
         friend class internalLS::ProductGetter;
         friend class fwlite::Event;

         LuminosityBlock(const LuminosityBlock&); // stop default

         const LuminosityBlock& operator=(const LuminosityBlock&); // stop default

         const edm::ProcessHistory& history() const;
         void updateAux(Long_t lumiIndex) const;
         void fillFileIndex() const;

         internalLS::Data& getBranchDataFor(const std::type_info&, const char*, const char*, const char*) const;

         // ---------- member data --------------------------------
         mutable fwlite::BranchMapReader branchMap_;


         typedef std::map<internalLS::DataKey, boost::shared_ptr<internalLS::Data> > KeyToDataMap;
         mutable KeyToDataMap data_;
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

         //references data in data_;
         mutable std::map<edm::ProductID,boost::shared_ptr<internalLS::Data> > idToData_;

         boost::shared_ptr<edm::EDProductGetter> getter_;
   };

}
#endif /*__CINT__ */
#endif
