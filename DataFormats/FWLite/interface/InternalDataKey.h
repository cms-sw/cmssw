#ifndef DataFormats_FWLite_InternalDataKey_h
#define DataFormats_FWLite_InternalDataKey_h

// -*- C++ -*-
//
// Package:     FWLite
// Class  :     internal::DataKey
//
/**\class DataKey InternalDataKey.h DataFormats/FWLite/interface/InternalDataKey.h

   Description: Split from fwlite::Event to be reused in Event, LuminosityBlock, Run

   Usage:
   <usage>

*/
//
// Original Author:  Eric Vaandering
//         Created:  Jan 29 09:01:20 CDT 2009
//
#if !defined(__CINT__) && !defined(__MAKECINT__)

#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "TBranch.h"

#include <cstring>

namespace fwlite {
   namespace internal {
      class DataKey {
         public:
            //NOTE: Do not take ownership of strings.  This is done to avoid
            // doing 'new's and string copies when we just want to lookup the data
            // This means something else is responsible for the pointers remaining
            // valid for the time for which the pointers are still in use
            DataKey(const edm::TypeID& iType,
                    char const* iModule,
                    char const* iProduct,
                    char const* iProcess) :
               type_(iType),
               module_(iModule!=0? iModule:kEmpty()),
               product_(iProduct!=0?iProduct:kEmpty()),
               process_(iProcess!=0?iProcess:kEmpty()) {}

            ~DataKey() {
            }

            bool operator<(const DataKey& iRHS) const {
               if(type_ < iRHS.type_) {
                  return true;
               }
               if(iRHS.type_ < type_) {
                  return false;
               }
               int comp = std::strcmp(module_,iRHS.module_);
               if(0 != comp) {
                  return comp < 0;
               }
               comp = std::strcmp(product_,iRHS.product_);
               if(0 != comp) {
                  return comp < 0;
               }
               comp = std::strcmp(process_,iRHS.process_);
               return comp < 0;
            }
            char const* kEmpty()  const {return "";}
            char const* module()  const {return module_;}
            char const* product() const {return product_;}
            char const* process() const {return process_;}
            const edm::TypeID& typeID() const {return type_;}

         private:
            edm::TypeID type_;
            char const* module_;
            char const* product_;
            char const* process_;
      };

      struct Data {
            TBranch* branch_;
            Long64_t lastProduct_;
            edm::ObjectWithDict obj_; // For wrapped object
            void * pObj_; // pointer to pProd_.  ROOT requires the address of the pointer be stable
            void * pProd_; // pointer to wrapped product
            edm::WrapperInterfaceBase * interface_;

            ~Data() {
               obj_.destruct();
            }
      };

      class ProductGetter;
   }

}

#endif /*__CINT__ */
#endif
