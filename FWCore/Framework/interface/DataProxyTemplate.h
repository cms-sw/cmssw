#ifndef FWCore_Framework_DataProxyTemplate_h
#define FWCore_Framework_DataProxyTemplate_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataProxyTemplate
// 
/**\class DataProxyTemplate DataProxyTemplate.h FWCore/Framework/interface/DataProxyTemplate.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 31 12:45:32 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/MakeDataException.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include <cassert>

// forward declarations

namespace edm {
   namespace eventsetup {
template<class RecordT, class DataT>
class DataProxyTemplate : public DataProxy
{

   public:
      typedef DataT value_type;
      typedef RecordT record_type;
   
      DataProxyTemplate() : cache_(0) {}
      //virtual ~DataProxyTemplate();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual const DataT* get(const RecordT& iRecord,
                                const DataKey& iKey) const {
         if(!cacheIsValid()) {
            cache_ = const_cast<DataProxyTemplate<RecordT, DataT>*>(this)->make(iRecord, iKey);
            const_cast<DataProxyTemplate<RecordT, DataT>*>(this)->setCacheIsValid();
         }
         if(0 == cache_) {
            throwMakeException(iRecord, iKey);
         }
         return cache_;
      }
      
      void doGet(const EventSetupRecord& iRecord, const DataKey& iKey) const {
         assert(iRecord.key() == RecordT::keyForClass());
         get(static_cast<const RecordT&>(iRecord), iKey);
      }
   protected:
      virtual const DataT* make(const RecordT&, const DataKey&) = 0;
      
      virtual void throwMakeException(const RecordT& /*iRecord*/,
                                       const DataKey& iKey) const {
         throw MakeDataException (MakeDataExceptionInfo<record_type, value_type>(iKey));
      }

   private:
      DataProxyTemplate(const DataProxyTemplate&); // stop default

      const DataProxyTemplate& operator=(const DataProxyTemplate&); // stop default

      // ---------- member data --------------------------------
      mutable const DataT* cache_;
};

   }
}
#endif
