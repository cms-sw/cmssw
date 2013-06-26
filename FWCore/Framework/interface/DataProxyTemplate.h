#ifndef FWCore_Framework_DataProxyTemplate_h
#define FWCore_Framework_DataProxyTemplate_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataProxyTemplate
// 
/**\class DataProxyTemplate DataProxyTemplate.h FWCore/Framework/interface/DataProxyTemplate.h

 Description: A DataProxy base class which allows one to write type-safe proxies

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
   
      DataProxyTemplate(){}
      //virtual ~DataProxyTemplate();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual const void* getImpl(const EventSetupRecord& iRecord,
                                  const DataKey& iKey) {
         assert(iRecord.key() == RecordT::keyForClass());
         return this->make(static_cast<const RecordT&>(iRecord), iKey);
      }
      
   protected:
      virtual const DataT* make(const RecordT&, const DataKey&) = 0;
      
   private:
      DataProxyTemplate(const DataProxyTemplate&); // stop default

      const DataProxyTemplate& operator=(const DataProxyTemplate&); // stop default

      // ---------- member data --------------------------------
};

   }
}
#endif
