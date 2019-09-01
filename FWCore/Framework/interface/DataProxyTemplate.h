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
#include <limits>

// forward declarations

namespace edm {
  namespace eventsetup {
    template <class RecordT, class DataT>
    class DataProxyTemplate : public DataProxy {
    public:
      typedef DataT value_type;
      typedef RecordT record_type;

      DataProxyTemplate() {}
      //virtual ~DataProxyTemplate();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      const void* getImpl(const EventSetupRecordImpl& iRecord, const DataKey& iKey) override {
        assert(iRecord.key() == RecordT::keyForClass());
        RecordT rec;
        rec.setImpl(&iRecord, std::numeric_limits<unsigned int>::max(), nullptr);
        return this->make(rec, iKey);
      }

    protected:
      virtual const DataT* make(const RecordT&, const DataKey&) = 0;

    private:
      DataProxyTemplate(const DataProxyTemplate&) = delete;  // stop default

      const DataProxyTemplate& operator=(const DataProxyTemplate&) = delete;  // stop default

      // ---------- member data --------------------------------
    };

  }  // namespace eventsetup
}  // namespace edm
#endif
