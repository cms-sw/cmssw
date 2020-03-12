#ifndef FWCore_Framework_DataProxyTemplate_h
#define FWCore_Framework_DataProxyTemplate_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataProxyTemplate
//
/**\class DataProxyTemplate DataProxyTemplate.h FWCore/Framework/interface/DataProxyTemplate.h

 Description: A DataProxy base class which allows one to write type-safe proxies

              Note that DataProxy types that inherit from this are not allowed
              to get data from the EventSetup (they cannot consume anything).
              This is intended mainly for use with ESSources that are also
              not allowed to get data from the EventSetup. Currently (as of
              April 2019), this class is used only in PoolDBESSource and
              Framework unit tests.

              This is also not used with ESProducers that inherit from
              the ESProducer base class and use the setWhatProduced interface.
              This class is used instead of CallProxy.

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

  class EventSetupImpl;

  namespace eventsetup {

    template <class RecordT, class DataT>
    class DataProxyTemplate : public DataProxy {
    public:
      typedef DataT value_type;
      typedef RecordT record_type;

      DataProxyTemplate() {}

      const void* getImpl(const EventSetupRecordImpl& iRecord,
                          const DataKey& iKey,
                          EventSetupImpl const* iEventSetupImpl) override {
        assert(iRecord.key() == RecordT::keyForClass());
        RecordT rec;
        rec.setImpl(&iRecord, std::numeric_limits<unsigned int>::max(), nullptr, iEventSetupImpl, true);
        return this->make(rec, iKey);
      }

    protected:
      virtual const DataT* make(const RecordT&, const DataKey&) = 0;
    };

  }  // namespace eventsetup
}  // namespace edm
#endif
