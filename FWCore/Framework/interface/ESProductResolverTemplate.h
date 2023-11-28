#ifndef FWCore_Framework_ESProductResolverTemplate_h
#define FWCore_Framework_ESProductResolverTemplate_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProductResolverTemplate
//
/**\class ESProductResolverTemplate ESProductResolverTemplate.h FWCore/Framework/interface/ESProductResolverTemplate.h

 Description: A ESProductResolver base class which allows one to write type-safe proxies

              Note that ESProductResolver types that inherit from this are not allowed
              to get data from the EventSetup (they cannot consume anything).
              This is intended mainly for use with ESSources that are also
              not allowed to get data from the EventSetup. Currently (as of
              April 2019), this class is used only in PoolDBESSource and
              Framework unit tests.

              This is also not used with ESProducers that inherit from
              the ESProducer base class and use the setWhatProduced interface.
              This class is used instead of CallbackProductResolver.

 Usage:
    <usage>
*/
//
// Author:      Chris Jones
// Created:     Thu Mar 31 12:45:32 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESProductResolver.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include <cassert>
#include <limits>
#include <atomic>

// forward declarations

namespace edm {

  class EventSetupImpl;

  namespace eventsetup {

    template <class RecordT, class DataT>
    class ESProductResolverTemplate : public ESProductResolver {
    public:
      typedef DataT value_type;
      typedef RecordT record_type;

      ESProductResolverTemplate() {}

      void prefetchAsyncImpl(WaitingTaskHolder iTask,
                             const EventSetupRecordImpl& iRecord,
                             const DataKey& iKey,
                             EventSetupImpl const* iEventSetupImpl,
                             edm::ServiceToken const& iToken,
                             edm::ESParentContext const& iParent) override {
        assert(iRecord.key() == RecordT::keyForClass());
        bool expected = false;
        bool doPrefetch = prefetching_.compare_exchange_strong(expected, true);
        taskList_.add(iTask);

        if (doPrefetch) {
          iTask.group()->run([this, &iRecord, iKey, iEventSetupImpl, iToken, iParent]() {
            try {
              RecordT rec;
              rec.setImpl(&iRecord, std::numeric_limits<unsigned int>::max(), nullptr, iEventSetupImpl, &iParent);
              ServiceRegistry::Operate operate(iToken);
              this->make(rec, iKey);
            } catch (...) {
              this->taskList_.doneWaiting(std::current_exception());
              return;
            }
            this->taskList_.doneWaiting(std::exception_ptr{});
          });
        }
      }

    protected:
      void invalidateCache() override {
        taskList_.reset();
        prefetching_ = false;
      }

      virtual const DataT* make(const RecordT&, const DataKey&) = 0;

    private:
      WaitingTaskList taskList_;
      std::atomic<bool> prefetching_{false};
    };

  }  // namespace eventsetup
}  // namespace edm
#endif
