#ifndef FWCore_Framework_Callback_h
#define FWCore_Framework_Callback_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     Callback
//
/**\class edm::eventsetup::Callback

 Description: Functional object used as the 'callback' for the CallbackProxy

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Sun Apr 17 14:30:24 EDT 2005
//

// system include files
#include <vector>
#include <type_traits>
#include <atomic>
// user include files
#include "FWCore/Framework/interface/produce_helpers.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/ESIndices.h"

namespace edm {
  namespace eventsetup {
    class EventSetupRecordImpl;

    // The default decorator that does nothing
    template <typename TRecord>
    struct CallbackSimpleDecorator {
      void pre(const TRecord&) {}
      void post(const TRecord&) {}
    };

    template <typename T,          //producer's type
              typename TReturn,    //return type of the producer's method
              typename TRecord,    //the record passed in as an argument
              typename TDecorator  //allows customization using pre/post calls
              = CallbackSimpleDecorator<TRecord>>
    class Callback {
    public:
      using method_type = TReturn (T ::*)(const TRecord&);

      Callback(T* iProd, method_type iMethod, unsigned int iID, const TDecorator& iDec = TDecorator())
          : proxyData_{},
            producer_(iProd),
            method_(iMethod),
            id_(iID),
            wasCalledForThisRecord_(false),
            decorator_(iDec) {}

      Callback* clone() { return new Callback(producer_.get(), method_, id_, decorator_); }

      Callback(const Callback&) = delete;
      const Callback& operator=(const Callback&) = delete;

      void operator()(EventSetupRecordImpl const* iRecord, EventSetupImpl const* iEventSetupImpl) {
        bool expected = false;
        if (wasCalledForThisRecord_.compare_exchange_strong(expected, true)) {
          //Get everything we can before knowing about the mayGets
          prefetch(iEventSetupImpl, getTokenIndices());

          if (handleMayGet(iRecord, iEventSetupImpl)) {
            prefetch(iEventSetupImpl, &((*postMayGetProxies_).front()));
          }

          runProducer(iRecord, iEventSetupImpl);
        }
      }

      template <class DataT>
      void holdOntoPointer(DataT* iData) {
        proxyData_[produce::find_index<TReturn, DataT>::value] = iData;
      }

      void storeReturnedValues(TReturn iReturn) {
        using type = typename produce::product_traits<TReturn>::type;
        setData<typename type::head_type, typename type::tail_type>(iReturn);
      }

      template <class RemainingContainerT, class DataT, class ProductsT>
      void setData(ProductsT& iProducts) {
        DataT* temp = reinterpret_cast<DataT*>(proxyData_[produce::find_index<TReturn, DataT>::value]);
        if (nullptr != temp) {
          moveFromTo(iProducts, *temp);
        }
        if constexpr (not std::is_same_v<produce::Null, RemainingContainerT>) {
          setData<typename RemainingContainerT::head_type, typename RemainingContainerT::tail_type>(iProducts);
        }
      }
      void newRecordComing() { wasCalledForThisRecord_ = false; }

      unsigned int transitionID() const { return id_; }
      ESProxyIndex const* getTokenIndices() const { return producer_->getTokenIndices(id_); }

    private:
      void prefetch(EventSetupImpl const* iImpl, ESProxyIndex const* proxies) const {
        auto recs = producer_->getTokenRecordIndices(id_);
        auto n = producer_->numberOfTokenIndices(id_);
        for (size_t i = 0; i != n; ++i) {
          auto rec = iImpl->findImpl(recs[i]);
          if (rec) {
            rec->doGet(proxies[i], iImpl, true);
          }
        }
      }

      bool handleMayGet(EventSetupRecordImpl const* iRecord, EventSetupImpl const* iEventSetupImpl) {
        //Handle mayGets
        TRecord rec;
        rec.setImpl(iRecord, transitionID(), getTokenIndices(), iEventSetupImpl, true);
        postMayGetProxies_ = producer_->updateFromMayConsumes(id_, rec);
        return static_cast<bool>(postMayGetProxies_);
      }

      void runProducer(EventSetupRecordImpl const* iRecord, EventSetupImpl const* iEventSetupImpl) {
        auto proxies = getTokenIndices();
        if (postMayGetProxies_) {
          proxies = &((*postMayGetProxies_).front());
        }
        TRecord rec;
        rec.setImpl(iRecord, transitionID(), proxies, iEventSetupImpl, true);
        decorator_.pre(rec);
        storeReturnedValues((producer_->*method_)(rec));
        decorator_.post(rec);
      }

      std::array<void*, produce::size<TReturn>::value> proxyData_;
      std::optional<std::vector<ESProxyIndex>> postMayGetProxies_;
      edm::propagate_const<T*> producer_;
      method_type method_;
      // This transition id identifies which setWhatProduced call this Callback is associated with
      const unsigned int id_;
      std::atomic<bool> wasCalledForThisRecord_;
      TDecorator decorator_;
    };
  }  // namespace eventsetup
}  // namespace edm

#endif
