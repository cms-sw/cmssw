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
// user include files
#include "FWCore/Framework/interface/produce_helpers.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/ESIndices.h"

namespace edm {
  namespace eventsetup {

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
              = CallbackSimpleDecorator<TRecord> >
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

      void operator()(const TRecord& iRecord) {
        if (!wasCalledForThisRecord_) {
          producer_->updateFromMayConsumes(id_, iRecord);
          decorator_.pre(iRecord);
          storeReturnedValues((producer_->*method_)(iRecord));
          wasCalledForThisRecord_ = true;
          decorator_.post(iRecord);
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
      std::array<void*, produce::size<TReturn>::value> proxyData_;
      edm::propagate_const<T*> producer_;
      method_type method_;
      // This transition id identifies which setWhatProduced call this Callback is associated with
      unsigned int id_;
      bool wasCalledForThisRecord_;
      TDecorator decorator_;
    };
  }  // namespace eventsetup
}  // namespace edm

#endif
