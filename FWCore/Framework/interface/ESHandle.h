#ifndef FWCore_Framework_ESHandle_h
#define FWCore_Framework_ESHandle_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESHandle
//
/**\class ESHandle ESHandle.h FWCore/Framework/interface/ESHandle.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Apr  1 14:47:35 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/ESHandleExceptionFactory.h"

#include <exception>
#include <memory>
#include <utility>

namespace edm {

class ESHandleBase {
   public:
      ESHandleBase() = default;
      ESHandleBase(void const* iData, edm::eventsetup::ComponentDescription const* desc)
           : data_(iData), description_(desc) {}

      ///Used when the attempt to get the data failed
      ESHandleBase(std::shared_ptr<ESHandleExceptionFactory>&& iWhyFailed) :
        whyFailedFactory_(std::move(iWhyFailed)) {}

      edm::eventsetup::ComponentDescription const* description() const;

      bool isValid() const { return nullptr != data_ && nullptr != description_; }

      bool failedToGet() const { return bool(whyFailedFactory_); }

      void swap(ESHandleBase& iOther) {
         std::swap(data_, iOther.data_);
         std::swap(description_, iOther.description_);
         std::swap(whyFailedFactory_, iOther.whyFailedFactory_);
      }

      std::shared_ptr<ESHandleExceptionFactory> const&
      whyFailedFactory() const { return whyFailedFactory_;}

   protected:
      void const *productStorage() const {
        if (whyFailedFactory_) {
          std::rethrow_exception(whyFailedFactory_->make());
        }
        return data_;
      }

   private:
      // ---------- member data --------------------------------
      void const* data_{nullptr};
      edm::eventsetup::ComponentDescription const* description_{nullptr};
      std::shared_ptr<ESHandleExceptionFactory> whyFailedFactory_{nullptr};
};

template<typename T>
class ESHandle : public ESHandleBase {
   public:
      typedef T value_type;

      ESHandle() = default;
      ESHandle(T const* iData) : ESHandleBase(iData, nullptr) {}
      ESHandle(T const* iData, edm::eventsetup::ComponentDescription const* desc) : ESHandleBase(iData, desc) {}
      ESHandle(std::shared_ptr<ESHandleExceptionFactory> &&);

      // ---------- const member functions ---------------------
      T const* product() const { return static_cast<T const *>(productStorage()); }
      T const* operator->() const { return product(); }
      T const& operator*() const { return *product(); }
      // ---------- static member functions --------------------
      static constexpr bool transientAccessOnly = false;

      // ---------- member functions ---------------------------

};

template <class T>
ESHandle<T>::ESHandle(std::shared_ptr<edm::ESHandleExceptionFactory> && iWhyFailed) :
  ESHandleBase(std::move(iWhyFailed))
{ }

  // Free swap function
  inline
  void
  swap(ESHandleBase& a, ESHandleBase& b)
  {
    a.swap(b);
  }
}
#endif
