#ifndef FWCore_Framework_ESValidHandle_h
#define FWCore_Framework_ESValidHandle_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESValidHandle
//
/**\class ESValidHandle ESValidHandle.h FWCore/Framework/interface/ESValidHandle.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Tue Feb  5 14:47:35 EST 2019
//

// system include files

// user include files
#include "FWCore/Framework/interface/ComponentDescription.h"

#include <exception>
#include <memory>
#include <utility>

namespace edm {

  namespace esvhhelper {
    void throwIfNotValid(const void*) noexcept(false);
  }

  template <typename T>
  class ESValidHandle {
  public:
    typedef T value_type;

    ESValidHandle() = delete;
    ESValidHandle(T const& iData, edm::eventsetup::ComponentDescription const* desc) noexcept
        : data_{&iData}, description_{desc} {}

    ESValidHandle(ESValidHandle<T> const&) = default;
    ESValidHandle(ESValidHandle<T>&&) = default;
    ESValidHandle& operator=(ESValidHandle<T> const&) = default;
    ESValidHandle& operator=(ESValidHandle<T>&&) = default;

    // ---------- const member functions ---------------------
    T const* product() const noexcept { return data_; }
    T const* operator->() const noexcept { return product(); }
    T const& operator*() const noexcept { return *product(); }
    // ---------- static member functions --------------------
    static constexpr bool transientAccessOnly = false;

    // ---------- member functions ---------------------------
  private:
    T const* data_{nullptr};
    edm::eventsetup::ComponentDescription const* description_{nullptr};
  };

  /** Take a handle (e.g. edm::ESHandle<T>) and
   create a edm::ESValidHandle<T>. If the argument is an invalid handle,
   an exception will be thrown.
   */
  template <typename U>
  auto makeESValid(const U& iOtherHandleType) noexcept(false) {
    esvhhelper::throwIfNotValid(iOtherHandleType.product());
    //because of the check, we know this is valid and do not have to check again
    return ESValidHandle<typename U::value_type>(*iOtherHandleType.product(), iOtherHandleType.description());
  }

}  // namespace edm
#endif
