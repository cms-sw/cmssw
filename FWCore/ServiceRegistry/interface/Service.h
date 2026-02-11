#ifndef ServiceRegistry_Service_h
#define ServiceRegistry_Service_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     Service
//
/**\class Service Service.h FWCore/ServiceRegistry/interface/Service.h

 Description: Smart pointer used to give easy access to Service's

 Usage:
    

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Sep  7 15:17:17 EDT 2005
//

// system include files
#include <optional>

// user include files
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

// forward declarations

namespace edm {
  template <typename T>
  class Service {
  public:
    Service() {}
    //virtual ~Service();

    // ---------- const member functions ---------------------
    T* operator->() const { return &(ServiceRegistry::instance().template get<T>()); }

    T& operator*() const { return ServiceRegistry::instance().template get<T>(); }

    bool isAvailable() const { return ServiceRegistry::instance().template isAvailable<T>(); }

    operator bool() const { return isAvailable(); }

    ///iF should be a functor and will only be called if the Service is available
    template <typename F>
      requires(!requires(F&& iF, T* t) {
                { iF(*t) } -> std::same_as<void>;
              })
    auto and_then(F&& iF, T* t) -> std::optional<decltype(iF(*t))> const {
      if (isAvailable()) {
        return iF(*(operator->()));
      }
      return std::nullopt;
    }

    template <typename F>
      requires(requires(F&& iF, T* t) {
        { iF(*t) } -> std::same_as<void>;
      })
    void and_then(F&& iF) const {
      if (isAvailable()) {
        iF(*(operator->()));
      }
    }

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------

  private:
  };

}  // namespace edm

#endif
