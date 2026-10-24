#ifndef ServiceRegistry_ServiceRegistry_h
#define ServiceRegistry_ServiceRegistry_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServiceRegistry
//
/**\class ServiceRegistry ServiceRegistry.h FWCore/ServiceRegistry/interface/ServiceRegistry.h

 Description: Manages the 'thread specific' instance of Services

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 13:33:00 EDT 2005
//

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ServiceLegacy.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServicesManager.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"

// system include files
#include <utility>
#include <vector>

// forward declarations
namespace edm {
  class FwkImpl;
  namespace serviceregistry {
    template <typename T>
    class ServiceWrapper;
  }

  class ServiceRegistry {
  public:
    class Operate {
    public:
      Operate(ServiceToken const& iToken) : oldToken_(ServiceRegistry::instance().setContext(iToken)) {}
      Operate(Operate const&) = delete;                   //stop default
      Operate const& operator=(Operate const&) = delete;  //stop default
      ~Operate() { ServiceRegistry::instance().unsetContext(oldToken_); }

      //override operator new to stop use on heap?
    private:
      ServiceToken oldToken_;
    };

    /// Makes the services owned by an OwningServiceToken available, and keeps them alive, as long as it exists.
    /// When constructed from a WeakServiceToken, it locks it first. If the services have already been destroyed, no
    /// services are made available.
    class LockedOperate {
    public:
      explicit LockedOperate(OwningServiceToken iToken) : token_(std::move(iToken)), operate_(token_) {}
      explicit LockedOperate(WeakServiceToken const& iToken) : LockedOperate(iToken.lock()) {}
      LockedOperate(LockedOperate const&) = delete;                   //stop default
      LockedOperate const& operator=(LockedOperate const&) = delete;  //stop default

    private:
      OwningServiceToken token_;
      Operate operate_;
    };

    friend class edm::FwkImpl;
    friend int main(int argc, char* argv[]);
    friend class Operate;

    ServiceRegistry(ServiceRegistry const&) = delete;                   // stop default
    ServiceRegistry const& operator=(ServiceRegistry const&) = delete;  // stop default
    virtual ~ServiceRegistry();

    // ---------- const member functions ---------------------
    template <typename T>
    T& get() const {
      if (nullptr == manager_) {
        auto demangled = typeDemangle(typeid(T).name());
        Exception::throwThis(errors::NotFound,
                             "No ServiceRegistry has been set for this thread. Tried to get a Service ",
                             demangled.c_str());
      }
      return manager_->template get<T>();
    }

    template <typename T>
    bool isAvailable() const {
      if (nullptr == manager_) {
        auto demangled = typeDemangle(typeid(T).name());
        Exception::throwThis(errors::NotFound,
                             "No ServiceRegistry has been set for this thread. Tried to ask availability of a Service ",
                             demangled.c_str());
      }
      return manager_->template isAvailable<T>();
    }
    /** The token can be passed to another thread in order to have the
         same services available in the other thread. It does not own the
         services: use OwningServiceToken(presentToken()) to keep them alive.
         */
    ServiceToken presentToken() const;
    // ---------- static member functions --------------------
    static ServiceRegistry& instance();

    // ---------- member functions ---------------------------

    static OwningServiceToken createServicesFromConfig(std::unique_ptr<ParameterSet> params);

  public:  // Made public (temporarily) at the request of Emilio Meschi.
    static OwningServiceToken createSet(std::vector<ParameterSet>&);
    /// create a service token that holds the services described by the ParameterSets on top of those owned by the
    /// OwningServiceToken, sharing their ownership
    static OwningServiceToken createSet(std::vector<ParameterSet>&,
                                        OwningServiceToken const&,
                                        serviceregistry::ServiceLegacy);
    /// create a service token that holds the service defined by iService
    template <typename T>
    static OwningServiceToken createContaining(std::unique_ptr<T> iService) {
      std::vector<edm::ParameterSet> config;
      auto manager = serviceregistry::ServicesManager::create(config);
      auto wrapper = std::make_shared<serviceregistry::ServiceWrapper<T> >(std::move(iService));
      manager->put(wrapper);
      return manager;
    }
    template <typename T>
    static OwningServiceToken createContaining(std::unique_ptr<T> iService,
                                               OwningServiceToken const& iToken,
                                               bool iOverrideIfServiceInToken) {
      std::vector<edm::ParameterSet> config;
      auto manager =
          serviceregistry::ServicesManager::create(iToken.manager_, serviceregistry::kOverlapIsError, config);
      auto wrapper = std::make_shared<serviceregistry::ServiceWrapper<T> >(std::move(iService));
      manager->put(wrapper, iOverrideIfServiceInToken);
      return manager;
    }
    /// create a service token that holds the service held by iWrapper
    template <typename T>
    static OwningServiceToken createContaining(std::shared_ptr<serviceregistry::ServiceWrapper<T> > iWrapper) {
      std::vector<edm::ParameterSet> config;
      auto manager = serviceregistry::ServicesManager::create(config);
      manager->put(iWrapper);
      return manager;
    }
    template <typename T>
    static OwningServiceToken createContaining(std::shared_ptr<serviceregistry::ServiceWrapper<T> > iWrapper,
                                               OwningServiceToken const& iToken,
                                               serviceregistry::ServiceLegacy iLegacy) {
      std::vector<edm::ParameterSet> config;
      auto manager = serviceregistry::ServicesManager::create(iToken.manager_, iLegacy, config);
      manager->put(iWrapper);
      return manager;
    }

  private:
    //returns old token
    ServiceToken setContext(ServiceToken const& iNewToken);
    void unsetContext(ServiceToken const& iOldToken);

    ServiceRegistry();

    // ---------- member data --------------------------------
    // non-owning: the owner of the services is whoever created the ServiceRegistry::Operate
    serviceregistry::ServicesManager* manager_ = nullptr;
  };
}  // namespace edm

#endif
