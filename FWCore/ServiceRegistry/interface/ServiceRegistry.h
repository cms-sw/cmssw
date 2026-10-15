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

    /// Like Operate, but without copying the token: the services are made available through a non-owning
    /// pointer, and the previous context is saved and restored by moving it. This does not touch the reference
    /// count of the token, so it can be used in tasks that run concurrently on many threads without making them
    /// all update the same atomic counter. The caller must guarantee that iToken outlives the guard.
    class OperateNonOwning {
    public:
      OperateNonOwning(ServiceToken const& iToken) { ServiceRegistry::instance().setContextNonOwning(iToken, *this); }
      OperateNonOwning(OperateNonOwning const&) = delete;
      OperateNonOwning const& operator=(OperateNonOwning const&) = delete;
      ~OperateNonOwning() { ServiceRegistry::instance().unsetContextNonOwning(*this); }

    private:
      friend class ServiceRegistry;
      std::shared_ptr<serviceregistry::ServicesManager> oldManager_;
      ServiceToken const* oldOwner_ = nullptr;
    };

    friend class edm::FwkImpl;
    friend int main(int argc, char* argv[]);
    friend class Operate;
    friend class OperateNonOwning;

    ServiceRegistry(ServiceRegistry const&) = delete;                   // stop default
    ServiceRegistry const& operator=(ServiceRegistry const&) = delete;  // stop default
    virtual ~ServiceRegistry();

    // ---------- const member functions ---------------------
    template <typename T>
    T& get() const {
      if (nullptr == manager_.get()) {
        auto demangled = typeDemangle(typeid(T).name());
        Exception::throwThis(errors::NotFound,
                             "No ServiceRegistry has been set for this thread. Tried to get a Service ",
                             demangled.c_str());
      }
      return manager_->template get<T>();
    }

    template <typename T>
    bool isAvailable() const {
      if (nullptr == manager_.get()) {
        auto demangled = typeDemangle(typeid(T).name());
        Exception::throwThis(errors::NotFound,
                             "No ServiceRegistry has been set for this thread. Tried to ask availability of a Service ",
                             demangled.c_str());
      }
      return manager_->template isAvailable<T>();
    }
    /** The token can be passed to another thread in order to have the
         same services available in the other thread.
         */
    ServiceToken presentToken() const;
    // ---------- static member functions --------------------
    static ServiceRegistry& instance();

    // ---------- member functions ---------------------------

    static ServiceToken createServicesFromConfig(std::unique_ptr<ParameterSet> params);

  public:  // Made public (temporarily) at the request of Emilio Meschi.
    static ServiceToken createSet(std::vector<ParameterSet>&);
    static ServiceToken createSet(std::vector<ParameterSet>&, ServiceToken, serviceregistry::ServiceLegacy);
    /// create a service token that holds the service defined by iService
    template <typename T>
    static ServiceToken createContaining(std::unique_ptr<T> iService) {
      std::vector<edm::ParameterSet> config;
      auto manager = std::make_shared<serviceregistry::ServicesManager>(config);
      auto wrapper = std::make_shared<serviceregistry::ServiceWrapper<T> >(std::move(iService));
      manager->put(wrapper);
      return manager;
    }
    template <typename T>
    static ServiceToken createContaining(std::unique_ptr<T> iService,
                                         ServiceToken iToken,
                                         bool iOverrideIfServiceInToken) {
      std::vector<edm::ParameterSet> config;
      auto manager =
          std::make_shared<serviceregistry::ServicesManager>(iToken, serviceregistry::kOverlapIsError, config);
      auto wrapper = std::make_shared<serviceregistry::ServiceWrapper<T> >(std::move(iService));
      manager->put(wrapper, iOverrideIfServiceInToken);
      return manager;
    }
    /// create a service token that holds the service held by iWrapper
    template <typename T>
    static ServiceToken createContaining(std::shared_ptr<serviceregistry::ServiceWrapper<T> > iWrapper) {
      std::vector<edm::ParameterSet> config;
      auto manager = std::make_shared<serviceregistry::ServicesManager>(config);
      manager->put(iWrapper);
      return manager;
    }
    template <typename T>
    static ServiceToken createContaining(std::shared_ptr<serviceregistry::ServiceWrapper<T> > iWrapper,
                                         ServiceToken iToken,
                                         serviceregistry::ServiceLegacy iLegacy) {
      std::vector<edm::ParameterSet> config;
      auto manager = std::make_shared<serviceregistry::ServicesManager>(iToken, iLegacy, config);
      manager->put(iWrapper);
      return manager;
    }

  private:
    //returns old token
    ServiceToken setContext(ServiceToken const& iNewToken);
    void unsetContext(ServiceToken const& iOldToken);
    void setContextNonOwning(ServiceToken const& iNewToken, OperateNonOwning& iSaved);
    void unsetContextNonOwning(OperateNonOwning& iSaved);

    ServiceRegistry();

    // ---------- member data --------------------------------
    std::shared_ptr<serviceregistry::ServicesManager> manager_;
    // owning token of the services made available by OperateNonOwning, while manager_ is a non-owning pointer to them;
    // used by presentToken() to return an owning token
    ServiceToken const* owner_ = nullptr;
  };
}  // namespace edm

#endif
