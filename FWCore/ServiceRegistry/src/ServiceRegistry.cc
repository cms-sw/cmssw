// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServiceRegistry
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 13:33:19 EDT 2005
//

// user include files
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"

// system include files
#include <cassert>

namespace edm {
  //
  // constants, enums and typedefs
  //

  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //
  ServiceRegistry::ServiceRegistry() {}

  // ServiceRegistry::ServiceRegistry(ServiceRegistry const& rhs) {
  //    // do actual copying here;
  // }

  ServiceRegistry::~ServiceRegistry() {}

  //
  // assignment operators
  //
  // ServiceRegistry const& ServiceRegistry::operator=(ServiceRegistry const& rhs) {
  //   //An exception safe implementation is
  //   ServiceRegistry temp(rhs);
  //   swap(rhs);
  //
  //   return *this;
  // }

  //
  // member functions
  //
  ServiceToken ServiceRegistry::setContext(ServiceToken const& iNewToken) {
    ServiceToken returnValue(manager_);
    manager_ = iNewToken.manager_;
    return returnValue;
  }

  void ServiceRegistry::unsetContext(ServiceToken const& iOldToken) { manager_ = iOldToken.manager_; }

  void ServiceRegistry::setContextNonOwning(ServiceToken const& iNewToken, OperateNonOwning& iSaved) {
    iSaved.oldManager_ = std::move(manager_);
    iSaved.oldOwner_ = owner_;
    // aliasing constructor with an empty owner: a non-owning pointer without a control block
    manager_ = std::shared_ptr<serviceregistry::ServicesManager>(std::shared_ptr<serviceregistry::ServicesManager>(),
                                                                 iNewToken.manager_.get());
    owner_ = &iNewToken;
  }

  void ServiceRegistry::unsetContextNonOwning(OperateNonOwning& iSaved) {
    manager_ = std::move(iSaved.oldManager_);
    owner_ = iSaved.oldOwner_;
  }

  //
  // const member functions
  //
  ServiceToken ServiceRegistry::presentToken() const {
    // while the services are made available by OperateNonOwning, manager_ does not own them: return the owning token
    if (owner_ and owner_->manager_.get() == manager_.get()) {
      return *owner_;
    }
    // never hand out the non-owning pointer set by OperateNonOwning: it has no control block (use_count() == 0)
    assert(not manager_ or manager_.use_count() != 0);
    return manager_;
  }

  //
  // static member functions
  //

  ServiceToken ServiceRegistry::createServicesFromConfig(std::unique_ptr<ParameterSet> params) {
    auto serviceSets = params->popVParameterSet(std::string("services"));
    //create the services
    return ServiceToken(ServiceRegistry::createSet(serviceSets));
  }

  ServiceToken ServiceRegistry::createSet(std::vector<ParameterSet>& iPS) {
    using namespace serviceregistry;
    auto returnValue = std::make_shared<ServicesManager>(iPS);
    return ServiceToken(returnValue);
  }

  ServiceToken ServiceRegistry::createSet(std::vector<ParameterSet>& iPS,
                                          ServiceToken iToken,
                                          serviceregistry::ServiceLegacy iLegacy) {
    using namespace serviceregistry;
    auto returnValue = std::make_shared<ServicesManager>(iToken, iLegacy, iPS);
    return ServiceToken(returnValue);
  }

  ServiceRegistry& ServiceRegistry::instance() {
    static thread_local ServiceRegistry s_registry;
    return s_registry;
  }
}  // namespace edm
