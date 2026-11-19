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

  //
  // const member functions
  //
  ServiceToken ServiceRegistry::presentToken() const { return ServiceToken(manager_); }

  //
  // static member functions
  //

  OwningServiceToken ServiceRegistry::createServicesFromConfig(std::unique_ptr<ParameterSet> params) {
    auto serviceSets = params->popVParameterSet(std::string("services"));
    //create the services
    return ServiceRegistry::createSet(serviceSets);
  }

  OwningServiceToken ServiceRegistry::createSet(std::vector<ParameterSet>& iPS) {
    using namespace serviceregistry;
    return OwningServiceToken(ServicesManager::create(iPS));
  }

  OwningServiceToken ServiceRegistry::createSet(std::vector<ParameterSet>& iPS,
                                                OwningServiceToken const& iToken,
                                                serviceregistry::ServiceLegacy iLegacy) {
    using namespace serviceregistry;
    return OwningServiceToken(ServicesManager::create(iToken.manager_, iLegacy, iPS));
  }

  ServiceRegistry& ServiceRegistry::instance() {
    static thread_local ServiceRegistry s_registry;
    return s_registry;
  }
}  // namespace edm
