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
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"

// system include files
#include "boost/thread/tss.hpp"

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
   ServiceRegistry::ServiceRegistry() {
   }

   // ServiceRegistry::ServiceRegistry(ServiceRegistry const& rhs) {
   //    // do actual copying here;
   // }

   ServiceRegistry::~ServiceRegistry() {
   }

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
   ServiceToken 
   ServiceRegistry::setContext(ServiceToken const& iNewToken) {
      ServiceToken returnValue(manager_);
      manager_ = iNewToken.manager_;
      return returnValue;
   }

   void 
   ServiceRegistry::unsetContext(ServiceToken const& iOldToken) {
      manager_ = iOldToken.manager_;
   }

   //
   // const member functions
   //
   ServiceToken 
   ServiceRegistry::presentToken() const {
      return manager_;
   }

   //
   // static member functions
   //

   ServiceToken
   ServiceRegistry::createServicesFromConfig(std::string const& config) {
      boost::shared_ptr<ParameterSet> params;
      makeParameterSets(config, params);

      std::auto_ptr<std::vector<ParameterSet> > serviceSets = params->popVParameterSet(std::string("services"));
      //create the services
      return ServiceToken(ServiceRegistry::createSet(*serviceSets));
   }

   ServiceToken 
   ServiceRegistry::createSet(std::vector<ParameterSet>& iPS) {
      using namespace serviceregistry;
      boost::shared_ptr<ServicesManager> returnValue(new ServicesManager(iPS));
      return ServiceToken(returnValue);
   }

   ServiceToken 
   ServiceRegistry::createSet(std::vector<ParameterSet>& iPS,
                                   ServiceToken iToken,
                                   serviceregistry::ServiceLegacy iLegacy,
                                   bool associate) {
      using namespace serviceregistry;
      boost::shared_ptr<ServicesManager> returnValue(new ServicesManager(iToken, iLegacy, iPS, associate));
      return ServiceToken(returnValue);
   }

   ServiceRegistry& 
   ServiceRegistry::instance() {
      static boost::thread_specific_ptr<ServiceRegistry> s_registry;
      if(0 == s_registry.get()){
         s_registry.reset(new ServiceRegistry);
      }
      return *s_registry;
   }
}
