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
// $Id: ServiceRegistry.cc,v 1.2 2005/09/08 18:08:49 chrjones Exp $
//

// system include files
#include "boost/thread/tss.hpp"

// user include files
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
edm::ServiceRegistry::ServiceRegistry()
{
}

// ServiceRegistry::ServiceRegistry(const ServiceRegistry& rhs)
// {
//    // do actual copying here;
// }

edm::ServiceRegistry::~ServiceRegistry()
{
}

//
// assignment operators
//
// const ServiceRegistry& ServiceRegistry::operator=(const ServiceRegistry& rhs)
// {
//   //An exception safe implementation is
//   ServiceRegistry temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
edm::ServiceToken 
edm::ServiceRegistry::setContext(const edm::ServiceToken& iNewToken)
{
   edm::ServiceToken returnValue(manager_);
   manager_ = iNewToken.manager_;
   return returnValue;
}

void 
edm::ServiceRegistry::unsetContext(const ServiceToken& iOldToken)
{
   manager_ = iOldToken.manager_;
}

//
// const member functions
//
edm::ServiceToken 
edm::ServiceRegistry::presentToken() const
{
   return manager_;
}

//
// static member functions
//
edm::ServiceToken 
edm::ServiceRegistry::createSet(const std::vector<ParameterSet>& iPS)
{
   using namespace edm::serviceregistry;
   boost::shared_ptr<ServicesManager> returnValue(new ServicesManager(iPS));
   return edm::ServiceToken(returnValue);
}
edm::ServiceToken 
edm::ServiceRegistry::createSet(const std::vector<ParameterSet>& iPS,
                                ServiceToken iToken,
                                serviceregistry::ServiceLegacy iLegacy)
{
   using namespace edm::serviceregistry;
   boost::shared_ptr<ServicesManager> returnValue(new ServicesManager(iToken,iLegacy,iPS));
   return edm::ServiceToken(returnValue);
}

edm::ServiceRegistry& 
edm::ServiceRegistry::instance()
{
   static boost::thread_specific_ptr<ServiceRegistry> s_registry;
   if(0 == s_registry.get()){
      s_registry.reset(new ServiceRegistry);
   }
   return *s_registry;
}
