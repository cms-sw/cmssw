// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     ServicePluginFactory
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 13:33:19 EDT 2005
// $Id$
//

// system include files

// user include files
#include "FWCore/ServiceRegistry/interface/ServicePluginFactory.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
static edm::serviceregistry::ServicePluginFactory s_factory;

//
// constructors and destructor
//
edm::serviceregistry::ServicePluginFactory::ServicePluginFactory() :
seal::PluginFactory< ServiceMakerBase* ()>("CMS EDM Framework Service")
{
}

// ServicePluginFactory::ServicePluginFactory(const ServicePluginFactory& rhs)
// {
//    // do actual copying here;
// }

//ServicePluginFactory::~ServicePluginFactory()
//{
//}

//
// assignment operators
//
// const ServicePluginFactory& ServicePluginFactory::operator=(const ServicePluginFactory& rhs)
// {
//   //An exception safe implementation is
//   ServicePluginFactory temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

//
// static member functions
//
edm::serviceregistry::ServicePluginFactory*
edm::serviceregistry::ServicePluginFactory::get()
{
   return &s_factory;
}

