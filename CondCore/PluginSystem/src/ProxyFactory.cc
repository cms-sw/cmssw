// -*- C++ -*-
//
// Package:     PluginSystem
// Class  :     ProxyFactory
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Jul 23 19:14:11 EDT 2005
// $Id$
//

// system include files

// user include files
#include "CondCore/PluginSystem/interface/ProxyFactory.h"
#include <map>
#include <string>
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
cond::ProxyFactory::ProxyFactory() 
  : seal::PluginFactory< edm::eventsetup::DataProxy*( pool::IDataSvc*, std::map<std::string,std::string>::iterator& ) >( "CondProxyFactory")
{
}

// TestCondProxyFactory::TestCondProxyFactory( const TestCondProxyFactory& rhs )
// {
//    // do actual copying here;
// }

cond::ProxyFactory::~ProxyFactory()
{
}

//
// assignment operators
//
// const TestCondProxyFactory& TestCondProxyFactory::operator=( const TestCondProxyFactory& rhs )
// {
//   //An exception safe implementation is
//   TestCondProxyFactory temp(rhs);
//   swap( rhs );
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
static
cond::ProxyFactory s_factory;

cond::ProxyFactory* 
cond::ProxyFactory::get()
{
  return &s_factory;
}
