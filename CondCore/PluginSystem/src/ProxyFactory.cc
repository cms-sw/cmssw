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
// $Id: ProxyFactory.cc,v 1.3.2.2 2006/12/07 15:37:22 xiezhen Exp $
//

// system include files

// user include files
#include "CondCore/PluginSystem/interface/ProxyFactory.h"
#include <map>
#include <string>
//#include <iostream>
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
/*cond::ProxyFactory::ProxyFactory() 
  : seal::PluginFactory< edm::eventsetup::DataProxy*( pool::IDataSvc*, std::map<std::string,std::string>::iterator& ) >(pluginCategory())
{
}
*/
ProxyFactory::ProxyFactory() 
  : seal::PluginFactory< edm::eventsetup::DataProxy*( 
cond::PoolStorageManager* pooldb, std::map<std::string,std::string>::iterator& ) >(pluginCategory())
{
  //std::cout<<"ProxyFactory::ProxyFactory"<<std::endl;
}
ProxyFactory::~ProxyFactory()
{
  //std::cout<<"ProxyFactory::~ProxyFactory"<<std::endl;
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
ProxyFactory s_factory;

ProxyFactory* 
ProxyFactory::get()
{
  return &s_factory;
}

const char*
ProxyFactory::pluginCategory()
{
  return  "CondProxyFactory";
}
