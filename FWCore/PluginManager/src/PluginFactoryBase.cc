// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     PluginFactoryBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 13:09:50 EDT 2007
// $Id: PluginFactoryBase.cc,v 1.2 2007/04/12 12:51:12 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/PluginManager/interface/PluginFactoryBase.h"
#include "FWCore/PluginManager/interface/PluginFactoryManager.h"
#include "FWCore/PluginManager/interface/PluginManager.h"

namespace edmplugin {
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

// PluginFactoryBase::PluginFactoryBase(const PluginFactoryBase& rhs)
// {
//    // do actual copying here;
// }

PluginFactoryBase::~PluginFactoryBase()
{
}

//
// assignment operators
//
// const PluginFactoryBase& PluginFactoryBase::operator=(const PluginFactoryBase& rhs)
// {
//   //An exception safe implementation is
//   PluginFactoryBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
PluginFactoryBase::finishedConstruction()
{
   PluginFactoryManager::get()->addFactory(this);
}

void
PluginFactoryBase::newPlugin(const std::string& iName)
{
  PluginInfo info;
  info.loadable_=boost::filesystem::path(PluginManager::loadingFile(),boost::filesystem::no_check);
  info.name_=iName;
  newPluginAdded_(category(),info);
}
//
// const member functions
//

//
// static member functions
//
}
