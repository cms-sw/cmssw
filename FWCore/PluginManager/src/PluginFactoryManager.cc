// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     PluginFactoryManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 13:09:31 EDT 2007
//

// system include files

// user include files
#include "FWCore/PluginManager/interface/PluginFactoryManager.h"

namespace edmplugin{
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PluginFactoryManager::PluginFactoryManager()
{
}

// PluginFactoryManager::PluginFactoryManager(const PluginFactoryManager& rhs)
// {
//    // do actual copying here;
// }

PluginFactoryManager::~PluginFactoryManager()
{
}

//
// assignment operators
//
// const PluginFactoryManager& PluginFactoryManager::operator=(const PluginFactoryManager& rhs)
// {
//   //An exception safe implementation is
//   PluginFactoryManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
PluginFactoryManager::addFactory(const PluginFactoryBase* iFactory)
{
   factories_.push_back(iFactory);
   newFactory_(iFactory);
}

//
// const member functions
//
PluginFactoryManager::const_iterator
PluginFactoryManager::begin() const
{
   return factories_.begin();
}

PluginFactoryManager::const_iterator
PluginFactoryManager::end() const
{
   return factories_.end();
}

//
// static member functions
//
PluginFactoryManager*
PluginFactoryManager::get()
{
   static PluginFactoryManager s_instance;
   return &s_instance;
}
}
