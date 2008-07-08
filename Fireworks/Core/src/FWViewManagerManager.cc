// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewManagerManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Jan 15 10:27:12 EST 2008
// $Id: FWViewManagerManager.cc,v 1.8 2008/06/09 20:18:22 chrjones Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>

// user include files
#include "Fireworks/Core/interface/FWViewManagerManager.h"
#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "FWCore/PluginManager/interface/PluginCapabilities.h"
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWViewManagerManager::FWViewManagerManager(FWModelChangeManager* iCM):
m_changeManager(iCM)
{
}

// FWViewManagerManager::FWViewManagerManager(const FWViewManagerManager& rhs)
// {
//    // do actual copying here;
// }

FWViewManagerManager::~FWViewManagerManager()
{
}

//
// assignment operators
//
// const FWViewManagerManager& FWViewManagerManager::operator=(const FWViewManagerManager& rhs)
// {
//   //An exception safe implementation is
//   FWViewManagerManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWViewManagerManager::add( boost::shared_ptr<FWViewManagerBase> iManager)
{
   m_viewManagers.push_back(iManager);
   iManager->setChangeManager(m_changeManager);

   for(std::map<std::string,const FWEventItem*>::iterator it=m_typeToItems.begin(), itEnd=m_typeToItems.end();
       it != itEnd;
       ++it) {
      iManager->newItem(it->second);
   }
}

void 
FWViewManagerManager::registerEventItem(const FWEventItem*iItem)
{
   if ( m_typeToItems.find(iItem->name()) != m_typeToItems.end() ) {
      printf("WARNING: item %s was already registered. Request ignored.\n", iItem->name().c_str() );
      return;
   }
   m_typeToItems[iItem->name()]=iItem;
   //std::map<std::string, std::vector<std::string> >::iterator itFind = m_typeToBuilders.find(iItem->name());
   for(std::vector<boost::shared_ptr<FWViewManagerBase> >::iterator itVM = m_viewManagers.begin();
       itVM != m_viewManagers.end();
       ++itVM) {
      (*itVM)->newItem(iItem);
   }
}

//
// const member functions
//
std::set<std::pair<std::string,std::string> > 
FWViewManagerManager::supportedTypesAndPurpose() const
{
   std::set<std::pair<std::string,std::string> > returnValue;
   for(std::vector<boost::shared_ptr<FWViewManagerBase> >::const_iterator itVM = m_viewManagers.begin();
       itVM != m_viewManagers.end();
       ++itVM) {
      std::set<std::pair<std::string,std::string> > v = (*itVM)->supportedTypesAndPurpose();
      returnValue.insert(v.begin(),v.end());
   }
   return returnValue;
}


//
// static member functions
//
