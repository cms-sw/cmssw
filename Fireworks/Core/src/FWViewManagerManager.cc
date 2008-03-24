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
// $Id: FWViewManagerManager.cc,v 1.6 2008/03/20 20:14:35 chrjones Exp $
//

// system include files
#include <iostream>

// user include files
#include "Fireworks/Core/interface/FWViewManagerManager.h"
#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"

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
   for(std::vector<boost::shared_ptr<FWViewManagerBase> >::iterator itVM = m_viewManagers.begin();
       itVM != m_viewManagers.end();
       ++itVM) {
      (*itVM)->newItem(iItem);
   }
}
void 
FWViewManagerManager::registerProxyBuilder(const std::string& type, 
                                           const std::string& proxyBuilderName)
{
   bool foundOwner = false;
   std::map<std::string,const FWEventItem*>::iterator itFind = m_typeToItems.find(type);
   const FWEventItem* matchedItem=0;
   if( itFind != m_typeToItems.end() ) {
      matchedItem = itFind->second;
   }
   for(std::vector<boost::shared_ptr<FWViewManagerBase> >::iterator itVM = m_viewManagers.begin();
       itVM != m_viewManagers.end();
       ++itVM) {
      if((*itVM)->useableBuilder(proxyBuilderName)) {
         std::cout <<"REGISTERING "<<type << ", " << proxyBuilderName <<std::endl;
         (*itVM)->registerProxyBuilder(type,proxyBuilderName,matchedItem);
         //return;
         foundOwner = true;
      }
   }
   if(not foundOwner) {
      std::cout << "rejecting " << type << ", " << proxyBuilderName <<std::endl;
   }
}

//
// const member functions
//

//
// static member functions
//
