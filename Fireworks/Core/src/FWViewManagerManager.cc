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
// $Id: FWViewManagerManager.cc,v 1.2 2008/01/21 01:17:20 chrjones Exp $
//

// system include files
#include <iostream>

// user include files
#include "Fireworks/Core/interface/FWViewManagerManager.h"
#include "Fireworks/Core/interface/FWViewManagerBase.h"


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
}
void 
FWViewManagerManager::registerEventItem(const FWEventItem*iItem)
{
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
   for(std::vector<boost::shared_ptr<FWViewManagerBase> >::iterator itVM = m_viewManagers.begin();
       itVM != m_viewManagers.end();
       ++itVM) {
      if((*itVM)->useableBuilder(proxyBuilderName)) {
         //std::cout <<"REGISTERING "<<type<<std::endl;
         (*itVM)->registerProxyBuilder(type,proxyBuilderName);
         break;
      }
   }
}

//
// const member functions
//

//
// static member functions
//
