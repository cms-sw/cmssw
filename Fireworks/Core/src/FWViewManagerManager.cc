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
//

// system include files
#include <iostream>
#include <boost/bind.hpp>

// user include files
#include "Fireworks/Core/interface/FWViewManagerManager.h"
#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWTypeToRepresentations.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWViewManagerManager::FWViewManagerManager(FWModelChangeManager* iCM,
                                           FWColorManager* iColorM) :
   m_changeManager(iCM),
   m_colorManager(iColorM)
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
   iManager->setColorManager(m_colorManager);

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
      fwLog(fwlog::kWarning) << "WARNING: item "<< iItem->name() <<" was already registered. Request ignored.\n";
      return;
   }
   m_typeToItems[iItem->name()]=iItem;
   iItem->goingToBeDestroyed_.connect(boost::bind(&FWViewManagerManager::removeEventItem,this,_1));

   //std::map<std::string, std::vector<std::string> >::iterator itFind = m_typeToBuilders.find(iItem->name());
   for(std::vector<boost::shared_ptr<FWViewManagerBase> >::iterator itVM = m_viewManagers.begin();
       itVM != m_viewManagers.end();
       ++itVM) {
      (*itVM)->newItem(iItem);
   }
}

void
FWViewManagerManager::removeEventItem(const FWEventItem* iItem)
{
   std::map<std::string, const FWEventItem*>::iterator itr =
      m_typeToItems.find(iItem->name());
   if ( itr != m_typeToItems.end() ) m_typeToItems.erase( itr );
}


//
// const member functions
//
FWTypeToRepresentations
FWViewManagerManager::supportedTypesAndRepresentations() const
{
   FWTypeToRepresentations returnValue;
   for(std::vector<boost::shared_ptr<FWViewManagerBase> >::const_iterator itVM = m_viewManagers.begin();
       itVM != m_viewManagers.end();
       ++itVM) {
      FWTypeToRepresentations v = (*itVM)->supportedTypesAndRepresentations();
      returnValue.insert(v);
   }
   return returnValue;
}

void
FWViewManagerManager::eventBegin()
{
   for (auto i = m_viewManagers.begin(); i != m_viewManagers.end(); ++i)
      (*i)->eventBegin();
}

void
FWViewManagerManager::eventEnd()
{
   for (auto i = m_viewManagers.begin(); i != m_viewManagers.end(); ++i)
      (*i)->eventEnd();
}

//
// static member functions
//
