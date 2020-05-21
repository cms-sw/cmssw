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
FWViewManagerManager::FWViewManagerManager(FWModelChangeManager* iCM, FWColorManager* iColorM)
    : m_changeManager(iCM), m_colorManager(iColorM) {}

// FWViewManagerManager::FWViewManagerManager(const FWViewManagerManager& rhs)
// {
//    // do actual copying here;
// }

FWViewManagerManager::~FWViewManagerManager() {}

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
void FWViewManagerManager::add(std::shared_ptr<FWViewManagerBase> iManager) {
  m_viewManagers.push_back(iManager);
  iManager->setChangeManager(m_changeManager);
  iManager->setColorManager(m_colorManager);

  for (auto& m_typeToItem : m_typeToItems) {
    iManager->newItem(m_typeToItem.second);
  }
}

void FWViewManagerManager::registerEventItem(const FWEventItem* iItem) {
  if (m_typeToItems.find(iItem->name()) != m_typeToItems.end()) {
    fwLog(fwlog::kWarning) << "WARNING: item " << iItem->name() << " was already registered. Request ignored.\n";
    return;
  }
  m_typeToItems[iItem->name()] = iItem;
  iItem->goingToBeDestroyed_.connect(boost::bind(&FWViewManagerManager::removeEventItem, this, _1));

  //std::map<std::string, std::vector<std::string> >::iterator itFind = m_typeToBuilders.find(iItem->name());
  for (auto& m_viewManager : m_viewManagers) {
    m_viewManager->newItem(iItem);
  }
}

void FWViewManagerManager::removeEventItem(const FWEventItem* iItem) {
  std::map<std::string, const FWEventItem*>::iterator itr = m_typeToItems.find(iItem->name());
  if (itr != m_typeToItems.end())
    m_typeToItems.erase(itr);
}

//
// const member functions
//
FWTypeToRepresentations FWViewManagerManager::supportedTypesAndRepresentations() const {
  FWTypeToRepresentations returnValue;
  for (const auto& m_viewManager : m_viewManagers) {
    FWTypeToRepresentations v = m_viewManager->supportedTypesAndRepresentations();
    returnValue.insert(v);
  }
  return returnValue;
}

void FWViewManagerManager::eventBegin() {
  for (auto& m_viewManager : m_viewManagers)
    m_viewManager->eventBegin();
}

void FWViewManagerManager::eventEnd() {
  for (auto& m_viewManager : m_viewManagers)
    m_viewManager->eventEnd();
}

//
// static member functions
//
