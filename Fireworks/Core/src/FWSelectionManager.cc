// -*- C++ -*-
//
// Package:     Core
// Class  :     FWSelectionManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jan 18 14:40:51 EST 2008
// $Id: FWSelectionManager.cc,v 1.1 2008/01/21 01:17:42 chrjones Exp $
//

// system include files
#include <boost/bind.hpp>

// user include files
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWSelectionManager::FWSelectionManager(FWModelChangeManager* iCM):
m_changeManager(iCM)
{
   assert(0!=m_changeManager);
   m_changeManager->changeSignalsAreDone_.connect(boost::bind(&FWSelectionManager::finishedAllSelections,this));
}

// FWSelectionManager::FWSelectionManager(const FWSelectionManager& rhs)
// {
//    // do actual copying here;
// }

/*FWSelectionManager::~FWSelectionManager()
{
}*/

//
// assignment operators
//
// const FWSelectionManager& FWSelectionManager::operator=(const FWSelectionManager& rhs)
// {
//   //An exception safe implementation is
//   FWSelectionManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWSelectionManager::clearSelection()
{  
   FWChangeSentry sentry(*m_changeManager);
   for(std::set<FWModelId>::iterator it = m_selection.begin(), itEnd = m_selection.end();
       it != itEnd;
       ++it) {
      //NOTE: this will cause 
      it->unselect();
   }
}

void 
FWSelectionManager::finishedAllSelections()
{
   if(m_wasChanged) {
      m_selection=m_newSelection;
      selectionChanged_(*this);
      m_wasChanged = false;
   }
}

void 
FWSelectionManager::select(const FWModelId& iId)
{
   bool changed = m_newSelection.insert(iId).second;
   m_wasChanged |=changed;
}

void 
FWSelectionManager::unselect(const FWModelId& iId)
{
   bool changed = (0 != m_newSelection.erase(iId));
   m_wasChanged |=changed;
}

//
// const member functions
//
const std::set<FWModelId>& 
FWSelectionManager::selected() const
{
   return m_selection;
}

//
// static member functions
//
