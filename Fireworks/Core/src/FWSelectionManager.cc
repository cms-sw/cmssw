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
// $Id: FWSelectionManager.cc,v 1.12 2012/09/21 09:26:26 eulisse Exp $
//

// system include files
#include <boost/bind.hpp>
#include <iostream>
#include <cassert>

// user include files
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
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
FWSelectionManager::FWSelectionManager(FWModelChangeManager* iCM) :
   m_changeManager(iCM),
   m_wasChanged(false)
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
   clearItemSelection();
}

void
FWSelectionManager::clearItemSelection()
{
   //may need this in the future 
   //FWChangeSentry sentry(*m_changeManager);
   std::set<FWEventItem*> items;
   items.swap(m_itemSelection);
   for(std::set<FWEventItem*>::iterator it = items.begin(), itEnd = items.end();
       it != itEnd;
       ++it) {
      //NOTE: this will cause
      (*it)->unselectItem();
   }
}

void 
FWSelectionManager::clearModelSelectionLeaveItem()
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
   if(changed) {
      //if this is new, we need to connect to the 'item' just incase it changes
      if(m_itemConnectionCount.size()<= iId.item()->id()) {
         m_itemConnectionCount.resize(iId.item()->id()+1);
      }
      if(1 ==++(m_itemConnectionCount[iId.item()->id()].first) ) {
         //want to know early about item change so we can send the 'selectionChanged' message
         // as part of the itemChange message from the ChangeManager
         // This way if more than one Item has changed, we still only send one 'selectionChanged' message
         m_itemConnectionCount[iId.item()->id()].second =
            iId.item()->preItemChanged_.connect(boost::bind(&FWSelectionManager::itemChanged,this,_1));
      }
   }
}

void
FWSelectionManager::unselect(const FWModelId& iId)
{
   bool changed = (0 != m_newSelection.erase(iId));
   m_wasChanged |=changed;
   if(changed) {
      assert(m_itemConnectionCount.size() > iId.item()->id());
      //was this the last model selected for this item?
      if(0 ==--(m_itemConnectionCount[iId.item()->id()].first)) {
         m_itemConnectionCount[iId.item()->id()].second.disconnect();
      }
   }
}

void
FWSelectionManager::itemChanged(const FWEventItem* iItem)
{
   assert(0!=iItem);
   assert(m_itemConnectionCount.size() > iItem->id());
   //if this appears in any of our models we need to remove them
   FWModelId low(iItem,0);
   FWModelId high(iItem,0x7FFFFFFF); //largest signed 32 bit number
   bool someoneChanged = false;
   {
      std::set<FWModelId>::iterator itL=m_newSelection.lower_bound(low),
                                    itH=m_newSelection.upper_bound(high);
      if(itL!=itH) {
         m_wasChanged =true;
         someoneChanged=true;
         m_newSelection.erase(itL,itH);
      }
   }
   {
      std::set<FWModelId>::iterator itL=m_selection.lower_bound(low),
                                    itH=m_selection.upper_bound(high);
      if(itL!=itH) {
         m_wasChanged =true;
         someoneChanged = true;
         //Don't need to erase here since will happen in 'finishedAllSelection'
      }
   }
   assert(someoneChanged);
   m_itemConnectionCount[iItem->id()].second.disconnect();
   m_itemConnectionCount[iItem->id()].first = 0;
}

void 
FWSelectionManager::selectItem(FWEventItem* iItem)
{
   m_itemSelection.insert(iItem);
   itemSelectionChanged_(*this);
}
void 
FWSelectionManager::unselectItem(FWEventItem* iItem)
{
   m_itemSelection.erase(iItem);
   itemSelectionChanged_(*this);
}
//
// const member functions
//
const std::set<FWModelId>&
FWSelectionManager::selected() const
{
   return m_selection;
}

const std::set<FWEventItem*>&
FWSelectionManager::selectedItems() const
{
   return m_itemSelection;
}

//
// static member functions
//
