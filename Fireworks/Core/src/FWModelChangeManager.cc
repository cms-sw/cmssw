// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelChangeManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Jan 17 19:13:46 EST 2008
// $Id: FWModelChangeManager.cc,v 1.3 2008/01/23 15:02:14 chrjones Exp $
//

// system include files
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>

// user include files
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
FWModelChangeManager::FWModelChangeManager():
m_depth(0)
{
}

// FWModelChangeManager::FWModelChangeManager(const FWModelChangeManager& rhs)
// {
//    // do actual copying here;
// }

FWModelChangeManager::~FWModelChangeManager()
{
}

//
// assignment operators
//
// const FWModelChangeManager& FWModelChangeManager::operator=(const FWModelChangeManager& rhs)
// {
//   //An exception safe implementation is
//   FWModelChangeManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWModelChangeManager::beginChanges()
{++m_depth;}

void 
FWModelChangeManager::changed(const FWModelId& iID)
{  
   FWChangeSentry sentry(*this);
   assert(iID.item());
   assert(iID.item()->id() < m_changes.size());
   m_changes[iID.item()->id()].insert(iID);   
}

void
FWModelChangeManager::changed(const FWEventItem* iItem)
{
   FWChangeSentry sentry(*this);
   assert(0!= iItem);
   m_itemChanges.insert(iItem);
   //remove any pending changes on models owned by this item
   assert(iItem->id() <m_changes.size());
   m_changes[iItem->id()].clear();
}

void 
FWModelChangeManager::endChanges()
{
   assert(m_depth !=0);
   //makes sure that 'changeSignalsAreDone is called if changeSignalsAreComing_ is sent
   boost::shared_ptr<void> guard;
   if(0 == --m_depth) {
      std::vector<FWModelChangeSignal>::iterator itSignal = m_changeSignals.begin();
      for(std::vector<FWModelIds>::iterator itChanges = m_changes.begin();
          itChanges != m_changes.end();
          ++itChanges,++itSignal) {
         if(not itChanges->empty()) {
            if(0 == guard.get()) {
               boost::shared_ptr<sigc::signal<void> > done(&changeSignalsAreDone_,
                                                            boost::mem_fn(&sigc::signal<void>::operator()));
               guard = done;
               changeSignalsAreComing_();
            }
            (*itSignal)(*itChanges);
            itChanges->clear();
         }
      }
      for(std::set<const FWEventItem*>::iterator itChanges = m_itemChanges.begin();
          itChanges != m_itemChanges.end();
          ++itChanges) {
         if(0 == guard.get()) {
            boost::shared_ptr<sigc::signal<void> > done(&changeSignalsAreDone_,
                                                        boost::mem_fn(&sigc::signal<void>::operator()));
            guard = done;
            changeSignalsAreComing_();
         }
         m_itemChangeSignals[(*itChanges)->id()](*itChanges);
      }
      m_itemChanges.clear();
   }
}

void 
FWModelChangeManager::newItemSlot(FWEventItem* iItem)
{
   assert(0!=iItem);
   assert(iItem->id() == m_changes.size());
   assert(iItem->id() == m_changeSignals.size());
   m_changes.push_back(FWModelIds());
   m_changeSignals.push_back(FWModelChangeSignal());
   m_itemChangeSignals.push_back(FWItemChangeSignal());
   //propagate our signal to the item
   m_changeSignals.back().connect(iItem->changed_);
   m_itemChangeSignals.back().connect(iItem->itemChanged_);
}

//
// const member functions
//

//
// static member functions
//
