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
// $Id: FWModelChangeManager.cc,v 1.10 2008/11/06 22:05:26 amraktad Exp $
//

// system include files
#include <boost/shared_ptr.hpp>
#include <boost/mem_fn.hpp>
#include <exception>

// user include files
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "FWCore/Utilities/interface/Exception.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWModelChangeManager::FWModelChangeManager() :
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
{
   ++m_depth;
}

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

static
void sendChangeSignalsAreDone(FWModelChangeManager* iCM)
{
   //since this can cause other changes, we might as well aggregate them
   FWChangeSentry sentry(*iCM);
   iCM->changeSignalsAreDone_();
}

void
FWModelChangeManager::endChanges()
{
   assert(m_depth !=0);
   //makes sure that 'changeSignalsAreDone is called if changeSignalsAreComing_ is sent
   bool guard(false);
   if(0 == --m_depth) {
      unsigned int index = 0;
      for(std::set<const FWEventItem*>::iterator itChanges = m_itemChanges.begin();
          itChanges != m_itemChanges.end();
          ++itChanges,++index) {
         if( !guard ) {
            // boost::shared_ptr<FWModelChangeManager> done(this, &sendChangeSignalsAreDone);
            guard = true;
            changeSignalsAreComing_();
         }
         FWItemChangeSignal& signal = m_itemChangeSignals[(*itChanges)->id()];
         //loop over the slots ourself so we can control the behavior in case of a failure
         FWItemChangeSignal::slot_list_type slots = signal.slots();
         for(FWItemChangeSignal::slot_list_type::iterator itSlot=slots.begin(), itEnd = slots.end();
             itSlot != itEnd;
             ++itSlot) {
            try {
               (*itSlot)(*itChanges);
            } catch(const cms::Exception& iE) {
               std::cerr <<(*itChanges)->name()<<" had the failure\n"<<iE.what()<<std::endl;
            } catch(const std::bad_alloc& iE) {
               std::cerr <<"Ran out of memory while processing "<<(*itChanges)->name()<<std::endl;
               exit(1);
            } catch(const std::exception& iE) {
               std::cerr <<(*itChanges)->name()<<" had the failure\n"<<iE.what()<<std::endl;
            }
         }
      }
      m_itemChanges.clear();

      std::vector<FWModelChangeSignal>::iterator itSignal = m_changeSignals.begin();
      for(std::vector<FWModelIds>::iterator itChanges = m_changes.begin();
          itChanges != m_changes.end();
          ++itChanges,++itSignal) {
         if(not itChanges->empty()) {
            if( !guard ) {
               // boost::shared_ptr<FWModelChangeManager> done(this, &sendChangeSignalsAreDone);
               guard = true;
               changeSignalsAreComing_();
            }
            //loop over the slots ourself so we can control the behavior in case of a failure
            FWModelChangeSignal::slot_list_type slots = itSignal->slots();
            for(FWModelChangeSignal::slot_list_type::iterator itSlot=slots.begin(), itEnd = slots.end();
                itSlot != itEnd;
                ++itSlot) {
               try {
                  (*itSlot)(*itChanges);
               } catch(const cms::Exception& iE) {
                  std::cerr <<(*itChanges).begin()->item()->name()<<" had the failure\n"<<iE.what()<<std::endl;
               } catch(const std::bad_alloc& iE) {
                  std::cerr <<"Ran out of memory while processing "<<(*itChanges).begin()->item()->name()<<std::endl;
                  exit(1);
               } catch(const std::exception& iE) {
                  std::cerr <<(*itChanges).begin()->item()->name()<<" had the failure\n"<<iE.what()<<std::endl;
               }
            }
            itChanges->clear();
         }
      }
   }
   if (guard) sendChangeSignalsAreDone(this);
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
