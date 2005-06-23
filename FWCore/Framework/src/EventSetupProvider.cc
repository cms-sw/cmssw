// -*- C++ -*-
//
// Package:     CoreFramework
// Module:      EventSetupProvider
// 
// Description: <one line class summary>
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 24 16:27:14 EST 2005
//

// system include files
#include <set>

// user include files
#include "FWCore/CoreFramework/interface/EventSetupProvider.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordProvider.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordProviderFactoryManager.h"
#include "FWCore/CoreFramework/interface/DataProxyProvider.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordIntervalFinder.h"

namespace edm {
   namespace eventsetup {
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EventSetupProvider::EventSetupProvider() :
mustFinishConfiguration_(true)
{
}

// EventSetupProvider::EventSetupProvider(const EventSetupProvider& rhs)
// {
//    // do actual copying here;
// }

EventSetupProvider::~EventSetupProvider()
{
}

//
// assignment operators
//
// const EventSetupProvider& EventSetupProvider::operator=(const EventSetupProvider& rhs)
// {
//   //An exception safe implementation is
//   EventSetupProvider temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
EventSetupProvider::insert(const EventSetupRecordKey& iKey, std::auto_ptr<EventSetupRecordProvider> iProvider)
{
   boost::shared_ptr<EventSetupRecordProvider> temp(iProvider.release());
   providers_[iKey] = temp;
   //temp->addRecordTo(*this);
}

void 
EventSetupProvider::add(boost::shared_ptr<DataProxyProvider> iProvider)
{
   mustFinishConfiguration_ = true;
   assert(&(*iProvider) != 0);
   typedef std::set<EventSetupRecordKey> Keys;
   const Keys recordsUsing = iProvider->usingRecords();

   Keys::const_iterator itEnd = recordsUsing.end();
   for(Keys::const_iterator itKey = recordsUsing.begin();
       itKey != itEnd;
       ++itKey) {
      Providers::iterator itFound = providers_.find(*itKey);
      if(providers_.end() == itFound) {
         //create a provider for this record
         insert(*itKey, EventSetupRecordProviderFactoryManager::instance().makeRecordProvider(*itKey));
         itFound = providers_.find(*itKey);
      }
      itFound->second->add(iProvider);
   }
}


void 
EventSetupProvider::add(boost::shared_ptr<EventSetupRecordIntervalFinder> iFinder)
{
   mustFinishConfiguration_ = true;
   assert(&(*iFinder) != 0);
   typedef std::set<EventSetupRecordKey> Keys;
   const Keys recordsUsing = iFinder->findingForRecords();
   
   Keys::const_iterator itEnd = recordsUsing.end();
   for(Keys::const_iterator itKey = recordsUsing.begin();
       itKey != itEnd;
       ++itKey) {
      Providers::iterator itFound = providers_.find(*itKey);
      if(providers_.end() == itFound) {
         //create a provider for this record
         insert(*itKey, EventSetupRecordProviderFactoryManager::instance().makeRecordProvider(*itKey));
         itFound = providers_.find(*itKey);
      }
      itFound->second->addFinder(iFinder);
   }
}
void
EventSetupProvider::finishConfiguration()
{
   //For each Provider, find all the Providers it depends on.  If a dependent Provider
   // can not be found pass in an empty list
   Providers::iterator itEnd = providers_.end();
   for(Providers::iterator itProvider = providers_.begin();
        itProvider != itEnd;
        ++itProvider) {
      std::set<EventSetupRecordKey> records = itProvider->second->dependentRecords();
      if(records.size() != 0) {
         std::vector<boost::shared_ptr<EventSetupRecordProvider> > depProviders;
         depProviders.reserve(records.size());
         bool foundAllProviders = true;
         for(std::set<EventSetupRecordKey>::iterator itRecord = records.begin();
              itRecord != records.end();
              ++itRecord) {
            Providers::iterator itFound =providers_.find(*itRecord);
            if(itFound == providers_.end()) {
               foundAllProviders = false;
               break;
            }
            depProviders.push_back(itFound->second);
         }
         if(!foundAllProviders) {
            depProviders.clear();
         }
         itProvider->second->setDependentProviders(depProviders);
      }
   }
   mustFinishConfiguration_ = false;
   
}
//
// const member functions
//
EventSetup const&
EventSetupProvider::eventSetupForInstance(const Timestamp& iValue)
{
   eventSetup_.setTimestamp(iValue);

   eventSetup_.clear();
   if(mustFinishConfiguration_) {
      const_cast<EventSetupProvider*>(this)->finishConfiguration();
   }

   Providers::iterator itEnd = providers_.end();
   for(Providers::iterator itProvider = providers_.begin();
        itProvider != itEnd;
        ++itProvider) {
      itProvider->second->addRecordToIfValid(*this, iValue);
   }
   
   return eventSetup_;

   
}
//
// static member functions
//
   }
}
