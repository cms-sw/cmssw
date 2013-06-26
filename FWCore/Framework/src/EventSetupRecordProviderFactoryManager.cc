// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordProviderFactoryManager
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Mon Mar 28 16:58:39 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/EventSetupRecordProviderFactoryManager.h"
#include "FWCore/Framework/interface/EventSetupRecordProviderFactory.h"
#include <cassert>


//
// constants, enums and typedefs
//
namespace edm {
   namespace eventsetup {
//
// static data member definitions
//

//
// constructors and destructor
//
EventSetupRecordProviderFactoryManager::EventSetupRecordProviderFactoryManager() : factories_()
{
}

// EventSetupRecordProviderFactoryManager::EventSetupRecordProviderFactoryManager(const EventSetupRecordProviderFactoryManager& rhs)
// {
//    // do actual copying here;
// }

EventSetupRecordProviderFactoryManager::~EventSetupRecordProviderFactoryManager()
{
}

//
// assignment operators
//
// const EventSetupRecordProviderFactoryManager& EventSetupRecordProviderFactoryManager::operator=(const EventSetupRecordProviderFactoryManager& rhs)
// {
//   //An exception safe implementation is
//   EventSetupRecordProviderFactoryManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
EventSetupRecordProviderFactoryManager::addFactory(const EventSetupRecordProviderFactory& iFactory, 
                                                 const EventSetupRecordKey& iKey) {
   factories_[iKey] = &iFactory;
}


//
// const member functions
//
std::auto_ptr<EventSetupRecordProvider> 
EventSetupRecordProviderFactoryManager::makeRecordProvider(const EventSetupRecordKey& iKey) const
{
   std::map<EventSetupRecordKey, const EventSetupRecordProviderFactory*>::const_iterator itFound= factories_.find(iKey);
   //should be impossible to have a key without a factory being available
   assert(itFound != factories_.end());
   
   const EventSetupRecordProviderFactory* factory = itFound->second;
   assert(0 != factory);
   return std::auto_ptr<EventSetupRecordProvider>(factory->makeRecordProvider());
}

//
// static member functions
//
EventSetupRecordProviderFactoryManager&
EventSetupRecordProviderFactoryManager::instance() {
   static EventSetupRecordProviderFactoryManager s_instance;
   return s_instance;
}
   }
}
