// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     EventSetupRecord
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Sat Mar 26 18:06:32 EST 2005
//

// system include files

// user include files
#include "FWCore/CoreFramework/interface/EventSetupRecord.h"

namespace edm {
   namespace eventsetup {
//
// constants, enums and typedefs
//
      typedef std::map< DataKey , const DataProxy* > Proxies;
//
// static data member definitions
//

//
// constructors and destructor
//
EventSetupRecord::EventSetupRecord():
eventSetup_(0)
{
}

// EventSetupRecord::EventSetupRecord(const EventSetupRecord& rhs)
// {
//    // do actual copying here;
// }

EventSetupRecord::~EventSetupRecord()
{
}

//
// assignment operators
//
// const EventSetupRecord& EventSetupRecord::operator=(const EventSetupRecord& rhs)
// {
//   //An exception safe implementation is
//   EventSetupRecord temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
EventSetupRecord::set(const ValidityInterval& iInterval) 
{
   validity_ = iInterval;
}

bool 
EventSetupRecord::add(const DataKey& iKey ,
                    const DataProxy* iProxy)
{
   //
   if (0 != find(iKey)) {
      //
      // we already know the field exist, so do not need to check against end()
      //
      (*proxies_.find(iKey)).second = iProxy ;
   }
   else {
      proxies_.insert(Proxies::value_type(iKey , iProxy)) ;
   }
   return true ;
}

void 
EventSetupRecord::removeAll() 
{
}

//
// const member functions
//
const DataProxy* 
EventSetupRecord::find(const DataKey& iKey) const 
{
   Proxies::const_iterator entry(proxies_.find(iKey)) ;
   if (entry != proxies_.end()) {
      return entry->second;
   }
   return 0;
}

//
// static member functions
//
   }
}
