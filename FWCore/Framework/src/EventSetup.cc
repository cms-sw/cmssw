// -*- C++ -*-
//
// Package:     Framework
// Module:      EventSetup
// 
// Description: <one line class summary>
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Thu Mar 24 16:27:10 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"

namespace edm {
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
   EventSetup::EventSetup() : syncValue_(IOVSyncValue::invalidIOVSyncValue()), recordMap_()
{
}

// EventSetup::EventSetup(EventSetup const& rhs)
// {
//    // do actual copying here;
// }

EventSetup::~EventSetup()
{
}

//
// assignment operators
//
// EventSetup const& EventSetup::operator=(EventSetup const& rhs)
// {
//   //An exception safe implementation is
//   EventSetup temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
EventSetup::setIOVSyncValue(const IOVSyncValue& iTime) {
   //will ultimately build our list of records
   syncValue_ = iTime;
}

void 
EventSetup::insert(const eventsetup::EventSetupRecordKey& iKey,
                const eventsetup::EventSetupRecord* iRecord)
{
   recordMap_[iKey]= iRecord;
}

void
EventSetup::clear()
{
   recordMap_.clear();
}
   
void 
EventSetup::add(const eventsetup::EventSetupRecord& iRecord) 
{
   insert(iRecord.key(), &iRecord);
}
   
//
// const member functions
//
const eventsetup::EventSetupRecord* 
EventSetup::find(const eventsetup::EventSetupRecordKey& iKey) const
{
   std::map<eventsetup::EventSetupRecordKey, eventsetup::EventSetupRecord const *>::const_iterator itFind
   = recordMap_.find(iKey);
   if(itFind == recordMap_.end()) {
      return 0;
   }
   return itFind->second;
}

void 
EventSetup::fillAvailableRecordKeys(std::vector<eventsetup::EventSetupRecordKey>& oToFill) const
{
  oToFill.clear();
  oToFill.reserve(recordMap_.size());
  
  typedef std::map<eventsetup::EventSetupRecordKey, eventsetup::EventSetupRecord const *> KeyToRecordMap;
  for(KeyToRecordMap::const_iterator it = recordMap_.begin(), itEnd=recordMap_.end();
      it != itEnd;
      ++it) {
    oToFill.push_back(it->first);
  }
}

//
// static member functions
//
}
