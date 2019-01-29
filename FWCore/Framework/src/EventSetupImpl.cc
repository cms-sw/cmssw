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
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupKnownRecordsSupplier.h"

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
EventSetupImpl::EventSetupImpl(ActivityRegistry const* activityRegistry) :
   recordMap_(),
   activityRegistry_(activityRegistry)

{
}

// EventSetupImpl::EventSetupImpl(EventSetupImpl const& rhs)
// {
//    // do actual copying here;
// }

EventSetupImpl::~EventSetupImpl()
{
}

//
// assignment operators
//
// EventSetupImpl const& EventSetupImpl::operator=(EventSetupImpl const& rhs)
// {
//   //An exception safe implementation is
//   EventSetupImpl temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
EventSetupImpl::insert(const eventsetup::EventSetupRecordKey& iKey,
                const eventsetup::EventSetupRecordImpl* iRecord)
{
   recordMap_[iKey]= iRecord;
}

void
EventSetupImpl::clear()
{
   recordMap_.clear();
}

void
EventSetupImpl::add(const eventsetup::EventSetupRecordImpl& iRecord)
{
   insert(iRecord.key(), &iRecord);
}

//
// const member functions
//
std::optional<eventsetup::EventSetupRecordGeneric>
EventSetupImpl::find(const eventsetup::EventSetupRecordKey& iKey) const
{
   auto itFind = recordMap_.find(iKey);
   if(itFind == recordMap_.end()) {
     return std::nullopt;
   }
  return eventsetup::EventSetupRecordGeneric(itFind->second);
}

eventsetup::EventSetupRecordImpl const*
EventSetupImpl::findImpl(const eventsetup::EventSetupRecordKey& iKey) const
{
  auto itFind = recordMap_.find(iKey);
  if(itFind == recordMap_.end()) {
    return nullptr;
  }
  return itFind->second;
}

void
EventSetupImpl::fillAvailableRecordKeys(std::vector<eventsetup::EventSetupRecordKey>& oToFill) const
{
  oToFill.clear();
  oToFill.reserve(recordMap_.size());

  for(auto it = recordMap_.begin(), itEnd=recordMap_.end();
      it != itEnd;
      ++it) {
    oToFill.push_back(it->first);
  }
}

bool
EventSetupImpl::recordIsProvidedByAModule( eventsetup::EventSetupRecordKey const& iKey) const
{
  return knownRecords_->isKnown(iKey);
}

//
// static member functions
//
}
