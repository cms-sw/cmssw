// -*- C++ -*-
//
// Package:     Framework
// Class  :     DependentRecordIntervalFinder
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Sat Apr 30 19:37:22 EDT 2005
// $Id: DependentRecordIntervalFinder.cc,v 1.5 2006/06/05 18:12:20 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/DependentRecordIntervalFinder.h"
#include "FWCore/Framework/interface/EventSetupRecordProvider.h"


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
DependentRecordIntervalFinder::DependentRecordIntervalFinder(const EventSetupRecordKey& iKey) :
  providers_()
{
   findingRecordWithKey(iKey);
}

// DependentRecordIntervalFinder::DependentRecordIntervalFinder(const DependentRecordIntervalFinder& rhs)
// {
//    // do actual copying here;
// }

DependentRecordIntervalFinder::~DependentRecordIntervalFinder()
{
}

//
// assignment operators
//
// const DependentRecordIntervalFinder& DependentRecordIntervalFinder::operator=(const DependentRecordIntervalFinder& rhs)
// {
//   //An exception safe implementation is
//   DependentRecordIntervalFinder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
DependentRecordIntervalFinder::addProviderWeAreDependentOn(boost::shared_ptr<EventSetupRecordProvider> iProvider)
{
   providers_.push_back(iProvider);
}

void 
DependentRecordIntervalFinder::setIntervalFor(const EventSetupRecordKey&,
                                               const IOVSyncValue& iTime, 
                                               ValidityInterval& oInterval)
{
   //I am assuming that an invalidTime is always less then the first valid time
   assert(IOVSyncValue::invalidIOVSyncValue() < IOVSyncValue::beginOfTime());
   if(providers_.size() == 0) {
      oInterval = ValidityInterval::invalidInterval();
      return;
   }
   bool haveAValidDependentRecord = false;
   ValidityInterval newInterval(IOVSyncValue::beginOfTime(), IOVSyncValue::endOfTime());
   for(Providers::iterator itProvider = providers_.begin();
       itProvider != providers_.end();
       ++itProvider) {
      if((*itProvider)->setValidityIntervalFor(iTime)) {
         haveAValidDependentRecord=true;
         ValidityInterval providerInterval = (*itProvider)->validityInterval();
         if(newInterval.first() < providerInterval.first()) {
            newInterval.setFirst(providerInterval.first());
         }
         if(newInterval.last() > providerInterval.last()) {
            newInterval.setLast(providerInterval.last());
         }
      }
   }
   if(!haveAValidDependentRecord) {
     //If no Finder has no valid time, then this record is also invalid for this time
     newInterval = ValidityInterval::invalidInterval();
   }
   oInterval = newInterval;
}

//
// const member functions
//

//
// static member functions
//
   }
}
