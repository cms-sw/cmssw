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
// $Id: DependentRecordIntervalFinder.cc,v 1.8 2007/06/13 11:10:26 chrjones Exp $
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
  providers_(),
  m_previousSyncTo(IOVSyncValue::invalidIOVSyncValue())
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
DependentRecordIntervalFinder::setAlternateFinder(boost::shared_ptr<EventSetupRecordIntervalFinder> iOther)
{
  alternate_ = iOther;
}

void 
DependentRecordIntervalFinder::setIntervalFor(const EventSetupRecordKey& iKey,
                                               const IOVSyncValue& iTime, 
                                               ValidityInterval& oInterval)
{
   //I am assuming that an invalidTime is always less then the first valid time
   assert(IOVSyncValue::invalidIOVSyncValue() < IOVSyncValue::beginOfTime());
   if(providers_.size() == 0 && alternate_.get() == 0 ) {
      oInterval = ValidityInterval::invalidInterval();
      return;
   }
   bool haveAValidDependentRecord = false;
   ValidityInterval newInterval(IOVSyncValue::beginOfTime(), IOVSyncValue::endOfTime());
   
   if (alternate_.get() != 0) {
     ValidityInterval test = alternate_->findIntervalFor(iKey, iTime);
     if ( test != ValidityInterval::invalidInterval() ) {
       haveAValidDependentRecord =true;
       newInterval = test;
     }
   }
   bool intervalsWereComparible = true;
   for(Providers::iterator itProvider = providers_.begin(), itProviderEnd = providers_.end();
       itProvider != itProviderEnd;
       ++itProvider) {
      if((*itProvider)->setValidityIntervalFor(iTime)) {
         haveAValidDependentRecord=true;
         ValidityInterval providerInterval = (*itProvider)->validityInterval();
	 if( (!newInterval.first().comparable(providerInterval.first())) ||
	     (!newInterval.last().comparable(providerInterval.last()))) {
	   intervalsWereComparible=false;
	   break;
	 }
         if(newInterval.first() < providerInterval.first()) {
            newInterval.setFirst(providerInterval.first());
         }
         if(newInterval.last() > providerInterval.last()) {
            newInterval.setLast(providerInterval.last());
         }
      }
   }
   if(intervalsWereComparible) {
     if(!haveAValidDependentRecord) {
       //If no Finder has no valid time, then this record is also invalid for this time
       newInterval = ValidityInterval::invalidInterval();
     }
     oInterval = newInterval;
     return;
   }
   //handle the case where some providers use time and others use run/lumi/event
   // in this case all we can do is find an IOV which changed since last time
   // and use its start time to do the synching and use an 'invalid' end time
   // so the system always checks back to see if something has changed

   for(Providers::iterator itProvider = providers_.begin(), itProviderEnd = providers_.end();
       itProvider != itProviderEnd;
       ++itProvider) {
      if((*itProvider)->setValidityIntervalFor(iTime)) {
         ValidityInterval providerInterval = (*itProvider)->validityInterval();
	 if(!providerInterval.validFor(m_previousSyncTo)) {
	   //NOTE if the above is never true than old interval should be fine
	   providerInterval.setLast(IOVSyncValue::invalidIOVSyncValue());
	   oInterval = providerInterval;
	   break;
         }
      }
   }
   m_previousSyncTo = iTime;
}

//
// const member functions
//

//
// static member functions
//
   }
}
