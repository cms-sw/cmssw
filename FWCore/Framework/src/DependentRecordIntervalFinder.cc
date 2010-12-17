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
// $Id: DependentRecordIntervalFinder.cc,v 1.12 2010/12/17 04:31:54 chrjones Exp $
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
DependentRecordIntervalFinder::setAlternateFinder(boost::shared_ptr<EventSetupRecordIntervalFinder> iOther)
{
  alternate_ = iOther;
}

void 
DependentRecordIntervalFinder::setIntervalFor(const EventSetupRecordKey& iKey,
                                               const IOVSyncValue& iTime, 
                                               ValidityInterval& oInterval)
{
   //NOTE: oInterval is the last value that was used so if nothing changed do not modify oInterval
   
   //I am assuming that an invalidTime is always less then the first valid time
   assert(IOVSyncValue::invalidIOVSyncValue() < IOVSyncValue::beginOfTime());
   if(providers_.size() == 0 && alternate_.get() == 0 ) {
      oInterval = ValidityInterval::invalidInterval();
      return;
   }
   bool haveAValidDependentRecord = false;
   bool allRecordsValid = true;
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
      } else {
	allRecordsValid = false;
      }
   }
   if(intervalsWereComparible) {
     if(!haveAValidDependentRecord) {
       //If no Finder has no valid time, then this record is also invalid for this time
       newInterval = ValidityInterval::invalidInterval();
     }
     if(!allRecordsValid) {
       //since some of the dependent providers do not have a valid IOV for this syncvalue
       // we do not know what the true end IOV is.  Therefore we must make it open ended
       // so that we can check each time to see if those providers become valid.
       newInterval.setLast(IOVSyncValue::invalidIOVSyncValue());
     }
     oInterval = newInterval;
     return;
   }
   //handle the case where some providers use time and others use run/lumi/event
   // in this case all we can do is find an IOV which changed since last time
   // and use its start time to do the synching and use an 'invalid' end time
   // so the system always checks back to see if something has changed
   if (previousIOVs_.empty()) {
      std::vector<ValidityInterval> tmp(providers_.size(),ValidityInterval());
      previousIOVs_.swap(tmp);
   }

   //I'm using an heuristic to pick a reasonable starting point for the IOV. The idea is to
   // assume that lumi sections are 23 seconds long and therefore if we take a difference between
   // iTime and the beginning of a changed IOV we can pick the changed IOV with the start time 
   // closest to iTime. This doesn't have to be perfect, we just want to reduce the dependency
   // on provider order to make the jobs more deterministic
   
   bool hadChangedIOV = false;
   //both start at the smallest value
   EventID closestID;
   Timestamp closestTimeStamp(0);
   std::vector<ValidityInterval>::iterator itIOVs = previousIOVs_.begin();
   for(Providers::iterator itProvider = providers_.begin(), itProviderEnd = providers_.end();
       itProvider != itProviderEnd;
       ++itProvider, ++itIOVs) {
      if((*itProvider)->setValidityIntervalFor(iTime)) {
         ValidityInterval providerInterval = (*itProvider)->validityInterval();
	 if(*itIOVs != providerInterval) {
            hadChangedIOV = true;
            if(providerInterval.first().time().value() == 0) {
               //this is a run/lumi based one
               if( closestID < providerInterval.first().eventID()) {
                  closestID = providerInterval.first().eventID();
               }
            } else {
               if(closestTimeStamp < providerInterval.first().time()) {
                  closestTimeStamp = providerInterval.first().time();
               }
            }
            *itIOVs = providerInterval;
         }
      }
   }
   if(hadChangedIOV) {
      if(closestID.run() !=0) {
         if(closestTimeStamp.value() == 0) {
            //no time
            oInterval = ValidityInterval(IOVSyncValue(closestID), IOVSyncValue::invalidIOVSyncValue());
         } else {
            if(closestID.run() == iTime.eventID().run()) {
               //can compare time to lumi
               const unsigned long long kLumiTimeLength = 23;
               
               if( (iTime.eventID().luminosityBlock() - closestID.luminosityBlock())*kLumiTimeLength < 
                  iTime.time().unixTime() - closestTimeStamp.unixTime() ) {
                  //closestID was closer
                  oInterval = ValidityInterval(IOVSyncValue(closestID), IOVSyncValue::invalidIOVSyncValue());
               } else {
                  oInterval = ValidityInterval(IOVSyncValue(closestTimeStamp), IOVSyncValue::invalidIOVSyncValue());
               }
            } else {
               //since we don't know how to change run # into time we can't compare
               // so if we have a time just use it
               oInterval = ValidityInterval(IOVSyncValue(closestTimeStamp), IOVSyncValue::invalidIOVSyncValue());
            }
         }
      } else {
         oInterval = ValidityInterval( IOVSyncValue(closestTimeStamp), IOVSyncValue::invalidIOVSyncValue());
      }
   }
}

//
// const member functions
//

//
// static member functions
//
   }
}
