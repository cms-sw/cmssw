// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     DependentRecordIntervalFinder
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Sat Apr 30 19:37:22 EDT 2005
// $Id: DependentRecordIntervalFinder.cc,v 1.1 2005/05/03 19:33:40 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/CoreFramework/interface/DependentRecordIntervalFinder.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordProvider.h"


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
DependentRecordIntervalFinder::DependentRecordIntervalFinder(const EventSetupRecordKey& iKey )
{
   findingRecordWithKey(iKey);
}

// DependentRecordIntervalFinder::DependentRecordIntervalFinder( const DependentRecordIntervalFinder& rhs )
// {
//    // do actual copying here;
// }

DependentRecordIntervalFinder::~DependentRecordIntervalFinder()
{
}

//
// assignment operators
//
// const DependentRecordIntervalFinder& DependentRecordIntervalFinder::operator=( const DependentRecordIntervalFinder& rhs )
// {
//   //An exception safe implementation is
//   DependentRecordIntervalFinder temp(rhs);
//   swap( rhs );
//
//   return *this;
// }

//
// member functions
//
void 
DependentRecordIntervalFinder::addProviderWeAreDependentOn( boost::shared_ptr<EventSetupRecordProvider> iProvider )
{
   providers_.push_back(iProvider );
}

void 
DependentRecordIntervalFinder::setIntervalFor( const EventSetupRecordKey&,
                                               const Timestamp& iTime, 
                                               ValidityInterval& oInterval)
{
   //I am assuming that an invalidTime is always less then the first valid time
   assert( Timestamp::invalidTimestamp() < Timestamp::beginOfTime() );
   if( providers_.size() == 0 ) {
      oInterval = ValidityInterval::invalidInterval();
      return;
   }
   ValidityInterval newInterval( Timestamp::beginOfTime(), Timestamp::endOfTime() );
   for(Providers::iterator itProvider = providers_.begin();
       itProvider != providers_.end();
       ++itProvider ) {
      if( ! (*itProvider)->setValidityIntervalFor( iTime ) ) {
         //If one Finder has no valid time, then this record is also invalid for this time
         newInterval = ValidityInterval::invalidInterval();
         break;
      } else {
         ValidityInterval providerInterval = (*itProvider)->validityInterval();
         if( newInterval.first() < providerInterval.first() ) {
            newInterval.setFirst( providerInterval.first() );
         }
         if( newInterval.last() > providerInterval.last() ) {
            newInterval.setLast( providerInterval.last() );
         }
      }
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
