#ifndef Framework_EventSetupRecordIntervalFinder_h
#define Framework_EventSetupRecordIntervalFinder_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     EventSetupRecordIntervalFinder
// 
/**\class EventSetupRecordIntervalFinder EventSetupRecordIntervalFinder.h FWCore/Framework/interface/EventSetupRecordIntervalFinder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Tue Mar 29 16:15:11 EST 2005
// $Id: EventSetupRecordIntervalFinder.h,v 1.5 2005/09/01 05:35:01 wmtan Exp $
//

// system include files
#include <map>
#include <set>

// user include files
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

// forward declarations
namespace edm {
   namespace eventsetup {

class EventSetupRecordIntervalFinder
{

   public:
      EventSetupRecordIntervalFinder() {}
      virtual ~EventSetupRecordIntervalFinder();

      // ---------- const member functions ---------------------
      std::set<EventSetupRecordKey> findingForRecords() const ;
   
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   /**returns the 'default constructed' ValidityInterval if no valid interval.
   If upperbound is not known, it should be set to IOVSyncValue::invalidIOVSyncValue()
   */
   const ValidityInterval& findIntervalFor(const EventSetupRecordKey&,
                                            const IOVSyncValue&);
   
   protected:
      virtual void setIntervalFor(const EventSetupRecordKey&,
                                   const IOVSyncValue& , 
                                   ValidityInterval&) = 0;

      template< class T>
         void findingRecord() {
            findingRecordWithKey(EventSetupRecordKey::makeKey<T>());
         }
      
      void findingRecordWithKey(const EventSetupRecordKey&);
      
   private:
      EventSetupRecordIntervalFinder(const EventSetupRecordIntervalFinder&); // stop default

      const EventSetupRecordIntervalFinder& operator=(const EventSetupRecordIntervalFinder&); // stop default

      // ---------- member data --------------------------------
      typedef  std::map<EventSetupRecordKey,ValidityInterval> Intervals;
      Intervals intervals_;
      
};

   }
}
#endif
