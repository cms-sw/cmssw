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
// $Id: EventSetupRecordIntervalFinder.h,v 1.7 2005/10/03 23:20:51 chrjones Exp $
//

// system include files
#include <map>
#include <set>

// user include files
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

// forward declarations
namespace edm {

class EventSetupRecordIntervalFinder
{

   public:
      EventSetupRecordIntervalFinder() : intervals_() {}
      virtual ~EventSetupRecordIntervalFinder();

      // ---------- const member functions ---------------------
      std::set<eventsetup::EventSetupRecordKey> findingForRecords() const ;
   
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   /**returns the 'default constructed' ValidityInterval if no valid interval.
   If upperbound is not known, it should be set to IOVSyncValue::invalidIOVSyncValue()
   */
   const ValidityInterval& findIntervalFor(const eventsetup::EventSetupRecordKey&,
                                            const IOVSyncValue&);
   
   protected:
      virtual void setIntervalFor(const eventsetup::EventSetupRecordKey&,
                                   const IOVSyncValue& , 
                                   ValidityInterval&) = 0;

      template< class T>
         void findingRecord() {
            findingRecordWithKey(eventsetup::EventSetupRecordKey::makeKey<T>());
         }
      
      void findingRecordWithKey(const eventsetup::EventSetupRecordKey&);
      
   private:
      EventSetupRecordIntervalFinder(const EventSetupRecordIntervalFinder&); // stop default

      const EventSetupRecordIntervalFinder& operator=(const EventSetupRecordIntervalFinder&); // stop default

      // ---------- member data --------------------------------
      typedef  std::map<eventsetup::EventSetupRecordKey,ValidityInterval> Intervals;
      Intervals intervals_;
      
};

}
#endif
