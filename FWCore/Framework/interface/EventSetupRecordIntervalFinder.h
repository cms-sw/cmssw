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
//

// system include files
#include <map>
#include <set>

// user include files
#include "FWCore/Framework/interface/ValidityInterval.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/ComponentDescription.h"

// forward declarations
namespace edm {

class EventSetupRecordIntervalFinder
{

   public:
      EventSetupRecordIntervalFinder() : intervals_() {}
      virtual ~EventSetupRecordIntervalFinder();

      // ---------- const member functions ---------------------
      std::set<eventsetup::EventSetupRecordKey> findingForRecords() const ;
   
      const eventsetup::ComponentDescription& descriptionForFinder() const { return description_;}
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      /**returns the 'default constructed' ValidityInterval if no valid interval.
       If upperbound is not known, it should be set to IOVSyncValue::invalidIOVSyncValue()
      */
      const ValidityInterval& findIntervalFor(const eventsetup::EventSetupRecordKey&,
                                            const IOVSyncValue&);
   
      void setDescriptionForFinder(const eventsetup::ComponentDescription& iDescription) {
        description_ = iDescription;
      }
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

      /** override this method if you need to delay setting what records you will be using until after all modules are loaded*/
      virtual void delaySettingRecords();
      // ---------- member data --------------------------------
      typedef  std::map<eventsetup::EventSetupRecordKey,ValidityInterval> Intervals;
      Intervals intervals_;

      eventsetup::ComponentDescription description_;
};

}
#endif
