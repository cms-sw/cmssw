#ifndef Framework_DummyFinder_h
#define Framework_DummyFinder_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DummyFinder
// 
/**\class DummyFinder DummyFinder.h FWCore/Framework/interface/DummyFinder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Sat Apr 16 18:47:04 EDT 2005
// $Id: DummyFinder.h,v 1.9 2006/08/31 23:26:25 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/test/DummyRecord.h"

// forward declarations

class DummyFinder : public edm::EventSetupRecordIntervalFinder {
public:
   DummyFinder() :edm::EventSetupRecordIntervalFinder(), interval_() {
      this->findingRecord<DummyRecord>();
   }
   
   void setInterval(const edm::ValidityInterval& iInterval) {
      interval_ = iInterval;
   }
protected:
   virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                const edm::IOVSyncValue& iTime, 
                                edm::ValidityInterval& iInterval) {
      if(interval_.validFor(iTime)) {
         iInterval = interval_;
      } else {
         if(interval_.last() == edm::IOVSyncValue::invalidIOVSyncValue() &&
             interval_.first() != edm::IOVSyncValue::invalidIOVSyncValue() &&
             interval_.first() <= iTime) {
            iInterval = interval_;
         }else {
            iInterval = edm::ValidityInterval();
         }
      }
   }
private:
   edm::ValidityInterval interval_;   
};


#endif
