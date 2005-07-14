#ifndef COREFRAMEWORK_DUMMYEVENTSETUPRECORDRETRIEVER_H
#define COREFRAMEWORK_DUMMYEVENTSETUPRECORDRETRIEVER_H
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DummyEventSetupRecordRetriever
// 
/**\class DummyEventSetupRecordRetriever DummyEventSetupRecordRetriever.h FWCore/Framework/interface/DummyEventSetupRecordRetriever.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Fri Apr 22 14:14:09 EDT 2005
// $Id: DummyEventSetupRecordRetriever.h,v 1.2 2005/06/23 20:01:12 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/test/DummyEventSetupRecord.h"
#include "FWCore/Framework/test/DummyEventSetupData.h"

// forward declarations
namespace edm {
   class DummyEventSetupRecordRetriever :
     public eventsetup::EventSetupRecordIntervalFinder, 
     public eventsetup::ESProducer
   {
   
   public:
      DummyEventSetupRecordRetriever() {
         this->findingRecord<DummyEventSetupRecord>();
         setWhatProduced(this);
      }
      
      std::auto_ptr<DummyEventSetupData> produce(const DummyEventSetupRecord&) {
         std::auto_ptr<DummyEventSetupData> data(new DummyEventSetupData(1));
         return data;
      }
   protected:

      virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                   const edm::Timestamp& iTime, 
                                   edm::ValidityInterval& iInterval) {
         iInterval = edm::ValidityInterval(Timestamp(1),
                                            Timestamp::endOfTime());
      }

   private:
      DummyEventSetupRecordRetriever(const DummyEventSetupRecordRetriever&); // stop default

      const DummyEventSetupRecordRetriever& operator=(const DummyEventSetupRecordRetriever&); // stop default

      // ---------- member data --------------------------------

};
}
#endif /* COREFRAMEWORK_DUMMYEVENTSETUPRECORDRETRIEVER_H */
