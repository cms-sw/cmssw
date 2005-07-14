#ifndef EVENTSETUP_DUMMYPROVIDER_H
#define EVENTSETUP_DUMMYPROVIDER_H
// -*- C++ -*-
//
// Package:     Framework
// Class  :     DummyProvider
// 
/**\class DummyProvider DummyProvider.h FWCore/Framework/interface/DummyProvider.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu May 26 13:37:48 EDT 2005
// $Id: DummyProxyProvider.h,v 1.2 2005/06/23 20:01:12 wmtan Exp $
//

// system include files

// user include files
#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/Framework/test/DummyData.h"

#include "FWCore/Framework/interface/DataProxyTemplate.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      namespace test {
class WorkingDummyProxy : public edm::eventsetup::DataProxyTemplate<DummyRecord, DummyData> {
public:
   WorkingDummyProxy(const DummyData* iDummy) : data_(iDummy) {}
   
protected:
   
   const value_type* make(const record_type&, const DataKey&) {
      return data_ ;
   }
   void invalidateCache() {
   }   
private:
   const DummyData* data_;
};


class DummyProxyProvider : public edm::eventsetup::DataProxyProvider {
public:
   DummyProxyProvider() {
      //std::cout <<"constructed provider"<<std::endl;
      usingRecord<DummyRecord>();
   }
   void newInterval(const eventsetup::EventSetupRecordKey& iRecordType,
                     const ValidityInterval& iInterval) {
      //do nothing
   }
protected:
   void registerProxies(const eventsetup::EventSetupRecordKey&, KeyedProxies& iProxies) {
      //std::cout <<"registered proxy"<<std::endl;
      
      boost::shared_ptr<WorkingDummyProxy> pProxy(new WorkingDummyProxy(&dummy_));
      insertProxy(iProxies, pProxy);
   }
   
private:
   DummyData dummy_;
};

      }
   }
}
#endif /* EVENTSETUP_DUMMYPROVIDER_H */
