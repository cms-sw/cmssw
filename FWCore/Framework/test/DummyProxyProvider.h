#ifndef EVENTSETUP_DUMMYPROVIDER_H
#define EVENTSETUP_DUMMYPROVIDER_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     DummyProvider
// 
/**\class DummyProvider DummyProvider.h Core/CoreFramework/interface/DummyProvider.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu May 26 13:37:48 EDT 2005
// $Id: DummyProxyProvider.h,v 1.1 2005/05/26 20:58:38 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/CoreFramework/test/DummyRecord.h"
#include "FWCore/CoreFramework/test/DummyData.h"

#include "FWCore/CoreFramework/interface/DataProxyTemplate.h"
#include "FWCore/CoreFramework/interface/DataProxyProvider.h"

// forward declarations
namespace edm {
   namespace eventsetup {
      namespace test {
class WorkingDummyProxy : public edm::eventsetup::DataProxyTemplate<DummyRecord, DummyData> {
public:
   WorkingDummyProxy( const DummyData* iDummy ) : data_(iDummy) {}
   
protected:
   
   const value_type* make( const record_type&, const DataKey&) {
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
   void newInterval( const eventsetup::EventSetupRecordKey& iRecordType,
                     const ValidityInterval& iInterval ) {
      //do nothing
   }
protected:
   void registerProxies( const eventsetup::EventSetupRecordKey&, KeyedProxies& iProxies ) {
      //std::cout <<"registered proxy"<<std::endl;
      
      boost::shared_ptr<WorkingDummyProxy> pProxy( new WorkingDummyProxy(&dummy_) );
      insertProxy(iProxies, pProxy);
   }
   
private:
   DummyData dummy_;
};

      }
   }
}
#endif /* EVENTSETUP_DUMMYPROVIDER_H */
