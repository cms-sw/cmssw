/*
 *  eventsetuprecord_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 3/29/05.
 *  Changed by Viji on 06/07/2005
 */

#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/CoreFramework/interface/EventSetupRecordImplementation.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordProviderTemplate.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordProviderFactoryManager.h"
#include "FWCore/CoreFramework/interface/EventSetupRecordProviderFactoryTemplate.h"

using namespace edm;
using namespace edm::eventsetup;

class DummyRecord : public edm::eventsetup::EventSetupRecordImplementation<DummyRecord> { public:
   const DataProxy* find(const edm::eventsetup::DataKey& iKey ) const {
      return edm::eventsetup::EventSetupRecord::find( iKey );
   }
};

#include "FWCore/CoreFramework/interface/HCMethods.icc"
#include "FWCore/CoreFramework/interface/HCTypeTagTemplate.icc"
//HCMethods<T, T, EventSetup, EventSetupRecordKey, EventSetupRecordKey::IdTag >
template<>
const char*
edm::eventsetup::heterocontainer::HCTypeTagTemplate<DummyRecord, edm::eventsetup::EventSetupRecordKey>::className() {
   return "DummyRecord";
}

//create an instance of the factory
static eventsetup::EventSetupRecordProviderFactoryTemplate<DummyRecord> s_factory;

namespace eventsetuprecord_t {
class Dummy {};
}
using eventsetuprecord_t::Dummy;
typedef edm::eventsetup::MakeDataException<DummyRecord,Dummy> ExceptionType;
typedef edm::eventsetup::NoDataException<Dummy> NoDataExceptionType;

class testEventsetupRecord: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testEventsetupRecord);

CPPUNIT_TEST(factoryTest);
CPPUNIT_TEST_EXCEPTION(doGetTest,ExceptionType);
CPPUNIT_TEST_EXCEPTION(get2Test,NoDataExceptionType);
CPPUNIT_TEST_EXCEPTION(getTest,ExceptionType);
CPPUNIT_TEST(proxyTest);

CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void factoryTest();
  void getTest();
  void get2Test();
  void doGetTest();
  void proxyTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEventsetupRecord);
void testEventsetupRecord::factoryTest()
{
   std::auto_ptr<EventSetupRecordProvider> dummyProvider =
   EventSetupRecordProviderFactoryManager::instance().makeRecordProvider(
                              EventSetupRecordKey::makeKey<DummyRecord>());
   
   CPPUNIT_ASSERT(0 != dynamic_cast<EventSetupRecordProviderTemplate<DummyRecord>*>(&(*dummyProvider)));

}   

//namespace eventsetuprecord_t {
 //  class Dummy {};
//}
//using eventsetuprecord_t::Dummy;

template<>
const char*
edm::eventsetup::heterocontainer::HCTypeTagTemplate<Dummy, edm::eventsetup::DataKey>::className() {
   return "Dummy";
}

#include "FWCore/CoreFramework/interface/DataProxyTemplate.h"

class FailingDummyProxy : public eventsetup::DataProxyTemplate<DummyRecord, Dummy> {
protected:
   const value_type* make(const record_type&, const DataKey&) {
      return 0 ;
   }
   void invalidateCache() {
   }   
};

class WorkingDummyProxy : public eventsetup::DataProxyTemplate<DummyRecord, Dummy> {
public:
   WorkingDummyProxy(const Dummy* iDummy) : data_(iDummy) {}

protected:
   
   const value_type* make(const record_type&, const DataKey&) {
      return data_ ;
   }
   void invalidateCache() {
   }   
private:
   const Dummy* data_;
};

void testEventsetupRecord::proxyTest()
{
   DummyRecord dummyRecord;
   FailingDummyProxy dummyProxy;
   
   const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(),
                              "");
   
   CPPUNIT_ASSERT(0 == dummyRecord.find(dummyDataKey));
   //BOOST_CHECK_THROW(dummyRecord.get(dummyPtr), edm::eventsetup::MakeDataException<DummyRecord,Dummy>);

   
   dummyRecord.add(dummyDataKey,
                    &dummyProxy);
   
   CPPUNIT_ASSERT(&dummyProxy == dummyRecord.find(dummyDataKey));

   const DataKey dummyFredDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(),
                                  "fred");
   CPPUNIT_ASSERT(0 == dummyRecord.find(dummyFredDataKey));

}

#include "FWCore/CoreFramework/interface/ESHandle.h"
#include "FWCore/CoreFramework/interface/recordGetImplementation.icc"

void testEventsetupRecord::getTest()
{
   DummyRecord dummyRecord;
   FailingDummyProxy dummyProxy;

   const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(),
                              "");

   ESHandle<Dummy> dummyPtr;
   //typedef edm::eventsetup::NoDataException<Dummy> NoDataExceptionType;
   //dummyRecord.get(dummyPtr);
   //BOOST_CHECK_THROW(dummyRecord.get(dummyPtr), NoDataExceptionType) ;
   
   dummyRecord.add(dummyDataKey,
                    &dummyProxy);

   typedef edm::eventsetup::MakeDataException<DummyRecord,Dummy> ExceptionType;
   dummyRecord.get(dummyPtr);
   //BOOST_CHECK_THROW(dummyRecord.get(dummyPtr), ExceptionType);

   Dummy myDummy;
   WorkingDummyProxy workingProxy(&myDummy);
   
   const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyProxy::value_type>(),
                              "working");

   dummyRecord.add(workingDataKey,
                    &workingProxy);

   dummyRecord.get(dummyPtr, "working");
   
   CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);

   const std::string workingString("working");
   
   dummyRecord.get(dummyPtr, workingString);
   CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);
   
}

void testEventsetupRecord::get2Test()
{
   DummyRecord dummyRecord;
   FailingDummyProxy dummyProxy;

   const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(),
                              "");

   ESHandle<Dummy> dummyPtr;
   typedef edm::eventsetup::NoDataException<Dummy> NoDataExceptionType;
   dummyRecord.get(dummyPtr);
   //BOOST_CHECK_THROW(dummyRecord.get(dummyPtr), NoDataExceptionType) ;
   
   dummyRecord.add(dummyDataKey,
                    &dummyProxy);

   //typedef edm::eventsetup::MakeDataException<DummyRecord,Dummy> ExceptionType;
   //dummyRecord.get(dummyPtr);
   //BOOST_CHECK_THROW(dummyRecord.get(dummyPtr), ExceptionType);

   Dummy myDummy;
   WorkingDummyProxy workingProxy(&myDummy);
   
   const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyProxy::value_type>(),
                              "working");

   dummyRecord.add(workingDataKey,
                    &workingProxy);

   dummyRecord.get(dummyPtr, "working");
   
   CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);

   const std::string workingString("working");
   
   dummyRecord.get(dummyPtr, workingString);
   CPPUNIT_ASSERT(&(*dummyPtr) == &myDummy);
}

void testEventsetupRecord::doGetTest()
{
   DummyRecord dummyRecord;
   FailingDummyProxy dummyProxy;
   
   const DataKey dummyDataKey(DataKey::makeTypeTag<FailingDummyProxy::value_type>(),
                              "");
   
   CPPUNIT_ASSERT(!dummyRecord.doGet(dummyDataKey)) ;
   
   dummyRecord.add(dummyDataKey,
                   &dummyProxy);
   
   typedef edm::eventsetup::MakeDataException<DummyRecord,Dummy> ExceptionType;
   dummyRecord.doGet(dummyDataKey);
   //BOOST_CHECK_THROW(dummyRecord.doGet(dummyDataKey), ExceptionType);
   
   Dummy myDummy;
   WorkingDummyProxy workingProxy(&myDummy);
   
   const DataKey workingDataKey(DataKey::makeTypeTag<WorkingDummyProxy::value_type>(),
                                "working");
   
   dummyRecord.add(workingDataKey,
                   &workingProxy);
   
   CPPUNIT_ASSERT(dummyRecord.doGet(workingDataKey) );
   
}
