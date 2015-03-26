/*
 *  proxyfactoryproducer_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 4/8/05.
 *  Changed by Viji Sundararajan on 28-Jun-05
 */
#include <iostream>
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/test/DummyData.h"
#include "FWCore/Framework/test/DummyRecord.h"
#include "FWCore/Framework/test/DummyFinder.h"
#include "FWCore/Framework/test/DepRecord.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "cppunit/extensions/HelperMacros.h"

#include "FWCore/Utilities/interface/Exception.h"
using edm::eventsetup::test::DummyData;
using namespace edm::eventsetup;
using edm::ESProducer;
using edm::EventSetupRecordIntervalFinder;

class testEsproducer: public CppUnit::TestFixture 
{
CPPUNIT_TEST_SUITE(testEsproducer);

CPPUNIT_TEST(registerTest);
CPPUNIT_TEST(getFromTest);
CPPUNIT_TEST(getfromShareTest);
CPPUNIT_TEST(decoratorTest);
CPPUNIT_TEST(dependsOnTest);
CPPUNIT_TEST(labelTest);
CPPUNIT_TEST_EXCEPTION(failMultipleRegistration,cms::Exception);
CPPUNIT_TEST(forceCacheClearTest);
   
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}

  void registerTest();
  void getFromTest();
  void getfromShareTest();
  void decoratorTest();
  void dependsOnTest();
  void labelTest();
  void failMultipleRegistration();
  void forceCacheClearTest();

private:
class Test1Producer : public ESProducer {
public:
   Test1Producer() : ESProducer(), data_() {
      data_.value_ = 0;
      setWhatProduced(this);
   }
   const DummyData* produce(const DummyRecord& /*iRecord*/) {
      ++data_.value_;
      std::cout <<"produce called "<<data_.value_<<std::endl;
      return &data_;
   }
private:
   DummyData data_;
};

class MultiRegisterProducer : public ESProducer {
public:
   MultiRegisterProducer() : ESProducer(), data_() {
      setWhatProduced(this);
      setWhatProduced(this);
   }
   const DummyData* produce(const DummyRecord& /*iRecord*/) {
      return &data_;
   }
private:
   DummyData data_;
};


class ShareProducer : public ESProducer {
public:
   ShareProducer(): ptr_(new DummyData){
      ptr_->value_ = 0;
      setWhatProduced(this);
   }
   boost::shared_ptr<DummyData> produce(const DummyRecord& /*iRecord*/) {
      ++ptr_->value_;
      std::cout <<"produce called "<<ptr_->value_<<std::endl;
      return ptr_;
   }
private:
   boost::shared_ptr<DummyData> ptr_;
};

class LabelledProducer : public ESProducer {
public:
   enum {kFi, kFum};
   typedef edm::ESProducts< edm::es::L<DummyData,kFi>, edm::es::L<DummyData,kFum> > ReturnProducts;
   LabelledProducer(): ptr_(new DummyData), fi_(new DummyData){
      ptr_->value_ = 0;
      fi_->value_=0;
      setWhatProduced(this,"foo");
      setWhatProduced(this, &LabelledProducer::produceMore, edm::es::label("fi",kFi)("fum",kFum));
   }
   
   boost::shared_ptr<DummyData> produce(const DummyRecord& /*iRecord*/) {
      ++ptr_->value_;
      std::cout <<"\"foo\" produce called "<<ptr_->value_<<std::endl;
      return ptr_;
   }
   
   ReturnProducts produceMore(const DummyRecord&){
      using edm::es::L;
      using namespace edm;
      ++fi_->value_;

      L<DummyData,kFum> fum( new DummyData);
      fum->value_ = fi_->value_;
      
      return edm::es::products(fum, es::l<kFi>(fi_) );
   }
private:
   boost::shared_ptr<DummyData> ptr_;
   boost::shared_ptr<DummyData> fi_;
};

};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEsproducer);


void testEsproducer::registerTest()
{
   Test1Producer testProd;
   EventSetupRecordKey dummyRecordKey = EventSetupRecordKey::makeKey<DummyRecord>();
   CPPUNIT_ASSERT(testProd.isUsingRecord(dummyRecordKey));

   const DataProxyProvider::KeyedProxies& keyedProxies =
      testProd.keyedProxies(dummyRecordKey);

   CPPUNIT_ASSERT(keyedProxies.size() == 1);
}

void testEsproducer::getFromTest()
{
   EventSetupProvider provider;
   
   boost::shared_ptr<DataProxyProvider> pProxyProv(new Test1Producer);
   provider.add(pProxyProv);
   
   boost::shared_ptr<DummyFinder> pFinder(new DummyFinder);
   provider.add(boost::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));
   
   for(int iTime=1; iTime != 6; ++iTime) {
      const edm::Timestamp time(iTime);
      pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time) , edm::IOVSyncValue(time)));
      const edm::EventSetup& eventSetup = provider.eventSetupForInstance(edm::IOVSyncValue(time));
      edm::ESHandle<DummyData> pDummy;
      eventSetup.get<DummyRecord>().get(pDummy);
      CPPUNIT_ASSERT(0 != pDummy.product());
      std::cout <<pDummy->value_ << std::endl;
      CPPUNIT_ASSERT(iTime == pDummy->value_);
   }
}

void testEsproducer::getfromShareTest()
{
   EventSetupProvider provider;
   
   boost::shared_ptr<DataProxyProvider> pProxyProv(new ShareProducer);
   provider.add(pProxyProv);
   
   boost::shared_ptr<DummyFinder> pFinder(new DummyFinder);
   provider.add(boost::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));
   
   for(int iTime=1; iTime != 6; ++iTime) {
      const edm::Timestamp time(iTime);
      pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time) , edm::IOVSyncValue(time)));
      const edm::EventSetup& eventSetup = provider.eventSetupForInstance(edm::IOVSyncValue(time));
      edm::ESHandle<DummyData> pDummy;
      eventSetup.get<DummyRecord>().get(pDummy);
      CPPUNIT_ASSERT(0 != pDummy.product());
      std::cout <<pDummy->value_ << std::endl;
      CPPUNIT_ASSERT(iTime == pDummy->value_);
   }
}

void testEsproducer::labelTest()
{
   try {
   EventSetupProvider provider;
   
   boost::shared_ptr<DataProxyProvider> pProxyProv(new LabelledProducer);
   provider.add(pProxyProv);
   
   boost::shared_ptr<DummyFinder> pFinder(new DummyFinder);
   provider.add(boost::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));
   
   for(int iTime=1; iTime != 6; ++iTime) {
      const edm::Timestamp time(iTime);
      pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time) , edm::IOVSyncValue(time)));
      const edm::EventSetup& eventSetup = provider.eventSetupForInstance(edm::IOVSyncValue(time));
      edm::ESHandle<DummyData> pDummy;
      eventSetup.get<DummyRecord>().get("foo",pDummy);
      CPPUNIT_ASSERT(0 != pDummy.product());
      std::cout <<pDummy->value_ << std::endl;
      CPPUNIT_ASSERT(iTime == pDummy->value_);
      
      eventSetup.get<DummyRecord>().get("fi",pDummy);
      CPPUNIT_ASSERT(0 != pDummy.product());
      std::cout <<pDummy->value_ << std::endl;
      CPPUNIT_ASSERT(iTime == pDummy->value_);
      
      eventSetup.get<DummyRecord>().get("fum",pDummy);
      CPPUNIT_ASSERT(0 != pDummy.product());
      std::cout <<pDummy->value_ << std::endl;
      CPPUNIT_ASSERT(iTime == pDummy->value_);
   }
   } catch(const cms::Exception& iException) {
      std::cout <<"caught exception "<<iException.explainSelf()<<std::endl;
      throw;
   }
}

struct TestDecorator {
   static int s_pre;
   static int s_post;
   
   void pre(const DummyRecord&) {
      ++s_pre;
   }

   void post(const DummyRecord&) {
      ++s_post;
   }   
};

int TestDecorator::s_pre = 0;
int TestDecorator::s_post = 0;

class DecoratorProducer : public ESProducer {
public:
   DecoratorProducer(): ptr_(new DummyData){
      ptr_->value_ = 0;
      setWhatProduced(this, TestDecorator());
   }
   boost::shared_ptr<DummyData> produce(const DummyRecord& /*iRecord*/) {
      ++ptr_->value_;
      std::cout <<"produce called "<<ptr_->value_<<std::endl;
      return ptr_;
   }
private:
   boost::shared_ptr<DummyData> ptr_;
};

void testEsproducer::decoratorTest()
{
   EventSetupProvider provider;
   
   boost::shared_ptr<DataProxyProvider> pProxyProv(new DecoratorProducer);
   provider.add(pProxyProv);
   
   boost::shared_ptr<DummyFinder> pFinder(new DummyFinder);
   provider.add(boost::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));
   
   for(int iTime=1; iTime != 6; ++iTime) {
      const edm::Timestamp time(iTime);
      pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
      const edm::EventSetup& eventSetup = provider.eventSetupForInstance(edm::IOVSyncValue(time));
      edm::ESHandle<DummyData> pDummy;
      
      CPPUNIT_ASSERT(iTime - 1 == TestDecorator::s_pre);
      CPPUNIT_ASSERT(iTime - 1 == TestDecorator::s_post);
      eventSetup.get<DummyRecord>().get(pDummy);
      CPPUNIT_ASSERT(0 != pDummy.product());
      std::cout <<"pre "<<TestDecorator::s_pre << " post " << TestDecorator::s_post << std::endl;
      CPPUNIT_ASSERT(iTime == TestDecorator::s_pre);
      CPPUNIT_ASSERT(iTime == TestDecorator::s_post);
      CPPUNIT_ASSERT(iTime == pDummy->value_);
   }
}

class DepProducer : public ESProducer {
public:
   DepProducer(): ptr_(new DummyData){
      ptr_->value_ = 0;
      setWhatProduced(this , dependsOn(&DepProducer::callWhenDummyChanges, 
                                        &DepProducer::callWhenDummyChanges2,
                                        &DepProducer::callWhenDummyChanges3));
   }
   boost::shared_ptr<DummyData> produce(const DepRecord& /*iRecord*/) {
      return ptr_;
   }
   void callWhenDummyChanges(const DummyRecord&) {
      ++ptr_->value_;
      std::cout <<"callWhenDummyChanges called "<<ptr_->value_<<std::endl;
   }
   void callWhenDummyChanges2(const DummyRecord&) {
      ++ptr_->value_;
      std::cout <<"callWhenDummyChanges2 called "<<ptr_->value_<<std::endl;
   }
   void callWhenDummyChanges3(const DummyRecord&) {
      ++ptr_->value_;
      std::cout <<"callWhenDummyChanges3 called "<<ptr_->value_<<std::endl;
   }
   
private:
   boost::shared_ptr<DummyData> ptr_;
};

void testEsproducer::dependsOnTest()
{
   EventSetupProvider provider;
   
   boost::shared_ptr<DataProxyProvider> pProxyProv(new DepProducer);
   provider.add(pProxyProv);
   
   boost::shared_ptr<DummyFinder> pFinder(new DummyFinder);
   provider.add(boost::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));
   
   for(int iTime=1; iTime != 6; ++iTime) {
      const edm::Timestamp time(iTime);
      pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time), edm::IOVSyncValue(time)));
      const edm::EventSetup& eventSetup = provider.eventSetupForInstance(edm::IOVSyncValue(time));
      edm::ESHandle<DummyData> pDummy;
      
      eventSetup.get<DepRecord>().get(pDummy);
      CPPUNIT_ASSERT(0 != pDummy.product());
      CPPUNIT_ASSERT(3*iTime == pDummy->value_);
   }
}

void testEsproducer::failMultipleRegistration()
{
   MultiRegisterProducer dummy;
}

void testEsproducer::forceCacheClearTest()
{
   EventSetupProvider provider;
   
   boost::shared_ptr<DataProxyProvider> pProxyProv(new Test1Producer);
   provider.add(pProxyProv);
   
   boost::shared_ptr<DummyFinder> pFinder(new DummyFinder);
   provider.add(boost::shared_ptr<EventSetupRecordIntervalFinder>(pFinder));
   
   const edm::Timestamp time(1);
   pFinder->setInterval(edm::ValidityInterval(edm::IOVSyncValue(time) , edm::IOVSyncValue(time)));
   const edm::EventSetup& eventSetup = provider.eventSetupForInstance(edm::IOVSyncValue(time));
   {
      edm::ESHandle<DummyData> pDummy;
      eventSetup.get<DummyRecord>().get(pDummy);
      CPPUNIT_ASSERT(0 != pDummy.product());
      std::cout <<pDummy->value_ << std::endl;
      CPPUNIT_ASSERT(1 == pDummy->value_);
   }
   provider.forceCacheClear();
   {
      edm::ESHandle<DummyData> pDummy;
      eventSetup.get<DummyRecord>().get(pDummy);
      CPPUNIT_ASSERT(0 != pDummy.product());
      std::cout <<pDummy->value_ << std::endl;
      CPPUNIT_ASSERT(2 == pDummy->value_);
   }
}

