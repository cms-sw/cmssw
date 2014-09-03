/*
 *  serviceregistry_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 9/7/05.
 *
 */

//need to open a 'back door' to be able to setup the ServiceRegistry
#define private public
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#undef private
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/test/stubs/DummyService.h"

#include <cppunit/extensions/HelperMacros.h>

#include "FWCore/PluginManager/interface/ProblemTracker.h"

#include "boost/thread/thread.hpp"

#include <atomic>

class testServiceRegistry: public CppUnit::TestFixture
{
   CPPUNIT_TEST_SUITE(testServiceRegistry);
   
   CPPUNIT_TEST(loadTest);
   CPPUNIT_TEST(hierarchyTest);
   CPPUNIT_TEST(threadTest);
   CPPUNIT_TEST(externalServiceTest);
   CPPUNIT_TEST(saveConfigWithExternalTest);

   
   CPPUNIT_TEST_SUITE_END();
public:
      void setUp(){}
   void tearDown(){}
   
   void loadTest();
   void hierarchyTest();
   void threadTest();
   void externalServiceTest();
   void saveConfigWithExternalTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testServiceRegistry);

void
testServiceRegistry::loadTest()
{
   edm::AssertHandler ah;

   std::vector<edm::ParameterSet> pss;
   
   edm::ParameterSet ps;
   std::string typeName("DummyService");
   ps.addParameter("@service_type", typeName);
   int value = 2;
   ps.addParameter("value", value);
   pss.push_back(ps);

   edm::ServiceToken token(edm::ServiceRegistry::createSet(pss));

   edm::ServiceRegistry::Operate operate(token);
   edm::Service<testserviceregistry::DummyService> dummy;
   CPPUNIT_ASSERT(dummy);
   CPPUNIT_ASSERT(dummy.isAvailable());
   CPPUNIT_ASSERT(dummy->value() == 2);
}

namespace {
   struct DummyService { int value_; };
}

void
testServiceRegistry::externalServiceTest()
{
   edm::AssertHandler ah;

   {
      std::auto_ptr<DummyService> dummyPtr(new DummyService);
      dummyPtr->value_ = 2;
      edm::ServiceToken token(edm::ServiceRegistry::createContaining(dummyPtr));
      {         
         edm::ServiceRegistry::Operate operate(token);
         edm::Service<DummyService> dummy;
         CPPUNIT_ASSERT(dummy);
         CPPUNIT_ASSERT(dummy.isAvailable());
         CPPUNIT_ASSERT(dummy->value_ == 2);
      }
      {
         std::vector<edm::ParameterSet> pss;
         
         edm::ParameterSet ps;
         std::string typeName("DummyService");
         ps.addParameter("@service_type", typeName);
         int value = 2;
         ps.addParameter("value", value);
         pss.push_back(ps);
      
         edm::ServiceToken token(edm::ServiceRegistry::createSet(pss));
         edm::ServiceToken token2(edm::ServiceRegistry::createContaining(dummyPtr,
                                                                         token,
                                                                         edm::serviceregistry::kOverlapIsError));
         
         edm::ServiceRegistry::Operate operate(token2);
         edm::Service<testserviceregistry::DummyService> dummy;
         CPPUNIT_ASSERT(dummy);
         CPPUNIT_ASSERT(dummy.isAvailable());
         CPPUNIT_ASSERT(dummy->value() == 2);
      }
   }

   {
      std::auto_ptr<DummyService> dummyPtr(new DummyService);
      auto wrapper = std::make_shared<edm::serviceregistry::ServiceWrapper<DummyService> >(dummyPtr);
      edm::ServiceToken token(edm::ServiceRegistry::createContaining(wrapper));

      wrapper->get().value_ = 2;

      {
         edm::ServiceRegistry::Operate operate(token);
         edm::Service<DummyService> dummy;
         CPPUNIT_ASSERT(dummy);
         CPPUNIT_ASSERT(dummy.isAvailable());
         CPPUNIT_ASSERT(dummy->value_ == 2);
      }
      {
         std::vector<edm::ParameterSet> pss;
         
         edm::ParameterSet ps;
         std::string typeName("DummyService");
         ps.addParameter("@service_type", typeName);
         int value = 2;
         ps.addParameter("value", value);
         pss.push_back(ps);
         
         edm::ServiceToken token(edm::ServiceRegistry::createSet(pss));
         edm::ServiceToken token2(edm::ServiceRegistry::createContaining(dummyPtr,
                                                                         token,
                                                                         edm::serviceregistry::kOverlapIsError));
         
         edm::ServiceRegistry::Operate operate(token2);
         edm::Service<testserviceregistry::DummyService> dummy;
         CPPUNIT_ASSERT(dummy);
         CPPUNIT_ASSERT(dummy.isAvailable());
         CPPUNIT_ASSERT(dummy->value() == 2);
      }
      
   }
}

void 
testServiceRegistry::saveConfigWithExternalTest()
{
   //In the HLT the PrescaleService is created once and then used again
   // even if a new EventProcessor is created.  However, the '@save_config' must
   // still be added to the Service's PSet even if that service is not created
   edm::AssertHandler ah;

   std::vector<edm::ParameterSet> pss;
   
   {
      edm::ParameterSet ps;
      std::string typeName("DummyStoreConfigService");
      ps.addParameter("@service_type", typeName);
      pss.push_back(ps);
   }
   edm::ServiceToken token(edm::ServiceRegistry::createSet(pss));
   CPPUNIT_ASSERT( pss[0].exists("@save_config") );

   pss.clear();

   {
      edm::ParameterSet ps;
      std::string typeName("DummyStoreConfigService");
      ps.addParameter("@service_type", typeName);
      pss.push_back(ps);
   }
   
   //create the services
   edm::ServiceToken token2(edm::ServiceRegistry::createSet(pss, token, edm::serviceregistry::kTokenOverrides));

   CPPUNIT_ASSERT( pss[0].exists("@save_config") );
   
}

void
testServiceRegistry::hierarchyTest()
{
   edm::AssertHandler ah;
   
   std::vector<edm::ParameterSet> pss;
   {
      edm::ParameterSet ps;
      std::string typeName("DummyService");
      ps.addParameter("@service_type", typeName);
      int value = 1;
      ps.addParameter("value", value);
      pss.push_back(ps);
   }
   edm::ServiceToken token1(edm::ServiceRegistry::createSet(pss));

   pss.clear();
   {
      edm::ParameterSet ps;
      std::string typeName("DummyService");
      ps.addParameter("@service_type", typeName);
      int value = 2;
      ps.addParameter("value", value);
      pss.push_back(ps);
   }
   edm::ServiceToken token2(edm::ServiceRegistry::createSet(pss));
   
   
   edm::ServiceRegistry::Operate operate1(token1);
   {
      edm::Service<testserviceregistry::DummyService> dummy;
      CPPUNIT_ASSERT(dummy->value() == 1);
   }
   {
      edm::ServiceRegistry::Operate operate2(token2);
      edm::Service<testserviceregistry::DummyService> dummy;
      CPPUNIT_ASSERT(dummy->value() == 2);
   }
   {
      edm::Service<testserviceregistry::DummyService> dummy;
      CPPUNIT_ASSERT(dummy->value() == 1);
   }
}

namespace {
   struct UniqueRegistry {

      UniqueRegistry(void* iReg) : otherRegistry_(iReg) {}
      
      void operator()(){
         isUnique_ = (otherRegistry_ != &(edm::ServiceRegistry::instance()));
      }
      void* otherRegistry_;
      static std::atomic<bool> isUnique_;
   };
   std::atomic<bool> UniqueRegistry::isUnique_{false};

   struct PassServices {
      PassServices(edm::ServiceToken iToken,
                    bool& oSucceeded,
                    bool& oCaughtException) :
                    token_(iToken), success_(&oSucceeded), caught_(&oCaughtException)
   { *success_ = false; *caught_ = false; }

      void operator()() {
         try  {
            edm::ServiceRegistry::Operate operate(token_);
            edm::Service<testserviceregistry::DummyService> dummy;
            *success_ = dummy->value()==1;
         } catch(...){
            *caught_=true;
         }
      }
      
      edm::ServiceToken token_;
      bool* success_;
      bool* caught_;
      
   };
}


void 
testServiceRegistry::threadTest()
{
   UniqueRegistry::isUnique_ = false;
   void* value = &(edm::ServiceRegistry::instance());
   UniqueRegistry unique(value);
   boost::thread testUniqueness(unique);
   testUniqueness.join();
   CPPUNIT_ASSERT(UniqueRegistry::isUnique_);



   edm::AssertHandler ah;
   
   std::vector<edm::ParameterSet> pss;
   {
      edm::ParameterSet ps;
      std::string typeName("DummyService");
      ps.addParameter("@service_type", typeName);
      int value = 1;
      ps.addParameter("value", value);
      pss.push_back(ps);
   }
   edm::ServiceToken token(edm::ServiceRegistry::createSet(pss));
   
   bool succeededToPassServices = false;
   bool exceptionWasThrown = false;
   
   PassServices passRun(token, succeededToPassServices, exceptionWasThrown);
   boost::thread testPassing(passRun);
   testPassing.join();
   CPPUNIT_ASSERT(!exceptionWasThrown);
   CPPUNIT_ASSERT(succeededToPassServices);
}
#include <Utilities/Testing/interface/CppUnit_testdriver.icpp>
