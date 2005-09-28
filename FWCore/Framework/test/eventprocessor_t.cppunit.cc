/*----------------------------------------------------------------------

Test of the EventProcessor class.

$Id: eventprocessor_t.cppunit.cc,v 1.9 2005/09/09 14:56:50 chrjones Exp $

----------------------------------------------------------------------*/  
#include <exception>
#include <iostream>
#include <string>

//I need to open a 'back door' in order to test the functionality
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#define private public
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#undef private
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/test/stubs/TestBeginEndJobAnalyzer.h"

#include <cppunit/extensions/HelperMacros.h>

class testeventprocessor: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testeventprocessor);
CPPUNIT_TEST(parseTest);
CPPUNIT_TEST(prepostTest);
CPPUNIT_TEST(beginEndJobTest);
CPPUNIT_TEST(activityRegistryTest);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void parseTest();
  void prepostTest();
  void beginEndJobTest();
  void activityRegistryTest();
private:
void work()
{
  std::string configuration("process p = {\n"
			    "source = EmptySource { untracked int32 maxEvents = 5 }\n"
			    "module m1 = TestMod { int32 ivalue = 10 }\n"
			    "module m2 = TestMod { int32 ivalue = -3 }\n"
                            "path p1 = { m1,m2 }\n"
			    "}\n");
  edm::EventProcessor proc(configuration);
  proc.beginJob();
  proc.run(0);
  proc.endJob();
}
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testeventprocessor);

void testeventprocessor::parseTest()
{
  int rc = -1;                // we should never return this value!
  try { work(); rc = 0;}
  catch (seal::Error& e)
    {
      std::cerr << "Application exception caught: "
		<< e.explainSelf() << std::endl;
      CPPUNIT_ASSERT("Caught seal::Error " == 0);
    }
  catch (std::exception& e)
    {
      std::cerr << "Standard library exception caught: "
		<< e.what() << std::endl;
     CPPUNIT_ASSERT("Caught std::exception " == 0);
    }
  catch (...)
    {
     CPPUNIT_ASSERT("Caught unknown exception " == 0);
    }
}

static int g_pre = 0;
static int g_post = 0;
static
void doPre(const edm::EventID&, const edm::Timestamp&) 
{
   ++g_pre;
}
static
void doPost(const edm::Event&, const edm::EventSetup&) 
{
   CPPUNIT_ASSERT(g_pre == ++g_post);
}

void testeventprocessor::prepostTest()
{
   std::string configuration("process p = {\n"
                             "source = EmptySource { untracked int32 maxEvents = 5 }\n"
                             "module m1 = TestMod { int32 ivalue = -3 }\n"
                             "path p1 = { m1 }\n"
                             "}\n");
   edm::EventProcessor proc(configuration);
   
   proc.preProcessEventSignal.connect(&doPre);
   proc.postProcessEventSignal.connect(&doPost);
   proc.beginJob();
   proc.run(0);
   proc.endJob();
   CPPUNIT_ASSERT(5 == g_pre);
   CPPUNIT_ASSERT(5 == g_post);
}

void testeventprocessor::beginEndJobTest()
{
   std::string configuration("process p = {\n"
                             "source = EmptySource { untracked int32 maxEvents = 2 }\n"
                             "module m1 = TestBeginEndJobAnalyzer { }\n"
                             "path p1 = { m1 }\n"
                             "}\n");
   {
      edm::EventProcessor proc(configuration);
      
      CPPUNIT_ASSERT(!TestBeginEndJobAnalyzer::beginJobCalled);
      proc.beginJob();
      CPPUNIT_ASSERT(TestBeginEndJobAnalyzer::beginJobCalled);
      CPPUNIT_ASSERT(!TestBeginEndJobAnalyzer::endJobCalled);
      proc.endJob();
      CPPUNIT_ASSERT(TestBeginEndJobAnalyzer::endJobCalled);
   }
   {
      TestBeginEndJobAnalyzer::beginJobCalled = false;
      TestBeginEndJobAnalyzer::endJobCalled = false;

      edm::EventProcessor proc(configuration);
      //run should call beginJob if it hasn't happened already
      CPPUNIT_ASSERT(!TestBeginEndJobAnalyzer::beginJobCalled);
      proc.run(1);
      CPPUNIT_ASSERT(TestBeginEndJobAnalyzer::beginJobCalled);

      //second call to run should NOT call beginJob
      TestBeginEndJobAnalyzer::beginJobCalled = false;
      proc.run(1);
      CPPUNIT_ASSERT(!TestBeginEndJobAnalyzer::beginJobCalled);

   }
   //In this case, endJob should not have been called since was not done explicitly
   CPPUNIT_ASSERT(!TestBeginEndJobAnalyzer::endJobCalled);
   
}

namespace {
   struct Listener{
      Listener(edm::ActivityRegistry& iAR) :
      postBeginJob_(false),
      postEndJob_(false),
      preEventProcessing_(false),
      postEventProcessing_(false),
      preModule_(false),
      postModule_(false){
         iAR.watchPostBeginJob(this,&Listener::postBeginJob);
         iAR.watchPostEndJob(this,&Listener::postEndJob);

         iAR.watchPreProcessEvent(this,&Listener::preEventProcessing);
         iAR.watchPostProcessEvent(this,&Listener::postEventProcessing);

         iAR.watchPreModule(this, &Listener::preModule);
         iAR.watchPostModule(this, &Listener::postModule);
      }
         
      void postBeginJob() {postBeginJob_=true;}
      void postEndJob() {postEndJob_=true;}
      
      void preEventProcessing(const edm::EventID&, const edm::Timestamp&){
         preEventProcessing_=true;}
      void postEventProcessing(const edm::Event&, const edm::EventSetup&){
         postEventProcessing_=true;}
      
      void preModule(const edm::ModuleDescription&){
         preModule_=true;
      }
      void postModule(const edm::ModuleDescription&){
         postModule_=true;
      }
      
      bool allCalled() const {
         return postBeginJob_&&postEndJob_
         &&preEventProcessing_&&postEventProcessing_
         &&preModule_&&postModule_;
      }
      
      bool postBeginJob_;
      bool postEndJob_;
      bool preEventProcessing_;
      bool postEventProcessing_;
      bool preModule_;
      bool postModule_;      
   };
}

void 
testeventprocessor::activityRegistryTest()
{
   std::string configuration("process p = {\n"
                             "source = EmptySource { untracked int32 maxEvents = 5 }\n"
                             "module m1 = TestMod { int32 ivalue = -3 }\n"
                             "path p1 = { m1 }\n"
                             "}\n");
   
   //We don't want any services, we just want an ActivityRegistry to be created
   // We then use this ActivityRegistry to 'spy on' the signals being produced
   // inside the EventProcessor
   std::vector<edm::ParameterSet> serviceConfigs;
   edm::ServiceToken token = edm::ServiceRegistry::createSet(serviceConfigs);

   edm::ActivityRegistry ar;
   token.connect(ar);
   Listener listener(ar);
   
   edm::EventProcessor proc(configuration,token, edm::serviceregistry::kOverlapIsError);
   
   proc.beginJob();
   proc.run(0);
   proc.endJob();
   
   CPPUNIT_ASSERT(listener.postBeginJob_);
   CPPUNIT_ASSERT(listener.postEndJob_);
   CPPUNIT_ASSERT(listener.preEventProcessing_);
   CPPUNIT_ASSERT(listener.postEventProcessing_);
   CPPUNIT_ASSERT(listener.preModule_);
   CPPUNIT_ASSERT(listener.postModule_);      
   
   CPPUNIT_ASSERT(listener.allCalled());
}
