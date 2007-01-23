/*----------------------------------------------------------------------

Test of the EventProcessor class.

$Id: eventprocessor_t.cppunit.cc,v 1.20 2006/07/06 19:11:44 wmtan Exp $

----------------------------------------------------------------------*/  
#include <exception>
#include <iostream>
#include <string>
#include "boost/regex.hpp"

//I need to open a 'back door' in order to test the functionality
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#define private public
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#undef private
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/test/stubs/TestBeginEndJobAnalyzer.h"

#include "FWCore/Utilities/interface/ProblemTracker.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DataFormats/Common/interface/ModuleDescription.h"


#include "cppunit/extensions/HelperMacros.h"

class testeventprocessor: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(testeventprocessor);
  CPPUNIT_TEST(parseTest);
  CPPUNIT_TEST(prepostTest);
  CPPUNIT_TEST(beginEndJobTest);
  CPPUNIT_TEST(cleanupJobTest);
  CPPUNIT_TEST(activityRegistryTest);
  CPPUNIT_TEST(moduleFailureTest);
  CPPUNIT_TEST(endpathTest);
  CPPUNIT_TEST_SUITE_END();
 public:
  void setUp(){m_handler = std::auto_ptr<edm::AssertHandler>(new edm::AssertHandler());}
  void tearDown(){ m_handler.reset();}
  void parseTest();
  void prepostTest();
  void beginEndJobTest();
  void cleanupJobTest();
  void activityRegistryTest();
  void moduleFailureTest();
  void endpathTest();
 private:
  std::auto_ptr<edm::AssertHandler> m_handler;
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
  catch (cms::Exception& e) {
      std::cerr << "cms exception caught: "
		<< e.explainSelf() << std::endl;
      CPPUNIT_ASSERT("Caught cms::Exception " == 0);
  }
  catch (seal::Error& e) {
      std::cerr << "Application exception caught: "
		<< e.explainSelf() << std::endl;
      CPPUNIT_ASSERT("Caught seal::Error " == 0);
  }
  catch (std::exception& e) {
      std::cerr << "Standard library exception caught: "
		<< e.what() << std::endl;
      CPPUNIT_ASSERT("Caught std::exception " == 0);
  }
  catch (...) {
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
  {
    edm::EventProcessor const& crProc(proc);
    typedef std::vector<edm::ModuleDescription const*> ModuleDescs;
    ModuleDescs  allModules = crProc.getAllModuleDescriptions();
    CPPUNIT_ASSERT(1 == allModules.size());
    std::cout << "\nModuleDescriptions in testeventprocessor::prepostTest()---\n";
    for (ModuleDescs::const_iterator i = allModules.begin(),
	    e = allModules.end() ; 
	  i != e ; 
	  ++i)
      {
	CPPUNIT_ASSERT(*i != 0);
	std::cout << **i << '\n';
      }
    std::cout << "--- end of ModuleDescriptions\n";

    CPPUNIT_ASSERT(5 == crProc.totalEvents());
    CPPUNIT_ASSERT(5 == crProc.totalEventsPassed());    
  }
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

void testeventprocessor::cleanupJobTest()
{
  std::string configuration("process p = {\n"
			    "source = EmptySource { untracked int32 maxEvents = 2 }\n"
			    "module m1 = TestBeginEndJobAnalyzer { }\n"
			    "path p1 = { m1 }\n"
			    "}\n");
  {
    TestBeginEndJobAnalyzer::destructorCalled = false;
    edm::EventProcessor proc(configuration);
      
    CPPUNIT_ASSERT(!TestBeginEndJobAnalyzer::destructorCalled);
    proc.beginJob();
    CPPUNIT_ASSERT(!TestBeginEndJobAnalyzer::destructorCalled);
    proc.endJob();
    CPPUNIT_ASSERT(!TestBeginEndJobAnalyzer::destructorCalled);
  }
  CPPUNIT_ASSERT(TestBeginEndJobAnalyzer::destructorCalled);
  {
    TestBeginEndJobAnalyzer::destructorCalled = false;
    edm::EventProcessor proc(configuration);

    CPPUNIT_ASSERT(!TestBeginEndJobAnalyzer::destructorCalled);
    proc.run(1);
    CPPUNIT_ASSERT(!TestBeginEndJobAnalyzer::destructorCalled);
    proc.run(1);
    CPPUNIT_ASSERT(!TestBeginEndJobAnalyzer::destructorCalled);

  }
  CPPUNIT_ASSERT(TestBeginEndJobAnalyzer::destructorCalled);
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

static
bool
findModuleName(const std::string& iMessage) {
  static const boost::regex expr("TestFailuresAnalyzer");
  return regex_search(iMessage,expr);
}

void 
testeventprocessor::moduleFailureTest()
{
  try {
    const std::string preC("process p = {\n"
			   "source = EmptySource { untracked int32 maxEvents = 2 }\n"
			   "module m1 = TestFailuresAnalyzer { int32 whichFailure =");
    const std::string postC(" }\n"
			    "path p1 = { m1 }\n"
			    "}\n");
    {
      const std::string configuration = preC +"0"+postC;
      bool threw = true;
      try {
	edm::EventProcessor proc(configuration);
	threw = false;
      } catch(const cms::Exception& iException){
	if(!findModuleName(iException.explainSelf())) {
	  std::cout <<iException.explainSelf()<<std::endl;
	  CPPUNIT_ASSERT(0 == "module name not in exception message");
	}
      }
      CPPUNIT_ASSERT(threw && 0 != "exception never thrown");
    }
    {
      const std::string configuration = preC +"1"+postC;
      bool threw = true;
      edm::EventProcessor proc(configuration);
         
      try {
	proc.beginJob();
	threw = false;
      } catch(const cms::Exception& iException){
	if(!findModuleName(iException.explainSelf())) {
	  std::cout <<iException.explainSelf()<<std::endl;
	  CPPUNIT_ASSERT(0 == "module name not in exception message");
	}
      }
      CPPUNIT_ASSERT(threw && 0 != "exception never thrown");
    }
      
    {
      const std::string configuration = preC +"2"+postC;
      bool threw = true;
      edm::EventProcessor proc(configuration);
         
      proc.beginJob();
      try {
	proc.run(1);
	threw = false;
      } catch(const cms::Exception& iException){
	if(!findModuleName(iException.explainSelf())) {
	  std::cout <<iException.explainSelf()<<std::endl;
	  CPPUNIT_ASSERT(0 == "module name not in exception message");
	}
      }
      CPPUNIT_ASSERT(threw && 0 != "exception never thrown");
      proc.endJob();
    }
    {
      const std::string configuration = preC +"3"+postC;
      bool threw = true;
      edm::EventProcessor proc(configuration);
         
      proc.beginJob();
      try {
	proc.endJob();
	threw = false;
      } catch(const cms::Exception& iException){
	if(!findModuleName(iException.explainSelf())) {
	  std::cout <<iException.explainSelf()<<std::endl;
	  CPPUNIT_ASSERT(0 == "module name not in exception message");
	}
      }
      CPPUNIT_ASSERT(threw && 0 != "exception never thrown");
    }
    ///
    {
      bool threw = true;
      try {
        const std::string configuration("process p = {\n"
                               "source = EmptySource { untracked int32 maxEvents = 2 }\n"
                                "path p1 = { m1 }\n"
                                "}\n");
        edm::EventProcessor proc(configuration);
      
	threw = false;
      } catch(const cms::Exception& iException){
        static const boost::regex expr("m1");
	if(!regex_search(iException.explainSelf(),expr)) {
	  std::cout <<iException.explainSelf()<<std::endl;
	  CPPUNIT_ASSERT(0 == "module name not in exception message");
	}
      }
      CPPUNIT_ASSERT(threw && 0 != "exception never thrown");
    }
    
  } catch(const cms::Exception& iException) {
    std::cout <<"Unexpected exception "<<iException.explainSelf()<<std::endl;
    throw;
  }
}

void
testeventprocessor::endpathTest()
{
  std::string configuration("process p = {\n"
			    "source = EmptySource { untracked int32 maxEvents = 5 }\n"
			    "module m1 = TestMod { int32 ivalue = -3 }\n"
			    "path p1 = { m1 }\n"
			    "}\n");
  
}
