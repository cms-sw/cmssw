/*
 *  stream_module_test.cppunit.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 2/8/2013.
 */
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDProducerAdaptor.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"


#include "FWCore/Utilities/interface/Exception.h"

#include "cppunit/extensions/HelperMacros.h"

class testStreamModule: public CppUnit::TestFixture 
{
  CPPUNIT_TEST_SUITE(testStreamModule);
  
  CPPUNIT_TEST(basicTest);
  CPPUNIT_TEST(globalTest);
  CPPUNIT_TEST(runTest);
  CPPUNIT_TEST(runSummaryTest);
  CPPUNIT_TEST(lumiTest);
  CPPUNIT_TEST(lumiSummaryTest);
  CPPUNIT_TEST(beginRunProdTest);
  CPPUNIT_TEST(beginLumiProdTest);
  CPPUNIT_TEST(endRunProdTest);
  CPPUNIT_TEST(endLumiProdTest);
  CPPUNIT_TEST(endRunSummaryProdTest);
  CPPUNIT_TEST(endLumiSummaryProdTest);
  
  CPPUNIT_TEST_SUITE_END();
public:
  testStreamModule();
  
  void setUp(){}
  void tearDown(){}

  void basicTest();
  void globalTest();
  void runTest();
  void runSummaryTest();
  void lumiTest();
  void lumiSummaryTest();
  void beginRunProdTest();
  void beginLumiProdTest();
  void endRunProdTest();
  void endLumiProdTest();
  void endRunSummaryProdTest();
  void endLumiSummaryProdTest();

private:

  enum class Trans {
    kBeginJob, //0
    kBeginStream, //1
    kGlobalBeginRun, //2
    kStreamBeginRun, //3
    kGlobalBeginLuminosityBlock, //4
    kStreamBeginLuminosityBlock, //5
    kEvent, //6
    kStreamEndLuminosityBlock, //7
    kGlobalEndLuminosityBlock, //8
    kStreamEndRun, //9
    kGlobalEndRun, //10
    kEndStream, //11
    kEndJob //12
  };
  
  std::map<Trans,std::function<void(edm::Worker*)>> m_transToFunc;
  typedef std::vector<Trans> Expectations;
  
  edm::ProcessConfiguration m_procConfig;
  boost::shared_ptr<edm::ProductRegistry> m_prodReg;
  boost::shared_ptr<edm::BranchIDListHelper> m_idHelper;
  std::unique_ptr<edm::EventPrincipal> m_ep;
  edm::HistoryAppender historyAppender_;
  boost::shared_ptr<edm::LuminosityBlockPrincipal> m_lbp;
  boost::shared_ptr<edm::RunPrincipal> m_rp;
  edm::EventSetup* m_es = nullptr;
  edm::CurrentProcessingContext* m_context = nullptr;
  edm::ModuleDescription m_desc = {"Dummy","dummy"};
  edm::CPUTimer* m_timer = nullptr;
  
  template<typename T, typename U>
  void testTransitions(U* iMod, Expectations const& iExpect);
  
  template<typename T>
  void runTest(Expectations const& iExpect);
  
  class BasicProd : public edm::stream::EDProducer<> {
  public:
    static unsigned int m_count;
    
    BasicProd(edm::ParameterSet const&) {}
    
    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
    }
  };
  
  class GlobalProd : public edm::stream::EDProducer<edm::GlobalCache<int>> {
  public:
    static unsigned int m_count;
    
    static std::unique_ptr<int> initializeGlobalCache(edm::ParameterSet const&) {
      return std::unique_ptr<int>{new int{1}};
    }
    GlobalProd(edm::ParameterSet const&, const int* iGlobal) { CPPUNIT_ASSERT(*iGlobal == 1); }
    
    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
    }
    
    static void globalEndJob(int* iGlobal) {
      CPPUNIT_ASSERT(1==*iGlobal);
      ++m_count;
    }
    
  };
  class RunProd : public edm::stream::EDProducer<edm::RunCache<int>> {
  public:
    static unsigned int m_count;
    RunProd(edm::ParameterSet const&) {}
    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
    }
    
    static std::shared_ptr<int> globalBeginRun(edm::Run const&, edm::EventSetup const&, GlobalCache const*) {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    static void globalEndRun(edm::Run const&, edm::EventSetup const&, RunProd::RunContext const*) {
      ++m_count;
    }
  };


  class LumiProd : public edm::stream::EDProducer<edm::LuminosityBlockCache<int>> {
  public:
    static unsigned int m_count;
    LumiProd(edm::ParameterSet const&) {}
    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
    }
    
    static std::shared_ptr<int> globalBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    static void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&, LuminosityBlockContext const*) {
      ++m_count;
    }
  };
  
  class RunSummaryProd : public edm::stream::EDProducer<edm::RunSummaryCache<int>> {
  public:
    static unsigned int m_count;
    RunSummaryProd(edm::ParameterSet const&) {}
    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
    }
    
    static std::shared_ptr<int> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&, GlobalCache const*) {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void endRunSummary(edm::Run const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    static void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, RunContext const*, int*){
      ++m_count;
    }
  };

  class LumiSummaryProd : public edm::stream::EDProducer<edm::LuminosityBlockSummaryCache<int>> {
  public:
    static unsigned int m_count;
    LumiSummaryProd(edm::ParameterSet const&) {}
    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
    }
    
    static std::shared_ptr<int> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, LuminosityBlockContext const*){
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void endLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, LuminosityBlockContext const*, int*){
      ++m_count;
    }
  };

  class BeginRunProd : public edm::stream::EDProducer<edm::BeginRunProducer> {
  public:
    static unsigned int m_count;
    BeginRunProd(edm::ParameterSet const&) {}

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
    }

    static void globalBeginRunProduce(edm::Run&, edm::EventSetup const&, RunContext const*) {
      ++m_count;
    }
  };

  class BeginLumiProd : public edm::stream::EDProducer<edm::BeginLuminosityBlockProducer> {
  public:
    static unsigned int m_count;
    BeginLumiProd(edm::ParameterSet const&) {}

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
    }
    
    static void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&, LuminosityBlockContext const*) {
      ++m_count;
    }
  };

  class EndRunProd : public edm::stream::EDProducer<edm::EndRunProducer> {
  public:
    static unsigned int m_count;
    EndRunProd(edm::ParameterSet const&) {}

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
    }
    
    static void globalEndRunProduce(edm::Run&, edm::EventSetup const&, RunContext const*) {
      ++m_count;
    }
  };
  
  class EndLumiProd : public edm::stream::EDProducer<edm::EndLuminosityBlockProducer> {
  public:
    static unsigned int m_count;
    EndLumiProd(edm::ParameterSet const&) {}

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
    }
    
    static void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&, LuminosityBlockContext const*) {
      ++m_count;
    }
  };


  class EndRunSummaryProd : public edm::stream::EDProducer<edm::EndRunProducer, edm::RunSummaryCache<int>> {
  public:
    static unsigned int m_count;
    EndRunSummaryProd(edm::ParameterSet const&) {}

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
    }
    
    static std::shared_ptr<int> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&, RunContext const*) {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void endRunSummary(edm::Run const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    static void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, RunContext const*, int*) {
      ++m_count;
    }

    static void globalEndRunProduce(edm::Run&, edm::EventSetup const&, RunContext const*, int const*) {
      ++m_count;
    }
  };
  
  class EndLumiSummaryProd : public edm::stream::EDProducer<edm::EndLuminosityBlockProducer, edm::LuminosityBlockSummaryCache<int>> {
  public:
    static unsigned int m_count;
    EndLumiSummaryProd(edm::ParameterSet const&) {}

    void produce(edm::Event&, edm::EventSetup const&) override {
      ++m_count;
    }

    static std::shared_ptr<int> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, LuminosityBlockContext const*) {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void endLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, LuminosityBlockContext const*, int*) {
      ++m_count;
    }

    static void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&, LuminosityBlockContext const*, int const*) {
      ++m_count;
    }
  };
   
};
unsigned int testStreamModule::BasicProd::m_count = 0;
unsigned int testStreamModule::GlobalProd::m_count = 0;
unsigned int testStreamModule::RunProd::m_count = 0;
unsigned int testStreamModule::LumiProd::m_count = 0;
unsigned int testStreamModule::RunSummaryProd::m_count = 0;
unsigned int testStreamModule::LumiSummaryProd::m_count = 0;
unsigned int testStreamModule::BeginRunProd::m_count = 0;
unsigned int testStreamModule::EndRunProd::m_count = 0;
unsigned int testStreamModule::BeginLumiProd::m_count = 0;
unsigned int testStreamModule::EndLumiProd::m_count = 0;
unsigned int testStreamModule::EndRunSummaryProd::m_count = 0;
unsigned int testStreamModule::EndLumiSummaryProd::m_count = 0;
///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testStreamModule);

namespace {
  struct ShadowStreamID {
    constexpr ShadowStreamID():value(0){}
    unsigned int value;
  };
  
  union IDUnion {
    IDUnion(): m_shadow() {}
    ShadowStreamID m_shadow;
    edm::StreamID m_id;
  };
}
static edm::StreamID makeID() {
  IDUnion u;
  assert(u.m_id.value() == 0);
  return u.m_id;
}
static const edm::StreamID s_streamID0 = makeID();


testStreamModule::testStreamModule():
m_prodReg(new edm::ProductRegistry{}),
m_idHelper(new edm::BranchIDListHelper{}),
m_ep()
{
  //Setup the principals
  m_prodReg->setFrozen();
  m_idHelper->updateRegistries(*m_prodReg);
  edm::EventID eventID;
  
  std::string uuid = edm::createGlobalIdentifier();
  edm::Timestamp now(1234567UL);
  boost::shared_ptr<edm::RunAuxiliary> runAux(new edm::RunAuxiliary(eventID.run(), now, now));
  m_rp.reset(new edm::RunPrincipal(runAux, m_prodReg, m_procConfig, &historyAppender_,0));
  boost::shared_ptr<edm::LuminosityBlockAuxiliary> lumiAux(new edm::LuminosityBlockAuxiliary(m_rp->run(), 1, now, now));
  m_lbp.reset(new edm::LuminosityBlockPrincipal(lumiAux, m_prodReg, m_procConfig, &historyAppender_,0));
  m_lbp->setRunPrincipal(m_rp);
  edm::EventAuxiliary eventAux(eventID, uuid, now, true);

  //Only an EventProcessor or SubProcess is allowed to create a StreamID but I need one
  ShadowStreamID shadowID;
  shadowID.value = 0;
  edm::StreamID* pID = reinterpret_cast<edm::StreamID*>(&shadowID);
  assert(pID->value() == 0);
  
  m_ep.reset(new edm::EventPrincipal(m_prodReg,
                                     m_idHelper,
                                     m_procConfig,nullptr,*pID));
  m_ep->fillEventPrincipal(eventAux);
  m_ep->setLuminosityBlockPrincipal(m_lbp);


  //For each transition, bind a lambda which will call the proper method of the Worker
  m_transToFunc[Trans::kBeginStream] = [this](edm::Worker* iBase) {
    edm::StreamContext streamContext(s_streamID0, nullptr);
    iBase->beginStream(s_streamID0, streamContext); };
  
  m_transToFunc[Trans::kGlobalBeginRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionGlobalBegin> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_rp,*m_es,m_context,m_timer, s_streamID0, parentContext, nullptr); };
  m_transToFunc[Trans::kStreamBeginRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_rp,*m_es,m_context,m_timer, s_streamID0, parentContext, nullptr); };
  
  m_transToFunc[Trans::kGlobalBeginLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalBegin> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_lbp,*m_es,m_context,m_timer, s_streamID0, parentContext, nullptr); };
  m_transToFunc[Trans::kStreamBeginLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_lbp,*m_es,m_context,m_timer, s_streamID0, parentContext, nullptr); };
  
  m_transToFunc[Trans::kEvent] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::EventPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_ep,*m_es,m_context,m_timer, s_streamID0, parentContext, nullptr); };

  m_transToFunc[Trans::kStreamEndLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionStreamEnd> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_lbp,*m_es,m_context,m_timer, s_streamID0, parentContext, nullptr); };
  m_transToFunc[Trans::kGlobalEndLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalEnd> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_lbp,*m_es,m_context,m_timer, s_streamID0, parentContext, nullptr); };

  m_transToFunc[Trans::kStreamEndRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionStreamEnd> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_rp,*m_es,m_context,m_timer, s_streamID0, parentContext, nullptr); };
  m_transToFunc[Trans::kGlobalEndRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionGlobalEnd> Traits;
    edm::ParentContext parentContext;
    iBase->doWork<Traits>(*m_rp,*m_es,m_context,m_timer, s_streamID0, parentContext, nullptr); };

  m_transToFunc[Trans::kEndStream] = [this](edm::Worker* iBase) {
    edm::StreamContext streamContext(s_streamID0, nullptr);
    iBase->endStream(s_streamID0, streamContext); };

}


namespace {
  template<typename T>
  std::unique_ptr<edm::stream::EDProducerAdaptorBase> createModule() {
    edm::ParameterSet pset;
    return std::unique_ptr<edm::stream::EDProducerAdaptorBase>(new edm::stream::EDProducerAdaptor<T>(pset));
  }
  template<typename T>
  void
  testTransition(edm::Worker* iWorker, testStreamModule::Trans iTrans, testStreamModule::Expectations const& iExpect, std::function<void(edm::Worker*)> iFunc) {
    assert(0==T::m_count);
    iFunc(iWorker);
    auto count = std::count(iExpect.begin(),iExpect.end(),iTrans);
    if(count != T::m_count) {
      std::cout<<"For trans " <<static_cast<std::underlying_type<testStreamModule::Trans>::type >(iTrans)<< " expected "<<count<<" and got "<<T::m_count<<std::endl;
    }
    CPPUNIT_ASSERT(T::m_count == count);
    T::m_count = 0;
    iWorker->reset();
  }
}

template<typename T, typename U>
void
testStreamModule::testTransitions(U* iMod, Expectations const& iExpect) {
  edm::WorkerT<edm::stream::EDProducerAdaptorBase> w{iMod,m_desc,nullptr};
  for(auto& keyVal: m_transToFunc) {
    testTransition<T>(&w,keyVal.first,iExpect,keyVal.second);
  }
}
template<typename T>
void
testStreamModule::runTest(Expectations const& iExpect) {
  auto mod = createModule<T>();
  CPPUNIT_ASSERT(0 == T::m_count);
  testTransitions<T>(mod.get(),iExpect);
}


void testStreamModule::basicTest()
{
  runTest<BasicProd>({Trans::kEvent} );
}

void testStreamModule::globalTest()
{
  runTest<GlobalProd>({Trans::kBeginJob, Trans::kEvent, Trans::kEndJob} );
}

void testStreamModule::runTest()
{
  runTest<RunProd>({Trans::kGlobalBeginRun, Trans::kEvent, Trans::kGlobalEndRun} );
}

void testStreamModule::runSummaryTest()
{
  runTest<RunSummaryProd>({Trans::kGlobalBeginRun, Trans::kEvent, Trans::kStreamEndRun, Trans::kGlobalEndRun} );
}

void testStreamModule::lumiTest()
{
  runTest<LumiProd>({Trans::kGlobalBeginLuminosityBlock, Trans::kEvent, Trans::kGlobalEndLuminosityBlock} );
}

void testStreamModule::lumiSummaryTest()
{
  runTest<LumiSummaryProd>({Trans::kGlobalBeginLuminosityBlock, Trans::kEvent, Trans::kStreamEndLuminosityBlock, Trans::kGlobalEndLuminosityBlock} );
}

void testStreamModule::beginRunProdTest()
{
  runTest<BeginRunProd>({Trans::kGlobalBeginRun, Trans::kEvent} );
}

void testStreamModule::beginLumiProdTest()
{
  runTest<BeginLumiProd>({Trans::kGlobalBeginLuminosityBlock, Trans::kEvent} );
}

void testStreamModule::endRunProdTest()
{
  runTest<EndRunProd>({Trans::kGlobalEndRun, Trans::kEvent} );
}

void testStreamModule::endLumiProdTest()
{
  runTest<EndLumiProd>({Trans::kGlobalEndLuminosityBlock, Trans::kEvent} );
}

void testStreamModule::endRunSummaryProdTest()
{
  runTest<EndRunSummaryProd>({Trans::kGlobalEndRun, Trans::kEvent, Trans::kGlobalBeginRun, Trans::kStreamEndRun, Trans::kGlobalEndRun} );
}

void testStreamModule::endLumiSummaryProdTest()
{
  runTest<EndLumiSummaryProd>({Trans::kGlobalEndLuminosityBlock, Trans::kEvent, Trans::kGlobalBeginLuminosityBlock, Trans::kStreamEndLuminosityBlock, Trans::kGlobalEndLuminosityBlock} );
}

