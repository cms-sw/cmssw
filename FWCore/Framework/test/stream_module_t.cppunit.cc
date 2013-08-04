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
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/stream/EDProducerAdapter.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"


#include "FWCore/Utilities/interface/Exception.h"

#include <cppunit/extensions/HelperMacros.h>

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
    kBeginJob,
    kBeginStream,
    kGlobalBeginRun,
    kGlobalBeginRunProduce,
    kStreamBeginRun,
    kGlobalBeginLuminosityBlock,
    kStreamBeginLuminosityBlock,
    kEvent,
    kStreamEndLuminosityBlock,
    kGlobalEndLuminosityBlock,
    kStreamEndRun,
    kGlobalEndRun,
    kEndStream,
    kEndJob
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
  edm::WorkerParams m_params;
  
  template<typename T, typename U>
  void testTransitions(std::unique_ptr<U>&& iMod, Expectations const& iExpect);
  
  template<typename T>
  static void runTest(Expectations const& iExpect);
  
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
  /*
  class RunProd : public edm::global::EDProducer<edm::RunCache<int>> {
  public:
    static unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    std::shared_ptr<int> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }

    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
    }
  };


  class LumiProd : public edm::global::EDProducer<edm::LuminosityBlockCache<int>> {
  public:
    static unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    std::shared_ptr<int> globalBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
    }
  };
  
  class RunSummaryProd : public edm::global::EDProducer<edm::RunSummaryCache<int>> {
  public:
    static unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    std::shared_ptr<int> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void streamEndRunSummary(edm::StreamID, edm::Run const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
  };

  class LumiSummaryProd : public edm::global::EDProducer<edm::LuminosityBlockSummaryCache<int>> {
  public:
    static unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    std::shared_ptr<int> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void streamEndLuminosityBlockSummary(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
  };

  class BeginRunProd : public edm::global::EDProducer<edm::BeginRunProducer> {
  public:
    static unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }

    void globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const override {
      ++m_count;
    }
  };

  class BeginLumiProd : public edm::global::EDProducer<edm::BeginLuminosityBlockProducer> {
  public:
    static unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override {
      ++m_count;
    }
  };

  class EndRunProd : public edm::global::EDProducer<edm::EndRunProducer> {
  public:
    static unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    void globalEndRunProduce(edm::Run&, edm::EventSetup const&) const override {
      ++m_count;
    }
  };
  
  class EndLumiProd : public edm::global::EDProducer<edm::EndLuminosityBlockProducer> {
  public:
    static unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override {
      ++m_count;
    }
  };


  class EndRunSummaryProd : public edm::global::EDProducer<edm::EndRunProducer, edm::RunSummaryCache<int>> {
  public:
    static unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    std::shared_ptr<int> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void streamEndRunSummary(edm::StreamID, edm::Run const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }

    void globalEndRunProduce(edm::Run&, edm::EventSetup const&, int const*) const override {
      ++m_count;
    }
  };
  
  class EndLumiSummaryProd : public edm::global::EDProducer<edm::EndLuminosityBlockProducer, edm::LuminosityBlockSummaryCache<int>> {
  public:
    static unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }

    std::shared_ptr<int> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
      return std::shared_ptr<int>{};
    }
    
    void streamEndLuminosityBlockSummary(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }
    
    void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&, edm::EventSetup const&, int*) const override {
      ++m_count;
    }

    void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&, int const*) const override {
      ++m_count;
    }
  };
   */
};
unsigned int testStreamModule::BasicProd::m_count = 0;
unsigned int testStreamModule::GlobalProd::m_count = 0;

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testStreamModule);

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

  m_ep.reset(new edm::EventPrincipal(m_prodReg,
                                     m_idHelper,
                                     m_procConfig,nullptr));
  m_ep->fillEventPrincipal(eventAux);
  m_ep->setLuminosityBlockPrincipal(m_lbp);

  //For each transition, bind a lambda which will call the proper method of the Worker
  m_transToFunc[Trans::kBeginStream] = [this](edm::Worker* iBase) {
    iBase->beginStream(edm::StreamID::invalidStreamID()); };
  
  m_transToFunc[Trans::kGlobalBeginRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionGlobalBegin> Traits;
    iBase->doWork<Traits>(*m_rp,*m_es,m_context,m_timer, edm::StreamID::invalidStreamID()); };
  m_transToFunc[Trans::kStreamBeginRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionStreamBegin> Traits;
    iBase->doWork<Traits>(*m_rp,*m_es,m_context,m_timer, edm::StreamID::invalidStreamID()); };
  
  m_transToFunc[Trans::kGlobalBeginLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalBegin> Traits;
    iBase->doWork<Traits>(*m_lbp,*m_es,m_context,m_timer, edm::StreamID::invalidStreamID()); };
  m_transToFunc[Trans::kStreamBeginLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionStreamBegin> Traits;
    iBase->doWork<Traits>(*m_lbp,*m_es,m_context,m_timer, edm::StreamID::invalidStreamID()); };
  
  m_transToFunc[Trans::kEvent] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::EventPrincipal, edm::BranchActionStreamBegin> Traits;
    iBase->doWork<Traits>(*m_ep,*m_es,m_context,m_timer, edm::StreamID::invalidStreamID()); };

  m_transToFunc[Trans::kStreamEndLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionStreamEnd> Traits;
    iBase->doWork<Traits>(*m_lbp,*m_es,m_context,m_timer, edm::StreamID::invalidStreamID()); };
  m_transToFunc[Trans::kGlobalEndLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalEnd> Traits;
    iBase->doWork<Traits>(*m_lbp,*m_es,m_context,m_timer, edm::StreamID::invalidStreamID()); };

  m_transToFunc[Trans::kStreamEndRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionStreamEnd> Traits;
    iBase->doWork<Traits>(*m_rp,*m_es,m_context,m_timer, edm::StreamID::invalidStreamID()); };
  m_transToFunc[Trans::kGlobalEndRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionGlobalEnd> Traits;
    iBase->doWork<Traits>(*m_rp,*m_es,m_context,m_timer, edm::StreamID::invalidStreamID()); };

  m_transToFunc[Trans::kEndStream] = [this](edm::Worker* iBase) {
    iBase->endStream(edm::StreamID::invalidStreamID()); };

}


namespace {
  template<typename T>
  void * createModule() {
    edm::ParameterSet pset;
    edm::stream::EDProducerAdapter<T> t(pset);
    return nullptr;
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
testStreamModule::testTransitions(std::unique_ptr<U>&& iMod, Expectations const& iExpect) {
  U* pMod = iMod.get();
  edm::WorkerT<edm::stream::EDProducerAdapterBase> w{std::move(iMod),m_desc,m_params};
  for(auto& keyVal: m_transToFunc) {
    testTransition(pMod,&w,keyVal.first,iExpect,keyVal.second);
  }
}
template<typename T>
void
testStreamModule::runTest(Expectations const& iExpect) {
  auto mod = createModule<T>();
  CPPUNIT_ASSERT(0 == T::m_count);
  testTransitions<T>(std::move(mod),iExpect);
}


void testStreamModule::basicTest()
{
  auto mod = createModule<BasicProd>();
  CPPUNIT_ASSERT(mod == nullptr);
  /*
  std::unique_ptr<BasicProd> testProd{ new BasicProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(std::move(testProd), {Trans::kEvent});
   */
}

void testStreamModule::globalTest()
{
  auto mod = createModule<GlobalProd>();
  CPPUNIT_ASSERT(mod == nullptr);
  /*
  std::unique_ptr<StreamProd> testProd{ new StreamProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(std::move(testProd), {Trans::kBeginStream, Trans::kStreamBeginRun, Trans::kStreamBeginLuminosityBlock, Trans::kEvent,
    Trans::kStreamEndLuminosityBlock,Trans::kStreamEndRun,Trans::kEndStream});
   */
}

void testStreamModule::runTest()
{
   /*
  std::unique_ptr<RunProd> testProd{ new RunProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(std::move(testProd), {Trans::kGlobalBeginRun, Trans::kEvent, Trans::kGlobalEndRun});
    */
}

void testStreamModule::runSummaryTest()
{
   /*
  std::unique_ptr<RunSummaryProd> testProd{ new RunSummaryProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(std::move(testProd), {Trans::kGlobalBeginRun, Trans::kEvent, Trans::kStreamEndRun, Trans::kGlobalEndRun});
    */
}

void testStreamModule::lumiTest()
{
   /*
  std::unique_ptr<LumiProd> testProd{ new LumiProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(std::move(testProd), {Trans::kGlobalBeginLuminosityBlock, Trans::kEvent, Trans::kGlobalEndLuminosityBlock});
    */
}

void testStreamModule::lumiSummaryTest()
{
   /*
  std::unique_ptr<LumiSummaryProd> testProd{ new LumiSummaryProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(std::move(testProd), {Trans::kGlobalBeginLuminosityBlock, Trans::kEvent, Trans::kStreamEndLuminosityBlock, Trans::kGlobalEndLuminosityBlock});
    */
}

void testStreamModule::beginRunProdTest()
{
   /*
  std::unique_ptr<BeginRunProd> testProd{ new BeginRunProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(std::move(testProd), {Trans::kGlobalBeginRun, Trans::kEvent});
    */
}

void testStreamModule::beginLumiProdTest()
{
   /*
  std::unique_ptr<BeginLumiProd> testProd{ new BeginLumiProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(std::move(testProd), {Trans::kGlobalBeginLuminosityBlock, Trans::kEvent});
    */
}

void testStreamModule::endRunProdTest()
{
   /*
  std::unique_ptr<EndRunProd> testProd{ new EndRunProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(std::move(testProd), {Trans::kGlobalEndRun, Trans::kEvent});
    */
}

void testStreamModule::endLumiProdTest()
{
   /*
  std::unique_ptr<EndLumiProd> testProd{ new EndLumiProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(std::move(testProd), {Trans::kGlobalEndLuminosityBlock, Trans::kEvent});
    */
}

void testStreamModule::endRunSummaryProdTest()
{
   /*
  std::unique_ptr<EndRunSummaryProd> testProd{ new EndRunSummaryProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(std::move(testProd), {Trans::kGlobalEndRun, Trans::kEvent, Trans::kGlobalBeginRun, Trans::kStreamEndRun, Trans::kGlobalEndRun});
    */
}

void testStreamModule::endLumiSummaryProdTest()
{
   /*
  std::unique_ptr<EndLumiSummaryProd> testProd{ new EndLumiSummaryProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(std::move(testProd), {Trans::kGlobalEndLuminosityBlock, Trans::kEvent, Trans::kGlobalBeginLuminosityBlock, Trans::kStreamEndLuminosityBlock, Trans::kGlobalEndLuminosityBlock}); 
    */
}

