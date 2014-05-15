/*
 *  proxyfactoryproducer_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 4/8/05.
 *  Changed by Viji Sundararajan on 28-Jun-05
 */
#include <iostream>
#include <atomic>
#include <vector>
#include <map>
#include <functional>
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/src/ModuleHolder.h"
#include "FWCore/Framework/src/PreallocationConfiguration.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"


#include "FWCore/Utilities/interface/Exception.h"

#include "cppunit/extensions/HelperMacros.h"

class testGlobalModule: public CppUnit::TestFixture 
{
  CPPUNIT_TEST_SUITE(testGlobalModule);
  
  CPPUNIT_TEST(basicTest);
  CPPUNIT_TEST(streamTest);
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
  testGlobalModule();
  
  void setUp(){}
  void tearDown(){}

  void basicTest();
  void streamTest();
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
  typedef std::vector<Trans> Expectations;

private:

  
  std::map<Trans,std::function<void(edm::Worker*)>> m_transToFunc;
  
  edm::ProcessConfiguration m_procConfig;
  boost::shared_ptr<edm::ProductRegistry> m_prodReg;
  boost::shared_ptr<edm::BranchIDListHelper> m_idHelper;
  std::unique_ptr<edm::EventPrincipal> m_ep;
  edm::HistoryAppender historyAppender_;
  boost::shared_ptr<edm::LuminosityBlockPrincipal> m_lbp;
  boost::shared_ptr<edm::RunPrincipal> m_rp;
  boost::shared_ptr<edm::ActivityRegistry> m_actReg;
  edm::EventSetup* m_es = nullptr;
  edm::ModuleDescription m_desc = {"Dummy","dummy"};
  edm::CPUTimer* m_timer = nullptr;
  
  template<typename T>
  void testTransitions(T* iMod, Expectations const& iExpect);
  
  class BasicProd : public edm::global::EDProducer<> {
  public:
    mutable unsigned int m_count = 0; //[[cms-thread-safe]]
    
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
  };
  class StreamProd : public edm::global::EDProducer<edm::StreamCache<int>> {
  public:
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    std::unique_ptr<int> beginStream(edm::StreamID) const override {
      ++m_count;
      return std::unique_ptr<int>{};
    }
    
    virtual void streamBeginRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const  override{
      ++m_count;
    }
    virtual void streamBeginLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    virtual void streamEndLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    virtual void streamEndRun(edm::StreamID, edm::Run const&, edm::EventSetup const&) const override {
      ++m_count;
    }
    void endStream(edm::StreamID) const override {
      ++m_count;
    }
  };
  
  class RunProd : public edm::global::EDProducer<edm::RunCache<int>> {
  public:
    mutable unsigned int m_count = 0;
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
    mutable unsigned int m_count = 0;
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
    mutable unsigned int m_count = 0;
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
    mutable unsigned int m_count = 0;
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
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }

    void globalBeginRunProduce(edm::Run&, edm::EventSetup const&) const override {
      ++m_count;
    }
  };

  class BeginLumiProd : public edm::global::EDProducer<edm::BeginLuminosityBlockProducer> {
  public:
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override {
      ++m_count;
    }
  };

  class EndRunProd : public edm::global::EDProducer<edm::EndRunProducer> {
  public:
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    void globalEndRunProduce(edm::Run&, edm::EventSetup const&) const override {
      ++m_count;
    }
  };
  
  class EndLumiProd : public edm::global::EDProducer<edm::EndLuminosityBlockProducer> {
  public:
    mutable unsigned int m_count = 0;
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {
      ++m_count;
    }
    
    void globalEndLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) const override {
      ++m_count;
    }
  };


  class EndRunSummaryProd : public edm::global::EDProducer<edm::EndRunProducer, edm::RunSummaryCache<int>> {
  public:
    mutable unsigned int m_count = 0;
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
    mutable unsigned int m_count = 0;
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
};

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


///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testGlobalModule);

testGlobalModule::testGlobalModule():
m_prodReg(new edm::ProductRegistry{}),
m_idHelper(new edm::BranchIDListHelper{}),
m_ep()
{
  //Setup the principals
  m_prodReg->setFrozen();
  m_idHelper->updateFromRegistry(*m_prodReg);
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
  edm::ProcessHistoryRegistry phr;
  m_ep->fillEventPrincipal(eventAux, phr);
  m_ep->setLuminosityBlockPrincipal(m_lbp);
  m_actReg.reset(new edm::ActivityRegistry);


  //For each transition, bind a lambda which will call the proper method of the Worker
  m_transToFunc[Trans::kBeginStream] = [this](edm::Worker* iBase) {
    edm::StreamContext streamContext(s_streamID0, nullptr);
    iBase->beginStream(s_streamID0, streamContext); };

  
  m_transToFunc[Trans::kGlobalBeginRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionGlobalBegin> Traits;
    edm::ParentContext nullParentContext;
    iBase->doWork<Traits>(*m_rp,*m_es,m_timer, s_streamID0, nullParentContext, nullptr); };
  m_transToFunc[Trans::kStreamBeginRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::ParentContext nullParentContext;
    iBase->doWork<Traits>(*m_rp,*m_es,m_timer, s_streamID0, nullParentContext, nullptr); };
  
  m_transToFunc[Trans::kGlobalBeginLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalBegin> Traits;
    edm::ParentContext nullParentContext;
    iBase->doWork<Traits>(*m_lbp,*m_es,m_timer, s_streamID0, nullParentContext, nullptr); };
  m_transToFunc[Trans::kStreamBeginLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::ParentContext nullParentContext;
    iBase->doWork<Traits>(*m_lbp,*m_es,m_timer, s_streamID0, nullParentContext, nullptr); };
  
  m_transToFunc[Trans::kEvent] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::EventPrincipal, edm::BranchActionStreamBegin> Traits;
    edm::StreamContext streamContext(s_streamID0, nullptr);
    edm::ParentContext nullParentContext(&streamContext);
    iBase->setActivityRegistry(m_actReg);
    iBase->doWork<Traits>(*m_ep,*m_es,m_timer, s_streamID0, nullParentContext, nullptr); };

  m_transToFunc[Trans::kStreamEndLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionStreamEnd> Traits;
    edm::ParentContext nullParentContext;
    iBase->doWork<Traits>(*m_lbp,*m_es,m_timer, s_streamID0, nullParentContext, nullptr); };
  m_transToFunc[Trans::kGlobalEndLuminosityBlock] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::LuminosityBlockPrincipal, edm::BranchActionGlobalEnd> Traits;
    edm::ParentContext nullParentContext;
    iBase->doWork<Traits>(*m_lbp,*m_es,m_timer, s_streamID0, nullParentContext, nullptr); };

  m_transToFunc[Trans::kStreamEndRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionStreamEnd> Traits;
    edm::ParentContext nullParentContext;
    iBase->doWork<Traits>(*m_rp,*m_es,m_timer, s_streamID0, nullParentContext, nullptr); };
  m_transToFunc[Trans::kGlobalEndRun] = [this](edm::Worker* iBase) {
    typedef edm::OccurrenceTraits<edm::RunPrincipal, edm::BranchActionGlobalEnd> Traits;
    edm::ParentContext nullParentContext;
    iBase->doWork<Traits>(*m_rp,*m_es,m_timer, s_streamID0, nullParentContext, nullptr); };

  m_transToFunc[Trans::kEndStream] = [this](edm::Worker* iBase) {
    edm::StreamContext streamContext(s_streamID0, nullptr);
    iBase->endStream(s_streamID0, streamContext); };

}


namespace {
  template<typename T>
  void
  testTransition(T* iMod, edm::Worker* iWorker, testGlobalModule::Trans iTrans, testGlobalModule::Expectations const& iExpect, std::function<void(edm::Worker*)> iFunc) {
    assert(0==iMod->m_count);
    iFunc(iWorker);
    auto count = std::count(iExpect.begin(),iExpect.end(),iTrans);
    if(count != iMod->m_count) {
      std::cout<<"For trans " <<static_cast<std::underlying_type<testGlobalModule::Trans>::type >(iTrans)<< " expected "<<count<<" and got "<<iMod->m_count<<std::endl;
    }
    CPPUNIT_ASSERT(iMod->m_count == count);
    iMod->m_count = 0;
    iWorker->reset();
  }
}

template<typename T>
void
testGlobalModule::testTransitions(T* iMod, Expectations const& iExpect) {
  edm::WorkerT<edm::global::EDProducerBase> w{iMod,m_desc,nullptr};
  for(auto& keyVal: m_transToFunc) {
    testTransition(iMod,&w,keyVal.first,iExpect,keyVal.second);
  }
}


void testGlobalModule::basicTest()
{
  std::unique_ptr<BasicProd> testProd{ new BasicProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd.get(), {Trans::kEvent});
}

void testGlobalModule::streamTest()
{
  std::unique_ptr<StreamProd> testProd{ new StreamProd };
  edm::maker::ModuleHolderT<edm::global::EDProducerBase> h(testProd.get(),nullptr);
  h.preallocate(edm::PreallocationConfiguration{});
  h.release();
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd.get(), {Trans::kBeginStream, Trans::kStreamBeginRun, Trans::kStreamBeginLuminosityBlock, Trans::kEvent,
    Trans::kStreamEndLuminosityBlock,Trans::kStreamEndRun,Trans::kEndStream});
}

void testGlobalModule::runTest()
{
  std::unique_ptr<RunProd> testProd{ new RunProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd.get(), {Trans::kGlobalBeginRun, Trans::kEvent, Trans::kGlobalEndRun});
}

void testGlobalModule::runSummaryTest()
{
  std::unique_ptr<RunSummaryProd> testProd{ new RunSummaryProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd.get(), {Trans::kGlobalBeginRun, Trans::kEvent, Trans::kStreamEndRun, Trans::kGlobalEndRun});
}

void testGlobalModule::lumiTest()
{
  std::unique_ptr<LumiProd> testProd{ new LumiProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd.get(), {Trans::kGlobalBeginLuminosityBlock, Trans::kEvent, Trans::kGlobalEndLuminosityBlock});
}

void testGlobalModule::lumiSummaryTest()
{
  std::unique_ptr<LumiSummaryProd> testProd{ new LumiSummaryProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd.get(), {Trans::kGlobalBeginLuminosityBlock, Trans::kEvent, Trans::kStreamEndLuminosityBlock, Trans::kGlobalEndLuminosityBlock});
}

void testGlobalModule::beginRunProdTest()
{
  std::unique_ptr<BeginRunProd> testProd{ new BeginRunProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd.get(), {Trans::kGlobalBeginRun, Trans::kEvent});
}

void testGlobalModule::beginLumiProdTest()
{
  std::unique_ptr<BeginLumiProd> testProd{ new BeginLumiProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd.get(), {Trans::kGlobalBeginLuminosityBlock, Trans::kEvent});
}

void testGlobalModule::endRunProdTest()
{
  std::unique_ptr<EndRunProd> testProd{ new EndRunProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd.get(), {Trans::kGlobalEndRun, Trans::kEvent});
}

void testGlobalModule::endLumiProdTest()
{
  std::unique_ptr<EndLumiProd> testProd{ new EndLumiProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd.get(), {Trans::kGlobalEndLuminosityBlock, Trans::kEvent});
}

void testGlobalModule::endRunSummaryProdTest()
{
  std::unique_ptr<EndRunSummaryProd> testProd{ new EndRunSummaryProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd.get(), {Trans::kGlobalEndRun, Trans::kEvent, Trans::kGlobalBeginRun, Trans::kStreamEndRun, Trans::kGlobalEndRun});
}

void testGlobalModule::endLumiSummaryProdTest()
{
  std::unique_ptr<EndLumiSummaryProd> testProd{ new EndLumiSummaryProd };
  
  CPPUNIT_ASSERT(0 == testProd->m_count);
  testTransitions(testProd.get(), {Trans::kGlobalEndLuminosityBlock, Trans::kEvent, Trans::kGlobalBeginLuminosityBlock, Trans::kStreamEndLuminosityBlock, Trans::kGlobalEndLuminosityBlock}); 
}

