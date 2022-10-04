
/*----------------------------------------------------------------------

Toy edm::stream::EDFilter modules of
edm::*Cache templates and edm::*Producer classes
for testing purposes only.

----------------------------------------------------------------------*/

#include <atomic>
#include <functional>
#include <iostream>
#include <map>
#include <tuple>
#include <unistd.h>
#include <vector>

#include "FWCore/Framework/interface/CacheHandle.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/maker/WorkerT.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

namespace edmtest {
  namespace stream {

    // anonymous namespace here causes build warnings
    namespace cache {
      struct Cache {
        Cache() : value(0), run(0), lumi(0) {}
        //Using mutable since we want to update the value.
        mutable std::atomic<unsigned int> value;
        mutable std::atomic<unsigned int> run;
        mutable std::atomic<unsigned int> lumi;
      };

      struct SummaryCache {
        // Intentionally not thread safe, not atomic
        unsigned int value = 0;
      };

      struct TestGlobalCacheFil {
        CMS_THREAD_SAFE mutable edm::EDPutTokenT<unsigned int> token_;
        CMS_THREAD_SAFE mutable edm::EDGetTokenT<unsigned int> getTokenBegin_;
        CMS_THREAD_SAFE mutable edm::EDGetTokenT<unsigned int> getTokenEnd_;
        unsigned int trans_{0};
        mutable std::atomic<unsigned int> m_count{0};
      };
    }  // namespace cache

    using Cache = cache::Cache;
    using SummaryCache = cache::SummaryCache;
    using TestGlobalCacheFil = cache::TestGlobalCacheFil;

    class GlobalIntFilter : public edm::stream::EDFilter<edm::GlobalCache<Cache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;

      static std::unique_ptr<Cache> initializeGlobalCache(edm::ParameterSet const&) {
        ++m_count;
        return std::make_unique<Cache>();
      }

      GlobalIntFilter(edm::ParameterSet const& p, const Cache* iGlobal) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
      }

      static void globalBeginJob(Cache* iGlobal) {
        ++m_count;
        if (iGlobal->value != 0) {
          throw cms::Exception("cache value") << iGlobal->value << " but it was supposed to be 0";
        }
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++((globalCache())->value);

        return true;
      }

      static void globalEndJob(Cache* iGlobal) {
        ++m_count;
        if (iGlobal->value != cvalue_) {
          throw cms::Exception("cache value") << iGlobal->value << " but it was supposed to be " << cvalue_;
        }
      }

      ~GlobalIntFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunIntFilter : public edm::stream::EDFilter<edm::RunCache<Cache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;

      RunIntFilter(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        m_count = 0;
        produces<unsigned int>();
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++(runCache()->value);
        return true;
      }

      static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
        ++m_count;
        auto pCache = std::make_shared<Cache>();
        pCache->run = iRun.runAuxiliary().run();
        return pCache;
      }

      void beginRun(edm::Run const& iRun, edm::EventSetup const&) override {
        if (runCache()->run != iRun.runAuxiliary().run()) {
          throw cms::Exception("begin out of sequence") << "beginRun seen before globalBeginRun";
        }
      }

      void endRun(edm::Run const& iRun, edm::EventSetup const&) override {
        if (runCache()->run != iRun.runAuxiliary().run()) {
          throw cms::Exception("end out of sequence") << "globalEndRun seen before endRun";
        }
      }

      static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
        ++m_count;
        auto pCache = iContext->run();
        if (pCache->run != iRun.runAuxiliary().run()) {
          throw cms::Exception("end out of sequence") << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
        }
        pCache->run = 0;
        if (iContext->run()->value != cvalue_) {
          throw cms::Exception("cache value") << iContext->run()->value << " but it was supposed to be " << cvalue_;
        }
      }

      ~RunIntFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiIntFilter : public edm::stream::EDFilter<edm::LuminosityBlockCache<Cache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;

      LumiIntFilter(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        m_count = 0;
        produces<unsigned int>();
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++(luminosityBlockCache()->value);

        return true;
      }

      static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB,
                                                               edm::EventSetup const&,
                                                               RunContext const*) {
        ++m_count;
        auto pCache = std::make_shared<Cache>();
        pCache->run = iLB.luminosityBlockAuxiliary().run();
        pCache->lumi = iLB.luminosityBlockAuxiliary().luminosityBlock();
        return pCache;
      }

      void beginLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) override {
        if (luminosityBlockCache()->run != iLB.luminosityBlockAuxiliary().run() ||
            luminosityBlockCache()->lumi != iLB.luminosityBlockAuxiliary().luminosityBlock()) {
          throw cms::Exception("begin out of sequence")
              << "beginLuminosityBlock seen before globalBeginLuminosityBlock " << luminosityBlockCache()->run << " "
              << iLB.luminosityBlockAuxiliary().run();
        }
      }

      static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           LuminosityBlockContext const* iLBContext) {
        ++m_count;
        auto pCache = iLBContext->luminosityBlock();
        if (pCache->run != iLB.luminosityBlockAuxiliary().run() ||
            pCache->lumi != iLB.luminosityBlockAuxiliary().luminosityBlock()) {
          throw cms::Exception("end out of sequence")
              << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock"
              << iLB.luminosityBlock();
        }
        pCache->run = 0;
        pCache->lumi = 0;
        if (pCache->value != cvalue_) {
          throw cms::Exception("cache value") << "LumiIntFilter cache value " << iLBContext->luminosityBlock()->value
                                              << " but it was supposed to be " << cvalue_;
        }
      }

      void endLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) override {
        if (luminosityBlockCache()->run != iLB.luminosityBlockAuxiliary().run() ||
            luminosityBlockCache()->lumi != iLB.luminosityBlockAuxiliary().luminosityBlock()) {
          throw cms::Exception("end out of sequence") << "globalEndLuminosityBlock seen before endLuminosityBlock";
        }
      }

      ~LumiIntFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunSummaryIntFilter : public edm::stream::EDFilter<edm::RunCache<Cache>, edm::RunSummaryCache<SummaryCache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> globalBeginRunCalled_;
      unsigned int valueAccumulatedForStream_ = 0;
      bool endRunWasCalled_ = false;

      RunSummaryIntFilter(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        m_count = 0;
        produces<unsigned int>();
      }

      void beginRun(edm::Run const&, edm::EventSetup const&) override {
        valueAccumulatedForStream_ = 0;
        endRunWasCalled_ = false;
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++(runCache()->value);
        ++valueAccumulatedForStream_;
        return true;
      }

      static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
        ++m_count;
        globalBeginRunCalled_ = true;
        auto pCache = std::make_shared<Cache>();
        ++(pCache->run);
        return pCache;
      }

      static std::shared_ptr<SummaryCache> globalBeginRunSummary(edm::Run const&,
                                                                 edm::EventSetup const&,
                                                                 GlobalCache const*) {
        ++m_count;
        if (!globalBeginRunCalled_) {
          throw cms::Exception("begin out of sequence") << "globalBeginRunSummary seen before globalBeginRun";
        }
        globalBeginRunCalled_ = false;
        return std::make_shared<SummaryCache>();
      }

      void endRunSummary(edm::Run const&, edm::EventSetup const&, SummaryCache* runSummaryCache) const override {
        runSummaryCache->value += valueAccumulatedForStream_;
        if (!endRunWasCalled_) {
          throw cms::Exception("end out of sequence") << "endRunSummary seen before endRun";
        }
      }

      static void globalEndRunSummary(edm::Run const&,
                                      edm::EventSetup const&,
                                      RunContext const*,
                                      SummaryCache* runSummaryCache) {
        ++m_count;
        if (runSummaryCache->value != cvalue_) {
          throw cms::Exception("unexpectedValue")
              << "run summary cache value = " << runSummaryCache->value << " but it was supposed to be " << cvalue_;
        }
      }

      static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
        ++m_count;
        auto pCache = iContext->run();
        if (pCache->value != cvalue_) {
          throw cms::Exception("unExpectedValue")
              << "run cache value " << pCache->value << " but it was supposed to be " << cvalue_;
        }
        if (pCache->run != 1) {
          throw cms::Exception("end out of sequence") << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
        }
      }

      void endRun(edm::Run const&, edm::EventSetup const&) override { endRunWasCalled_ = true; }

      ~RunSummaryIntFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiSummaryIntFilter
        : public edm::stream::EDFilter<edm::LuminosityBlockCache<Cache>, edm::LuminosityBlockSummaryCache<SummaryCache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> globalBeginLumiCalled_;
      unsigned int valueAccumulatedForStream_ = 0;
      bool endLumiWasCalled_ = false;

      LumiSummaryIntFilter(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        m_count = 0;
        produces<unsigned int>();
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++(luminosityBlockCache()->value);
        ++valueAccumulatedForStream_;
        return true;
      }

      void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
        valueAccumulatedForStream_ = 0;
        valueAccumulatedForStream_ = 0;
        endLumiWasCalled_ = false;
      }

      static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB,
                                                               edm::EventSetup const&,
                                                               RunContext const*) {
        ++m_count;
        globalBeginLumiCalled_ = true;
        auto pCache = std::make_shared<Cache>();
        ++(pCache->lumi);
        return pCache;
      }

      static std::shared_ptr<SummaryCache> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                             edm::EventSetup const&,
                                                                             LuminosityBlockContext const*) {
        ++m_count;
        if (!globalBeginLumiCalled_) {
          throw cms::Exception("begin out of sequence")
              << "globalBeginLuminosityBlockSummary seen before globalBeginLuminosityBlock";
        }
        globalBeginLumiCalled_ = false;
        return std::make_shared<SummaryCache>();
      }

      void endLuminosityBlockSummary(edm::LuminosityBlock const&,
                                     edm::EventSetup const&,
                                     SummaryCache* lumiSummaryCache) const override {
        lumiSummaryCache->value += valueAccumulatedForStream_;
        if (!endLumiWasCalled_) {
          throw cms::Exception("end out of sequence") << "endLuminosityBlockSummary seen before endLuminosityBlock";
        }
      }

      static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                  edm::EventSetup const&,
                                                  LuminosityBlockContext const* iLBContext,
                                                  SummaryCache* lumiSummaryCache) {
        ++m_count;
        if (lumiSummaryCache->value != cvalue_) {
          throw cms::Exception("unexpectedValue")
              << "lumi summary cache value = " << lumiSummaryCache->value << " but it was supposed to be " << cvalue_;
        }
        auto pCache = iLBContext->luminosityBlock();
        // Add one so globalEndLuminosityBlock can check this function was called first
        ++pCache->value;
      }

      static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           LuminosityBlockContext const* iLBContext) {
        ++m_count;
        auto pCache = iLBContext->luminosityBlock();
        if (pCache->value != cvalue_ + 1) {
          throw cms::Exception("unexpectedValue")
              << "lumi cache value " << pCache->value << " but it was supposed to be " << cvalue_ + 1;
        }
        if (pCache->lumi != 1) {
          throw cms::Exception("end out of sequence")
              << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock"
              << iLB.luminosityBlock();
        }
      }

      void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
        endLumiWasCalled_ = true;
      }

      ~LumiSummaryIntFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class ProcessBlockIntFilter
        : public edm::stream::EDFilter<edm::WatchProcessBlock, edm::GlobalCache<TestGlobalCacheFil>> {
    public:
      explicit ProcessBlockIntFilter(edm::ParameterSet const& pset, TestGlobalCacheFil const* testGlobalCache) {
        produces<unsigned int>();

        {
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock");
          if (not tag.label().empty()) {
            testGlobalCache->getTokenBegin_ = consumes<unsigned int, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock");
          if (not tag.label().empty()) {
            testGlobalCache->getTokenEnd_ = consumes<unsigned int, edm::InProcess>(tag);
          }
        }
      }

      static std::unique_ptr<TestGlobalCacheFil> initializeGlobalCache(edm::ParameterSet const& pset) {
        auto testGlobalCache = std::make_unique<TestGlobalCacheFil>();
        testGlobalCache->trans_ = pset.getParameter<int>("transitions");
        return testGlobalCache;
      }

      static void beginProcessBlock(edm::ProcessBlock const& processBlock, TestGlobalCacheFil* testGlobalCache) {
        if (testGlobalCache->m_count != 0) {
          throw cms::Exception("transitions") << "ProcessBlockIntFilter::begin transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << 0;
        }
        ++testGlobalCache->m_count;
        const unsigned int valueToGet = 71;
        if (not testGlobalCache->getTokenBegin_.isUninitialized()) {
          if (processBlock.get(testGlobalCache->getTokenBegin_) != valueToGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToGet << " but got " << processBlock.get(testGlobalCache->getTokenBegin_);
          }
        }
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        TestGlobalCacheFil const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count < 1u) {
          throw cms::Exception("out of sequence") << "produce before beginProcessBlock " << testGlobalCache->m_count;
        }
        ++testGlobalCache->m_count;
        return true;
      }

      static void endProcessBlock(edm::ProcessBlock const& processBlock, TestGlobalCacheFil* testGlobalCache) {
        ++testGlobalCache->m_count;
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "ProcessBlockIntFilter::end transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
        {
          const unsigned int valueToGet = 71;
          if (not testGlobalCache->getTokenBegin_.isUninitialized()) {
            if (processBlock.get(testGlobalCache->getTokenBegin_) != valueToGet) {
              throw cms::Exception("BadValue")
                  << "expected " << valueToGet << " but got " << processBlock.get(testGlobalCache->getTokenBegin_);
            }
          }
        }
        {
          const unsigned int valueToGet = 81;
          if (not testGlobalCache->getTokenEnd_.isUninitialized()) {
            if (processBlock.get(testGlobalCache->getTokenEnd_) != valueToGet) {
              throw cms::Exception("BadValue")
                  << "expected " << valueToGet << " but got " << processBlock.get(testGlobalCache->getTokenEnd_);
            }
          }
        }
      }

      static void globalEndJob(TestGlobalCacheFil* testGlobalCache) {
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "ProcessBlockIntFilter transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }

      ~ProcessBlockIntFilter() {
        TestGlobalCacheFil const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "ProcessBlockIntFilter transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }
    };

    class TestBeginProcessBlockFilter
        : public edm::stream::EDFilter<edm::BeginProcessBlockProducer, edm::GlobalCache<TestGlobalCacheFil>> {
    public:
      explicit TestBeginProcessBlockFilter(edm::ParameterSet const& pset, TestGlobalCacheFil const* testGlobalCache) {
        testGlobalCache->token_ = produces<unsigned int, edm::Transition::BeginProcessBlock>("begin");
        produces<unsigned int>();

        auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock");
        if (not tag.label().empty()) {
          testGlobalCache->getTokenBegin_ = consumes<unsigned int, edm::InProcess>(tag);
        }
      }

      static std::unique_ptr<TestGlobalCacheFil> initializeGlobalCache(edm::ParameterSet const& pset) {
        auto testGlobalCache = std::make_unique<TestGlobalCacheFil>();
        testGlobalCache->trans_ = pset.getParameter<int>("transitions");
        return testGlobalCache;
      }

      static void beginProcessBlockProduce(edm::ProcessBlock& processBlock, TestGlobalCacheFil const* testGlobalCache) {
        if (testGlobalCache->m_count != 0) {
          throw cms::Exception("transitions") << "TestBeginProcessBlockFilter transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << 0;
        }
        ++testGlobalCache->m_count;

        const unsigned int valueToPutAndGet = 71;
        processBlock.emplace(testGlobalCache->token_, valueToPutAndGet);

        if (not testGlobalCache->getTokenBegin_.isUninitialized()) {
          if (processBlock.get(testGlobalCache->getTokenBegin_) != valueToPutAndGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToPutAndGet << " but got " << processBlock.get(testGlobalCache->getTokenBegin_);
          }
        }
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        TestGlobalCacheFil const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count < 1u) {
          throw cms::Exception("out of sequence")
              << "produce before beginProcessBlockProduce " << testGlobalCache->m_count;
        }
        ++testGlobalCache->m_count;
        return true;
      }

      static void globalEndJob(TestGlobalCacheFil* testGlobalCache) {
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "TestBeginProcessBlockFilter transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }

      ~TestBeginProcessBlockFilter() {
        TestGlobalCacheFil const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "TestBeginProcessBlockFilter transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }
    };

    class TestEndProcessBlockFilter
        : public edm::stream::EDFilter<edm::EndProcessBlockProducer, edm::GlobalCache<TestGlobalCacheFil>> {
    public:
      explicit TestEndProcessBlockFilter(edm::ParameterSet const& pset, TestGlobalCacheFil const* testGlobalCache) {
        testGlobalCache->token_ = produces<unsigned int, edm::Transition::EndProcessBlock>("end");
        produces<unsigned int>();

        auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock");
        if (not tag.label().empty()) {
          testGlobalCache->getTokenEnd_ = consumes<unsigned int, edm::InProcess>(tag);
        }
      }

      static std::unique_ptr<TestGlobalCacheFil> initializeGlobalCache(edm::ParameterSet const& pset) {
        auto testGlobalCache = std::make_unique<TestGlobalCacheFil>();
        testGlobalCache->trans_ = pset.getParameter<int>("transitions");
        return testGlobalCache;
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        TestGlobalCacheFil const* testGlobalCache = globalCache();
        ++testGlobalCache->m_count;
        return true;
      }

      static void endProcessBlockProduce(edm::ProcessBlock& processBlock, TestGlobalCacheFil const* testGlobalCache) {
        ++testGlobalCache->m_count;
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "TestEndProcessBlockFilter transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }

        const unsigned int valueToPutAndGet = 81;
        processBlock.emplace(testGlobalCache->token_, valueToPutAndGet);
        if (not testGlobalCache->getTokenEnd_.isUninitialized()) {
          if (processBlock.get(testGlobalCache->getTokenEnd_) != valueToPutAndGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToPutAndGet << " but got " << processBlock.get(testGlobalCache->getTokenEnd_);
          }
        }
      }

      static void globalEndJob(TestGlobalCacheFil* testGlobalCache) {
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "TestEndProcessBlockFilter transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }

      ~TestEndProcessBlockFilter() {
        TestGlobalCacheFil const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "~TestEndProcessBlockFilter transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }
    };

    class TestBeginRunFilter : public edm::stream::EDFilter<edm::RunCache<Cache>, edm::BeginRunProducer> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbr;
      static std::atomic<bool> ger;

      TestBeginRunFilter(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        m_count = 0;
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::BeginRun>("a");
      }

      static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
        ++m_count;
        gbr = true;
        ger = false;
        auto pCache = std::make_shared<Cache>();
        ++(pCache->run);
        return pCache;
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        return true;
      }

      static void globalBeginRunProduce(edm::Run& iRun, edm::EventSetup const&, RunContext const*) {
        ++m_count;
        if (!gbr) {
          throw cms::Exception("begin out of sequence") << "globalBeginRunProduce seen before globalBeginRun";
        }
      }

      static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
        ++m_count;
        auto pCache = iContext->run();
        if (pCache->run != 1) {
          throw cms::Exception("end out of sequence") << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
        }
        gbr = false;
        ger = true;
      }

      ~TestBeginRunFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestEndRunFilter : public edm::stream::EDFilter<edm::RunCache<Cache>, edm::EndRunProducer> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbr;
      static std::atomic<bool> ger;

      static std::shared_ptr<Cache> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
        ++m_count;
        gbr = true;
        ger = false;
        auto pCache = std::make_shared<Cache>();
        ++(pCache->run);
        return pCache;
      }

      TestEndRunFilter(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        m_count = 0;
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::EndRun>("a");
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;

        return true;
      }

      static void globalEndRunProduce(edm::Run& iRun, edm::EventSetup const&, RunContext const*) {
        ++m_count;
        if (ger) {
          throw cms::Exception("end out of sequence") << "globalEndRun seen before globalEndRunProduce";
        }
      }

      static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
        ++m_count;
        auto pCache = iContext->run();
        if (pCache->run != 1) {
          throw cms::Exception("end out of sequence") << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
        }
        gbr = false;
        ger = true;
      }

      ~TestEndRunFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestBeginLumiBlockFilter
        : public edm::stream::EDFilter<edm::LuminosityBlockCache<Cache>, edm::BeginLuminosityBlockProducer> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbl;
      static std::atomic<bool> gel;

      TestBeginLumiBlockFilter(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        m_count = 0;
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::BeginLuminosityBlock>("a");
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;

        return true;
      }

      static void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&,
                                                    edm::EventSetup const&,
                                                    LuminosityBlockContext const*) {
        ++m_count;
        if (!gbl) {
          throw cms::Exception("begin out of sequence")
              << "globalBeginLumiBlockProduce seen before globalBeginLumiBlock";
        }
      }

      static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB,
                                                               edm::EventSetup const&,
                                                               RunContext const*) {
        ++m_count;
        gbl = true;
        gel = false;
        auto pCache = std::make_shared<Cache>();
        ++(pCache->lumi);
        return pCache;
      }

      static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           LuminosityBlockContext const* iLBContext) {
        ++m_count;
        auto pCache = iLBContext->luminosityBlock();
        if (pCache->lumi != 1) {
          throw cms::Exception("end out of sequence")
              << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock"
              << iLB.luminosityBlock();
        }
        gel = true;
        gbl = false;
      }

      ~TestBeginLumiBlockFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestEndLumiBlockFilter
        : public edm::stream::EDFilter<edm::LuminosityBlockCache<Cache>, edm::EndLuminosityBlockProducer> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbl;
      static std::atomic<bool> gel;

      TestEndLumiBlockFilter(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        m_count = 0;
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::EndLuminosityBlock>("a");
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;

        return true;
      }

      static std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB,
                                                               edm::EventSetup const&,
                                                               RunContext const*) {
        ++m_count;
        gbl = true;
        gel = false;
        auto pCache = std::make_shared<Cache>();
        ++(pCache->lumi);
        return pCache;
      }

      static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           LuminosityBlockContext const* iLBContext) {
        ++m_count;
        auto pCache = iLBContext->luminosityBlock();
        if (pCache->lumi != 1) {
          throw cms::Exception("end out of sequence")
              << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock"
              << iLB.luminosityBlock();
        }
        gel = true;
        gbl = false;
      }

      static void globalEndLuminosityBlockProduce(edm::LuminosityBlock&,
                                                  edm::EventSetup const&,
                                                  LuminosityBlockContext const*) {
        ++m_count;
      }

      ~TestEndLumiBlockFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestInputProcessBlockCache {
    public:
      long long int value_ = 0;
    };

    class TestInputProcessBlockCache1 {
    public:
      long long int value_ = 0;
    };

    class InputProcessBlockIntFilter
        : public edm::stream::EDFilter<
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>> {
    public:
      explicit InputProcessBlockIntFilter(edm::ParameterSet const& pset) {
        {
          expectedByRun_ = pset.getParameter<std::vector<int>>("expectedByRun");
          sleepTime_ = pset.getParameter<unsigned int>("sleepTime");
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock");
          if (not tag.label().empty()) {
            getTokenBegin_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock");
          if (not tag.label().empty()) {
            getTokenEnd_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        registerProcessBlockCacheFiller<TestInputProcessBlockCache1>(
            getTokenBegin_,
            [this](edm::ProcessBlock const& processBlock,
                   std::shared_ptr<TestInputProcessBlockCache1> const& previousCache) {
              auto returnValue = std::make_shared<TestInputProcessBlockCache1>();
              returnValue->value_ += processBlock.get(getTokenBegin_).value;
              returnValue->value_ += processBlock.get(getTokenEnd_).value;
              return returnValue;
            });
      }

      static void accessInputProcessBlock(edm::ProcessBlock const&) {
        edm::LogAbsolute("InputProcessBlockIntFilter") << "InputProcessBlockIntFilter::accessInputProcessBlock";
      }

      bool filter(edm::Event& event, edm::EventSetup const&) override {
        auto cacheTuple = processBlockCaches(event);
        if (!expectedByRun_.empty()) {
          if (expectedByRun_.at(event.run() - 1) !=
              std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntFilter::filter cached value was "
                << std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_
                << " but it was supposed to be " << expectedByRun_.at(event.run() - 1);
          }
        }
        // Force events to be processed concurrently
        if (sleepTime_ > 0) {
          usleep(sleepTime_);
        }
        return true;
      }

    private:
      edm::EDGetTokenT<IntProduct> getTokenBegin_;
      edm::EDGetTokenT<IntProduct> getTokenEnd_;
      std::vector<int> expectedByRun_;
      unsigned int sleepTime_{0};
    };

    struct InputProcessBlockGlobalCacheAn {
      // The tokens are duplicated in this test module to prove that they
      // work both as GlobalCache members and module data members.
      // We need them as GlobalCache members for accessInputProcessBlock.
      // In registerProcessBlockCacheFiller we use tokens that are member
      // variables of the class and because the lambda captures the "this"
      // pointer of the zeroth stream module instance. We always
      // use the zeroth EDConsumer. In the case of registerProcessBlockCacheFiller,
      // either set of tokens would work. Note that in the GlobalCache case
      // there is a slight weirdness that the zeroth consumer is used but
      // the token comes from the last consumer instance. It works because
      // all the stream module instances have EDConsumer base classes with
      // containers with the same contents in the same order (not 100% guaranteed,
      // but it would be difficult to implement a module where this isn't true).
      CMS_THREAD_SAFE mutable edm::EDGetTokenT<IntProduct> getTokenBegin_;
      CMS_THREAD_SAFE mutable edm::EDGetTokenT<IntProduct> getTokenEnd_;
      CMS_THREAD_SAFE mutable edm::EDGetTokenT<IntProduct> getTokenBeginM_;
      CMS_THREAD_SAFE mutable edm::EDGetTokenT<IntProduct> getTokenEndM_;
      mutable std::atomic<unsigned int> transitions_{0};
      int sum_{0};
      unsigned int expectedTransitions_{0};
      std::vector<int> expectedByRun_;
      int expectedSum_{0};
      unsigned int sleepTime_{0};
    };

    // Same thing as previous class except with a GlobalCache added
    class InputProcessBlockIntFilterG
        : public edm::stream::EDFilter<
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>,
              edm::GlobalCache<InputProcessBlockGlobalCacheAn>> {
    public:
      explicit InputProcessBlockIntFilterG(edm::ParameterSet const& pset,
                                           InputProcessBlockGlobalCacheAn const* testGlobalCache) {
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock");
          if (not tag.label().empty()) {
            getTokenBegin_ = consumes<IntProduct, edm::InProcess>(tag);
            testGlobalCache->getTokenBegin_ = getTokenBegin_;
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock");
          if (not tag.label().empty()) {
            getTokenEnd_ = consumes<IntProduct, edm::InProcess>(tag);
            testGlobalCache->getTokenEnd_ = getTokenEnd_;
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlockM");
          if (not tag.label().empty()) {
            getTokenBeginM_ = consumes<IntProduct, edm::InProcess>(tag);
            testGlobalCache->getTokenBeginM_ = getTokenBeginM_;
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlockM");
          if (not tag.label().empty()) {
            getTokenEndM_ = consumes<IntProduct, edm::InProcess>(tag);
            testGlobalCache->getTokenEndM_ = getTokenEndM_;
          }
        }
        registerProcessBlockCacheFiller<int>(
            getTokenBegin_, [this](edm::ProcessBlock const& processBlock, std::shared_ptr<int> const& previousCache) {
              auto returnValue = std::make_shared<int>(0);
              *returnValue += processBlock.get(getTokenBegin_).value;
              *returnValue += processBlock.get(getTokenEnd_).value;
              ++globalCache()->transitions_;
              return returnValue;
            });
        registerProcessBlockCacheFiller<1>(getTokenBegin_,
                                           [this](edm::ProcessBlock const& processBlock,
                                                  std::shared_ptr<TestInputProcessBlockCache> const& previousCache) {
                                             auto returnValue = std::make_shared<TestInputProcessBlockCache>();
                                             returnValue->value_ += processBlock.get(getTokenBegin_).value;
                                             returnValue->value_ += processBlock.get(getTokenEnd_).value;
                                             ++globalCache()->transitions_;
                                             return returnValue;
                                           });
        registerProcessBlockCacheFiller<TestInputProcessBlockCache1>(
            getTokenBegin_,
            [this](edm::ProcessBlock const& processBlock,
                   std::shared_ptr<TestInputProcessBlockCache1> const& previousCache) {
              auto returnValue = std::make_shared<TestInputProcessBlockCache1>();
              returnValue->value_ += processBlock.get(getTokenBegin_).value;
              returnValue->value_ += processBlock.get(getTokenEnd_).value;
              ++globalCache()->transitions_;
              return returnValue;
            });
      }

      static std::unique_ptr<InputProcessBlockGlobalCacheAn> initializeGlobalCache(edm::ParameterSet const& pset) {
        auto testGlobalCache = std::make_unique<InputProcessBlockGlobalCacheAn>();
        testGlobalCache->expectedTransitions_ = pset.getParameter<int>("transitions");
        testGlobalCache->expectedByRun_ = pset.getParameter<std::vector<int>>("expectedByRun");
        testGlobalCache->expectedSum_ = pset.getParameter<int>("expectedSum");
        testGlobalCache->sleepTime_ = pset.getParameter<unsigned int>("sleepTime");
        return testGlobalCache;
      }

      static void accessInputProcessBlock(edm::ProcessBlock const& processBlock,
                                          InputProcessBlockGlobalCacheAn* testGlobalCache) {
        if (processBlock.processName() == "PROD1") {
          testGlobalCache->sum_ += processBlock.get(testGlobalCache->getTokenBegin_).value;
          testGlobalCache->sum_ += processBlock.get(testGlobalCache->getTokenEnd_).value;
        }
        if (processBlock.processName() == "MERGE") {
          testGlobalCache->sum_ += processBlock.get(testGlobalCache->getTokenBeginM_).value;
          testGlobalCache->sum_ += processBlock.get(testGlobalCache->getTokenEndM_).value;
        }
        ++testGlobalCache->transitions_;
      }

      bool filter(edm::Event& event, edm::EventSetup const&) override {
        auto cacheTuple = processBlockCaches(event);
        auto testGlobalCache = globalCache();
        if (!testGlobalCache->expectedByRun_.empty()) {
          if (testGlobalCache->expectedByRun_.at(event.run() - 1) != *std::get<edm::CacheHandle<int>>(cacheTuple)) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntFilterG::filter cached value was "
                << *std::get<edm::CacheHandle<int>>(cacheTuple) << " but it was supposed to be "
                << testGlobalCache->expectedByRun_.at(event.run() - 1);
          }
          if (testGlobalCache->expectedByRun_.at(event.run() - 1) != std::get<1>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntFilterG::filter second cached value was " << std::get<1>(cacheTuple)->value_
                << " but it was supposed to be " << testGlobalCache->expectedByRun_.at(event.run() - 1);
          }
          if (testGlobalCache->expectedByRun_.at(event.run() - 1) !=
              std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntFilterG::filter third cached value was "
                << std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_
                << " but it was supposed to be " << testGlobalCache->expectedByRun_.at(event.run() - 1);
          }
        }
        ++testGlobalCache->transitions_;

        // Force events to be processed concurrently
        if (testGlobalCache->sleepTime_ > 0) {
          usleep(testGlobalCache->sleepTime_);
        }
        return true;
      }

      static void globalEndJob(InputProcessBlockGlobalCacheAn* testGlobalCache) {
        if (testGlobalCache->transitions_ != testGlobalCache->expectedTransitions_) {
          throw cms::Exception("transitions")
              << "InputProcessBlockIntFilterG transitions " << testGlobalCache->transitions_
              << " but it was supposed to be " << testGlobalCache->expectedTransitions_;
        }

        if (testGlobalCache->sum_ != testGlobalCache->expectedSum_) {
          throw cms::Exception("UnexpectedValue") << "InputProcessBlockIntFilterG sum " << testGlobalCache->sum_
                                                  << " but it was supposed to be " << testGlobalCache->expectedSum_;
        }
      }

    private:
      edm::EDGetTokenT<IntProduct> getTokenBegin_;
      edm::EDGetTokenT<IntProduct> getTokenEnd_;
      edm::EDGetTokenT<IntProduct> getTokenBeginM_;
      edm::EDGetTokenT<IntProduct> getTokenEndM_;
    };

  }  // namespace stream
}  // namespace edmtest
std::atomic<unsigned int> edmtest::stream::GlobalIntFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunIntFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiIntFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunSummaryIntFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestBeginRunFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestEndRunFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestBeginLumiBlockFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestEndLumiBlockFilter::m_count{0};
std::atomic<unsigned int> edmtest::stream::GlobalIntFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::RunIntFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::LumiIntFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::RunSummaryIntFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestBeginRunFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestEndRunFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestBeginLumiBlockFilter::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestEndLumiBlockFilter::cvalue_{0};
std::atomic<bool> edmtest::stream::RunSummaryIntFilter::globalBeginRunCalled_{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntFilter::globalBeginLumiCalled_{false};
std::atomic<bool> edmtest::stream::TestBeginRunFilter::gbr{false};
std::atomic<bool> edmtest::stream::TestBeginRunFilter::ger{false};
std::atomic<bool> edmtest::stream::TestEndRunFilter::gbr{false};
std::atomic<bool> edmtest::stream::TestEndRunFilter::ger{false};
std::atomic<bool> edmtest::stream::TestBeginLumiBlockFilter::gbl{false};
std::atomic<bool> edmtest::stream::TestBeginLumiBlockFilter::gel{false};
std::atomic<bool> edmtest::stream::TestEndLumiBlockFilter::gbl{false};
std::atomic<bool> edmtest::stream::TestEndLumiBlockFilter::gel{false};
DEFINE_FWK_MODULE(edmtest::stream::GlobalIntFilter);
DEFINE_FWK_MODULE(edmtest::stream::RunIntFilter);
DEFINE_FWK_MODULE(edmtest::stream::LumiIntFilter);
DEFINE_FWK_MODULE(edmtest::stream::RunSummaryIntFilter);
DEFINE_FWK_MODULE(edmtest::stream::LumiSummaryIntFilter);
DEFINE_FWK_MODULE(edmtest::stream::ProcessBlockIntFilter);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginProcessBlockFilter);
DEFINE_FWK_MODULE(edmtest::stream::TestEndProcessBlockFilter);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginRunFilter);
DEFINE_FWK_MODULE(edmtest::stream::TestEndRunFilter);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginLumiBlockFilter);
DEFINE_FWK_MODULE(edmtest::stream::TestEndLumiBlockFilter);
DEFINE_FWK_MODULE(edmtest::stream::InputProcessBlockIntFilter);
DEFINE_FWK_MODULE(edmtest::stream::InputProcessBlockIntFilterG);
