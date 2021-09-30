
/*----------------------------------------------------------------------

Toy edm::stream::EDProducer modules of
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
#include "FWCore/Framework/interface/stream/EDProducer.h"
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

      struct TestGlobalCache {
        CMS_THREAD_SAFE mutable edm::EDPutTokenT<unsigned int> token_;
        CMS_THREAD_SAFE mutable edm::EDGetTokenT<unsigned int> getTokenBegin_;
        CMS_THREAD_SAFE mutable edm::EDGetTokenT<unsigned int> getTokenEnd_;
        unsigned int trans_{0};
        mutable std::atomic<unsigned int> m_count{0};
      };

    }  // namespace cache

    using Cache = cache::Cache;
    using SummaryCache = cache::SummaryCache;
    using TestGlobalCache = cache::TestGlobalCache;

    class GlobalIntProducer : public edm::stream::EDProducer<edm::GlobalCache<Cache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;

      static std::unique_ptr<Cache> initializeGlobalCache(edm::ParameterSet const&) {
        ++m_count;
        return std::make_unique<Cache>();
      }

      GlobalIntProducer(edm::ParameterSet const& p, const Cache* iGlobal) {
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

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++((globalCache())->value);
      }

      static void globalEndJob(Cache* iGlobal) {
        ++m_count;
        if (iGlobal->value != cvalue_) {
          throw cms::Exception("cache value") << iGlobal->value << " but it was supposed to be " << cvalue_;
        }
      }

      ~GlobalIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunIntProducer : public edm::stream::EDProducer<edm::RunCache<Cache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;

      RunIntProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++(runCache()->value);
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

      void endRun(edm::Run const& iRun, edm::EventSetup const&) override {
        if (runCache()->run != iRun.runAuxiliary().run()) {
          throw cms::Exception("end out of sequence") << "globalEndRun seen before endRun";
        }
      }

      ~RunIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiIntProducer : public edm::stream::EDProducer<edm::LuminosityBlockCache<Cache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;

      LumiIntProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++(luminosityBlockCache()->value);
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
        if (iLBContext->luminosityBlock()->value != cvalue_) {
          throw cms::Exception("cache value")
              << iLBContext->luminosityBlock()->value << " but it was supposed to be " << cvalue_;
        }
      }

      void endLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) override {
        if (luminosityBlockCache()->run != iLB.luminosityBlockAuxiliary().run() ||
            luminosityBlockCache()->lumi != iLB.luminosityBlockAuxiliary().luminosityBlock()) {
          throw cms::Exception("end out of sequence") << "globalEndLuminosityBlock seen before endLuminosityBlock";
        }
      }

      ~LumiIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunSummaryIntProducer
        : public edm::stream::EDProducer<edm::RunCache<Cache>, edm::RunSummaryCache<SummaryCache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> globalBeginRunCalled_;
      unsigned int valueAccumulatedForStream_ = 0;
      bool endRunWasCalled_ = false;

      RunSummaryIntProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
      }

      void beginRun(edm::Run const&, edm::EventSetup const&) override {
        valueAccumulatedForStream_ = 0;
        endRunWasCalled_ = false;
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++(runCache()->value);
        ++valueAccumulatedForStream_;
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

      ~RunSummaryIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiSummaryIntProducer : public edm::stream::EDProducer<edm::LuminosityBlockCache<Cache>,
                                                                  edm::LuminosityBlockSummaryCache<SummaryCache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> globalBeginLumiCalled_;
      unsigned int valueAccumulatedForStream_ = 0;
      bool endLumiWasCalled_ = false;

      LumiSummaryIntProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        ++(luminosityBlockCache()->value);
        ++valueAccumulatedForStream_;
      }

      void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
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

      ~LumiSummaryIntProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class ProcessBlockIntProducer
        : public edm::stream::EDProducer<edm::WatchProcessBlock, edm::GlobalCache<TestGlobalCache>> {
    public:
      explicit ProcessBlockIntProducer(edm::ParameterSet const& pset, TestGlobalCache const* testGlobalCache) {
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

      static std::unique_ptr<TestGlobalCache> initializeGlobalCache(edm::ParameterSet const& pset) {
        auto testGlobalCache = std::make_unique<TestGlobalCache>();
        testGlobalCache->trans_ = pset.getParameter<int>("transitions");
        return testGlobalCache;
      }

      static void beginProcessBlock(edm::ProcessBlock const& processBlock, TestGlobalCache* testGlobalCache) {
        if (testGlobalCache->m_count != 0) {
          throw cms::Exception("transitions") << "ProcessBlockIntProducer::begin transitions "
                                              << testGlobalCache->m_count << " but it was supposed to be " << 0;
        }
        ++testGlobalCache->m_count;

        const unsigned int valueToGet = 51;
        if (not testGlobalCache->getTokenBegin_.isUninitialized()) {
          if (processBlock.get(testGlobalCache->getTokenBegin_) != valueToGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToGet << " but got " << processBlock.get(testGlobalCache->getTokenBegin_);
          }
        }
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        TestGlobalCache const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count < 1u) {
          throw cms::Exception("out of sequence") << "produce before beginProcessBlock " << testGlobalCache->m_count;
        }
        ++testGlobalCache->m_count;
      }

      static void endProcessBlock(edm::ProcessBlock const& processBlock, TestGlobalCache* testGlobalCache) {
        ++testGlobalCache->m_count;
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "ProcessBlockIntProducer::end transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
        {
          const unsigned int valueToGet = 51;
          if (not testGlobalCache->getTokenBegin_.isUninitialized()) {
            if (processBlock.get(testGlobalCache->getTokenBegin_) != valueToGet) {
              throw cms::Exception("BadValue")
                  << "expected " << valueToGet << " but got " << processBlock.get(testGlobalCache->getTokenBegin_);
            }
          }
        }
        {
          const unsigned int valueToGet = 61;
          if (not testGlobalCache->getTokenEnd_.isUninitialized()) {
            if (processBlock.get(testGlobalCache->getTokenEnd_) != valueToGet) {
              throw cms::Exception("BadValue")
                  << "expected " << valueToGet << " but got " << processBlock.get(testGlobalCache->getTokenEnd_);
            }
          }
        }
      }

      static void globalEndJob(TestGlobalCache* testGlobalCache) {
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions")
              << "TestBeginProcessBlockProducer transitions " << testGlobalCache->m_count
              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }

      ~ProcessBlockIntProducer() {
        TestGlobalCache const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "ProcessBlockIntProducer transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }
    };

    class TestBeginProcessBlockProducer
        : public edm::stream::EDProducer<edm::BeginProcessBlockProducer, edm::GlobalCache<TestGlobalCache>> {
    public:
      explicit TestBeginProcessBlockProducer(edm::ParameterSet const& pset, TestGlobalCache const* testGlobalCache) {
        testGlobalCache->token_ = produces<unsigned int, edm::Transition::BeginProcessBlock>("begin");

        auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock");
        if (not tag.label().empty()) {
          testGlobalCache->getTokenBegin_ = consumes<unsigned int, edm::InProcess>(tag);
        }
      }

      static std::unique_ptr<TestGlobalCache> initializeGlobalCache(edm::ParameterSet const& pset) {
        auto testGlobalCache = std::make_unique<TestGlobalCache>();
        testGlobalCache->trans_ = pset.getParameter<int>("transitions");
        return testGlobalCache;
      }

      static void beginProcessBlockProduce(edm::ProcessBlock& processBlock, TestGlobalCache const* testGlobalCache) {
        if (testGlobalCache->m_count != 0) {
          throw cms::Exception("transitions") << "TestBeginProcessBlockProducer transitions "
                                              << testGlobalCache->m_count << " but it was supposed to be " << 0;
        }
        ++testGlobalCache->m_count;

        const unsigned int valueToPutAndGet = 51;
        processBlock.emplace(testGlobalCache->token_, valueToPutAndGet);

        if (not testGlobalCache->getTokenBegin_.isUninitialized()) {
          if (processBlock.get(testGlobalCache->getTokenBegin_) != valueToPutAndGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToPutAndGet << " but got " << processBlock.get(testGlobalCache->getTokenBegin_);
          }
        }
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        TestGlobalCache const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count < 1u) {
          throw cms::Exception("out of sequence")
              << "produce before beginProcessBlockProduce " << testGlobalCache->m_count;
        }
        ++testGlobalCache->m_count;
      }

      static void globalEndJob(TestGlobalCache* testGlobalCache) {
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions")
              << "TestBeginProcessBlockProducer transitions " << testGlobalCache->m_count
              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }

      ~TestBeginProcessBlockProducer() {
        TestGlobalCache const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions")
              << "TestBeginProcessBlockProducer transitions " << testGlobalCache->m_count
              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }
    };

    class TestEndProcessBlockProducer
        : public edm::stream::EDProducer<edm::EndProcessBlockProducer, edm::GlobalCache<TestGlobalCache>> {
    public:
      explicit TestEndProcessBlockProducer(edm::ParameterSet const& pset, TestGlobalCache const* testGlobalCache) {
        testGlobalCache->token_ = produces<unsigned int, edm::Transition::EndProcessBlock>("end");

        auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock");
        if (not tag.label().empty()) {
          testGlobalCache->getTokenEnd_ = consumes<unsigned int, edm::InProcess>(tag);
        }
      }

      static std::unique_ptr<TestGlobalCache> initializeGlobalCache(edm::ParameterSet const& pset) {
        auto testGlobalCache = std::make_unique<TestGlobalCache>();
        testGlobalCache->trans_ = pset.getParameter<int>("transitions");
        return testGlobalCache;
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        TestGlobalCache const* testGlobalCache = globalCache();
        ++testGlobalCache->m_count;
      }

      static void endProcessBlockProduce(edm::ProcessBlock& processBlock, TestGlobalCache const* testGlobalCache) {
        ++testGlobalCache->m_count;
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "TestEndProcessBlockProducer transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
        const unsigned int valueToPutAndGet = 61;
        processBlock.emplace(testGlobalCache->token_, valueToPutAndGet);

        if (not testGlobalCache->getTokenEnd_.isUninitialized()) {
          if (processBlock.get(testGlobalCache->getTokenEnd_) != valueToPutAndGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToPutAndGet << " but got " << processBlock.get(testGlobalCache->getTokenEnd_);
          }
        }
      }

      static void globalEndJob(TestGlobalCache* testGlobalCache) {
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions")
              << "TestBeginProcessBlockProducer transitions " << testGlobalCache->m_count
              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }

      ~TestEndProcessBlockProducer() {
        TestGlobalCache const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "~TestEndProcessBlockProducer transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }
    };

    class ProcessBlockIntProducerNoGlobalCache : public edm::stream::EDProducer<edm::WatchProcessBlock> {
    public:
      explicit ProcessBlockIntProducerNoGlobalCache(edm::ParameterSet const&) {}

      static void beginProcessBlock(edm::ProcessBlock const&) {}

      void produce(edm::Event&, edm::EventSetup const&) override {}

      static void endProcessBlock(edm::ProcessBlock const&) {}
    };

    class TestBeginProcessBlockProducerNoGlobalCache : public edm::stream::EDProducer<edm::BeginProcessBlockProducer> {
    public:
      explicit TestBeginProcessBlockProducerNoGlobalCache(edm::ParameterSet const&) {}

      static void beginProcessBlockProduce(edm::ProcessBlock&) {}

      void produce(edm::Event&, edm::EventSetup const&) override {}
    };

    class TestEndProcessBlockProducerNoGlobalCache : public edm::stream::EDProducer<edm::EndProcessBlockProducer> {
    public:
      explicit TestEndProcessBlockProducerNoGlobalCache(edm::ParameterSet const&) {}

      void produce(edm::Event&, edm::EventSetup const&) override {}

      static void endProcessBlockProduce(edm::ProcessBlock&) {}
    };

    class TestBeginRunProducer : public edm::stream::EDProducer<edm::RunCache<bool>, edm::BeginRunProducer> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbr;
      static std::atomic<bool> ger;
      static std::atomic<bool> gbrp;

      TestBeginRunProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::BeginRun>("a");
      }

      static std::shared_ptr<bool> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
        gbr = true;
        ger = false;
        gbrp = false;
        return std::shared_ptr<bool>{};
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        if (!gbrp) {
          throw cms::Exception("out of sequence") << "produce before globalBeginRunProduce";
        }
      }

      static void globalBeginRunProduce(edm::Run& iRun, edm::EventSetup const&, RunContext const*) {
        gbrp = true;
        ++m_count;
        if (!gbr) {
          throw cms::Exception("begin out of sequence") << "globalBeginRunProduce seen before globalBeginRun";
        }
      }

      static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
        if (!gbr) {
          throw cms::Exception("end out of sequence") << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
        }
        gbr = false;
        ger = true;
      }

      ~TestBeginRunProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestEndRunProducer : public edm::stream::EDProducer<edm::RunCache<bool>, edm::EndRunProducer> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbr;
      static std::atomic<bool> ger;
      static std::atomic<bool> p;

      static std::shared_ptr<bool> globalBeginRun(edm::Run const& iRun, edm::EventSetup const&, GlobalCache const*) {
        gbr = true;
        ger = false;
        p = false;
        return std::shared_ptr<bool>{};
      }

      TestEndRunProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::EndRun>("a");
      }

      void produce(edm::Event&, edm::EventSetup const&) override { p = true; }

      static void globalEndRunProduce(edm::Run& iRun, edm::EventSetup const&, RunContext const*) {
        ++m_count;
        if (!p) {
          throw cms::Exception("out of sequence") << "globalEndRunProduce seen before produce";
        }
        if (ger) {
          throw cms::Exception("out of sequence") << "globalEndRun seen before globalEndRunProduce";
        }
      }

      static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext) {
        if (!gbr) {
          throw cms::Exception("out of sequence") << "globalEndRun seen before globalBeginRun in Run" << iRun.run();
        }
        gbr = false;
        ger = true;
      }

      ~TestEndRunProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestBeginLumiBlockProducer
        : public edm::stream::EDProducer<edm::LuminosityBlockCache<bool>, edm::BeginLuminosityBlockProducer> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbl;
      static std::atomic<bool> gel;
      static std::atomic<bool> gblp;

      TestBeginLumiBlockProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::BeginLuminosityBlock>("a");
      }

      void produce(edm::Event&, edm::EventSetup const&) override {
        if (!gblp) {
          throw cms::Exception("begin out of sequence") << "produce seen before globalBeginLumiBlockProduce";
        }
      }

      static void globalBeginLuminosityBlockProduce(edm::LuminosityBlock&,
                                                    edm::EventSetup const&,
                                                    LuminosityBlockContext const*) {
        ++m_count;
        if (!gbl) {
          throw cms::Exception("begin out of sequence")
              << "globalBeginLumiBlockProduce seen before globalBeginLumiBlock";
        }
        gblp = true;
      }

      static std::shared_ptr<bool> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB,
                                                              edm::EventSetup const&,
                                                              RunContext const*) {
        gbl = true;
        gel = false;
        gblp = false;
        return std::shared_ptr<bool>();
      }

      static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           LuminosityBlockContext const* iLBContext) {
        if (!gbl) {
          throw cms::Exception("end out of sequence")
              << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock"
              << iLB.luminosityBlock();
        }
        gel = true;
        gbl = false;
      }

      ~TestBeginLumiBlockProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class TestEndLumiBlockProducer
        : public edm::stream::EDProducer<edm::LuminosityBlockCache<bool>, edm::EndLuminosityBlockProducer> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> gbl;
      static std::atomic<bool> gel;
      static std::atomic<bool> p;

      TestEndLumiBlockProducer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        produces<unsigned int>();
        produces<unsigned int, edm::Transition::EndLuminosityBlock>("a");
      }

      void produce(edm::Event&, edm::EventSetup const&) override { p = true; }

      static void globalEndLuminosityBlockProduce(edm::LuminosityBlock&,
                                                  edm::EventSetup const&,
                                                  LuminosityBlockContext const*) {
        ++m_count;
        if (!p) {
          throw cms::Exception("out of sequence") << "globalEndLumiBlockProduce seen before produce";
        }
      }

      static std::shared_ptr<bool> globalBeginLuminosityBlock(edm::LuminosityBlock const& iLB,
                                                              edm::EventSetup const&,
                                                              RunContext const*) {
        gbl = true;
        gel = false;
        p = false;
        return std::shared_ptr<bool>{};
      }

      static void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           LuminosityBlockContext const* iLBContext) {
        if (!gbl) {
          throw cms::Exception("end out of sequence")
              << "globalEndLuminosityBlock seen before globalBeginLuminosityBlock in LuminosityBlock"
              << iLB.luminosityBlock();
        }
      }

      ~TestEndLumiBlockProducer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    struct Count {
      Count() : m_value(0), m_expectedValue(0) {}
      //Using mutable since we want to update the value.
      mutable std::atomic<unsigned int> m_value;
      mutable std::atomic<unsigned int> m_expectedValue;
      // Set only in constructor, framework runs constructors serially
      CMS_THREAD_SAFE mutable edm::EDPutTokenT<unsigned int> m_putToken;
    };

    class TestAccumulator
        : public edm::stream::EDProducer<edm::GlobalCache<Count>, edm::Accumulator, edm::EndLuminosityBlockProducer> {
    public:
      static std::atomic<unsigned int> m_expectedCount;

      explicit TestAccumulator(edm::ParameterSet const& p, Count const* iCount) {
        iCount->m_expectedValue = p.getParameter<unsigned int>("expectedCount");
        if (iCount->m_putToken.isUninitialized()) {
          iCount->m_putToken = produces<unsigned int, edm::Transition::EndLuminosityBlock>();
        }
      }

      static std::unique_ptr<Count> initializeGlobalCache(edm::ParameterSet const&) {
        return std::unique_ptr<Count>(new Count());
      }

      void accumulate(edm::Event const&, edm::EventSetup const&) override { ++(globalCache()->m_value); }

      static void globalEndLuminosityBlockProduce(edm::LuminosityBlock& l,
                                                  edm::EventSetup const&,
                                                  LuminosityBlockContext const* ctx) {
        Count const* count = ctx->global();
        l.emplace(count->m_putToken, count->m_value.load());
      }

      static void globalEndJob(Count const* iCount) {
        if (iCount->m_value != iCount->m_expectedValue) {
          throw cms::Exception("CountEvents") << "Number of events seen = " << iCount->m_value
                                              << " but it was supposed to be " << iCount->m_expectedValue;
        }
      }

      ~TestAccumulator() {}
    };

    class TestInputProcessBlockCache {
    public:
      long long int value_ = 0;
    };

    class TestInputProcessBlockCache1 {
    public:
      long long int value_ = 0;
    };

    class InputProcessBlockIntProducer
        : public edm::stream::EDProducer<
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>> {
    public:
      explicit InputProcessBlockIntProducer(edm::ParameterSet const& pset) {
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
        edm::LogAbsolute("InputProcessBlockIntProducer") << "InputProcessBlockIntProducer::accessInputProcessBlock";
      }

      void produce(edm::Event& event, edm::EventSetup const&) override {
        auto cacheTuple = processBlockCaches(event);
        if (!expectedByRun_.empty()) {
          if (expectedByRun_.at(event.run() - 1) !=
              std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntProducer::produce cached value was "
                << std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_
                << " but it was supposed to be " << expectedByRun_.at(event.run() - 1);
          }
        }
        // Force events to be processed concurrently
        if (sleepTime_ > 0) {
          usleep(sleepTime_);
        }
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
    class InputProcessBlockIntProducerG
        : public edm::stream::EDProducer<
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>,
              edm::GlobalCache<InputProcessBlockGlobalCacheAn>> {
    public:
      explicit InputProcessBlockIntProducerG(edm::ParameterSet const& pset,
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

      void produce(edm::Event& event, edm::EventSetup const&) override {
        auto cacheTuple = processBlockCaches(event);
        auto testGlobalCache = globalCache();
        if (!testGlobalCache->expectedByRun_.empty()) {
          if (testGlobalCache->expectedByRun_.at(event.run() - 1) != *std::get<edm::CacheHandle<int>>(cacheTuple)) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntProducerG::produce cached value was "
                << *std::get<edm::CacheHandle<int>>(cacheTuple) << " but it was supposed to be "
                << testGlobalCache->expectedByRun_.at(event.run() - 1);
          }
          if (testGlobalCache->expectedByRun_.at(event.run() - 1) != std::get<1>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntProducerG::produce second cached value was " << std::get<1>(cacheTuple)->value_
                << " but it was supposed to be " << testGlobalCache->expectedByRun_.at(event.run() - 1);
          }
          if (testGlobalCache->expectedByRun_.at(event.run() - 1) !=
              std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntProducerG::produce third cached value was "
                << std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_
                << " but it was supposed to be " << testGlobalCache->expectedByRun_.at(event.run() - 1);
          }
        }
        ++testGlobalCache->transitions_;

        // Force events to be processed concurrently
        if (testGlobalCache->sleepTime_ > 0) {
          usleep(testGlobalCache->sleepTime_);
        }
      }

      static void globalEndJob(InputProcessBlockGlobalCacheAn* testGlobalCache) {
        if (testGlobalCache->transitions_ != testGlobalCache->expectedTransitions_) {
          throw cms::Exception("transitions")
              << "InputProcessBlockIntProducerG transitions " << testGlobalCache->transitions_
              << " but it was supposed to be " << testGlobalCache->expectedTransitions_;
        }

        if (testGlobalCache->sum_ != testGlobalCache->expectedSum_) {
          throw cms::Exception("UnexpectedValue") << "InputProcessBlockIntProducerG sum " << testGlobalCache->sum_
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
std::atomic<unsigned int> edmtest::stream::GlobalIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunSummaryIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestBeginRunProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestEndRunProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestBeginLumiBlockProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::TestEndLumiBlockProducer::m_count{0};
std::atomic<unsigned int> edmtest::stream::GlobalIntProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::RunIntProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::LumiIntProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::RunSummaryIntProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestBeginRunProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestEndRunProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestBeginLumiBlockProducer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::TestEndLumiBlockProducer::cvalue_{0};
std::atomic<bool> edmtest::stream::RunSummaryIntProducer::globalBeginRunCalled_{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntProducer::globalBeginLumiCalled_{false};
std::atomic<bool> edmtest::stream::TestBeginRunProducer::gbr{false};
std::atomic<bool> edmtest::stream::TestBeginRunProducer::gbrp{false};
std::atomic<bool> edmtest::stream::TestBeginRunProducer::ger{false};
std::atomic<bool> edmtest::stream::TestEndRunProducer::gbr{false};
std::atomic<bool> edmtest::stream::TestEndRunProducer::ger{false};
std::atomic<bool> edmtest::stream::TestEndRunProducer::p{false};
std::atomic<bool> edmtest::stream::TestBeginLumiBlockProducer::gbl{false};
std::atomic<bool> edmtest::stream::TestBeginLumiBlockProducer::gblp{false};
std::atomic<bool> edmtest::stream::TestBeginLumiBlockProducer::gel{false};
std::atomic<bool> edmtest::stream::TestEndLumiBlockProducer::gbl{false};
std::atomic<bool> edmtest::stream::TestEndLumiBlockProducer::gel{false};
std::atomic<bool> edmtest::stream::TestEndLumiBlockProducer::p{false};
DEFINE_FWK_MODULE(edmtest::stream::GlobalIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::RunIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::LumiIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::RunSummaryIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::LumiSummaryIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::ProcessBlockIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginProcessBlockProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestEndProcessBlockProducer);
DEFINE_FWK_MODULE(edmtest::stream::ProcessBlockIntProducerNoGlobalCache);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginProcessBlockProducerNoGlobalCache);
DEFINE_FWK_MODULE(edmtest::stream::TestEndProcessBlockProducerNoGlobalCache);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginRunProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestEndRunProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestBeginLumiBlockProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestEndLumiBlockProducer);
DEFINE_FWK_MODULE(edmtest::stream::TestAccumulator);
DEFINE_FWK_MODULE(edmtest::stream::InputProcessBlockIntProducer);
DEFINE_FWK_MODULE(edmtest::stream::InputProcessBlockIntProducerG);
