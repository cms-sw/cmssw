
/*----------------------------------------------------------------------

Toy edm::stream::EDAnalyzer modules of
edm::*Cache templates
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
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/maker/WorkerT.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/ServiceRegistry/interface/ParentContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
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
      struct TestGlobalCacheAn {
        CMS_THREAD_SAFE mutable edm::EDGetTokenT<unsigned int> getTokenBegin_;
        CMS_THREAD_SAFE mutable edm::EDGetTokenT<unsigned int> getTokenEnd_;
        unsigned int trans_{0};
        mutable std::atomic<unsigned int> m_count{0};
      };
    }  // namespace cache

    using Cache = cache::Cache;
    using SummaryCache = cache::SummaryCache;
    using TestGlobalCacheAn = cache::TestGlobalCacheAn;

    class GlobalIntAnalyzer : public edm::stream::EDAnalyzer<edm::GlobalCache<Cache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;

      static std::unique_ptr<Cache> initializeGlobalCache(edm::ParameterSet const& p) {
        ++m_count;
        return std::make_unique<Cache>();
      }

      GlobalIntAnalyzer(edm::ParameterSet const& p, Cache const* iGlobal) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        callWhenNewProductsRegistered([](edm::BranchDescription const& desc) {
          std::cout << "stream::GlobalIntAnalyzer " << desc.moduleLabel() << std::endl;
        });
      }

      static void globalBeginJob(Cache* iGlobal) {
        ++m_count;
        if (iGlobal->value != 0) {
          throw cms::Exception("cache value") << iGlobal->value << " but it was supposed to be 0";
        }
      }

      void analyze(edm::Event const&, edm::EventSetup const&) {
        ++m_count;
        ++((globalCache())->value);
      }

      static void globalEndJob(Cache* iGlobal) {
        ++m_count;
        if (iGlobal->value != cvalue_) {
          throw cms::Exception("cache value") << iGlobal->value << " but it was supposed to be " << cvalue_;
        }
      }

      ~GlobalIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunIntAnalyzer : public edm::stream::EDAnalyzer<edm::RunCache<Cache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;

      RunIntAnalyzer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        m_count = 0;
      }

      void analyze(edm::Event const&, edm::EventSetup const&) override {
        if (moduleDescription().processName() != edm::Service<edm::service::TriggerNamesService>()->getProcessName()) {
          throw cms::Exception("LogicError") << "module description not properly initialized in stream analyzer";
        }
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

      ~RunIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiIntAnalyzer : public edm::stream::EDAnalyzer<edm::LuminosityBlockCache<Cache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;

      LumiIntAnalyzer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        m_count = 0;
        // just to create a data dependence
        auto const& tag = p.getParameter<edm::InputTag>("moduleLabel");
        if (not tag.label().empty()) {
          consumes<unsigned int, edm::InLumi>(tag);
        }
      }

      void analyze(edm::Event const&, edm::EventSetup const&) override {
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
              << "beginLuminosityBlock seen before globalBeginLuminosityBlock";
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

      ~LumiIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunSummaryIntAnalyzer
        : public edm::stream::EDAnalyzer<edm::RunCache<Cache>, edm::RunSummaryCache<SummaryCache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> globalBeginRunCalled_;
      unsigned int valueAccumulatedForStream_ = 0;
      bool endRunWasCalled_ = false;

      RunSummaryIntAnalyzer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        m_count = 0;
      }

      void analyze(edm::Event const&, edm::EventSetup const&) override {
        ++m_count;
        ++(runCache()->value);
        ++valueAccumulatedForStream_;
      }

      void beginRun(edm::Run const&, edm::EventSetup const&) override {
        valueAccumulatedForStream_ = 0;
        endRunWasCalled_ = false;
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

      ~RunSummaryIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiSummaryIntAnalyzer : public edm::stream::EDAnalyzer<edm::LuminosityBlockCache<Cache>,
                                                                  edm::LuminosityBlockSummaryCache<SummaryCache>> {
    public:
      static std::atomic<unsigned int> m_count;
      unsigned int trans_;
      static std::atomic<unsigned int> cvalue_;
      static std::atomic<bool> globalBeginLumiCalled_;
      unsigned int valueAccumulatedForStream_ = 0;
      bool endLumiWasCalled_ = false;

      LumiSummaryIntAnalyzer(edm::ParameterSet const& p) {
        trans_ = p.getParameter<int>("transitions");
        cvalue_ = p.getParameter<int>("cachevalue");
        m_count = 0;
      }

      void analyze(edm::Event const&, edm::EventSetup const&) override {
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

      ~LumiSummaryIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions") << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class ProcessBlockIntAnalyzer
        : public edm::stream::EDAnalyzer<edm::WatchProcessBlock, edm::GlobalCache<TestGlobalCacheAn>> {
    public:
      explicit ProcessBlockIntAnalyzer(edm::ParameterSet const& pset, TestGlobalCacheAn const* testGlobalCache) {
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

      static std::unique_ptr<TestGlobalCacheAn> initializeGlobalCache(edm::ParameterSet const& pset) {
        auto testGlobalCache = std::make_unique<TestGlobalCacheAn>();
        testGlobalCache->trans_ = pset.getParameter<int>("transitions");
        return testGlobalCache;
      }

      static void beginProcessBlock(edm::ProcessBlock const& processBlock, TestGlobalCacheAn* testGlobalCache) {
        if (testGlobalCache->m_count != 0) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::begin transitions "
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

      void analyze(edm::Event const&, edm::EventSetup const&) override {
        TestGlobalCacheAn const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count < 1u) {
          throw cms::Exception("out of sequence") << "produce before beginProcessBlock " << testGlobalCache->m_count;
        }
        ++testGlobalCache->m_count;
      }

      static void endProcessBlock(edm::ProcessBlock const& processBlock, TestGlobalCacheAn* testGlobalCache) {
        ++testGlobalCache->m_count;
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::end transitions " << testGlobalCache->m_count
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

      static void globalEndJob(TestGlobalCacheAn* testGlobalCache) {
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions")
              << "TestBeginProcessBlockAnalyzer transitions " << testGlobalCache->m_count
              << " but it was supposed to be " << testGlobalCache->trans_;
        }
      }

      ~ProcessBlockIntAnalyzer() {
        TestGlobalCacheAn const* testGlobalCache = globalCache();
        if (testGlobalCache->m_count != testGlobalCache->trans_) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer transitions " << testGlobalCache->m_count
                                              << " but it was supposed to be " << testGlobalCache->trans_;
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

    class InputProcessBlockIntAnalyzer
        : public edm::stream::EDAnalyzer<
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>> {
    public:
      explicit InputProcessBlockIntAnalyzer(edm::ParameterSet const& pset) {
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
        edm::LogAbsolute("InputProcessBlockIntAnalyzer") << "InputProcessBlockIntAnalyzer::accessInputProcessBlock";
      }

      void analyze(edm::Event const& event, edm::EventSetup const&) override {
        auto cacheTuple = processBlockCaches(event);
        if (!expectedByRun_.empty()) {
          if (expectedByRun_.at(event.run() - 1) !=
              std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntAnalyzer::analyze cached value was "
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
    class InputProcessBlockIntAnalyzerG
        : public edm::stream::EDAnalyzer<
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>,
              edm::GlobalCache<InputProcessBlockGlobalCacheAn>> {
    public:
      explicit InputProcessBlockIntAnalyzerG(edm::ParameterSet const& pset,
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

      void analyze(edm::Event const& event, edm::EventSetup const&) override {
        auto cacheTuple = processBlockCaches(event);
        auto testGlobalCache = globalCache();
        if (!testGlobalCache->expectedByRun_.empty()) {
          if (testGlobalCache->expectedByRun_.at(event.run() - 1) != *std::get<edm::CacheHandle<int>>(cacheTuple)) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntAnalyzerG::analyze cached value was "
                << *std::get<edm::CacheHandle<int>>(cacheTuple) << " but it was supposed to be "
                << testGlobalCache->expectedByRun_.at(event.run() - 1);
          }
          if (testGlobalCache->expectedByRun_.at(event.run() - 1) != std::get<1>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntAnalyzerG::analyze second cached value was " << std::get<1>(cacheTuple)->value_
                << " but it was supposed to be " << testGlobalCache->expectedByRun_.at(event.run() - 1);
          }
          if (testGlobalCache->expectedByRun_.at(event.run() - 1) !=
              std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntAnalyzerG::analyze third cached value was "
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
              << "InputProcessBlockIntAnalyzerG transitions " << testGlobalCache->transitions_
              << " but it was supposed to be " << testGlobalCache->expectedTransitions_;
        }

        if (testGlobalCache->sum_ != testGlobalCache->expectedSum_) {
          throw cms::Exception("UnexpectedValue") << "InputProcessBlockIntAnalyzerG sum " << testGlobalCache->sum_
                                                  << " but it was supposed to be " << testGlobalCache->expectedSum_;
        }
      }

    private:
      edm::EDGetTokenT<IntProduct> getTokenBegin_;
      edm::EDGetTokenT<IntProduct> getTokenEnd_;
      edm::EDGetTokenT<IntProduct> getTokenBeginM_;
      edm::EDGetTokenT<IntProduct> getTokenEndM_;
    };

    // The next two test that modules without the
    // static accessInputProcessBlock function will build.
    // And also that modules with no functor registered run.

    class InputProcessBlockIntAnalyzerNS
        : public edm::stream::EDAnalyzer<edm::InputProcessBlockCache<int, TestInputProcessBlockCache>> {
    public:
      explicit InputProcessBlockIntAnalyzerNS(edm::ParameterSet const& pset) {}
      void analyze(edm::Event const&, edm::EventSetup const&) override {}
    };

    // Same thing as previous class except with a GlobalCache added
    class InputProcessBlockIntAnalyzerGNS
        : public edm::stream::EDAnalyzer<edm::InputProcessBlockCache<int, TestInputProcessBlockCache>,
                                         edm::GlobalCache<TestGlobalCacheAn>> {
    public:
      explicit InputProcessBlockIntAnalyzerGNS(edm::ParameterSet const& pset,
                                               TestGlobalCacheAn const* testGlobalCache) {}
      static std::unique_ptr<TestGlobalCacheAn> initializeGlobalCache(edm::ParameterSet const&) {
        return std::make_unique<TestGlobalCacheAn>();
      }
      void analyze(edm::Event const&, edm::EventSetup const&) override {}
      static void globalEndJob(TestGlobalCacheAn* testGlobalCache) {}
    };

  }  // namespace stream
}  // namespace edmtest
std::atomic<unsigned int> edmtest::stream::GlobalIntAnalyzer::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunIntAnalyzer::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiIntAnalyzer::m_count{0};
std::atomic<unsigned int> edmtest::stream::RunSummaryIntAnalyzer::m_count{0};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntAnalyzer::m_count{0};
std::atomic<unsigned int> edmtest::stream::GlobalIntAnalyzer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::RunIntAnalyzer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::LumiIntAnalyzer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::RunSummaryIntAnalyzer::cvalue_{0};
std::atomic<unsigned int> edmtest::stream::LumiSummaryIntAnalyzer::cvalue_{0};
std::atomic<bool> edmtest::stream::RunSummaryIntAnalyzer::globalBeginRunCalled_{false};
std::atomic<bool> edmtest::stream::LumiSummaryIntAnalyzer::globalBeginLumiCalled_{false};
DEFINE_FWK_MODULE(edmtest::stream::GlobalIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::stream::RunIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::stream::LumiIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::stream::RunSummaryIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::stream::LumiSummaryIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::stream::ProcessBlockIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::stream::InputProcessBlockIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::stream::InputProcessBlockIntAnalyzerG);
DEFINE_FWK_MODULE(edmtest::stream::InputProcessBlockIntAnalyzerNS);
DEFINE_FWK_MODULE(edmtest::stream::InputProcessBlockIntAnalyzerGNS);
