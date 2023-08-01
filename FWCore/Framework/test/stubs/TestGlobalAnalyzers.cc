
/*----------------------------------------------------------------------

Toy edm::global::EDAnalyzer modules of
edm::*Cache templates
for testing purposes only.

----------------------------------------------------------------------*/

#include <atomic>
#include <iostream>
#include <memory>
#include <tuple>
#include <vector>

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/CacheHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edmtest {
  namespace global {

    namespace {
      struct Cache {
        Cache() : value(0) {}
        //Using mutable since we want to update the value.
        mutable std::atomic<unsigned int> value;
      };

      struct UnsafeCache {
        UnsafeCache() : value(0), lumi(0) {}
        unsigned int value;
        unsigned int lumi;
      };

    }  //end anonymous namespace

    class StreamIntAnalyzer : public edm::global::EDAnalyzer<edm::StreamCache<UnsafeCache>> {
    public:
      explicit StreamIntAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        bool verbose = p.getUntrackedParameter<bool>("verbose", true);
        callWhenNewProductsRegistered([verbose](edm::BranchDescription const& desc) {
          if (verbose) {
            std::cout << "global::StreamIntAnalyzer " << desc.moduleLabel() << std::endl;
          }
        });
      }
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID iID) const override {
        ++m_count;
        auto pCache = std::make_unique<UnsafeCache>();
        pCache->value = iID.value();
        return pCache;
      }

      void streamBeginRun(edm::StreamID iID, edm::Run const&, edm::EventSetup const&) const override {
        ++m_count;
        if ((streamCache(iID))->value != iID.value()) {
          throw cms::Exception("cache value")
              << "StreamIntAnalyzer cache value " << (streamCache(iID))->value << " but it was supposed to be " << iID;
        }
      }

      void streamBeginLuminosityBlock(edm::StreamID iID,
                                      edm::LuminosityBlock const&,
                                      edm::EventSetup const&) const override {
        ++m_count;
        if ((streamCache(iID))->value != iID.value()) {
          throw cms::Exception("cache value")
              << "StreamIntAnalyzer cache value " << (streamCache(iID))->value << " but it was supposed to be " << iID;
        }
      }

      void analyze(edm::StreamID iID, const edm::Event&, const edm::EventSetup&) const override {
        ++m_count;
        if ((streamCache(iID))->value != iID.value()) {
          throw cms::Exception("cache value")
              << "StreamIntAnalyzer cache value " << (streamCache(iID))->value << " but it was supposed to be " << iID;
        }
      }

      void streamEndLuminosityBlock(edm::StreamID iID,
                                    edm::LuminosityBlock const&,
                                    edm::EventSetup const&) const override {
        ++m_count;
        if ((streamCache(iID))->value != iID.value()) {
          throw cms::Exception("cache value")
              << "StreamIntAnalyzer cache value " << (streamCache(iID))->value << " but it was supposed to be " << iID;
        }
      }

      void streamEndRun(edm::StreamID iID, edm::Run const&, edm::EventSetup const&) const override {
        ++m_count;
        if ((streamCache(iID))->value != iID.value()) {
          throw cms::Exception("cache value")
              << "StreamIntAnalyzer cache value " << (streamCache(iID))->value << " but it was supposed to be " << iID;
        }
      }

      void endStream(edm::StreamID iID) const override {
        ++m_count;
        if ((streamCache(iID))->value != iID.value()) {
          throw cms::Exception("cache value")
              << "StreamIntAnalyzer cache value " << (streamCache(iID))->value << " but it was supposed to be " << iID;
        }
      }

      void endJob() override {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "StreamIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunIntAnalyzer : public edm::global::EDAnalyzer<edm::RunCache<Cache>> {
    public:
      explicit RunIntAnalyzer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")), cvalue_(p.getParameter<int>("cachevalue")) {}
      const unsigned int trans_;
      const unsigned int cvalue_;
      mutable std::atomic<unsigned int> m_count{0};

      std::shared_ptr<Cache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override {
        ++m_count;
        return std::make_shared<Cache>();
      }

      void analyze(edm::StreamID iID, const edm::Event& iEvent, const edm::EventSetup&) const override {
        ++m_count;
        ++((runCache(iEvent.getRun().index()))->value);
      }

      void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) const override {
        ++m_count;
        if ((runCache(iRun.index()))->value != cvalue_) {
          throw cms::Exception("cache value") << "RunIntAnalyzer cache value " << (runCache(iRun.index()))->value
                                              << " but it was supposed to be " << cvalue_;
        }
      }

      void endJob() override {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "RunIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiIntAnalyzer : public edm::global::EDAnalyzer<edm::LuminosityBlockCache<Cache>> {
    public:
      explicit LumiIntAnalyzer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")), cvalue_(p.getParameter<int>("cachevalue")) {
        // just to create a data dependence
        auto const& tag = p.getParameter<edm::InputTag>("moduleLabel");
        if (not tag.label().empty()) {
          consumes<unsigned int, edm::InLumi>(tag);
        }
      }
      const unsigned int trans_;
      const unsigned int cvalue_;
      mutable std::atomic<unsigned int> m_count{0};

      std::shared_ptr<Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                        edm::EventSetup const&) const override {
        ++m_count;
        return std::make_shared<Cache>();
      }

      void analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup&) const override {
        ++m_count;
        ++(luminosityBlockCache(iEvent.getLuminosityBlock().index())->value);
      }

      void globalEndLuminosityBlock(edm::LuminosityBlock const& iLB, edm::EventSetup const&) const override {
        ++m_count;
        if ((luminosityBlockCache(iLB.index()))->value != cvalue_) {
          throw cms::Exception("cache value")
              << "LumiIntAnalyzer cache value " << (luminosityBlockCache(iLB.index()))->value
              << " but it was supposed to be " << cvalue_;
        }
      }

      void endJob() override {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "LumiIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunSummaryIntAnalyzer
        : public edm::global::EDAnalyzer<edm::StreamCache<UnsafeCache>, edm::RunSummaryCache<UnsafeCache>> {
    public:
      explicit RunSummaryIntAnalyzer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")), cvalue_(p.getParameter<int>("cachevalue")) {}
      const unsigned int trans_;
      const unsigned int cvalue_;
      mutable std::atomic<unsigned int> m_count{0};

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override {
        ++m_count;
        return std::make_unique<UnsafeCache>();
      }

      std::shared_ptr<UnsafeCache> globalBeginRunSummary(edm::Run const&, edm::EventSetup const&) const override {
        ++m_count;
        return std::make_shared<UnsafeCache>();
      }

      void analyze(edm::StreamID iID, const edm::Event&, const edm::EventSetup&) const override {
        ++m_count;
        ++((streamCache(iID))->value);
      }

      void streamEndRunSummary(edm::StreamID iID,
                               edm::Run const&,
                               edm::EventSetup const&,
                               UnsafeCache* gCache) const override {
        ++m_count;
        gCache->value += (streamCache(iID))->value;
        (streamCache(iID))->value = 0;
      }

      void globalEndRunSummary(edm::Run const&, edm::EventSetup const&, UnsafeCache* gCache) const override {
        ++m_count;
        if (gCache->value != cvalue_) {
          throw cms::Exception("cache value")
              << "RunSummaryIntAnalyzer cache value " << gCache->value << " but it was supposed to be " << cvalue_;
        }
      }

      void endJob() override {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "RunSummaryIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiSummaryIntAnalyzer
        : public edm::global::EDAnalyzer<edm::StreamCache<UnsafeCache>, edm::LuminosityBlockSummaryCache<UnsafeCache>> {
    public:
      explicit LumiSummaryIntAnalyzer(edm::ParameterSet const& p)
          : trans_(p.getParameter<int>("transitions")), cvalue_(p.getParameter<int>("cachevalue")) {}
      const unsigned int trans_;
      const unsigned int cvalue_;
      mutable std::atomic<unsigned int> m_count{0};

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override {
        ++m_count;
        return std::make_unique<UnsafeCache>();
      }

      std::shared_ptr<UnsafeCache> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const& iLB,
                                                                     edm::EventSetup const&) const override {
        ++m_count;
        auto gCache = std::make_shared<UnsafeCache>();
        gCache->lumi = iLB.luminosityBlockAuxiliary().luminosityBlock();
        return gCache;
      }

      void analyze(edm::StreamID iID, const edm::Event& iEvent, const edm::EventSetup&) const override {
        ++m_count;
        ++((streamCache(iID))->value);
      }

      void streamEndLuminosityBlockSummary(edm::StreamID iID,
                                           edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           UnsafeCache* gCache) const override {
        ++m_count;
        if (gCache->lumi != iLB.luminosityBlockAuxiliary().luminosityBlock()) {
          throw cms::Exception("UnexpectedValue")
              << "streamEndLuminosityBlockSummary unexpected lumi number in Stream " << iID.value();
        }
        gCache->value += (streamCache(iID))->value;
        (streamCache(iID))->value = 0;
      }

      void globalEndLuminosityBlockSummary(edm::LuminosityBlock const& iLB,
                                           edm::EventSetup const&,
                                           UnsafeCache* gCache) const override {
        ++m_count;
        if (gCache->lumi != iLB.luminosityBlockAuxiliary().luminosityBlock()) {
          throw cms::Exception("UnexpectedValue") << "globalEndLuminosityBlockSummary unexpected lumi number";
        }
        if (gCache->value != cvalue_) {
          throw cms::Exception("cache value")
              << "LumiSummaryIntAnalyzer cache value " << gCache->value << " but it was supposed to be " << cvalue_;
        }
      }

      void endJob() override {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "LumiSummaryIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class ProcessBlockIntAnalyzer : public edm::global::EDAnalyzer<edm::WatchProcessBlock,
                                                                   edm::StreamCache<UnsafeCache>,
                                                                   edm::RunCache<UnsafeCache>> {
    public:
      explicit ProcessBlockIntAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        {
          auto tag = p.getParameter<edm::InputTag>("consumesBeginProcessBlock");
          if (not tag.label().empty()) {
            getTokenBegin_ = consumes<unsigned int, edm::InProcess>(tag);
          }
        }
        {
          auto tag = p.getParameter<edm::InputTag>("consumesEndProcessBlock");
          if (not tag.label().empty()) {
            getTokenEnd_ = consumes<unsigned int, edm::InProcess>(tag);
          }
        }
      }

      void beginJob() override {
        if (m_count != 0) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntAnalyzer::beginJob transition " << m_count << " but it was supposed to be " << 0;
        }
        ++m_count;
      }

      std::unique_ptr<UnsafeCache> beginStream(edm::StreamID) const override {
        if (m_count < 1) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::beginStream transition " << m_count
                                              << " but it was supposed to be at least 1";
        }
        ++m_count;
        return std::make_unique<UnsafeCache>();
      }

      void beginProcessBlock(edm::ProcessBlock const& processBlock) override {
        if (m_count != 5) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::beginProcessBlock transition " << m_count
                                              << " but it was supposed to be " << 5;
        }
        ++m_count;

        const unsigned int valueToGet = 11;
        if (not getTokenBegin_.isUninitialized()) {
          if (processBlock.get(getTokenBegin_) != valueToGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToGet << " but got " << processBlock.get(getTokenBegin_);
          }
        }
      }

      std::shared_ptr<UnsafeCache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const override {
        if (m_count < 6u) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::globalBeginRun transition " << m_count
                                              << " but it was supposed to be at least 6";
        }
        ++m_count;
        return std::make_shared<UnsafeCache>();
      }

      void analyze(edm::StreamID iID, edm::Event const&, edm::EventSetup const&) const override {
        if (m_count < 7u) {
          throw cms::Exception("out of sequence") << "analyze before beginProcessBlock " << m_count;
        }
        ++m_count;
      }

      void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {
        if (m_count < 15u) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::globalEndRun transition " << m_count
                                              << " but it was supposed to be at least 15";
        }
        ++m_count;
      }

      void endProcessBlock(edm::ProcessBlock const& processBlock) override {
        if (m_count != 646u) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::endProcessBlock transition " << m_count
                                              << " but it was supposed to be " << 646;
        }
        ++m_count;
        {
          const unsigned int valueToGet = 11;
          if (not getTokenBegin_.isUninitialized()) {
            if (processBlock.get(getTokenBegin_) != valueToGet) {
              throw cms::Exception("BadValue")
                  << "expected " << valueToGet << " but got " << processBlock.get(getTokenBegin_);
            }
          }
        }
        {
          const unsigned int valueToGet = 21;
          if (not getTokenEnd_.isUninitialized()) {
            if (processBlock.get(getTokenEnd_) != valueToGet) {
              throw cms::Exception("BadValue")
                  << "expected " << valueToGet << " but got " << processBlock.get(getTokenEnd_);
            }
          }
        }
      }

      void endStream(edm::StreamID) const override {
        if (m_count < 647u) {
          throw cms::Exception("transitions") << "ProcessBlockIntAnalyzer::endStream transition " << m_count
                                              << " but it was supposed to be at least 647";
        }
        ++m_count;
      }

      void endJob() override {
        if (m_count != 651u) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntAnalyzer::endJob transition " << m_count << " but it was supposed to be " << 651;
        }
        ++m_count;
      }

      ~ProcessBlockIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }

    private:
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};
      edm::EDGetTokenT<unsigned int> getTokenBegin_;
      edm::EDGetTokenT<unsigned int> getTokenEnd_;
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
        : public edm::global::EDAnalyzer<
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>> {
    public:
      explicit InputProcessBlockIntAnalyzer(edm::ParameterSet const& pset) {
        expectedTransitions_ = pset.getParameter<int>("transitions");
        expectedByRun_ = pset.getParameter<std::vector<int>>("expectedByRun");
        expectedSum_ = pset.getParameter<int>("expectedSum");
        {
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
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlockM");
          if (not tag.label().empty()) {
            getTokenBeginM_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlockM");
          if (not tag.label().empty()) {
            getTokenEndM_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        registerProcessBlockCacheFiller<int>(
            getTokenBegin_, [this](edm::ProcessBlock const& processBlock, std::shared_ptr<int> const& previousCache) {
              auto returnValue = std::make_shared<int>(0);
              *returnValue += processBlock.get(getTokenBegin_).value;
              *returnValue += processBlock.get(getTokenEnd_).value;
              ++transitions_;
              return returnValue;
            });
        registerProcessBlockCacheFiller<1>(getTokenBegin_,
                                           [this](edm::ProcessBlock const& processBlock,
                                                  std::shared_ptr<TestInputProcessBlockCache> const& previousCache) {
                                             auto returnValue = std::make_shared<TestInputProcessBlockCache>();
                                             returnValue->value_ += processBlock.get(getTokenBegin_).value;
                                             returnValue->value_ += processBlock.get(getTokenEnd_).value;
                                             ++transitions_;
                                             return returnValue;
                                           });
        registerProcessBlockCacheFiller<TestInputProcessBlockCache1>(
            getTokenBegin_,
            [this](edm::ProcessBlock const& processBlock,
                   std::shared_ptr<TestInputProcessBlockCache1> const& previousCache) {
              auto returnValue = std::make_shared<TestInputProcessBlockCache1>();
              returnValue->value_ += processBlock.get(getTokenBegin_).value;
              returnValue->value_ += processBlock.get(getTokenEnd_).value;
              ++transitions_;
              return returnValue;
            });
      }

      void accessInputProcessBlock(edm::ProcessBlock const& processBlock) override {
        if (processBlock.processName() == "PROD1") {
          sum_ += processBlock.get(getTokenBegin_).value;
          sum_ += processBlock.get(getTokenEnd_).value;
        }
        if (processBlock.processName() == "MERGE") {
          sum_ += processBlock.get(getTokenBeginM_).value;
          sum_ += processBlock.get(getTokenEndM_).value;
        }
        ++transitions_;
      }

      void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const override {
        auto cacheTuple = processBlockCaches(event);
        if (!expectedByRun_.empty()) {
          if (expectedByRun_.at(event.run() - 1) != *std::get<edm::CacheHandle<int>>(cacheTuple)) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntAnalyzer::analyze cached value was "
                << *std::get<edm::CacheHandle<int>>(cacheTuple) << " but it was supposed to be "
                << expectedByRun_.at(event.run() - 1);
          }
          if (expectedByRun_.at(event.run() - 1) != std::get<1>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntAnalyzer::analyze second cached value was " << std::get<1>(cacheTuple)->value_
                << " but it was supposed to be " << expectedByRun_.at(event.run() - 1);
          }
          if (expectedByRun_.at(event.run() - 1) !=
              std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntAnalyzer::analyze third cached value was "
                << std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_
                << " but it was supposed to be " << expectedByRun_.at(event.run() - 1);
          }
        }
        ++transitions_;
      }

      void endJob() override {
        if (transitions_ != expectedTransitions_) {
          throw cms::Exception("transitions") << "InputProcessBlockIntAnalyzer transitions " << transitions_
                                              << " but it was supposed to be " << expectedTransitions_;
        }
        if (sum_ != expectedSum_) {
          throw cms::Exception("UnexpectedValue")
              << "InputProcessBlockIntAnalyzer sum " << sum_ << " but it was supposed to be " << expectedSum_;
        }
        if (cacheSize() > 0u) {
          throw cms::Exception("UnexpectedValue")
              << "InputProcessBlockIntAnalyzer cache size not zero at endJob " << cacheSize();
        }
      }

    private:
      edm::EDGetTokenT<IntProduct> getTokenBegin_;
      edm::EDGetTokenT<IntProduct> getTokenEnd_;
      edm::EDGetTokenT<IntProduct> getTokenBeginM_;
      edm::EDGetTokenT<IntProduct> getTokenEndM_;
      mutable std::atomic<unsigned int> transitions_{0};
      int sum_{0};
      unsigned int expectedTransitions_{0};
      std::vector<int> expectedByRun_;
      int expectedSum_{0};
    };

    class InputProcessBlockAnalyzerThreeTags
        : public edm::global::EDAnalyzer<
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>> {
    public:
      explicit InputProcessBlockAnalyzerThreeTags(edm::ParameterSet const& pset) {
        expectedTransitions_ = pset.getParameter<int>("transitions");
        expectedByRun0_ = pset.getParameter<std::vector<int>>("expectedByRun0");
        expectedByRun1_ = pset.getParameter<std::vector<int>>("expectedByRun1");
        expectedByRun2_ = pset.getParameter<std::vector<int>>("expectedByRun2");
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock0");
          if (not tag.label().empty()) {
            getTokenBegin0_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock0");
          if (not tag.label().empty()) {
            getTokenEnd0_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock1");
          if (not tag.label().empty()) {
            getTokenBegin1_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock1");
          if (not tag.label().empty()) {
            getTokenEnd1_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock2");
          if (not tag.label().empty()) {
            getTokenBegin2_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock2");
          if (not tag.label().empty()) {
            getTokenEnd2_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        registerProcessBlockCacheFiller<int>(
            getTokenBegin0_, [this](edm::ProcessBlock const& processBlock, std::shared_ptr<int> const& previousCache) {
              auto returnValue = std::make_shared<int>(0);
              *returnValue += processBlock.get(getTokenBegin0_).value;
              *returnValue += processBlock.get(getTokenEnd0_).value;
              ++transitions_;
              return returnValue;
            });
        registerProcessBlockCacheFiller<1>(getTokenBegin1_,
                                           [this](edm::ProcessBlock const& processBlock,
                                                  std::shared_ptr<TestInputProcessBlockCache> const& previousCache) {
                                             auto returnValue = std::make_shared<TestInputProcessBlockCache>();
                                             returnValue->value_ += processBlock.get(getTokenBegin1_).value;
                                             returnValue->value_ += processBlock.get(getTokenEnd1_).value;
                                             ++transitions_;
                                             return returnValue;
                                           });
        registerProcessBlockCacheFiller<TestInputProcessBlockCache1>(
            getTokenBegin2_,
            [this](edm::ProcessBlock const& processBlock,
                   std::shared_ptr<TestInputProcessBlockCache1> const& previousCache) {
              auto returnValue = std::make_shared<TestInputProcessBlockCache1>();
              returnValue->value_ += processBlock.get(getTokenBegin2_).value;
              returnValue->value_ += processBlock.get(getTokenEnd2_).value;
              ++transitions_;
              return returnValue;
            });
      }

      void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const override {
        auto cacheTuple = processBlockCaches(event);
        if (expectedByRun0_.empty()) {
          if (std::get<edm::CacheHandle<int>>(cacheTuple).isValid()) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockAnalyzerThreeTags::analyze expected invalid CacheHandle for cache 0";
          }
        } else {
          if (expectedByRun0_.at(event.run() - 1) != *std::get<edm::CacheHandle<int>>(cacheTuple)) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockAnalyzerThreeTags::analyze zeroth cached value was "
                << *std::get<edm::CacheHandle<int>>(cacheTuple) << " but it was supposed to be "
                << expectedByRun0_.at(event.run() - 1);
          }
        }
        if (expectedByRun1_.empty()) {
          if (std::get<1>(cacheTuple).isValid()) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockAnalyzerThreeTags::analyze expected invalid CacheHandle for cache 1";
          }
        } else {
          if (expectedByRun1_.at(event.run() - 1) != std::get<1>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockAnalyzerThreeTags::analyze first cached value was "
                << std::get<1>(cacheTuple)->value_ << " but it was supposed to be "
                << expectedByRun1_.at(event.run() - 1);
          }
        }
        if (expectedByRun2_.empty()) {
          if (std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple).isValid()) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockAnalyzerThreeTags::analyze expected invalid CacheHandle for cache 2";
          }
        } else {
          if (expectedByRun2_.at(event.run() - 1) !=
              std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockAnalyzerThreeTags::analyze second cached value was "
                << std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_
                << " but it was supposed to be " << expectedByRun2_.at(event.run() - 1);
          }
        }
        ++transitions_;
      }

      void endJob() override {
        if (transitions_ != expectedTransitions_) {
          throw cms::Exception("transitions") << "InputProcessBlockAnalyzerThreeTags transitions " << transitions_
                                              << " but it was supposed to be " << expectedTransitions_;
        }
        if (cacheSize() > 0u) {
          throw cms::Exception("UnexpectedValue")
              << "InputProcessBlockAnalyzerThreeTags cache size not zero at endJob " << cacheSize();
        }
      }

    private:
      edm::EDGetTokenT<IntProduct> getTokenBegin0_;
      edm::EDGetTokenT<IntProduct> getTokenEnd0_;
      edm::EDGetTokenT<IntProduct> getTokenBegin1_;
      edm::EDGetTokenT<IntProduct> getTokenEnd1_;
      edm::EDGetTokenT<IntProduct> getTokenBegin2_;
      edm::EDGetTokenT<IntProduct> getTokenEnd2_;
      mutable std::atomic<unsigned int> transitions_{0};
      unsigned int expectedTransitions_{0};
      std::vector<int> expectedByRun0_;
      std::vector<int> expectedByRun1_;
      std::vector<int> expectedByRun2_;
    };

    class InputProcessBlockAnalyzerReuseCache
        : public edm::global::EDAnalyzer<
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>> {
    public:
      explicit InputProcessBlockAnalyzerReuseCache(edm::ParameterSet const& pset) {
        expectedTransitions_ = pset.getParameter<int>("transitions");
        expectedByRun_ = pset.getParameter<std::vector<int>>("expectedByRun");
        {
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
              ++transitions_;
              auto returnValue = std::make_shared<TestInputProcessBlockCache1>();
              if (previousCache) {
                returnValue = previousCache;
                return returnValue;
              }
              returnValue->value_ += processBlock.get(getTokenBegin_).value;
              returnValue->value_ += processBlock.get(getTokenEnd_).value;
              return returnValue;
            });
      }

      void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const override {
        auto cacheTuple = processBlockCaches(event);
        if (!expectedByRun_.empty()) {
          if (expectedByRun_.at(event.run() - 1) !=
              std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockAnalyzerReuseCache::analyze cached value was "
                << std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_
                << " but it was supposed to be " << expectedByRun_.at(event.run() - 1);
          }
        }
        ++transitions_;
      }

      void endJob() override {
        if (transitions_ != expectedTransitions_) {
          throw cms::Exception("transitions") << "InputProcessBlockAnalyzerReuseCache transitions " << transitions_
                                              << " but it was supposed to be " << expectedTransitions_;
        }
        if (cacheSize() > 0u) {
          throw cms::Exception("UnexpectedValue")
              << "InputProcessBlockAnalyzerReuseCache cache size not zero at endJob " << cacheSize();
        }
      }

    private:
      edm::EDGetTokenT<IntProduct> getTokenBegin_;
      edm::EDGetTokenT<IntProduct> getTokenEnd_;
      mutable std::atomic<unsigned int> transitions_{0};
      unsigned int expectedTransitions_{0};
      std::vector<int> expectedByRun_;
    };

    class InputProcessBlockIntAnalyzerNoRegistration
        : public edm::global::EDAnalyzer<
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>> {
    public:
      explicit InputProcessBlockIntAnalyzerNoRegistration(edm::ParameterSet const& pset) {
        expectedTransitions_ = pset.getParameter<int>("transitions");
      }

      void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const override {
        auto cacheTuple = processBlockCaches(event);
        ++transitions_;
        if (std::get<0>(cacheTuple).isValid() || std::get<1>(cacheTuple).isValid() ||
            std::get<2>(cacheTuple).isValid()) {
          throw cms::Exception("LogicError")
              << "InputProcessBlockIntAnalyzerNoRegistration expected cacheTuple full of invalid CacheHandles";
        }
      }

      void endJob() override {
        if (transitions_ != expectedTransitions_) {
          throw cms::Exception("transitions") << "InputProcessBlockIntAnalyzerNoRegistration transitions "
                                              << transitions_ << " but it was supposed to be " << expectedTransitions_;
        }
      }

    private:
      mutable std::atomic<unsigned int> transitions_{0};
      unsigned int expectedTransitions_{0};
    };

  }  // namespace global
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::global::StreamIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::RunIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::LumiIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::RunSummaryIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::LumiSummaryIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::ProcessBlockIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::InputProcessBlockIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::global::InputProcessBlockAnalyzerThreeTags);
DEFINE_FWK_MODULE(edmtest::global::InputProcessBlockAnalyzerReuseCache);
DEFINE_FWK_MODULE(edmtest::global::InputProcessBlockIntAnalyzerNoRegistration);
