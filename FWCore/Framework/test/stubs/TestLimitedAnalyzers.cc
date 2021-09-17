
/*----------------------------------------------------------------------

Toy edm::limited::EDAnalyzer modules of
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
#include "FWCore/Framework/interface/limited/EDAnalyzer.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edmtest {
  namespace limited {

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

    class StreamIntAnalyzer : public edm::limited::EDAnalyzer<edm::StreamCache<UnsafeCache>> {
    public:
      explicit StreamIntAnalyzer(edm::ParameterSet const& p)
          : edm::limited::EDAnalyzerBase(p),
            edm::limited::EDAnalyzer<edm::StreamCache<UnsafeCache>>(p),
            trans_(p.getParameter<int>("transitions")) {
        callWhenNewProductsRegistered([](edm::BranchDescription const& desc) {
          std::cout << "limited::StreamIntAnalyzer " << desc.moduleLabel() << std::endl;
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

      ~StreamIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "StreamIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunIntAnalyzer : public edm::limited::EDAnalyzer<edm::RunCache<Cache>> {
    public:
      explicit RunIntAnalyzer(edm::ParameterSet const& p)
          : edm::limited::EDAnalyzerBase(p),
            edm::limited::EDAnalyzer<edm::RunCache<Cache>>(p),
            trans_(p.getParameter<int>("transitions")),
            cvalue_(p.getParameter<int>("cachevalue")) {}
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

      ~RunIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "RunIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiIntAnalyzer : public edm::limited::EDAnalyzer<edm::LuminosityBlockCache<Cache>> {
    public:
      explicit LumiIntAnalyzer(edm::ParameterSet const& p)
          : edm::limited::EDAnalyzerBase(p),
            edm::limited::EDAnalyzer<edm::LuminosityBlockCache<Cache>>(p),
            trans_(p.getParameter<int>("transitions")),
            cvalue_(p.getParameter<int>("cachevalue")) {
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

      ~LumiIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "LumiIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class RunSummaryIntAnalyzer
        : public edm::limited::EDAnalyzer<edm::StreamCache<UnsafeCache>, edm::RunSummaryCache<UnsafeCache>> {
    public:
      explicit RunSummaryIntAnalyzer(edm::ParameterSet const& p)
          : edm::limited::EDAnalyzerBase(p),
            edm::limited::EDAnalyzer<edm::StreamCache<UnsafeCache>, edm::RunSummaryCache<UnsafeCache>>(p),
            trans_(p.getParameter<int>("transitions")),
            cvalue_(p.getParameter<int>("cachevalue")) {}
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

      ~RunSummaryIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "RunSummaryIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiSummaryIntAnalyzer
        : public edm::limited::EDAnalyzer<edm::StreamCache<UnsafeCache>, edm::LuminosityBlockSummaryCache<UnsafeCache>> {
    public:
      explicit LumiSummaryIntAnalyzer(edm::ParameterSet const& p)
          : edm::limited::EDAnalyzerBase(p),
            edm::limited::EDAnalyzer<edm::StreamCache<UnsafeCache>, edm::LuminosityBlockSummaryCache<UnsafeCache>>(p),
            trans_(p.getParameter<int>("transitions")),
            cvalue_(p.getParameter<int>("cachevalue")) {}
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

      ~LumiSummaryIntAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "LumiSummaryIntAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class ProcessBlockIntAnalyzer : public edm::limited::EDAnalyzer<edm::WatchProcessBlock> {
    public:
      explicit ProcessBlockIntAnalyzer(edm::ParameterSet const& pset)
          : edm::limited::EDAnalyzerBase(pset),
            edm::limited::EDAnalyzer<edm::WatchProcessBlock>(pset),
            trans_(pset.getParameter<int>("transitions")) {
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock");
          if (not tag.label().empty()) {
            getTokenBegin_ = consumes<unsigned int, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock");
          if (not tag.label().empty()) {
            getTokenEnd_ = consumes<unsigned int, edm::InProcess>(tag);
          }
        }
      }

      void beginProcessBlock(edm::ProcessBlock const& processBlock) override {
        if (m_count != 0) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntAnalyzer::begin transitions " << m_count << " but it was supposed to be " << 0;
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

      void analyze(edm::StreamID iID, edm::Event const&, edm::EventSetup const&) const override {
        if (m_count < 1u) {
          throw cms::Exception("out of sequence") << "analyze before beginProcessBlock " << m_count;
        }
        ++m_count;
      }

      void endProcessBlock(edm::ProcessBlock const& processBlock) override {
        ++m_count;
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntAnalyzer::end transitions " << m_count << " but it was supposed to be " << trans_;
        }
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
        : public edm::limited::EDAnalyzer<
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>> {
    public:
      explicit InputProcessBlockIntAnalyzer(edm::ParameterSet const& pset)
          : edm::limited::EDAnalyzerBase(pset),
            edm::limited::EDAnalyzer<
                edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>>(pset) {
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

  }  // namespace limited
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::limited::StreamIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::limited::RunIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::limited::LumiIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::limited::RunSummaryIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::limited::LumiSummaryIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::limited::ProcessBlockIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::limited::InputProcessBlockIntAnalyzer);
