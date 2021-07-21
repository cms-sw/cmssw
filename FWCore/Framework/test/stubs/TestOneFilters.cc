
/*----------------------------------------------------------------------

Toy edm::one::EDFilter modules of
edm::one cache classes and edm::*Producer classes
for testing purposes only.

----------------------------------------------------------------------*/

#include <atomic>
#include <memory>
#include <tuple>
#include <vector>

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/CacheHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edmtest {
  namespace one {

    class SharedResourcesFilter : public edm::one::EDFilter<edm::one::SharedResources> {
    public:
      explicit SharedResourcesFilter(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<int>();
        usesResource();
      }
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};
      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        return true;
      }

      ~SharedResourcesFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "SharedResourcesFilter transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class WatchRunsFilter : public edm::one::EDFilter<edm::one::WatchRuns> {
    public:
      explicit WatchRunsFilter(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {}
      bool br = false;
      bool er = false;
      const unsigned int trans_;
      unsigned int m_count = 0;

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        if (!br || er) {
          throw cms::Exception("out of sequence") << " produce before beginRun or after endRun";
        }
        return true;
      }

      void beginRun(edm::Run const&, edm::EventSetup const&) override {
        ++m_count;
        if (br) {
          throw cms::Exception("out of sequence") << " beginRun seen multiple times";
        }
        br = true;
        er = false;
      }

      void endRun(edm::Run const&, edm::EventSetup const&) override {
        ++m_count;
        if (!br) {
          throw cms::Exception("out of sequence") << " endRun before beginRun";
        }
        br = false;
        er = true;
      }

      ~WatchRunsFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchRunsFilter transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class WatchLumiBlocksFilter : public edm::one::EDFilter<edm::one::WatchLuminosityBlocks> {
    public:
      explicit WatchLumiBlocksFilter(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {}
      const unsigned int trans_;
      bool bl = false;
      bool el = false;
      unsigned int m_count = 0;

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        if (!bl || el) {
          throw cms::Exception("out of sequence") << " produce before beginLumiBlock or after endLumiBlock";
        }
        return true;
      }

      void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
        ++m_count;
        if (bl) {
          throw cms::Exception("out of sequence") << " beginLumiBlock seen mutiple times";
        }
        bl = true;
        el = false;
      }

      void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {
        ++m_count;
        if (!bl) {
          throw cms::Exception("out of sequence") << " endLumiBlock before beginLumiBlock";
        }
        bl = false;
        el = true;
      }

      ~WatchLumiBlocksFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchLumiBlocksFilter transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    namespace fltr {
      struct Cache {
        bool begin = true;
        bool end = false;
      };
    }  // namespace fltr
    class RunCacheFilter : public edm::one::EDFilter<edm::RunCache<fltr::Cache>> {
    public:
      explicit RunCacheFilter(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {}
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count = 0;

      bool filter(edm::Event& iEvent, edm::EventSetup const&) override {
        ++m_count;
        auto c = runCache(iEvent.getRun().index());
        if (nullptr == c) {
          throw cms::Exception("Missing cache") << " no cache in analyze";
        }

        if (!c->begin || c->end) {
          throw cms::Exception("out of sequence") << " produce before beginRun or after endRun";
        }

        return true;
      }

      std::shared_ptr<fltr::Cache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const final {
        ++m_count;
        return std::make_shared<fltr::Cache>();
      }

      void globalEndRun(edm::Run const& iRun, edm::EventSetup const&) final {
        ++m_count;
        auto c = runCache(iRun.index());
        if (nullptr == c) {
          throw cms::Exception("Missing cache") << " no cache in globalEndRun";
        }
        if (!c->begin) {
          throw cms::Exception("out of sequence") << " endRun before beginRun";
        }
        c->begin = false;
        c->end = true;
      }

      ~RunCacheFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchRunsAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiBlockCacheFilter : public edm::one::EDFilter<edm::LuminosityBlockCache<fltr::Cache>> {
    public:
      explicit LumiBlockCacheFilter(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {}
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count = 0;

      bool filter(edm::Event& iEvent, edm::EventSetup const&) override {
        ++m_count;

        auto c = luminosityBlockCache(iEvent.getLuminosityBlock().index());
        if (nullptr == c) {
          throw cms::Exception("Missing cache") << " no cache in analyze";
        }

        if (!c->begin || c->end) {
          throw cms::Exception("out of sequence") << " produce before beginLumiBlock or after endLumiBlock";
        }
        return true;
      }

      std::shared_ptr<fltr::Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                              edm::EventSetup const&) const final {
        ++m_count;
        return std::make_shared<fltr::Cache>();
      }

      void globalEndLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) override {
        ++m_count;
        auto c = luminosityBlockCache(iLumi.index());
        if (!c->begin) {
          throw cms::Exception("out of sequence") << " endLumiBlock before beginLumiBlock";
        }
        c->begin = false;
        c->end = true;
      }

      ~LumiBlockCacheFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchLumiBlocksAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class ProcessBlockIntFilter : public edm::one::EDFilter<edm::WatchProcessBlock> {
    public:
      explicit ProcessBlockIntFilter(edm::ParameterSet const& pset) : trans_(pset.getParameter<int>("transitions")) {
        produces<unsigned int>();

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
              << "ProcessBlockIntFilter::begin transitions " << m_count << " but it was supposed to be " << 0;
        }
        ++m_count;

        const unsigned int valueToGet = 31;
        if (not getTokenBegin_.isUninitialized()) {
          if (processBlock.get(getTokenBegin_) != valueToGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToGet << " but got " << processBlock.get(getTokenBegin_);
          }
        }
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        if (m_count < 1u) {
          throw cms::Exception("out of sequence") << "produce before beginProcessBlock " << m_count;
        }
        ++m_count;
        return true;
      }

      void endProcessBlock(edm::ProcessBlock const& processBlock) override {
        ++m_count;
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntFilter::end transitions " << m_count << " but it was supposed to be " << trans_;
        }

        {
          const unsigned int valueToGet = 31;
          if (not getTokenBegin_.isUninitialized()) {
            if (processBlock.get(getTokenBegin_) != valueToGet) {
              throw cms::Exception("BadValue")
                  << "expected " << valueToGet << " but got " << processBlock.get(getTokenBegin_);
            }
          }
        }
        {
          const unsigned int valueToGet = 41;
          if (not getTokenEnd_.isUninitialized()) {
            if (processBlock.get(getTokenEnd_) != valueToGet) {
              throw cms::Exception("BadValue")
                  << "expected " << valueToGet << " but got " << processBlock.get(getTokenEnd_);
            }
          }
        }
      }

      ~ProcessBlockIntFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "ProcessBlockIntFilter transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }

    private:
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};
      edm::EDGetTokenT<unsigned int> getTokenBegin_;
      edm::EDGetTokenT<unsigned int> getTokenEnd_;
    };

    class TestBeginProcessBlockFilter : public edm::one::EDFilter<edm::BeginProcessBlockProducer> {
    public:
      explicit TestBeginProcessBlockFilter(edm::ParameterSet const& pset)
          : trans_(pset.getParameter<int>("transitions")),
            token_(produces<unsigned int, edm::Transition::BeginProcessBlock>("begin")) {
        produces<unsigned int>();

        auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlock");
        if (not tag.label().empty()) {
          getToken_ = consumes<unsigned int, edm::InProcess>(tag);
        }
      }

      void beginProcessBlockProduce(edm::ProcessBlock& processBlock) override {
        if (m_count != 0) {
          throw cms::Exception("transitions")
              << "TestBeginProcessBlockFilter transitions " << m_count << " but it was supposed to be " << 0;
        }
        ++m_count;

        const unsigned int valueToPutAndGet = 31;
        processBlock.emplace(token_, valueToPutAndGet);
        if (not getToken_.isUninitialized()) {
          if (processBlock.get(getToken_) != valueToPutAndGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToPutAndGet << " but got " << processBlock.get(getToken_);
          }
        }
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        if (m_count < 1u) {
          throw cms::Exception("out of sequence") << "produce before beginProcessBlockProduce " << m_count;
        }
        ++m_count;
        return true;
      }

      ~TestBeginProcessBlockFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "TestBeginProcessBlockFilter transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }

    private:
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};
      edm::EDPutTokenT<unsigned int> token_;
      edm::EDGetTokenT<unsigned int> getToken_;
    };

    class TestEndProcessBlockFilter : public edm::one::EDFilter<edm::EndProcessBlockProducer> {
    public:
      explicit TestEndProcessBlockFilter(edm::ParameterSet const& pset)
          : trans_(pset.getParameter<int>("transitions")),
            token_(produces<unsigned int, edm::Transition::EndProcessBlock>("end")) {
        produces<unsigned int>();

        auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlock");
        if (not tag.label().empty()) {
          getToken_ = consumes<unsigned int, edm::InProcess>(tag);
        }
      }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        return true;
      }

      void endProcessBlockProduce(edm::ProcessBlock& processBlock) override {
        ++m_count;
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "TestEndProcessBlockFilter transitions " << m_count << " but it was supposed to be " << trans_;
        }

        const unsigned int valueToPutAndGet = 41;
        processBlock.emplace(token_, valueToPutAndGet);
        if (not getToken_.isUninitialized()) {
          if (processBlock.get(getToken_) != valueToPutAndGet) {
            throw cms::Exception("BadValue")
                << "expected " << valueToPutAndGet << " but got " << processBlock.get(getToken_);
          }
        }
      }

      ~TestEndProcessBlockFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "~TestEndProcessBlockFilter transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }

    private:
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count{0};
      edm::EDPutTokenT<unsigned int> token_;
      edm::EDGetTokenT<unsigned int> getToken_;
    };

    class BeginRunFilter : public edm::one::EDFilter<edm::one::WatchRuns, edm::BeginRunProducer> {
    public:
      explicit BeginRunFilter(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<int>();
        produces<unsigned int, edm::Transition::BeginRun>("a");
      }
      const unsigned int trans_;
      unsigned int m_count = 0;
      bool p = false;

      void beginRun(edm::Run const&, edm::EventSetup const&) override { p = false; }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        p = true;
        return true;
      }

      void beginRunProduce(edm::Run&, edm::EventSetup const&) override {
        if (p) {
          throw cms::Exception("out of sequence") << "produce before beginRunProduce";
        }
        ++m_count;
      }

      void endRun(edm::Run const&, edm::EventSetup const&) override {}

      ~BeginRunFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "BeginRunFilter transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class BeginLumiBlockFilter
        : public edm::one::EDFilter<edm::one::WatchLuminosityBlocks, edm::BeginLuminosityBlockProducer> {
    public:
      explicit BeginLumiBlockFilter(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<unsigned int, edm::Transition::BeginLuminosityBlock>("a");
      }
      const unsigned int trans_;
      unsigned int m_count = 0;
      bool p = false;

      void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override { p = false; }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        p = true;
        return true;
      }

      void beginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) override {
        ++m_count;
        if (p) {
          throw cms::Exception("out of sequence") << "produce before beginLuminosityBlockProduce";
        }
      }

      void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {}

      ~BeginLumiBlockFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "BeginLumiBlockFilter transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class EndRunFilter : public edm::one::EDFilter<edm::one::WatchRuns, edm::EndRunProducer> {
    public:
      explicit EndRunFilter(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<unsigned int, edm::Transition::EndRun>("a");
      }
      const unsigned int trans_;
      bool erp = false;
      unsigned int m_count = 0;

      void beginRun(edm::Run const&, edm::EventSetup const&) override { erp = false; }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        if (erp) {
          throw cms::Exception("out of sequence") << "endRunProduce before produce";
        }
        return true;
      }

      void endRunProduce(edm::Run&, edm::EventSetup const&) override { ++m_count; }

      void endRun(edm::Run const&, edm::EventSetup const&) override {}

      ~EndRunFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "EndRunFilter transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class EndLumiBlockFilter
        : public edm::one::EDFilter<edm::one::WatchLuminosityBlocks, edm::EndLuminosityBlockProducer> {
    public:
      explicit EndLumiBlockFilter(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        produces<unsigned int, edm::Transition::EndLuminosityBlock>("a");
      }
      const unsigned int trans_;
      bool elbp = false;
      unsigned int m_count = 0;

      void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override { elbp = false; }

      bool filter(edm::Event&, edm::EventSetup const&) override {
        ++m_count;
        if (elbp) {
          throw cms::Exception("out of sequence") << "endLumiBlockProduce before produce";
        }
        return true;
      }

      void endLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) override {
        ++m_count;
        elbp = true;
      }

      void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {}

      ~EndLumiBlockFilter() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "EndLumiBlockFilter transitions " << m_count << " but it was supposed to be " << trans_;
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
        : public edm::one::EDFilter<
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>> {
    public:
      explicit InputProcessBlockIntFilter(edm::ParameterSet const& pset) {
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

      bool filter(edm::Event& event, edm::EventSetup const&) override {
        auto cacheTuple = processBlockCaches(event);
        if (!expectedByRun_.empty()) {
          if (expectedByRun_.at(event.run() - 1) != *std::get<edm::CacheHandle<int>>(cacheTuple)) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntFilter::filter cached value was "
                << *std::get<edm::CacheHandle<int>>(cacheTuple) << " but it was supposed to be "
                << expectedByRun_.at(event.run() - 1);
          }
          if (expectedByRun_.at(event.run() - 1) != std::get<1>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntFilter::filter second cached value was " << std::get<1>(cacheTuple)->value_
                << " but it was supposed to be " << expectedByRun_.at(event.run() - 1);
          }
          if (expectedByRun_.at(event.run() - 1) !=
              std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_) {
            throw cms::Exception("UnexpectedValue")
                << "InputProcessBlockIntFilter::filter third cached value was "
                << std::get<edm::CacheHandle<TestInputProcessBlockCache1>>(cacheTuple)->value_
                << " but it was supposed to be " << expectedByRun_.at(event.run() - 1);
          }
        }
        ++transitions_;
        return true;
      }

      void endJob() override {
        if (transitions_ != expectedTransitions_) {
          throw cms::Exception("transitions") << "InputProcessBlockIntFilter transitions " << transitions_
                                              << " but it was supposed to be " << expectedTransitions_;
        }
        if (sum_ != expectedSum_) {
          throw cms::Exception("UnexpectedValue")
              << "InputProcessBlockIntFilter sum " << sum_ << " but it was supposed to be " << expectedSum_;
        }
        if (cacheSize() > 0u) {
          throw cms::Exception("UnexpectedValue")
              << "InputProcessBlockIntFilter cache size not zero at endJob " << cacheSize();
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

  }  // namespace one
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::one::SharedResourcesFilter);
DEFINE_FWK_MODULE(edmtest::one::WatchRunsFilter);
DEFINE_FWK_MODULE(edmtest::one::WatchLumiBlocksFilter);
DEFINE_FWK_MODULE(edmtest::one::RunCacheFilter);
DEFINE_FWK_MODULE(edmtest::one::LumiBlockCacheFilter);
DEFINE_FWK_MODULE(edmtest::one::ProcessBlockIntFilter);
DEFINE_FWK_MODULE(edmtest::one::TestBeginProcessBlockFilter);
DEFINE_FWK_MODULE(edmtest::one::TestEndProcessBlockFilter);
DEFINE_FWK_MODULE(edmtest::one::BeginRunFilter);
DEFINE_FWK_MODULE(edmtest::one::BeginLumiBlockFilter);
DEFINE_FWK_MODULE(edmtest::one::EndRunFilter);
DEFINE_FWK_MODULE(edmtest::one::EndLumiBlockFilter);
DEFINE_FWK_MODULE(edmtest::one::InputProcessBlockIntFilter);
