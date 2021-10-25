
/*----------------------------------------------------------------------

Toy edm::one::EDAnalyzer modules of
edm::one cache templates
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
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ProcessBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/BranchType.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edmtest {
  namespace one {

    class SharedResourcesAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
    public:
      explicit SharedResourcesAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        usesResource();
        callWhenNewProductsRegistered([](edm::BranchDescription const& desc) {
          std::cout << "one::SharedResourcesAnalyzer " << desc.moduleLabel() << std::endl;
        });
      }
      const unsigned int trans_;
      unsigned int m_count = 0;

      void analyze(edm::Event const&, edm::EventSetup const&) override { ++m_count; }

      ~SharedResourcesAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "SharedResourcesAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class WatchRunsAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
    public:
      explicit WatchRunsAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {}
      bool br = false;
      bool er = false;
      const unsigned int trans_;
      unsigned int m_count = 0;

      void analyze(edm::Event const&, edm::EventSetup const&) override {
        ++m_count;
        if (!br || er) {
          throw cms::Exception("out of sequence") << " produce before beginRun or after endRun";
        }
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

      ~WatchRunsAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchRunsAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class WatchLumiBlocksAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchLuminosityBlocks> {
    public:
      explicit WatchLumiBlocksAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {
        // just to create a data dependence
        auto const& tag = p.getParameter<edm::InputTag>("moduleLabel");
        if (not tag.label().empty()) {
          consumes<unsigned int, edm::InLumi>(tag);
        }
      }
      const unsigned int trans_;
      bool bl = false;
      bool el = false;
      unsigned int m_count = 0;

      void analyze(edm::Event const&, edm::EventSetup const&) override {
        ++m_count;
        if (!bl || el) {
          throw cms::Exception("out of sequence") << " produce before beginLumiBlock or after endLumiBlock";
        }
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

      ~WatchLumiBlocksAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchLumiBlocksAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    namespace an {
      struct Cache {
        bool begin = true;
        bool end = false;
      };
    }  // namespace an
    class RunCacheAnalyzer : public edm::one::EDAnalyzer<edm::RunCache<an::Cache>> {
    public:
      explicit RunCacheAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {}
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count = 0;

      void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {
        ++m_count;
        auto c = runCache(iEvent.getRun().index());
        if (nullptr == c) {
          throw cms::Exception("Missing cache") << " no cache in analyze";
        }

        if (!c->begin || c->end) {
          throw cms::Exception("out of sequence") << " produce before beginRun or after endRun";
        }
      }

      std::shared_ptr<an::Cache> globalBeginRun(edm::Run const&, edm::EventSetup const&) const final {
        ++m_count;
        return std::make_shared<an::Cache>();
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

      ~RunCacheAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchRunsAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class LumiBlockCacheAnalyzer : public edm::one::EDAnalyzer<edm::LuminosityBlockCache<an::Cache>> {
    public:
      explicit LumiBlockCacheAnalyzer(edm::ParameterSet const& p) : trans_(p.getParameter<int>("transitions")) {}
      const unsigned int trans_;
      mutable std::atomic<unsigned int> m_count = 0;

      void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {
        ++m_count;

        auto c = luminosityBlockCache(iEvent.getLuminosityBlock().index());
        if (nullptr == c) {
          throw cms::Exception("Missing cache") << " no cache in analyze";
        }

        if (!c->begin || c->end) {
          throw cms::Exception("out of sequence") << " produce before beginLumiBlock or after endLumiBlock";
        }
      }

      std::shared_ptr<an::Cache> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                            edm::EventSetup const&) const final {
        ++m_count;
        return std::make_shared<an::Cache>();
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

      ~LumiBlockCacheAnalyzer() {
        if (m_count != trans_) {
          throw cms::Exception("transitions")
              << "WatchLumiBlocksAnalyzer transitions " << m_count << " but it was supposed to be " << trans_;
        }
      }
    };

    class ProcessBlockIntAnalyzer : public edm::one::EDAnalyzer<edm::WatchProcessBlock> {
    public:
      explicit ProcessBlockIntAnalyzer(edm::ParameterSet const& pset) : trans_(pset.getParameter<int>("transitions")) {
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

      void analyze(edm::Event const&, edm::EventSetup const&) override {
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
        : public edm::one::EDAnalyzer<
              edm::WatchProcessBlock,
              edm::InputProcessBlockCache<int, TestInputProcessBlockCache, TestInputProcessBlockCache1>> {
    public:
      explicit InputProcessBlockIntAnalyzer(edm::ParameterSet const& pset) {
        expectedTransitions_ = pset.getParameter<int>("transitions");
        expectedByRun_ = pset.getParameter<std::vector<int>>("expectedByRun");
        expectedSum_ = pset.getParameter<int>("expectedSum");
        expectedFillerSum_ = pset.getUntrackedParameter<int>("expectedFillerSum", 0);
        expectedCacheSize_ = pset.getUntrackedParameter<unsigned int>("expectedCacheSize", 0);
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
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesBeginProcessBlockNotFound");
          if (not tag.label().empty()) {
            getTokenBeginNotFound_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesEndProcessBlockNotFound");
          if (not tag.label().empty()) {
            getTokenEndNotFound_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesProcessBlockNotFound1");
          if (not tag.label().empty()) {
            getTokenNotFound1_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesProcessBlockNotFound2");
          if (not tag.label().empty()) {
            getTokenNotFound2_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesProcessBlockNotFound3");
          if (not tag.label().empty()) {
            getTokenNotFound3_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }
        {
          auto tag = pset.getParameter<edm::InputTag>("consumesProcessBlockNotFound4");
          if (not tag.label().empty()) {
            getTokenNotFound4_ = consumes<IntProduct, edm::InProcess>(tag);
          }
        }

        if (!getTokenBegin_.isUninitialized()) {
          registerProcessBlockCacheFiller<int>(
              getTokenBegin_, [this](edm::ProcessBlock const& processBlock, std::shared_ptr<int> const& previousCache) {
                auto returnValue = std::make_shared<int>(0);
                *returnValue += processBlock.get(getTokenBegin_).value;
                *returnValue += processBlock.get(getTokenEnd_).value;
                fillerSum_ += processBlock.get(getTokenBegin_).value;
                fillerSum_ += processBlock.get(getTokenEnd_).value;
                ++transitions_;
                return returnValue;
              });
        }
        if (!getTokenBegin_.isUninitialized()) {
          registerProcessBlockCacheFiller<1>(getTokenBegin_,
                                             [this](edm::ProcessBlock const& processBlock,
                                                    std::shared_ptr<TestInputProcessBlockCache> const& previousCache) {
                                               auto returnValue = std::make_shared<TestInputProcessBlockCache>();
                                               returnValue->value_ += processBlock.get(getTokenBegin_).value;
                                               returnValue->value_ += processBlock.get(getTokenEnd_).value;
                                               fillerSum_ += processBlock.get(getTokenBegin_).value;
                                               fillerSum_ += processBlock.get(getTokenEnd_).value;
                                               ++transitions_;
                                               return returnValue;
                                             });
        }
        if (!getTokenBegin_.isUninitialized()) {
          registerProcessBlockCacheFiller<TestInputProcessBlockCache1>(
              getTokenBegin_,
              [this](edm::ProcessBlock const& processBlock,
                     std::shared_ptr<TestInputProcessBlockCache1> const& previousCache) {
                auto returnValue = std::make_shared<TestInputProcessBlockCache1>();
                returnValue->value_ += processBlock.get(getTokenBegin_).value;
                returnValue->value_ += processBlock.get(getTokenEnd_).value;
                fillerSum_ += processBlock.get(getTokenBegin_).value;
                fillerSum_ += processBlock.get(getTokenEnd_).value;
                ++transitions_;
                return returnValue;
              });
        }
      }

      void beginProcessBlock(edm::ProcessBlock const& processBlock) override {
        if (!getTokenBeginNotFound_.isUninitialized() && processBlock.getHandle(getTokenBeginNotFound_).isValid()) {
          throw cms::Exception("TestFailure") << "Expected handle to be invalid but it is valid (begin)";
        }
      }

      void endProcessBlock(edm::ProcessBlock const& processBlock) override {
        if (!getTokenEndNotFound_.isUninitialized() && processBlock.getHandle(getTokenEndNotFound_).isValid()) {
          throw cms::Exception("TestFailure") << "Expected handle to be invalid but it is valid (end)";
        }
      }

      void accessInputProcessBlock(edm::ProcessBlock const& processBlock) override {
        if (processBlock.processName() == "PROD1") {
          if (!getTokenBegin_.isUninitialized() && processBlock.getHandle(getTokenBegin_).isValid()) {
            sum_ += processBlock.get(getTokenBegin_).value;
            sum_ += processBlock.get(getTokenEnd_).value;
          }
        }
        if (processBlock.processName() == "MERGE") {
          if (!getTokenBeginM_.isUninitialized() && processBlock.getHandle(getTokenBeginM_).isValid()) {
            sum_ += processBlock.get(getTokenBeginM_).value;
            sum_ += processBlock.get(getTokenEndM_).value;
          }
        }

        if (!getTokenNotFound1_.isUninitialized() && processBlock.getHandle(getTokenNotFound1_).isValid()) {
          throw cms::Exception("TestFailure") << "Expected handle to be invalid but it is valid (token 1)";
        }
        if (!getTokenNotFound2_.isUninitialized() && processBlock.getHandle(getTokenNotFound2_).isValid()) {
          throw cms::Exception("TestFailure") << "Expected handle to be invalid but it is valid (token 2)";
        }
        if (!getTokenNotFound3_.isUninitialized() && processBlock.getHandle(getTokenNotFound3_).isValid()) {
          throw cms::Exception("TestFailure") << "Expected handle to be invalid but it is valid (token 3)";
        }
        if (!getTokenNotFound4_.isUninitialized() && processBlock.getHandle(getTokenNotFound4_).isValid()) {
          throw cms::Exception("TestFailure") << "Expected handle to be invalid but it is valid (token 4)";
        }

        ++transitions_;
      }

      void analyze(edm::Event const& event, edm::EventSetup const&) override {
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

        if (expectedCacheSize_ != 0u && expectedCacheSize_ != cacheSize()) {
          throw cms::Exception("UnexpectedValue") << "InputProcessBlockIntAnalyzer::analyze, unexpected cacheSize "
                                                  << cacheSize() << " but it was supposed to be " << expectedCacheSize_;
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
        if (expectedFillerSum_ != 0 && fillerSum_ != expectedFillerSum_) {
          throw cms::Exception("UnexpectedValue") << "InputProcessBlockIntAnalyzer fillerSum " << fillerSum_
                                                  << " but it was supposed to be " << expectedFillerSum_;
        }
        if (cacheSize() > 0u) {
          throw cms::Exception("UnexpectedValue")
              << "InputProcessBlockIntAnalyzer cache size not zero at endJob " << cacheSize();
        }
      }

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        edm::ParameterSetDescription desc;
        desc.add<int>("transitions");
        desc.add<std::vector<int>>("expectedByRun", std::vector<int>());
        desc.add<int>("expectedSum");
        edm::InputTag defaultInputTag;
        desc.addUntracked<int>("expectedFillerSum", 0);
        desc.addUntracked<unsigned int>("expectedCacheSize", 0);
        desc.add<edm::InputTag>("consumesBeginProcessBlock", defaultInputTag);
        desc.add<edm::InputTag>("consumesEndProcessBlock", defaultInputTag);
        desc.add<edm::InputTag>("consumesBeginProcessBlockM", defaultInputTag);
        desc.add<edm::InputTag>("consumesEndProcessBlockM", defaultInputTag);
        desc.add<edm::InputTag>("consumesBeginProcessBlockNotFound", defaultInputTag);
        desc.add<edm::InputTag>("consumesEndProcessBlockNotFound", defaultInputTag);
        desc.add<edm::InputTag>("consumesProcessBlockNotFound1", defaultInputTag);
        desc.add<edm::InputTag>("consumesProcessBlockNotFound2", defaultInputTag);
        desc.add<edm::InputTag>("consumesProcessBlockNotFound3", defaultInputTag);
        desc.add<edm::InputTag>("consumesProcessBlockNotFound4", defaultInputTag);
        descriptions.addDefault(desc);
      }

    private:
      edm::EDGetTokenT<IntProduct> getTokenBegin_;
      edm::EDGetTokenT<IntProduct> getTokenEnd_;
      edm::EDGetTokenT<IntProduct> getTokenBeginM_;
      edm::EDGetTokenT<IntProduct> getTokenEndM_;
      edm::EDGetTokenT<IntProduct> getTokenBeginNotFound_;
      edm::EDGetTokenT<IntProduct> getTokenEndNotFound_;
      edm::EDGetTokenT<IntProduct> getTokenNotFound1_;
      edm::EDGetTokenT<IntProduct> getTokenNotFound2_;
      edm::EDGetTokenT<IntProduct> getTokenNotFound3_;
      edm::EDGetTokenT<IntProduct> getTokenNotFound4_;
      mutable std::atomic<unsigned int> transitions_{0};
      int sum_{0};
      unsigned int expectedTransitions_{0};
      std::vector<int> expectedByRun_;
      int expectedSum_{0};
      int fillerSum_{0};
      int expectedFillerSum_{0};
      unsigned int expectedCacheSize_{0};
    };

  }  // namespace one
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::one::SharedResourcesAnalyzer);
DEFINE_FWK_MODULE(edmtest::one::WatchRunsAnalyzer);
DEFINE_FWK_MODULE(edmtest::one::WatchLumiBlocksAnalyzer);
DEFINE_FWK_MODULE(edmtest::one::RunCacheAnalyzer);
DEFINE_FWK_MODULE(edmtest::one::LumiBlockCacheAnalyzer);
DEFINE_FWK_MODULE(edmtest::one::ProcessBlockIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::one::InputProcessBlockIntAnalyzer);
