#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/ESProducerLooper.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LooperFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"

#include "FWCore/Framework/test/DummyData.h"
#include "FWCore/Framework/test/DummyRecord.h"

namespace edmtest {
  class IntTestLooper : public edm::ESProducerLooper {
  public:
    IntTestLooper(edm::ParameterSet const& iPSet)
        : tokensBeginRun_(edm::vector_transform(
              iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginRun"),
              [this](edm::InputTag const& tag) { return this->consumes<IntProduct, edm::InRun>(tag); })),
          tokensBeginLumi_(edm::vector_transform(
              iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("srcBeginLumi"),
              [this](edm::InputTag const& tag) { return this->consumes<IntProduct, edm::InLumi>(tag); })),
          tokensEvent_(
              edm::vector_transform(iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("srcEvent"),
                                    [this](edm::InputTag const& tag) { return this->consumes<IntProduct>(tag); })),
          tokensEndLumi_(edm::vector_transform(
              iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("srcEndLumi"),
              [this](edm::InputTag const& tag) { return this->consumes<IntProduct, edm::InLumi>(tag); })),
          tokensEndRun_(edm::vector_transform(
              iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("srcEndRun"),
              [this](edm::InputTag const& tag) { return this->consumes<IntProduct, edm::InRun>(tag); })),
          esTokenBeginRun_(esConsumes<edm::Transition::BeginRun>()),
          esToken_(esConsumes()),
          valuesBeginRun_(iPSet.getUntrackedParameter<std::vector<int>>("expectBeginRunValues")),
          valuesBeginLumi_(iPSet.getUntrackedParameter<std::vector<int>>("expectBeginLumiValues")),
          valuesEvent_(iPSet.getUntrackedParameter<std::vector<int>>("expectEventValues")),
          valuesEndLumi_(iPSet.getUntrackedParameter<std::vector<int>>("expectEndLumiValues")),
          valuesEndRun_(iPSet.getUntrackedParameter<std::vector<int>>("expectEndRunValues")),
          valueES_(iPSet.getUntrackedParameter<int>("expectESValue")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.setComment(
          "This test EDLooper can consume arbitrary number of IntProduct's at Run, Lumi, and Event transitions. It "
          "always consumes DummyData EventSetup product in beginRun and Event transitions. For all the consumed "
          "products it requires the value of the product to correspond the value in configuratoin. It stops looping "
          "after 3rd time.");
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginRun", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcBeginLumi", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcEvent", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcEndLumi", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<edm::InputTag>>("srcEndRun", std::vector<edm::InputTag>{});
      desc.addUntracked<std::vector<int>>("expectBeginRunValues", std::vector<int>{});
      desc.addUntracked<std::vector<int>>("expectBeginLumiValues", std::vector<int>{});
      desc.addUntracked<std::vector<int>>("expectEventValues", std::vector<int>{});
      desc.addUntracked<std::vector<int>>("expectEndLumiValues", std::vector<int>{});
      desc.addUntracked<std::vector<int>>("expectEndRunValues", std::vector<int>{});
      desc.addUntracked<int>("expectESValue");
      descriptions.addDefault(desc);
    }

    void beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) override {
      for (size_t i = 0; i < tokensBeginRun_.size(); ++i) {
        int const expectedValue = valuesBeginRun_[i];
        int const value = iRun.get(tokensBeginRun_[i]).value;
        if (expectedValue != value) {
          throw cms::Exception("WrongValue")
              << "expected value " << expectedValue << " but got " << value << " for beginRun product " << i;
        }
      }

      auto const& data = iSetup.getData(esTokenBeginRun_);
      if (data.value_ != valueES_) {
        throw cms::Exception("WrongValue")
            << " expected value " << valueES_ << " but got " << data.value_ << " for EventSetup product in beginRun";
      }
    }

    void beginLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) override {
      for (size_t i = 0; i < tokensBeginLumi_.size(); ++i) {
        int const expectedValue = valuesBeginLumi_[i];
        int const value = iLumi.get(tokensBeginLumi_[i]).value;
        if (expectedValue != value) {
          throw cms::Exception("WrongValue")
              << "expected value " << expectedValue << " but got " << value << " for beginLumi product " << i;
        }
      }
    }

    void endLuminosityBlock(edm::LuminosityBlock const& iLumi, edm::EventSetup const&) override {
      for (size_t i = 0; i < tokensEndLumi_.size(); ++i) {
        int const expectedValue = valuesEndLumi_[i];
        int const value = iLumi.get(tokensEndLumi_[i]).value;
        if (expectedValue != value) {
          throw cms::Exception("WrongValue")
              << "expected value " << expectedValue << " but got " << value << " for endLumi product " << i;
        }
      }
    }

    void endRun(edm::Run const& iRun, edm::EventSetup const&) override {
      for (size_t i = 0; i < tokensEndRun_.size(); ++i) {
        int const expectedValue = valuesEndRun_[i];
        int const value = iRun.get(tokensEndRun_[i]).value;
        if (expectedValue != value) {
          throw cms::Exception("WrongValue")
              << "expected value " << expectedValue << " but got " << value << " for endRun product " << i;
        }
      }
    }

    void startingNewLoop(unsigned int) override {}

    Status duringLoop(edm::Event const& iEvent, edm::EventSetup const& iSetup) override {
      for (size_t i = 0; i < tokensEvent_.size(); ++i) {
        int const expectedValue = valuesEvent_[i];
        int const value = iEvent.get(tokensEvent_[i]).value;
        if (expectedValue != value) {
          throw cms::Exception("WrongValue")
              << "expected value " << expectedValue << " but got " << value << " for Event product " << i;
        }
      }

      auto const& data = iSetup.getData(esToken_);
      if (data.value_ != valueES_) {
        throw cms::Exception("WrongValue")
            << " expected value " << valueES_ << " but got " << data.value_ << " for EventSetup product in duringLoop";
      }

      return kContinue;
    }

    Status endOfLoop(edm::EventSetup const&, unsigned int iCount) override { return iCount == 2 ? kStop : kContinue; }

  private:
    const std::vector<edm::EDGetTokenT<IntProduct>> tokensBeginRun_;
    const std::vector<edm::EDGetTokenT<IntProduct>> tokensBeginLumi_;
    const std::vector<edm::EDGetTokenT<IntProduct>> tokensEvent_;
    const std::vector<edm::EDGetTokenT<IntProduct>> tokensEndLumi_;
    const std::vector<edm::EDGetTokenT<IntProduct>> tokensEndRun_;
    edm::ESGetToken<edm::eventsetup::test::DummyData, edm::DefaultRecord> const esTokenBeginRun_;
    edm::ESGetToken<edm::eventsetup::test::DummyData, edm::DefaultRecord> const esToken_;
    const std::vector<int> valuesBeginRun_;
    const std::vector<int> valuesBeginLumi_;
    const std::vector<int> valuesEvent_;
    const std::vector<int> valuesEndLumi_;
    const std::vector<int> valuesEndRun_;
    const int valueES_;
  };
}  // namespace edmtest

using IntTestLooper = edmtest::IntTestLooper;
DEFINE_FWK_LOOPER(IntTestLooper);
