// The purpose of this EDProducer is to test the existence of
// dictionaries for certain types, especially those from the standard
// library.

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include <cassert>
#include <memory>
#include <vector>

namespace edm {
  class EventSetup;
}

namespace edmtest {
  class ExistingDictionaryTestProducer : public edm::global::EDProducer<> {
  public:
    explicit ExistingDictionaryTestProducer(edm::ParameterSet const&)
        : intToken_{produces<int>()},
          vecUniqIntToken_{produces<std::vector<std::unique_ptr<int>>>()},
          vecUniqIntProdToken_{produces<std::vector<std::unique_ptr<IntProduct>>>()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addDefault(desc);
    }

    void produce(edm::StreamID id, edm::Event& iEvent, edm::EventSetup const&) const override {
      iEvent.emplace(intToken_, 1);

      std::vector<std::unique_ptr<int>> foo;
      foo.emplace_back(std::make_unique<int>(1));
      iEvent.emplace(vecUniqIntToken_, std::move(foo));

      std::vector<std::unique_ptr<IntProduct>> foo2;
      foo2.emplace_back(std::make_unique<IntProduct>(1));
      iEvent.emplace(vecUniqIntProdToken_, std::move(foo2));
    }

  private:
    const edm::EDPutTokenT<int> intToken_;
    const edm::EDPutTokenT<std::vector<std::unique_ptr<int>>> vecUniqIntToken_;
    const edm::EDPutTokenT<std::vector<std::unique_ptr<IntProduct>>> vecUniqIntProdToken_;
  };

  class ExistingDictionaryTestAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    explicit ExistingDictionaryTestAnalyzer(edm::ParameterSet const& iConfig)
        : intToken_{consumes(iConfig.getParameter<edm::InputTag>("src"))},
          vecUniqIntToken_{consumes(iConfig.getParameter<edm::InputTag>("src"))},
          vecUniqIntProdToken_{consumes(iConfig.getParameter<edm::InputTag>("src"))},
          testVecUniqInt_{iConfig.getParameter<bool>("testVecUniqInt")} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src", edm::InputTag{"prod"});
      desc.add<bool>("testVecUniqInt", true);
      descriptions.addDefault(desc);
    }

    void analyze(edm::StreamID id, edm::Event const& iEvent, edm::EventSetup const&) const override {
      assert(iEvent.get(intToken_) == 1);

      const auto& vecUniqInt = iEvent.get(vecUniqIntToken_);
      assert(vecUniqInt.size() == 1);
      for (const auto& elem : vecUniqInt) {
        if (testVecUniqInt_) {
          assert(elem.get() != nullptr);
          assert(*elem == 1);
        } else {
          if (elem) {
            edm::LogError("ExistingDictionaryTestAnalyzer")
                << "I am now getting a valid pointer, please update the test";
          }
        }
      }

      const auto& vecUniqIntProd = iEvent.get(vecUniqIntProdToken_);
      assert(vecUniqIntProd.size() == 1);
      for (const auto& elem : vecUniqIntProd) {
        assert(elem.get() != nullptr);
        assert(elem->value == 1);
      }
    }

  private:
    const edm::EDGetTokenT<int> intToken_;
    const edm::EDGetTokenT<std::vector<std::unique_ptr<int>>> vecUniqIntToken_;
    const edm::EDGetTokenT<std::vector<std::unique_ptr<IntProduct>>> vecUniqIntProdToken_;
    const bool testVecUniqInt_;
  };
}  // namespace edmtest

using edmtest::ExistingDictionaryTestAnalyzer;
using edmtest::ExistingDictionaryTestProducer;
DEFINE_FWK_MODULE(ExistingDictionaryTestProducer);
DEFINE_FWK_MODULE(ExistingDictionaryTestAnalyzer);
