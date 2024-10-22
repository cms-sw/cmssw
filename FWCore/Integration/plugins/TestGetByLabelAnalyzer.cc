#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"

namespace edmtest {
  template <typename T>
  class TestGetByLabelAnalyzerT : public edm::global::EDAnalyzer<> {
  public:
    TestGetByLabelAnalyzerT(edm::ParameterSet const& iPSet)
        : src_(iPSet.getUntrackedParameter<edm::InputTag>("src")),
          getCategory_(iPSet.getUntrackedParameter<std::string>("getExceptionCategory")),
          accessCategory_(iPSet.getUntrackedParameter<std::string>("accessExceptionCategory")) {
      if (iPSet.getUntrackedParameter<bool>("consumes")) {
        consumes<T>(src_);
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<edm::InputTag>("src");
      desc.addUntracked<std::string>("getExceptionCategory", "");
      desc.addUntracked<std::string>("accessExceptionCategory", "");
      desc.addUntracked<bool>("consumes", false);
      descriptions.addDefault(desc);
    }

    void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const override {
      edm::Handle<T> handle;

      auto test = [](std::string const& category, std::string const& msg, auto func) {
        bool caught = false;
        try {
          func();
        } catch (cms::Exception& e) {
          caught = true;
          if (category.empty())
            throw;
          if (e.category() != category) {
            throw cms::Exception("Assert")
                << "Expected cms::Exception from " << msg << " with category " << category << ", got " << e.category();
          }
          return false;
        }
        if (not category.empty() and not caught) {
          throw cms::Exception("Assert") << "Expected cms::Exception to be thrown from " << msg << ", but got nothing";
        }
        return true;
      };

      bool noException = test(getCategory_, "getByLabel(InputTag)", [&]() { event.getByLabel(src_, handle); });
      if (noException) {
        test(accessCategory_, "*handle from InputTag", [&]() { *handle; });
      }

      noException =
          test(getCategory_, "getByLabel(strings)", [&]() { event.getByLabel(src_.label(), src_.instance(), handle); });
      if (noException) {
        test(accessCategory_, "*handle from strings", [&]() { *handle; });
      }
    }

  private:
    edm::InputTag const src_;
    std::string const getCategory_;
    std::string const accessCategory_;
  };

  using TestGetByLabelIntAnalyzer = TestGetByLabelAnalyzerT<edmtest::IntProduct>;
  using TestGetByLabelThingAnalyzer = TestGetByLabelAnalyzerT<edmtest::ThingCollection>;
}  // namespace edmtest

DEFINE_FWK_MODULE(edmtest::TestGetByLabelIntAnalyzer);
DEFINE_FWK_MODULE(edmtest::TestGetByLabelThingAnalyzer);
