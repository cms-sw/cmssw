#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edmtest {

  class MaybeUninitializedIntAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    MaybeUninitializedIntAnalyzer(edm::ParameterSet const& config)
        : value_{config.getParameter<int32_t>("value")},
          token_{consumes(config.getParameter<edm::InputTag>("source"))} {}

    void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const final {
      MaybeUninitializedIntProduct const& product = event.get(token_);
      if (product.value != value_) {
        throw cms::Exception("Inconsistent Data", "MaybeUninitializedIntAnalyzer::analyze")
            << "Found value " << product.value << " while expecting value " << value_;
      }
    }

  private:
    const cms_int32_t value_;
    const edm::EDGetTokenT<MaybeUninitializedIntProduct> token_;
  };

}  // namespace edmtest

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(edmtest::MaybeUninitializedIntAnalyzer);
