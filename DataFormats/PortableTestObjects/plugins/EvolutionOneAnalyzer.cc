#include "DataFormats/PortableTestObjects/interface/SchemaEvolutionSoA.h"
#include "DataFormats/PortableTestObjects/interface/SchemaEvolutionHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

using CollectionVersion = portabletest::HostCollectionEvolutionOne;

class EvolutionOneAnalyzer : public edm::global::EDAnalyzer<> {
public:
  EvolutionOneAnalyzer(edm::ParameterSet const& config)
      : source_{config.getParameter<edm::InputTag>("source")}, soaToken_{consumes(source_)} {}

  void analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const&) const override {
    CollectionVersion const& product = event.get(soaToken_);
    auto const& view = product.const_view();

    assert(view.metadata().size() == 23);

    constexpr float EPS_F = 1e-5f;

    for (int i = 0; i < view.metadata().size(); i++) {
      auto element = view[i];

      // Check that float has been casted to double
      double expectedFloat = static_cast<double>(i) + 0.1f;
      assert(std::abs(element.cFloat() - expectedFloat) < EPS_F);

      // Check that int has been casted to float
      float expectedInt = static_cast<float>(i);
      assert(std::abs(element.cInt() - expectedInt) < EPS_F);

      // Check that double has been casted to int
      int expectedDouble = static_cast<int>(std::sin(static_cast<double>(i)) * 1e3);
      assert(element.cDouble() == expectedDouble);

      // Check that enum has been casted to the new enum type
      assert(element.cEnum() == portabletest::SEEnumTypeTwo::s2);

      // Check that Eigen matrix has been casted to Eigen matrix with type double
      portabletest::SEEigenObjectTwo expectedEigen;
      expectedEigen << 10 * i + 1.1, -10 * i - 1.2, 10 * i + 2.3, -10 * i - 2.4, 10 * i + 3.5, -10 * i - 3.6,
          10 * i + 4.7, -10 * i - 4.8;
      for (int r = 0; r < element.eEigenObject().rows(); ++r) {
        for (int c = 0; c < element.eEigenObject().cols(); ++c) {
          assert(std::abs(element.eEigenObject()(r, c) - expectedEigen(r, c)) < EPS_F);
        }
      }
    }

    // Check Scalars
    assert(std::abs(view.sInt() - static_cast<double>(std::numeric_limits<int>::max() - 7)) < EPS_F);
    assert(view.sFloat() == static_cast<int8_t>(1.0f / 3.0f));
    assert(std::abs(view.sDouble() - (1.0f / 10.0f)) < EPS_F);
    assert(view.sEnum() == portabletest::SEEnumTypeTwo::s1);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source");
  }

private:
  const edm::InputTag source_;
  const edm::EDGetTokenT<CollectionVersion> soaToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EvolutionOneAnalyzer);
