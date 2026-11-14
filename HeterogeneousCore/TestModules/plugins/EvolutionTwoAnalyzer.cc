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

#include "HeterogeneousCore/TestModules/interface/SchemaEvolutionSoA.h"
#include "HeterogeneousCore/TestModules/interface/SchemaEvolutionHostCollection.h"

using CollectionVersion = testmodules::HostCollectionEvolutionTwo;

class EvolutionTwoAnalyzer : public edm::global::EDAnalyzer<> {
public:
  EvolutionTwoAnalyzer(edm::ParameterSet const& config)
      : source_{config.getParameter<edm::InputTag>("source")}, soaToken_{consumes(source_)} {}

  void analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const&) const override {
    CollectionVersion const& product = event.get(soaToken_);
    auto const& view = product.const_view();

    assert(view.metadata().size() == 23);

    constexpr float EPS_F = 1e-5f;
    constexpr double EPS_D = 1e-10;

    for (int i = 0; i < view.metadata().size(); i++) {
      auto element = view[i];

      // Check that float is still correctly read
      double expectedFloat = static_cast<double>(i) + 0.1f;
      assert(std::abs(element.cFloat() - expectedFloat) < EPS_F);

      // Check that new column is zero initialized when reading old data
      assert(element.newColumn() == static_cast<int>(0));

      // Check that enum is still correct
      assert(element.cEnum() == testmodules::SEEnumType::s2);

      // Check that Eigen matrix is still correct
      testmodules::SEEigenObject expectedEigen;
      expectedEigen << 10 * i + 1.1, -10 * i - 1.2, 10 * i + 2.3, -10 * i - 2.4, 10 * i + 3.5, -10 * i - 3.6,
          10 * i + 4.7, -10 * i - 4.8;
      for (int r = 0; r < element.eEigenObject().rows(); ++r) {
        for (int c = 0; c < element.eEigenObject().cols(); ++c) {
          assert(std::abs(element.eEigenObject()(r, c) - expectedEigen(r, c)) < EPS_F);
        }
      }

      // Check that new Eigen matrix is zero initialized when reading old data
      for (int r = 0; r < element.newEigenObject().rows(); ++r) {
        for (int c = 0; c < element.newEigenObject().cols(); ++c) {
          assert(std::abs(element.newEigenObject()(r, c) - static_cast<double>(0)) < EPS_D);
        }
      }

      for (std::size_t j = 0; j < element.cArray().size(); j++) {
        int expectedArrayValue = i * static_cast<int>(element.cArray().size()) + static_cast<int>(j);
        assert(element.cArray()[j] == expectedArrayValue);
      }
    }

    // Check Scalars
    assert(std::abs(view.sFloatNewName() - static_cast<float>(0.0f)) < EPS_F);
    assert(view.newScalar() == static_cast<int8_t>(0));
    assert(view.sEnum() == testmodules::SEEnumType::s1);
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
DEFINE_FWK_MODULE(EvolutionTwoAnalyzer);
