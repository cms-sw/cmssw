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

using CollectionVersion = testmodules::HostCollectionEvolutionThree;

class EvolutionThreeAnalyzer : public edm::global::EDAnalyzer<> {
public:
  EvolutionThreeAnalyzer(edm::ParameterSet const& config)
      : source_{config.getParameter<edm::InputTag>("source")}, soaToken_{consumes(source_)} {}

  void analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const&) const override {
    CollectionVersion const& product = event.get(soaToken_);
    auto const& view = product.const_view();

    assert(view.metadata().size() == 23);

    constexpr float EPS_F = 1e-5f;
    for (int i = 0; i < view.metadata().size(); i++) {
      auto element = view[i];

      // Check ints
      int expectedInt = i;
      assert(element.cInt() == expectedInt);

      // Check enum
      assert(element.cEnum() == testmodules::SEEnumType::s2);

      // Check Eigen matrix
      testmodules::SEEigenObject expectedEigen;
      expectedEigen << 10 * i + 1.1f, -10 * i - 1.2f, 10 * i + 2.3f, -10 * i - 2.4f, 10 * i + 3.5f, -10 * i - 3.6f,
          10 * i + 4.7f, -10 * i - 4.8f;
      assert((element.eEigenObject() - expectedEigen).cwiseAbs().maxCoeff() < EPS_F);

      for (std::size_t j = 0; j < element.cArray().size(); j++) {
        int expectedArrayValue = i * static_cast<int>(element.cArray().size()) + static_cast<int>(j);
        assert(element.cArray()[j] == expectedArrayValue);
      }
    }

    // Check Scalars
    assert(view.sInt() == std::numeric_limits<int>::max() - 7);
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
DEFINE_FWK_MODULE(EvolutionThreeAnalyzer);
