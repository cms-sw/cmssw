#include <Eigen/Core>

#include <cstdint>

#include "DataFormats/CaloRecHit/test/CaloClusterSoATestValues.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edmtest {

  // Reads a reco::CaloClusterHostCollection and throws if it differs from the known values of TestWriteCaloClusterSoA.
  class TestReadCaloClusterSoA : public edm::global::EDAnalyzer<> {
  public:
    explicit TestReadCaloClusterSoA(edm::ParameterSet const& config)
        : token_(consumes(config.getParameter<edm::InputTag>("source"))) {}

    void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const override {
      const auto n = static_cast<int32_t>((event.id().event() - 1) % 4);
      caloclustertest::check(event.get(token_).const_view(), n);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("source", edm::InputTag("writer"));
      descriptions.addDefault(desc);
    }

  private:
    const edm::EDGetTokenT<reco::CaloClusterHostCollection> token_;
  };

}  // namespace edmtest

using edmtest::TestReadCaloClusterSoA;
DEFINE_FWK_MODULE(TestReadCaloClusterSoA);
