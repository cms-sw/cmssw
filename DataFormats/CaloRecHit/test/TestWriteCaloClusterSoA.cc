#include <Eigen/Core>

#include <cstdint>

#include "DataFormats/CaloRecHit/test/CaloClusterSoATestValues.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

namespace edmtest {

  // Puts a reco::CaloClusterHostCollection with the known values of the tests: event e has (e - 1) % 4 elements.
  class TestWriteCaloClusterSoA : public edm::global::EDProducer<> {
  public:
    explicit TestWriteCaloClusterSoA(edm::ParameterSet const&) : token_(produces()) {}

    void produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const override {
      const auto n = static_cast<int32_t>((event.id().event() - 1) % 4);
      reco::CaloClusterHostCollection clusters(n, n, n, n);
      caloclustertest::fill(clusters.view());
      event.emplace(token_, std::move(clusters));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addDefault(desc);
    }

  private:
    const edm::EDPutTokenT<reco::CaloClusterHostCollection> token_;
  };

}  // namespace edmtest

using edmtest::TestWriteCaloClusterSoA;
DEFINE_FWK_MODULE(TestWriteCaloClusterSoA);
