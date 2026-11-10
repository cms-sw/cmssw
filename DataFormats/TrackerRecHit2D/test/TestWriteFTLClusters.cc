#include <cstdint>
#include <memory>

#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

class TestWriteFTLClusters : public edm::global::EDProducer<> {
public:
  explicit TestWriteFTLClusters(edm::ParameterSet const&) : topologyToken_(esConsumes<MTDTopology, MTDTopologyRcd>()) {
    produces<FTLClusterCollection>("FTLBarrel");
    produces<FTLClusterCollection>("FTLEndcap");
  }

  void produce(edm::StreamID, edm::Event& event, edm::EventSetup const& es) const override {
    auto const& topology = es.getData(topologyToken_);
    auto const detId = topology.btlidFromIndex(1, 1);
    if (detId == 0)
      throw cms::Exception("TestWriteFTLClusters") << "No BTL module at topology indices (1, 1).";

    constexpr float energies[] = {4.f};
    constexpr float times[] = {10.f};
    constexpr float timeErrors[] = {0.03f};
    constexpr uint16_t rows[] = {0};
    constexpr uint16_t columns[] = {0};

    FTLCluster cluster(DetId(detId), 1, energies, times, timeErrors, rows, columns, rows[0], columns[0]);

    cluster.setClusterPosX(0.0f);
    cluster.setClusterErrorX(0.15f);
    cluster.setClusterErrorTime(0.03f);

    auto barrel = std::make_unique<FTLClusterCollection>();
    {
      FTLClusterCollection::FastFiller filler(*barrel, detId);
      filler.push_back(cluster);
    }
    event.put(std::move(barrel), "FTLBarrel");
    event.put(std::make_unique<FTLClusterCollection>(), "FTLEndcap");
    //one cluster in the barrel, none in the endcap
  }

private:
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> const topologyToken_;
};

DEFINE_FWK_MODULE(TestWriteFTLClusters);
