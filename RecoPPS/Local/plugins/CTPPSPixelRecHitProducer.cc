/**********************************************************************
 *
 * Author: F.Ferro - INFN Genova
 *
 **********************************************************************/

#include "CondFormats/DataRecord/interface/PPSPixelTopologyRcd.h"
#include "CondFormats/PPSObjects/interface/PPSPixelTopology.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelCluster.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoPPS/Local/interface/RPixClusterToHit.h"

class CTPPSPixelRecHitProducer : public edm::global::EDProducer<> {
public:
  explicit CTPPSPixelRecHitProducer(const edm::ParameterSet &param);

  ~CTPPSPixelRecHitProducer() override = default;

  void produce(edm::StreamID, edm::Event &, edm::EventSetup const &) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  const edm::ESGetToken<PPSPixelTopology, PPSPixelTopologyRcd> pixelTopologyToken_;
  const edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelCluster>> clustersToken_;
  const edm::EDPutTokenT<edm::DetSetVector<CTPPSPixelRecHit>> rechitsToken_;
  const RPixClusterToHit clusterToHit_;
};

CTPPSPixelRecHitProducer::CTPPSPixelRecHitProducer(const edm::ParameterSet &config)
    : pixelTopologyToken_(esConsumes<PPSPixelTopology, PPSPixelTopologyRcd>()),
      clustersToken_(
          consumes<edm::DetSetVector<CTPPSPixelCluster>>(config.getParameter<edm::InputTag>("RPixClusterTag"))),
      rechitsToken_(produces<edm::DetSetVector<CTPPSPixelRecHit>>()),
      clusterToHit_(config) {}

void CTPPSPixelRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("RPixClusterTag", edm::InputTag("ctppsPixelClusters"));
  desc.addUntracked<int>("RPixVerbosity", 0);  // used by RPixClusterToHit
  descriptions.add("ctppsPixelRecHits", desc);
}

void CTPPSPixelRecHitProducer::produce(edm::StreamID, edm::Event &event, edm::EventSetup const &setup) const {
  PPSPixelTopology const &thePixelTopology = setup.getData(pixelTopologyToken_);
  edm::DetSetVector<CTPPSPixelCluster> const &clusters = event.get(clustersToken_);
  edm::DetSetVector<CTPPSPixelRecHit> rechits;
  rechits.reserve(clusters.size());

  // run the reconstruction
  for (auto const &cluster : clusters) {
    edm::DetSet<CTPPSPixelRecHit> &rechit = rechits.find_or_insert(cluster.id);
    rechit.data.reserve(cluster.data.size());

    // calculate the cluster parameters and convert it into a rechit
    clusterToHit_.buildHits(cluster.id, cluster.data, rechit.data, thePixelTopology);
  }

  event.emplace(rechitsToken_, std::move(rechits));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CTPPSPixelRecHitProducer);
