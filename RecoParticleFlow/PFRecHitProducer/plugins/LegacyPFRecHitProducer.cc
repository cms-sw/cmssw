#include <unordered_map>

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

class LegacyPFRecHitProducer : public edm::stream::EDProducer<> {
public:
  LegacyPFRecHitProducer(edm::ParameterSet const& config)
      : alpakaPfRecHitsToken_(consumes(config.getParameter<edm::InputTag>("src"))),
        legacyPfRecHitsToken_(produces()),
        geomToken_(esConsumes<edm::Transition::BeginRun>()) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<reco::PFRecHitHostCollection> alpakaPfRecHitsToken_;
  const edm::EDPutTokenT<reco::PFRecHitCollection> legacyPfRecHitsToken_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  std::unordered_map<PFLayer::Layer, const CaloSubdetectorGeometry*> caloGeo_;
};

void LegacyPFRecHitProducer::beginRun(edm::Run const&, const edm::EventSetup& setup) {
  const CaloGeometry& geo = setup.getData(geomToken_);
  caloGeo_[PFLayer::HCAL_BARREL1] = geo.getSubdetectorGeometry(DetId::Hcal, HcalSubdetector::HcalBarrel);
  caloGeo_[PFLayer::HCAL_ENDCAP] = geo.getSubdetectorGeometry(DetId::Hcal, HcalSubdetector::HcalEndcap);
  caloGeo_[PFLayer::ECAL_BARREL] = geo.getSubdetectorGeometry(DetId::Ecal, EcalSubdetector::EcalBarrel);
  caloGeo_[PFLayer::ECAL_ENDCAP] = geo.getSubdetectorGeometry(DetId::Ecal, EcalSubdetector::EcalEndcap);
}

void LegacyPFRecHitProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  const reco::PFRecHitHostCollection& pfRecHitsAlpakaSoA = event.get(alpakaPfRecHitsToken_);
  const reco::PFRecHitHostCollection::ConstView& alpakaPfRecHits = pfRecHitsAlpakaSoA.const_view();

  reco::PFRecHitCollection out;
  out.reserve(alpakaPfRecHits.size());

  for (size_t i = 0; i < alpakaPfRecHits.size(); i++) {
    reco::PFRecHit& pfrh =
        out.emplace_back(caloGeo_.at(alpakaPfRecHits[i].layer())->getGeometry(alpakaPfRecHits[i].detId()),
                         alpakaPfRecHits[i].detId(),
                         alpakaPfRecHits[i].layer(),
                         alpakaPfRecHits[i].energy());
    pfrh.setTime(alpakaPfRecHits[i].time());
    pfrh.setDepth(alpakaPfRecHits[i].depth());

    // order in Alpaka:   N, S, E, W,NE,SW,SE,NW
    const short eta[8] = {0, 0, 1, -1, 1, -1, 1, -1};
    const short phi[8] = {1, -1, 0, 0, 1, -1, -1, 1};
    for (size_t k = 0; k < 8; k++)
      if (alpakaPfRecHits[i].neighbours()(k) != -1)
        pfrh.addNeighbour(eta[k], phi[k], 0, alpakaPfRecHits[i].neighbours()(k));
  }

  event.emplace(legacyPfRecHitsToken_, out);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LegacyPFRecHitProducer);
