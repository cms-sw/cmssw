#include <iostream>
#include <string>

#include "DataFormats/HcalRecHit/interface/HcalRecHitHostCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

class HcalRecHitSoAToLegacy : public edm::stream::EDProducer<> {
public:
  explicit HcalRecHitSoAToLegacy(edm::ParameterSet const& ps);
  ~HcalRecHitSoAToLegacy() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, edm::EventSetup const&) override;

private:
  const edm::EDGetTokenT<hcal::RecHitHostCollection> recHitsTokenIn_;
  const edm::EDPutTokenT<HBHERecHitCollection> recHitsLegacyTokenOut_;
};

void HcalRecHitSoAToLegacy::fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag{"hbheRecHitProducerPortable"});

  confDesc.addWithDefaultLabel(desc);
}

HcalRecHitSoAToLegacy::HcalRecHitSoAToLegacy(const edm::ParameterSet& ps)
    : recHitsTokenIn_{consumes<hcal::RecHitHostCollection>(ps.getParameter<edm::InputTag>("src"))},
      recHitsLegacyTokenOut_{produces<HBHERecHitCollection>()} {}

void HcalRecHitSoAToLegacy::produce(edm::Event& event, edm::EventSetup const& setup) {
  // populate the legacy collection
  auto recHitsLegacy = std::make_unique<HBHERecHitCollection>();

  // get input from host SoA
  auto const& hcalRechitSoAView = event.get(recHitsTokenIn_).const_view();

  recHitsLegacy->reserve(hcalRechitSoAView.metadata().size());

  for (auto i = 0; i < hcalRechitSoAView.metadata().size(); i++) {
    auto const& rechit = hcalRechitSoAView[i];
    // skip bad channels
    if (rechit.chi2() < 0)
      continue;

    // build a legacy rechit with the computed detid and MAHI energy
    recHitsLegacy->emplace_back(HcalDetId{rechit.detId()},
                                rechit.energy(),
                                rechit.timeM0()  // timeRising
    );
    // update the legacy rechit with the Chi2 and M0 values
    recHitsLegacy->back().setChiSquared(rechit.chi2());
    recHitsLegacy->back().setRawEnergy(rechit.energyM0());
  }

  // put the legacy collection
  event.put(recHitsLegacyTokenOut_, std::move(recHitsLegacy));
}

DEFINE_FWK_MODULE(HcalRecHitSoAToLegacy);
