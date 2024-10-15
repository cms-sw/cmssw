#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

class EcalRecHitSoAToLegacy : public edm::stream::EDProducer<> {
public:
  explicit EcalRecHitSoAToLegacy(edm::ParameterSet const &ps);
  ~EcalRecHitSoAToLegacy() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  using InputProduct = EcalRecHitHostCollection;
  void produce(edm::Event &, edm::EventSetup const &) override;

private:
  const bool isPhase2_;
  const edm::EDGetTokenT<InputProduct> recHitsPortableEB_;
  const edm::EDGetTokenT<InputProduct> recHitsPortableEE_;
  const edm::EDPutTokenT<EBRecHitCollection> recHitsCPUEBToken_;
  const edm::EDPutTokenT<EERecHitCollection> recHitsCPUEEToken_;
};

void EcalRecHitSoAToLegacy::fillDescriptions(edm::ConfigurationDescriptions &confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recHitsPortableEB", edm::InputTag("ecalRecHitPortable", "EcalRecHitsEB"));
  desc.add<std::string>("recHitsLabelCPUEB", "EcalRecHitsEB");
  desc.ifValue(edm::ParameterDescription<bool>("isPhase2", false, true),
               false >> (edm::ParameterDescription<edm::InputTag>(
                             "recHitsPortableEE", edm::InputTag("ecalRecHitPortable", "EcalRecHitsEE"), true) and
                         edm::ParameterDescription<std::string>("recHitsLabelCPUEE", "EcalRecHitsEE", true)) or
                   true >> edm::EmptyGroupDescription());
  confDesc.add("ecalRecHitSoAToLegacy", desc);
}

EcalRecHitSoAToLegacy::EcalRecHitSoAToLegacy(edm::ParameterSet const &ps)
    : isPhase2_{ps.getParameter<bool>("isPhase2")},
      recHitsPortableEB_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("recHitsPortableEB"))},
      recHitsPortableEE_{isPhase2_ ? edm::EDGetTokenT<InputProduct>{}
                                   : consumes<InputProduct>(ps.getParameter<edm::InputTag>("recHitsPortableEE"))},
      recHitsCPUEBToken_{produces<EBRecHitCollection>(ps.getParameter<std::string>("recHitsLabelCPUEB"))},
      recHitsCPUEEToken_{isPhase2_ ? edm::EDPutTokenT<EERecHitCollection>{}
                                   : produces<EERecHitCollection>(ps.getParameter<std::string>("recHitsLabelCPUEE"))} {}

void EcalRecHitSoAToLegacy::produce(edm::Event &event, edm::EventSetup const &setup) {
  auto const &recHitsEBColl = event.get(recHitsPortableEB_);
  auto const &recHitsEBCollView = recHitsEBColl.const_view();
  auto recHitsCPUEB = std::make_unique<EBRecHitCollection>();
  recHitsCPUEB->reserve(recHitsEBCollView.size());

  for (uint32_t i = 0; i < recHitsEBCollView.size(); ++i) {
    recHitsCPUEB->emplace_back(DetId{recHitsEBCollView.id()[i]},
                               recHitsEBCollView.energy()[i],
                               recHitsEBCollView.time()[i],
                               recHitsEBCollView.extra()[i],
                               recHitsEBCollView.flagBits()[i]);
  }
  event.put(recHitsCPUEBToken_, std::move(recHitsCPUEB));

  if (!isPhase2_) {
    auto const &recHitsEEColl = event.get(recHitsPortableEE_);
    auto const &recHitsEECollView = recHitsEEColl.const_view();
    auto recHitsCPUEE = std::make_unique<EERecHitCollection>();
    recHitsCPUEE->reserve(recHitsEECollView.size());

    for (uint32_t i = 0; i < recHitsEECollView.size(); ++i) {
      recHitsCPUEE->emplace_back(DetId{recHitsEECollView.id()[i]},
                                 recHitsEECollView.energy()[i],
                                 recHitsEECollView.time()[i],
                                 recHitsEECollView.extra()[i],
                                 recHitsEECollView.flagBits()[i]);
    }
    event.put(recHitsCPUEEToken_, std::move(recHitsCPUEE));
  }
}

DEFINE_FWK_MODULE(EcalRecHitSoAToLegacy);
