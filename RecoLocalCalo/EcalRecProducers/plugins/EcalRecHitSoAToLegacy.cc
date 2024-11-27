#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

class EcalRecHitSoAToLegacy : public edm::global::EDProducer<> {
public:
  explicit EcalRecHitSoAToLegacy(edm::ParameterSet const &ps);
  ~EcalRecHitSoAToLegacy() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  using InputProduct = EcalRecHitHostCollection;
  void produce(edm::StreamID, edm::Event &, edm::EventSetup const &) const override;

private:
  const bool isPhase2_;
  const edm::EDGetTokenT<InputProduct> inputTokenEB_;
  const edm::EDGetTokenT<InputProduct> inputTokenEE_;
  const edm::EDPutTokenT<EBRecHitCollection> outputTokenEB_;
  const edm::EDPutTokenT<EERecHitCollection> outputTokenEE_;
};

void EcalRecHitSoAToLegacy::fillDescriptions(edm::ConfigurationDescriptions &confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("inputCollectionEB", edm::InputTag("ecalRecHitPortable", "EcalRecHitsEB"));
  desc.add<std::string>("outputLabelEB", "EcalRecHitsEB");
  desc.ifValue(edm::ParameterDescription<bool>("isPhase2", false, true),
               false >> (edm::ParameterDescription<edm::InputTag>(
                             "inputCollectionEE", edm::InputTag("ecalRecHitPortable", "EcalRecHitsEE"), true) and
                         edm::ParameterDescription<std::string>("outputLabelEE", "EcalRecHitsEE", true)) or
                   true >> edm::EmptyGroupDescription());
  confDesc.add("ecalRecHitSoAToLegacy", desc);
}

EcalRecHitSoAToLegacy::EcalRecHitSoAToLegacy(edm::ParameterSet const &ps)
    : isPhase2_{ps.getParameter<bool>("isPhase2")},
      inputTokenEB_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("inputCollectionEB"))},
      inputTokenEE_{isPhase2_ ? edm::EDGetTokenT<InputProduct>{}
                              : consumes<InputProduct>(ps.getParameter<edm::InputTag>("inputCollectionEE"))},
      outputTokenEB_{produces<EBRecHitCollection>(ps.getParameter<std::string>("outputLabelEB"))},
      outputTokenEE_{isPhase2_ ? edm::EDPutTokenT<EERecHitCollection>{}
                               : produces<EERecHitCollection>(ps.getParameter<std::string>("outputLabelEE"))} {}

void EcalRecHitSoAToLegacy::produce(edm::StreamID sid, edm::Event &event, edm::EventSetup const &setup) const {
  auto const &inputCollEB = event.get(inputTokenEB_);
  auto const &inputCollEBView = inputCollEB.const_view();
  auto outputCollEB = std::make_unique<EBRecHitCollection>();
  outputCollEB->reserve(inputCollEBView.size());

  for (uint32_t i = 0; i < inputCollEBView.size(); ++i) {
    // Save only if energy is >= 0 !
    // This is important because the channels that were supposed
    // to be excluded get "-1" as energy
    if (inputCollEBView.energy()[i] >= 0.) {
      outputCollEB->emplace_back(DetId{inputCollEBView.id()[i]},
                                 inputCollEBView.energy()[i],
                                 inputCollEBView.time()[i],
                                 inputCollEBView.extra()[i],
                                 inputCollEBView.flagBits()[i]);
    }
  }
  event.put(outputTokenEB_, std::move(outputCollEB));

  if (!isPhase2_) {
    auto const &inputCollEE = event.get(inputTokenEE_);
    auto const &inputCollEEView = inputCollEE.const_view();
    auto outputCollEE = std::make_unique<EERecHitCollection>();
    outputCollEE->reserve(inputCollEEView.size());

    for (uint32_t i = 0; i < inputCollEEView.size(); ++i) {
      // Save only if energy is >= 0 !
      // This is important because the channels that were supposed
      // to be excluded get "-1" as energy
      if (inputCollEEView.energy()[i] >= 0.) {
        outputCollEE->emplace_back(DetId{inputCollEEView.id()[i]},
                                   inputCollEEView.energy()[i],
                                   inputCollEEView.time()[i],
                                   inputCollEEView.extra()[i],
                                   inputCollEEView.flagBits()[i]);
      }
    }
    event.put(outputTokenEE_, std::move(outputCollEE));
  }
}

DEFINE_FWK_MODULE(EcalRecHitSoAToLegacy);
