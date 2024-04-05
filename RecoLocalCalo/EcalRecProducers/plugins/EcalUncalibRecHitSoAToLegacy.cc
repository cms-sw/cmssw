#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHitHostCollection.h"

class EcalUncalibRecHitSoAToLegacy : public edm::global::EDProducer<> {
public:
  explicit EcalUncalibRecHitSoAToLegacy(edm::ParameterSet const &ps);
  ~EcalUncalibRecHitSoAToLegacy() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  using InputProduct = EcalUncalibratedRecHitHostCollection;
  void produce(edm::StreamID, edm::Event &, edm::EventSetup const &) const override;

private:
  const bool isPhase2_;
  const edm::EDGetTokenT<InputProduct> inputTokenEB_;
  const edm::EDGetTokenT<InputProduct> inputTokenEE_;
  const edm::EDPutTokenT<EBUncalibratedRecHitCollection> outputTokenEB_;
  const edm::EDPutTokenT<EEUncalibratedRecHitCollection> outputTokenEE_;
};

void EcalUncalibRecHitSoAToLegacy::fillDescriptions(edm::ConfigurationDescriptions &confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("outputLabelEB", "EcalUncalibRecHitsEB");
  desc.ifValue(edm::ParameterDescription<bool>("isPhase2", false, true),
               false >> (edm::ParameterDescription<edm::InputTag>(
			     "inputCollectionEB",
                             edm::InputTag("ecalMultiFitUncalibRecHitPortable", "EcalUncalibRecHitsEB"),
                             true) and
                         edm::ParameterDescription<edm::InputTag>(
                             "inputCollectionEE",
                             edm::InputTag("ecalMultiFitUncalibRecHitPortable", "EcalUncalibRecHitsEE"),
                             true) and
                         edm::ParameterDescription<std::string>("outputLabelEE", "EcalUncalibRecHitsEE", true)) or
	       true >> (edm::ParameterDescription<edm::InputTag>(
                               "inputCollectionEB",
                               edm::InputTag("ecalUncalibRecHitPhase2Portable", "EcalUncalibRecHitsEB"),
                               true)));
  confDesc.add("ecalUncalibRecHitSoAToLegacy", desc);
}

EcalUncalibRecHitSoAToLegacy::EcalUncalibRecHitSoAToLegacy(edm::ParameterSet const &ps)
    : isPhase2_{ps.getParameter<bool>("isPhase2")},
      inputTokenEB_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("inputCollectionEB"))},
      inputTokenEE_{isPhase2_ ? edm::EDGetTokenT<InputProduct>{}
                              : consumes<InputProduct>(ps.getParameter<edm::InputTag>("inputCollectionEE"))},
      outputTokenEB_{produces<EBUncalibratedRecHitCollection>(ps.getParameter<std::string>("outputLabelEB"))},
      outputTokenEE_{isPhase2_
                         ? edm::EDPutTokenT<EEUncalibratedRecHitCollection>{}
                         : produces<EEUncalibratedRecHitCollection>(ps.getParameter<std::string>("outputLabelEE"))} {}

void EcalUncalibRecHitSoAToLegacy::produce(edm::StreamID sid, edm::Event &event, edm::EventSetup const &setup) const {
  auto const &inputCollEB = event.get(inputTokenEB_);
  auto const &inputCollEBView = inputCollEB.const_view();
  auto outputCollEB = std::make_unique<EBUncalibratedRecHitCollection>();
  outputCollEB->reserve(inputCollEBView.size());

  for (uint32_t i = 0; i < inputCollEBView.size(); ++i) {
    outputCollEB->emplace_back(DetId{inputCollEBView.id()[i]},
                               inputCollEBView.amplitude()[i],
                               inputCollEBView.pedestal()[i],
                               inputCollEBView.jitter()[i],
                               inputCollEBView.chi2()[i],
                               inputCollEBView.flags()[i]);
    if (isPhase2_) {
      (*outputCollEB)[i].setAmplitudeError(inputCollEBView.amplitudeError()[i]);
    }
    (*outputCollEB)[i].setJitterError(inputCollEBView.jitterError()[i]);
    for (uint32_t sample = 0; sample < EcalDataFrame::MAXSAMPLES; ++sample) {
      (*outputCollEB)[i].setOutOfTimeAmplitude(sample, inputCollEBView.outOfTimeAmplitudes()[i][sample]);
    }
  }
  event.put(outputTokenEB_, std::move(outputCollEB));

  if (!isPhase2_) {
    auto const &inputCollEE = event.get(inputTokenEE_);
    auto const &inputCollEEView = inputCollEE.const_view();
    auto outputCollEE = std::make_unique<EEUncalibratedRecHitCollection>();
    outputCollEE->reserve(inputCollEEView.size());

    for (uint32_t i = 0; i < inputCollEEView.size(); ++i) {
      outputCollEE->emplace_back(DetId{inputCollEEView.id()[i]},
                                 inputCollEEView.amplitude()[i],
                                 inputCollEEView.pedestal()[i],
                                 inputCollEEView.jitter()[i],
                                 inputCollEEView.chi2()[i],
                                 inputCollEEView.flags()[i]);
      (*outputCollEE)[i].setJitterError(inputCollEEView.jitterError()[i]);
      for (uint32_t sample = 0; sample < EcalDataFrame::MAXSAMPLES; ++sample) {
        (*outputCollEE)[i].setOutOfTimeAmplitude(sample, inputCollEEView.outOfTimeAmplitudes()[i][sample]);
      }
    }
    event.put(outputTokenEE_, std::move(outputCollEE));
  }
}

DEFINE_FWK_MODULE(EcalUncalibRecHitSoAToLegacy);
