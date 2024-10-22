#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHitHostCollection.h"

class EcalUncalibRecHitSoAToLegacy : public edm::stream::EDProducer<> {
public:
  explicit EcalUncalibRecHitSoAToLegacy(edm::ParameterSet const &ps);
  ~EcalUncalibRecHitSoAToLegacy() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  using InputProduct = EcalUncalibratedRecHitHostCollection;
  void produce(edm::Event &, edm::EventSetup const &) override;

private:
  const bool isPhase2_;
  const edm::EDGetTokenT<InputProduct> uncalibRecHitsPortableEB_;
  const edm::EDGetTokenT<InputProduct> uncalibRecHitsPortableEE_;
  const edm::EDPutTokenT<EBUncalibratedRecHitCollection> uncalibRecHitsCPUEBToken_;
  const edm::EDPutTokenT<EEUncalibratedRecHitCollection> uncalibRecHitsCPUEEToken_;
};

void EcalUncalibRecHitSoAToLegacy::fillDescriptions(edm::ConfigurationDescriptions &confDesc) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("uncalibRecHitsPortableEB",
                          edm::InputTag("ecalMultiFitUncalibRecHitPortable", "EcalUncalibRecHitsEB"));
  desc.add<std::string>("recHitsLabelCPUEB", "EcalUncalibRecHitsEB");
  desc.ifValue(edm::ParameterDescription<bool>("isPhase2", false, true),
               false >> (edm::ParameterDescription<edm::InputTag>(
                             "uncalibRecHitsPortableEE",
                             edm::InputTag("ecalMultiFitUncalibRecHitPortable", "EcalUncalibRecHitsEE"),
                             true) and
                         edm::ParameterDescription<std::string>("recHitsLabelCPUEE", "EcalUncalibRecHitsEE", true)) or
                   true >> edm::EmptyGroupDescription());
  confDesc.add("ecalUncalibRecHitSoAToLegacy", desc);
}

EcalUncalibRecHitSoAToLegacy::EcalUncalibRecHitSoAToLegacy(edm::ParameterSet const &ps)
    : isPhase2_{ps.getParameter<bool>("isPhase2")},
      uncalibRecHitsPortableEB_{consumes<InputProduct>(ps.getParameter<edm::InputTag>("uncalibRecHitsPortableEB"))},
      uncalibRecHitsPortableEE_{
          isPhase2_ ? edm::EDGetTokenT<InputProduct>{}
                    : consumes<InputProduct>(ps.getParameter<edm::InputTag>("uncalibRecHitsPortableEE"))},
      uncalibRecHitsCPUEBToken_{
          produces<EBUncalibratedRecHitCollection>(ps.getParameter<std::string>("recHitsLabelCPUEB"))},
      uncalibRecHitsCPUEEToken_{
          isPhase2_ ? edm::EDPutTokenT<EEUncalibratedRecHitCollection>{}
                    : produces<EEUncalibratedRecHitCollection>(ps.getParameter<std::string>("recHitsLabelCPUEE"))} {}

void EcalUncalibRecHitSoAToLegacy::produce(edm::Event &event, edm::EventSetup const &setup) {
  auto const &uncalRecHitsEBColl = event.get(uncalibRecHitsPortableEB_);
  auto const &uncalRecHitsEBCollView = uncalRecHitsEBColl.const_view();
  auto recHitsCPUEB = std::make_unique<EBUncalibratedRecHitCollection>();
  recHitsCPUEB->reserve(uncalRecHitsEBCollView.size());

  for (uint32_t i = 0; i < uncalRecHitsEBCollView.size(); ++i) {
    recHitsCPUEB->emplace_back(DetId{uncalRecHitsEBCollView.id()[i]},
                               uncalRecHitsEBCollView.amplitude()[i],
                               uncalRecHitsEBCollView.pedestal()[i],
                               uncalRecHitsEBCollView.jitter()[i],
                               uncalRecHitsEBCollView.chi2()[i],
                               uncalRecHitsEBCollView.flags()[i]);
    if (isPhase2_) {
      (*recHitsCPUEB)[i].setAmplitudeError(uncalRecHitsEBCollView.amplitudeError()[i]);
    }
    (*recHitsCPUEB)[i].setJitterError(uncalRecHitsEBCollView.jitterError()[i]);
    for (uint32_t sample = 0; sample < EcalDataFrame::MAXSAMPLES; ++sample) {
      (*recHitsCPUEB)[i].setOutOfTimeAmplitude(sample, uncalRecHitsEBCollView.outOfTimeAmplitudes()[i][sample]);
    }
  }
  event.put(uncalibRecHitsCPUEBToken_, std::move(recHitsCPUEB));

  if (!isPhase2_) {
    auto const &uncalRecHitsEEColl = event.get(uncalibRecHitsPortableEE_);
    auto const &uncalRecHitsEECollView = uncalRecHitsEEColl.const_view();
    auto recHitsCPUEE = std::make_unique<EEUncalibratedRecHitCollection>();
    recHitsCPUEE->reserve(uncalRecHitsEECollView.size());

    for (uint32_t i = 0; i < uncalRecHitsEECollView.size(); ++i) {
      recHitsCPUEE->emplace_back(DetId{uncalRecHitsEECollView.id()[i]},
                                 uncalRecHitsEECollView.amplitude()[i],
                                 uncalRecHitsEECollView.pedestal()[i],
                                 uncalRecHitsEECollView.jitter()[i],
                                 uncalRecHitsEECollView.chi2()[i],
                                 uncalRecHitsEECollView.flags()[i]);
      (*recHitsCPUEE)[i].setJitterError(uncalRecHitsEECollView.jitterError()[i]);
      for (uint32_t sample = 0; sample < EcalDataFrame::MAXSAMPLES; ++sample) {
        (*recHitsCPUEE)[i].setOutOfTimeAmplitude(sample, uncalRecHitsEECollView.outOfTimeAmplitudes()[i][sample]);
      }
    }
    event.put(uncalibRecHitsCPUEEToken_, std::move(recHitsCPUEE));
  }
}

DEFINE_FWK_MODULE(EcalUncalibRecHitSoAToLegacy);
