#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"

#include "DataFormats/EcalDigi/interface/EcalConstants.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class EcalUncalibRecHitPhase2WeightsProducer : public edm::stream::EDProducer<> {
public:
  explicit EcalUncalibRecHitPhase2WeightsProducer(const edm::ParameterSet& ps);
  void produce(edm::Event& evt, const edm::EventSetup& es) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const float tRise_;
  const float tFall_;
  const std::vector<double> ampWeights_;
  const std::vector<double> timeWeights_;

  const edm::EDGetTokenT<EBDigiCollectionPh2> ebDigiCollectionToken_;
  const edm::EDPutTokenT<EBUncalibratedRecHitCollection> ebUncalibRecHitCollectionToken_;
};

void EcalUncalibRecHitPhase2WeightsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("EBhitCollection", "EcalUncalibRecHitsEB");
  desc.add<double>("tRise", 0.2);
  desc.add<double>("tFall", 2.);
  desc.add<std::vector<double>>("ampWeights",
                                {-0.121016,
                                 -0.119899,
                                 -0.120923,
                                 -0.0848959,
                                 0.261041,
                                 0.509881,
                                 0.373591,
                                 0.134899,
                                 -0.0233605,
                                 -0.0913195,
                                 -0.112452,
                                 -0.118596,
                                 -0.121737,
                                 -0.121737,
                                 -0.121737,
                                 -0.121737});
  desc.add<std::vector<double>>("timeWeights",
                                {0.429452,
                                 0.442762,
                                 0.413327,
                                 0.858327,
                                 4.42324,
                                 2.04369,
                                 -3.42426,
                                 -4.16258,
                                 -2.36061,
                                 -0.725371,
                                 0.0727267,
                                 0.326005,
                                 0.402035,
                                 0.404287,
                                 0.434207,
                                 0.422775});

  desc.add<edm::InputTag>("BarrelDigis", edm::InputTag("simEcalUnsuppressedDigis", ""));

  descriptions.addWithDefaultLabel(desc);
}

EcalUncalibRecHitPhase2WeightsProducer::EcalUncalibRecHitPhase2WeightsProducer(const edm::ParameterSet& ps)
    : tRise_(ps.getParameter<double>("tRise")),
      tFall_(ps.getParameter<double>("tFall")),
      ampWeights_(ps.getParameter<std::vector<double>>("ampWeights")),
      timeWeights_(ps.getParameter<std::vector<double>>("timeWeights")),
      ebDigiCollectionToken_(consumes<EBDigiCollectionPh2>(ps.getParameter<edm::InputTag>("BarrelDigis"))),
      ebUncalibRecHitCollectionToken_(
          produces<EBUncalibratedRecHitCollection>(ps.getParameter<std::string>("EBhitCollection"))) {}

void EcalUncalibRecHitPhase2WeightsProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  // retrieve digis
  const EBDigiCollectionPh2* pdigis = &evt.get(ebDigiCollectionToken_);

  // prepare output
  auto ebUncalibRechits = std::make_unique<EBUncalibratedRecHitCollection>();

  for (auto itdg = pdigis->begin(); itdg != pdigis->end(); ++itdg) {
    EBDataFrame digi(*itdg);
    EcalDataFrame_Ph2 dataFrame(*itdg);
    DetId detId(digi.id());

    bool g1 = false;

    std::vector<float> timetrace;
    std::vector<float> adctrace;

    int nSamples = digi.size();

    timetrace.reserve(nSamples);
    adctrace.reserve(nSamples);

    float amp = 0;
    float t0 = 0;
    float gratio;

    for (int sample = 0; sample < nSamples; ++sample) {
      EcalLiteDTUSample thisSample = dataFrame[sample];
      gratio = ecalPh2::gains[thisSample.gainId()];
      adctrace.push_back(thisSample.adc() * gratio);

      amp = amp + adctrace[sample] * ampWeights_[sample];

      t0 = t0 + adctrace[sample] * timeWeights_[sample];

      if (thisSample.gainId() == 1)
        g1 = true;

      timetrace.push_back(sample);

    }  // loop on samples

    float amp_e = 1;
    float t0_e = 0;

    EcalUncalibratedRecHit rhit(detId, amp, 0., t0, 0., 0);  // rhit(detIt, amp, pedestal, t0, chi2, flags)
    rhit.setAmplitudeError(amp_e);
    rhit.setJitterError(t0_e);
    if (g1)
      rhit.setFlagBit(EcalUncalibratedRecHit::kHasSwitchToGain1);

    ebUncalibRechits->push_back(rhit);

  }  // loop on digis

  evt.put(ebUncalibRecHitCollectionToken_, std::move(ebUncalibRechits));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalUncalibRecHitPhase2WeightsProducer);
