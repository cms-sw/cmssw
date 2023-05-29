#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <iostream>

class EcalUncalibRecHitDump : public edm::stream::EDAnalyzer<> {
public:
  explicit EcalUncalibRecHitDump(const edm::ParameterSet&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<EcalUncalibratedRecHitCollection> EBUncalicRecHitCollectionT_;
  edm::EDGetTokenT<EcalUncalibratedRecHitCollection> EEUncalicRecHitCollectionT_;
};

EcalUncalibRecHitDump::EcalUncalibRecHitDump(const edm::ParameterSet& iConfig) {
  EBUncalicRecHitCollectionT_ =
      consumes<EcalUncalibratedRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBUncalibRecHitCollection"));
  EEUncalicRecHitCollectionT_ =
      consumes<EcalUncalibratedRecHitCollection>(iConfig.getParameter<edm::InputTag>("EEUncalibRecHitCollection"));
}

void EcalUncalibRecHitDump::analyze(const edm::Event& ev, const edm::EventSetup&) {
  edm::Handle<EcalUncalibratedRecHitCollection> EBURecHits_;
  edm::Handle<EcalUncalibratedRecHitCollection> EEURecHits_;

  ev.getByToken(EBUncalicRecHitCollectionT_, EBURecHits_);
  ev.getByToken(EEUncalicRecHitCollectionT_, EEURecHits_);

  for (auto const& h : (*EBURecHits_))
    std::cout << "EB id: " << h.id() << " amplitude: " << h.amplitude() << " pedestal: " << h.pedestal()
              << " jitter: " << h.jitter() << " chi2: " << h.chi2() << " amplitudeError: " << h.amplitudeError()
              << " jitterErrorBits: " << int(h.jitterErrorBits()) << std::endl;

  for (auto const& h : (*EEURecHits_))
    std::cout << "EE id: " << h.id() << " amplitude: " << h.amplitude() << " pedestal: " << h.pedestal()
              << " jitter: " << h.jitter() << " chi2: " << h.chi2() << " amplitudeError: " << h.amplitudeError()
              << " jitterErrorBits: " << int(h.jitterErrorBits()) << std::endl;
}

DEFINE_FWK_MODULE(EcalUncalibRecHitDump);
