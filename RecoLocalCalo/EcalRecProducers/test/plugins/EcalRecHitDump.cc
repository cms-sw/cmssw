#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <iostream>

class EcalRecHitDump : public edm::stream::EDAnalyzer<> {
public:
  explicit EcalRecHitDump(const edm::ParameterSet&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<EcalRecHitCollection> EBRecHitCollectionT_;
  const edm::EDGetTokenT<EcalRecHitCollection> EERecHitCollectionT_;
};

EcalRecHitDump::EcalRecHitDump(const edm::ParameterSet& iConfig)
    : EBRecHitCollectionT_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHitCollection"))),
      EERecHitCollectionT_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHitCollection"))) {}

void EcalRecHitDump::analyze(const edm::Event& ev, const edm::EventSetup&) {
  edm::Handle<EcalRecHitCollection> EBRecHits_;
  edm::Handle<EcalRecHitCollection> EERecHits_;

  ev.getByToken(EBRecHitCollectionT_, EBRecHits_);
  ev.getByToken(EERecHitCollectionT_, EERecHits_);

  for (auto const& h : (*EBRecHits_))
    std::cout << h << std::endl;

  for (auto const& h : (*EERecHits_))
    std::cout << h << std::endl;
}

DEFINE_FWK_MODULE(EcalRecHitDump);
