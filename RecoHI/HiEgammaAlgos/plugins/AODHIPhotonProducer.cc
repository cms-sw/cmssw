#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/AODHIPhoton.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#ifndef AODHIPhotonProducer_h
#define AODHIPhotonProducer_h

class AODHIPhotonProducer : public edm::stream::EDProducer<> {

 public:

  explicit AODHIPhotonProducer (const edm::ParameterSet& ps);
  ~AODHIPhotonProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:

  virtual void beginRun (edm::Run const& r, edm::EventSetup const & es) override final;
  virtual void endRun(edm::Run const&,  edm::EventSetup const&) override final;
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

  edm::EDGetTokenT<reco::PhotonCollection> photonProducer_;
  edm::EDGetTokenT<EcalRecHitCollection> barrelEcalHits_;
  edm::EDGetTokenT<EcalRecHitCollection> endcapEcalHits_;
};

#endif

AODHIPhotonProducer::AODHIPhotonProducer(const edm::ParameterSet& config)
{
  photonProducer_   =
    consumes<reco::PhotonCollection>(config.getParameter<edm::InputTag>("photonProducer"));
  barrelEcalHits_   =
    consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("barrelEcalHits"));
  endcapEcalHits_   =
    consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("endcapEcalHits"));

  produces< aod::AODHIPhotonCollection >();
}

AODHIPhotonProducer::~AODHIPhotonProducer() {}

void
AODHIPhotonProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::auto_ptr<aod::AODHIPhotonCollection> outputAODHIPhotonCollection (new aod::AODHIPhotonCollection);

  edm::Handle<reco::PhotonCollection> photons;
  evt.getByToken(photonProducer_, photons);

  for (reco::PhotonCollection::const_iterator phoItr = photons->begin(); phoItr != photons->end(); ++phoItr) {
    aod::AODHIPhoton newphoton(*phoItr);
    outputAODHIPhotonCollection->push_back(newphoton);
  }

  evt.put(outputAODHIPhotonCollection);
}

void AODHIPhotonProducer::beginRun (edm::Run const& r, edm::EventSetup const & es) {}
void AODHIPhotonProducer::endRun(edm::Run const&,  edm::EventSetup const&) {}

void
AODHIPhotonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(AODHIPhotonProducer);
