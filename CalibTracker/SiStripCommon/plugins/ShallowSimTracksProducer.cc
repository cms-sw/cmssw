#include "ShallowSimTracksProducer.h"
#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

ShallowSimTracksProducer::ShallowSimTracksProducer(const edm::ParameterSet& conf)
    : Prefix(conf.getParameter<std::string>("Prefix")),
      Suffix(conf.getParameter<std::string>("Suffix")),
      trackingParticles_token_(
          consumes<TrackingParticleCollection>(conf.getParameter<edm::InputTag>("TrackingParticles"))),
      associator_token_(
          consumes<reco::TrackToTrackingParticleAssociator>(conf.getParameter<edm::InputTag>("Associator"))),
      tracks_token_(consumes<edm::View<reco::Track>>(conf.getParameter<edm::InputTag>("Tracks"))) {
  produces<std::vector<unsigned>>(Prefix + "multi" + Suffix);
  produces<std::vector<int>>(Prefix + "type" + Suffix);
  produces<std::vector<float>>(Prefix + "charge" + Suffix);
  produces<std::vector<float>>(Prefix + "momentum" + Suffix);
  produces<std::vector<float>>(Prefix + "pt" + Suffix);
  produces<std::vector<double>>(Prefix + "theta" + Suffix);
  produces<std::vector<double>>(Prefix + "phi" + Suffix);
  produces<std::vector<double>>(Prefix + "eta" + Suffix);
  produces<std::vector<double>>(Prefix + "qoverp" + Suffix);
  produces<std::vector<double>>(Prefix + "vx" + Suffix);
  produces<std::vector<double>>(Prefix + "vy" + Suffix);
  produces<std::vector<double>>(Prefix + "vz" + Suffix);
}

void ShallowSimTracksProducer::produce(edm::Event& event, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<reco::Track>> tracks;
  event.getByToken(tracks_token_, tracks);
  edm::Handle<TrackingParticleCollection> trackingParticles;
  event.getByToken(trackingParticles_token_, trackingParticles);
  edm::Handle<reco::TrackToTrackingParticleAssociator> associator;
  event.getByToken(associator_token_, associator);

  unsigned size = tracks->size();
  auto multi = std::make_unique<std::vector<unsigned>>(size, 0);
  auto type = std::make_unique<std::vector<int>>(size, 0);
  auto charge = std::make_unique<std::vector<float>>(size, 0);
  auto momentum = std::make_unique<std::vector<float>>(size, -1);
  auto pt = std::make_unique<std::vector<float>>(size, -1);
  auto theta = std::make_unique<std::vector<double>>(size, -1000);
  auto phi = std::make_unique<std::vector<double>>(size, -1000);
  auto eta = std::make_unique<std::vector<double>>(size, -1000);
  auto dxy = std::make_unique<std::vector<double>>(size, -1000);
  auto dsz = std::make_unique<std::vector<double>>(size, -1000);
  auto qoverp = std::make_unique<std::vector<double>>(size, -1000);
  auto vx = std::make_unique<std::vector<double>>(size, -1000);
  auto vy = std::make_unique<std::vector<double>>(size, -1000);
  auto vz = std::make_unique<std::vector<double>>(size, -1000);

  reco::RecoToSimCollection associations = associator->associateRecoToSim(tracks, trackingParticles);

  for (reco::RecoToSimCollection::const_iterator association = associations.begin(); association != associations.end();
       association++) {
    const reco::Track* track = association->key.get();
    const int matches = association->val.size();
    if (matches > 0) {
      const TrackingParticle* tparticle = association->val[0].first.get();
      unsigned i = shallow::findTrackIndex(tracks, track);

      multi->at(i) = matches;
      type->at(i) = tparticle->pdgId();
      charge->at(i) = tparticle->charge();
      momentum->at(i) = tparticle->p();
      pt->at(i) = tparticle->pt();
      theta->at(i) = tparticle->theta();
      phi->at(i) = tparticle->phi();
      eta->at(i) = tparticle->eta();
      qoverp->at(i) = tparticle->charge() / tparticle->p();

      const TrackingVertex* tvertex = tparticle->parentVertex().get();
      vx->at(i) = tvertex->position().x();
      vy->at(i) = tvertex->position().y();
      vz->at(i) = tvertex->position().z();
    }
  }

  event.put(std::move(multi), Prefix + "multi" + Suffix);
  event.put(std::move(type), Prefix + "type" + Suffix);
  event.put(std::move(charge), Prefix + "charge" + Suffix);
  event.put(std::move(momentum), Prefix + "momentum" + Suffix);
  event.put(std::move(pt), Prefix + "pt" + Suffix);
  event.put(std::move(theta), Prefix + "theta" + Suffix);
  event.put(std::move(phi), Prefix + "phi" + Suffix);
  event.put(std::move(eta), Prefix + "eta" + Suffix);
  event.put(std::move(qoverp), Prefix + "qoverp" + Suffix);
  event.put(std::move(vx), Prefix + "vx" + Suffix);
  event.put(std::move(vy), Prefix + "vy" + Suffix);
  event.put(std::move(vz), Prefix + "vz" + Suffix);
}
