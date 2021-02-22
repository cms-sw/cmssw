#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
typedef SimpleFlatTableProducer<PSimHit> SimplePSimHitFlatTableProducer;

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
typedef SimpleFlatTableProducer<TrackingParticle> SimpleTrackingParticleFlatTableProducer;

#include "SimDataFormats/Track/interface/SimTrack.h"
typedef SimpleFlatTableProducer<SimTrack> SimpleSimTrackFlatTableProducer;
#include "DataFormats/TrackReco/interface/Track.h"
typedef SimpleFlatTableProducer<reco::Track> SimpleTrackFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimplePSimHitFlatTableProducer);
DEFINE_FWK_MODULE(SimpleSimTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTrackingParticleFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTrackFlatTableProducer);
