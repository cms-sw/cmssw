#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
typedef SimpleFlatTableProducer<pat::CompositeCandidate> SimpleCompositeCandidateFlatTableProducer;

//not really useful in the end because lowptgsf tracks come with BDT taht is not part of GsfTracks
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
typedef SimpleFlatTableProducer<reco::GsfTrack> SimpleGsfTrackFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleCompositeCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleGsfTrackFlatTableProducer);
