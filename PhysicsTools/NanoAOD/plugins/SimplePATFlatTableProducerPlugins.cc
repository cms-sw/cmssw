#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
typedef SimpleFlatTableProducer<pat::Electron> SimplePATElectronFlatTableProducer;

#include "DataFormats/PatCandidates/interface/Muon.h"
typedef SimpleFlatTableProducer<pat::Muon> SimplePATMuonFlatTableProducer;

#include "DataFormats/PatCandidates/interface/Tau.h"
typedef SimpleFlatTableProducer<pat::Tau> SimplePATTauFlatTableProducer;

#include "DataFormats/PatCandidates/interface/Photon.h"
typedef SimpleFlatTableProducer<pat::Photon> SimplePATPhotonFlatTableProducer;

#include "DataFormats/PatCandidates/interface/Jet.h"
typedef SimpleFlatTableProducer<pat::Jet> SimplePATJetFlatTableProducer;

#include "DataFormats/PatCandidates/interface/IsolatedTrack.h"
typedef SimpleFlatTableProducer<pat::IsolatedTrack> SimplePATIsolatedTrackFlatTableProducer;

#include "DataFormats/PatCandidates/interface/GenericParticle.h"
typedef SimpleFlatTableProducer<pat::GenericParticle> SimplePATGenericParticleFlatTableProducer;

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
typedef SimpleFlatTableProducer<pat::PackedCandidate> SimplePATCandidateFlatTableProducer;

#include "DataFormats/PatCandidates/interface/MET.h"
typedef SimpleFlatTableProducer<pat::MET> SimplePATMETFlatTableProducer;

#include "DataFormats/VertexReco/interface/TrackTimeLifeInfo.h"
typedef SimpleTypedExternalFlatTableProducer<pat::Electron, TrackTimeLifeInfo>
    SimplePATElectron2TrackTimeLifeInfoFlatTableProducer;
typedef SimpleTypedExternalFlatTableProducer<pat::Muon, TrackTimeLifeInfo>
    SimplePATMuon2TrackTimeLifeInfoFlatTableProducer;
typedef SimpleTypedExternalFlatTableProducer<pat::Tau, TrackTimeLifeInfo>
    SimplePATTau2TrackTimeLifeInfoFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimplePATElectronFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATMuonFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATTauFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATPhotonFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATJetFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATIsolatedTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATGenericParticleFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATMETFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATElectron2TrackTimeLifeInfoFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATMuon2TrackTimeLifeInfoFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATTau2TrackTimeLifeInfoFlatTableProducer);
