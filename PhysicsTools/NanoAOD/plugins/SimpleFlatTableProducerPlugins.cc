#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"
typedef SimpleFlatTableProducer<reco::Candidate> SimpleCandidateFlatTableProducer;

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
typedef SimpleFlatTableProducer<reco::PFCandidate> SimplePFCandidateFlatTableProducer;

#include "DataFormats/TauReco/interface/PFTau.h"
typedef SimpleFlatTableProducer<reco::PFTau> SimplePFTauCandidateFlatTableProducer;

typedef SimpleCollectionFlatTableProducer<reco::Candidate> SimpleCandidateCollectionFlatTableProducer;

#include "DataFormats/TrackReco/interface/Track.h"
typedef SimpleFlatTableProducer<reco::Track> SimpleTrackFlatTableProducer;

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
typedef SimpleFlatTableProducer<reco::SuperCluster> SimpleSuperclusterFlatTableProducer;

#include "DataFormats/JetReco/interface/PFJet.h"
typedef SimpleFlatTableProducer<reco::PFJet> SimplePFJetFlatTableProducer;

#include "DataFormats/JetReco/interface/GenJet.h"
typedef SimpleFlatTableProducer<reco::GenJet> SimpleGenJetFlatTableProducer;

#include "DataFormats/VertexReco/interface/Vertex.h"
typedef SimpleFlatTableProducer<reco::Vertex> SimpleVertexFlatTableProducer;

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
typedef SimpleFlatTableProducer<reco::VertexCompositePtrCandidate> SimpleSecondaryVertexFlatTableProducer;

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
typedef SimpleFlatTableProducer<reco::GenParticle> SimpleGenParticleFlatTableProducer;

typedef SimpleTypedExternalFlatTableProducer<reco::Candidate, reco::Candidate>
    SimpleCandidate2CandidateFlatTableProducer;

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
typedef EventSingletonSimpleFlatTableProducer<GenEventInfoProduct> SimpleGenEventFlatTableProducer;

#include "SimDataFormats/GeneratorProducts/interface/GenFilterInfo.h"
typedef LumiSingletonSimpleFlatTableProducer<GenFilterInfo> SimpleGenFilterFlatTableProducerLumi;

#include "SimDataFormats/HTXS/interface/HiggsTemplateCrossSections.h"
typedef EventSingletonSimpleFlatTableProducer<HTXS::HiggsClassification> SimpleHTXSFlatTableProducer;

#include "DataFormats/ProtonReco/interface/ForwardProton.h"
typedef SimpleFlatTableProducer<reco::ForwardProton> SimpleProtonTrackFlatTableProducer;

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
typedef SimpleFlatTableProducer<CTPPSLocalTrackLite> SimpleLocalTrackFlatTableProducer;

#include "DataFormats/Math/interface/Point3D.h"
typedef EventSingletonSimpleFlatTableProducer<math::XYZPointF> SimpleXYZPointFlatTableProducer;

#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
typedef EventSingletonSimpleFlatTableProducer<OnlineLuminosityRecord> SimpleOnlineLuminosityFlatTableProducer;

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
typedef EventSingletonSimpleFlatTableProducer<reco::BeamSpot> SimpleBeamspotFlatTableProducer;

#include "DataFormats/TrackReco/interface/Track.h"
typedef SimpleFlatTableProducer<TrajectorySeed> SimpleTrajectorySeedFlatTableProducer;

#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
typedef SimpleFlatTableProducer<L2MuonTrajectorySeed> SimpleL2MuonTrajectorySeedFlatTableProducer;

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
typedef SimpleFlatTableProducer<reco::Track> SimpleTriggerTrackFlatTableProducer;

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
typedef SimpleFlatTableProducer<reco::GsfTrack> SimpleGsfTrackFlatTableProducer;

#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
typedef SimpleFlatTableProducer<pat::CompositeCandidate> SimpleCompositeCandidateFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimplePFCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimplePFTauCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleCandidateCollectionFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleSuperclusterFlatTableProducer);
DEFINE_FWK_MODULE(SimplePFJetFlatTableProducer);
DEFINE_FWK_MODULE(SimpleGenJetFlatTableProducer);
DEFINE_FWK_MODULE(SimpleVertexFlatTableProducer);
DEFINE_FWK_MODULE(SimpleSecondaryVertexFlatTableProducer);
DEFINE_FWK_MODULE(SimpleGenParticleFlatTableProducer);
DEFINE_FWK_MODULE(SimpleCandidate2CandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleGenEventFlatTableProducer);
DEFINE_FWK_MODULE(SimpleGenFilterFlatTableProducerLumi);
DEFINE_FWK_MODULE(SimpleHTXSFlatTableProducer);
DEFINE_FWK_MODULE(SimpleProtonTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleLocalTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleXYZPointFlatTableProducer);
DEFINE_FWK_MODULE(SimpleOnlineLuminosityFlatTableProducer);
DEFINE_FWK_MODULE(SimpleBeamspotFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTrajectorySeedFlatTableProducer);
DEFINE_FWK_MODULE(SimpleL2MuonTrajectorySeedFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleGsfTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleCompositeCandidateFlatTableProducer);
