#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/Candidate/interface/Candidate.h"
typedef SimpleFlatTableProducer<reco::Candidate> SimpleCandidateFlatTableProducer;

#include "DataFormats/TrackReco/interface/Track.h"
typedef SimpleFlatTableProducer<reco::Track> SimpleTrackFlatTableProducer;

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
typedef SimpleFlatTableProducer<reco::SuperCluster> SimpleSuperclusterFlatTableProducer;

#include "DataFormats/JetReco/interface/PFJet.h"
typedef SimpleFlatTableProducer<reco::PFJet> SimplePFJetFlatTableProducer;

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

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleSuperclusterFlatTableProducer);
DEFINE_FWK_MODULE(SimplePFJetFlatTableProducer);
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
