#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/Candidate/interface/Candidate.h"
typedef SimpleFlatTableProducer<reco::Candidate> SimpleCandidateFlatTableProducer;

#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
typedef SimpleFlatTableProducer<SimCluster> SimpleSimClusterFlatTableProducer;

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
typedef SimpleFlatTableProducer<PCaloHit> SimplePCaloHitFlatTableProducer;

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
typedef SimpleFlatTableProducer<PSimHit> SimplePSimHitFlatTableProducer;

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
typedef SimpleFlatTableProducer<TrackingParticle> SimpleTrackingParticleFlatTableProducer;
#include "SimDataFormats/PFAnalysis/interface/PFParticle.h"
typedef SimpleFlatTableProducer<PFParticle> SimplePFParticleFlatTableProducer;
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
typedef SimpleFlatTableProducer<CaloParticle> SimpleCaloParticleFlatTableProducer;
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
typedef SimpleFlatTableProducer<reco::PFCandidate> SimplePFCandidateFlatTableProducer;

#include "SimDataFormats/Track/interface/SimTrack.h"
typedef SimpleFlatTableProducer<SimTrack> SimpleSimTrackFlatTableProducer;

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
typedef EventSingletonSimpleFlatTableProducer<GenEventInfoProduct> SimpleGenEventFlatTableProducer;

#include "SimDataFormats/HTXS/interface/HiggsTemplateCrossSections.h"
typedef EventSingletonSimpleFlatTableProducer<HTXS::HiggsClassification> SimpleHTXSFlatTableProducer;

#include "DataFormats/ProtonReco/interface/ForwardProton.h"
typedef SimpleFlatTableProducer<reco::ForwardProton> SimpleProtonTrackFlatTableProducer;

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
typedef SimpleFlatTableProducer<CTPPSLocalTrackLite> SimpleLocalTrackFlatTableProducer;

#include "DataFormats/Math/interface/Point3D.h"
typedef EventSingletonSimpleFlatTableProducer<math::XYZPointF> SimpleXYZPointFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimplePCaloHitFlatTableProducer);
DEFINE_FWK_MODULE(SimplePSimHitFlatTableProducer);
DEFINE_FWK_MODULE(SimpleSimTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleSimClusterFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTrackingParticleFlatTableProducer);
DEFINE_FWK_MODULE(SimplePFParticleFlatTableProducer);
DEFINE_FWK_MODULE(SimplePFCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleCaloParticleFlatTableProducer);
DEFINE_FWK_MODULE(SimpleCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleGenEventFlatTableProducer);
DEFINE_FWK_MODULE(SimpleHTXSFlatTableProducer);
<<<<<<< HEAD
DEFINE_FWK_MODULE(SimpleProtonTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleLocalTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleXYZPointFlatTableProducer);
=======
DEFINE_FWK_MODULE(SimpleXYZPointFlatTableProducer);
>>>>>>> Add PFParticle outline and ntuples
