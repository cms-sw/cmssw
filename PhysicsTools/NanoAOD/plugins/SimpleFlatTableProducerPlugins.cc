#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/Candidate/interface/Candidate.h"
typedef SimpleFlatTableProducer<reco::Candidate> SimpleCandidateFlatTableProducer;

#include "DataFormats/TrackReco/interface/Track.h"
typedef SimpleFlatTableProducer<reco::Track> SimpleTrackFlatTableProducer;

#include "DataFormats/JetReco/interface/PFJet.h"
typedef SimpleFlatTableProducer<reco::PFJet> SimplePFJetFlatTableProducer;

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
typedef SimpleFlatTableProducer<reco::VertexCompositePtrCandidate> SimpleSecondaryVertexFlatTableProducer;

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
typedef SimpleFlatTableProducer<reco::GenParticle> SimpleGenParticleFlatTableProducer;

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

#include "DataFormats/L1Trigger/interface/EGamma.h"
typedef BXVectorSimpleFlatTableProducer<l1t::EGamma> SimpleTriggerL1EGFlatTableProducer;

#include "DataFormats/L1Trigger/interface/Jet.h"
typedef BXVectorSimpleFlatTableProducer<l1t::Jet> SimpleTriggerL1JetFlatTableProducer;

#include "DataFormats/L1Trigger/interface/Tau.h"
typedef BXVectorSimpleFlatTableProducer<l1t::Tau> SimpleTriggerL1TauFlatTableProducer;

#include "DataFormats/L1Trigger/interface/Muon.h"
typedef BXVectorSimpleFlatTableProducer<l1t::Muon> SimpleTriggerL1MuonFlatTableProducer;

#include "DataFormats/L1Trigger/interface/EtSum.h"
typedef BXVectorSimpleFlatTableProducer<l1t::EtSum> SimpleTriggerL1EtSumFlatTableProducer;

#include "DataFormats/Scouting/interface/Run3ScoutingVertex.h"
typedef SimpleFlatTableProducer<Run3ScoutingVertex> SimpleRun3ScoutingVertexFlatTableProducer;

#include "DataFormats/Scouting/interface/Run3ScoutingPhoton.h"
typedef SimpleFlatTableProducer<Run3ScoutingPhoton> SimpleRun3ScoutingPhotonFlatTableProducer;

#include "DataFormats/Scouting/interface/Run3ScoutingMuon.h"
typedef SimpleFlatTableProducer<Run3ScoutingMuon> SimpleRun3ScoutingMuonFlatTableProducer;

#include "DataFormats/Scouting/interface/Run3ScoutingElectron.h"
typedef SimpleFlatTableProducer<Run3ScoutingElectron> SimpleRun3ScoutingElectronFlatTableProducer;

#include "DataFormats/Scouting/interface/Run3ScoutingTrack.h"
typedef SimpleFlatTableProducer<Run3ScoutingTrack> SimpleRun3ScoutingTrackFlatTableProducer;

#include "DataFormats/VertexReco/interface/Vertex.h"
typedef SimpleFlatTableProducer<reco::Vertex> SimpleVertexFlatTableProducer;

#include "DataFormats/VertexReco/interface/TrackTimeLifeInfo.h"
typedef SimpleTypedExternalFlatTableProducer<reco::Candidate, TrackTimeLifeInfo>
    SimpleCandidate2TrackTimeLifeInfoFlatTableProducer;
typedef SimpleTypedExternalFlatTableProducer<pat::Electron, TrackTimeLifeInfo>
    SimplePATElectron2TrackTimeLifeInfoFlatTableProducer;
typedef SimpleTypedExternalFlatTableProducer<pat::Muon, TrackTimeLifeInfo>
    SimplePATMuon2TrackTimeLifeInfoFlatTableProducer;
typedef SimpleTypedExternalFlatTableProducer<pat::Tau, TrackTimeLifeInfo>
    SimplePATTau2TrackTimeLifeInfoFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimplePFJetFlatTableProducer);
DEFINE_FWK_MODULE(SimpleSecondaryVertexFlatTableProducer);
DEFINE_FWK_MODULE(SimpleGenParticleFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATElectronFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATMuonFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATTauFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATPhotonFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATJetFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATIsolatedTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATGenericParticleFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATMETFlatTableProducer);
DEFINE_FWK_MODULE(SimpleCandidate2CandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleGenEventFlatTableProducer);
DEFINE_FWK_MODULE(SimpleGenFilterFlatTableProducerLumi);
DEFINE_FWK_MODULE(SimpleHTXSFlatTableProducer);
DEFINE_FWK_MODULE(SimpleProtonTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleLocalTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleXYZPointFlatTableProducer);
DEFINE_FWK_MODULE(SimpleOnlineLuminosityFlatTableProducer);
DEFINE_FWK_MODULE(SimpleBeamspotFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1EGFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1JetFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1MuonFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1TauFlatTableProducer);
DEFINE_FWK_MODULE(SimpleTriggerL1EtSumFlatTableProducer);
DEFINE_FWK_MODULE(SimpleRun3ScoutingVertexFlatTableProducer);
DEFINE_FWK_MODULE(SimpleRun3ScoutingPhotonFlatTableProducer);
DEFINE_FWK_MODULE(SimpleRun3ScoutingMuonFlatTableProducer);
DEFINE_FWK_MODULE(SimpleRun3ScoutingElectronFlatTableProducer);
DEFINE_FWK_MODULE(SimpleRun3ScoutingTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleVertexFlatTableProducer);
DEFINE_FWK_MODULE(SimpleCandidate2TrackTimeLifeInfoFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATElectron2TrackTimeLifeInfoFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATMuon2TrackTimeLifeInfoFlatTableProducer);
DEFINE_FWK_MODULE(SimplePATTau2TrackTimeLifeInfoFlatTableProducer);
