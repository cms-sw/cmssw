#include "PhysicsTools/NanoAOD/interface/SimpleFlatTableProducer.h"

#include "DataFormats/Candidate/interface/Candidate.h"
typedef SimpleFlatTableProducer<reco::Candidate> SimpleCandidateFlatTableProducer;

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

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleCandidateFlatTableProducer);
DEFINE_FWK_MODULE(SimpleGenEventFlatTableProducer);
DEFINE_FWK_MODULE(SimpleGenFilterFlatTableProducerLumi);
DEFINE_FWK_MODULE(SimpleHTXSFlatTableProducer);
DEFINE_FWK_MODULE(SimpleProtonTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleLocalTrackFlatTableProducer);
DEFINE_FWK_MODULE(SimpleXYZPointFlatTableProducer);
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
