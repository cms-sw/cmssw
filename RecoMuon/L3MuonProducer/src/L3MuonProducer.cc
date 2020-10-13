/**  \class L3MuonProducer
 *
 *   L3 muon reconstructor:
 *   reconstructs muons using DT, CSC, RPC and tracker
 *   information,<BR>
 *   starting from a L2 reonstructed muon.
 *
 *   \author  A. Everett - Purdue University
 */

// Framework
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/L3MuonProducer/src/L3MuonProducer.h"

// TrackFinder and specific GLB Trajectory Builder
#include "RecoMuon/L3TrackFinder/interface/L3MuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"

using namespace edm;
using namespace std;

//
// constructor with config
//
L3MuonProducer::L3MuonProducer(const ParameterSet& parameterSet) {
  LogTrace("L3MuonProducer") << "constructor called" << endl;

  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("L3TrajBuilderParameters");

  // L2 Muon Collection Label
  theL2CollectionLabel = parameterSet.getParameter<InputTag>("MuonCollectionLabel");
  l2MuonToken_ = consumes<reco::TrackCollection>(theL2CollectionLabel);
  l2MuonTrajToken_ = consumes<std::vector<Trajectory>>(theL2CollectionLabel.label());
  l2AssoMapToken_ = consumes<TrajTrackAssociationCollection>(theL2CollectionLabel.label());
  updatedL2AssoMapToken_ = consumes<reco::TrackToTrackMap>(theL2CollectionLabel.label());

  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");

  // TrackLoader parameters
  ParameterSet trackLoaderParameters = parameterSet.getParameter<ParameterSet>("TrackLoaderParameters");

  // the services
  theService = std::make_unique<MuonServiceProxy>(serviceParameters, consumesCollector());
  ConsumesCollector iC = consumesCollector();

  // instantiate the concrete trajectory builder in the Track Finder
  auto mtl = std::make_unique<MuonTrackLoader>(trackLoaderParameters, iC, theService.get());
  auto l3mtb = std::make_unique<L3MuonTrajectoryBuilder>(trajectoryBuilderParameters, theService.get(), iC);
  theTrackFinder = std::make_unique<MuonTrackFinder>(std::move(l3mtb), std::move(mtl), iC);

  theL2SeededTkLabel =
      trackLoaderParameters.getUntrackedParameter<std::string>("MuonSeededTracksInstance", std::string());

  produces<reco::TrackCollection>(theL2SeededTkLabel);
  produces<TrackingRecHitCollection>(theL2SeededTkLabel);
  produces<reco::TrackExtraCollection>(theL2SeededTkLabel);
  produces<vector<Trajectory>>(theL2SeededTkLabel);
  produces<TrajTrackAssociationCollection>(theL2SeededTkLabel);

  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<vector<Trajectory>>();
  produces<TrajTrackAssociationCollection>();

  produces<reco::MuonTrackLinksCollection>();
}

//
// destructor
//
L3MuonProducer::~L3MuonProducer() { LogTrace("L3MuonProducer") << "destructor called" << endl; }

//
// reconstruct muons
//
void L3MuonProducer::produce(Event& event, const EventSetup& eventSetup) {
  const string metname = "Muon|RecoMuon|L3MuonProducer";
  LogTrace(metname) << endl << endl << endl;
  LogTrace(metname) << "L3 Muon Reconstruction started" << endl;

  typedef vector<Trajectory> TrajColl;

  // Update the services
  theService->update(eventSetup);

  // Take the L2 muon container(s)
  LogTrace(metname) << "Taking the L2 Muons " << theL2CollectionLabel << endl;

  Handle<reco::TrackCollection> L2Muons;
  event.getByToken(l2MuonToken_, L2Muons);

  Handle<vector<Trajectory>> L2MuonsTraj;
  vector<MuonTrajectoryBuilder::TrackCand> L2TrackCands;

  event.getByToken(l2MuonTrajToken_, L2MuonsTraj);

  edm::Handle<TrajTrackAssociationCollection> L2AssoMap;
  event.getByToken(l2AssoMapToken_, L2AssoMap);

  edm::Handle<reco::TrackToTrackMap> updatedL2AssoMap;
  event.getByToken(updatedL2AssoMapToken_, updatedL2AssoMap);

  for (TrajTrackAssociationCollection::const_iterator it = L2AssoMap->begin(); it != L2AssoMap->end(); ++it) {
    const Ref<vector<Trajectory>> traj = it->key;
    const reco::TrackRef tkRegular = it->val;
    reco::TrackRef tkUpdated;
    reco::TrackToTrackMap::const_iterator iEnd;
    reco::TrackToTrackMap::const_iterator iii;
    if (theL2CollectionLabel.instance() == "UpdatedAtVtx") {
      iEnd = updatedL2AssoMap->end();
      iii = updatedL2AssoMap->find(it->val);
      if (iii != iEnd)
        tkUpdated = (*updatedL2AssoMap)[it->val];
    }

    const reco::TrackRef tk = (tkUpdated.isNonnull()) ? tkUpdated : tkRegular;

    MuonTrajectoryBuilder::TrackCand L2Cand = MuonTrajectoryBuilder::TrackCand((Trajectory*)nullptr, tk);
    if (traj->isValid())
      L2Cand.first = &*traj;
    L2TrackCands.push_back(L2Cand);
  }

  theTrackFinder->reconstruct(L2TrackCands, event, eventSetup);

  LogTrace(metname) << "Event loaded"
                    << "================================" << endl
                    << endl;
}

void L3MuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;
    psd0.addUntracked<std::vector<std::string>>("Propagators",
                                                {
                                                    "hltESPSmartPropagatorAny",
                                                    "SteppingHelixPropagatorAny",
                                                    "hltESPSmartPropagator",
                                                    "hltESPSteppingHelixPropagatorOpposite",
                                                });
    psd0.add<bool>("RPCLayers", true);
    psd0.addUntracked<bool>("UseMuonNavigation", true);
    desc.add<edm::ParameterSetDescription>("ServiceParameters", psd0);
  }
  desc.add<edm::InputTag>("MuonCollectionLabel", edm::InputTag("hltL2Muons", "UpdatedAtVtx"));
  {
    edm::ParameterSetDescription psd0;
    psd0.addUntracked<bool>("PutTkTrackIntoEvent", false);
    psd0.add<std::string>("TTRHBuilder", "hltESPTTRHBWithTrackAngle");
    psd0.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
    psd0.addUntracked<bool>("SmoothTkTrack", false);
    psd0.addUntracked<std::string>("MuonSeededTracksInstance", "L2Seeded");
    psd0.add<std::string>("Smoother", "hltESPKFTrajectorySmootherForMuonTrackLoader");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("MaxChi2", 1000000.0);
      psd1.add<std::string>("Propagator", "hltESPSteppingHelixPropagatorOpposite");
      psd1.add<std::vector<double>>("BeamSpotPositionErrors",
                                    {
                                        0.1,
                                        0.1,
                                        5.3,
                                    });
      psd0.add<edm::ParameterSetDescription>("MuonUpdatorAtVertexParameters", psd1);
    }
    psd0.add<bool>("VertexConstraint", false);
    psd0.add<bool>("DoSmoothing", false);
    desc.add<edm::ParameterSetDescription>("TrackLoaderParameters", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<double>("ScaleTECyFactor", -1.0);
    psd0.add<edm::InputTag>("tkTrajVertex", edm::InputTag("hltPixelVertices"));
    psd0.add<bool>("tkTrajUseVertex", false);
    {
      edm::ParameterSetDescription psd1;
      psd1.add<int>("TrackerSkipSection", -1);
      psd1.add<bool>("DoPredictionsOnly", false);
      psd1.add<bool>("PropDirForCosmics", false);
      psd1.add<int>("HitThreshold", 1);
      psd1.add<int>("MuonHitsOption", 1);
      psd1.add<bool>("RefitFlag", true);
      psd1.add<std::string>("Fitter", "hltESPL3MuKFTrajectoryFitter");
      psd1.add<int>("SkipStation", -1);
      psd1.add<std::string>("TrackerRecHitBuilder", "hltESPTTRHBWithTrackAngle");
      psd1.add<double>("Chi2CutRPC", 1.0);
      psd1.add<std::string>("MuonRecHitBuilder", "hltESPMuonTransientTrackingRecHitBuilder");
      psd1.add<std::string>("RefitDirection", "insideOut");
      psd1.add<edm::InputTag>("CSCRecSegmentLabel", edm::InputTag("hltCscSegments"));
      psd1.add<edm::InputTag>("GEMRecHitLabel", edm::InputTag("gemRecHits"));
      psd1.add<edm::InputTag>("ME0RecHitLabel", edm::InputTag("me0Segments"));
      psd1.add<std::vector<int>>("DYTthrs",
                                 {
                                     30,
                                     15,
                                 });
      psd1.add<int>("DYTselector", 1);
      psd1.add<bool>("DYTupdator", false);
      psd1.add<bool>("DYTuseAPE", false);
      psd1.add<bool>("DYTuseThrsParametrization", true);
      {
        edm::ParameterSetDescription psd2;
        psd2.add<std::vector<double>>("eta0p8", {1, -0.919853, 0.990742});
        psd2.add<std::vector<double>>("eta1p2", {1, -0.897354, 0.987738});
        psd2.add<std::vector<double>>("eta2p0", {1, -0.986855, 0.998516});
        psd2.add<std::vector<double>>("eta2p2", {1, -0.940342, 0.992955});
        psd2.add<std::vector<double>>("eta2p4", {1, -0.947633, 0.993762});
        psd1.add<edm::ParameterSetDescription>("DYTthrsParameters", psd2);
      }
      psd1.add<double>("Chi2CutCSC", 150.0);
      psd1.add<double>("Chi2CutDT", 10.0);
      psd1.add<double>("Chi2CutGEM", 1.0);
      psd1.add<double>("Chi2CutME0", 1.0);
      psd1.add<bool>("RefitRPCHits", true);
      psd1.add<edm::InputTag>("DTRecSegmentLabel", edm::InputTag("hltDt4DSegments"));
      psd1.add<std::string>("Propagator", "hltESPSmartPropagatorAny");
      psd1.add<int>("TrackerSkipSystem", -1);
      psd0.add<edm::ParameterSetDescription>("GlbRefitterParameters", psd1);
    }
    psd0.add<double>("tkTrajMaxChi2", 9999.0);
    psd0.add<double>("ScaleTECxFactor", -1.0);
    psd0.add<std::string>("TrackerRecHitBuilder", "hltESPTTRHBWithTrackAngle");
    psd0.add<edm::InputTag>("tkTrajBeamSpot", edm::InputTag("hltOnlineBeamSpot"));
    psd0.add<std::string>("MuonRecHitBuilder", "hltESPMuonTransientTrackingRecHitBuilder");
    psd0.add<double>("tkTrajMaxDXYBeamSpot", 9999.0);
    psd0.add<std::string>("TrackerPropagator", "SteppingHelixPropagatorAny");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<bool>("precise", true);
      psd1.add<bool>("Eta_fixed", true);
      psd1.add<double>("Eta_min", 0.1);
      psd1.add<bool>("Z_fixed", false);
      psd1.add<edm::InputTag>("MeasurementTrackerName", edm::InputTag("hltESPMeasurementTracker"));
      psd1.add<int>("maxRegions", 2);
      psd1.add<double>("Pt_min", 3.0);
      psd1.add<double>("Rescale_Dz", 4.0);
      psd1.add<double>("PhiR_UpperLimit_Par1", 0.6);
      psd1.add<double>("PhiR_UpperLimit_Par2", 0.2);
      psd1.add<edm::InputTag>("vertexCollection", edm::InputTag("pixelVertices"));
      psd1.add<bool>("Phi_fixed", true);
      psd1.add<edm::InputTag>("input", edm::InputTag("hltL2Muons", "UpdatedAtVtx"));
      psd1.add<double>("DeltaR", 0.025);
      psd1.add<int>("OnDemand", -1);
      psd1.add<double>("DeltaZ", 24.2);
      psd1.add<double>("Rescale_phi", 3.0);
      psd1.add<double>("Rescale_eta", 3.0);
      psd1.add<double>("DeltaEta", 0.04);
      psd1.add<double>("DeltaPhi", 0.15);
      psd1.add<double>("Phi_min", 0.1);
      psd1.add<bool>("UseVertex", false);
      psd1.add<double>("EtaR_UpperLimit_Par1", 0.25);
      psd1.add<double>("EtaR_UpperLimit_Par2", 0.15);
      psd1.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
      psd1.add<double>("EscapePt", 3.0);
      psd1.add<bool>("Pt_fixed", false);
      psd0.add<edm::ParameterSetDescription>("MuonTrackingRegionBuilder", psd1);
    }
    psd0.add<bool>("RefitRPCHits", true);
    psd0.add<double>("PCut", 2.5);
    {
      edm::ParameterSetDescription psd1;
      TrackTransformer::fillPSetDescription(psd1,
                                            false,                                           // do predictions only
                                            "hltESPL3MuKFTrajectoryFitter",                  // fitter
                                            "hltESPKFTrajectorySmootherForMuonTrackLoader",  // smoother
                                            "hltESPSmartPropagatorAny",                      // propagator
                                            "insideOut",                                     // refit direction
                                            true,                                            // refit rpc hits
                                            "hltESPTTRHBWithTrackAngle",                     // tracker rechit builder
                                            "hltESPMuonTransientTrackingRecHitBuilder"       // muon rechit builder
      );
      psd0.add<edm::ParameterSetDescription>("TrackTransformer", psd1);
    }
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("Quality_3", 7.0);
      psd1.add<double>("DeltaRCut_1", 0.1);
      psd1.add<double>("MinP", 2.5);
      psd1.add<double>("MinPt", 1.0);
      psd1.add<double>("Quality_2", 15.0);
      psd1.add<double>("Pt_threshold2", 999999999.0);
      psd1.add<double>("LocChi2Cut", 0.001);
      psd1.add<double>("Eta_threshold", 1.2);
      psd1.add<double>("Pt_threshold1", 0.0);
      psd1.add<double>("Chi2Cut_1", 50.0);
      psd1.add<double>("Quality_1", 20.0);
      psd1.add<double>("Chi2Cut_3", 200.0);
      psd1.add<double>("DeltaRCut_3", 1.0);
      psd1.add<double>("DeltaRCut_2", 0.2);
      psd1.add<double>("DeltaDCut_1", 40.0);
      psd1.add<double>("DeltaDCut_2", 10.0);
      psd1.add<double>("DeltaDCut_3", 15.0);
      psd1.add<double>("Chi2Cut_2", 50.0);
      psd1.add<std::string>("Propagator", "hltESPSmartPropagator");
      psd0.add<edm::ParameterSetDescription>("GlobalMuonTrackMatcher", psd1);
    }
    psd0.add<double>("PtCut", 1.0);
    psd0.add<bool>("matchToSeeds", true);
    psd0.add<edm::InputTag>("tkTrajLabel", edm::InputTag("hltBRSMuonSeededTracksOutIn"));
    desc.add<edm::ParameterSetDescription>("L3TrajBuilderParameters", psd0);
  }
  descriptions.add("L3MuonProducer", desc);
}
