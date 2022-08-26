//-------------------------------------------------
//
/**  \class L2MuonProducer
 * 
 *   L2 muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from internal seeds (L2 muon track segments).
 *
 *
 *
 *   \author  R.Bellan - INFN TO
 *
 *   modified by A. Sharma to add fillDescription function
 */
//
//--------------------------------------------------

#include <memory>
#include <string>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoMuon/StandAloneTrackFinder/interface/ExhaustiveMuonTrajectoryBuilder.h"
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

class L2MuonProducer : public edm::stream::EDProducer<> {
public:
  /// constructor with config
  L2MuonProducer(const edm::ParameterSet&);

  /// destructor
  ~L2MuonProducer() override;

  /// reconstruct muons
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // MuonSeed Collection Label
  edm::InputTag theSeedCollectionLabel;

  /// the track finder
  std::unique_ptr<MuonTrackFinder> theTrackFinder;  //It isn't the same as in ORCA

  /// the event setup proxy, it takes care the services update
  std::unique_ptr<MuonServiceProxy> theService;

  edm::EDGetTokenT<edm::View<TrajectorySeed>> seedsToken;
};

/// constructor with config
L2MuonProducer::L2MuonProducer(const edm::ParameterSet& parameterSet) {
  LogTrace("Muon|RecoMuon|L2MuonProducer") << "constructor called" << endl;

  // Parameter set for the Builder
  edm::ParameterSet trajectoryBuilderParameters =
      parameterSet.getParameter<edm::ParameterSet>("L2TrajBuilderParameters");

  // MuonSeed Collection Label
  theSeedCollectionLabel = parameterSet.getParameter<edm::InputTag>("InputObjects");
  seedsToken = consumes<edm::View<TrajectorySeed>>(theSeedCollectionLabel);
  // service parameters
  edm::ParameterSet serviceParameters = parameterSet.getParameter<edm::ParameterSet>("ServiceParameters");

  // TrackLoader parameters
  edm::ParameterSet trackLoaderParameters = parameterSet.getParameter<edm::ParameterSet>("TrackLoaderParameters");

  // the services
  theService = std::make_unique<MuonServiceProxy>(serviceParameters, consumesCollector());

  std::unique_ptr<MuonTrajectoryBuilder> trajectoryBuilder = nullptr;
  // instantiate the concrete trajectory builder in the Track Finder

  edm::ConsumesCollector iC = consumesCollector();
  string typeOfBuilder = parameterSet.getParameter<string>("MuonTrajectoryBuilder");
  if (typeOfBuilder == "StandAloneMuonTrajectoryBuilder" || typeOfBuilder.empty())
    trajectoryBuilder =
        std::make_unique<StandAloneMuonTrajectoryBuilder>(trajectoryBuilderParameters, theService.get(), iC);
  else if (typeOfBuilder == "Exhaustive")
    trajectoryBuilder =
        std::make_unique<ExhaustiveMuonTrajectoryBuilder>(trajectoryBuilderParameters, theService.get(), iC);
  else {
    edm::LogWarning("Muon|RecoMuon|StandAloneMuonProducer")
        << "No Trajectory builder associated with " << typeOfBuilder
        << ". Falling down to the default (StandAloneMuonTrajectoryBuilder)";
    trajectoryBuilder =
        std::make_unique<StandAloneMuonTrajectoryBuilder>(trajectoryBuilderParameters, theService.get(), iC);
  }
  theTrackFinder =
      std::make_unique<MuonTrackFinder>(std::move(trajectoryBuilder),
                                        std::make_unique<MuonTrackLoader>(trackLoaderParameters, iC, theService.get()),
                                        std::make_unique<MuonTrajectoryCleaner>(true),
                                        iC);

  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  // TrackCollection refers to TrackingRechit and TrackExtra
  // collections, need to declare its production after them to work
  // around a rare race condition in framework scheduling
  produces<reco::TrackCollection>();
  produces<reco::TrackCollection>("UpdatedAtVtx");
  produces<reco::TrackToTrackMap>();

  produces<std::vector<Trajectory>>();
  produces<TrajTrackAssociationCollection>();

  produces<edm::AssociationMap<edm::OneToMany<std::vector<L2MuonTrajectorySeed>, std::vector<L2MuonTrajectorySeed>>>>();
}

/// destructor
L2MuonProducer::~L2MuonProducer() {
  LogTrace("Muon|RecoMuon|L2eMuonProducer") << "L2MuonProducer destructor called" << endl;
}

/// reconstruct muons
void L2MuonProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  const std::string metname = "Muon|RecoMuon|L2MuonProducer";

  LogTrace(metname) << endl << endl << endl;
  LogTrace(metname) << "L2 Muon Reconstruction Started" << endl;

  // Take the seeds container
  LogTrace(metname) << "Taking the seeds: " << theSeedCollectionLabel.label() << endl;
  edm::Handle<edm::View<TrajectorySeed>> seeds;
  event.getByToken(seedsToken, seeds);

  // Update the services
  theService->update(eventSetup);

  // Reconstruct
  LogTrace(metname) << "Track Reconstruction" << endl;
  theTrackFinder->reconstruct(seeds, event, eventSetup);

  LogTrace(metname) << "edm::Event loaded"
                    << "================================" << endl
                    << endl;
}

void L2MuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;
    psd0.addUntracked<std::vector<std::string>>("Propagators",
                                                {
                                                    "hltESPFastSteppingHelixPropagatorAny"
                                                    "hltESPFastSteppingHelixPropagatorOpposite",
                                                });
    psd0.add<bool>("RPCLayers", true);
    psd0.addUntracked<bool>("UseMuonNavigation", true);
    desc.add<edm::ParameterSetDescription>("ServiceParameters", psd0);
  }

  desc.add<edm::InputTag>("InputObjects", edm::InputTag("hltL2MuonSeeds"));
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("Fitter", "hltESPKFFittingSmootherForL2Muon");
    psd0.add<std::string>("MuonRecHitBuilder", "hltESPMuonTransientTrackingRecHitBuilder");
    psd0.add<unsigned int>("NMinRecHits", 2);
    psd0.add<bool>("UseSubRecHits", false);
    psd0.add<std::string>("Propagator", "hltESPFastSteppingHelixPropagatorAny");
    psd0.add<double>("RescaleError", 100.0);
    desc.add<edm::ParameterSetDescription>("SeedTransformerParameters", psd0);
  }

  {
    edm::ParameterSetDescription psd0;
    psd0.add<bool>("DoRefit", false);
    psd0.add<std::string>("SeedPropagator", "hltESPFastSteppingHelixPropagatorAny");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("NumberOfSigma", 3.0);
      psd1.add<std::string>("FitDirection", "insideOut");
      psd1.add<edm::InputTag>("DTRecSegmentLabel", edm::InputTag("hltDt4DSegments"));
      psd1.add<double>("MaxChi2", 1000.0);
      {
        edm::ParameterSetDescription psd2;
        psd2.add<double>("MaxChi2", 25.0);
        psd2.add<double>("RescaleErrorFactor", 100.0);
        psd2.add<int>("Granularity", 0);
        psd2.add<bool>("ExcludeRPCFromFit", false);
        psd2.add<bool>("UseInvalidHits", true);
        psd2.add<bool>("RescaleError", false);
        psd1.add<edm::ParameterSetDescription>("MuonTrajectoryUpdatorParameters", psd2);
      }
      psd1.add<bool>("EnableRPCMeasurement", true);
      psd1.add<edm::InputTag>("CSCRecSegmentLabel", edm::InputTag("hltCscSegments"));
      psd1.add<bool>("EnableDTMeasurement", true);
      psd1.add<edm::InputTag>("RPCRecSegmentLabel", edm::InputTag("hltRpcRecHits"));
      psd1.add<std::string>("Propagator", "hltESPFastSteppingHelixPropagatorAny");
      psd1.add<bool>("EnableGEMMeasurement", false);
      psd1.add<edm::InputTag>("GEMRecSegmentLabel", edm::InputTag("gemRecHits"));
      psd1.add<bool>("EnableME0Measurement", false);
      psd1.add<edm::InputTag>("ME0RecSegmentLabel", edm::InputTag("me0Segments"));
      psd1.add<bool>("EnableCSCMeasurement", true);
      psd0.add<edm::ParameterSetDescription>("FilterParameters", psd1);
    }
    psd0.add<std::string>("NavigationType", "Standard");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<std::string>("Fitter", "hltESPKFFittingSmootherForL2Muon");
      psd1.add<std::string>("MuonRecHitBuilder", "hltESPMuonTransientTrackingRecHitBuilder");
      psd1.add<unsigned int>("NMinRecHits", 2);
      psd1.add<bool>("UseSubRecHits", false);
      psd1.add<std::string>("Propagator", "hltESPFastSteppingHelixPropagatorAny");
      psd1.add<double>("RescaleError", 100.0);
      psd0.add<edm::ParameterSetDescription>("SeedTransformerParameters", psd1);
    }
    psd0.add<bool>("DoBackwardFilter", true);
    psd0.add<std::string>("SeedPosition", "in");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("NumberOfSigma", 3.0);
      psd1.add<edm::InputTag>("CSCRecSegmentLabel", edm::InputTag("hltCscSegments"));
      psd1.add<std::string>("FitDirection", "outsideIn");
      psd1.add<edm::InputTag>("DTRecSegmentLabel", edm::InputTag("hltDt4DSegments"));
      psd1.add<double>("MaxChi2", 100.0);
      {
        edm::ParameterSetDescription psd2;
        psd2.add<double>("MaxChi2", 25.0);
        psd2.add<double>("RescaleErrorFactor", 100.0);
        psd2.add<int>("Granularity", 0);
        psd2.add<bool>("ExcludeRPCFromFit", false);
        psd2.add<bool>("UseInvalidHits", true);
        psd2.add<bool>("RescaleError", false);
        psd1.add<edm::ParameterSetDescription>("MuonTrajectoryUpdatorParameters", psd2);
      }
      psd1.add<bool>("EnableRPCMeasurement", true);
      psd1.add<std::string>("BWSeedType", "fromGenerator");
      psd1.add<bool>("EnableDTMeasurement", true);
      psd1.add<edm::InputTag>("RPCRecSegmentLabel", edm::InputTag("hltRpcRecHits"));
      psd1.add<std::string>("Propagator", "hltESPFastSteppingHelixPropagatorAny");
      psd1.add<bool>("EnableGEMMeasurement", false);
      psd1.add<edm::InputTag>("GEMRecSegmentLabel", edm::InputTag("gemRecHits"));
      psd1.add<bool>("EnableME0Measurement", false);
      psd1.add<edm::InputTag>("ME0RecSegmentLabel", edm::InputTag("me0Segments"));
      psd1.add<bool>("EnableCSCMeasurement", true);
      psd0.add<edm::ParameterSetDescription>("BWFilterParameters", psd1);
    }
    psd0.add<bool>("DoSeedRefit", false);
    desc.add<edm::ParameterSetDescription>("L2TrajBuilderParameters", psd0);
  }
  desc.add<bool>("DoSeedRefit", false);
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("Smoother", "hltESPKFTrajectorySmootherForMuonTrackLoader");
    psd0.add<bool>("DoSmoothing", false);
    psd0.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("MaxChi2", 1000000.0);
      psd1.add<std::vector<double>>("BeamSpotPosition",
                                    {
                                        0.0,
                                        0.0,
                                        0.0,
                                    });
      psd1.add<std::string>("Propagator", "hltESPFastSteppingHelixPropagatorOpposite");
      psd1.add<std::vector<double>>("BeamSpotPositionErrors",
                                    {
                                        0.1,
                                        0.1,
                                        5.3,
                                    });
      psd0.add<edm::ParameterSetDescription>("MuonUpdatorAtVertexParameters", psd1);
    }
    psd0.add<bool>("VertexConstraint", true);
    psd0.add<std::string>("TTRHBuilder", "hltESPTTRHBWithTrackAngle");
    desc.add<edm::ParameterSetDescription>("TrackLoaderParameters", psd0);
  }
  desc.add<std::string>("MuonTrajectoryBuilder", "Exhaustive");
  descriptions.add("L2MuonProducer", desc);
}

// declare as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L2MuonProducer);
