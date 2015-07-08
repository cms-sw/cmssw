//-------------------------------------------------
//
/**  \class L2MuonProducer
 * 
 *   Level-2 muon reconstructor:
 *   reconstructs muons using DT, CSC and RPC
 *   information,<BR>
 *   starting from Level-1 trigger seeds.
 *
 *
 *
 *   \author  R.Bellan - INFN TO
 *
 *   modified by A. Sharma to add fillDescription function
 */
//
//--------------------------------------------------

#include "RecoMuon/L2MuonProducer/src/L2MuonProducer.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// TrackFinder and Specific STA/L2 Trajectory Builder
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneTrajectoryBuilder.h"
#include "RecoMuon/StandAloneTrackFinder/interface/ExhaustiveMuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrackLoader.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackToTrackMap.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"

#include <string>

using namespace edm;
using namespace std;

/// constructor with config
L2MuonProducer::L2MuonProducer(const ParameterSet& parameterSet){
  LogTrace("Muon|RecoMuon|L2MuonProducer")<<"constructor called"<<endl;

  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("L2TrajBuilderParameters");

  // MuonSeed Collection Label
  theSeedCollectionLabel = parameterSet.getParameter<InputTag>("InputObjects");
  seedsToken = consumes<edm::View<TrajectorySeed> >(theSeedCollectionLabel);
  // service parameters
  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");

  // TrackLoader parameters
  ParameterSet trackLoaderParameters = parameterSet.getParameter<ParameterSet>("TrackLoaderParameters");

  // the services
  theService = new MuonServiceProxy(serviceParameters);

  MuonTrajectoryBuilder * trajectoryBuilder = 0;
  // instantiate the concrete trajectory builder in the Track Finder

  edm::ConsumesCollector  iC = consumesCollector();
  string typeOfBuilder = parameterSet.existsAs<string>("MuonTrajectoryBuilder") ? 
    parameterSet.getParameter<string>("MuonTrajectoryBuilder") : "StandAloneMuonTrajectoryBuilder";
  if(typeOfBuilder == "StandAloneMuonTrajectoryBuilder" || typeOfBuilder == "")
    trajectoryBuilder = new StandAloneMuonTrajectoryBuilder(trajectoryBuilderParameters,theService,iC);
  else if(typeOfBuilder == "Exhaustive")
    trajectoryBuilder = new ExhaustiveMuonTrajectoryBuilder(trajectoryBuilderParameters,theService,iC);
  else{
    LogWarning("Muon|RecoMuon|StandAloneMuonProducer") << "No Trajectory builder associated with "<<typeOfBuilder
    						       << ". Falling down to the default (StandAloneMuonTrajectoryBuilder)";
    trajectoryBuilder = new StandAloneMuonTrajectoryBuilder(trajectoryBuilderParameters,theService,iC);
  }
  theTrackFinder = new MuonTrackFinder(trajectoryBuilder,
				       new MuonTrackLoader(trackLoaderParameters, iC, theService),
				       new MuonTrajectoryCleaner(true));
  
  produces<reco::TrackCollection>();
  produces<reco::TrackCollection>("UpdatedAtVtx");
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();
  produces<reco::TrackToTrackMap>();

  produces<std::vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();

  produces<edm::AssociationMap<edm::OneToMany<std::vector<L2MuonTrajectorySeed>, std::vector<L2MuonTrajectorySeed> > > >();
}
  
/// destructor
L2MuonProducer::~L2MuonProducer(){
  LogTrace("Muon|RecoMuon|L2eMuonProducer")<<"L2MuonProducer destructor called"<<endl;
  delete theService;
  delete theTrackFinder;
}


/// reconstruct muons
void L2MuonProducer::produce(Event& event, const EventSetup& eventSetup){
  
 const std::string metname = "Muon|RecoMuon|L2MuonProducer";
  
  LogTrace(metname)<<endl<<endl<<endl;
  LogTrace(metname)<<"L2 Muon Reconstruction Started"<<endl;
  
  // Take the seeds container
  LogTrace(metname)<<"Taking the seeds: "<<theSeedCollectionLabel.label()<<endl;
  Handle<View<TrajectorySeed> > seeds; 
  event.getByToken(seedsToken,seeds);

  // Update the services
  theService->update(eventSetup);
  
  // Reconstruct 
  LogTrace(metname)<<"Track Reconstruction"<<endl;
  theTrackFinder->reconstruct(seeds,event, eventSetup);
  
  LogTrace(metname)<<"Event loaded"
		   <<"================================"
		   <<endl<<endl;
}

void L2MuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    {
        edm::ParameterSetDescription psd0;
        psd0.addUntracked<std::vector<std::string>>("Propagators", {
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
        psd0.add<unsigned int>("NMinRecHits",2);
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
            psd1.add<bool>("EnableCSCMeasurement", true);
            psd0.add<edm::ParameterSetDescription>("FilterParameters", psd1);
        }
        psd0.add<std::string>("NavigationType", "Standard");
        {
            edm::ParameterSetDescription psd1;
            psd1.add<std::string>("Fitter", "hltESPKFFittingSmootherForL2Muon");
            psd1.add<std::string>("MuonRecHitBuilder", "hltESPMuonTransientTrackingRecHitBuilder");
            psd1.add<unsigned int>("NMinRecHits",2);
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
            psd1.add<std::vector<double>>("BeamSpotPosition", {
                0.0,
                0.0,
                0.0,
            });
            psd1.add<std::string>("Propagator", "hltESPFastSteppingHelixPropagatorOpposite");
            psd1.add<std::vector<double>>("BeamSpotPositionErrors", {
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
