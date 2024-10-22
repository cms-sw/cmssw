/** \class SETMuonSeedProducer
    I. Bloch, E. James, S. Stoynev
 */

#include "RecoMuon/MuonSeedGenerator/plugins/SETMuonSeedProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TMath.h"

using namespace edm;
using namespace std;

SETMuonSeedProducer::SETMuonSeedProducer(const ParameterSet& parameterSet)
    : theSeedFinder(parameterSet), theBeamSpotTag(parameterSet.getParameter<edm::InputTag>("beamSpotTag")) {
  edm::ConsumesCollector iC = consumesCollector();
  thePatternRecognition = new SETPatternRecognition(parameterSet, iC);

  const string metname = "Muon|RecoMuon|SETMuonSeedSeed";
  //std::cout<<" The SET SEED producer started."<<std::endl;

  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters, consumesCollector());
  thePatternRecognition->setServiceProxy(theService);
  theSeedFinder.setServiceProxy(theService);
  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("SETTrajBuilderParameters");

  LogTrace(metname) << "constructor called" << endl;

  apply_prePruning = trajectoryBuilderParameters.getParameter<bool>("Apply_prePruning");

  useSegmentsInTrajectory = trajectoryBuilderParameters.getParameter<bool>("UseSegmentsInTrajectory");

  // The inward-outward fitter (starts from seed state)
  ParameterSet filterPSet = trajectoryBuilderParameters.getParameter<ParameterSet>("FilterParameters");
  filterPSet.addUntrackedParameter("UseSegmentsInTrajectory", useSegmentsInTrajectory);
  theFilter = new SETFilter(filterPSet, theService);

  //----

  beamspotToken = consumes<reco::BeamSpot>(theBeamSpotTag);
  produces<TrajectorySeedCollection>();
}

SETMuonSeedProducer::~SETMuonSeedProducer() {
  LogTrace("Muon|RecoMuon|SETMuonSeedProducer") << "SETMuonSeedProducer destructor called" << endl;

  if (theFilter)
    delete theFilter;
  if (theService)
    delete theService;
  if (thePatternRecognition)
    delete thePatternRecognition;
}

void SETMuonSeedProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  //std::cout<<" start producing..."<<std::endl;
  const string metname = "Muon|RecoMuon|SETMuonSeedSeed";

  MuonPatternRecoDumper debug;

  //Get the CSC Geometry :
  theService->update(eventSetup);

  auto output = std::make_unique<TrajectorySeedCollection>();

  Handle<View<TrajectorySeed> > seeds;

  setEvent(event);

  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> beamSpotHandle;
  event.getByToken(beamspotToken, beamSpotHandle);
  if (beamSpotHandle.isValid()) {
    beamSpot = *beamSpotHandle;

  } else {
    edm::LogInfo("MuonSeedGenerator") << "No beam spot available from EventSetup \n";
  }

  // make it a vector so we can subtract it from position vectors
  GlobalVector gv(beamSpot.x0(), beamSpot.y0(), beamSpot.z0());
  theSeedFinder.setBeamSpot(gv);

  bool fwFitFailed = true;

  std::vector<SeedCandidate> seedCandidates_AllChosen;
  std::vector<MuonRecHitContainer> MuonRecHitContainer_clusters;
  //---- this is "clustering"; later a trajectory can not use hits from different clusters
  thePatternRecognition->produce(event, eventSetup, MuonRecHitContainer_clusters);

  //std::cout<<"We have formed "<<MuonRecHitContainer_clusters.size()<<" clusters"<<std::endl;
  //---- for each cluster,
  for (unsigned int cluster = 0; cluster < MuonRecHitContainer_clusters.size(); ++cluster) {
    //std::cout<<" This is cluster number : "<<cluster<<std::endl;
    std::vector<SeedCandidate> seedCandidates_inCluster;
    //---- group hits in detector layers (if in same layer); the idea is that
    //---- some hits could not belong to a track simultaneously - these will be in a
    //---- group; two hits from one and the same group will not go to the same track
    std::vector<MuonRecHitContainer> MuonRecHitContainer_perLayer =
        theSeedFinder.sortByLayer(MuonRecHitContainer_clusters[cluster]);
    //---- Add protection against huge memory consumption
    //---- Delete busy layers if needed (due to combinatorics)
    theSeedFinder.limitCombinatorics(MuonRecHitContainer_perLayer);
    //---- build all possible combinations (valid sets)
    std::vector<MuonRecHitContainer> allValidSets = theSeedFinder.findAllValidSets(MuonRecHitContainer_perLayer);
    if (apply_prePruning) {
      //---- remove "wild" segments from the combination
      theSeedFinder.validSetsPrePruning(allValidSets);
    }

    //---- build the appropriate output: seedCandidates_inCluster
    //---- if too many (?) valid sets in a cluster - skip it
    if (allValidSets.size() < 500) {  // hardcoded - remove it
      seedCandidates_inCluster = theSeedFinder.fillSeedCandidates(allValidSets);
    }
    //---- find the best valid combinations using simple (no propagation errors) chi2-fit
    //std::cout<<"  Found "<<seedCandidates_inCluster.size()<<" valid sets in the current cluster."<<std::endl;
    if (!seedCandidates_inCluster.empty()) {
      //---- this is the forward fitter (segments); choose which of the SETs in a cluster to be considered further
      std::vector<SeedCandidate> bestSets_inCluster;
      fwFitFailed = !(filter()->fwfit_SET(seedCandidates_inCluster, bestSets_inCluster));

      //---- has the fit failed? continue to the next cluster instead of returning the empty trajectoryContainer and stop the loop IBL 080903
      if (fwFitFailed) {
        //std::cout<<"  fwfit_SET failed!"<<std::endl;
        continue;
      }
      for (unsigned int iSet = 0; iSet < bestSets_inCluster.size(); ++iSet) {
        seedCandidates_AllChosen.push_back(bestSets_inCluster[iSet]);
      }
    }
  }
  //---- loop over all the SETs candidates
  for (unsigned int iMuon = 0; iMuon < seedCandidates_AllChosen.size(); ++iMuon) {
    //std::cout<<" chosen iMuon = "<<iMuon<<std::endl;
    Trajectory::DataContainer finalCandidate;
    SeedCandidate* aFinalSet = &(seedCandidates_AllChosen[iMuon]);
    fwFitFailed = !(filter()->buildTrajectoryMeasurements(aFinalSet, finalCandidate));
    if (fwFitFailed) {
      //std::cout<<"  buildTrajectoryMeasurements failed!"<<std::endl;
      continue;
    }
    //---- are there measurements (or detLayers) used at all?
    if (!filter()->layers().empty())
      LogTrace(metname) << debug.dumpLayer(filter()->lastDetLayer());
    else {
      continue;
    }
    //std::cout<<"  chambers used - all : "<<filter()->getTotalChamberUsed()<<", DT : "<<filter()->getDTChamberUsed()<<
    //", CSC : "<<filter()->getCSCChamberUsed()<<", RPC : "<<filter()->getRPCChamberUsed()<<std::endl;
    //---- ask for some "reasonable" conditions to build a STA muon;
    //---- (totalChambers >= 2, dtChambers + cscChambers >0)
    if (filter()->goodState()) {
      TransientTrackingRecHit::ConstRecHitContainer hitContainer;
      TrajectoryStateOnSurface firstTSOS;
      bool conversionPassed = false;
      if (!useSegmentsInTrajectory) {
        //---- transforms set of segment measurements to a set of rechit measurements
        conversionPassed = filter()->transform(finalCandidate, hitContainer, firstTSOS);
      } else {
        //---- transforms set of segment measurements to a set of segment measurements
        conversionPassed = filter()->transformLight(finalCandidate, hitContainer, firstTSOS);
      }
      if (conversionPassed && !finalCandidate.empty() && !hitContainer.empty()) {
        //---- doesn't work...
        //output->push_back( theSeedFinder.makeSeed(firstTSOS, hitContainer) );

        edm::OwnVector<TrackingRecHit> recHitsContainer;
        for (unsigned int iHit = 0; iHit < hitContainer.size(); ++iHit) {
          recHitsContainer.push_back(hitContainer.at(iHit)->hit()->clone());
        }
        PropagationDirection dir = oppositeToMomentum;
        if (useSegmentsInTrajectory) {
          dir = alongMomentum;  // why forward (for rechits) later?
        }

        PTrajectoryStateOnDet seedTSOS =
            trajectoryStateTransform::persistentState(firstTSOS, hitContainer.at(0)->geographicalId().rawId());
        TrajectorySeed seed(seedTSOS, recHitsContainer, dir);

        //MuonPatternRecoDumper debug;
        //std::cout<<"  firstTSOS (not IP) = "<<debug.dumpTSOS(firstTSOS)<<std::endl;
        //std::cout<<"  hits(from range) = "<<range.second-range.first<<" hits (from size) = "<<hitContainer.size()<<std::endl;
        //for(unsigned int iRH=0;iRH<hitContainer.size();++iRH){
        //std::cout<<"  RH = "<<iRH+1<<" globPos = "<<hitContainer.at(iRH)->globalPosition()<<std::endl;
        //}
        output->push_back(seed);
      } else {
        //std::cout<<" Transformation from TrajectoryMeasurements to RecHitContainer faild - skip "<<std::endl;
        continue;
      }
    } else {
      //std::cout<<" Not enough (as defined) measurements to build trajectory - skip"<<std::endl;
      continue;
    }
  }
  event.put(std::move(output));
  theFilter->reset();
}

//
void SETMuonSeedProducer::setEvent(const edm::Event& event) { theFilter->setEvent(event); }
