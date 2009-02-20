/** \class SETMuonSeedProducer
    I. Bloch, E. James, S. Stoynev
 */

#include "RecoMuon/MuonSeedGenerator/plugins/SETMuonSeedProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
: thePatternRecognition(parameterSet),
  theSeedFinder(parameterSet)
{
  const string metname = "Muon|RecoMuon|SETMuonSeedSeed";  
  //std::cout<<" The SET SEED"<<std::endl;

  ParameterSet serviceParameters = parameterSet.getParameter<ParameterSet>("ServiceParameters");
  theService        = new MuonServiceProxy(serviceParameters);
  thePatternRecognition.setServiceProxy(theService);
  theSeedFinder.setServiceProxy(theService);
  // Parameter set for the Builder
  ParameterSet trajectoryBuilderParameters = parameterSet.getParameter<ParameterSet>("SETTrajBuilderParameters");

  LogTrace(metname) << "constructor called" << endl;
  
  apply_prePruning = trajectoryBuilderParameters.getParameter<bool>("Apply_prePruning");

  useSegmentsInTrajectory = trajectoryBuilderParameters.getParameter<bool>("UseSegmentsInTrajectory");

  // The inward-outward fitter (starts from seed state)
  ParameterSet filterPSet = trajectoryBuilderParameters.getParameter<ParameterSet>("FilterParameters");
  filterPSet.addUntrackedParameter("UseSegmentsInTrajectory", useSegmentsInTrajectory);
  theFilter = new SETFilter(filterPSet,theService);

  //----

  produces<TrajectorySeedCollection>();

} 

SETMuonSeedProducer::~SETMuonSeedProducer(){

  LogTrace("Muon|RecoMuon|SETMuonSeedProducer") 
    << "SETMuonSeedProducer destructor called" << endl;
  
  if(theFilter) delete theFilter;
}

void SETMuonSeedProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup){
  //std::cout<<" start producing..."<<std::endl;  
  const string metname = "Muon|RecoMuon|SETMuonSeedSeed";  

  MuonPatternRecoDumper debug;

  //Get the CSC Geometry :
  theService->update(eventSetup);

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());
  
  Handle<View<TrajectorySeed> > seeds; 

  setEvent(event);

  std::vector < TrajectoryMeasurement > trajectoryMeasurementsFW;

  bool fwFitFailed = true;


  std::vector< MuonRecHitContainer > MuonRecHitContainer_clusters;
  thePatternRecognition.produce(event, eventSetup, MuonRecHitContainer_clusters);

  //std::cout<<"We have formed "<<MuonRecHitContainer_clusters.size()<<" clusters"<<std::endl;
  // for each cluster,
  for(unsigned int cluster = 0; cluster < MuonRecHitContainer_clusters.size(); ++cluster) {
    //std::cout<<" This is cluster number : "<<cluster<<std::endl;
     std::vector <seedSet> seedSets_inCluster; 
    //---- group hits in detector layers (if in same layer); the idea is that
    //---- some hits could not belong to a track simultaneously - these will be in a 
    //---- group; two hits from one and the same group will not go to the same track 
    std::vector< MuonRecHitContainer > MuonRecHitContainer_perLayer = theSeedFinder.sortByLayer(MuonRecHitContainer_clusters[cluster]);
    //---- build all possible combinations (valid sets)
    std::vector <MuonRecHitContainer> allValidSets = theSeedFinder.findAllValidSets(MuonRecHitContainer_perLayer);
    if(apply_prePruning){
      //---- remove "wild" segments from the combination
      theSeedFinder.validSetsPrePruning(allValidSets);
    }
    
    //---- build the appropriate output: seedSets_inCluster
    //---- if too many (?) valid sets in a cluster - skip it 
    if(allValidSets.size()<500){// hardcoded - remove it
      seedSets_inCluster = theSeedFinder.fillSeedSets(allValidSets);
    }
    //// find the best valid combinations using simple (no material effects) chi2-fit 
    //std::cout<<"Found "<<seedSets_inCluster.size()<<" valid sets in the current cluster."<<std::endl;
    if(!seedSets_inCluster.empty()){
      //TrajectorySeed seed;
      //PropagationDirection dir = alongMomentum;
      //Trajectory trajectoryNew(seed, dir);
      //fwFitFailed = !(filter()->refit(seedSets_inCluster, trajectoryNew));

      //---- this is the forward fitter (segments)
      trajectoryMeasurementsFW.clear();
      fwFitFailed = !(filter()->fwfit_SET(seedSets_inCluster, trajectoryMeasurementsFW));
      //std::cout<<"after refit : fwFitFailed = "<<fwFitFailed<<std::endl;
      //trajectoryFW = trajectoryNew;

    // has the fit failed? continue to the next cluster instead of returning the empty trajectoryContainer and stop the loop IBL 080903
    if(fwFitFailed || !trajectoryMeasurementsFW.at(trajectoryMeasurementsFW.size()-1).forwardPredictedState().isValid()) continue;

    //TrajectoryStateOnSurface tsosAfterRefit = trajectoryMeasurementsFW.at(trajectoryMeasurementsFW.size()-1).forwardPredictedState();

      // are there measurements (or detLayers) used at all?
      if( !filter()->layers().empty() )
        LogTrace(metname) << debug.dumpLayer( filter()->lastDetLayer());
      else {
        continue;
      }

      //---- ask for some "reasonable" conditions to build a STA muon; 
      //---- (totalChambers >= 2, dtChambers + cscChambers >0)
      if (filter()->goodState()) {
	TransientTrackingRecHit::ConstRecHitContainer hitContainer;
	TrajectoryStateOnSurface firstTSOS;
	bool conversionPassed = false;
	if(!useSegmentsInTrajectory){
	// transforms set of segment measurements to a set of rechit measurements
	  conversionPassed = filter()->transform(trajectoryMeasurementsFW, hitContainer, firstTSOS);
	}
	else{
	// transforms set of segment measurements to a set of segment measurements
          conversionPassed = filter()->transformLight(trajectoryMeasurementsFW, hitContainer, firstTSOS);
	}
	if ( conversionPassed && !trajectoryMeasurementsFW.empty() && !hitContainer.empty()) {
          output->push_back( theSeedFinder.makeSeed(firstTSOS, hitContainer) );
	}
	else{
          continue;
	}
      }else{
        continue;
      }
    }
  }
  event.put(output);
}


void SETMuonSeedProducer::setEvent(const edm::Event& event){
  theFilter->setEvent(event);
}
