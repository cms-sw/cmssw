/** \class StandAloneMuonRefitter
 *  The inward-outward fitter (starts from seed state).
 *
 *  $Date: 2006/05/22 12:11:22 $
 *  $Revision: 1.3 $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonRefitter.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "Utilities/Timing/interface/TimingReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "RecoMuon/TrackingTools/interface/MuonBestMeasurementFinder.h"

#include <vector>

using namespace edm;
using namespace std;

StandAloneMuonRefitter::StandAloneMuonRefitter(const ParameterSet& par){

  // The max allowed chi2 to accept a rechit in the fit
  theMaxChi2 = par.getParameter<double>("MaxChi2");

  // The errors of the trajectory state are multiplied by nSigma 
  // to define acceptance of BoundPlane and maximalLocalDisplacement
  theNSigma = par.getParameter<double>("NumberOfSigma"); // default = 3.

  // The propagator: it propagates a state
  thePropagator = new SteppingHelixPropagator();

  // The estimator: makes the decision wheter a measure is good or not for the fit
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2,theNSigma);

  // FIXME: Do I want different propagator and estimator??
  // The best measurement finder: search for the best measurement among the TMs available
  theBestMeasurementFinder = new MuonBestMeasurementFinder(thePropagator,theEstimator);
}

StandAloneMuonRefitter::~StandAloneMuonRefitter(){
  delete thePropagator;
  delete theEstimator;
  delete theBestMeasurementFinder;
}


void StandAloneMuonRefitter::reset(){
  totalChambers = dtChambers = cscChambers = rpcChambers = 0;
  
  theLastUpdatedTSOS =  theLastButOneUpdatedTSOS = TrajectoryStateOnSurface();

  // FIXME
  // theNavLayers = vector<const DetLayer*>() ;
}

void StandAloneMuonRefitter::setES(const EventSetup& setup){
  
  // Set the muon DetLayer geometry in the navigation school
//   edm::ESHandle<MuonDetLayerGeometry> muonDetLayerGeometry;
//   setup.get<MuonRecoGeometryRecord>().get(muonDetLayerGeometry);     

//   MuonNavigationSchool muonSchool(&*muonDetLayerGeometry);
//   NavigationSetter setter(muonSchool);
}

void StandAloneMuonRefitter::setEvent(const Event& event){
  theCachedEvent = &event;
}

void StandAloneMuonRefitter::refit(TrajectoryStateOnSurface& initialTSOS,const DetLayer* initialLayer){
  
  std::string metname = "StandAloneMuonRefitter::refit";
  bool timing = true;
  
  MuonPatternRecoDumper debug;
  LogDebug(metname) << "Starting the refit"; 
  TimeMe t(metname,timing);
  
  //   // this is the most outward FTS updated with a recHit
  //   FreeTrajectoryState lastUpdatedFts;
  //   // this is the last but one most outward FTS updated with a recHit
  //   FreeTrajectoryState lastButOneUpdatedFts;
  //   // this is the most outward FTS (updated or predicted)
  //   FreeTrajectoryState lastFts;
  //   lastUpdatedFts = lastButOneUpdatedFts = lastFts = *(initialTSOS.freeTrajectoryState());
  
  // this is the most outward TSOS updated with a recHit
  TrajectoryStateOnSurface lastUpdatedTSOS;
  // this is the last but one most outward TSOS updated with a recHit
  TrajectoryStateOnSurface lastButOneUpdatedTSOS;
  // this is the most outward TSOS (updated or predicted)
  TrajectoryStateOnSurface lastTSOS;
  
  lastUpdatedTSOS = lastButOneUpdatedTSOS = lastTSOS = initialTSOS;
  
  // FIXME: check the prop direction!
  vector<const DetLayer*> nLayers = initialLayer->compatibleLayers(*initialTSOS.freeTrajectoryState(),
								   alongMomentum);  
    
  
  // FIXME: is it right?Or have I to invert the iterator/prop direction??
  vector<const DetLayer*>::const_iterator layer;

  // FIXME: begin() in rbegin() and end() in rend()??
  for ( layer = nLayers.begin(); layer!= nLayers.end(); ++layer ) {
    
    //  const DetLayer* layer = *nextlayer;
    debug.dumpLayer(*layer,metname);
    
    LogDebug(metname) << "search Trajectory Measurement from: " << lastTSOS.globalPosition();

    vector<TrajectoryMeasurement> measL = 
      theMeasurementExtractor.measurements(*layer,
      					   lastTSOS, 
      					   propagator(), 
					   estimator(),
      					   *theCachedEvent);
    LogDebug(metname) << "Number of Trajectory Measurement:" << measL.size();
    
    TrajectoryMeasurement* bestMeasurement = bestMeasurementFinder()->findBestMeasurement(measL);
    
  }
}

