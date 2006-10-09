/** \class StandAloneMuonRefitter
 *  The inward-outward fitter (starts from seed state).
 *
 *  $Date: 2006/09/04 17:11:49 $
 *  $Revision: 1.29 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \author S. Lacaprara - INFN Legnaro
 */
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonRefitter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// FIXME: remove this
#include "FWCore/Framework/interface/Event.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"

#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "Utilities/Timing/interface/TimingReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>

using namespace edm;
using namespace std;

StandAloneMuonRefitter::StandAloneMuonRefitter(const ParameterSet& par,
					       const MuonServiceProxy* service):theService(service){
  
  // Fit direction
  string fitDirectionName = par.getParameter<string>("FitDirection");

  if (fitDirectionName == "insideOut" ) theFitDirection = recoMuon::insideOut;
  else if (fitDirectionName == "outsideIn" ) theFitDirection = recoMuon::outsideIn;
  else 
    throw cms::Exception("StandAloneMuonRefitter constructor") 
      <<"Wrong fit direction chosen in StandAloneMuonRefitter::StandAloneMuonRefitter ParameterSet"
      << "\n"
      << "Possible choices are:"
      << "\n"
      << "FitDirection = insideOut or FitDirection = outsideIn";
  
  // The max allowed chi2 to accept a rechit in the fit
  theMaxChi2 = par.getParameter<double>("MaxChi2");

  // The errors of the trajectory state are multiplied by nSigma 
  // to define acceptance of BoundPlane and maximalLocalDisplacement
  theNSigma = par.getParameter<double>("NumberOfSigma"); // default = 3.

  // The navigation type:
  // "Direct","Standard"
  theNavigationType = par.getParameter<string>("NavigationType");
  
  // The estimator: makes the decision wheter a measure is good or not
  // it isn't used by the updator which does the real fit. In fact, in principle,
  // a looser request onto the measure set can be requested 
  // (w.r.t. the request on the accept/reject measure in the fit)
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2,theNSigma);
  
  thePropagatorName = par.getParameter<string>("Propagator");

  theBestMeasurementFinder = new MuonBestMeasurementFinder();

  // Muon trajectory updator parameters
  ParameterSet muonUpdatorPSet = par.getParameter<ParameterSet>("MuonTrajectoryUpdatorParameters");
  
  // the updator needs the fit direction
  theMuonUpdator = new MuonTrajectoryUpdator(muonUpdatorPSet,
					     fitDirection() );

  // Measurement Extractor: enable the measure for each muon sub detector
  bool enableDTMeasurement = par.getParameter<bool>("EnableDTMeasurement");
  bool enableCSCMeasurement = par.getParameter<bool>("EnableCSCMeasurement");
  bool enableRPCMeasurement = par.getParameter<bool>("EnableRPCMeasurement");

  theMeasurementExtractor = new MuonDetLayerMeasurements(enableDTMeasurement,
							 enableCSCMeasurement,
							 enableRPCMeasurement);
}

StandAloneMuonRefitter::~StandAloneMuonRefitter(){

  LogDebug("Muon|RecoMuon|StandAloneMuonRefitter")
    <<"StandAloneMuonRefitter destructor called"<<endl;
  
  delete theEstimator;
  delete theMuonUpdator;
  delete theMeasurementExtractor;
  delete theBestMeasurementFinder;
}

/// Return the propagation direction
PropagationDirection StandAloneMuonRefitter::propagationDirection() const{
  if( fitDirection() == 0 ) return alongMomentum;
  else if ( fitDirection() == 1 ) return oppositeToMomentum;
  else return anyDirection;
}


void StandAloneMuonRefitter::reset(){
  totalChambers = dtChambers = cscChambers = rpcChambers = 0;
  
  theLastUpdatedTSOS =  theLastButOneUpdatedTSOS = TrajectoryStateOnSurface();

  theDetLayers.clear();
}

void StandAloneMuonRefitter::setEvent(const Event& event){
  theMeasurementExtractor->setEvent(event);
}


void StandAloneMuonRefitter::incrementChamberCounters(const DetLayer *layer){

  if(layer->subDetector()==GeomDetEnumerators::DT) dtChambers++; 
  else if(layer->subDetector()==GeomDetEnumerators::CSC) cscChambers++; 
  else if(layer->subDetector()==GeomDetEnumerators::RPCBarrel || layer->subDetector()==GeomDetEnumerators::RPCEndcap) rpcChambers++; 
  else 
    LogError("Muon|RecoMuon|StandAloneMuonRefitter")
      << "Unrecognized module type in incrementChamberCounters";
  // FIXME:
  //   << layer->module() << " " <<layer->Part() << endl;
  
  totalChambers++;
}


vector<const DetLayer*> StandAloneMuonRefitter::compatibleLayers(const DetLayer *initialLayer,
								 FreeTrajectoryState& fts,
								 PropagationDirection propDir){
  vector<const DetLayer*> detLayers;

  if(theNavigationType == "Standard"){
    // ask for compatible layers
    detLayers = initialLayer->compatibleLayers(fts,propDir);  
    // I have to fit by hand the first layer until the seedTSOS is defined on the first rechit layer
    // In fact the first layer is not returned by initialLayer->compatibleLayers.
    detLayers.insert(detLayers.begin(),initialLayer);
  }
  else if (theNavigationType == "Direct"){
    DirectMuonNavigation navigation(&*theService->detLayerGeometry());
    detLayers = navigation.compatibleLayers(fts,propDir);
  }
  else
    edm::LogError("Muon|RecoMuon|StandAloneMuonRefitter") << "No Properly Navigation Selected!!"<<endl;
  
  return detLayers;
}


void StandAloneMuonRefitter::refit(const TrajectoryStateOnSurface& initialTSOS,
				   const DetLayer* initialLayer, Trajectory &trajectory){
  
  const std::string metname = "Muon|RecoMuon|StandAloneMuonRefitter";
  bool timing = true;

  // reset the refitter each seed refinement
  reset();
  
  MuonPatternRecoDumper debug;
  
  LogDebug(metname) << "Starting the refit"<<endl; 
  TimeMe t(metname,timing);

  // this is the most outward TSOS updated with a recHit onto a DetLayer
  TrajectoryStateOnSurface lastUpdatedTSOS;
  // this is the last but one most outward TSOS updated with a recHit onto a DetLayer
  TrajectoryStateOnSurface lastButOneUpdatedTSOS;
  // this is the most outward TSOS (updated or predicted) onto a DetLayer
  TrajectoryStateOnSurface lastTSOS;
  
  lastUpdatedTSOS = lastButOneUpdatedTSOS = lastTSOS = initialTSOS;
  
  vector<const DetLayer*> detLayers = compatibleLayers(initialLayer,*initialTSOS.freeTrajectoryState(),
						       propagationDirection());  
  
  LogDebug(metname)<<"compatible layers found: "<<detLayers.size()<<endl;
  
  vector<const DetLayer*>::const_iterator layer;

  // the layers are ordered in agreement with the fit/propagation direction 
  for ( layer = detLayers.begin(); layer!= detLayers.end(); ++layer ) {
    
    //    bool firstTime = true;

    LogDebug(metname) << debug.dumpLayer(*layer);
    
    LogDebug(metname) << "search Trajectory Measurement from: " << lastTSOS.globalPosition();
    
    vector<TrajectoryMeasurement> measL = 
      theMeasurementExtractor->measurements(*layer,
      					   lastTSOS, 
      					   *propagator(), 
					   *estimator());

    LogDebug(metname) << "Number of Trajectory Measurement: " << measL.size();
        
    TrajectoryMeasurement* bestMeasurement = bestMeasurementFinder()->findBestMeasurement(measL, propagator());

    // RB: Different ways can be choosen if no bestMeasurement is available:
    // 1- check on lastTSOS-initialTSOS eta difference
    // 2- check on lastTSOS-lastButOneUpdatedTSOS eta difference
    // After this choice:
    // A- extract the measurements compatible with the initialTSOS (seed)
    // B- extract the measurements compatible with the lastButOneUpdatedTSOS
    // In ORCA the choice was 1A. Here I will try 1B and if it fail I'll try 1A
    // another possibility could be 2B and then 1A.

    // if no measurement found and the current TSOS has an eta very different
    // wrt the initial one (i.e. seed), then try to find the measurements
    // according to the lastButOne FTS. (1B)
    if( !bestMeasurement && 
	fabs(lastTSOS.freeTrajectoryState()->momentum().eta() - 
	     initialTSOS.freeTrajectoryState()->momentum().eta())>0.1 ) {

      LogDebug(metname) << "No measurement and big eta variation wrt seed" << endl
			<< "trying with lastButOneUpdatedTSOS";
      measL = theMeasurementExtractor->measurements(*layer,
						   lastButOneUpdatedTSOS, 
						   *propagator(), 
						   *estimator());
      bestMeasurement = bestMeasurementFinder()->findBestMeasurement(measL, propagator());
    }
    
    //if no measurement found and the current FTS has an eta very different
    //wrt the initial one (i.e. seed), then try to find the measurements
    //according to the initial FTS. (1A)
    if( !bestMeasurement && 
	fabs(lastTSOS.freeTrajectoryState()->momentum().eta() - 
	     initialTSOS.freeTrajectoryState()->momentum().eta())>0.1 ) {

      LogDebug(metname) << "No measurement and big eta variation wrt seed" << endl
			<< "tryng with seed TSOS";

      measL = theMeasurementExtractor->measurements(*layer,
						   initialTSOS, 
						   *propagator(), 
						   *estimator());
      bestMeasurement = bestMeasurementFinder()->findBestMeasurement(measL, propagator());
    }
    
    // FIXME: uncomment this line!!
    // if(!bestMeasurement && firstTime) break;

    // check if the there is a measurement
    if(bestMeasurement){
      LogDebug(metname)<<"best measurement found" << "\n"
		       <<"updating the trajectory..."<<endl;
      pair<bool,TrajectoryStateOnSurface> result = updator()->update(bestMeasurement,
								     trajectory,
								     propagator());
      LogDebug(metname)<<"trajectory updated: "<<result.first<<endl;
      LogDebug(metname) << debug.dumpTSOS(result.second);

      if(result.first){ 
	lastTSOS = result.second;
	incrementChamberCounters(*layer);
	theDetLayers.push_back(*layer);
	
	lastButOneUpdatedTSOS = lastUpdatedTSOS;
	lastUpdatedTSOS = lastTSOS;
      }
    }
    // SL in case no valid mesurement is found, still I want to use the predicted
    // state for the following measurement serches. I take the first in the
    // container. FIXME!!! I want to carefully check this!!!!!
    else{
      LogDebug(metname)<<"No best measurement found"<<endl;
      if (measL.size()>0){
	LogDebug(metname)<<"but the #of measurement is "<<measL.size()<<endl;
        lastTSOS = measL.front().predictedState();
      }
    }

  }
  setLastUpdatedTSOS(lastUpdatedTSOS);
  setLastButOneUpdatedTSOS(lastButOneUpdatedTSOS);
}


