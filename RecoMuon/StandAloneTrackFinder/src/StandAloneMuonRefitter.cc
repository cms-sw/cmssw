/** \class StandAloneMuonRefitter
 *  The inward-outward fitter (starts from seed state).
 *
 *  $Date: 2006/07/26 09:27:14 $
 *  $Revision: 1.20 $
 *  \author R. Bellan - INFN Torino
 *  \author S. Lacaprara - INFN Legnaro
 */
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonRefitter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonBestMeasurementFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
// FIXME!! use this (i.e. take the estimator from the esetup)
// #include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "Utilities/Timing/interface/TimingReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <vector>

using namespace edm;
using namespace std;

StandAloneMuonRefitter::StandAloneMuonRefitter(const ParameterSet& par):thePropagator(0),theDetLayerGeometry(0){
  
  // FIXME
  // I am not yet sure if I want to pass it by paset or have it hard-coded...
  // Propagation direction
  string propagationName = par.getParameter<string>("PropagationDirection");
  if (propagationName == "alongMomentum" ) thePropagationDirection = alongMomentum;
  else if (propagationName == "oppositeToMomentum" ) thePropagationDirection = oppositeToMomentum;
  else 
    throw cms::Exception("StandAloneMuonRefitter constructor") 
      <<"Wrong propagation direction chosen in StandAloneMuonRefitter::StandAloneMuonRefitter ParameterSet"
      << endl
      << "Possible choices are:"
      << endl
      << "PropagationDirection = alongMomentum or PropagationDirection = oppositeToMomentum";
  
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

  // FIXME: Do I need it??
  // theMuonUpdatorName = par.getParameter<string>("MuonUpdatorName");
  ParameterSet muonUpdatorPSet = par.getParameter<ParameterSet>("MuonTrajectoryUpdatorParameters");
  theMuonUpdator = new MuonTrajectoryUpdator(muonUpdatorPSet); //FIXME this is very very temp!!!

  thePropagatorName = par.getParameter<string>("Propagator");

  // Measurement Extractor: enable the measure for each muon sub detector
  bool enableDTMeasurement = par.getUntrackedParameter<bool>("EnableDTMeasurement",true);
  bool enableCSCMeasurement = par.getUntrackedParameter<bool>("EnableCSCMeasurement",true);
  bool enableRPCMeasurement = par.getUntrackedParameter<bool>("EnableRPCMeasurement",true);

  theMeasurementExtractor = new MuonDetLayerMeasurements(enableDTMeasurement,
							 enableCSCMeasurement,
							 enableRPCMeasurement);
}

StandAloneMuonRefitter::~StandAloneMuonRefitter(){

  LogDebug("Muon|RecoMuon|StandAloneMuonRefitter")
    <<"StandAloneMuonRefitter destructor called"<<endl;
  
  // FIXME
  //delete thePropagator;
  delete theEstimator;
  delete theMuonUpdator;
  delete theMeasurementExtractor;
}


void StandAloneMuonRefitter::reset(){
  totalChambers = dtChambers = cscChambers = rpcChambers = 0;
  
  theLastUpdatedTSOS =  theLastButOneUpdatedTSOS = TrajectoryStateOnSurface();

  theDetLayers.clear();
}

void StandAloneMuonRefitter::setES(const EventSetup& setup){
  init(setup);
  
  edm::ESHandle<MuonDetLayerGeometry> detLayerGeometry;
  setup.get<MuonRecoGeometryRecord>().get(detLayerGeometry);

  if(theDetLayerGeometry) delete theDetLayerGeometry;
  theDetLayerGeometry = detLayerGeometry->clone();
}

void StandAloneMuonRefitter::setEvent(const Event& event){
  theMeasurementExtractor->setEvent(event);
}

void StandAloneMuonRefitter::init(const EventSetup& setup){
  // Init each event the members of the class
  
  ESHandle<Propagator> eshPropagator;
  setup.get<TrackingComponentsRecord>().get(thePropagatorName, eshPropagator);

  if(thePropagator) delete thePropagator;

  thePropagator = eshPropagator->clone();

  // Consistency check
  // FIXME NCA
//   if( propagator()->propagationDirection() != propagationDirection() )
//     propagator()->setPropagationDirection( propagationDirection() );
  

  // the muon updator (it doesn't inhert from an updator, but it has one!)
  // the updator is suitable both for FW and BW filtering. The difference between the two fitter are two:
  // the granularity of the updating (i.e.: segment position or 1D rechit position), which can be set via
  // parameter set, and the propagation direction which is embeded in the propagator.

  
  // FIXME: it is temporary solution waiting for the es_producers...

  // TODO fare che theMuonUpdatorName = MuonUpdator + nome Propagator
  // FIXME put it into the event setup and extract it from es. i.e.:
  // theSetup.get<TrackingComponentsRecord>().get(theMuonUpdatorName,theMuonUpdator);
  theMuonUpdator->setPropagator( propagator() ); // FIXME this function will disappear asap!!!
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
								 PropagationDirection &propDir){
  vector<const DetLayer*> detLayers;

  if(theNavigationType == "Standard"){
    // ask for compatible layers
    detLayers = initialLayer->compatibleLayers(fts,propDir);  
    // I have to fit by hand the first layer until the seedTSOS is defined on the first rechit layer
    // In fact the first layer is not returned by initialLayer->compatibleLayers.
    detLayers.insert(detLayers.begin(),initialLayer);
  }
  else if (theNavigationType == "Direct"){
    DirectMuonNavigation navigation(theDetLayerGeometry);
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
  
  // The best measurement finder: search for the best measurement among the TMs available
  MuonBestMeasurementFinder bestMeasurementFinder( propagator() );
  
  // this is the most outward TSOS updated with a recHit onto a DetLayer
  TrajectoryStateOnSurface lastUpdatedTSOS;
  // this is the last but one most outward TSOS updated with a recHit onto a DetLayer
  TrajectoryStateOnSurface lastButOneUpdatedTSOS;
  // this is the most outward TSOS (updated or predicted) onto a DetLayer
  TrajectoryStateOnSurface lastTSOS;
  
  lastUpdatedTSOS = lastButOneUpdatedTSOS = lastTSOS = initialTSOS;
  
  // ask for compatible layers
  vector<const DetLayer*> detLayers = initialLayer->compatibleLayers(*initialTSOS.freeTrajectoryState(),
								     propagationDirection());  
    
  // I have to fit by hand the first layer until the seedTSOS is defined on the first rechit layer
  // In fact the first layer is not returned by initialLayer->compatibleLayers.
  detLayers.insert(detLayers.begin(),initialLayer);
  
  LogDebug(metname)<<"compatible layers found: "<<detLayers.size()<<endl;
  
  vector<const DetLayer*>::const_iterator layer;

  // the layers are ordered in agreement with the propagation direction 
  for ( layer = detLayers.begin(); layer!= detLayers.end(); ++layer ) {
    
    //    bool firstTime = true;

    debug.dumpLayer(*layer,metname);
    
    LogDebug(metname) << "search Trajectory Measurement from: " << lastTSOS.globalPosition();
    
    vector<TrajectoryMeasurement> measL = 
      theMeasurementExtractor->measurements(*layer,
      					   lastTSOS, 
      					   *propagator(), 
					   *estimator());

    LogDebug(metname) << "Number of Trajectory Measurement: " << measL.size();
        
    TrajectoryMeasurement* bestMeasurement = bestMeasurementFinder.findBestMeasurement(measL);
    
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
      bestMeasurement = bestMeasurementFinder.findBestMeasurement(measL);
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
      bestMeasurement = bestMeasurementFinder.findBestMeasurement(measL);
    }
    
    // FIXME: uncomment this line!!
    // if(!bestMeasurement && firstTime) break;

    // check if the there is a measurement
    if(bestMeasurement){
      LogDebug(metname)<<"best measurement found" << "\n"
		       <<"updating the trajectory..."<<endl;
      pair<bool,TrajectoryStateOnSurface> result = updator()->update(bestMeasurement,trajectory);
      LogDebug(metname)<<"trajectory updated: "<<result.first<<endl;
      debug.dumpTSOS(result.second,metname);

      if(result.first){ 
	lastTSOS = result.second;
	incrementChamberCounters(*layer);
	theDetLayers.push_back(*layer);
	
	lastButOneUpdatedTSOS = lastUpdatedTSOS;
	lastUpdatedTSOS = lastTSOS;
      }
    }
    //SL in case no valid mesurement is found, still I want to use the predicted
    //state for the following measurement serches. I take the first in the
    //container. FIXME!!! I want to carefully check this!!!!!
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


