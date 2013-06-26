/** \class StandAloneMuonFilter
 *  The inward-outward fitter (starts from seed state).
 *
 *  $Date: 2009/09/17 20:02:50 $
 *  $Revision: 1.10 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *          D. Trocino - INFN Torino <daniele.trocino@to.infn.it>
 */
#include "RecoMuon/StandAloneTrackFinder/interface/StandAloneMuonFilter.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// FIXME: remove this
#include "FWCore/Framework/interface/Event.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <vector>

using namespace edm;
using namespace std;

StandAloneMuonFilter::StandAloneMuonFilter(const ParameterSet& par,
					       const MuonServiceProxy* service)
:theService(service),
 theOverlappingChambersFlag(true)
{
  // Fit direction
  string fitDirectionName = par.getParameter<string>("FitDirection");

  if (fitDirectionName == "insideOut" ) theFitDirection = insideOut;
  else if (fitDirectionName == "outsideIn" ) theFitDirection = outsideIn;
  else 
    throw cms::Exception("StandAloneMuonFilter constructor") 
      <<"Wrong fit direction chosen in StandAloneMuonFilter::StandAloneMuonFilter ParameterSet"
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

  theMeasurementExtractor = new MuonDetLayerMeasurements(par.getParameter<InputTag>("DTRecSegmentLabel"),
							 par.getParameter<InputTag>("CSCRecSegmentLabel"),
							 par.getParameter<InputTag>("RPCRecSegmentLabel"),
							 enableDTMeasurement,
							 enableCSCMeasurement,
							 enableRPCMeasurement);
  
  theRPCLoneliness = (!(enableDTMeasurement && enableCSCMeasurement)) ? enableRPCMeasurement : false;
}

StandAloneMuonFilter::~StandAloneMuonFilter(){

  LogTrace("Muon|RecoMuon|StandAloneMuonFilter")
    <<"StandAloneMuonFilter destructor called"<<endl;
  
  delete theEstimator;
  delete theMuonUpdator;
  delete theMeasurementExtractor;
  delete theBestMeasurementFinder;
}

const Propagator* StandAloneMuonFilter::propagator() const { 
  return &*theService->propagator(thePropagatorName); 
}

/// Return the propagation direction
PropagationDirection StandAloneMuonFilter::propagationDirection() const{
  if( fitDirection() == 0 ) return alongMomentum;
  else if ( fitDirection() == 1 ) return oppositeToMomentum;
  else return anyDirection;
}


void StandAloneMuonFilter::reset(){
  totalChambers = dtChambers = cscChambers = rpcChambers = 0;
  totalCompatibleChambers = dtCompatibleChambers = cscCompatibleChambers = rpcCompatibleChambers = 0;
  
  theLastCompatibleTSOS = theLastUpdatedTSOS = theLastButOneUpdatedTSOS = TrajectoryStateOnSurface();

  theMuonUpdator->makeFirstTime();

  theDetLayers.clear();
}

void StandAloneMuonFilter::setEvent(const Event& event){
  theMeasurementExtractor->setEvent(event);
}


void StandAloneMuonFilter::incrementChamberCounters(const DetLayer *layer){

  if(layer->subDetector()==GeomDetEnumerators::DT) dtChambers++; 
  else if(layer->subDetector()==GeomDetEnumerators::CSC) cscChambers++; 
  else if(layer->subDetector()==GeomDetEnumerators::RPCBarrel || layer->subDetector()==GeomDetEnumerators::RPCEndcap) rpcChambers++; 
  else 
    LogError("Muon|RecoMuon|StandAloneMuonFilter")
      << "Unrecognized module type in incrementChamberCounters";
  // FIXME:
  //   << layer->module() << " " <<layer->Part() << endl;
  
  totalChambers++;
}

void StandAloneMuonFilter::incrementCompatibleChamberCounters(const DetLayer *layer){

  if(layer->subDetector()==GeomDetEnumerators::DT) dtCompatibleChambers++; 
  else if(layer->subDetector()==GeomDetEnumerators::CSC) cscCompatibleChambers++; 
  else if(layer->subDetector()==GeomDetEnumerators::RPCBarrel || layer->subDetector()==GeomDetEnumerators::RPCEndcap) rpcCompatibleChambers++; 
  else 
    LogError("Muon|RecoMuon|StandAloneMuonFilter")
      << "Unrecognized module type in incrementCompatibleChamberCounters";
  
  totalCompatibleChambers++;
}


vector<const DetLayer*> StandAloneMuonFilter::compatibleLayers(const DetLayer *initialLayer,
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
    DirectMuonNavigation navigation(theService->detLayerGeometry());
    detLayers = navigation.compatibleLayers(fts,propDir);
  }
  else
    edm::LogError("Muon|RecoMuon|StandAloneMuonFilter") << "No Properly Navigation Selected!!"<<endl;
  
  return detLayers;
}


void StandAloneMuonFilter::refit(const TrajectoryStateOnSurface& initialTSOS,
				   const DetLayer* initialLayer, Trajectory &trajectory){
  
  const std::string metname = "Muon|RecoMuon|StandAloneMuonFilter";

  // reset the refitter each seed refinement
  reset();
  
  MuonPatternRecoDumper debug;
  
  LogTrace(metname) << "Starting the refit"<<endl; 

  // this is the most outward TSOS (updated or predicted) onto a DetLayer
  TrajectoryStateOnSurface lastTSOS = theLastCompatibleTSOS = theLastUpdatedTSOS = theLastButOneUpdatedTSOS = initialTSOS;
  
  double eta0 = initialTSOS.freeTrajectoryState()->momentum().eta();
  vector<const DetLayer*> detLayers = compatibleLayers(initialLayer,*initialTSOS.freeTrajectoryState(),
						       propagationDirection());  
  
  LogTrace(metname)<<"compatible layers found: "<<detLayers.size()<<endl;
  
  vector<const DetLayer*>::const_iterator layer;

  // the layers are ordered in agreement with the fit/propagation direction 
  for ( layer = detLayers.begin(); layer!= detLayers.end(); ++layer ) {
    
    //    bool firstTime = true;

    LogTrace(metname) << debug.dumpLayer(*layer);
    
    LogTrace(metname) << "search Trajectory Measurement from: " << lastTSOS.globalPosition();
    
    // pick the best measurement from each group
    std::vector<TrajectoryMeasurement> bestMeasurements = findBestMeasurements(*layer, lastTSOS);

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
    double lastdEta = fabs(lastTSOS.freeTrajectoryState()->momentum().eta() - eta0);
    if( bestMeasurements.empty() && lastdEta > 0.1) {
      LogTrace(metname) << "No measurement and big eta variation wrt seed" << endl
			<< "trying with lastButOneUpdatedTSOS";
      bestMeasurements = findBestMeasurements(*layer, theLastButOneUpdatedTSOS);
    }
    
    //if no measurement found and the current FTS has an eta very different
    //wrt the initial one (i.e. seed), then try to find the measurements
    //according to the initial FTS. (1A)
    if( bestMeasurements.empty() && lastdEta > 0.1) {
      LogTrace(metname) << "No measurement and big eta variation wrt seed" << endl
			<< "tryng with seed TSOS";
      bestMeasurements = findBestMeasurements(*layer, initialTSOS);
    }
    
    // FIXME: uncomment this line!!
    // if(!bestMeasurement && firstTime) break;

    if(!bestMeasurements.empty()) {
      incrementCompatibleChamberCounters(*layer);
      bool added = false;
      for(std::vector<TrajectoryMeasurement>::const_iterator tmItr = bestMeasurements.begin();
          tmItr != bestMeasurements.end(); ++tmItr){
        added |= update(*layer, &(*tmItr), trajectory);
        lastTSOS = theLastUpdatedTSOS;
      }
      if(added) {
        incrementChamberCounters(*layer);
        theDetLayers.push_back(*layer);
      }
    }
    // SL in case no valid mesurement is found, still I want to use the predicted
    // state for the following measurement serches. I take the first in the
    // container. FIXME!!! I want to carefully check this!!!!!
    else{
      LogTrace(metname)<<"No best measurement found"<<endl;
      //       if (!theMeasurementCache.empty()){
      // 	LogTrace(metname)<<"but the #of measurement is "<<theMeasurementCache.size()<<endl;
      //         lastTSOS = theMeasurementCache.front().predictedState();
      //       }
    }
  } // loop over layers
}


std::vector<TrajectoryMeasurement>
StandAloneMuonFilter::findBestMeasurements(const DetLayer* layer,
                                             const TrajectoryStateOnSurface& tsos){

  const std::string metname = "Muon|RecoMuon|StandAloneMuonFilter";

  std::vector<TrajectoryMeasurement> result;
  std::vector<TrajectoryMeasurement> measurements;

  if(theOverlappingChambersFlag && layer->hasGroups()){
    
    std::vector<TrajectoryMeasurementGroup> measurementGroups =
      theMeasurementExtractor->groupedMeasurements(layer, tsos, *propagator(), *estimator());

    if(theFitDirection == outsideIn){
      LogTrace(metname) << "Reversing the order of groupedMeasurements as the direction of the fit is outside-in";
      reverse(measurementGroups.begin(),measurementGroups.end());
      // this should be fixed either in RecoMuon/MeasurementDet/MuonDetLayerMeasurements or
      // RecoMuon/DetLayers/MuRingForwardDoubleLayer
    }


    for(std::vector<TrajectoryMeasurementGroup>::const_iterator tmGroupItr = measurementGroups.begin();
        tmGroupItr != measurementGroups.end(); ++tmGroupItr){
    
      measurements = tmGroupItr->measurements();
      LogTrace(metname) << "Number of Trajectory Measurement: " << measurements.size();
      
      const TrajectoryMeasurement* bestMeasurement 
	= bestMeasurementFinder()->findBestMeasurement(measurements,  propagator());
      
      if(bestMeasurement) result.push_back(*bestMeasurement);
    }
  } 
  else{
    measurements = theMeasurementExtractor->measurements(layer, tsos, *propagator(), *estimator());
    LogTrace(metname) << "Number of Trajectory Measurement: " << measurements.size();
    const TrajectoryMeasurement* bestMeasurement 
      = bestMeasurementFinder()->findBestMeasurement(measurements,  
						     propagator());
    if(bestMeasurement) result.push_back(*bestMeasurement);
  }
  return result;
}




bool StandAloneMuonFilter::update(const DetLayer * layer, 
                                    const TrajectoryMeasurement * meas, 
                                    Trajectory & trajectory)
{
  const std::string metname = "Muon|RecoMuon|StandAloneMuonFilter";
  MuonPatternRecoDumper debug;

  LogTrace(metname)<<"best measurement found" << "\n"
                   <<"updating the trajectory..."<<endl;
  pair<bool,TrajectoryStateOnSurface> result = updator()->update(meas,
                                                                 trajectory,
                                                                 propagator());
  LogTrace(metname)<<"trajectory updated: "<<result.first<<endl;
  LogTrace(metname) << debug.dumpTSOS(result.second);

  if(result.first){
    theLastButOneUpdatedTSOS = theLastUpdatedTSOS;
    theLastUpdatedTSOS = result.second;
  }

  if(result.second.isValid())
    theLastCompatibleTSOS = result.second;

  return result.first;
}


void StandAloneMuonFilter::createDefaultTrajectory(const Trajectory & oldTraj, Trajectory & defTraj) {

  Trajectory::DataContainer oldMeas = oldTraj.measurements();
  defTraj.reserve(oldMeas.size());

  for (Trajectory::DataContainer::const_iterator itm = oldMeas.begin(); itm != oldMeas.end(); itm++) {
    if( !(*itm).recHit()->isValid() )
      defTraj.push( *itm, (*itm).estimate() );
    else {
      MuonTransientTrackingRecHit::MuonRecHitPointer invRhPtr = MuonTransientTrackingRecHit::specificBuild( (*itm).recHit()->det(), (*itm).recHit()->hit() );
      invRhPtr->invalidateHit();
      TrajectoryMeasurement invRhMeas( (*itm).forwardPredictedState(), (*itm).updatedState(), invRhPtr.get(), (*itm).estimate(), (*itm).layer() );
      defTraj.push( invRhMeas, (*itm).estimate() );	  
    }

  } // end for
}
