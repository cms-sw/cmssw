#include "RecoMuon/GlobalTrackFinder/interface/MuonCkfTrajectoryBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"


MuonCkfTrajectoryBuilder::MuonCkfTrajectoryBuilder(const edm::ParameterSet&              conf,
						   const TrajectoryStateUpdator*         updator,
						   const Propagator*                     propagatorAlong,
						   const Propagator*                     propagatorOpposite,
						   const Chi2MeasurementEstimatorBase*   estimator,
						   const TransientTrackingRecHitBuilder* RecHitBuilder,
						   const MeasurementTracker*             measurementTracker): 
  CkfTrajectoryBuilder(conf,updator,propagatorAlong,propagatorOpposite,estimator,RecHitBuilder,measurementTracker)
{
  //and something specific to me ?
  theUseSeedLayer = conf.getParameter<bool>("useSeedLayer");
  theRescaleErrorIfFail = conf.getParameter<double>("rescaleErrorIfFail");
}

MuonCkfTrajectoryBuilder::~MuonCkfTrajectoryBuilder()
{  delete (CkfTrajectoryBuilder*)this;}


void MuonCkfTrajectoryBuilder::collectMeasurement(const std::vector<const DetLayer*>& nl,const TrajectoryStateOnSurface & currentState, std::vector<TM>& result,int& invalidHits) const{
  for (std::vector<const DetLayer*>::const_iterator il = nl.begin();
       il != nl.end(); il++) {
    std::vector<TM> tmp =
      theLayerMeasurements->measurements((**il),currentState, *theForwardPropagator, *theEstimator);
    
    LogDebug("CkfPattern")<<tmp.size()<<" measurements returned by LayerMeasurements";
    
    if ( !tmp.empty()) {
      // FIXME durty-durty-durty cleaning: never do that please !
      /*      for (vector<TM>::iterator it = tmp.begin(); it!=tmp.end(); ++it)
              {if (it->recHit()->det()==0) it=tmp.erase(it)--;}*/
      
      if ( result.empty()) result = tmp;
      else {
        // keep one dummy TM at the end, skip the others
        result.insert( result.end()-invalidHits, tmp.begin(), tmp.end());
      }
      invalidHits++;
    }
  }
  
  LogDebug("CkfPattern")<<result.size()<<" total measurements";
  for (std::vector<TrajectoryMeasurement>::iterator it = result.begin(); it!=result.end();++it){
    LogDebug("CkfPattern")<<"layer pointer: "<<it->layer()<<"\n"
                          <<"estimate: "<<it->estimate()<<"\n"
                          <<"forward state: \n"<<it->forwardPredictedState()
                          <<"geomdet pointer from rechit: "<<it->recHit()->det()<<"\n"
                          <<"detId: "<<it->recHit()->geographicalId().rawId();
  }
  
}


void 
MuonCkfTrajectoryBuilder::findCompatibleMeasurements( const TempTrajectory& traj, 
						  std::vector<TrajectoryMeasurement> & result) const
{
  int invalidHits = 0;


  std::vector<const DetLayer*> nl;

  if (traj.empty())
    {
      edm::LogInfo("CkfPattern")<<"using JR patch for no measurement case";           
     //what if there are no measurement on the Trajectory

      //set the currentState to be the one from the trajectory seed starting point
      PTrajectoryStateOnDet ptod =traj.seed().startingState();
      DetId id(ptod.detId());
      const GeomDet * g = theMeasurementTracker->geomTracker()->idToDet(id);
      const Surface * surface=&g->surface();
      TrajectoryStateTransform tsTransform;
      TrajectoryStateOnSurface currentState(tsTransform.transientState(ptod,surface,theForwardPropagator->magneticField()));

      //set the next layers to be that one the state is on
      const DetLayer * l=theMeasurementTracker->geometricSearchTracker()->detLayer(id);

      if (theUseSeedLayer){
        //get the measurementsin the the layer first
        //      if ( traj.direction() == alongMomentum ){
        {
	  edm::LogInfo("CkfPattern")<<"using the layer of the seed first.";
          //will fail if the building is outside-in
          //because the propagator will cross over the barrel and give measurement on the other side of the barrel
          nl.clear();
          nl.push_back(l);
          collectMeasurement(nl,currentState,result,invalidHits);
        }


        //if fails: try to rescale locally the state to find measurements
        if ((result.size()==0 || ((uint)invalidHits==result.size())) && theRescaleErrorIfFail!=1.0)
          {
	    result.clear();
	    edm::LogInfo("CkfPattern")<<"using a rescale by "<< theRescaleErrorIfFail <<" to find measurements.";
	    TrajectoryStateOnSurface rescaledCurrentState = currentState;
	    rescaledCurrentState.rescaleError(theRescaleErrorIfFail);
	    invalidHits=0;
	    collectMeasurement(nl,rescaledCurrentState,result,invalidHits);
          }
      }

      //if fails: go to next layers
      if (result.size()==0 || ((uint)invalidHits==result.size()))
        {
          result.clear();
	  edm::LogInfo("CkfPattern")<<"using JR patch: need to go to next layer to get measurements";
          //the following will "JUMP" the first layer measurements
          nl = l->nextLayers(*currentState.freeState(), traj.direction());
          invalidHits=0;
          collectMeasurement(nl,currentState,result,invalidHits);
        }

      //if fails: this is on the next layers already, try rescaling locally the state
      if ((result.size()==0 || ((uint)invalidHits==result.size())) && theRescaleErrorIfFail!=1.0)
        {
          result.clear();
	  edm::LogInfo("CkfPattern")<<"using a rescale by "<< theRescaleErrorIfFail <<" to find measurements on next layers.";
          TrajectoryStateOnSurface rescaledCurrentState = currentState;
          rescaledCurrentState.rescaleError(theRescaleErrorIfFail);
          invalidHits=0;
          collectMeasurement(nl,rescaledCurrentState, result,invalidHits);
        }

    }
  else //regular case
    {

      TSOS currentState( traj.lastMeasurement().updatedState());

      nl = traj.lastLayer()->nextLayers( *currentState.freeState(), traj.direction());
  
      if (nl.empty()) return;

      for (std::vector<const DetLayer*>::iterator il = nl.begin(); 
	   il != nl.end(); il++) {
	std::vector<TM> tmp = 
	  theLayerMeasurements->measurements((**il),currentState, *theForwardPropagator, *theEstimator);

	if ( !tmp.empty()) {
	  if ( result.empty()) result = tmp;
	  else {
	    // keep one dummy TM at the end, skip the others
	    result.insert( result.end()-invalidHits, tmp.begin(), tmp.end());
	  }
	  invalidHits++;
	}
      }
    }


  // sort the final result, keep dummy measurements at the end
  if ( result.size() > 1) {
    sort( result.begin(), result.end()-invalidHits, TrajMeasLessEstim());
  }

#ifdef DEBUG_INVALID
  bool afterInvalid = false;
  for (std::vector<TM>::const_iterator i=result.begin();
       i!=result.end(); i++) {
    if ( ! i->recHit().isValid()) afterInvalid = true;
    if (afterInvalid && i->recHit().isValid()) {
      edm::LogError("CkfPattern") << "CkfTrajectoryBuilder error: valid hit after invalid!" ;
    }
  }
#endif

  //analyseMeasurements( result, traj);

}


