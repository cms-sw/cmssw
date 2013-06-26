#include "RecoMuon/L3TrackFinder/interface/MuonCkfTrajectoryBuilder.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorBase.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "RecoMuon/L3TrackFinder/src/EtaPhiEstimator.h"
#include <sstream>

MuonCkfTrajectoryBuilder::MuonCkfTrajectoryBuilder(const edm::ParameterSet&              conf,
						   const TrajectoryStateUpdator*         updator,
						   const Propagator*                     propagatorAlong,
						   const Propagator*                     propagatorOpposite,
						   const Propagator*                     propagatorProximity,
						   const Chi2MeasurementEstimatorBase*   estimator,
						   const TransientTrackingRecHitBuilder* RecHitBuilder,
						   const MeasurementTracker*             measurementTracker,
						   const TrajectoryFilter*               filter): 
  CkfTrajectoryBuilder(conf,updator,propagatorAlong,propagatorOpposite,estimator,RecHitBuilder,measurementTracker,filter),
  theProximityPropagator(propagatorProximity)
{
  //and something specific to me ?
  theUseSeedLayer = conf.getParameter<bool>("useSeedLayer");
  theRescaleErrorIfFail = conf.getParameter<double>("rescaleErrorIfFail");
  double dEta=conf.getParameter<double>("deltaEta");
  double dPhi=conf.getParameter<double>("deltaPhi");
  if (dEta>0 && dPhi>0)
    theEtaPhiEstimator = new EtaPhiEstimator(dEta,dPhi,theEstimator);
  else theEtaPhiEstimator = (Chi2MeasurementEstimatorBase*)0;


}

MuonCkfTrajectoryBuilder::~MuonCkfTrajectoryBuilder()
{
  if (theEtaPhiEstimator) delete theEtaPhiEstimator;
}

/*
std::string dumpMeasurement(const TrajectoryMeasurement & tm)
{
  std::stringstream buffer;
  buffer<<"layer pointer: "<<tm.layer()<<"\n"
        <<"estimate: "<<tm.estimate()<<"\n"
        <<"forward state: \n"
        <<"x: "<<tm.forwardPredictedState().globalPosition()<<"\n"
        <<"p: "<<tm.forwardPredictedState().globalMomentum()<<"\n"
        <<"geomdet pointer from rechit: "<<tm.recHit()->det()<<"\n"
        <<"detId: "<<tm.recHit()->geographicalId().rawId();
  if (tm.recHit()->isValid()){
    buffer<<"\n hit global x: "<<tm.recHit()->globalPosition()
	  <<"\n hit global error: "<<tm.recHit()->globalPositionError().matrix()
	  <<"\n hit local x:"<<tm.recHit()->localPosition()
	  <<"\n hit local error"<<tm.recHit()->localPositionError();
  }
  return buffer.str();
}
std::string dumpMeasurements(const std::vector<TrajectoryMeasurement> & v)
{
  std::stringstream buffer;
  buffer<<v.size()<<" total measurements\n";
  for (std::vector<TrajectoryMeasurement>::const_iterator it = v.begin(); it!=v.end();++it){
    buffer<<dumpMeasurement(*it);
    buffer<<"\n";}
  return buffer.str();
}
*/

void MuonCkfTrajectoryBuilder::collectMeasurement(const DetLayer* layer,
						  const std::vector<const DetLayer*>& nl,
						  const TrajectoryStateOnSurface & currentState,
						  std::vector<TM>& result,int& invalidHits,
						  const Propagator * prop) const{
  for (std::vector<const DetLayer*>::const_iterator il = nl.begin();
       il != nl.end(); il++) {

    TSOS stateToUse = currentState;
    
    if (layer == (*il)){
      LogDebug("CkfPattern")<<" self propagating in findCompatibleMeasurements.\n from: \n"<<stateToUse;
      //self navigation case
      // go to a middle point first
      TransverseImpactPointExtrapolator middle;
      GlobalPoint center(0,0,0);
      stateToUse = middle.extrapolate(stateToUse, center, *prop);

      if (!stateToUse.isValid()) continue;
      LogDebug("CkfPattern")<<"to: "<<stateToUse;
    }

    std::vector<TM> tmp =
      theLayerMeasurements->measurements((**il),stateToUse, *prop, *theEstimator);
    
    if (tmp.size()==1 && theEtaPhiEstimator){
      LogDebug("CkfPattern")<<"only an invalid hit is found. trying differently";
      tmp = theLayerMeasurements->measurements((**il),stateToUse, *prop, *theEtaPhiEstimator);
    }
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
  
  LogDebug("CkfPattern")<<"starting from:\n"
                        <<"x: "<<currentState.globalPosition()<<"\n"
                        <<"p: "<<currentState.globalMomentum()<<"\n"
                        <<PrintoutHelper::dumpMeasurements(result);
}
 


void 
MuonCkfTrajectoryBuilder::findCompatibleMeasurements(const TrajectorySeed&seed,
						     const TempTrajectory& traj, 
						     std::vector<TrajectoryMeasurement> & result) const
{
  int invalidHits = 0;


  std::vector<const DetLayer*> nl;

  if (traj.empty())
    {
      LogDebug("CkfPattern")<<"using JR patch for no measurement case";           
     //what if there are no measurement on the Trajectory

      //set the currentState to be the one from the trajectory seed starting point
      PTrajectoryStateOnDet ptod =seed.startingState();
      DetId id(ptod.detId());
      const GeomDet * g = theMeasurementTracker->geomTracker()->idToDet(id);
      const Surface * surface=&g->surface();
      
      TrajectoryStateOnSurface currentState(trajectoryStateTransform::transientState(ptod,surface,theForwardPropagator->magneticField()));

      //set the next layers to be that one the state is on
      const DetLayer * l=theMeasurementTracker->geometricSearchTracker()->detLayer(id);

      if (theUseSeedLayer){
        {
	  //get the measurements on the layer first
	  LogDebug("CkfPattern")<<"using the layer of the seed first.";
          nl.push_back(l);
          collectMeasurement(l,nl,currentState,result,invalidHits,theProximityPropagator);
        }
	
        //if fails: try to rescale locally the state to find measurements
        if ((unsigned int)invalidHits==result.size() && theRescaleErrorIfFail!=1.0 && result.size()!=0)
          {
	    result.clear();
	    LogDebug("CkfPattern")<<"using a rescale by "<< theRescaleErrorIfFail <<" to find measurements.";
	    TrajectoryStateOnSurface rescaledCurrentState = currentState;
	    rescaledCurrentState.rescaleError(theRescaleErrorIfFail);
	    invalidHits=0;
	    collectMeasurement(l,nl,rescaledCurrentState,result,invalidHits,theProximityPropagator);
          }
      }

      //if fails: go to next layers
      if (result.size()==0 || (unsigned int)invalidHits==result.size())
        {
          result.clear();
	  LogDebug("CkfPattern")<<"Need to go to next layer to get measurements";
          //the following will "JUMP" the first layer measurements
	  nl = l->nextLayers(*currentState.freeState(), traj.direction());
	  if (nl.size()==0){
            LogDebug("CkfPattern")<<" there was no next layer with wellInside. Use the next with no check.";
            //means you did not get any compatible layer on the next 1/2 tracker layer.
            // use the next layers with no checking
            nl = l->nextLayers(((traj.direction()==alongMomentum)?insideOut:outsideIn));
          }
          invalidHits=0;
          collectMeasurement(l,nl,currentState,result,invalidHits,theForwardPropagator);
        }

      //if fails: this is on the next layers already, try rescaling locally the state
      if (result.size()!=0 && (unsigned int)invalidHits==result.size() && theRescaleErrorIfFail!=1.0)
        {
          result.clear();
	  LogDebug("CkfPattern")<<"using a rescale by "<< theRescaleErrorIfFail <<" to find measurements on next layers.";
          TrajectoryStateOnSurface rescaledCurrentState = currentState;
          rescaledCurrentState.rescaleError(theRescaleErrorIfFail);
          invalidHits=0;
          collectMeasurement(l,nl,rescaledCurrentState, result,invalidHits,theForwardPropagator);
        }

    }
  else //regular case
    {

      TSOS currentState( traj.lastMeasurement().updatedState());

      nl = traj.lastLayer()->nextLayers( *currentState.freeState(), traj.direction());
      if (nl.empty()){LogDebug("CkfPattern")<<" no next layers... going "<<traj.direction()<<"\n from: \n"<<currentState<<"\n from detId: "<<traj.lastMeasurement().recHit()->geographicalId().rawId(); return ;}

      collectMeasurement(traj.lastLayer(),nl,currentState,result,invalidHits,theForwardPropagator);
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


