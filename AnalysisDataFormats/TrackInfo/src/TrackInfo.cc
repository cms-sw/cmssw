#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;


TrackInfo::TrackInfo( const TrajectorySeed & seed ,const  TrajectoryInfo & trajstates): seed_(seed),trajstates_(trajstates){}

const TrajectorySeed & TrackInfo::seed() const {return seed_;}

const reco::TrackInfo::TrajectoryInfo & TrackInfo::trajStateMap() const {return trajstates_;}

const RecHitType  TrackInfo::type(TrackingRecHitRef hit) const {
  TrajectoryInfo::const_iterator states=trajstates_.find(hit);
  if(states!=trajstates_.end())return states->second.type();
  else edm::LogError("TrackInfo")<<"This rechit does not exist";
  return Null;
}

const PTrajectoryStateOnDet * TrackInfo::stateOnDet(StateType statetype,TrackingRecHitRef hit)const {
  TrajectoryInfo::const_iterator states=trajstates_.find(hit);
  if(states!=trajstates_.end())return states->second.stateOnDet(statetype);
  else edm::LogError("TrackInfo")<<"This rechit does not exist";
  return 0;
}

const LocalVector  TrackInfo::localTrackMomentum(StateType statetype,TrackingRecHitRef hit)const{ 

  TrajectoryInfo::const_iterator states=trajstates_.find(hit);
  if(states!=trajstates_.end())
    {
      const PTrajectoryStateOnDet * state=states->second.stateOnDet(statetype);
      if(state!=0) return state->parameters().momentum();
    }
  else edm::LogError("TrackInfo")<<"This rechit does not exist";
  return LocalVector(0,0,0); 
}

const LocalVector  TrackInfo::localTrackMomentumOnMono(StateType statetype,TrackingRecHitRef hit)const{ 

  TrajectoryInfo::const_iterator states=trajstates_.find(hit);
  if(states!=trajstates_.end())return states->second.localTrackMomentumOnMono(statetype);
  else edm::LogError("TrackInfo")<<"This rechit does not exist";
  return LocalVector(0,0,0); 
}

const LocalVector  TrackInfo::localTrackMomentumOnStereo(StateType statetype,TrackingRecHitRef hit)const{

  TrajectoryInfo::const_iterator states=trajstates_.find(hit);
  if(states!=trajstates_.end())return states->second.localTrackMomentumOnStereo(statetype);
  return LocalVector(0,0,0); 
}

const LocalPoint  TrackInfo::localTrackPosition(StateType statetype,TrackingRecHitRef hit)const { 

  TrajectoryInfo::const_iterator states=trajstates_.find(hit);
  if(states!=trajstates_.end())
    {
      const PTrajectoryStateOnDet * state=states->second.stateOnDet(statetype);
      if(state!=0) return state->parameters().position();
    }
  else edm::LogError("TrackInfo")<<"This rechit does not exist";
  return LocalPoint(0,0,0);
}


const LocalPoint  TrackInfo::localTrackPositionOnMono(StateType statetype,TrackingRecHitRef hit)const{ 

  TrajectoryInfo::const_iterator states=trajstates_.find(hit);
  if(states!=trajstates_.end())return states->second.localTrackPositionOnMono(statetype);
  else edm::LogError("TrackInfo")<<"This rechit does not exist";
  return LocalPoint(0,0,0); 
}

const LocalPoint  TrackInfo::localTrackPositionOnStereo(StateType statetype,TrackingRecHitRef hit)const{ 

  TrajectoryInfo::const_iterator states=trajstates_.find(hit);
  if(states!=trajstates_.end())return states->second.localTrackPositionOnStereo(statetype);
  else edm::LogError("TrackInfo")<<"This rechit does not exist";
  return LocalPoint(0,0,0); 
}


