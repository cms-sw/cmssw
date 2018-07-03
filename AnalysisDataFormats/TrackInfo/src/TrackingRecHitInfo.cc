#include "AnalysisDataFormats/TrackInfo/interface/TrackingRecHitInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;

const LocalVector TrackingRecHitInfo::localTrackMomentumOnMono(StateType statetype) const {
  TrackingStates::const_iterator state=states_.find(statetype);
  if(state!=states_.end())return state->second.localTrackMomentumOnMono();
  else edm::LogError("TrackingRecHitInfo")<<"This state does not exist";
  return LocalVector(0,0,0); 
}


const LocalVector TrackingRecHitInfo::localTrackMomentumOnStereo(StateType statetype)const {
  TrackingStates::const_iterator state=states_.find(statetype);
  if(state!=states_.end())return state->second.localTrackMomentumOnStereo();
  else edm::LogError("TrackingRecHitInfo")<<"This state does not exist";
  return LocalVector(0,0,0); 
}

const LocalPoint TrackingRecHitInfo::localTrackPositionOnMono(StateType statetype) const {
  TrackingStates::const_iterator state=states_.find(statetype);
  if(state!=states_.end())return state->second.localTrackPositionOnMono();
  else edm::LogError("TrackingRecHitInfo")<<"This state does not exist";
  return LocalPoint(0,0,0); 
}

const LocalPoint TrackingRecHitInfo::localTrackPositionOnStereo(StateType statetype)const {
  TrackingStates::const_iterator state=states_.find(statetype);
  if(state!=states_.end())return state->second.localTrackPositionOnStereo();
  else edm::LogError("TrackingRecHitInfo")<<"This state does not exist";
  return LocalPoint(0,0,0); 
}

const PTrajectoryStateOnDet * TrackingRecHitInfo::stateOnDet(StateType statetype)const {
  TrackingStates::const_iterator state=states_.find(statetype);
  if(state!=states_.end())return state->second.stateOnDet();
  else edm::LogError("TrackInfo")<<"This rechit does not exist";
  return nullptr;
}
