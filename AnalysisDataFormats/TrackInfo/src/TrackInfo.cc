#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;


TrackInfo::TrackInfo( const TrajectorySeed & seed ,const  TrajectoryInfo & trajstates): seed_(seed),trajstates_(trajstates){}

//TrackRef TrackInfo::track() {return track_;}

const TrajectorySeed & TrackInfo::seed() const {return seed_;}

const reco::TrackInfo::TrajectoryInfo & TrackInfo::trajStateMap() const {return trajstates_;}

//continuare inversione vector->map
const reco::TrackingRecHitInfo::RecHitType  TrackInfo::type(StateType type, TrackingRecHitRef hit) const {
  std::vector<TrackingRecHitInfo>::const_iterator  trackingrechitinfo;
  const std::vector<TrackingRecHitInfo> * trackingrechitinfos=&trajstates_.find(hit)->second;
  for(trackingrechitinfo=trackingrechitinfos->begin();trackingrechitinfo!=trackingrechitinfos->end(); trackingrechitinfo++ ){
    if( trackingrechitinfo->statetype()==type)return trackingrechitinfo->type();
  }
  edm::LogError("TrackInfo")<<"This state does not exist";
  return TrackingRecHitInfo::Single;
}

const PTrajectoryStateOnDet * TrackInfo::stateOnDet(StateType type,TrackingRecHitRef hit)const {
    std::vector<TrackingRecHitInfo>::const_iterator  trackingrechitinfo;
  const std::vector<TrackingRecHitInfo> * trackingrechitinfos=&trajstates_.find(hit)->second;
  for(trackingrechitinfo=trackingrechitinfos->begin();trackingrechitinfo!=trackingrechitinfos->end(); trackingrechitinfo++ ){
    if( trackingrechitinfo->statetype()==type)return &(trackingrechitinfo->stateOnDet());
  }
  edm::LogError("TrackInfo")<<"This state does not exist";
  return 0;
}

const LocalVector  TrackInfo::localTrackMomentum(StateType type,TrackingRecHitRef hit)const{ 
    std::vector<TrackingRecHitInfo>::const_iterator  trackingrechitinfo;
  const std::vector<TrackingRecHitInfo> * trackingrechitinfos=&trajstates_.find(hit)->second;
  for(trackingrechitinfo=trackingrechitinfos->begin();trackingrechitinfo!=trackingrechitinfos->end(); trackingrechitinfo++ ){
    if( trackingrechitinfo->statetype()==type) return (trackingrechitinfo->stateOnDet().parameters()).momentum();
  }
  edm::LogError("TrackInfo")<<"This state does not exist";
  return LocalVector(0,0,0); 
}

const LocalVector  TrackInfo::localTrackMomentumOnMono(StateType type,TrackingRecHitRef hit)const{ 

   std::vector<TrackingRecHitInfo>::const_iterator  trackingrechitinfo;
  const std::vector<TrackingRecHitInfo> * trackingrechitinfos=&trajstates_.find(hit)->second;
 for(trackingrechitinfo=trackingrechitinfos->begin();trackingrechitinfo!=trackingrechitinfos->end(); trackingrechitinfo++ ){
    if( trackingrechitinfo->statetype()==type) return trackingrechitinfo->localTrackMomentumOnMono();
  }
  edm::LogError("TrackInfo")<<"This state does not exist";
  return LocalVector(0,0,0); 
}

const LocalVector  TrackInfo::localTrackMomentumOnStereo(StateType type,TrackingRecHitRef hit)const{

   std::vector<TrackingRecHitInfo>::const_iterator  trackingrechitinfo;
  const std::vector<TrackingRecHitInfo> * trackingrechitinfos=&trajstates_.find(hit)->second;

for(trackingrechitinfo=trackingrechitinfos->begin();trackingrechitinfo!=trackingrechitinfos->end(); trackingrechitinfo++ ){
    if( trackingrechitinfo->statetype()==type) return trackingrechitinfo->localTrackMomentumOnStereo();
  }
  edm::LogError("TrackInfo")<<"This state does not exist";
  return LocalVector(0,0,0); 
}

const LocalPoint  TrackInfo::localTrackPosition(StateType type,TrackingRecHitRef hit)const { 

    std::vector<TrackingRecHitInfo>::const_iterator  trackingrechitinfo;
  const std::vector<TrackingRecHitInfo> * trackingrechitinfos=&trajstates_.find(hit)->second;
  for(trackingrechitinfo=trackingrechitinfos->begin();trackingrechitinfo!=trackingrechitinfos->end(); trackingrechitinfo++ ){
    if( trackingrechitinfo->statetype()==type) return (trackingrechitinfo->stateOnDet().parameters()).position();
  } 
  edm::LogError("TrackInfo")<<"This state does not exist";
  return LocalPoint(0,0,0);
}


const LocalPoint  TrackInfo::localTrackPositionOnMono(StateType type,TrackingRecHitRef hit)const{ 

   std::vector<TrackingRecHitInfo>::const_iterator  trackingrechitinfo;
  const std::vector<TrackingRecHitInfo> * trackingrechitinfos=&trajstates_.find(hit)->second;
  for(trackingrechitinfo=trackingrechitinfos->begin();trackingrechitinfo!=trackingrechitinfos->end(); trackingrechitinfo++ ){
    if( trackingrechitinfo->statetype()==type) return trackingrechitinfo->localTrackPositionOnMono();
  }
  edm::LogError("TrackInfo")<<"This state does not exist";
  return LocalPoint(0,0,0); 
}

const LocalPoint  TrackInfo::localTrackPositionOnStereo(StateType type,TrackingRecHitRef hit)const{ 
    std::vector<TrackingRecHitInfo>::const_iterator  trackingrechitinfo;
  const std::vector<TrackingRecHitInfo> * trackingrechitinfos=&trajstates_.find(hit)->second;
  for(trackingrechitinfo=trackingrechitinfos->begin();trackingrechitinfo!=trackingrechitinfos->end(); trackingrechitinfo++ ){
    if( trackingrechitinfo->statetype()==type) return trackingrechitinfo->localTrackPositionOnStereo();
  }
  edm::LogError("TrackInfo")<<"This state does not exist";
  return LocalPoint(0,0,0); 
}


