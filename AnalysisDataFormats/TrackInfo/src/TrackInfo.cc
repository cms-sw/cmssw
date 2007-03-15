#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
using namespace reco;

TrackInfo::TrackInfo( const TrajectorySeed & seed ,const  TrajectoryInfo & trajstate): seed_(seed),trajstate_(trajstate){}

//TrackRef TrackInfo::track() {return track_;}

const TrajectorySeed & TrackInfo::seed() const {return seed_;}

const TrackInfo::TrajectoryInfo & TrackInfo::trajStateMap() const {return trajstate_;}

const reco::TrackingRecHitInfo::RecHitType  TrackInfo::type(TrackingRecHitRef hit) const {return trajstate_.find(hit)->second.type();}

const PTrajectoryStateOnDet & TrackInfo::stateOnDet(TrackingRecHitRef hit)const {return trajstate_.find(hit)->second.stateOnDet();}

const LocalVector  TrackInfo::localTrackMomentum(TrackingRecHitRef hit)const{ return ((trajstate_.find(hit))->second.stateOnDet().parameters()).momentum();} 

const LocalVector  TrackInfo::localTrackMomentumOnMono(TrackingRecHitRef hit)const{ return (trajstate_.find(hit))->second.localTrackMomentumOnMono();} 

const LocalVector  TrackInfo::localTrackMomentumOnStereo(TrackingRecHitRef hit)const{ return (trajstate_.find(hit))->second.localTrackMomentumOnStereo();} 

const LocalPoint  TrackInfo::localTrackPosition(TrackingRecHitRef hit)const { return ((trajstate_.find(hit))->second.stateOnDet().parameters()).position();} 

