#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
using namespace reco;

TrackInfo::TrackInfo( const TrajectorySeed & seed ,const  TrajectoryInfo & trajstate): seed_(seed),trajstate_(trajstate){}

//TrackRef TrackInfo::track() {return track_;}

const TrajectorySeed & TrackInfo::seed() const {return seed_;}

const TrackInfo::TrajectoryInfo & TrackInfo::trajStateMap() const {return trajstate_;}

const PTrajectoryStateOnDet & TrackInfo::stateOnDet(TrackingRecHitRef hit)const {return trajstate_.find(hit)->second;}

const LocalVector  TrackInfo::localTrackMomentum(TrackingRecHitRef hit)const{ return ((trajstate_.find(hit))->second.parameters()).momentum();} 

const LocalPoint  TrackInfo::localTrackPosition(TrackingRecHitRef hit)const { return ((trajstate_.find(hit))->second.parameters()).position();} 

