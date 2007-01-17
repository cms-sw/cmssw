#include "AnalysisDataFormats/TrackInfo/interface/TrackInfo.h"
using namespace reco;

TrackInfo::TrackInfo( const TrajectorySeed & seed ,const  TrajectoryInfo & trajstate): seed_(seed),trajstate_(trajstate){}

//TrackRef TrackInfo::track() {return track_;}

const TrajectorySeed & TrackInfo::seed() const {return seed_;}

const TrackInfo::TrajectoryInfo & TrackInfo::trajStateMap() const {return trajstate_;}

const PTrajectoryStateOnDet & TrackInfo::stateOnDet(TrackingRecHitRef hit) {return trajstate_[hit];}

const LocalVector  TrackInfo::localTrackMomentum(TrackingRecHitRef hit){ return ((trajstate_[hit]).parameters()).momentum();} 

const LocalPoint  TrackInfo::localTrackPosition(TrackingRecHitRef hit){ return ((trajstate_[hit]).parameters()).position();} 

