#ifndef TrackInfo_TrackingRecHitInfo_h
#define TrackInfo_TrackingRecHitInfo_h
/** \class reco::TrackingRecHitInfo TrackingRecHitInfo.h DataFormats/TrackAnalysisInfo/interface/TrackingRecHitInfo.h
 *
 * It contains additional info
 * for tracker studies
 * 
 *
 * \author Chiara Genta
 *
 * \version $Id: TrackingRecHitInfo.h,v 1.2 2007/04/12 14:06:05 genta Exp $
 *
 */

#include "AnalysisDataFormats/TrackInfo/interface/TrackingStateInfo.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

namespace reco {
  class TrackingRecHitInfo{
    
  public:
    enum StateType { Updated=0, Combined=1, FwPredicted=2, BwPredicted=3};
    
    enum RecHitType { Single=0, Matched=1, Projected=2, Null=3};
    typedef std::map<StateType, TrackingStateInfo > TrackingStates;
    
    TrackingRecHitInfo(){}
    TrackingRecHitInfo(RecHitType type, TrackingStates & states):type_(type),states_(states){}
        const RecHitType  type() const {return type_;}
    const LocalVector localTrackMomentumOnMono(StateType statetype) const;
    const LocalVector localTrackMomentumOnStereo(StateType statetype)const;
    const LocalPoint localTrackPositionOnMono(StateType statetype) const;
    const LocalPoint localTrackPositionOnStereo(StateType statetype)const;
    const TrackingStates &statesOnDet() const{return states_;}
    const PTrajectoryStateOnDet *stateOnDet(StateType statetype) const;
      
  private:
      RecHitType type_;
      TrackingStates states_;
  };
  
}
#endif
