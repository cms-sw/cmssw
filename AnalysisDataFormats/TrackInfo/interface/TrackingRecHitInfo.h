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
 * \version $Id: TrackingRecHitInfo.h,v 1.4 2007/04/18 16:46:37 genta Exp $
 *
 */

#include "AnalysisDataFormats/TrackInfo/interface/TrackingStateInfo.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoEnum.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"


namespace reco {
  class TrackingRecHitInfo{
    
  public:
    typedef reco::StateType StateType;
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
