#ifndef TrackInfo_TrackingStateInfo_h
#define TrackInfo_TrackingStateInfo_h
/** \class reco::TrackingStateInfo TrackingStateInfo.h DataFormats/TrackAnalysisInfo/interface/TrackingStateInfo.h
 *
 * It contains additional info
 * for tracker studies
 * 
 *
 * \author Chiara Genta
 *
 * \version $Id: TrackingStateInfo.h,v 1.2 2011/12/22 19:19:31 innocent Exp $
 *
 */

#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoFwd.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

namespace reco {
  class TrackingStateInfo{
    
  public:
    //    enum StateType { Updated=0, Combined=1, FwPredicted=2, BwPredicted=3};
    
    //enum RecHitType { Single=0, Matched=1, Projected=2};
    
    TrackingStateInfo(){}
    TrackingStateInfo(std::pair<LocalVector, LocalVector> trackdirections ,std::pair<LocalPoint, LocalPoint> trackpositions , PTrajectoryStateOnDet const &trajstate ): 
      trackdirections_(trackdirections),  trackpositions_(trackpositions), trajstate_(trajstate) {}
      //const RecHitType  type() const {return type_;}
      //const StateType  statetype() const {return statetype_;}
      const LocalVector localTrackMomentumOnMono() const {return trackdirections_.first;}
      const LocalVector localTrackMomentumOnStereo()const {return trackdirections_.second;}
      const LocalPoint localTrackPositionOnMono() const {return trackpositions_.first;}
      const LocalPoint localTrackPositionOnStereo()const {return trackpositions_.second;}
      const PTrajectoryStateOnDet *stateOnDet() const {return &trajstate_;};
      
  private:
 
      std::pair<LocalVector, LocalVector> trackdirections_;
      std::pair<LocalPoint, LocalPoint> trackpositions_;
      PTrajectoryStateOnDet trajstate_;
      
  };
  
}
#endif
