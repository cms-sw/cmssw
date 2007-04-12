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
 * \version $Id: TrackingRecHitInfo.h,v 1.1 2007/03/15 13:53:58 genta Exp $
 *
 */

#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

namespace reco {
  class TrackingRecHitInfo{
    
  public:
    enum StateType { Updated=0, Combined=1, FwPredicted=2, BwPredicted=3};
    
    enum RecHitType { Single=0, Matched=1, Projected=2};
    
    TrackingRecHitInfo(){}
    TrackingRecHitInfo(StateType statetype, RecHitType type, std::pair<LocalVector, LocalVector> trackdirections ,std::pair<LocalPoint, LocalPoint> trackpositions , PTrajectoryStateOnDet &trajstate ): 
      statetype_(statetype), type_(type),  trackdirections_(trackdirections),  trackpositions_(trackpositions), trajstate_(trajstate) {}
      const RecHitType  type() const {return type_;}
      const StateType  statetype() const {return statetype_;}
      const LocalVector localTrackMomentumOnMono() const {return trackdirections_.first;}
      const LocalVector localTrackMomentumOnStereo()const {return trackdirections_.second;}
      const LocalPoint localTrackPositionOnMono() const {return trackpositions_.first;}
      const LocalPoint localTrackPositionOnStereo()const {return trackpositions_.second;}
      const PTrajectoryStateOnDet &stateOnDet() const {return trajstate_;};
      
  private:
      StateType statetype_;
      RecHitType type_;
      std::pair<LocalVector, LocalVector> trackdirections_;
      std::pair<LocalPoint, LocalPoint> trackpositions_;
      PTrajectoryStateOnDet trajstate_;
      
  };
  
}
#endif
