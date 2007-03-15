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
 * \version $Id: $
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
     enum RecHitType { Single=0, Matched=1, Projected=2};
     TrackingRecHitInfo(){}
     TrackingRecHitInfo(RecHitType type, std::pair<LocalVector, LocalVector> trackdirections,PTrajectoryStateOnDet trajstate ): 
     type_(type),  trackdirections_(trackdirections), trajstate_(trajstate) {}
     const RecHitType  type() const {return type_;}
     const LocalVector localTrackMomentumOnMono() const {return trackdirections_.first;}
     const LocalVector localTrackMomentumOnStereo()const {return trackdirections_.second;}
     const PTrajectoryStateOnDet &stateOnDet() const {return trajstate_;};

  private:
     RecHitType type_;
     std::pair<LocalVector, LocalVector> trackdirections_;
     PTrajectoryStateOnDet trajstate_;

  };

}
#endif
