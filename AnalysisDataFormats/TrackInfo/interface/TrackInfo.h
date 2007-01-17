#ifndef TrackInfo_TrackInfo_h
#define TrackInfo_TrackInfo_h
/** \class reco::TrackInfo TrackInfo.h DataFormats/TrackAnalysisInfo/interface/TrackInfo.h
 *
 * It contains additional info
 * for tracker studies
 * 
 *
 * \author Chiara Genta
 *
 * \version $Id: TrackInfo.h,v 1.1 2006/12/22 10:36:47 genta Exp $
 *
 */

#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/Common/interface/AssociationMap.h"
namespace reco {
   class TrackInfo{
  public:
    /// default constructor
        typedef std::map<TrackingRecHitRef , const PTrajectoryStateOnDet >  TrajectoryInfo;
    TrackInfo() {}

    TrackInfo( const TrajectorySeed & seed_, const TrajectoryInfo & trajstate);

    //TrackRef track();

    const TrajectorySeed &seed() const;

    const PTrajectoryStateOnDet &stateOnDet(TrackingRecHitRef);

    const LocalVector localTrackMomentum(TrackingRecHitRef);

    const LocalPoint localTrackPosition(TrackingRecHitRef);

    const TrajectoryInfo &trajStateMap() const;

    //    void add(PTrajectoryStateOnDet  state,const TrackingRecHitRef hitref);

  private:
    TrajectorySeed  seed_ ;
    TrajectoryInfo   trajstate_;
  };

}

#endif
