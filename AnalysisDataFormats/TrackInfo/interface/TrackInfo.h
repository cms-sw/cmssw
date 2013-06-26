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
 * \version $Id: TrackInfo.h,v 1.7 2007/04/18 16:46:37 genta Exp $
 *
 */

#include "AnalysisDataFormats/TrackInfo/interface/TrackInfoFwd.h"
#include "AnalysisDataFormats/TrackInfo/interface/TrackingRecHitInfo.h"
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
    typedef std::map<TrackingRecHitRef , TrackingRecHitInfo >  TrajectoryInfo;
    typedef reco::StateType StateType;

    TrackInfo() {}

    TrackInfo( const TrajectorySeed & seed_, const TrajectoryInfo & trajstate);

    //TrackRef track();

    const TrajectorySeed &seed() const;

    const RecHitType  type(TrackingRecHitRef ) const;
    
    const PTrajectoryStateOnDet *stateOnDet(StateType,TrackingRecHitRef ) const;

    const LocalVector localTrackMomentum(StateType,TrackingRecHitRef ) const;

    const LocalVector localTrackMomentumOnMono(StateType,TrackingRecHitRef ) const;

    const LocalVector localTrackMomentumOnStereo(StateType,TrackingRecHitRef ) const;

    const LocalPoint localTrackPosition(StateType, TrackingRecHitRef ) const;

    const LocalPoint localTrackPositionOnMono(StateType,TrackingRecHitRef ) const;

    const LocalPoint localTrackPositionOnStereo(StateType,TrackingRecHitRef ) const;

    const TrajectoryInfo &trajStateMap() const;

    //    void add(PTrajectoryStateOnDet  state,const TrackingRecHitRef hitref);

  private:
    TrajectorySeed  seed_ ;
    TrajectoryInfo    trajstates_;
  };

}

#endif
