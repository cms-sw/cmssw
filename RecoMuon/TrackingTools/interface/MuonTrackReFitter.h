#ifndef RecoMuon_TrackingTools_MuonTrackReFitter_H
#define RecoMuon_TrackingTools_MuonTrackReFitter_H

/**  \class MuonTrackReFitter
 *
 *   Algorithm to refit a muon track in the
 *   muon chambers and the tracker.
 *   It consists of a standard Kalman forward fit
 *   and a Kalman backward smoother.
 *
 *
 *   $Date: 2007/04/12 00:39:14 $
 *   $Revision: 1.5 $
 *
 *   \author   N. Neumeister            Purdue University
 */

#include <vector>
                                                                    
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

class Propagator;
class TrajectoryStateUpdator;
class MuonServiceProxy;
class MeasurementEstimator;

namespace edm {class ParameterSet; class EventSetup;}
 
//              ---------------------
//              -- Class Interface --
//              ---------------------

class MuonTrackReFitter : public TrajectorySmoother {

  public:
  
    typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;

    /// default constructor
    MuonTrackReFitter(const edm::ParameterSet&, const MuonServiceProxy*);

    /// destructor
    virtual ~MuonTrackReFitter();

    /// refit trajectory
    virtual TrajectoryContainer trajectories(const Trajectory&) const;

    /// refit trajectory
    virtual TrajectoryContainer trajectories(const TrajectorySeed& seed,
				             const ConstRecHitContainer& hits, 
				             const TrajectoryStateOnSurface& firstPredTsos) const;

    /// refit trajectory
    virtual TrajectoryContainer trajectory(const Trajectory&) const;

    /// refit trajectory
    virtual TrajectoryContainer trajectory(const TrajectorySeed& seed,
					   const ConstRecHitContainer& hits, 
					   const TrajectoryStateOnSurface& firstPredTsos) const;
    /// clone
    MuonTrackReFitter* clone() const {
      return new MuonTrackReFitter(*this);
    }

    std::string propagatorAlongMomentum() const {return theAlongMomentumProp;}

  private:    

    std::vector<Trajectory> fit(const Trajectory&) const;
    std::vector<Trajectory> fit(const TrajectorySeed& seed,
                                const ConstRecHitContainer& hits,
                                const TrajectoryStateOnSurface& firstPredTsos) const;
    std::vector<Trajectory> smooth(const std::vector<Trajectory>& ) const;
    std::vector<Trajectory> smooth(const Trajectory&) const;

  private:

    typedef TrajectoryStateOnSurface TSOS;
    typedef TrajectoryMeasurement TM;

  private:

    const MuonServiceProxy *theService;
    const TrajectoryStateUpdator* theUpdator;
    const MeasurementEstimator* theEstimator;
    float theErrorRescaling;

    std::string theOppositeToMomentumProp;
    std::string theAlongMomentumProp;
};

#endif
