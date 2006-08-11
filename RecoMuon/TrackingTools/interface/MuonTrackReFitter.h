#ifndef TrackingTools_MuonTrackReFitter_H
#define TrackingTools_MuonTrackReFitter_H

/**  \class MuonTrackReFitter
 *
 *   Algorithm to refit a muon track in the
 *   muon chambers and the tracker.
 *   It consists of a standard Kalman forward fit
 *   and a Kalman backward smoother.
 *
 *
 *   $Date: 2006/08/03 $
 *   $Revision: $
 *
 *   \author   N. Neumeister            Purdue University
 */

#include <vector>
                                                                    
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "FWCore/Framework/interface/ESHandle.h"

class Propagator;
class TrajectoryStateUpdator;
class MagneticField;
class MeasurementEstimator;

namespace edm {class ParameterSet; class EventSetup;}
 
//              ---------------------
//              -- Class Interface --
//              ---------------------

class MuonTrackReFitter : public TrajectorySmoother {

  public:
  
    typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;

    /// default constructor
    MuonTrackReFitter(const edm::ParameterSet&, const edm::EventSetup&);

    /// destructor
    virtual ~MuonTrackReFitter();

    /// refit trajectory
    virtual TrajectoryContainer trajectories(const Trajectory& t) const;

    /// refit trajectory
    virtual TrajectoryContainer trajectories(const TrajectorySeed& seed,
				             const ConstRecHitContainer& hits, 
				             const TrajectoryStateOnSurface& firstPredTsos) const;

    /// return propagator
    Propagator* propagator() const { return thePropagator1; }

    /// clone
    MuonTrackReFitter* clone() const {
      return new MuonTrackReFitter(*this);
    }

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

    Propagator* thePropagator1;
    Propagator* thePropagator2;
    const TrajectoryStateUpdator* theUpdator;
    const MeasurementEstimator* theEstimator;
    float theErrorRescaling;

    edm::ESHandle<MagneticField> theMagField;
  
};

#endif
