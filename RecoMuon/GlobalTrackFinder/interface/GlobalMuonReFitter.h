#ifndef GlobalTrackFinder_GlobalMuonReFitter_H
#define GlobalTrackFinder_GlobalMuonReFitter_H

/**  \class GlobalMuonReFitter
 *
 *   Algorithm to refit a muon track in the
 *   muon chambers and the tracker.
 *   It consists of a standard Kalman forward fit
 *   and a Kalman backward smoother.
 *
 *
 *   $Date: 2005/07/22 12:42:02 $
 *   $Revision: 1.1 $
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
class KFUpdator;
class MagneticField;
class MeasurementEstimator;
 
//              ---------------------
//              -- Class Interface --
//              ---------------------

class GlobalMuonReFitter : public TrajectorySmoother {

  public:
  
    /// default constructor
    GlobalMuonReFitter(const MagneticField *);

    /// destructor
    virtual ~GlobalMuonReFitter();

    ///
    virtual TrajectoryContainer trajectories(const Trajectory& t) const;

    ///
    virtual TrajectoryContainer trajectories(const TrajectorySeed& seed,
				            const edm::OwnVector<TransientTrackingRecHit>& hits, 
				            const TrajectoryStateOnSurface& firstPredTsos) const;

    ///
    GlobalMuonReFitter* clone() const {
      return new GlobalMuonReFitter(theField);
    }

  private:    

     vector<Trajectory> fit(const Trajectory&) const;
     vector<Trajectory> fit(const TrajectorySeed& seed,
                            const edm::OwnVector<TransientTrackingRecHit>& hits,
                            const TrajectoryStateOnSurface& firstPredTsos) const;
     vector<Trajectory> smooth(const Trajectory&) const;
     vector<Trajectory> smooth(vector<Trajectory>& ) const;

  private:

    typedef TrajectoryStateOnSurface TSOS;
    typedef TrajectoryMeasurement TM;

  private:
    const MagneticField* theField;
    const Propagator* thePropagator1;
    const Propagator* thePropagator2;
    const KFUpdator* theUpdator;
    const MeasurementEstimator* theEstimator;
    float theErrorRescaling;
  
};

#endif

