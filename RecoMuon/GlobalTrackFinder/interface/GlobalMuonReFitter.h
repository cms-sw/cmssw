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
 *   $Date: 2006/07/24 01:20:31 $
 *   $Revision: 1.6 $
 *
 *   \author   N. Neumeister            Purdue University
 */

#include <vector>
                                                                    
#include "TrackingTools/PatternTools/interface/TrajectorySmoother.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class Propagator;
class KFUpdator;
class MuonTrajectoryUpdator;
class MagneticField;
class MeasurementEstimator;
 
//              ---------------------
//              -- Class Interface --
//              ---------------------

class GlobalMuonReFitter : public TrajectorySmoother {

  public:
  
    typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;

    /// default constructor
    GlobalMuonReFitter(const edm::ParameterSet&);

    /// destructor
    virtual ~GlobalMuonReFitter();

    /// initialize propagators
    void setES(const edm::EventSetup& iSetup);

    ///
    virtual TrajectoryContainer trajectories(const Trajectory& t) const;

    ///
    virtual TrajectoryContainer trajectories(const TrajectorySeed& seed,
				             const ConstRecHitContainer& hits, 
				             const TrajectoryStateOnSurface& firstPredTsos) const;

    ///
    GlobalMuonReFitter* clone() const {
      return new GlobalMuonReFitter(*this);
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
    const KFUpdator* theUpdator;
    MuonTrajectoryUpdator* theTrajectoryUpdator;
    const MeasurementEstimator* theEstimator;
    float theErrorRescaling;

    std::string theInPropagatorAlongMom;
    std::string theOutPropagatorAlongMom;
    std::string theInPropagatorOppositeToMom;
    std::string theOutPropagatorOppositeToMom;
    edm::ESHandle<MagneticField> theField;
  
};

#endif

