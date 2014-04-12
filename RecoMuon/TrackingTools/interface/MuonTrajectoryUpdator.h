#ifndef RecoMuon_TrackingTools_MuonTrajectoryUpdator_H
#define RecoMuon_TrackingTools_MuonTrajectoryUpdator_H

/** \class MuonTrajectoryUpdator
 *  An updator for the Muon system
 *  This class update a trajectory with a muon chamber measurement.
 *  In spite of the name, it is NOT an updator, but has one.
 *  A muon RecHit is a segment (for DT and CSC) or a "hit" (RPC).
 *  This updator is suitable both for FW and BW filtering. The difference between the two fitter are two:
 *  the granularity of the updating (i.e.: segment position or 1D rechit position), which can be set via
 *  parameter set, and the propagation direction which is embeded in the propagator set in the c'tor.
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \author S. Lacaprara - INFN Legnaro
 */

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/DetLayers/interface/NavigationDirection.h"

#include <functional>

class Propagator;
class MeasurementEstimator;
class TrajectoryMeasurement;
class Trajectory;
class TrajectoryStateOnSurface;
class TrajectoryStateUpdator;
class DetLayer;

namespace edm{class ParameterSet;}

class MuonTrajectoryUpdator {
  
 public:

  /// Constructor from Propagator and Parameter set
  MuonTrajectoryUpdator(const edm::ParameterSet& par,
			NavigationDirection fitDirection);

  /// Constructor from Propagator, chi2 and the granularity flag
  MuonTrajectoryUpdator(NavigationDirection fitDirection,
			double chi2, int granularity);
  
  /// Destructor
  virtual ~MuonTrajectoryUpdator();
  
  // Operations

  /// update the Trajectory with the TrajectoryMeasurement
  virtual std::pair<bool,TrajectoryStateOnSurface>  update(const TrajectoryMeasurement* measurement, 
							   Trajectory& trajectory,
							   const Propagator *propagator);

  /// accasso at the propagator
  const MeasurementEstimator *estimator() const {return theEstimator;}
  const TrajectoryStateUpdator *measurementUpdator() const {return theUpdator;}

  /// get the max chi2 allowed
  double maxChi2() const {return theMaxChi2 ;}
  
  /// get the fit direction
  NavigationDirection fitDirection() {return theFitDirection;}

  /// set max chi2
  void setMaxChi2(double chi2) {theMaxChi2 = chi2;}

  /// set fit direction
  void setFitDirection(NavigationDirection fitDirection) {theFitDirection = fitDirection;}

  /// reset the theFirstTSOSFlag
  void makeFirstTime();
  
 protected:
  
 private:

  /// Propagate the state to the hit surface if it's a multi hit RecHit.
  /// i.e.: if "current" is a sub-rechit of the mesurement (i.e. a 1/2D RecHit)
  /// the state will be propagated to the surface where lies the "current" rechit 
  TrajectoryStateOnSurface propagateState(const TrajectoryStateOnSurface& state,
					  const TrajectoryMeasurement* measurement, 
					  const TransientTrackingRecHit::ConstRecHitPointer& current,
					  const Propagator *propagator) const;
  
  ///  the max chi2 allowed
  double theMaxChi2;
  
  /// the granularity
  /// if 0 4D-segments are used both for the DT and CSC,
  /// if 1 2D-segments are used for the DT and the 2D-points for the CSC
  /// if 2 the 1D rec hit for the DT are used, while the 2D rechit for the CSC are used
  /// Maybe in a second step there will be more than 3 option
  /// i.e. max granularity for DT but not for the CSC and the viceversa
  int theGranularity; 
  // FIXME: ask Tim if the CSC segments can be used, since in ORCA they wasn't.

  /// Ordering along increasing radius (for DT rechits)
  struct RadiusComparatorInOut{
    bool operator()(const TransientTrackingRecHit::ConstRecHitPointer &a,
		    const TransientTrackingRecHit::ConstRecHitPointer &b) const{ 
      return a->det()->surface().position().perp() < b->det()->surface().position().perp(); 
    }
  };
  
  /// Ordering along decreasing radius (for DT rechits)
  struct RadiusComparatorOutIn{
    bool operator()(const TransientTrackingRecHit::ConstRecHitPointer &a, 
		    const TransientTrackingRecHit::ConstRecHitPointer &b) const{ 
      return a->det()->surface().position().perp() > b->det()->surface().position().perp();
    }
  };
  
  /// Ordering along increasing zed (for CSC rechits)
  struct ZedComparatorInOut{  
    bool operator()( const TransientTrackingRecHit::ConstRecHitPointer &a, 
		     const TransientTrackingRecHit::ConstRecHitPointer &b) const{ 
      return fabs(a->globalPosition().z()) < fabs(b->globalPosition().z()); 
    }
  };
  
  /// Ordering along decreasing zed (for CSC rechits)
  struct ZedComparatorOutIn{
    bool operator()( const TransientTrackingRecHit::ConstRecHitPointer &a, 
		     const TransientTrackingRecHit::ConstRecHitPointer &b) const{ 
      return fabs(a->globalPosition().z()) > fabs(b->globalPosition().z()); 
    }
  };

  void sort(TransientTrackingRecHit::ConstRecHitContainer&, const DetLayer*);
  
  /// Return the trajectory measurement. It handles both the fw and the bw propagation
  TrajectoryMeasurement updateMeasurement( const TrajectoryStateOnSurface &propagatedTSOS, 
					   const TrajectoryStateOnSurface &lastUpdatedTSOS, 
					   const TransientTrackingRecHit::ConstRecHitPointer &recHit,
					   const double &chi2, const DetLayer *detLayer, 
					   const TrajectoryMeasurement *initialMeasurement);
  

  // FIXME: change in a ESHandle
  MeasurementEstimator *theEstimator;
  TrajectoryStateUpdator *theUpdator;

  // The fit direction.This is the global fit direction and it could be (LOCALLY!) different w.r.t. the 
  // propagation direction embeeded in the propagator (i.e. when it is used in the "anyDirection" mode)
  // This data member is not set via parameter set since it must be consistent with the RefitterParameter.
  NavigationDirection theFitDirection;
  
  // Parameters for the error rescaling
  bool theFirstTSOSFlag;
  bool theRescaleErrorFlag;
  double theRescaleFactor;

  // parameter to use the "bad hits"
  bool useInvalidHits;
  // exclude RPC from the fit (but allows to keep it in the muon signal confirmation)
  bool theRPCExFlag;

};
#endif

