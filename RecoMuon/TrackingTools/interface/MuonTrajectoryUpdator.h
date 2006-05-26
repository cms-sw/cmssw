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
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \author S. Lacaprara - INFN Legnaro
 */

class Propagator;
class MeasurementEstimator;
class TrajectoryMeasurement;
class Trajectory;
class TrajectoryStateOnSurface;
class MuonTransientTrackingRecHit;

namespace edm{class ParameterSet;}

class MuonTrajectoryUpdator {
public:

  /// Constructor with Propagator and Parameter set
  MuonTrajectoryUpdator(Propagator *propagator,
			const edm::ParameterSet& par);

  /// Destructor
  virtual ~MuonTrajectoryUpdator();
  
  //   /// virtual construcor
  //   virtual MuonTrajectoryUpdator* clone() const {
  //     return new MuonTrajectoryUpdator(*this);
  //   }
  
  // Operations

  /// update the Trajectory with the TrajectoryMeasurement
  virtual bool update(const TrajectoryMeasurement& theMeas, 
		      Trajectory& theTraj) const;
  
  /// accasso at the propagator
  const Propagator& propagator() const {return *thePropagator;}
  const MeasurementEstimator& estimator() const {return *theEstimator;}

  /// get the max chi2 allowed
  double maxChi2() const {return theMaxChi2 ;}
  
  /// set max chi2
  void setMaxChi2(double chi2) { theMaxChi2=chi2; }

protected:
  
private:

  /// Propagate the state to the hit surface if it's a multi hit RecHit.
  /// i.e.: if "current" is a sub-rechit of the mesurement (i.e. a 1/2D RecHit)
  /// the state will be propagated to the surface where lies the "current" rechit 
  TrajectoryStateOnSurface propagateState(const TrajectoryStateOnSurface& state,
					  const TrajectoryMeasurement& theMeas, 
					  const MuonTransientTrackingRecHit& current) const;
  
  ///  the max chi2 allowed
  double theMaxChi2;
  
  /// the granularity
  /// if 0 segments are used, if 1 the 1/2D rechit are used
  /// Maybe in a second step there will be more than 2 option
  /// i.e. max granularity for DT but not for the CSC and the viceversa
  int theGranularity; 
  
  // FIXME: ask Tim if the CSC segments can be used, since in ORCA they wasn't.
  
  Propagator* thePropagator;
  MeasurementEstimator* theEstimator;
};
#endif

