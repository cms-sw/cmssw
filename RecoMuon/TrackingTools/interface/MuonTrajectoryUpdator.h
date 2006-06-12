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
 *  $Date: 2006/05/29 17:22:59 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 *  \author S. Lacaprara - INFN Legnaro
 */

#include "DataFormats/Common/interface/OwnVector.h"

class Propagator;
class MeasurementEstimator;
class TrajectoryMeasurement;
class Trajectory;
class TrajectoryStateOnSurface;
class TransientTrackingRecHit;
class TrajectoryStateUpdator;
class DetLayer;

namespace edm{class ParameterSet;}

class MuonTrajectoryUpdator {
public:

  //<< very temp
  
  // FIXME: this c'tor is temp!!
  // It will dis as the Updator will be loaded in the es
  /// Constructor with Parameter set
  MuonTrajectoryUpdator(const edm::ParameterSet& par);
  // FIXME: this function is temp!!
  // It will dis as the Updator will be loaded in the es
  void setPropagator(Propagator* prop) {thePropagator = prop;}

  //>> end tmp


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
  virtual std::pair<bool,TrajectoryStateOnSurface>  update(const TrajectoryMeasurement* theMeas, 
							   Trajectory& theTraj);
  
  /// accasso at the propagator
  const Propagator *propagator() const {return thePropagator;}
  const MeasurementEstimator *estimator() const {return theEstimator;}
  const TrajectoryStateUpdator *measurementUpdator() const {return theUpdator;}

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
					  const TrajectoryMeasurement* theMeas, 
					  const TransientTrackingRecHit& current) const;
  
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

  /// I have to use this method since I have to cope with two propagation direction
  void ownVectorLimits(edm::OwnVector<const TransientTrackingRecHit> &ownvector,
		       edm::OwnVector<const TransientTrackingRecHit>::iterator &recHitsForFit_begin,
		       edm::OwnVector<const TransientTrackingRecHit>::iterator &recHitsForFit_end);

  /// I have to use this method since I have to cope with two propagation direction
  void incrementIterator(edm::OwnVector<const TransientTrackingRecHit>::iterator &recHitIterator);

  /// copy objs from an OwnVector to another one
  void insert(edm::OwnVector<const TransientTrackingRecHit> & to,
	      edm::OwnVector<const TransientTrackingRecHit> & from);


  /// Return the trajectory measurement. It handles both the fw and the bw propagation
  TrajectoryMeasurement updateMeasurement( const TrajectoryStateOnSurface &propagatedTSOS, 
					   const TrajectoryStateOnSurface &lastUpdatedTSOS, 
					   const TransientTrackingRecHit &recHit,
					   const double &chi2, const DetLayer *detLayer, 
					   const TrajectoryMeasurement *initialMeasurement);


  Propagator *thePropagator;
  MeasurementEstimator *theEstimator;
  TrajectoryStateUpdator *theUpdator;
  
};
#endif

