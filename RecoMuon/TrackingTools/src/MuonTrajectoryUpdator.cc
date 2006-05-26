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


#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

// FIXME: can I change it in TransientTrackingRecHit??
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "Utilities/Timing/interface/TimingReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;

/// Constructor with Propagator and Parameter set
MuonTrajectoryUpdator::MuonTrajectoryUpdator(Propagator *propagator,
					     const edm::ParameterSet& par):thePropagator(propagator){
  
  // The max allowed chi2 to accept a rechit in the fit
  theMaxChi2 = par.getParameter<double>("MaxChi2");
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2);

  // The granularity
  theGranularity = par.getParameter<int>("Granularity");
}

/// Destructor
MuonTrajectoryUpdator::~MuonTrajectoryUpdator(){
  Propagator* thePropagator;
  MeasurementEstimator* theEstimator;
}



bool MuonTrajectoryUpdator::update(const TrajectoryMeasurement& theMeas, 
				   Trajectory& theTraj) const{}



TrajectoryStateOnSurface 
MuonTrajectoryUpdator::propagateState(const TrajectoryStateOnSurface& state,
				      const TrajectoryMeasurement& theMeas, 
				      const MuonTransientTrackingRecHit& current) const{}
