#ifndef SimpleTrackRefitter_H_
#define SimpleTrackRefitter_H_

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Utilities/interface/InputTag.h>
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include <DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h>
#include <TrackingTools/PatternTools/interface/Trajectory.h>
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include <TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h>
#include <Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h>
#include <MagneticField/Engine/interface/MagneticField.h>
#include <vector>
#include <string>


/** \class SimpleTrackRefitter 
 * This class refits a track (from cosmic or standard tracking).
 * The final result is a Trajectory refitted and smoothed.
 * To make the refitting (and the smoothing) the usual KF tools are used.
 * An arbitrary DetUnit can be ignored during the fit, so that the 
 * resulting trajectory can be used for unbiased residual studies 
 * on that Det.
 */

class Propagator;
class KFUpdator;
class KFUpdator;
class Chi2MeasurementEstimator;
class TransientTrackingRecHitBuilder;
class TrajectorySmoother;
class TrajectoryFitter;

class SimpleTrackRefitter{

public:
  /// Constructor
  SimpleTrackRefitter(const edm::ParameterSet&);

  /// Destructor
  virtual ~SimpleTrackRefitter();

  /// The main methods
  std::vector<Trajectory> refitTrack(const reco::Track& newTrack, 
                                     const uint32_t ExcludedDetId = 0);
  std::vector<Trajectory> refitTrack(const TrajectorySeed& seed,
                                     const TrackingRecHitCollection &hits,
                                     const uint32_t ExcludedDetid = 0);
  std::vector<Trajectory> refitTrack(const TrajectorySeed& seed,
                                     const reco::Track& theT,
                                     const uint32_t ExcludedDetid = 0);

  // to be called before the refit
  void setServices(const edm::EventSetup& es);

private: 
  // methods for cosmic tracking
  void initServices(const bool& seedAlongMomentum);
  void destroyServices();
  std::vector<TrajectoryMeasurement> seedMeasurements(const TrajectorySeed& seed) const;
  TrajectoryStateOnSurface startingTSOS(const TrajectorySeed& seed)const;
  Trajectory createStartingTrajectory( const TrajectorySeed& seed) const;
  // members
  const Propagator* thePropagator;
  const Propagator* thePropagatorOp;
  const KFUpdator*  theUpdator;
  const Chi2MeasurementEstimator *theEstimator;
  const TransientTrackingRecHitBuilder *RHBuilder;
  const TrajectorySmoother * theSmoother;
  const TrajectoryFitter * theFitter;
  const TrajectoryStateTransform tsTransform;
  const TrackerGeometry* tracker;
  const MagneticField* magfield;
  edm::ParameterSet conf_;
  bool isCosmics_;
};  

#endif
