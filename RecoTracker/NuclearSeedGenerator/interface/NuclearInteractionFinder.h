#ifndef CD_NuclearInteractionFinder_H_
#define CD_NuclearInteractionFinder_H_

//----------------------------------------------------------------------------
//! \class NuclearInteractionFinder
//! \brief Class used to obtain vector of all compatible TMs associated to a trajectory to be used by the NuclearTester.
//!
//!
//! \description The method run gets all compatible TMs of all TMs associated of a trajectory.
//! Then it uses the NuclearTester class to decide whether the trajectory has interacted nuclearly or not.
//! It finally returns a pair of the TM where the nuclear interaction occurs and all compatible TMs associated.
//-----------------------------------------------------------------------------

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "RecoTracker/NuclearSeedGenerator/interface/NuclearTester.h"
#include "RecoTracker/NuclearSeedGenerator/interface/SeedFromNuclearInteraction.h"
#include "RecoTracker/NuclearSeedGenerator/interface/TangentHelix.h"

#include "TrackingTools/DetLayers/interface/NavigationSchool.h"

class NuclearInteractionFinder {
private:
  typedef TrajectoryStateOnSurface TSOS;
  typedef FreeTrajectoryState FTS;
  typedef TrajectoryMeasurement TM;
  typedef std::vector<Trajectory> TrajectoryContainer;
  typedef TrajectoryMeasurement::ConstRecHitPointer ConstRecHitPointer;

  /// get the seeds at the interaction point
  void fillSeeds(const std::pair<TrajectoryMeasurement, std::vector<TrajectoryMeasurement> >& tmPairs);

  /// Find compatible TM of a TM with error rescaled by rescaleFactor
  std::vector<TrajectoryMeasurement> findCompatibleMeasurements(const TM& lastMeas,
                                                                double rescaleFactor,
                                                                const LayerMeasurements& layerMeasurements) const;

  std::vector<TrajectoryMeasurement> findMeasurementsFromTSOS(const TSOS& currentState,
                                                              DetId detid,
                                                              const LayerMeasurements& layerMeasurements) const;

  /// Calculate the parameters of the circle representing the primary track at the interaction point
  void definePrimaryHelix(std::vector<TrajectoryMeasurement>::const_iterator it_meas);

public:
  NuclearInteractionFinder() {}

  NuclearInteractionFinder(const edm::EventSetup& es, const edm::ParameterSet& iConfig);

  virtual ~NuclearInteractionFinder();

  /// Run the Finder
  bool run(const Trajectory& traj, const MeasurementTrackerEvent& event);

  /// Improve the seeds with a third RecHit
  void improveSeeds(const MeasurementTrackerEvent& event);

  /// Fill 'output' with persistent nuclear seeds
  std::unique_ptr<TrajectorySeedCollection> getPersistentSeeds();

  TrajectoryStateOnSurface rescaleError(float rescale, const TSOS& state) const;

  const NavigationSchool* nav() const { return theNavigationSchool; }

private:
  const Propagator* thePropagator;
  const MeasurementEstimator* theEstimator;
  const MeasurementTracker* theMeasurementTracker;
  const GeometricSearchTracker* theGeomSearchTracker;
  const NavigationSchool* theNavigationSchool;
  edm::ESHandle<MagneticField> theMagField;

  NuclearTester* nuclTester;
  SeedFromNuclearInteraction* currentSeed;
  std::vector<SeedFromNuclearInteraction> allSeeds;
  TangentHelix* thePrimaryHelix;

  // parameters
  double ptMin;
  unsigned int maxHits;
  double rescaleErrorFactor;
  bool checkCompletedTrack; /**< If set to true check all the tracks, even those reaching the edge of the tracker */
  std::string navigationSchoolName;
};
#endif
