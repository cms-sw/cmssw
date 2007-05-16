#ifndef CD_NuclearInteractionFinder_H_
#define CD_NuclearInteractionFinder_H_

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TrajectoryStateUpdator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"

#include "RecoTracker/NuclearSeedGenerator/interface/NuclearTester.h"

class NuclearInteractionFinder {
private:

  typedef TrajectoryStateOnSurface TSOS;
  typedef FreeTrajectoryState FTS;
  typedef TrajectoryMeasurement TM;
  typedef std::vector<Trajectory> TrajectoryContainer;
  typedef TrajectoryMeasurement::ConstRecHitPointer    ConstRecHitPointer;

public:

  NuclearInteractionFinder(){}
  NuclearInteractionFinder(const edm::EventSetup& es, const edm::ParameterSet& iConfig);
  virtual ~NuclearInteractionFinder();
  std::vector<std::pair< TM, std::vector<TM> > > run(const TrajectoryContainer& vTraj) const;
  std::vector<TrajectoryMeasurement>
         findCompatibleMeasurements( const TM& lastMeas, double rescaleFactor) const;
  std::vector<TrajectoryMeasurement>
         findMeasurementsFromTSOS(const TSOS& currentState, const TM& lastMeas) const;
  //TSOS stateWithLargeError(const TSOS& state, double min_pt,  int sign) const;
  void setEvent(const edm::Event& event) const;

private:

  const Propagator*               thePropagator;
  const MeasurementEstimator*     theEstimator;
  const MeasurementTracker*       theMeasurementTracker;
  const LayerMeasurements*        theLayerMeasurements;
  const GeometricSearchTracker*   theGeomSearchTracker;
  const NavigationSchool*         theNavigationSchool;
  edm::ESHandle<MagneticField>    theMagField;

  NuclearTester*  nuclTester;

  // parameters
  double ptMin;
  unsigned int maxPrimaryHits;
  double rescaleErrorFactor;
  bool checkCompletedTrack;

};
#endif
