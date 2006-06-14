#ifndef CosmicMuonTrajectoryBuilder_H
#define CosmicMuonTrajectoryBuilder_H
/** \file CosmicMuonTrajectoryBuilder
 *
 *  $Date: $
 *  $Revision: $
 *  \author Chang Liu  -  Purdue University
 */


#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackingTools/PatternTools/interface/MeasurementEstimator.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"


namespace edm {class ParameterSet; class Event; class EventSetup;}

class Trajectory;
class TrajectoryMeasurement;
class MuonBestMeasurementFinder;
class MagneticField;
//class CosmicSeedGenerator;
class CosmicNavigation;
class GenericTransientTrackingRecHit;

class CosmicMuonTrajectoryBuilder {
public:

  CosmicMuonTrajectoryBuilder(const MagneticField *);
  virtual ~CosmicMuonTrajectoryBuilder();


  std::vector<Trajectory> trajectories(const edm::Event&, const edm::EventSetup&) const;


  const Propagator& propagator() const {return *thePropagator;}
  MeasurementEstimator* estimator() const {return theEstimator;}
  MuonBestMeasurementFinder* measFinder() const {return theBestMeasurementFinder;}
  MuonTrajectoryUpdator* updator() const {return theUpdator;}
//  const CosmicNavigation& navigation() const {return *theNavigation;}
  double maxChi2() const {return theMaxChi2 ;}

private:

  double theMaxChi2;
  Propagator* thePropagator;
  MeasurementEstimator* theEstimator;
  MuonBestMeasurementFinder *theBestMeasurementFinder;
  MuonTrajectoryUpdator* theUpdator;
//  CosmicNavigation* theNavigation; 
  const MagneticField* theField;
//  const CosmicSeedGenerator* theSeedGenerator;
  double theMaxEta; 
  std::string theSeedCollectionLabel;
//  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  
};
#endif
