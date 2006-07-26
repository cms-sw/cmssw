#ifndef CosmicMuonTrajectoryBuilder_H
#define CosmicMuonTrajectoryBuilder_H
/** \file CosmicMuonTrajectoryBuilder
 *
 *  $Date: 2006/07/21 03:13:19 $
 *  $Revision: 1.3 $
 *  \author Chang Liu  -  Purdue University
 */

#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"

#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
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
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"


namespace edm {class ParameterSet; class Event; class EventSetup;}

class Trajectory;
class TrajectoryMeasurement;
class MuonBestMeasurementFinder;
class MuonDetLayerMeasurements;

class CosmicMuonTrajectoryBuilder : public MuonTrajectoryBuilder{
public:

  CosmicMuonTrajectoryBuilder(const edm::ParameterSet&);
  virtual ~CosmicMuonTrajectoryBuilder();

  std::vector<Trajectory*> trajectories(const TrajectorySeed&);

   // fake implementation 
   // return a container reconstructed muons starting from a given track
  virtual CandidateContainer trajectories(const reco::TrackRef&) {
    return CandidateContainer();
  }

  virtual void setES(const edm::EventSetup&);

  virtual void setEvent(const edm::Event&);

  const Propagator* propagator() const {return thePropagator;}
  const MeasurementEstimator* estimator() const {return theEstimator;}
  MuonBestMeasurementFinder* measFinder() const {return theBestMeasurementFinder;}
  MuonTrajectoryUpdator* updator() const {return theUpdator;}
  double maxChi2() const {return theMaxChi2 ;}

private:

  double theMaxChi2;
  Propagator* thePropagator;
  MeasurementEstimator* theEstimator;
  MuonBestMeasurementFinder *theBestMeasurementFinder;
  MuonTrajectoryUpdator* theUpdator;
  MuonDetLayerMeasurements* theLayerMeasurements;
  std::string thePropagatorName;

  edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  edm::ESHandle<MagneticField> theField;
  edm::ESHandle<MuonDetLayerGeometry> theDetLayerGeometry;
  
};
#endif
