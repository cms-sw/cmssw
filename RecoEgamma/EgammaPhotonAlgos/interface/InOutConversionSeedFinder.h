#ifndef RecoEGAMMA_ConversionSeed_InOutConversionSeedFinder_h
#define RecoEGAMMA_ConversionSeed_InOutConversionSeedFinder_h

/** \class InOutConversionSeedFinder
 **  
 **
 **  \author Nancy Marinelli, U. of Notre Dame, US
 **
 ***/
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionSeedFinder.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include <string>
#include <vector>

class MagneticField;
class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class TrajectoryMeasurement;

class InOutConversionSeedFinder : public ConversionSeedFinder {
private:
  typedef FreeTrajectoryState FTS;
  typedef TrajectoryStateOnSurface TSOS;

public:
  InOutConversionSeedFinder(const edm::ParameterSet& config, edm::ConsumesCollector&& iC);

  ~InOutConversionSeedFinder() override;

  void makeSeeds(const edm::Handle<edm::View<reco::CaloCluster> >& allBc) override;

  void setTracks(std::vector<Trajectory> const& in) { theOutInTracks_ = in; }

private:
  edm::ParameterSet conf_;
  virtual void fillClusterSeeds();
  void startSeed(const FreeTrajectoryState* fts,
                 const TrajectoryStateOnSurface& stateAtPreviousLayer,
                 int charge,
                 int layer);
  virtual void findSeeds(const TrajectoryStateOnSurface& startingState, float signedpt, unsigned int startingLayer);

  std::vector<const reco::CaloCluster*> getSecondCaloClusters(const GlobalPoint& conversionPosition,
                                                              float charge) const;
  void completeSeed(const TrajectoryMeasurement& m1,
                    const FreeTrajectoryState& fts,
                    const Propagator* propagator,
                    int ilayer);
  void createSeed(const TrajectoryMeasurement& m1, const TrajectoryMeasurement& m2);

private:
  float the2ndHitdphi_;
  float the2ndHitdzConst_;
  float the2ndHitdznSigma_;
  int track2Charge_;
  GlobalVector track2InitialMomentum_;
  int nSeedsPerInputTrack_;
  int maxNumberOfInOutSeedsPerInputTrack_;

  TrajectoryMeasurement* myPointer;

  std::vector<Trajectory> inputTracks_;
  std::vector<Trajectory> theOutInTracks_;
  std::vector<TrajectoryMeasurement> theFirstMeasurements_;

  reco::CaloCluster theSecondBC_;
  edm::Handle<edm::View<reco::CaloCluster> > bcCollection_;
};

#endif
