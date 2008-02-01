#ifndef RecoMuon_TrackerSeedGenerator_TSGFromPropagation_H
#define RecoMuon_TrackerSeedGenerator_TSGFromPropagation_H

/** \class TSGFromPropagation
 *  Tracker Seed Generator by propagating and updating a standAlone muon
 *  to the first 2 (or 1) rechits it meets in tracker system 
 *
 *  $Date: 2007/11/30 15:48:55 $
 *  $Revision: 1.10 $
 *  \author Chang Liu - Purdue University 
 */

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoMuon/TrackingTools/interface/MuonUpdatorAtVertex.h"
#include "RecoMuon/GlobalTrackingTools/interface/DirectTrackerNavigation.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

class TSGFromPropagation : public TrackerSeedGenerator {

public:
  TSGFromPropagation(const edm::ParameterSet &pset);

  TSGFromPropagation(const edm::ParameterSet& par, const MuonServiceProxy*);

  virtual ~TSGFromPropagation();

  void  trackerSeeds(const TrackCand&, const TrackingRegion&, std::vector<TrajectorySeed>&);
    
  void init(const MuonServiceProxy*);

  void setEvent(const edm::Event&);


private:

  TrajectoryStateOnSurface innerState(const TrackCand&) const;

  const LayerMeasurements* tkLayerMeasurements() const { return theTkLayerMeasurements; } 

  const Chi2MeasurementEstimator* estimator() const { return theEstimator; }

  edm::ESHandle<Propagator> propagator() const {return theService->propagator(thePropagatorName); }

  TrajectorySeed createSeed(const TrajectoryMeasurement&) const;

  void selectMeasurements(std::vector<TrajectoryMeasurement>&) const;

  void validMeasurements(std::vector<TrajectoryMeasurement>&) const;

  std::vector<TrajectoryMeasurement> findMeasurements(const DetLayer*, const TrajectoryStateOnSurface&) const;

  void findSecondMeasurements(std::vector<TrajectoryMeasurement>&, const std::vector<const DetLayer*>& ) const;

  struct IncreasingEstimate{
    bool operator()(const TrajectoryMeasurement& lhs,
		    const TrajectoryMeasurement& rhs) const{ 
    return lhs.estimate() < rhs.estimate();
    }
  };

  unsigned long long theCacheId_MT;

  const LayerMeasurements*  theTkLayerMeasurements;

  edm::ESHandle<GeometricSearchTracker> theTracker;

  edm::ESHandle<MeasurementTracker> theMeasTracker;

  const DirectTrackerNavigation* theNavigation;

  const MuonServiceProxy* theService;

  const Chi2MeasurementEstimator* theEstimator;

  edm::ParameterSet theConfig;

  double theMaxChi2;

  double theErrorRescaling;

  bool theUseSecondMeasurementsFlag;

  std::string thePropagatorName;

};

#endif 
