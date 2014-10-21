#ifndef RecoMuon_L3TrackFinder_MuonCkfTrajectoryBuilder_H
#define RecoMuon_L3TrackFinder_MuonCkfTrajectoryBuilder_H

#include "RecoTracker/CkfPattern/interface/CkfTrajectoryBuilder.h"
#include "FWCore/Framework/interface/ESWatcher.h"

class MuonCkfTrajectoryBuilder : public CkfTrajectoryBuilder {
 public:
  MuonCkfTrajectoryBuilder(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);
  virtual ~MuonCkfTrajectoryBuilder();

 protected:
  void setEvent_(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  void collectMeasurement(const DetLayer * layer, const std::vector<const DetLayer*>& nl,const TrajectoryStateOnSurface & currentState, std::vector<TM>& result,int& invalidHits,const Propagator *) const;

  virtual void findCompatibleMeasurements(const TrajectorySeed&seed, const TempTrajectory& traj, std::vector<TrajectoryMeasurement> & result) const;
  
  //and other fields
  bool theUseSeedLayer;
  double theRescaleErrorIfFail;
  const double theDeltaEta;
  const double theDeltaPhi;
  const std::string theProximityPropagatorName;
  const Propagator * theProximityPropagator;
  edm::ESWatcher<BaseCkfTrajectoryBuilder::Chi2MeasurementEstimatorRecord> theEstimatorWatcher;
  std::unique_ptr<Chi2MeasurementEstimatorBase> theEtaPhiEstimator;
  
};


#endif
