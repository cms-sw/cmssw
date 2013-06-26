#ifndef RecoMuon_L3TrackFinder_MuonCkfTrajectoryBuilderESProducer_h
#define RecoMuon_L3TrackFinder_MuonCkfTrajectoryBuilderESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"

#include "RecoMuon/L3TrackFinder/interface/MuonCkfTrajectoryBuilder.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

class  MuonCkfTrajectoryBuilderESProducer: public edm::ESProducer{
 public:
  MuonCkfTrajectoryBuilderESProducer(const edm::ParameterSet & p);
  virtual ~MuonCkfTrajectoryBuilderESProducer(); 
  boost::shared_ptr<TrajectoryBuilder> produce(const CkfComponentsRecord &);
 private:
  boost::shared_ptr<TrajectoryBuilder> _trajectoryBuilder;
  edm::ParameterSet pset_;
};

#endif
