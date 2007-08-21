#ifndef RecoMuon_L3TrackFinder_MuonCkfTrajectoryBuilderESProducer_h
#define RecoMuon_L3TrackFinder_MuonCkfTrajectoryBuilderESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"
#include <boost/shared_ptr.hpp>

class  MuonCkfTrajectoryBuilderESProducer: public edm::ESProducer{
 public:
  MuonCkfTrajectoryBuilderESProducer(const edm::ParameterSet & p);
  virtual ~MuonCkfTrajectoryBuilderESProducer(); 
  boost::shared_ptr<TrackerTrajectoryBuilder> produce(const CkfComponentsRecord &);
 private:
  boost::shared_ptr<TrackerTrajectoryBuilder> _trajectoryBuilder;
  edm::ParameterSet pset_;
};

#endif
