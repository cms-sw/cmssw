#ifndef RecoTracker_CkfPattern_GroupedCkfTrajectoryBuilderESProducer_h
#define RecoTracker_CkfPattern_GroupedCkfTrajectoryBuilderESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"
#include <boost/shared_ptr.hpp>

class  GroupedCkfTrajectoryBuilderESProducer: public edm::ESProducer{
 public:
  GroupedCkfTrajectoryBuilderESProducer(const edm::ParameterSet & p);
  virtual ~GroupedCkfTrajectoryBuilderESProducer(); 
  boost::shared_ptr<TrackerTrajectoryBuilder> produce(const CkfComponentsRecord &);
 private:
  boost::shared_ptr<TrackerTrajectoryBuilder> _trajectoryBuilder;
  edm::ParameterSet pset_;
};

#endif
