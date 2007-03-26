#ifndef RecoTracker_CkfPattern_CkfDebugTrajectoryBuilderESProducer_h
#define RecoTracker_CkfPattern_CkfDebugTrajectoryBuilderESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"
#include <boost/shared_ptr.hpp>

class  CkfDebugTrajectoryBuilderESProducer: public edm::ESProducer{
 public:
  CkfDebugTrajectoryBuilderESProducer(const edm::ParameterSet & p);
  virtual ~CkfDebugTrajectoryBuilderESProducer(); 
  boost::shared_ptr<TrackerTrajectoryBuilder> produce(const CkfComponentsRecord &);
 private:
  boost::shared_ptr<TrackerTrajectoryBuilder> _trajectoryBuilder;
  edm::ParameterSet pset_;
};

#endif
