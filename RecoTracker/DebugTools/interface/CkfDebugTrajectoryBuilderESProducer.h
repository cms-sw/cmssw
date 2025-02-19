#ifndef RecoTracker_CkfPattern_CkfDebugTrajectoryBuilderESProducer_h
#define RecoTracker_CkfPattern_CkfDebugTrajectoryBuilderESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "TrackingTools/PatternTools/interface/TrajectoryBuilder.h"
#include <boost/shared_ptr.hpp>

class  CkfDebugTrajectoryBuilderESProducer: public edm::ESProducer{
 public:
  CkfDebugTrajectoryBuilderESProducer(const edm::ParameterSet & p);
  virtual ~CkfDebugTrajectoryBuilderESProducer(); 
  boost::shared_ptr<TrajectoryBuilder> produce(const CkfComponentsRecord&);
 private:
  boost::shared_ptr<TrajectoryBuilder> _trajectoryBuilder;
  edm::ParameterSet pset_;
};

#endif
