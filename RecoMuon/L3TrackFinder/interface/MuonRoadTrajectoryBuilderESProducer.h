#ifndef RecoMuon_L3TrackFinder_MuonRoadTrajectoryBuilderESProducer_h
#define RecoMuon_L3TrackFinder_MuonRoadTrajectoryBuilderESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"

#include <RecoMuon/L3TrackFinder/interface/MuonRoadTrajectoryBuilder.h> 
#include <RecoTracker/Record/interface/CkfComponentsRecord.h>

class MuonRoadTrajectoryBuilderESProducer : public edm::ESProducer {
   public:
      MuonRoadTrajectoryBuilderESProducer(const edm::ParameterSet&);
      ~MuonRoadTrajectoryBuilderESProducer();

  
  boost::shared_ptr<TrajectoryBuilder> produce(const CkfComponentsRecord&);
   private:
  boost::shared_ptr<TrajectoryBuilder> _trajectorybuilder;
  edm::ParameterSet pset_;
  std::string measurementTrackerName;
  std::string propagatorName;
};

#endif
