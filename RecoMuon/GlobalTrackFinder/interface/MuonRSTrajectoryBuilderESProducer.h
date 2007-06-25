#ifndef MuonRSTrajectoryBuilderESProducer_h
#define MuonRSTrajectoryBuilderESProducer_h

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"


#include <RecoMuon/GlobalTrackFinder/interface/MuonRSTrajectoryBuilder.h> 
#include <RecoTracker/Record/interface/CkfComponentsRecord.h>
#include <RecoTracker/MeasurementDet/interface/MeasurementTracker.h>
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

//
// class decleration
//

class MuonRSTrajectoryBuilderESProducer : public edm::ESProducer {
   public:
      MuonRSTrajectoryBuilderESProducer(const edm::ParameterSet&);
      ~MuonRSTrajectoryBuilderESProducer();

  
  boost::shared_ptr<TrackerTrajectoryBuilder> produce(const CkfComponentsRecord&);
   private:
  boost::shared_ptr<TrackerTrajectoryBuilder> _trajectorybuilder;
  edm::ParameterSet pset_;
  std::string measurementTrackerName;
  std::string propagatorName;
};

#endif
