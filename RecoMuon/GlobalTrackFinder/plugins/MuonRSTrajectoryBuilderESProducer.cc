// -*- C++ -*-
//
// Package:    MuonRSTrajectoryBuilderESProducer
// Class:      MuonRSTrajectoryBuilderESProducer
// 
/**\class MuonRSTrajectoryBuilderESProducer MuonRSTrajectoryBuilderESProducer.h RecoMuon/MuonRSTrajectoryBuilderESProducer/interface/MuonRSTrajectoryBuilderESProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Wed May 16 20:31:23 CEST 2007
// $Id$
//
//


#include "RecoMuon/GlobalTrackFinder/interface/MuonRSTrajectoryBuilderESProducer.h"

//
// constructors and destructor
//
MuonRSTrajectoryBuilderESProducer::MuonRSTrajectoryBuilderESProducer(const edm::ParameterSet& iConfig)
{
  std::string myName = iConfig.getParameter<std::string>("ComponentName");
  measurementTrackerName = iConfig.getParameter<std::string>("measurementTrackerName");
  propagatorName = iConfig.getParameter<std::string>("propagatorName");
  pset_ = iConfig;
  setWhatProduced(this,myName);
}


MuonRSTrajectoryBuilderESProducer::~MuonRSTrajectoryBuilderESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
boost::shared_ptr<TrackerTrajectoryBuilder>
MuonRSTrajectoryBuilderESProducer::produce(const CkfComponentsRecord& iRecord)
{
   using namespace edm::es;

   edm::ESHandle<MeasurementTracker>             measurementTrackerHandle;
   edm::ESHandle<Propagator> propagatorHandle;
   
   iRecord.get(measurementTrackerName,measurementTrackerHandle);
   iRecord.getRecord<TrackingComponentsRecord>().get(propagatorName,propagatorHandle);
   
   _trajectorybuilder = boost::shared_ptr<TrackerTrajectoryBuilder>(new  MuonRSTrajectoryBuilder(pset_,
												measurementTrackerHandle.product(),
												propagatorHandle->magneticField(),
												propagatorHandle.product())
								   );
   return _trajectorybuilder;
}

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(MuonRSTrajectoryBuilderESProducer);
