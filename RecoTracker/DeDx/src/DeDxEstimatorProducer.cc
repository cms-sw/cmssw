// -*- C++ -*-
//
// Package:    DeDxEstimatorProducer
// Class:      DeDxEstimatorProducer
// 
/**\class DeDxEstimatorProducer DeDxEstimatorProducer.cc RecoTracker/DeDxEstimatorProducer/src/DeDxEstimatorProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  andrea
//         Created:  Thu May 31 14:09:02 CEST 2007
//    Code Updates:  loic Quertenmont (querten)
//         Created:  Thu May 10 14:09:02 CEST 2008
// $Id: DeDxEstimatorProducer.cc,v 1.9 2008/05/10 18:57:23 querten Exp $
//
//


// system include files
#include <memory>

#include "RecoTracker/DeDx/interface/DeDxEstimatorProducer.h"
#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrajectorySateOnDetInfo.h"
#include "DataFormats/TrackReco/interface/TrackTrajectorySateOnDetInfos.h"

#include "RecoTracker/DeDx/interface/GenericAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/TruncatedAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/MedianDeDxEstimator.h"



#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"



using namespace reco;
using namespace std;
using namespace edm;

DeDxEstimatorProducer::DeDxEstimatorProducer(const edm::ParameterSet& iConfig)
{
   produces<TrackDeDxEstimateCollection>();

   string estimatorName = iConfig.getParameter<string>("estimator");
   if(estimatorName == "median")     m_estimator = new MedianDeDxEstimator(-2.);
   if(estimatorName == "generic")    m_estimator = new GenericAverageDeDxEstimator  (iConfig.getParameter<double>("exponent"));
   if(estimatorName == "truncated")  m_estimator = new TruncatedAverageDeDxEstimator(iConfig.getParameter<double>("fraction"));

   m_FromTrajectory            = iConfig.getParameter<     bool    >("BuildFromTrajectory"); 
   if(!m_FromTrajectory){
      m_TsodiTag                  = iConfig.getParameter<edm::InputTag>("TrajectoryStateOnDetInfo");
   }else{
      m_TSODIProducer             = new TrajectorySateOnDetInfosProducer(iConfig);

   }
}


DeDxEstimatorProducer::~DeDxEstimatorProducer()
{
  delete m_estimator;
}


void
DeDxEstimatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );

   if(!m_FromTrajectory){
      produce_from_tsodi      (iEvent, iSetup, tkGeom);
   }else{
      produce_from_trajectory (iEvent, iSetup, tkGeom);
   }

}

void
DeDxEstimatorProducer::produce_from_tsodi(edm::Event& iEvent, const edm::EventSetup& iSetup, edm::ESHandle<TrackerGeometry> tkGeom)
{
   edm::Handle<reco::TrackTrajectorySateOnDetInfosCollection> trackTrajectorySateOnDetInfosCollectionHandle;
   iEvent.getByLabel(m_TsodiTag,trackTrajectorySateOnDetInfosCollectionHandle);
   const reco::TrackTrajectorySateOnDetInfosCollection* tsodis = trackTrajectorySateOnDetInfosCollectionHandle.product();

   TrackDeDxEstimateCollection * outputCollection = new TrackDeDxEstimateCollection(tsodis->keyProduct());

   reco::TrackTrajectorySateOnDetInfosCollection::const_iterator tsodis_it= tsodis->begin();
   for(int j=0;tsodis_it!=tsodis->end();++tsodis_it,j++)
   {
      TrajectorySateOnDetInfoCollection tmp = (*tsodis_it).second;
      float val=m_estimator->dedx( (*tsodis_it).second, tkGeom );
      outputCollection->setValue(j, val);
   }

   std::auto_ptr<TrackDeDxEstimateCollection> estimator(outputCollection);
   iEvent.put(estimator);
}

void
DeDxEstimatorProducer::produce_from_trajectory(edm::Event& iEvent, const edm::EventSetup& iSetup, edm::ESHandle<TrackerGeometry> tkGeom)
{
   TrackTrajectorySateOnDetInfosCollection* tsodis = m_TSODIProducer->Get_TSODICollection(iEvent, iSetup);
   TrackDeDxEstimateCollection * outputCollection = new TrackDeDxEstimateCollection(tsodis->keyProduct());

   reco::TrackTrajectorySateOnDetInfosCollection::const_iterator tsodis_it= tsodis->begin();
   for(int j=0;tsodis_it!=tsodis->end();++tsodis_it,j++)
   {
      TrajectorySateOnDetInfoCollection tmp = (*tsodis_it).second;
      float val=m_estimator->dedx( (*tsodis_it).second, tkGeom );
      outputCollection->setValue(j, val);
   }

   std::auto_ptr<TrackDeDxEstimateCollection> estimator(outputCollection);
   iEvent.put(estimator);
}


// ------------ method called once each job just before starting event loop  ------------
void 
DeDxEstimatorProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DeDxEstimatorProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeDxEstimatorProducer);
