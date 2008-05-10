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
// $Id: DeDxEstimatorProducer.cc,v 1.6 2007/06/18 13:49:40 arizzi Exp $
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
//
// constructors and destructor
//
DeDxEstimatorProducer::DeDxEstimatorProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   produces<TrackDeDxEstimateCollection>();
   m_TsodiTag = iConfig.getParameter<edm::InputTag>("TrajectoryStateOnDetInfo");

   //FIXME: configurable, use ES?
   string estimatorName = iConfig.getParameter<string>("estimator");
   if(estimatorName == "median")     m_estimator = new MedianDeDxEstimator(-2.);
   if(estimatorName == "generic")    m_estimator = new GenericAverageDeDxEstimator  (iConfig.getParameter<double>("exponent"));
   if(estimatorName == "truncated")  m_estimator = new TruncatedAverageDeDxEstimator(iConfig.getParameter<double>("fraction"));

}


DeDxEstimatorProducer::~DeDxEstimatorProducer()
{
 //FIXME: only if doen't come from ES
  delete m_estimator;
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DeDxEstimatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );

   edm::Handle<reco::TrackTrajectorySateOnDetInfosCollection> trackTrajectorySateOnDetInfosCollectionHandle;
   iEvent.getByLabel(m_TsodiTag,trackTrajectorySateOnDetInfosCollectionHandle);
   const reco::TrackTrajectorySateOnDetInfosCollection& tsodis = *trackTrajectorySateOnDetInfosCollectionHandle.product();

   TrackDeDxEstimateCollection * outputCollection = new TrackDeDxEstimateCollection(tsodis.keyProduct());

   reco::TrackTrajectorySateOnDetInfosCollection::const_iterator tsodis_it= tsodis.begin();
   for(int j=0;tsodis_it!=tsodis.end();++tsodis_it,j++)
   {
      TrajectorySateOnDetInfoCollection tmp = (*tsodis_it).second;
//      printf("Track %i contains %i hits\n",j,tmp.size());

      float val=m_estimator->dedx( (*tsodis_it).second, tkGeom );
   //   float val=0;
      outputCollection->setValue(j, val);      
   }   

   //put in the event the result
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
