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
#include "DataFormats/TrackReco/interface/TrajectoryStateOnDetInfo.h"
#include "DataFormats/TrackReco/interface/TrackTrajectoryStateOnDetInfos.h"

#include "RecoTracker/DeDx/interface/GenericAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/TruncatedAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/MedianDeDxEstimator.h"



#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoTracker/DeDx/interface/TrajectoryStateOnDetInfosTools.h"

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
      m_trajTrackAssociationTag   = iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation");
      m_tracksTag                 = iConfig.getParameter<edm::InputTag>("Track");
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

   TrackTrajectoryStateOnDetInfosCollection* tsodis;
   if(!m_FromTrajectory){
      edm::Handle<reco::TrackTrajectoryStateOnDetInfosCollection> trackTrajectoryStateOnDetInfosCollectionHandle;
      iEvent.getByLabel(m_TsodiTag,trackTrajectoryStateOnDetInfosCollectionHandle);
      tsodis = (TrackTrajectoryStateOnDetInfosCollection*) trackTrajectoryStateOnDetInfosCollectionHandle.product();
   }else{
      Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
      iEvent.getByLabel(m_trajTrackAssociationTag, trajTrackAssociationHandle);
      const TrajTrackAssociationCollection TrajToTrackMap = *trajTrackAssociationHandle.product();

      edm::Handle<reco::TrackCollection> trackCollectionHandle;
      iEvent.getByLabel(m_tracksTag,trackCollectionHandle);

      tsodis = new TrackTrajectoryStateOnDetInfosCollection(reco::TrackRefProd(trackCollectionHandle) );
      TSODI::Fill_TSODICollection(TrajToTrackMap, tsodis);

//      tsodis = TSODI::Fill_TSODICollection(TrajToTrackMap,trackCollectionHandle);
   }

   TrackDeDxEstimateCollection* outputCollection = new TrackDeDxEstimateCollection(tsodis->keyProduct());

   reco::TrackTrajectoryStateOnDetInfosCollection::const_iterator tsodis_it= tsodis->begin();
   for(int j=0;tsodis_it!=tsodis->end();++tsodis_it,j++)
   {
      TrajectoryStateOnDetInfoCollection TsodiColl = (*tsodis_it).second;
      Measurement1D val=m_estimator->dedx( GetMeasurements( TsodiColl, tkGeom)  );
      outputCollection->setValue(j, val);
   }

   std::auto_ptr<TrackDeDxEstimateCollection> estimator(outputCollection);
   iEvent.put(estimator);

   if(m_FromTrajectory){
      delete tsodis;
   }
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

std::vector<Measurement1D>
DeDxEstimatorProducer::GetMeasurements(TrajectoryStateOnDetInfoCollection TsodiColl, edm::ESHandle<TrackerGeometry> tkGeom){
   std::vector<Measurement1D> to_return;

   for(unsigned int i=0;i<TsodiColl.size();i++){
      float ChargeN = TSODI::chargeOverPath(&TsodiColl[i], tkGeom);
      if(ChargeN>=0) to_return.push_back( Measurement1D(ChargeN, 0) );
   }
   return to_return; 
}



//define this as a plug-in
DEFINE_FWK_MODULE(DeDxEstimatorProducer);
