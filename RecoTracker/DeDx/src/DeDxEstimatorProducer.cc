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
// $Id: DeDxEstimatorProducer.cc,v 1.15 2008/05/15 16:54:22 querten Exp $
//
//


// system include files
#include <memory>
#include "DataFormats/Common/interface/ValueMap.h"

#include "RecoTracker/DeDx/interface/DeDxEstimatorProducer.h"
#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
//#include "DataFormats/TrackReco/interface/TrajectoryStateOnDetInfo.h"
//#include "DataFormats/TrackReco/interface/TrackTrajectoryStateOnDetInfos.h"

#include "RecoTracker/DeDx/interface/GenericAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/TruncatedAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/MedianDeDxEstimator.h"



#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//#include "RecoTracker/DeDx/interface/TrajectoryStateOnDetInfosTools.h"

using namespace reco;
using namespace std;
using namespace edm;

DeDxEstimatorProducer::DeDxEstimatorProducer(const edm::ParameterSet& iConfig)
{

   produces<ValueMap<TrackDeDxEstimate> >();

   string estimatorName = iConfig.getParameter<string>("estimator");
   if(estimatorName == "median")     m_estimator = new MedianDeDxEstimator(-2.);
   if(estimatorName == "generic")    m_estimator = new GenericAverageDeDxEstimator  (iConfig.getParameter<double>("exponent"));
   if(estimatorName == "truncated")  m_estimator = new TruncatedAverageDeDxEstimator(iConfig.getParameter<double>("fraction"));


   m_tracksTag = iConfig.getParameter<edm::InputTag>("tracks");
   m_refittedTracksTag = iConfig.getParameter<edm::InputTag>("refittedTracks");
//   m_trajectoriesTag   = iConfig.getParameter<edm::InputTag>("trajectories");
   m_trajTrackAssociationTag   = iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation");

   usePixel = iConfig.getParameter<bool>("UsePixel"); 
   useStrip = iConfig.getParameter<bool>("UseStrip");
   MeVperADCPixel = iConfig.getParameter<double>("MeVperADCPixel"); 
   MeVperADCStrip = iConfig.getParameter<double>("MeVperADCStrip"); 


//    m_FromTrajectory            = iConfig.getParameter<     bool    >("BuildFromTrajectory"); 
//    if(!m_FromTrajectory){
//       m_TsodiTag                  = iConfig.getParameter<edm::InputTag>("TrajectoryStateOnDetInfo");
//    }else{
//       m_trajTrackAssociationTag   = iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation");
//       m_tracksTag                 = iConfig.getParameter<edm::InputTag>("Track");
//    }


}


DeDxEstimatorProducer::~DeDxEstimatorProducer()
{
  delete m_estimator;
}


void DeDxEstimatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  auto_ptr<ValueMap<TrackDeDxEstimate> > trackDeDxEstimateAssociation(new ValueMap<TrackDeDxEstimate> );  
  ValueMap<TrackDeDxEstimate>::Filler filler(*trackDeDxEstimateAssociation);


  edm::ESHandle<TrackerGeometry> tkGeom;
  iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
  
  //   TrackTrajectoryStateOnDetInfosCollection* tsodis;
  //   if(!m_FromTrajectory){
  
  //      edm::Handle<reco::TrackTrajectoryStateOnDetInfosCollection> trackTrajectoryStateOnDetInfosCollectionHandle;
  //   iEvent.getByLabel(m_TsodiTag,trackTrajectoryStateOnDetInfosCollectionHandle);
  //   tsodis = (TrackTrajectoryStateOnDetInfosCollection*) trackTrajectoryStateOnDetInfosCollectionHandle.product();
  //  }else{
  
  Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  iEvent.getByLabel(m_trajTrackAssociationTag, trajTrackAssociationHandle);
  const TrajTrackAssociationCollection TrajToTrackMap = *trajTrackAssociationHandle.product();

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByLabel(m_tracksTag,trackCollectionHandle);



   //      tsodis = new TrackTrajectoryStateOnDetInfosCollection(reco::TrackRefProd(trackCollectionHandle) );
   //    TSODI::Fill_TSODICollection(TrajToTrackMap, tsodis);
   //      tsodis = TSODI::Fill_TSODICollection(TrajToTrackMap,trackCollectionHandle);
   //}

  //reverse the track-trajectory map
//    std::map<const reco::Track *,const Trajectory *> trackToTrajectoryMap;
//    for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap->begin(); it!=TrajToTrackMap->end(); ++it)
//    {
//     trackToTrajectoryMap[&(*it->val)]=&(*it->key);
//    }


  size_t n =  TrajToTrackMap->size();
  std::vector<TrackDeDxEstimate> dedxEstimate(n);



  //assume trajectory collection size is equal to track collection size and that order is kept
  int j=0;
  for(TrajTrackAssociationCollection::const_iterator cit=TrajToTrackMap.begin(); cit=TrajToTrackMap.end(); cit++){
     
    const edm::Ref<std::vector<Trajectory> > traj = cit->key;
    const reco::TrackRef track = cit->val;
    

     //     DeDxHitCollection dedxHits; // the output hits for this track

     //     const Trajectory * trajectory= cit-> ...;
     //     if(trajectory) 
     //     {  

     DeDxHitCollection dedxHits;
     vector<DeDxTools::RawHits> hits; 
     DeDxTools::trajectoryRawHits(traj, hits);
  
     for(size_t i=0; i < hits.size(); i++)
       {
	 float pathLen=thickness(hits[i].detId)/std::abs(hits[i].angleCosine);
	 float charge=normalize(hits[i].detId,hits[i].charge*std::abs(hits[i].angleCosine)); 
	 dedxHits.push_back( DeDxHit( charge, distance(hits[i].detId), pathLen, hits[i].detId) );
       }
  
     sort(dedxHits.begin(),dedxHits.end(),less<DeDxHit>());
   
     float val=m_estimator->dedx(*it);



     dedxEstimate[j] = TrackDeDxEstimate();


  }

  filler.insert(trackCollectionHandle, dedxEstimate.begin(), dedxEstimate.end());

  // really fill the association map
  filler.fill();
   // put into the event 
  evt.put(trackDeDxEstimateAssociation);


   
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
void  DeDxEstimatorProducer::beginJob(const edm::EventSetup&){}

// ------------ method called once each job just after ending the event loop  ------------
void  DeDxEstimatorProducer::endJob() {
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


double DeDxEstimatorProducer::thickness(DetId id)
{
 map<DetId,double>::iterator th=m_thicknessMap.find(id);
 if(th!=m_thicknessMap.end())
   return (*th).second;
 else
 {
  double detThickness=1.;
  //compute thickness normalization
  const GeomDetUnit* it = m_tracker->idToDetUnit(DetId(id));
  bool isPixel = dynamic_cast<const PixelGeomDetUnit*>(it)!=0;
  bool isStrip = dynamic_cast<const StripGeomDetUnit*>(it)!=0;
  if (!isPixel && ! isStrip) {
  //FIXME throw exception
edm::LogWarning("DeDxHitsProducer") << "\t\t this detID doesn't seem to belong to the Tracker";
  detThickness = 1.;
  }else{
    detThickness = it->surface().bounds().thickness();
  }

   m_thicknessMap[id]=detThickness;//computed value
   return detThickness;
 }

}


double DeDxEstimatorProducer::normalization(DetId id)
{
 map<DetId,double>::iterator norm=m_normalizationMap.find(id);
 if(norm!=m_normalizationMap.end()) 
   return (*norm).second;
 else
 {
  double detNormalization=1./thickness(id);
  
//compute other normalization
  const GeomDetUnit* it = m_tracker->idToDetUnit(DetId(id));
  bool isPixel = dynamic_cast<const PixelGeomDetUnit*>(it)!=0;
  bool isStrip = dynamic_cast<const StripGeomDetUnit*>(it)!=0;

  //FIXME: include gain et al calib
   if(isPixel) detNormalization*=3.61e-06;
   if(isStrip) detNormalization*=3.61e-06*250;
 
   m_normalizationMap[id]=detNormalization;//computed value
   return detNormalization;
 }
 
}




//define this as a plug-in
DEFINE_FWK_MODULE(DeDxEstimatorProducer);
