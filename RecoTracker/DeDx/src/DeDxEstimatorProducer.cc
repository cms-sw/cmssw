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
// $Id: DeDxEstimatorProducer.cc,v 1.20 2009/02/04 15:42:10 querten Exp $
//
//


// system include files
#include <memory>
#include "DataFormats/Common/interface/ValueMap.h"

#include "RecoTracker/DeDx/interface/DeDxEstimatorProducer.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "RecoTracker/DeDx/interface/GenericAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/TruncatedAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/MedianDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/UnbinnedFitDeDxEstimator.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

using namespace reco;
using namespace std;
using namespace edm;

DeDxEstimatorProducer::DeDxEstimatorProducer(const edm::ParameterSet& iConfig)
{

   produces<ValueMap<DeDxData> >();


   string estimatorName = iConfig.getParameter<string>("estimator");
   if(estimatorName == "median")      m_estimator = new MedianDeDxEstimator(-2.);
   if(estimatorName == "generic")     m_estimator = new GenericAverageDeDxEstimator  (iConfig.getParameter<double>("exponent"));
   if(estimatorName == "truncated")   m_estimator = new TruncatedAverageDeDxEstimator(iConfig.getParameter<double>("fraction"));
   if(estimatorName == "unbinnedFit") m_estimator = new UnbinnedFitDeDxEstimator();

   MaxNrStrips         = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"        ,  255);
   MinTrackHits        = iConfig.getUntrackedParameter<unsigned>("MinTrackHits"       ,  4);

   m_tracksTag = iConfig.getParameter<edm::InputTag>("tracks");
   m_trajTrackAssociationTag   = iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation");

   usePixel = iConfig.getParameter<bool>("UsePixel"); 
   useStrip = iConfig.getParameter<bool>("UseStrip");
   MeVperADCPixel = iConfig.getParameter<double>("MeVperADCPixel"); 
   MeVperADCStrip = iConfig.getParameter<double>("MeVperADCStrip"); 

   if(!usePixel && !useStrip)
   edm::LogWarning("DeDxHitsProducer") << "Pixel Hits AND Strip Hits will not be used to estimate dEdx --> BUG, Please Update the config file";
}


DeDxEstimatorProducer::~DeDxEstimatorProducer()
{
  delete m_estimator;
}


void DeDxEstimatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  auto_ptr<ValueMap<DeDxData> > trackDeDxEstimateAssociation(new ValueMap<DeDxData> );  
  ValueMap<DeDxData>::Filler filler(*trackDeDxEstimateAssociation);

  edm::ESHandle<TrackerGeometry> tkGeom;
  iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
  m_tracker=&(* tkGeom );
  
  Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  iEvent.getByLabel(m_trajTrackAssociationTag, trajTrackAssociationHandle);
  const TrajTrackAssociationCollection TrajToTrackMap = *trajTrackAssociationHandle.product();

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByLabel(m_tracksTag,trackCollectionHandle);

  size_t n =  TrajToTrackMap.size();
  std::vector<DeDxData> dedxEstimate(n);

  //assume trajectory collection size is equal to track collection size and that order is kept
  int j=0;
  for(TrajTrackAssociationCollection::const_iterator cit=TrajToTrackMap.begin(); cit!=TrajToTrackMap.end(); cit++,j++){
     
     const edm::Ref<std::vector<Trajectory> > traj = cit->key;
     const reco::TrackRef track = cit->val;

     DeDxHitCollection dedxHits;
     vector<DeDxTools::RawHits> hits; 
     DeDxTools::trajectoryRawHits(traj, hits, usePixel, useStrip);
  
     for(size_t i=0; i < hits.size(); i++)
     {
	 float pathLen=thickness(hits[i].detId)/std::abs(hits[i].angleCosine);
	 float charge=normalization(hits[i].detId)*hits[i].charge*std::abs(hits[i].angleCosine); 
	 dedxHits.push_back( DeDxHit( charge, distance(hits[i].detId), pathLen, hits[i].detId) );
     }
  
     sort(dedxHits.begin(),dedxHits.end(),less<DeDxHit>());   
     std::pair<float,float> val_and_error = m_estimator->dedx(dedxHits);

     dedxEstimate[j] = DeDxData(val_and_error.first, val_and_error.second, dedxHits.size() );
  }
  filler.insert(trackCollectionHandle, dedxEstimate.begin(), dedxEstimate.end());

  // really fill the association map
  filler.fill();
   // put into the event 
  iEvent.put(trackDeDxEstimateAssociation);   
}

// ------------ method called once each job just before starting event loop  ------------
void  DeDxEstimatorProducer::beginRun(edm::Run & run, const edm::EventSetup&){}

// ------------ method called once each job just after ending the event loop  ------------
void  DeDxEstimatorProducer::endJob() {}


double DeDxEstimatorProducer::thickness(DetId id)
{
 map<DetId,double>::iterator th=m_thicknessMap.find(id);
 if(th!=m_thicknessMap.end())
   return (*th).second;
 else {
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
  else {
     double detNormalization=1./thickness(id);
  
     //compute other normalization
     const GeomDetUnit* it = m_tracker->idToDetUnit(DetId(id));
     bool isPixel = dynamic_cast<const PixelGeomDetUnit*>(it)!=0;
     bool isStrip = dynamic_cast<const StripGeomDetUnit*>(it)!=0;

     //FIXME: include gain et al calib
//     if(isPixel) detNormalization*=3.61e-06;
//     if(isStrip) detNormalization*=3.61e-06*250;

     if(isPixel) detNormalization*=MeVperADCPixel;
     if(isStrip) detNormalization*=MeVperADCStrip;

     m_normalizationMap[id]=detNormalization;//computed value
     return detNormalization;
   } 
}


double DeDxEstimatorProducer::distance(DetId id)
{
 map<DetId,double>::iterator dist=m_distanceMap.find(id);
 if(dist!=m_distanceMap.end()) 
   return (*dist).second;
 else
 {
  const GeomDetUnit* it = m_tracker->idToDetUnit(DetId(id));
   float   d=it->position().mag();
   m_distanceMap[id]=d;
   return d;
 }
 
}





//define this as a plug-in
DEFINE_FWK_MODULE(DeDxEstimatorProducer);
