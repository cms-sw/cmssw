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
// $Id: DeDxEstimatorProducer.cc,v 1.24 2010/01/21 08:30:45 querten Exp $
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



// ------------ method called once each job just before starting event loop  ------------
void  DeDxEstimatorProducer::beginRun(edm::Run & run, const edm::EventSetup& iSetup)
{
   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
//   const TrackerGeometry* m_tracker = tkGeom.product();

   MODsColl.clear();
   vector<GeomDet*> Det = tkGeom->dets();
   for(unsigned int i=0;i<Det.size();i++){
      DetId  Detid  = Det[i]->geographicalId();

       StripGeomDetUnit* StripDetUnit = dynamic_cast<StripGeomDetUnit*> (Det[i]);
       PixelGeomDetUnit* PixelDetUnit = dynamic_cast<PixelGeomDetUnit*> (Det[i]);

       double Thick=-1, Dist=-1, Norma=-1;
       if(StripDetUnit){
          Dist      = StripDetUnit->position().mag();
          Thick     = StripDetUnit->surface().bounds().thickness();
          Norma     = MeVperADCStrip/Thick;
       }else if(PixelDetUnit){
          Dist      = PixelDetUnit->position().mag();
          Thick     = PixelDetUnit->surface().bounds().thickness();
          Norma     = MeVperADCPixel/Thick;
       }

       stModInfo* MOD       = new stModInfo;
       MOD->DetId           = Detid.rawId();
       MOD->Thickness       = Thick;
       MOD->Distance        = Dist;
       MOD->Normalization   = Norma;
       MODsColl[MOD->DetId] = MOD;
   }
}

// ------------ method called once each job just after ending the event loop  ------------
void  DeDxEstimatorProducer::endJob() {}




void DeDxEstimatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  auto_ptr<ValueMap<DeDxData> > trackDeDxEstimateAssociation(new ValueMap<DeDxData> );  
  ValueMap<DeDxData>::Filler filler(*trackDeDxEstimateAssociation);

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
 
     int NClusterSaturating = 0; 
     for(size_t i=0; i < hits.size(); i++)
     {
         stModInfo* MOD = MODsColl[hits[i].detId];
         float pathLen = MOD->Thickness/std::abs(hits[i].angleCosine);
         float charge  = MOD->Normalization*hits[i].charge*std::abs(hits[i].angleCosine);
         dedxHits.push_back( DeDxHit( charge, MOD->Distance, pathLen, hits[i].detId) );

         if(hits[i].NSaturating>0)NClusterSaturating++;
     }
  
     sort(dedxHits.begin(),dedxHits.end(),less<DeDxHit>());   
     std::pair<float,float> val_and_error = m_estimator->dedx(dedxHits);


     //WARNING: Since the dEdX Error is not properly computed for the moment
     //It was decided to store the number of saturating cluster in that dataformat
     val_and_error.second = NClusterSaturating; 

     dedxEstimate[j] = DeDxData(val_and_error.first, val_and_error.second, dedxHits.size() );
  }
  filler.insert(trackCollectionHandle, dedxEstimate.begin(), dedxEstimate.end());

  // really fill the association map
  filler.fill();
   // put into the event 
  iEvent.put(trackDeDxEstimateAssociation);   
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeDxEstimatorProducer);
