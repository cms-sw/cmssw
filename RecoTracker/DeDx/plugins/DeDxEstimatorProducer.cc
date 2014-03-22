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
//
//


// system include files
#include <memory>
#include "DataFormats/Common/interface/ValueMap.h"

#include "RecoTracker/DeDx/plugins/DeDxEstimatorProducer.h"
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

#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"


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

   m_tracksTag = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));
   m_trajTrackAssociationTag   = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation"));

   usePixel = iConfig.getParameter<bool>("UsePixel"); 
   useStrip = iConfig.getParameter<bool>("UseStrip");
   MeVperADCPixel = iConfig.getParameter<double>("MeVperADCPixel"); 
   MeVperADCStrip = iConfig.getParameter<double>("MeVperADCStrip"); 

   shapetest = iConfig.getParameter<bool>("ShapeTest");
   useCalibration = iConfig.getParameter<bool>("UseCalibration");
   m_calibrationPath = iConfig.getParameter<string>("calibrationPath");

   if(!usePixel && !useStrip)
   edm::LogWarning("DeDxHitsProducer") << "Pixel Hits AND Strip Hits will not be used to estimate dEdx --> BUG, Please Update the config file";
}


DeDxEstimatorProducer::~DeDxEstimatorProducer()
{
  delete m_estimator;
}

// ------------ method called once each job just before starting event loop  ------------
void  DeDxEstimatorProducer::beginRun(edm::Run const& run, const edm::EventSetup& iSetup)
{
   if(MODsColl.size()!=0)return;


   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );

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
       MOD->Gain            = 1;
       MODsColl[MOD->DetId] = MOD;
   }

   MakeCalibrationMap();
}



void DeDxEstimatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  auto_ptr<ValueMap<DeDxData> > trackDeDxEstimateAssociation(new ValueMap<DeDxData> );  
  ValueMap<DeDxData>::Filler filler(*trackDeDxEstimateAssociation);

  Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  iEvent.getByToken(m_trajTrackAssociationTag, trajTrackAssociationHandle);
  const TrajTrackAssociationCollection & TrajToTrackMap = *trajTrackAssociationHandle.product();

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByToken(m_tracksTag,trackCollectionHandle);

  size_t n =  TrajToTrackMap.size();
  std::vector<DeDxData> dedxEstimate(n);

  //assume trajectory collection size is equal to track collection size and that order is kept
  int j=0;
  for(TrajTrackAssociationCollection::const_iterator cit=TrajToTrackMap.begin(); cit!=TrajToTrackMap.end(); cit++,j++){
     
     const edm::Ref<std::vector<Trajectory> > traj = cit->key;
     const reco::TrackRef track = cit->val;

     DeDxHitCollection dedxHits;
     vector<DeDxTools::RawHits> hits; 
//     DeDxTools::trajectoryRawHits(traj, hits, usePixel, useStrip);

    const vector<TrajectoryMeasurement> & measurements = traj->measurements();
    for(vector<TrajectoryMeasurement>::const_iterator it = measurements.begin(); it!=measurements.end(); it++){
      TrajectoryStateOnSurface trajState=it->updatedState();
      if( !trajState.isValid()) continue;
     
      const TrackingRecHit * recHit=(*it->recHit()).hit();
      LocalVector trackDirection = trajState.localDirection();
      double cosine = trackDirection.z()/trackDirection.mag();
              
       if(const SiStripMatchedRecHit2D* matchedHit=dynamic_cast<const SiStripMatchedRecHit2D*>(recHit)){
	   if(!useStrip) continue;
	   DeDxTools::RawHits mono,stereo; 
	   mono.trajectoryMeasurement = &(*it);
	   stereo.trajectoryMeasurement = &(*it);
	   mono.angleCosine = cosine; 
	   stereo.angleCosine = cosine;

	   mono.charge = getCharge(DeDxTools::GetCluster(matchedHit->monoHit()),mono.NSaturating,matchedHit->monoId());
           stereo.charge = getCharge(DeDxTools::GetCluster(matchedHit->stereoHit()),stereo.NSaturating,matchedHit->stereoId());

	   mono.detId= matchedHit->monoId();
	   stereo.detId= matchedHit->stereoId();

           if(shapetest && !(DeDxTools::shapeSelection((DeDxTools::GetCluster(matchedHit->stereoHit()))->amplitudes()))) hits.push_back(stereo);
	   if(shapetest && !(DeDxTools::shapeSelection((DeDxTools::GetCluster(matchedHit->  monoHit()))->amplitudes()))) hits.push_back(mono);
        }else if(const ProjectedSiStripRecHit2D* projectedHit=dynamic_cast<const ProjectedSiStripRecHit2D*>(recHit)) {
           if(!useStrip) continue;
           DeDxTools::RawHits mono;

           mono.trajectoryMeasurement = &(*it);
           mono.angleCosine = cosine;
           mono.charge = getCharge(DeDxTools::GetCluster(projectedHit->originalHit()),mono.NSaturating,projectedHit->originalId());
           mono.detId= projectedHit->originalId();
	   if(shapetest && !(DeDxTools::shapeSelection((DeDxTools::GetCluster(projectedHit->originalHit()))->amplitudes()))) continue;
           hits.push_back(mono);
        }else if(const SiStripRecHit2D* singleHit=dynamic_cast<const SiStripRecHit2D*>(recHit)){
           if(!useStrip) continue;
           DeDxTools::RawHits mono;
	       
           mono.trajectoryMeasurement = &(*it);
           mono.angleCosine = cosine; 
           mono.charge = getCharge(DeDxTools::GetCluster(singleHit),mono.NSaturating,singleHit->geographicalId());
           mono.detId= singleHit->geographicalId();
	   if(shapetest && !(DeDxTools::shapeSelection((DeDxTools::GetCluster(singleHit))->amplitudes()))) continue;
           hits.push_back(mono); 
        }else if(const SiStripRecHit1D* single1DHit=dynamic_cast<const SiStripRecHit1D*>(recHit)){
           if(!useStrip) continue;
           DeDxTools::RawHits mono;
               
           mono.trajectoryMeasurement = &(*it);
           mono.angleCosine = cosine; 
           mono.charge = getCharge(DeDxTools::GetCluster(single1DHit),mono.NSaturating,single1DHit->geographicalId());
           mono.detId= single1DHit->geographicalId();
	   if(shapetest && !(DeDxTools::shapeSelection((DeDxTools::GetCluster(single1DHit))->amplitudes()))) continue;
           hits.push_back(mono); 
        }else if(const SiPixelRecHit* pixelHit=dynamic_cast<const SiPixelRecHit*>(recHit)){
           if(!usePixel) continue;

           DeDxTools::RawHits pixel;

           pixel.trajectoryMeasurement = &(*it);
           pixel.angleCosine = cosine; 
           pixel.charge = pixelHit->cluster()->charge();
           pixel.NSaturating=-1;
           pixel.detId= pixelHit->geographicalId();
           hits.push_back(pixel);
       } 
    }
     ///////////////////////////////////////

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



void DeDxEstimatorProducer::MakeCalibrationMap()
{
   if(!useCalibration)return;

   TChain* t1 = new TChain("SiStripCalib/APVGain");
   t1->Add(m_calibrationPath.c_str());

   unsigned int  tree_DetId;
   unsigned char tree_APVId;
   double        tree_Gain;

   t1->SetBranchAddress("DetId"             ,&tree_DetId      );
   t1->SetBranchAddress("APVId"             ,&tree_APVId      );
   t1->SetBranchAddress("Gain"              ,&tree_Gain       );



   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
       t1->GetEntry(ientry);
       stModInfo* MOD  = MODsColl[tree_DetId];
       MOD->Gain = tree_Gain;
   }

   delete t1;

}


int DeDxEstimatorProducer::getCharge(const SiStripCluster*   Cluster, int& Saturating_Strips,
				     const uint32_t & DetId)
{
   //const vector<uint8_t>&  Ampls       = Cluster->amplitudes();
   const vector<uint8_t>&  Ampls       = Cluster->amplitudes();
   //uint32_t                DetId       = Cluster->geographicalId();

//   float G=1.0f;

   int toReturn = 0;
   Saturating_Strips = 0;
   for(unsigned int i=0;i<Ampls.size();i++){
      int CalibratedCharge = Ampls[i];

      if(useCalibration){
         stModInfo* MOD = MODsColl[DetId];
//         G = MOD->Gain;
         CalibratedCharge = (int)(CalibratedCharge / MOD->Gain );
         if(CalibratedCharge>=1024){
            CalibratedCharge = 255;
         }else if(CalibratedCharge>=255){
            CalibratedCharge = 254;
         } 
      }

      toReturn+=CalibratedCharge;
      if(CalibratedCharge>=254)Saturating_Strips++;
   }

//   printf("Charge = %i --> %i  (Gain=%f)\n", accumulate(Ampls.begin(), Ampls.end(), 0), toReturn, G);         


   return toReturn;
}




//define this as a plug-in
DEFINE_FWK_MODULE(DeDxEstimatorProducer);
