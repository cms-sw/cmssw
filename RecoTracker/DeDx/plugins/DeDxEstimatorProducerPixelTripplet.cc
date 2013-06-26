// -*- C++ -*-
//
// Package:    DeDxEstimatorProducerPixelTripplet
// Class:      DeDxEstimatorProducerPixelTripplet
// 
/**\class DeDxEstimatorProducerPixelTripplet DeDxEstimatorProducerPixelTripplet.cc RecoTracker/DeDxEstimatorProducerPixelTripplet/src/DeDxEstimatorProducerPixelTripplet.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  loic Quertenmont (querten)
//         Created:  Fri Nov 19 14:09:02 CEST 2010


// system include files
#include <memory>
#include "DataFormats/Common/interface/ValueMap.h"

#include "RecoTracker/DeDx/plugins/DeDxEstimatorProducerPixelTripplet.h"
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

DeDxEstimatorProducerPixelTripplet::DeDxEstimatorProducerPixelTripplet(const edm::ParameterSet& iConfig)
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

   shapetest = iConfig.getParameter<bool>("ShapeTest");
   useCalibration = iConfig.getParameter<bool>("UseCalibration");
   m_calibrationPath = iConfig.getParameter<string>("calibrationPath");

   if(!usePixel && !useStrip)
   edm::LogWarning("DeDxHitsProducer") << "Pixel Hits AND Strip Hits will not be used to estimate dEdx --> BUG, Please Update the config file";
}


DeDxEstimatorProducerPixelTripplet::~DeDxEstimatorProducerPixelTripplet()
{
  delete m_estimator;
}



// ------------ method called once each job just before starting event loop  ------------
void  DeDxEstimatorProducerPixelTripplet::beginRun(edm::Run const& run, const edm::EventSetup& iSetup)
{
   if(MODsColl.size()!=0)return;


   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );

   vector<GeomDet*> Det = tkGeom->dets();
   for(unsigned int i=0;i<Det.size();i++){
      DetId  Detid  = Det[i]->geographicalId();

       StripGeomDetUnit* StripDetUnit = dynamic_cast<StripGeomDetUnit*> (Det[i]);
       PixelGeomDetUnit* PixelDetUnit = dynamic_cast<PixelGeomDetUnit*> (Det[i]);

       stModInfo* MOD       = new stModInfo;
       double Thick=-1, Dist=-1, Norma=-1;
       if(StripDetUnit){
          Dist      = StripDetUnit->position().mag();
          Thick     = StripDetUnit->surface().bounds().thickness();
          Norma     = MeVperADCStrip/Thick;
          MOD->Normal = StripDetUnit->surface().normalVector();
       }else if(PixelDetUnit){
          Dist      = PixelDetUnit->position().mag();
          Thick     = PixelDetUnit->surface().bounds().thickness();
          Norma     = MeVperADCPixel/Thick;
          MOD->Normal = PixelDetUnit->surface().normalVector();
       }

       MOD->DetId           = Detid.rawId();
       MOD->Thickness       = Thick;
       MOD->Distance        = Dist;
       MOD->Normalization   = Norma;
       MOD->Gain            = 1;
       MODsColl[MOD->DetId] = MOD;
   }

   MakeCalibrationMap();
}

// ------------ method called once each job just after ending the event loop  ------------
void  DeDxEstimatorProducerPixelTripplet::endJob(){
   MODsColl.clear();
}




void DeDxEstimatorProducerPixelTripplet::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  auto_ptr<ValueMap<DeDxData> > trackDeDxEstimateAssociation(new ValueMap<DeDxData> );  
  ValueMap<DeDxData>::Filler filler(*trackDeDxEstimateAssociation);

  Handle<TrackCollection> trackCollHandle;
  iEvent.getByLabel(m_trajTrackAssociationTag, trackCollHandle);
  const TrackCollection trackColl = *trackCollHandle.product();

  size_t n =  trackColl.size();
  std::vector<DeDxData> dedxEstimate(n);

  //assume trajectory collection size is equal to track collection size and that order is kept
  for(unsigned int j=0;j<trackColl.size();j++){
     const reco::TrackRef track = reco::TrackRef( trackCollHandle.product(), j );

     DeDxHitCollection dedxHits;
     vector<DeDxTools::RawHits> hits; 
//     DeDxTools::trajectoryRawHits(traj, hits, usePixel, useStrip);


     int NClusterSaturating = 0;
     for(unsigned int h=0;h<track->recHitsSize();h++){
         //SiStripDetId detid = SiStripDetId((track->recHit(h))->geographicalId());
         //TrackingRecHit* recHit = (track->recHit(h))->clone();
     
         TrackingRecHit* recHit=  (track->recHit(h))->clone();
              
         if(const SiPixelRecHit* pixelHit=dynamic_cast<const SiPixelRecHit*>(recHit)){
            if(!usePixel) continue;

            unsigned int detid = pixelHit->geographicalId();
            stModInfo* MOD = MODsColl[detid];

            double cosine = (track->px()*MOD->Normal.x()+track->py()*MOD->Normal.y()+track->pz()*MOD->Normal.z())/track->p();
            float pathLen = MOD->Thickness/std::abs(cosine);
            float charge  = MOD->Normalization*pixelHit->cluster()->charge()*std::abs(cosine);
            dedxHits.push_back( DeDxHit( charge, MOD->Distance, pathLen, detid) );       

        } 
        delete recHit;
    }
     ///////////////////////////////////////


     sort(dedxHits.begin(),dedxHits.end(),less<DeDxHit>());   
     std::pair<float,float> val_and_error = m_estimator->dedx(dedxHits);

     //WARNING: Since the dEdX Error is not properly computed for the moment
     //It was decided to store the number of saturating cluster in that dataformat
     val_and_error.second = NClusterSaturating; 

     dedxEstimate[j] = DeDxData(val_and_error.first, val_and_error.second, dedxHits.size() );
  }
  filler.insert(trackCollHandle, dedxEstimate.begin(), dedxEstimate.end());

  // really fill the association map
  filler.fill();
   // put into the event 
  iEvent.put(trackDeDxEstimateAssociation);   
}



void DeDxEstimatorProducerPixelTripplet::MakeCalibrationMap()
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


//define this as a plug-in
DEFINE_FWK_MODULE(DeDxEstimatorProducerPixelTripplet);
