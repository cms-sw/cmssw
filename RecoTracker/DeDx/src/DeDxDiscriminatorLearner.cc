// -*- C++ -*-
//
// Package:    DeDxDiscriminatorLearner
// Class:      DeDxDiscriminatorLearner
// 
/**\class DeDxDiscriminatorLearner DeDxDiscriminatorLearner.cc RecoTracker/DeDxDiscriminatorLearner/src/DeDxDiscriminatorLearner.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Loic Quertenmont(querten)
//         Created:  Mon October 20 10:09:02 CEST 2008
//


// system include files
#include <memory>
//#include "DataFormats/Common/interface/ValueMap.h"

#include "RecoTracker/DeDx/interface/DeDxDiscriminatorLearner.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
//#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
//#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

//#include "RecoTracker/DeDx/interface/GenericAverageDeDxEstimator.h"
//#include "RecoTracker/DeDx/interface/TruncatedAverageDeDxEstimator.h"
//#include "RecoTracker/DeDx/interface/MedianDeDxEstimator.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"


using namespace reco;
using namespace std;
using namespace edm;

DeDxDiscriminatorLearner::DeDxDiscriminatorLearner(const edm::ParameterSet& iConfig) : ConditionDBWriter<PhysicsTools::Calibration::HistogramD2D>::ConditionDBWriter<PhysicsTools::Calibration::HistogramD2D>(iConfig)
{
   m_tracksTag                 = iConfig.getParameter<edm::InputTag>("tracks");
   m_trajTrackAssociationTag   = iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation");

   usePixel = iConfig.getParameter<bool>("UsePixel"); 
   useStrip = iConfig.getParameter<bool>("UseStrip");
   if(!usePixel && !useStrip)
   edm::LogWarning("DeDxDiscriminatorLearner") << "Pixel Hits AND Strip Hits will not be used to estimate dEdx --> BUG, Please Update the config file";

//   MapFileName         = iConfig.getParameter<std::string>("MapFile");

   MinTrackMomentum    = iConfig.getUntrackedParameter<double>  ("minTrackMomentum"   ,  3.0);
   MaxTrackMomentum    = iConfig.getUntrackedParameter<double>  ("maxTrackMomentum"   ,  99999.0); 
   MinTrackEta         = iConfig.getUntrackedParameter<double>  ("minTrackEta"        , -5.0);
   MaxTrackEta         = iConfig.getUntrackedParameter<double>  ("maxTrackEta"        ,  5.0);
   MaxNrStrips         = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"        ,  2);
   MinTrackHits        = iConfig.getUntrackedParameter<unsigned>("MinTrackHits"       ,  8);
   AllowSaturation     = iConfig.getUntrackedParameter<bool>    ("AllowSaturation"    ,  false);
}


DeDxDiscriminatorLearner::~DeDxDiscriminatorLearner(){}

// ------------ method called once each job just before starting event loop  ------------
void  DeDxDiscriminatorLearner::algoBeginJob(const edm::EventSetup& iSetup){
//   TH1::AddDirectory(kTRUE);

//   iSetup_                  = &iSetup;
//   MapFile                   = new TFile(MapFileName.c_str(), "RECREATE");
   Charge_Vs_Path_Barrel     = new TH2F ("Charge_Vs_Path_Barrel"     , "Charge_Vs_Path_Barrel" , 25, 0.2, 1.4, 10, 0, 5000);
//   Charge_Vs_Path_Barrel     = new TH2F ("Charge_Vs_Path_Barrel"     , "Charge_Vs_Path_Barrel" , 250, 0.2, 1.4, 1000, 0, 5000);
   Charge_Vs_Path_Endcap     = new TH2F ("Charge_Vs_Path_Endcap"     , "Charge_Vs_Path_Endcap" , 250, 0.2, 1.4, 1000, 0, 5000);


   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
   //  m_tracker=&(* tkGeom );

   vector<GeomDet*> Det = tkGeom->dets();
   for(unsigned int i=0;i<Det.size();i++){
      DetId  Detid  = Det[i]->geographicalId();
      int    SubDet = Detid.subdetId();

      if( SubDet == StripSubdetector::TIB ||  SubDet == StripSubdetector::TID ||
          SubDet == StripSubdetector::TOB ||  SubDet == StripSubdetector::TEC  ){

          StripGeomDetUnit* DetUnit     = dynamic_cast<StripGeomDetUnit*> (Det[i]);
          if(!DetUnit)continue;

          const StripTopology& Topo     = DetUnit->specificTopology();
          unsigned int         NAPV     = Topo.nstrips()/128;

          double Eta           = DetUnit->position().basicVector().eta();
          double R             = DetUnit->position().basicVector().transverse();
          double Thick         = DetUnit->surface().bounds().thickness();

          stModInfo* MOD       = new stModInfo;
          MOD->DetId           = Detid.rawId();
          MOD->SubDet          = SubDet;
          MOD->Eta             = Eta;
          MOD->R               = R;
          MOD->Thickness       = Thick;
          MOD->NAPV            = NAPV;
          MODsColl[MOD->DetId] = MOD;
      }
   }


}

// ------------ method called once each job just after ending the event loop  ------------
void  DeDxDiscriminatorLearner::algoEndJob(){
//   MapFile->Write();
//   MapFile->Close();
}



void DeDxDiscriminatorLearner::algoAnalyze(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //  iEvent_ = &iEvent;  

  Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  iEvent.getByLabel(m_trajTrackAssociationTag, trajTrackAssociationHandle);
  const TrajTrackAssociationCollection TrajToTrackMap = *trajTrackAssociationHandle.product();

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByLabel(m_tracksTag,trackCollectionHandle);
 
  unsigned track_index = 0;
  for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap.begin(); it!=TrajToTrackMap.end(); ++it, track_index++) {
      const Track      track = *it->val;
      const Trajectory traj  = *it->key;

      if(track.eta()<MinTrackEta || track.eta()>MaxTrackEta){printf("Eta Cut\n");continue;}
      if(track.p()<MinTrackMomentum || track.p()>MaxTrackMomentum){printf("Pt Cut\n");continue;}
      if(track.found()<MinTrackHits){printf("Hits Cut\n");continue;}

      vector<TrajectoryMeasurement> measurements = traj.measurements();
      if(traj.foundHits()<(int)MinTrackHits)continue;

      MeasurementProbabilities.clear();
      for(vector<TrajectoryMeasurement>::const_iterator measurement_it = measurements.begin(); measurement_it!=measurements.end(); measurement_it++){

         TrajectoryStateOnSurface trajState = measurement_it->updatedState();
         if( !trajState.isValid() ) continue;

         const TrackingRecHit*         hit               = (*measurement_it->recHit()).hit();
         const SiStripRecHit2D*        sistripsimplehit  = dynamic_cast<const SiStripRecHit2D*>(hit);
         const SiStripMatchedRecHit2D* sistripmatchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);

         if(sistripsimplehit)
         {
             ComputeChargeOverPath(sistripsimplehit, trajState, &iSetup, &track, traj.chiSquared());
         }else if(sistripmatchedhit){
             ComputeChargeOverPath(sistripmatchedhit->monoHit()  ,trajState, &iSetup, &track, traj.chiSquared());
             ComputeChargeOverPath(sistripmatchedhit->stereoHit(),trajState, &iSetup, &track, traj.chiSquared());
         }else{
         }

      }
   }
}


double
DeDxDiscriminatorLearner::ComputeChargeOverPath(const SiStripRecHit2D* sistripsimplehit,TrajectoryStateOnSurface trajState, const edm::EventSetup* iSetup,  const Track* track, double trajChi2OverN)
{

   LocalVector          trackDirection = trajState.localDirection();
   double                  cosine      = trackDirection.z()/trackDirection.mag();
   const SiStripCluster*   Cluster     = (sistripsimplehit->cluster()).get();
   const vector<uint8_t>&  Ampls       = Cluster->amplitudes();
   uint32_t                DetId       = Cluster->geographicalId();
   int                     FirstStrip  = Cluster->firstStrip();
   bool                    Saturation  = false;
   bool                    Overlaping  = false;
   int                     Charge      = 0;
   stModInfo* MOD                      = MODsColl[DetId];


   if(!IsFarFromBorder(trajState,DetId, iSetup)){/*printf("tooCloseFromBorder\n");*/return -1;}


   if(FirstStrip==0                                  )Overlaping=true;
   if(FirstStrip==128                                )Overlaping=true;
   if(FirstStrip==256                                )Overlaping=true;
   if(FirstStrip==384                                )Overlaping=true;
   if(FirstStrip==512                                )Overlaping=true;
   if(FirstStrip==640                                )Overlaping=true;

   if(FirstStrip<=127 && FirstStrip+Ampls.size()>127)Overlaping=true;
   if(FirstStrip<=255 && FirstStrip+Ampls.size()>255)Overlaping=true;
   if(FirstStrip<=383 && FirstStrip+Ampls.size()>383)Overlaping=true;
   if(FirstStrip<=511 && FirstStrip+Ampls.size()>511)Overlaping=true;
   if(FirstStrip<=639 && FirstStrip+Ampls.size()>639)Overlaping=true;

   if(FirstStrip+Ampls.size()==127                   )Overlaping=true;
   if(FirstStrip+Ampls.size()==255                   )Overlaping=true;
   if(FirstStrip+Ampls.size()==383                   )Overlaping=true;
   if(FirstStrip+Ampls.size()==511                   )Overlaping=true;
   if(FirstStrip+Ampls.size()==639                   )Overlaping=true;
   if(FirstStrip+Ampls.size()==767                   )Overlaping=true;
//   if(!DiscriminatorMode && Overlaping){printf("Overlapping\n");return -1;}


   for(unsigned int a=0;a<Ampls.size();a++){Charge+=Ampls[a];if(Ampls[a]>=254)Saturation=true;}
   double path                    = (10.0*MOD->Thickness)/fabs(cosine);
   double ClusterChargeOverPath   = (double)Charge / path ;

   if(Ampls.size()>MaxNrStrips)      {/*printf("tooMuchStrips\n");*/return -1;}
//   if(!DiscriminatorMode && Saturation && !AllowSaturation){printf("Saturation\n");return -1;}

   if(MOD->SubDet == StripSubdetector::TIB || MOD->SubDet == StripSubdetector::TOB) Charge_Vs_Path_Barrel->Fill(path,ClusterChargeOverPath);
   if(MOD->SubDet == StripSubdetector::TID || MOD->SubDet == StripSubdetector::TEC) Charge_Vs_Path_Endcap->Fill(path,ClusterChargeOverPath);

   return ClusterChargeOverPath;
}


bool DeDxDiscriminatorLearner::IsFarFromBorder(TrajectoryStateOnSurface trajState, const uint32_t detid, const edm::EventSetup* iSetup)
{
  edm::ESHandle<TrackerGeometry> tkGeom; iSetup->get<TrackerDigiGeometryRecord>().get( tkGeom );

  LocalPoint  HitLocalPos   = trajState.localPosition();
  LocalError  HitLocalError = trajState.localError().positionError() ;

  const GeomDetUnit* it = tkGeom->idToDetUnit(DetId(detid));
  if (dynamic_cast<const StripGeomDetUnit*>(it)==0 && dynamic_cast<const PixelGeomDetUnit*>(it)==0) {
     std::cout << "this detID doesn't seem to belong to the Tracker" << std::endl;
     return false;
  }

  const BoundPlane plane = it->surface();
  const TrapezoidalPlaneBounds* trapezoidalBounds( dynamic_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
  const RectangularPlaneBounds* rectangularBounds( dynamic_cast<const RectangularPlaneBounds*>(&(plane.bounds())));

  double DistFromBorder = 1.0;
  double HalfWidth      = it->surface().bounds().width()  /2.0;
  double HalfLength     = it->surface().bounds().length() /2.0;

  if(trapezoidalBounds)
  {
     std::vector<float> const & parameters = (*trapezoidalBounds).parameters();
     HalfLength     = parameters[3];
     double t       = (HalfLength + HitLocalPos.y()) / (2*HalfLength) ;
     HalfWidth      = parameters[0] + (parameters[1]-parameters[0]) * t;
  }else if(rectangularBounds){
     HalfWidth      = it->surface().bounds().width()  /2.0;
     HalfLength     = it->surface().bounds().length() /2.0;
  }else{return false;}

//  if (fabs(HitLocalPos.x())+HitLocalError.xx() >= (HalfWidth  - DistFromBorder) ) return false;//Don't think is really necessary
  if (fabs(HitLocalPos.y())+HitLocalError.yy() >= (HalfLength - DistFromBorder) ) return false;

  return true;
}

PhysicsTools::Calibration::HistogramD2D* DeDxDiscriminatorLearner::getNewObject(){

   PhysicsTools::Calibration::HistogramD2D* obj = new PhysicsTools::Calibration::HistogramD2D(300, 0., 3., 1000,0.,1000.);

   for(int ix=0; ix<300; ix++){
     for(int iy=0; iy<1000; iy++){

       obj->setBinContent(ix, iy, iy/999. );
       
     }
   }

   return obj;
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeDxDiscriminatorLearner);
