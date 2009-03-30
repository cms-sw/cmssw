// -*- C++ -*-
//
// Package:    DeDxDiscriminatorLearner2D
// Class:      DeDxDiscriminatorLearner2D
// 
/**\class DeDxDiscriminatorLearner2D DeDxDiscriminatorLearner2D.cc RecoTracker/DeDxDiscriminatorLearner/src/DeDxDiscriminatorLearner2D.cc

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

#include "RecoTracker/DeDx/interface/DeDxDiscriminatorLearner2D.h"

//#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

using namespace reco;
using namespace std;
using namespace edm;

DeDxDiscriminatorLearner2D::DeDxDiscriminatorLearner2D(const edm::ParameterSet& iConfig) : ConditionDBWriter<PhysicsTools::Calibration::HistogramD2D>::ConditionDBWriter<PhysicsTools::Calibration::HistogramD2D>(iConfig)
{
   m_tracksTag                 = iConfig.getParameter<edm::InputTag>("tracks");
   m_trajTrackAssociationTag   = iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation");

   usePixel = iConfig.getParameter<bool>("UsePixel"); 
   useStrip = iConfig.getParameter<bool>("UseStrip");
   if(!usePixel && !useStrip)
   edm::LogWarning("DeDxDiscriminatorLearner2D") << "Pixel Hits AND Strip Hits will not be used to estimate dEdx --> BUG, Please Update the config file";

   MinTrackMomentum    = iConfig.getUntrackedParameter<double>  ("minTrackMomentum"   ,  3.0);
   MaxTrackMomentum    = iConfig.getUntrackedParameter<double>  ("maxTrackMomentum"   ,  99999.0); 
   MinTrackEta         = iConfig.getUntrackedParameter<double>  ("minTrackEta"        , -5.0);
   MaxTrackEta         = iConfig.getUntrackedParameter<double>  ("maxTrackEta"        ,  5.0);
   MaxNrStrips         = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"        ,  255);
   MinTrackHits        = iConfig.getUntrackedParameter<unsigned>("MinTrackHits"       ,  4);

   algoMode            = iConfig.getUntrackedParameter<string>  ("AlgoMode"           ,  "SingleJob");
   HistoFile           = iConfig.getUntrackedParameter<string>  ("HistoFile"        ,  "out.root");
}


DeDxDiscriminatorLearner2D::~DeDxDiscriminatorLearner2D(){}

// ------------ method called once each job just before starting event loop  ------------

void  DeDxDiscriminatorLearner2D::algoBeginJob(const edm::EventSetup& iSetup)
{
   Charge_Vs_Path = new TH2F ("Charge_Vs_Path"     , "Charge_Vs_Path" , 24, 0.2, 1.4, 250, 0, 5000);

   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
   m_tracker = tkGeom.product();

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

void DeDxDiscriminatorLearner2D::algoEndJob()
{
   if( strcmp(algoMode.c_str(),"MultiJob")==0){
	TFile* Output = new TFile(HistoFile.c_str(), "RECREATE");
      	Charge_Vs_Path->Write();
	Output->Write();
	Output->Close();
   }else if( strcmp(algoMode.c_str(),"WriteOnDB")==0){
        TFile* Input = new TFile(HistoFile.c_str() );
	Charge_Vs_Path = (TH2F*)(Input->FindObjectAny("Charge_Vs_Path"))->Clone();  
	Input->Close();
   }
}

void DeDxDiscriminatorLearner2D::algoAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  iEvent.getByLabel(m_trajTrackAssociationTag, trajTrackAssociationHandle);
  const TrajTrackAssociationCollection TrajToTrackMap = *trajTrackAssociationHandle.product();

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByLabel(m_tracksTag,trackCollectionHandle);

  unsigned track_index = 0;
  for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap.begin(); it!=TrajToTrackMap.end(); ++it, track_index++) {
      const Track      track = *it->val;
      const Trajectory traj  = *it->key;

      if(track.eta()  <MinTrackEta      || track.eta()>MaxTrackEta     ){printf("Eta  Cut\n");continue;}
      if(track.p()    <MinTrackMomentum || track.p()  >MaxTrackMomentum){printf("Pt   Cut\n");continue;}
      if(track.found()<MinTrackHits                                    ){printf("Hits Cut\n");continue;}

      vector<TrajectoryMeasurement> measurements = traj.measurements();
      for(vector<TrajectoryMeasurement>::const_iterator measurement_it = measurements.begin(); measurement_it!=measurements.end(); measurement_it++)
      {
         TrajectoryStateOnSurface trajState = measurement_it->updatedState();
         if( !trajState.isValid() ) continue;

         const TrackingRecHit*         hit               = (*measurement_it->recHit()).hit();
         const SiStripRecHit2D*        sistripsimplehit  = dynamic_cast<const SiStripRecHit2D*>(hit);
         const SiStripMatchedRecHit2D* sistripmatchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);

         if(sistripsimplehit)
         {
             Learn(sistripsimplehit, trajState);
         }else if(sistripmatchedhit){
             Learn(sistripmatchedhit->monoHit()  ,trajState);
             Learn(sistripmatchedhit->stereoHit(),trajState);
         }else{
         }

      }
   }
}


void DeDxDiscriminatorLearner2D::Learn(const SiStripRecHit2D* sistripsimplehit,TrajectoryStateOnSurface trajState)
{
   // Get All needed variables
   LocalVector             trackDirection = trajState.localDirection();
   double                  cosine         = trackDirection.z()/trackDirection.mag();
   const SiStripCluster*   cluster        = (sistripsimplehit->cluster()).get();
   const vector<uint8_t>&  ampls          = cluster->amplitudes();
   uint32_t                detId          = cluster->geographicalId();
   int                     firstStrip     = cluster->firstStrip();
   stModInfo* MOD                         = MODsColl[detId];

   // Sanity Checks
   if( ampls.size()>MaxNrStrips)                                                                      { return; }
// if( DeDxDiscriminatorTools::IsSaturatingStrip  (ampls))                                            { return; }
   if( DeDxDiscriminatorTools::IsSpanningOver2APV (firstStrip, ampls.size()))                         { return; }
   if(!DeDxDiscriminatorTools::IsFarFromBorder    (trajState, m_tracker->idToDetUnit(DetId(detId)) )) { return; }

   // Fill Histo Path Vs Charge/Path
   double charge = DeDxDiscriminatorTools::charge(ampls);
   double path   = DeDxDiscriminatorTools::path(cosine,MOD->Thickness);
   Charge_Vs_Path->Fill(path,charge/path);
}


PhysicsTools::Calibration::HistogramD2D* DeDxDiscriminatorLearner2D::getNewObject()
{
   if( strcmp(algoMode.c_str(),"MultiJob")==0)return NULL;

   PhysicsTools::Calibration::HistogramD2D* obj;
   obj = new PhysicsTools::Calibration::HistogramD2D(
                Charge_Vs_Path->GetNbinsX(), Charge_Vs_Path->GetXaxis()->GetXmin(),  Charge_Vs_Path->GetXaxis()->GetXmax(),
                Charge_Vs_Path->GetNbinsY(), Charge_Vs_Path->GetYaxis()->GetXmin(),  Charge_Vs_Path->GetYaxis()->GetXmax());

   for(int ix=0; ix<Charge_Vs_Path->GetNbinsX(); ix++){
      for(int iy=0; iy<Charge_Vs_Path->GetNbinsY(); iy++){
         obj->setBinContent(ix, iy, Charge_Vs_Path->GetBinContent(ix,iy) );       
//         if(Charge_Vs_Path->GetBinContent(ix,iy)!=0)printf("%i %i --> %f\n",ix,iy, Charge_Vs_Path->GetBinContent(ix,iy)); 
      }
   }

   return obj;
}



//define this as a plug-in
DEFINE_FWK_MODULE(DeDxDiscriminatorLearner2D);
