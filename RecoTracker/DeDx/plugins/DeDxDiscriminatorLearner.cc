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

#include "RecoTracker/DeDx/plugins/DeDxDiscriminatorLearner.h"

//#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

using namespace reco;
using namespace std;
using namespace edm;

DeDxDiscriminatorLearner::DeDxDiscriminatorLearner(const edm::ParameterSet& iConfig) : ConditionDBWriter<PhysicsTools::Calibration::HistogramD3D>(iConfig)
{
   m_tracksTag                 = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));
   m_trajTrackAssociationTag   = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation"));

   usePixel = iConfig.getParameter<bool>("UsePixel"); 
   useStrip = iConfig.getParameter<bool>("UseStrip");
   if(!usePixel && !useStrip)
   edm::LogWarning("DeDxDiscriminatorLearner") << "Pixel Hits AND Strip Hits will not be used to estimate dEdx --> BUG, Please Update the config file";

   P_Min	       = iConfig.getParameter<double>  ("P_Min"  );
   P_Max               = iConfig.getParameter<double>  ("P_Max"  );
   P_NBins             = iConfig.getParameter<int>     ("P_NBins");
   Path_Min            = iConfig.getParameter<double>  ("Path_Min"  );
   Path_Max            = iConfig.getParameter<double>  ("Path_Max"  );
   Path_NBins          = iConfig.getParameter<int>     ("Path_NBins");
   Charge_Min          = iConfig.getParameter<double>  ("Charge_Min"  );
   Charge_Max          = iConfig.getParameter<double>  ("Charge_Max"  );
   Charge_NBins        = iConfig.getParameter<int>     ("Charge_NBins");

   MinTrackMomentum    = iConfig.getUntrackedParameter<double>  ("minTrackMomentum"   ,  3.0);
   MaxTrackMomentum    = iConfig.getUntrackedParameter<double>  ("maxTrackMomentum"   ,  99999.0); 
   MinTrackEta         = iConfig.getUntrackedParameter<double>  ("minTrackEta"        , -5.0);
   MaxTrackEta         = iConfig.getUntrackedParameter<double>  ("maxTrackEta"        ,  5.0);
   MaxNrStrips         = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"        ,  255);
   MinTrackHits        = iConfig.getUntrackedParameter<unsigned>("MinTrackHits"       ,  4);

   algoMode            = iConfig.getUntrackedParameter<string>  ("AlgoMode"           ,  "SingleJob");
   HistoFile           = iConfig.getUntrackedParameter<string>  ("HistoFile"        ,  "out.root");
}


DeDxDiscriminatorLearner::~DeDxDiscriminatorLearner(){}

// ------------ method called once each job just before starting event loop  ------------

void  DeDxDiscriminatorLearner::algoBeginJob(const edm::EventSetup& iSetup)
{
//   Charge_Vs_Path = new TH2F ("Charge_Vs_Path"     , "Charge_Vs_Path" , 24, 0.2, 1.4, 250, 0, 5000);
   Charge_Vs_Path = new TH3F ("Charge_Vs_Path"     , "Charge_Vs_Path" , P_NBins, P_Min, P_Max, Path_NBins, Path_Min, Path_Max, Charge_NBins, Charge_Min, Charge_Max);


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

void DeDxDiscriminatorLearner::algoEndJob()
{

   if( strcmp(algoMode.c_str(),"MultiJob")==0){
	TFile* Output = new TFile(HistoFile.c_str(), "RECREATE");
      	Charge_Vs_Path->Write();
	Output->Write();
	Output->Close();
   }else if( strcmp(algoMode.c_str(),"WriteOnDB")==0){
        TFile* Input = new TFile(HistoFile.c_str() );
	Charge_Vs_Path = (TH3F*)(Input->FindObjectAny("Charge_Vs_Path"))->Clone();  
	Input->Close();
   }

}

void DeDxDiscriminatorLearner::algoAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
   m_tracker = tkGeom.product();



  Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  iEvent.getByToken(m_trajTrackAssociationTag, trajTrackAssociationHandle);
  const TrajTrackAssociationCollection TrajToTrackMap = *trajTrackAssociationHandle.product();

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByToken(m_tracksTag,trackCollectionHandle);

  unsigned track_index = 0;
  for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap.begin(); it!=TrajToTrackMap.end(); ++it, track_index++) {

      const Track      track = *it->val;
      const Trajectory traj  = *it->key;

      if(track.eta()  <MinTrackEta      || track.eta()>MaxTrackEta     ){continue;}
      if(track.p()    <MinTrackMomentum || track.p()  >MaxTrackMomentum){continue;}
      if(track.found()<MinTrackHits                                    ){continue;}

      vector<TrajectoryMeasurement> measurements = traj.measurements();
      for(vector<TrajectoryMeasurement>::const_iterator measurement_it = measurements.begin(); measurement_it!=measurements.end(); measurement_it++)
      {
         TrajectoryStateOnSurface trajState = measurement_it->updatedState();
         if( !trajState.isValid() ) continue;

         const TrackingRecHit*         hit               = (*measurement_it->recHit()).hit();
         const SiStripRecHit2D*        sistripsimplehit  = dynamic_cast<const SiStripRecHit2D*>(hit);
         const SiStripMatchedRecHit2D* sistripmatchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);
         const SiStripRecHit1D*        sistripsimple1dhit  = dynamic_cast<const SiStripRecHit1D*>(hit);

         if(sistripsimplehit){
             Learn((sistripsimplehit->cluster()).get(), trajState);
         }else if(sistripmatchedhit){
             Learn(&sistripmatchedhit->monoCluster(),trajState);
             Learn(&sistripmatchedhit->stereoCluster(),trajState);
         }else if(sistripsimple1dhit){
             Learn((sistripsimple1dhit->cluster()).get(), trajState);
         }else{
         }

      }
   }

}


void DeDxDiscriminatorLearner::Learn(const SiStripCluster*   cluster,TrajectoryStateOnSurface trajState)
{
   // Get All needed variables
   LocalVector             trackDirection = trajState.localDirection();
   double                  cosine         = trackDirection.z()/trackDirection.mag();
   const vector<uint8_t>&  ampls          = cluster->amplitudes();
   uint32_t                detId          = 0; // zero since long time cluster->geographicalId();
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
   Charge_Vs_Path->Fill(trajState.localMomentum().mag(),path,charge/path);
}


PhysicsTools::Calibration::HistogramD3D* DeDxDiscriminatorLearner::getNewObject()
{
//   if( strcmp(algoMode.c_str(),"MultiJob")==0)return NULL;

   PhysicsTools::Calibration::HistogramD3D* obj;
   obj = new PhysicsTools::Calibration::HistogramD3D(
                Charge_Vs_Path->GetNbinsX(), Charge_Vs_Path->GetXaxis()->GetXmin(),  Charge_Vs_Path->GetXaxis()->GetXmax(),
                Charge_Vs_Path->GetNbinsY(), Charge_Vs_Path->GetYaxis()->GetXmin(),  Charge_Vs_Path->GetYaxis()->GetXmax(),
	        Charge_Vs_Path->GetNbinsZ(), Charge_Vs_Path->GetZaxis()->GetXmin(),  Charge_Vs_Path->GetZaxis()->GetXmax());

   for(int ix=0; ix<=Charge_Vs_Path->GetNbinsX()+1; ix++){
      for(int iy=0; iy<=Charge_Vs_Path->GetNbinsY()+1; iy++){
         for(int iz=0; iz<=Charge_Vs_Path->GetNbinsZ()+1; iz++){
            obj->setBinContent(ix, iy, iz, Charge_Vs_Path->GetBinContent(ix,iy, iz) );       
//          if(Charge_Vs_Path->GetBinContent(ix,iy)!=0)printf("%i %i %i --> %f\n",ix,iy, iz, Charge_Vs_Path->GetBinContent(ix,iy,iz)); 
         }
      }
   }

   return obj;
}



//define this as a plug-in
DEFINE_FWK_MODULE(DeDxDiscriminatorLearner);
