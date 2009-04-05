// -*- C++ -*-
//
// Package:    DeDxDiscriminatorProducer
// Class:      DeDxDiscriminatorProducer
// 
/**\class DeDxDiscriminatorProducer DeDxDiscriminatorProducer.cc RecoTracker/DeDxDiscriminatorProducer/src/DeDxDiscriminatorProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  andrea
//         Created:  Thu May 31 14:09:02 CEST 2007
//    Code Updates:  loic Quertenmont (querten)
//         Created:  Thu May 10 14:09:02 CEST 2008
// $Id: DeDxDiscriminatorProducer.cc,v 1.8 2009/03/04 13:34:25 vlimant Exp $
//
//


// system include files
#include <memory>
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"

#include "RecoTracker/DeDx/interface/DeDxDiscriminatorProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "CondFormats/DataRecord/interface/SiStripDeDxMip_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxElectron_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxProton_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxPion_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxKaon_3D_Rcd.h"


#include "TFile.h"

using namespace reco;
using namespace std;
using namespace edm;

DeDxDiscriminatorProducer::DeDxDiscriminatorProducer(const edm::ParameterSet& iConfig)
{

   produces<ValueMap<DeDxData> >();

   m_tracksTag = iConfig.getParameter<edm::InputTag>("tracks");
   m_trajTrackAssociationTag   = iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation");

   usePixel = iConfig.getParameter<bool>("UsePixel"); 
   useStrip = iConfig.getParameter<bool>("UseStrip");
   if(!usePixel && !useStrip)
   edm::LogWarning("DeDxHitsProducer") << "Pixel Hits AND Strip Hits will not be used to estimate dEdx --> BUG, Please Update the config file";

   Formula             = iConfig.getUntrackedParameter<unsigned>("Formula"            ,  0);
   Reccord             = iConfig.getUntrackedParameter<string>  ("Reccord"            , "SiStripDeDxMip_3D_Rcd");

   MinTrackMomentum    = iConfig.getUntrackedParameter<double>  ("minTrackMomentum"   ,  3.0);
   MaxTrackMomentum    = iConfig.getUntrackedParameter<double>  ("maxTrackMomentum"   ,  99999.0); 
   MinTrackEta         = iConfig.getUntrackedParameter<double>  ("minTrackEta"        , -5.0);
   MaxTrackEta         = iConfig.getUntrackedParameter<double>  ("maxTrackEta"        ,  5.0);
   MaxNrStrips         = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"        ,  255);
   MinTrackHits        = iConfig.getUntrackedParameter<unsigned>("MinTrackHits"       ,  4);
}


DeDxDiscriminatorProducer::~DeDxDiscriminatorProducer(){}

// ------------ method called once each job just before starting event loop  ------------
void  DeDxDiscriminatorProducer::beginRun(edm::Run & run, const edm::EventSetup& iSetup)
{
   edm::ESHandle<PhysicsTools::Calibration::HistogramD3D> DeDxMapHandle_;    
   if(      strcmp(Reccord.c_str(),"SiStripDeDxMip_3D_Rcd")==0){
      iSetup.get<SiStripDeDxMip_3D_Rcd>() .get(DeDxMapHandle_);
   }else if(strcmp(Reccord.c_str(),"SiStripDeDxPion_3D_Rcd")==0){
      iSetup.get<SiStripDeDxPion_3D_Rcd>().get(DeDxMapHandle_);
   }else if(strcmp(Reccord.c_str(),"SiStripDeDxKaon_3D_Rcd")==0){
      iSetup.get<SiStripDeDxKaon_3D_Rcd>().get(DeDxMapHandle_);
   }else if(strcmp(Reccord.c_str(),"SiStripDeDxProton_3D_Rcd")==0){
      iSetup.get<SiStripDeDxProton_3D_Rcd>().get(DeDxMapHandle_);
   }else if(strcmp(Reccord.c_str(),"SiStripDeDxElectron_3D_Rcd")==0){
      iSetup.get<SiStripDeDxElectron_3D_Rcd>().get(DeDxMapHandle_);
   }else{
      printf("The reccord %s is not known by the DeDxDiscriminatorProducer\n", Reccord.c_str());
      printf("Program will exit now\n");
      exit(0);
   }
   DeDxMap_ = *DeDxMapHandle_.product();

   double xmin = DeDxMap_.rangeX().min;
   double xmax = DeDxMap_.rangeX().max;
   double ymin = DeDxMap_.rangeY().min;
   double ymax = DeDxMap_.rangeY().max;
   double zmin = DeDxMap_.rangeZ().min;
   double zmax = DeDxMap_.rangeZ().max;

   Prob_ChargePath  = new TH3D ("Prob_ChargePath"     , "Prob_ChargePath" , DeDxMap_.numberOfBinsX(), xmin, xmax, DeDxMap_.numberOfBinsY() , ymin, ymax, DeDxMap_.numberOfBinsZ(), zmin, zmax);

   
   for(int i=0;i<Prob_ChargePath->GetXaxis()->GetNbins();i++){
      for(int j=0;j<Prob_ChargePath->GetYaxis()->GetNbins();j++){
         double Ni = 0;
         for(int k=0;k<Prob_ChargePath->GetZaxis()->GetNbins();k++){ Ni+=DeDxMap_.binContent(i,j,k);} 

         for(int k=0;k<Prob_ChargePath->GetZaxis()->GetNbins();k++){
            double tmp = 0;
            for(int l=0;l<=k;l++){ tmp+=DeDxMap_.binContent(i,j,l);}

   	    if(Ni>0){
               Prob_ChargePath->SetBinContent (i, j, k, tmp/Ni);
  	    }else{
               Prob_ChargePath->SetBinContent (i, j, k, 0);
	    }
         }
      }
   }



/*
   for(int i=0;i<Prob_ChargePath->GetXaxis()->GetNbins();i++){
      for(int j=0;j<Prob_ChargePath->GetYaxis()->GetNbins();j++){
         double tmp = DeDxMap_.binContent(i,j);
         Prob_ChargePath->SetBinContent (i, j, tmp);
	 printf("%i %i --> %f\n",i,j,tmp);
      }
   }
*/


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

          double Eta     = DetUnit->position().basicVector().eta();
          double R       = DetUnit->position().basicVector().transverse();
          double Thick   = DetUnit->surface().bounds().thickness();

          stModInfo* MOD = new stModInfo;
          MOD->DetId     = Detid.rawId();
          MOD->SubDet    = SubDet;
          MOD->Eta       = Eta;
          MOD->R         = R;
          MOD->Thickness = Thick;
          MOD->NAPV      = NAPV;
          MODsColl[MOD->DetId] = MOD;
      }
   }
 
}

void  DeDxDiscriminatorProducer::endJob()
{
/*
   TFile* file = new TFile("MipsMap.root", "RECREATE");
   Prob_ChargePath->Write();
   file->Write();
   file->Close();
*/
}



void DeDxDiscriminatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  auto_ptr<ValueMap<DeDxData> > trackDeDxDiscrimAssociation(new ValueMap<DeDxData> );  
  ValueMap<DeDxData>::Filler filler(*trackDeDxDiscrimAssociation);

  Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  iEvent.getByLabel(m_trajTrackAssociationTag, trajTrackAssociationHandle);
  const TrajTrackAssociationCollection TrajToTrackMap = *trajTrackAssociationHandle.product();

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByLabel(m_tracksTag,trackCollectionHandle);
 
   std::vector<DeDxData> dEdxDiscrims( TrajToTrackMap.size() );

   unsigned track_index = 0;
   for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap.begin(); it!=TrajToTrackMap.end(); ++it, track_index++) {
      dEdxDiscrims[track_index] = DeDxData(-1, -2, 0 );

      const Track      track = *it->val;
      const Trajectory traj  = *it->key;

      if(track.eta()  <MinTrackEta      || track.eta()>MaxTrackEta     ){printf("Eta  Cut\n");continue;}
      if(track.p()    <MinTrackMomentum || track.p()  >MaxTrackMomentum){printf("Pt   Cut\n");continue;}
      if(track.found()<MinTrackHits                                    ){printf("Hits Cut\n");continue;}

      std::vector<double> vect_probs;
      vector<TrajectoryMeasurement> measurements = traj.measurements();
      for(vector<TrajectoryMeasurement>::const_iterator measurement_it = measurements.begin(); measurement_it!=measurements.end(); measurement_it++){

         TrajectoryStateOnSurface trajState = measurement_it->updatedState();
         if( !trajState.isValid() ) continue;

         const TrackingRecHit*         hit               = (*measurement_it->recHit()).hit();
         const SiStripRecHit2D*        sistripsimplehit  = dynamic_cast<const SiStripRecHit2D*>(hit);
         const SiStripMatchedRecHit2D* sistripmatchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);

	 double Prob;
         if(sistripsimplehit)
         {           
	     Prob = GetProbability(sistripsimplehit, trajState);	                 if(Prob>=0) vect_probs.push_back(Prob);
         }else if(sistripmatchedhit){
             Prob = GetProbability(sistripmatchedhit->monoHit(), trajState);             if(Prob>=0) vect_probs.push_back(Prob);
             Prob = GetProbability(sistripmatchedhit->stereoHit(), trajState);           if(Prob>=0) vect_probs.push_back(Prob);
         }else{
         }
      }

      double estimator          = ComputeDiscriminator(vect_probs);
      int    size               = vect_probs.size();
      dEdxDiscrims[track_index] = DeDxData(estimator, -1, size );
   }

  filler.insert(trackCollectionHandle, dEdxDiscrims.begin(), dEdxDiscrims.end());
  filler.fill();
  iEvent.put(trackDeDxDiscrimAssociation);
}


double DeDxDiscriminatorProducer::GetProbability(const SiStripRecHit2D* sistripsimplehit,TrajectoryStateOnSurface trajState)
{
   // Get All needed variables
   LocalVector             trackDirection = trajState.localDirection();
   double                  cosine         = trackDirection.z()/trackDirection.mag();
   const SiStripCluster*   cluster        = (sistripsimplehit->cluster()).get();
   const vector<uint8_t>&  ampls          = cluster->amplitudes();
   uint32_t                detId          = cluster->geographicalId();
//   int                     firstStrip     = cluster->firstStrip();
   stModInfo* MOD                         = MODsColl[detId];


   // Sanity Checks
   if( ampls.size()>MaxNrStrips)                                                                      {return -1;}
// if( DeDxDiscriminatorTools::IsSaturatingStrip  (ampls))                                            {return -1;}
// if( DeDxDiscriminatorTools::IsSpanningOver2APV (firstStrip, ampls.size()))                         {return -1;}
// if(!DeDxDiscriminatorTools::IsFarFromBorder    (trajState, m_tracker->idToDetUnit(DetId(detId)) )) {return -1;}


   // Find Probability for this given Charge and Path
   double charge = DeDxDiscriminatorTools::charge(ampls);
   double path   = DeDxDiscriminatorTools::path(cosine,MOD->Thickness);

   int    BinX   = Prob_ChargePath->GetXaxis()->FindBin(trackDirection.mag());
   int    BinY   = Prob_ChargePath->GetXaxis()->FindBin(path);
   int    BinZ   = Prob_ChargePath->GetYaxis()->FindBin(charge/path);
   
   return Prob_ChargePath->GetBinContent(BinX,BinY,BinZ);
}


double DeDxDiscriminatorProducer::ComputeDiscriminator(std::vector<double>& vect_probs)
{
   double estimator = -1;
   int    size      = vect_probs.size();
   if(size<=0)return estimator;

   if(Formula==0){
      double P = 1;
      for(int i=0;i<size;i++){
         if(vect_probs[i]<=0.0001){P *= pow(0.0001       , 1.0/size);}
         else                     {P *= pow(vect_probs[i], 1.0/size);}
      }
      estimator = P;
   }else if(Formula==1){
      std::sort(vect_probs.begin(), vect_probs.end(), std::less<double>() );
      for(int i=0;i<size;i++){if(vect_probs[i]<=0.0001)vect_probs[i] = 0.0001;    }

      double SumJet = 0.;
      for(int i=0;i<size;i++){ SumJet+= log(vect_probs[i]); }

      double Loginvlog=log(-SumJet);
      double Prob =1.;
      double lfact=1.;

      for(int l=1; l!=size; l++){
         lfact*=l;
         Prob+=exp(l*Loginvlog-log(1.*lfact));
      }

      double LogProb=log(Prob);
      double ProbJet=std::min(exp(std::max(LogProb+SumJet,-30.)),1.);
      estimator = -log10(ProbJet)/4.;
      estimator = 1-estimator;
   }else if(Formula==2){
      std::sort(vect_probs.begin(), vect_probs.end(), std::less<double>() );
      double P = 1.0/(12*size);
      for(int i=1;i<=size;i++){
         P += pow(vect_probs[i-1] - ((2.0*i-1.0)/(2.0*size)),2);
      }
      P *= (1.0/size);
      estimator = P;
      if(estimator>=0.333)printf("BUG --> Estimator>0.3333 for SMI --> %f\n",estimator);
      if(estimator>=0.333){
	 printf("DETAIL OF PREVIOUS BUG : \nPROBA = ");
         for(int i=1;i<=size;i++){printf(" %f ",vect_probs[i-1]);}
         P = 1.0/(12*size);
         printf("P = %f\n",P);
         for(int i=1;i<=size;i++){
            P += pow(vect_probs[i-1] - ((2.0*i-1.0)/(2.0*size)),2);
            printf("P = %f\n",P);
         }
         P *= (1.0/size);
         printf("P Normalise = %f\n",P);
      }
   }else{
      std::sort(vect_probs.begin(), vect_probs.end(), std::less<double>() );
      double P = 1.0/(12*size);
      for(int i=1;i<=size;i++){
         P += vect_probs[i-1] * pow(vect_probs[i-1] - ((2.0*i-1.0)/(2.0*size)),2);
      }
      P *= (1.0/size);
      estimator = P;
      if(estimator>=0.333)printf("BUG  --> Estimator>0.3333 for ASMI --> %f\n",estimator);
   }

   return estimator;
}


//define this as a plug-in
DEFINE_FWK_MODULE(DeDxDiscriminatorProducer);
