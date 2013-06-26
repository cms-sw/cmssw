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
// $Id: DeDxDiscriminatorProducer.cc,v 1.3 2013/02/27 13:28:30 muzaffar Exp $
//
//


// system include files
#include <memory>
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"

#include "RecoTracker/DeDx/plugins/DeDxDiscriminatorProducer.h"
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
   ProbabilityMode     = iConfig.getUntrackedParameter<string>  ("ProbabilityMode"    , "Accumulation");


   MinTrackMomentum    = iConfig.getUntrackedParameter<double>  ("minTrackMomentum"   ,  0.0);
   MaxTrackMomentum    = iConfig.getUntrackedParameter<double>  ("maxTrackMomentum"   ,  99999.0); 
   MinTrackEta         = iConfig.getUntrackedParameter<double>  ("minTrackEta"        , -5.0);
   MaxTrackEta         = iConfig.getUntrackedParameter<double>  ("maxTrackEta"        ,  5.0);
   MaxNrStrips         = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"        ,  255);
   MinTrackHits        = iConfig.getUntrackedParameter<unsigned>("MinTrackHits"       ,  3);

   shapetest           = iConfig.getParameter<bool>("ShapeTest");
   useCalibration      = iConfig.getParameter<bool>("UseCalibration");
   m_calibrationPath   = iConfig.getParameter<string>("calibrationPath");

   Prob_ChargePath = NULL;
}


DeDxDiscriminatorProducer::~DeDxDiscriminatorProducer(){}

// ------------ method called once each job just before starting event loop  ------------
void  DeDxDiscriminatorProducer::beginRun(edm::Run const& run, const edm::EventSetup& iSetup)
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
//      printf("The reccord %s is not known by the DeDxDiscriminatorProducer\n", Reccord.c_str());
//      printf("Program will exit now\n");
      exit(0);
   }
   DeDxMap_ = *DeDxMapHandle_.product();

   double xmin = DeDxMap_.rangeX().min;
   double xmax = DeDxMap_.rangeX().max;
   double ymin = DeDxMap_.rangeY().min;
   double ymax = DeDxMap_.rangeY().max;
   double zmin = DeDxMap_.rangeZ().min;
   double zmax = DeDxMap_.rangeZ().max;

   if(Prob_ChargePath)delete Prob_ChargePath;
   Prob_ChargePath  = new TH3D ("Prob_ChargePath"     , "Prob_ChargePath" , DeDxMap_.numberOfBinsX(), xmin, xmax, DeDxMap_.numberOfBinsY() , ymin, ymax, DeDxMap_.numberOfBinsZ(), zmin, zmax);

   

   if(strcmp(ProbabilityMode.c_str(),"Accumulation")==0){
//      printf("LOOOP ON P\n");
      for(int i=0;i<=Prob_ChargePath->GetXaxis()->GetNbins()+1;i++){
//         printf("LOOOP ON PATH\n");
         for(int j=0;j<=Prob_ChargePath->GetYaxis()->GetNbins()+1;j++){
//            printf("LOOOP ON CHARGE\n");

            double Ni = 0;
            for(int k=0;k<=Prob_ChargePath->GetZaxis()->GetNbins()+1;k++){ Ni+=DeDxMap_.binContent(i,j,k);} 

            for(int k=0;k<=Prob_ChargePath->GetZaxis()->GetNbins()+1;k++){
               double tmp = 0;
               for(int l=0;l<=k;l++){ tmp+=DeDxMap_.binContent(i,j,l);}

      	       if(Ni>0){
                  Prob_ChargePath->SetBinContent (i, j, k, tmp/Ni);
// 	          printf("P=%6.2f Path=%6.2f Charge%8.2f --> Prob=%8.3f\n",Prob_ChargePath->GetXaxis()->GetBinCenter(i), Prob_ChargePath->GetYaxis()->GetBinCenter(j), Prob_ChargePath->GetZaxis()->GetBinCenter(k), tmp/Ni);
  	       }else{
                  Prob_ChargePath->SetBinContent (i, j, k, 0);
	       }
            }
         }
      }
   }else if(strcmp(ProbabilityMode.c_str(),"Integral")==0){
//      printf("LOOOP ON P\n");
      for(int i=0;i<=Prob_ChargePath->GetXaxis()->GetNbins()+1;i++){
//         printf("LOOOP ON PATH\n");
         for(int j=0;j<=Prob_ChargePath->GetYaxis()->GetNbins()+1;j++){
//            printf("LOOOP ON CHARGE\n");

            double Ni = 0;
            for(int k=0;k<=Prob_ChargePath->GetZaxis()->GetNbins()+1;k++){ Ni+=DeDxMap_.binContent(i,j,k);}

            for(int k=0;k<=Prob_ChargePath->GetZaxis()->GetNbins()+1;k++){
               double tmp = DeDxMap_.binContent(i,j,k);

               if(Ni>0){
                  Prob_ChargePath->SetBinContent (i, j, k, tmp/Ni);
//                  printf("P=%6.2f Path=%6.2f Charge%8.2f --> Prob=%8.3f\n",Prob_ChargePath->GetXaxis()->GetBinCenter(i), Prob_ChargePath->GetYaxis()->GetBinCenter(j), Prob_ChargePath->GetZaxis()->GetBinCenter(k), tmp/Ni);
               }else{
                  Prob_ChargePath->SetBinContent (i, j, k, 0);
               }
            }
         }
      }   
   }else{
//      printf("The ProbabilityMode: %s is unknown\n",ProbabilityMode.c_str());
//      printf("The program will stop now\n");
      exit(0);
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



   if(MODsColl.size()==0){
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
 
      MakeCalibrationMap();
   }

}

void  DeDxDiscriminatorProducer::endJob()
{
   MODsColl.clear();
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

   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
   m_tracker = tkGeom.product();
 
   std::vector<DeDxData> dEdxDiscrims( TrajToTrackMap.size() );

   unsigned track_index = 0;
   for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap.begin(); it!=TrajToTrackMap.end(); ++it, track_index++) {
      dEdxDiscrims[track_index] = DeDxData(-1, -2, 0 );

      const Track      track = *it->val;
      const Trajectory traj  = *it->key;

      if(track.eta()  <MinTrackEta      || track.eta()>MaxTrackEta     ){continue;}
      if(track.p()    <MinTrackMomentum || track.p()  >MaxTrackMomentum){continue;}
      if(track.found()<MinTrackHits                                    ){continue;}

      std::vector<double> vect_probs;
      vector<TrajectoryMeasurement> measurements = traj.measurements();

      unsigned int NClusterSaturating = 0;
      for(vector<TrajectoryMeasurement>::const_iterator measurement_it = measurements.begin(); measurement_it!=measurements.end(); measurement_it++){

         TrajectoryStateOnSurface trajState = measurement_it->updatedState();
         if( !trajState.isValid() ) continue;

         const TrackingRecHit*         hit                 = (*measurement_it->recHit()).hit();
         const SiStripRecHit2D*        sistripsimplehit    = dynamic_cast<const SiStripRecHit2D*>(hit);
         const SiStripMatchedRecHit2D* sistripmatchedhit   = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);
         const SiStripRecHit1D*        sistripsimple1dhit  = dynamic_cast<const SiStripRecHit1D*>(hit);
	 
	 double Prob;
         if(sistripsimplehit){
       		Prob = GetProbability(DeDxTools::GetCluster(sistripsimplehit), trajState,sistripsimplehit->geographicalId());	    
	        if(shapetest && !(DeDxTools::shapeSelection(DeDxTools::GetCluster(sistripsimplehit)->amplitudes()))) Prob=-1.0;
                if(Prob>=0) vect_probs.push_back(Prob);             
            
		if(ClusterSaturatingStrip(DeDxTools::GetCluster(sistripsimplehit),sistripsimplehit->geographicalId())>0)NClusterSaturating++;

         }else if(sistripmatchedhit){
	        Prob = GetProbability(DeDxTools::GetCluster(sistripmatchedhit->monoHit()), trajState, sistripmatchedhit->monoId());
	        if(shapetest && !(DeDxTools::shapeSelection(DeDxTools::GetCluster(sistripmatchedhit->monoHit())->amplitudes()))) Prob=-1.0;
                if(Prob>=0) vect_probs.push_back(Prob);
           
                Prob = GetProbability(DeDxTools::GetCluster(sistripmatchedhit->stereoHit()), trajState,sistripmatchedhit->stereoId());
                if(Prob>=0) vect_probs.push_back(Prob);
            
		if(ClusterSaturatingStrip(DeDxTools::GetCluster(sistripmatchedhit->monoHit()),sistripmatchedhit->monoId())  >0)NClusterSaturating++;
		if(ClusterSaturatingStrip(DeDxTools::GetCluster(sistripmatchedhit->stereoHit()),sistripmatchedhit->stereoId())>0)NClusterSaturating++;
         }else if(sistripsimple1dhit){ 
	        Prob = GetProbability(DeDxTools::GetCluster(sistripsimple1dhit), trajState, sistripsimple1dhit->geographicalId());
	        if(shapetest && !(DeDxTools::shapeSelection(DeDxTools::GetCluster(sistripsimple1dhit)->amplitudes()))) Prob=-1.0;
                if(Prob>=0) vect_probs.push_back(Prob);
		if(ClusterSaturatingStrip(DeDxTools::GetCluster(sistripsimple1dhit),sistripsimple1dhit->geographicalId())>0)NClusterSaturating++;
         }else{
         }
      }

      double estimator          = ComputeDiscriminator(vect_probs);
      int    size               = vect_probs.size();
      float  Error              = -1;

      //WARNING: Since the dEdX Error is not properly computed for the moment
      //It was decided to store the number of saturating cluster in that dataformat
      Error = NClusterSaturating;
      dEdxDiscrims[track_index] = DeDxData(estimator, Error, size );

//      printf("%i --> %g\n",size,estimator);
   }

  filler.insert(trackCollectionHandle, dEdxDiscrims.begin(), dEdxDiscrims.end());
  filler.fill();
  iEvent.put(trackDeDxDiscrimAssociation);
}


int DeDxDiscriminatorProducer::ClusterSaturatingStrip(const SiStripCluster*   cluster,
						      const uint32_t &               detId){
   const vector<uint8_t>&  ampls          = cluster->amplitudes();

   stModInfo* MOD                         = NULL;
   if(useCalibration)MOD                  = MODsColl[detId];

   int SaturatingStrip = 0;
   for(unsigned int i=0;i<ampls.size();i++){
      int StripCharge = ampls[i];
      if(MOD){StripCharge = (int)(StripCharge / MOD->Gain);}
      if(StripCharge>=254)SaturatingStrip++;
   }
   return SaturatingStrip;
}

double DeDxDiscriminatorProducer::GetProbability(const SiStripCluster*   cluster, TrajectoryStateOnSurface trajState,
						 const uint32_t &               detId)
{
   // Get All needed variables
   LocalVector             trackDirection = trajState.localDirection();
   double                  cosine         = trackDirection.z()/trackDirection.mag();
   const vector<uint8_t>&  ampls          = cluster->amplitudes();
   //   uint32_t                detId          = cluster->geographicalId();
   //   int                     firstStrip     = cluster->firstStrip();
   stModInfo* MOD                         = MODsColl[detId];


   // Sanity Checks
   if( ampls.size()>MaxNrStrips)                                                                      {return -1;}
// if( DeDxDiscriminatorTools::IsSaturatingStrip  (ampls))                                            {return -1;}
// if( DeDxDiscriminatorTools::IsSpanningOver2APV (firstStrip, ampls.size()))                         {return -1;}
// if(!DeDxDiscriminatorTools::IsFarFromBorder    (trajState, m_tracker->idToDetUnit(DetId(detId)) )) {return -1;}


   // Find Probability for this given Charge and Path
   double charge = 0;
   if(useCalibration){
      for(unsigned int i=0;i<ampls.size();i++){
         int CalibratedCharge = ampls[i];
         CalibratedCharge = (int)(CalibratedCharge / MOD->Gain);
         if(CalibratedCharge>=1024){
            CalibratedCharge = 255;
         }else if(CalibratedCharge>254){
            CalibratedCharge = 254;
         }
         charge+=CalibratedCharge;
      }
   }else{
      charge = DeDxDiscriminatorTools::charge(ampls);
   }
   double path   = DeDxDiscriminatorTools::path(cosine,MOD->Thickness);

   int    BinX   = Prob_ChargePath->GetXaxis()->FindBin(trajState.localMomentum().mag());
   int    BinY   = Prob_ChargePath->GetYaxis()->FindBin(path);
   int    BinZ   = Prob_ChargePath->GetZaxis()->FindBin(charge/path);
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
      P *= (3.0/size);
      estimator = P;
   }else{
      std::sort(vect_probs.begin(), vect_probs.end(), std::less<double>() );
      double P = 1.0/(12*size);
      for(int i=1;i<=size;i++){
         P += vect_probs[i-1] * pow(vect_probs[i-1] - ((2.0*i-1.0)/(2.0*size)),2);
      }
      P *= (3.0/size);
      estimator = P;
   }

   return estimator;
}


void DeDxDiscriminatorProducer::MakeCalibrationMap(){
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
DEFINE_FWK_MODULE(DeDxDiscriminatorProducer);
