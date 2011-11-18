// -*- C++ -*-
//
// Package:    DeDxDiscriminatorLearnerFromCalibTree
// Class:      DeDxDiscriminatorLearnerFromCalibTree
// 
/**\class DeDxDiscriminatorLearnerFromCalibTree DeDxDiscriminatorLearnerFromCalibTree.cc RecoTracker/DeDxDiscriminatorLearnerFromCalibTree/src/DeDxDiscriminatorLearnerFromCalibTree.cc

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

#include "RecoTracker/DeDx/plugins/DeDxDiscriminatorLearnerFromCalibTree.h"

//#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

using namespace reco;
using namespace std;
using namespace edm;

DeDxDiscriminatorLearnerFromCalibTree::DeDxDiscriminatorLearnerFromCalibTree(const edm::ParameterSet& iConfig) : ConditionDBWriter<PhysicsTools::Calibration::HistogramD3D>(iConfig)
{
std::cout << "TEST 0 " << endl;

   P_Min	       = iConfig.getParameter<double>  ("P_Min"  );
   P_Max               = iConfig.getParameter<double>  ("P_Max"  );
   P_NBins             = iConfig.getParameter<int>     ("P_NBins");
   Path_Min            = iConfig.getParameter<double>  ("Path_Min"  );
   Path_Max            = iConfig.getParameter<double>  ("Path_Max"  );
   Path_NBins          = iConfig.getParameter<int>     ("Path_NBins");
   Charge_Min          = iConfig.getParameter<double>  ("Charge_Min"  );
   Charge_Max          = iConfig.getParameter<double>  ("Charge_Max"  );
   Charge_NBins        = iConfig.getParameter<int>     ("Charge_NBins");

   MinTrackTMomentum   = iConfig.getUntrackedParameter<double>  ("minTrackMomentum"   ,  5.0);
   MaxTrackTMomentum   = iConfig.getUntrackedParameter<double>  ("maxTrackMomentum"   ,  99999.0); 
   MinTrackEta         = iConfig.getUntrackedParameter<double>  ("minTrackEta"        , -5.0);
   MaxTrackEta         = iConfig.getUntrackedParameter<double>  ("maxTrackEta"        ,  5.0);
   MaxNrStrips         = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"        ,  255);
   MinTrackHits        = iConfig.getUntrackedParameter<unsigned>("MinTrackHits"       ,  4);

   HistoFile           = iConfig.getUntrackedParameter<string>  ("HistoFile"        ,  "out.root");

   VInputFiles         = iConfig.getParameter<vector<string> >  ("InputFiles");

std::cout << "TEST 1 " << endl;


   useCalibration      = iConfig.getUntrackedParameter<bool>("UseCalibration", false);
   m_calibrationPath   = iConfig.getUntrackedParameter<string>("calibrationPath");

std::cout << "TEST 2 " << endl;

}


DeDxDiscriminatorLearnerFromCalibTree::~DeDxDiscriminatorLearnerFromCalibTree(){

std::cout << "TEST Z " << endl;
}

// ------------ method called once each job just before starting event loop  ------------

void  DeDxDiscriminatorLearnerFromCalibTree::algoBeginJob(const edm::EventSetup& iSetup)
{
std::cout << "TEST 3 " << endl;

//   Charge_Vs_Path = new TH2F ("Charge_Vs_Path"     , "Charge_Vs_Path" , 24, 0.2, 1.4, 250, 0, 5000);
   Charge_Vs_Path = new TH3F ("Charge_Vs_Path"     , "Charge_Vs_Path" , P_NBins, P_Min, P_Max, Path_NBins, Path_Min, Path_Max, Charge_NBins, Charge_Min, Charge_Max);

std::cout << "TEST A " << endl;

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
          for(unsigned int j=0;j<NAPV;j++){

             stAPVInfo* APV       = new stAPVInfo;
             APV->DetId           = Detid.rawId();
             APV->SubDet          = SubDet;
             APV->Eta             = Eta;
             APV->R               = R;
             APV->Thickness       = Thick;
             APV->APVId           = j;
             APVsColl[APV->DetId<<3 | APV->APVId] = APV;
         }
      }
   }

std::cout << "TEST B " << endl;


   MakeCalibrationMap();
std::cout << "TEST C " << endl;


   algoAnalyzeTheTree();
std::cout << "TEST D " << endl;

}

// ------------ method called once each job just after ending the event loop  ------------


void DeDxDiscriminatorLearnerFromCalibTree::algoEndJob()
{
	TFile* Output = new TFile(HistoFile.c_str(), "RECREATE");
      	Charge_Vs_Path->Write();
	Output->Write();
	Output->Close();
        TFile* Input = new TFile(HistoFile.c_str() );
	Charge_Vs_Path = (TH3F*)(Input->FindObjectAny("Charge_Vs_Path"))->Clone();  
	Input->Close();
}

void DeDxDiscriminatorLearnerFromCalibTree::algoAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}


void DeDxDiscriminatorLearnerFromCalibTree::algoAnalyzeTheTree()
{
   unsigned int NEvent = 0;
   for(unsigned int i=0;i<VInputFiles.size();i++){
      printf("Openning file %3i/%3i --> %s\n",i+1, (int)VInputFiles.size(), (char*)(VInputFiles[i].c_str())); fflush(stdout);
      TChain* tree = new TChain("gainCalibrationTree/tree");
      tree->Add(VInputFiles[i].c_str());

      TString EventPrefix("");
      TString EventSuffix("");

      TString TrackPrefix("track");
      TString TrackSuffix("");

      TString CalibPrefix("GainCalibration");
      TString CalibSuffix("");

      unsigned int                 eventnumber    = 0;    tree->SetBranchAddress(EventPrefix + "event"          + EventSuffix, &eventnumber   , NULL);
      unsigned int                 runnumber      = 0;    tree->SetBranchAddress(EventPrefix + "run"            + EventSuffix, &runnumber     , NULL);
      std::vector<bool>*           TrigTech       = 0;    tree->SetBranchAddress(EventPrefix + "TrigTech"       + EventSuffix, &TrigTech   , NULL);

      std::vector<double>*         trackchi2ndof  = 0;    tree->SetBranchAddress(TrackPrefix + "chi2ndof"       + TrackSuffix, &trackchi2ndof , NULL);
      std::vector<float>*          trackp         = 0;    tree->SetBranchAddress(TrackPrefix + "momentum"       + TrackSuffix, &trackp        , NULL);
      std::vector<float>*          trackpt        = 0;    tree->SetBranchAddress(TrackPrefix + "pt"             + TrackSuffix, &trackpt       , NULL);
      std::vector<double>*         tracketa       = 0;    tree->SetBranchAddress(TrackPrefix + "eta"            + TrackSuffix, &tracketa      , NULL);
      std::vector<double>*         trackphi       = 0;    tree->SetBranchAddress(TrackPrefix + "phi"            + TrackSuffix, &trackphi      , NULL);
      std::vector<unsigned int>*   trackhitsvalid = 0;    tree->SetBranchAddress(TrackPrefix + "hitsvalid"      + TrackSuffix, &trackhitsvalid, NULL);

      std::vector<int>*            trackindex     = 0;    tree->SetBranchAddress(CalibPrefix + "trackindex"     + CalibSuffix, &trackindex    , NULL);
      std::vector<unsigned int>*   rawid          = 0;    tree->SetBranchAddress(CalibPrefix + "rawid"          + CalibSuffix, &rawid         , NULL);
      std::vector<unsigned short>* firststrip     = 0;    tree->SetBranchAddress(CalibPrefix + "firststrip"     + CalibSuffix, &firststrip    , NULL);
      std::vector<unsigned short>* nstrips        = 0;    tree->SetBranchAddress(CalibPrefix + "nstrips"        + CalibSuffix, &nstrips       , NULL);
      std::vector<unsigned int>*   charge         = 0;    tree->SetBranchAddress(CalibPrefix + "charge"         + CalibSuffix, &charge        , NULL);
      std::vector<float>*          path           = 0;    tree->SetBranchAddress(CalibPrefix + "path"           + CalibSuffix, &path          , NULL);
      std::vector<unsigned char>*  amplitude      = 0;    tree->SetBranchAddress(CalibPrefix + "amplitude"      + CalibSuffix, &amplitude     , NULL);
      std::vector<double>*         gainused       = 0;    tree->SetBranchAddress(CalibPrefix + "gainused"       + CalibSuffix, &gainused      , NULL);

      printf("Number of Events = %i + %i = %i\n",NEvent,(unsigned int)tree->GetEntries(),(unsigned int)(NEvent+tree->GetEntries()));NEvent+=tree->GetEntries();
      printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
      printf("Looping on the Tree          :");
      int TreeStep = tree->GetEntries()/50;if(TreeStep<=1)TreeStep=1;
      for (unsigned int ientry = 0; ientry < tree->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
         tree->GetEntry(ientry);

         int FirstAmplitude = 0;
         for(unsigned int c=0;c<(*path).size();c++){
            FirstAmplitude+=(*nstrips)[c];
            int t = (*trackindex)[c];
            if((*trackpt)[t]<5)continue;
            if((*trackhitsvalid)[t]<5)continue;

            int Charge = 0; 
            if(useCalibration){
               stAPVInfo* APV = APVsColl[((*rawid)[c]<<3) | (unsigned int)((*firststrip)[c]/128)];
               for(unsigned int s=0;s<(*nstrips)[c];s++){
                 int StripCharge =  (*amplitude)[FirstAmplitude-(*nstrips)[c]+s];
                 if(StripCharge<254){
                    StripCharge=(int)(StripCharge/APV->CalibGain);
                    if(StripCharge>=1024){
                       StripCharge = 255;
                    }else if(StripCharge>=254){
                       StripCharge = 254;
                    }
                 }
                 Charge += StripCharge;
               }
            }else{
               Charge = (*charge)[c];
            }

//          printf("ChargeDifference = %i Vs %i with Gain = %f\n",(*charge)[c],Charge,Gains[(*rawid)[c]]);
            double ClusterChargeOverPath   =  ( (double) Charge )/(*path)[c] ;       
            Charge_Vs_Path->Fill((*trackp)[t],(*path)[c],ClusterChargeOverPath);
         }
      }printf("\n");
  }
}


PhysicsTools::Calibration::HistogramD3D* DeDxDiscriminatorLearnerFromCalibTree::getNewObject()
{
std::cout << "TEST X " << endl;


//   if( strcmp(algoMode.c_str(),"MultiJob")==0)return NULL;

   PhysicsTools::Calibration::HistogramD3D* obj;
   obj = new PhysicsTools::Calibration::HistogramD3D(
                Charge_Vs_Path->GetNbinsX(), Charge_Vs_Path->GetXaxis()->GetXmin(),  Charge_Vs_Path->GetXaxis()->GetXmax(),
                Charge_Vs_Path->GetNbinsY(), Charge_Vs_Path->GetYaxis()->GetXmin(),  Charge_Vs_Path->GetYaxis()->GetXmax(),
	        Charge_Vs_Path->GetNbinsZ(), Charge_Vs_Path->GetZaxis()->GetXmin(),  Charge_Vs_Path->GetZaxis()->GetXmax());

std::cout << "TEST Y " << endl;


   for(int ix=0; ix<=Charge_Vs_Path->GetNbinsX()+1; ix++){
      for(int iy=0; iy<=Charge_Vs_Path->GetNbinsY()+1; iy++){
         for(int iz=0; iz<=Charge_Vs_Path->GetNbinsZ()+1; iz++){
            obj->setBinContent(ix, iy, iz, Charge_Vs_Path->GetBinContent(ix,iy, iz) );       
//          if(Charge_Vs_Path->GetBinContent(ix,iy)!=0)printf("%i %i %i --> %f\n",ix,iy, iz, Charge_Vs_Path->GetBinContent(ix,iy,iz)); 
         }
      }
   }

std::cout << "TEST W " << endl;

   return obj;
}



void DeDxDiscriminatorLearnerFromCalibTree::MakeCalibrationMap(){
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
       stAPVInfo* APV = APVsColl[(tree_DetId<<3) | (unsigned int)tree_APVId];
       APV->CalibGain = tree_Gain;
   }

}


//define this as a plug-in
DEFINE_FWK_MODULE(DeDxDiscriminatorLearnerFromCalibTree);
