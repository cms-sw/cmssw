// Original Author:  Loic QUERTENMONT
//         Created:  Wed Feb  6 08:55:18 CET 2008

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"

#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"


#include "TFile.h"
#include "TObjString.h"
#include "TString.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TF1.h"
#include "TROOT.h"

#include <ext/hash_map>



using namespace edm;
using namespace reco;
using namespace std;
using namespace __gnu_cxx;

struct stAPVPairGain{unsigned int Index; int DetId; int APVPairId; int SubDet; float Eta; float R; float Thickness; double MPV; double Gain;};

class SiStripGainFromData : public ConditionDBWriter<SiStripApvGain> {
   public:
      explicit SiStripGainFromData(const edm::ParameterSet&);
      ~SiStripGainFromData();


   private:
      virtual void algoBeginJob(const edm::EventSetup&) ;
      virtual void algoEndJob() ;
      virtual void algoAnalyze(const edm::Event &, const edm::EventSetup &);

      SiStripApvGain* getNewObject();


      double              ComputeChargeOverPath(const SiStripRecHit2D* sistripsimplehit,TrajectoryStateOnSurface trajState, const edm::EventSetup* iSetup, const Track* track, double trajChi2OverN);
      bool                IsFarFromBorder(LocalPoint HitLocalPos, const uint32_t detid, const edm::EventSetup* iSetup);

      void                getPeakOfLandau(TH1* InputHisto, double* FitResults, double LowRange=0, double HighRange=5400);

      const edm::EventSetup* iSetup_;
      const edm::Event*      iEvent_;

      bool         CheckLocalAngle;
      unsigned int MinNrEntries;
      double       MaxMPVError;
      double       MaxChi2OverNDF;
      double       MinTrackMomentum;
      double       MaxTrackMomentum;
      double       MinTrackEta;
      double       MaxTrackEta;
      unsigned int MaxNrStrips;
      unsigned int MinTrackHits;
      double       MaxTrackChiOverNdf;
      bool         AllowSaturation;

      std::string  AlgoMode;
      std::string  OutputGains;
      std::string  OutputHistos;
      std::string  TrajToTrackProducer;
      std::string  TrajToTrackLabel;

      vector<string> VInputFiles;


      TH2D*	   Tracks_P_Vs_Eta;
      TH2D*        Tracks_Pt_Vs_Eta;

      TH1D*        NumberOfEntriesByAPVPair;
      TH1D*        HChi2OverNDF;
      TH1D*        HTrackChi2OverNDF;
      TH1D*        HTrackHits;

      TH2D*        MPV_Vs_EtaTIB;
      TH2D*        MPV_Vs_EtaTID;
      TH2D*        MPV_Vs_EtaTOB;
      TH2D*        MPV_Vs_EtaTEC;
      TH2D*        MPV_Vs_EtaTEC1;
      TH2D*        MPV_Vs_EtaTEC2;

      TH2D*        Charge_Vs_PathTIB;
      TH2D*        Charge_Vs_PathTID;
      TH2D*        Charge_Vs_PathTOB;
      TH2D*        Charge_Vs_PathTEC;
      TH2D*        Charge_Vs_PathTEC1;
      TH2D*        Charge_Vs_PathTEC2;

      TH1D*        MPV_Vs_PathTIB; 
      TH1D*        MPV_Vs_PathTID;
      TH1D*        MPV_Vs_PathTOB;
      TH1D*        MPV_Vs_PathTEC;
      TH1D*        MPV_Vs_PathTEC1;
      TH1D*        MPV_Vs_PathTEC2;

      TH2D*        MPV_Vs_Eta;
      TH2D*        MPV_Vs_R;

      TH2D*        PD_Vs_Eta;
      TH2D*        PD_Vs_R;

      TH1D*        APV_DetId;
      TH1D*        APV_PairId;
      TH1D*        APV_Eta;
      TH1D*        APV_R;
      TH1D*        APV_SubDet;
      TH2D*        APV_Momentum;
      TH2D*        APV_Charge;
      TH1D*        APV_MPV;
      TH1D*        APV_Gain;
      TH1D*        APV_Thickness;

      TH1D*        MPVs;


      TH1D*        NHighStripInCluster;
      TH2D*        Charge_Vs_PathLength_Sat;
      TH2D*        Charge_Vs_PathLength_NoSat;

      TH2D*        Charge_Vs_PathLength;
      TH2D*        Charge_Vs_PathLength320;
      TH2D*        Charge_Vs_PathLength500;

      TH1D*        MPV_Vs_PathLength;
      TH1D*        MPV_Vs_PathLength320;
      TH1D*        MPV_Vs_PathLength500;

      TH1D*        FWHM_Vs_PathLength;
      TH1D*        FWHM_Vs_PathLength320;
      TH1D*        FWHM_Vs_PathLength500;


      TH2D*        Charge_Vs_TransversAngle;
      TH1D*        MPV_Vs_TransversAngle;
      TH2D*        NStrips_Vs_TransversAngle;

      TH2D*        Charge_Vs_Alpha;
      TH1D*        MPV_Vs_Alpha;
      TH2D*        NStrips_Vs_Alpha;

      TH2D*        Charge_Vs_Beta;
      TH1D*        MPV_Vs_Beta;
      TH2D*        NStrips_Vs_Beta;

      TH2D*        MPV_Vs_Error;
      TH2D*        Entries_Vs_Error;

      TH2D*        HitLocalPosition;
      TH2D*        HitLocalPositionBefCut;

      TFile*       Output;
      unsigned int NEvent;     

   private :
      class isEqual{
         public:
		 template <class T> bool operator () (const T& PseudoDetId1, const T& PseudoDetId2) { return PseudoDetId1==PseudoDetId2; }
      };

      std::vector<stAPVPairGain*> APVsCollOrdered;
      hash_map<unsigned int, stAPVPairGain*,  hash<unsigned int>, isEqual > APVsColl;
};

SiStripGainFromData::SiStripGainFromData(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripApvGain>::ConditionDBWriter<SiStripApvGain>(iConfig)
{
   AlgoMode            = iConfig.getParameter<std::string>("AlgoMode");

   OutputGains         = iConfig.getParameter<std::string>("OutputGains");
   OutputHistos        = iConfig.getParameter<std::string>("OutputHistos");

   TrajToTrackProducer = iConfig.getParameter<std::string>("TrajToTrackProducer");
   TrajToTrackLabel    = iConfig.getParameter<std::string>("TrajToTrackLabel");

   CheckLocalAngle     = iConfig.getUntrackedParameter<bool>    ("checkLocalAngle"   ,  false);
   MinNrEntries        = iConfig.getUntrackedParameter<unsigned>("minNrEntries"      ,  20);
   MaxMPVError         = iConfig.getUntrackedParameter<double>  ("maxMPVError"       ,  500.0);
   MaxChi2OverNDF      = iConfig.getUntrackedParameter<double>  ("maxChi2OverNDF"    ,  5.0);
   MinTrackMomentum    = iConfig.getUntrackedParameter<double>  ("minTrackMomentum"  ,  3.0);
   MaxTrackMomentum    = iConfig.getUntrackedParameter<double>  ("maxTrackMomentum"  ,  99999.0);
   MinTrackEta         = iConfig.getUntrackedParameter<double>  ("minTrackEta"       , -5.0);
   MaxTrackEta         = iConfig.getUntrackedParameter<double>  ("maxTrackEta"       ,  5.0);
   MaxNrStrips         = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"       ,  2);
   MinTrackHits        = iConfig.getUntrackedParameter<unsigned>("MinTrackHits"      ,  8);
   MaxTrackChiOverNdf  = iConfig.getUntrackedParameter<double>  ("MaxTrackChiOverNdf",  3);
   AllowSaturation     = iConfig.getUntrackedParameter<bool>    ("AllowSaturation"   ,  false);

   if( strcmp(AlgoMode.c_str(),"WriteOnDB")==0 )
   VInputFiles         = iConfig.getParameter<vector<string> >("VInputFiles");
}


SiStripGainFromData::~SiStripGainFromData()
{ 
}


void
SiStripGainFromData::algoBeginJob(const edm::EventSetup& iSetup)
{
   iSetup_                  = &iSetup;

   TH1::AddDirectory(kTRUE);
   Output                     = new TFile(OutputHistos.c_str(), "RECREATE");

   Tracks_P_Vs_Eta            = new TH2D ("Tracks_P_Vs_Eta"   , "Tracks_P_Vs_Eta" , 60, 0,3,500,0,100);
   Tracks_Pt_Vs_Eta           = new TH2D ("Tracks_Pt_Vs_Eta"  , "Tracks_Pt_Vs_Eta", 60, 0,3,500,0,100);

   MPV_Vs_EtaTIB              = new TH2D ("MPV_Vs_EtaTIB"     , "MPV_Vs_EtaTIB" , 50, -3.0, 3.0, 1350, 0, 1350);
   MPV_Vs_EtaTID              = new TH2D ("MPV_Vs_EtaTID"     , "MPV_Vs_EtaTID" , 50, -3.0, 3.0, 1350, 0, 1350);
   MPV_Vs_EtaTOB              = new TH2D ("MPV_Vs_EtaTOB"     , "MPV_Vs_EtaTOB" , 50, -3.0, 3.0, 1350, 0, 1350);
   MPV_Vs_EtaTEC              = new TH2D ("MPV_Vs_EtaTEC"     , "MPV_Vs_EtaTEC" , 50, -3.0, 3.0, 1350, 0, 1350);
   MPV_Vs_EtaTEC1             = new TH2D ("MPV_Vs_EtaTEC1"    , "MPV_Vs_EtaTEC1", 50, -3.0, 3.0, 1350, 0, 1350);
   MPV_Vs_EtaTEC2             = new TH2D ("MPV_Vs_EtaTEC2"    , "MPV_Vs_EtaTEC2", 50, -3.0, 3.0, 1350, 0, 1350);

   Charge_Vs_PathTIB          = new TH2D ("Charge_Vs_PathTIB" , "Charge_Vs_PathTIB" ,1000,0.2,1.4, 1000,0,2000);
   Charge_Vs_PathTID          = new TH2D ("Charge_Vs_PathTID" , "Charge_Vs_PathTID" ,1000,0.2,1.4, 1000,0,2000);
   Charge_Vs_PathTOB          = new TH2D ("Charge_Vs_PathTOB" , "Charge_Vs_PathTOB" ,1000,0.2,1.4, 1000,0,2000);
   Charge_Vs_PathTEC          = new TH2D ("Charge_Vs_PathTEC" , "Charge_Vs_PathTEC" ,1000,0.2,1.4, 1000,0,2000);
   Charge_Vs_PathTEC1         = new TH2D ("Charge_Vs_PathTEC1", "Charge_Vs_PathTEC1",1000,0.2,1.4, 1000,0,2000);
   Charge_Vs_PathTEC2         = new TH2D ("Charge_Vs_PathTEC2", "Charge_Vs_PathTEC2",1000,0.2,1.4, 1000,0,2000);

   MPV_Vs_PathTIB             = new TH1D ("MPV_Vs_PathTIB"    , "MPV_Vs_PathTIB"    ,1000,0.2,1.4);
   MPV_Vs_PathTID             = new TH1D ("MPV_Vs_PathTID"    , "MPV_Vs_PathTID"    ,1000,0.2,1.4);
   MPV_Vs_PathTOB             = new TH1D ("MPV_Vs_PathTOB"    , "MPV_Vs_PathTOB"    ,1000,0.2,1.4);
   MPV_Vs_PathTEC             = new TH1D ("MPV_Vs_PathTEC"    , "MPV_Vs_PathTEC"    ,1000,0.2,1.4);
   MPV_Vs_PathTEC1            = new TH1D ("MPV_Vs_PathTEC1"   , "MPV_Vs_PathTEC1"   ,1000,0.2,1.4);
   MPV_Vs_PathTEC2            = new TH1D ("MPV_Vs_PathTEC2"   , "MPV_Vs_PathTEC2"   ,1000,0.2,1.4);

   MPV_Vs_Eta                 = new TH2D ("MPV_Vs_Eta", "MPV_Vs_Eta", 50, -3.0, 3.0, 1350, 0, 1350);
   MPV_Vs_R                   = new TH2D ("MPV_Vs_R"  , "MPV_Vs_R"  , 150, 0.0, 150.0, 1350, 0, 1350);
   
   PD_Vs_Eta                  = new TH2D ("PD_Vs_Eta", "PD_Vs_Eta", 50, -3.0, 3.0, 500, 0, 100 );
   PD_Vs_R                    = new TH2D ("PD_Vs_R"  , "PD_Vs_R"  , 150, 0.0, 150.0, 500, 0, 100);

   NHighStripInCluster        = new TH1D ("NHighStripInCluster"        , "NHighStripInCluster"         ,15,0,14);
   Charge_Vs_PathLength_Sat   = new TH2D ("Charge_Vs_PathLength_Sat"   , "Charge_Vs_PathLength_Sat"    ,1000,0.2,1.4, 1000,0,2000);
   Charge_Vs_PathLength_NoSat = new TH2D ("Charge_Vs_PathLength_NoSat" , "Charge_Vs_PathLength_NoSat"  ,1000,0.2,1.4, 1000,0,2000);

   Charge_Vs_PathLength       = new TH2D ("Charge_Vs_PathLength"    , "Charge_Vs_PathLength"  ,1000,0.2,1.4, 1000,0,2000);
   Charge_Vs_PathLength320    = new TH2D ("Charge_Vs_PathLength320" , "Charge_Vs_PathLength"  ,1000,0.2,1.4, 1000,0,2000);
   Charge_Vs_PathLength500    = new TH2D ("Charge_Vs_PathLength500" , "Charge_Vs_PathLength"  ,1000,0.2,1.4, 1000,0,2000);

   MPV_Vs_PathLength          = new TH1D ("MPV_Vs_PathLength"       , "MPV_Vs_PathLength"  ,1000,0.2,1.4);
   MPV_Vs_PathLength320       = new TH1D ("MPV_Vs_PathLength320"    , "MPV_Vs_PathLength"  ,1000,0.2,1.4);
   MPV_Vs_PathLength500       = new TH1D ("MPV_Vs_PathLength500"    , "MPV_Vs_PathLength"  ,1000,0.2,1.4);

   FWHM_Vs_PathLength         = new TH1D ("FWHM_Vs_PathLength"      , "FWHM_Vs_PathLength"  ,1000,0.2,1.4);
   FWHM_Vs_PathLength320      = new TH1D ("FWHM_Vs_PathLength320"   , "FWHM_Vs_PathLength"  ,1000,0.2,1.4);
   FWHM_Vs_PathLength500      = new TH1D ("FWHM_Vs_PathLength500"   , "FWHM_Vs_PathLength"  ,1000,0.2,1.4);


   Charge_Vs_TransversAngle   = new TH2D ("Charge_Vs_TransversAngle" , "Charge_Vs_TransversAngle" ,220,-20,200, 1000,0,2000);
   MPV_Vs_TransversAngle      = new TH1D ("MPV_Vs_TransversAngle"    , "MPV_Vs_TransversAngle"    ,220,-20,200);
   NStrips_Vs_TransversAngle  = new TH2D ("NStrips_Vs_TransversAngle", "NStrips_Vs_TransversAngle",220,-20,200, 50,0,10);

   Charge_Vs_Alpha            = new TH2D ("Charge_Vs_Alpha"  , "Charge_Vs_Alpha"   ,220 ,-20,200, 1000,0,2000);
   MPV_Vs_Alpha               = new TH1D ("MPV_Vs_Alpha"     , "MPV_Vs_Alpha"      ,220 ,-20,200);
   NStrips_Vs_Alpha           = new TH2D ("NStrips_Vs_Alpha" , "NStrips_Vs_Alpha"  ,220 ,-20,200, 50,0,10);

   Charge_Vs_Beta             = new TH2D ("Charge_Vs_Beta"   , "Charge_Vs_Beta"    ,220,-20,200, 1000,0,2000);
   MPV_Vs_Beta                = new TH1D ("MPV_Vs_Beta"      , "MPV_Vs_Beta"       ,220,-20,200);
   NStrips_Vs_Beta            = new TH2D ("NStrips_Vs_Beta"  , "NStrips_Vs_Beta"   ,220,-20,200, 50,0,10);

   MPV_Vs_Error               = new TH2D ("MPV_Vs_Error"   , "MPV_Vs_Error"  ,1350, 0, 1350, 1000,0,100);
   Entries_Vs_Error           = new TH2D ("Entries_Vs_Error","Entries_Vs_Error",2000,0,2000,1000,0,100); 

   NumberOfEntriesByAPVPair   = new TH1D ("NumberOfEntriesByAPVPair", "NumberOfEntriesByAPVPair", 2500, 0,2500);
   HChi2OverNDF               = new TH1D ("Chi2OverNDF","Chi2OverNDF", 5000, 0,10);
   HTrackChi2OverNDF          = new TH1D ("TrackChi2OverNDF","TrackChi2OverNDF", 5000, 0,10);
   HTrackHits                 = new TH1D ("TrackHits","TrackHits", 40, 0,40);

   APV_DetId                  = new TH1D ("APV_DetId"    , "APV_DetId"   , 36393,0,36392);
   APV_PairId                 = new TH1D ("APV_PairId"   , "APV_PairId"  , 36393,0,36392);
   APV_Eta                    = new TH1D ("APV_Eta"      , "APV_Eta"     , 36393,0,36392);
   APV_R                      = new TH1D ("APV_R"        , "APV_R"       , 36393,0,36392);
   APV_SubDet                 = new TH1D ("APV_SubDet"   , "APV_SubDet"  , 36393,0,36392);
   APV_Momentum               = new TH2D ("APV_Momentum" , "APV_Momentum", 36393,0,36392, 200,0,100);
   APV_Charge                 = new TH2D ("APV_Charge"   , "APV_Charge"  , 36393,0,36392, 1000,0,2000);
   APV_MPV                    = new TH1D ("APV_MPV"      , "APV_MPV"     , 36393,0,36392);
   APV_Gain                   = new TH1D ("APV_Gain"     , "APV_Gain"    , 36393,0,36392);
   APV_Thickness              = new TH1D ("APV_Thickness", "APV_Thicknes", 36393,0,36392);

   MPVs                       = new TH1D ("MPVs", "MPVs", 600,0,600);
   
   gROOT->cd();

   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
   vector<GeomDet*> Det = tkGeom->dets();

   unsigned int Id=0;
   for(unsigned int i=0;i<Det.size();i++){
      DetId  Detid  = Det[i]->geographicalId(); 
      int    SubDet = Detid.subdetId();

      if( SubDet == StripSubdetector::TIB ||  SubDet == StripSubdetector::TID ||
          SubDet == StripSubdetector::TOB ||  SubDet == StripSubdetector::TEC  ){

          StripGeomDetUnit* DetUnit     = dynamic_cast<StripGeomDetUnit*> (Det[i]);
	  if(!DetUnit)continue;

          const StripTopology& Topo     = DetUnit->specificTopology();	
          unsigned int         NAPVPair = Topo.nstrips()/256;
	
          double Eta   = DetUnit->position().basicVector().eta();
          double R     = DetUnit->position().basicVector().transverse();
          double Thick = DetUnit->surface().bounds().thickness();

          for(unsigned int j=0;j<NAPVPair;j++){
                stAPVPairGain* APV = new stAPVPairGain;
                APV->Index         = Id;
                APV->DetId         = Detid.rawId();
                APV->APVPairId     = j;
                APV->SubDet        = SubDet;
                APV->MPV           = -1;
                APV->Gain          = -1;
                APV->Eta           = Eta;
                APV->R             = R;
                APV->Thickness     = Thick;
                APVsCollOrdered.push_back(APV);
		APVsColl[(APV->DetId<<2) | APV->APVPairId] = APV;
                Id++;

                APV_DetId    ->Fill(Id,APV->DetId);
                APV_PairId   ->Fill(Id,APV->APVPairId);
                APV_Eta      ->Fill(Id,APV->Eta);
                APV_R        ->Fill(Id,APV->R);
                APV_SubDet   ->Fill(Id,APV->SubDet);
		APV_Thickness->Fill(Id,APV->Thickness);
          }
      }
   }
   NEvent = 0;
}

void 
SiStripGainFromData::algoEndJob() {
   unsigned int I=0;

/*
   if( strcmp(AlgoMode.c_str(),"WriteOnDB")==0 ){
      TFile* file = NULL;
      for(unsigned int f=0;f<VInputFiles.size();f++){
         if(file!=NULL){
            printf("Delete Previous File\n");
//	    delete file;
//          file->Close();
            printf("Delete Previous File end\n");
         }
         file =  new TFile( VInputFiles[f].c_str() ); if(file==NULL){printf("Bug With File %s\n",VInputFiles[f].c_str()); exit(0);}


         for(map<uint32_t, GlobalVector*>::iterator it = ModulePositionMap.begin();it!=ModulePositionMap.end();it++){
         if( I%364==0 ) printf("Merging Histograms \t %6.2f%%\n",(100.0*I) / (ModulePositionMap.size()*VInputFiles.size()) );I++;
         for(unsigned int i=0;i<3;i++){
            int detId   = it->first;
            int APVPair = i;

            TString HistoName      = Form("ChargeAPVPair %i %i",detId,APVPair);
            TH1F*   FullHisto      = (TH1F*) FindHisto(HistoName.Data());
            if(FullHisto==NULL)continue;
            TH1F*   Histo          = (TH1F*) file->FindObjectAny(HistoName.Data());
            if(Histo==NULL){printf("BUG_MERGING\n");continue;}
            FullHisto->Add(Histo);
         }}  

      }
   }
*/



   I=0;
   for(hash_map<unsigned int, stAPVPairGain*,  hash<unsigned int>, isEqual >::iterator it = APVsColl.begin();it!=APVsColl.end();it++){
   if( I%1825==0 ) printf("Fitting Histograms \t %6.2f%%\n",(100.0*I)/APVsColl.size());I++;
      stAPVPairGain* APV = it->second;

      TH1D* PointerToHisto = APV_Charge->ProjectionY(" ",APV->Index-1,APV->Index,"e");
      if(PointerToHisto==NULL)continue;

      double* FitResults = new double[5];
      getPeakOfLandau(PointerToHisto,FitResults);
      APV->MPV = FitResults[0];
      if(FitResults[0]!=-0.5 && FitResults[1]<MaxMPVError){
         APV_MPV->Fill(APV->Index,APV->MPV);
         MPVs   ->Fill(APV->MPV);

         MPV_Vs_R->Fill(APV->R,APV->MPV);
         MPV_Vs_Eta->Fill(APV->Eta,APV->MPV);
         if(APV->SubDet==StripSubdetector::TIB)  MPV_Vs_EtaTIB ->Fill(APV->Eta,APV->MPV);
         if(APV->SubDet==StripSubdetector::TID)  MPV_Vs_EtaTID ->Fill(APV->Eta,APV->MPV);
         if(APV->SubDet==StripSubdetector::TOB)  MPV_Vs_EtaTOB ->Fill(APV->Eta,APV->MPV);
         if(APV->SubDet==StripSubdetector::TEC){ MPV_Vs_EtaTEC ->Fill(APV->Eta,APV->MPV);
         if(APV->Thickness<0.04)		 MPV_Vs_EtaTEC1->Fill(APV->Eta,APV->MPV);
         if(APV->Thickness>0.04)                 MPV_Vs_EtaTEC2->Fill(APV->Eta,APV->MPV);
         }
      }

      if(FitResults[0]!=-0.5){
         HChi2OverNDF->Fill(FitResults[4]);
         MPV_Vs_Error->Fill(FitResults[0],FitResults[1]);
         Entries_Vs_Error->Fill(PointerToHisto->GetEntries(),FitResults[1]);
      }
      NumberOfEntriesByAPVPair->Fill(PointerToHisto->GetEntries());

      delete PointerToHisto;
   }

   cout << "F1" << endl;

   double MPVmean = MPVs->GetMean();
   for(hash_map<unsigned int, stAPVPairGain*,  hash<unsigned int>, isEqual >::iterator it = APVsColl.begin();it!=APVsColl.end();it++){
      stAPVPairGain* APV = it->second;
      if(APV->MPV>0) APV->Gain = MPVmean / APV->MPV;
      else           APV->Gain = 1;     
      APV_Gain->Fill(APV->Index,APV->Gain); 
   }

   cout << "F2" << endl;


   double* FitResults = new double[5]; TH1D* Proj;
   for(int j=0;j<Charge_Vs_PathLength->GetXaxis()->GetNbins();j++){
      Proj      = Charge_Vs_PathLength->ProjectionY(" ",j-1,j,"e");
      getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
      MPV_Vs_PathLength->SetBinContent (j, FitResults[0]/Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength->SetBinError   (j, FitResults[1]/Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j));
      FWHM_Vs_PathLength->SetBinContent(j, FitResults[2]/(FitResults[0]/Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j)) );
      FWHM_Vs_PathLength->SetBinError  (j, FitResults[3]/(FitResults[0]/Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j)) );
      delete Proj;
   }

   cout << "F3" << endl;


   for(int j=0;j<Charge_Vs_PathLength320->GetXaxis()->GetNbins();j++){
      Proj      = Charge_Vs_PathLength320->ProjectionY(" ",j-1,j,"e");
      getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
      MPV_Vs_PathLength320->SetBinContent (j, FitResults[0]/Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength320->SetBinError   (j, FitResults[1]/Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j));
      FWHM_Vs_PathLength320->SetBinContent(j, FitResults[2]/(FitResults[0]/Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j)) );
      FWHM_Vs_PathLength320->SetBinError  (j, FitResults[3]/(FitResults[0]/Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j)) );
      delete Proj;
   }

   cout << "F4" << endl;


   for(int j=0;j<Charge_Vs_PathLength500->GetXaxis()->GetNbins();j++){
      Proj      = Charge_Vs_PathLength500->ProjectionY(" ",j-1,j,"e");
      getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;    
      MPV_Vs_PathLength500->SetBinContent (j, FitResults[0]/Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength500->SetBinError   (j, FitResults[1]/Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j));
      FWHM_Vs_PathLength500->SetBinContent(j, FitResults[2]/(FitResults[0]/Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j) ));
      FWHM_Vs_PathLength500->SetBinError  (j, FitResults[3]/(FitResults[0]/Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j) ));
      delete Proj;
   }

   cout << "F5" << endl;


   for(int j=0;j<Charge_Vs_PathTIB->GetXaxis()->GetNbins();j++){
      Proj      = Charge_Vs_PathTIB->ProjectionY(" ",j-1,j,"e");
      getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
      MPV_Vs_PathTIB->SetBinContent(j, FitResults[0]/Charge_Vs_PathTIB->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathTIB->SetBinError  (j, FitResults[1]/Charge_Vs_PathTIB->GetXaxis()->GetBinCenter(j));
      delete Proj;
   }

   cout << "F6" << endl;

   for(int j=0;j<Charge_Vs_PathTID->GetXaxis()->GetNbins();j++){
      Proj      = Charge_Vs_PathTID->ProjectionY(" ",j-1,j,"e");
      getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
      MPV_Vs_PathTID->SetBinContent(j, FitResults[0]/Charge_Vs_PathTID->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathTID->SetBinError  (j, FitResults[1]/Charge_Vs_PathTID->GetXaxis()->GetBinCenter(j));
      delete Proj;
   }

   cout << "F7" << endl;

   for(int j=0;j<Charge_Vs_PathTOB->GetXaxis()->GetNbins();j++){
      Proj      = Charge_Vs_PathTOB->ProjectionY(" ",j-1,j,"e");
      getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
      MPV_Vs_PathTOB->SetBinContent(j, FitResults[0]/Charge_Vs_PathTOB->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathTOB->SetBinError  (j, FitResults[1]/Charge_Vs_PathTOB->GetXaxis()->GetBinCenter(j));
      delete Proj;
   }

   cout << "F8" << endl;

   for(int j=0;j<Charge_Vs_PathTEC->GetXaxis()->GetNbins();j++){
      Proj      = Charge_Vs_PathTEC->ProjectionY(" ",j-1,j,"e");
      getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
      MPV_Vs_PathTEC->SetBinContent(j, FitResults[0]/Charge_Vs_PathTEC->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathTEC->SetBinError  (j, FitResults[1]/Charge_Vs_PathTEC->GetXaxis()->GetBinCenter(j));
      delete Proj;
   }

   cout << "F9" << endl;

   for(int j=0;j<Charge_Vs_PathTEC1->GetXaxis()->GetNbins();j++){
      Proj      = Charge_Vs_PathTEC1->ProjectionY(" ",j-1,j,"e");
      getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
      MPV_Vs_PathTEC1->SetBinContent(j, FitResults[0]/Charge_Vs_PathTEC1->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathTEC1->SetBinError  (j, FitResults[1]/Charge_Vs_PathTEC1->GetXaxis()->GetBinCenter(j));
      delete Proj;
   }

   for(int j=0;j<Charge_Vs_PathTEC2->GetXaxis()->GetNbins();j++){
      Proj      = Charge_Vs_PathTEC2->ProjectionY(" ",j-1,j,"e");
      getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
      MPV_Vs_PathTEC2->SetBinContent(j, FitResults[0]/Charge_Vs_PathTEC2->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathTEC2->SetBinError  (j, FitResults[1]/Charge_Vs_PathTEC2->GetXaxis()->GetBinCenter(j));
      delete Proj;
   }

   cout << "F10" << endl;



   for(int j=1;j<Charge_Vs_TransversAngle->GetXaxis()->GetNbins();j++){
      Proj      = Charge_Vs_TransversAngle->ProjectionY(" ",j-1,j,"e");
      getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
      MPV_Vs_TransversAngle->SetBinContent(j, FitResults[0]);
      MPV_Vs_TransversAngle->SetBinError  (j, FitResults[1]);
      delete Proj;
   }

   cout << "F11" << endl;


   for(int j=1;j<Charge_Vs_Alpha->GetXaxis()->GetNbins();j++){
      Proj      = Charge_Vs_Alpha->ProjectionY(" ",j-1,j,"e");
      getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
      MPV_Vs_Alpha->SetBinContent(j, FitResults[0]);
      MPV_Vs_Alpha->SetBinError  (j, FitResults[1]);
      delete Proj;
   }

   cout << "F12" << endl;


   for(int j=1;j<Charge_Vs_Beta->GetXaxis()->GetNbins();j++){
      Proj      = Charge_Vs_Beta->ProjectionY(" ",j-1,j,"e");
      getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
      MPV_Vs_Beta->SetBinContent(j, FitResults[0]);
      MPV_Vs_Beta->SetBinError  (j, FitResults[1]);
      delete Proj;
   }

   cout << "F13" << endl;


   if( (strcmp(AlgoMode.c_str(),"WriteOnDB")==0 || strcmp(AlgoMode.c_str(),"SingleJob")==0) ){
      FILE* Gains = fopen(OutputGains.c_str(),"w");
      fprintf(Gains,"NEvents = %i\n",NEvent);
      fprintf(Gains,"Number of APVs = %i\n",APVsColl.size());
      for(std::vector<stAPVPairGain*>::iterator it = APVsCollOrdered.begin();it!=APVsCollOrdered.end();it++){
         stAPVPairGain* APV = *it;
         fprintf(Gains,"%i | %i | %f\n", APV->DetId,APV->APVPairId,APV->Gain);
      }
      fclose(Gains);
   }

   cout << "F14" << endl;

   Output->cd();
   TObjString str1 = Form("_NEvents =  %i",NEvent);
   TObjString str2 = Form("_Begin Run =  %i Event = %i",0,0);
   TObjString str3 = Form("_End   Run =  %i Event = %i",9,9);
   str1.Write(); str2.Write(); str3.Write();
   Output->Write();
   cout << "F15" << endl;

   Output->Close();

   cout << "F16" << endl;

}



void
SiStripGainFromData::algoAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if( strcmp(AlgoMode.c_str(),"WriteOnDB")==0 ) return;

   NEvent++;
   iEvent_ = &iEvent;

   Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
   iEvent.getByLabel(TrajToTrackProducer, TrajToTrackLabel, trajTrackAssociationHandle);
   const TrajTrackAssociationCollection TrajToTrackMap = *trajTrackAssociationHandle.product();

  //reverse the track-trajectory map
   map<const Track* ,const Trajectory *> TrackToTrajMap;
   for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap.begin(); it!=TrajToTrackMap.end(); ++it) {
        TrackToTrajMap[&(*it->val)]=  &(*it->key);
   }

   for(map<const Track* ,const Trajectory *>::const_iterator it = TrackToTrajMap.begin(); it!=TrackToTrajMap.end(); ++it) {
      
      const Track      track = *it->first;   
      const Trajectory traj  = *it->second;

      if(track.p()<MinTrackMomentum || track.p()>MaxTrackMomentum || track.eta()<MinTrackEta || track.eta()>MaxTrackEta)  continue;

      Tracks_Pt_Vs_Eta->Fill(fabs(track.eta()),track.pt());
      Tracks_P_Vs_Eta ->Fill(fabs(track.eta()),track.p());

      //BEGIN TO COMPUTE NDOF FOR TRACKS NO IMPLEMENTED BEFORE 200pre3
      int ndof =0;
      const Trajectory::RecHitContainer transRecHits = traj.recHits();
  
      for(Trajectory::RecHitContainer::const_iterator rechit = transRecHits.begin(); rechit != transRecHits.end(); ++rechit)
         if ((*rechit)->isValid()) ndof += (*rechit)->dimension();  
         ndof -= 5;
      //END TO COMPUTE NDOF FOR TRACKS NO IMPLEMENTED BEFORE 200pre3

      HTrackChi2OverNDF->Fill(traj.chiSquared()/ndof);
      if(traj.chiSquared()/ndof>MaxTrackChiOverNdf)continue;

      vector<TrajectoryMeasurement> measurements = traj.measurements();
      HTrackHits->Fill(traj.foundHits());
      if(traj.foundHits()<(int)MinTrackHits)continue;
/*
      //BEGIN TO COMPUTE #MATCHEDRECHIT IN THE TRACK
      int NMatchedHit = 0;
      for(vector<TrajectoryMeasurement>::const_iterator measurement_it = measurements.begin(); measurement_it!=measurements.end(); measurement_it++){
         TrajectoryStateOnSurface trajState = measurement_it->updatedState();
         if( !trajState.isValid() ) continue;
         const TrackingRecHit*         hit               = (*measurement_it->recHit()).hit();
         const SiStripMatchedRecHit2D* sistripmatchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);
	 if(sistripmatchedhit)NMatchedHit++;
//       NMatchedHit++;

      }
      //END TO COMPUTE #MATCHEDRECHIT IN THE TRACK

      if(NMatchedHit<2){
	 printf("NOT ENOUGH MATCHED RECHITS : %i\n",NMatchedHit);
 	 continue;
      }
*/

      for(vector<TrajectoryMeasurement>::const_iterator measurement_it = measurements.begin(); measurement_it!=measurements.end(); measurement_it++){

         TrajectoryStateOnSurface trajState = measurement_it->updatedState();
         if( !trajState.isValid() ) continue;     

         const TrackingRecHit*         hit               = (*measurement_it->recHit()).hit();
         const SiStripRecHit2D*        sistripsimplehit  = dynamic_cast<const SiStripRecHit2D*>(hit);
         const SiStripMatchedRecHit2D* sistripmatchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);

	 if(sistripsimplehit)
	 {
	     ComputeChargeOverPath(sistripsimplehit, trajState, &iSetup, &track, traj.chiSquared()/ndof);
	 }else if(sistripmatchedhit){
             ComputeChargeOverPath(sistripmatchedhit->monoHit()  ,trajState, &iSetup, &track, traj.chiSquared()/ndof); 
             ComputeChargeOverPath(sistripmatchedhit->stereoHit(),trajState, &iSetup, &track, traj.chiSquared()/ndof);
	 }else{		
         }

      }

   }
}

double
SiStripGainFromData::ComputeChargeOverPath(const SiStripRecHit2D* sistripsimplehit,TrajectoryStateOnSurface trajState, const edm::EventSetup* iSetup,  const Track* track, double trajChi2OverN)
{
   LocalVector          trackDirection = trajState.localDirection();
   double                  cosine      = trackDirection.z()/trackDirection.mag();
   const SiStripCluster*   Cluster     = (sistripsimplehit->cluster()).get();
   const vector<uint16_t>& Ampls       = Cluster->amplitudes();
// const vector<uint8_t>&  Ampls       = Cluster->amplitudes();
   uint32_t                DetId       = Cluster->geographicalId();
   int                     FirstStrip  = Cluster->firstStrip();
   int                     APVPairId   = FirstStrip/256;
   stAPVPairGain*          APV         = APVsColl[(DetId<<2) | APVPairId];
   bool                    Saturation  = false;
   bool                    Overlaping  = false;
   int                     Charge      = 0;
   unsigned int            NHighStrip  = 0;

   if(!IsFarFromBorder(trajState.localPosition(),DetId, iSetup))return -1;

   if(FirstStrip==0                                 )Overlaping=true;
   if(FirstStrip<=255 && FirstStrip+Ampls.size()>255)Overlaping=true;
   if(FirstStrip<=511 && FirstStrip+Ampls.size()>511)Overlaping=true;
   if(FirstStrip+Ampls.size()==511                  )Overlaping=true;
   if(FirstStrip+Ampls.size()==767                  )Overlaping=true;
   if(Overlaping)return -1;

   for(unsigned int a=0;a<Ampls.size();a++){Charge+=Ampls[a];if(Ampls[a]>=254)Saturation=true;if(Ampls[a]>=20)NHighStrip++;}
   double path                    = (10.0*APV->Thickness)/fabs(cosine);
   double ClusterChargeOverPath   = (double)Charge / path ;

   if(Ampls.size()>MaxNrStrips)      return -1;
   if(Saturation && !AllowSaturation)return -1;
                                           Charge_Vs_PathLength   ->Fill(path,Charge);
   if(APV->Thickness<0.04)                 Charge_Vs_PathLength320->Fill(path,Charge);
   if(APV->Thickness>0.04)                 Charge_Vs_PathLength500->Fill(path,Charge);
   if(APV->SubDet==StripSubdetector::TIB)  Charge_Vs_PathTIB      ->Fill(path,Charge);
   if(APV->SubDet==StripSubdetector::TID)  Charge_Vs_PathTID      ->Fill(path,Charge);
   if(APV->SubDet==StripSubdetector::TOB)  Charge_Vs_PathTOB      ->Fill(path,Charge);
   if(APV->SubDet==StripSubdetector::TEC){ Charge_Vs_PathTEC      ->Fill(path,Charge);
   if(APV->Thickness<0.04)                 Charge_Vs_PathTEC1     ->Fill(path,Charge);
   if(APV->Thickness>0.04)                 Charge_Vs_PathTEC2     ->Fill(path,Charge); }   

   double trans = atan2(trackDirection.y(),trackDirection.x())*(180/3.14159265);
   double alpha = acos(trackDirection.x() / sqrt( pow(trackDirection.x(),2) +  pow(trackDirection.z(),2) ) ) * (180/3.14159265);
   double beta  = acos(trackDirection.y() / sqrt( pow(trackDirection.x(),2) +  pow(trackDirection.z(),2) ) ) * (180/3.14159265);

   if(path>0.4 && path<0.45){
      Charge_Vs_TransversAngle->Fill(trans,Charge/path);
      Charge_Vs_Alpha->Fill(alpha,Charge/path);
      Charge_Vs_Beta->Fill(beta,Charge/path);
   }

   NStrips_Vs_TransversAngle->Fill(trans,Ampls.size());
   NStrips_Vs_Alpha         ->Fill(alpha,Ampls.size());
   NStrips_Vs_Beta          ->Fill(beta ,Ampls.size());

   NHighStripInCluster->Fill(NHighStrip);
   if(NHighStrip==1)   Charge_Vs_PathLength_Sat  ->Fill(path, Charge );
   if(NHighStrip==2)   Charge_Vs_PathLength_NoSat->Fill(path, Charge );
 
   APV_Charge  ->Fill(APV->Index,ClusterChargeOverPath);
   APV_Momentum->Fill(APV->Index,trajState.globalMomentum().mag());

   return ClusterChargeOverPath;
}

bool SiStripGainFromData::IsFarFromBorder(LocalPoint HitLocalPos, const uint32_t detid, const edm::EventSetup* iSetup)
{ 
  edm::ESHandle<TrackerGeometry> tkGeom; iSetup->get<TrackerDigiGeometryRecord>().get( tkGeom );

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

  if (fabs(HitLocalPos.x()) >= (HalfWidth  - DistFromBorder) ) return false;//Don't think is really necessary
  if (fabs(HitLocalPos.y()) >= (HalfLength - DistFromBorder) ) return false;

  return true;
}

void SiStripGainFromData::getPeakOfLandau(TH1* InputHisto, double* FitResults, double LowRange, double HighRange)
{ 
   double adcs           = -0.5; 
   double adcs_err       = 0.; 
   double width          = -0.5;
   double width_err      = 0;
   double chi2overndf    = -0.5;

   double nr_of_entries  = InputHisto->GetEntries();

   if( (unsigned int)nr_of_entries < MinNrEntries){
       FitResults[0] = adcs;
       FitResults[1] = adcs_err;
       FitResults[2] = width;
       FitResults[3] = width_err;
       FitResults[4] = chi2overndf;
      return;
   }

   // perform fit with standard landau
   TF1* MyLandau = new TF1("MyLandau","landau",LowRange, HighRange);
   MyLandau->SetParameter("MPV",300);

   InputHisto->Fit("MyLandau","QR WW");
   TF1 * fitfunction = (TF1*) InputHisto->GetListOfFunctions()->First();

   // MPV is parameter 1 (0=constant, 1=MPV, 2=Sigma)
    adcs        = fitfunction->GetParameter("MPV");
    adcs_err    = fitfunction->GetParError(1);
    width       = fitfunction->GetParameter(2);
    width_err   = fitfunction->GetParError(2);
    chi2overndf = fitfunction->GetChisquare() / fitfunction->GetNDF();

    // if still wrong, give up
    if(adcs<2. || chi2overndf>MaxChi2OverNDF){
       adcs  = -0.5; adcs_err  = 0.;
       width = -0.5; width_err = 0;
       chi2overndf = -0.5;
    }

    FitResults[0] = adcs;
    FitResults[1] = adcs_err;
    FitResults[2] = width;
    FitResults[3] = width_err;
    FitResults[4] = chi2overndf;
}




SiStripApvGain* SiStripGainFromData::getNewObject() 
{
   cout << "DB1" << endl;

  if( !(strcmp(AlgoMode.c_str(),"WriteOnDB")==0 || strcmp(AlgoMode.c_str(),"SingleJob")==0) )return NULL;

   SiStripApvGain * obj = new SiStripApvGain();
   std::vector<float>* theSiStripVector = NULL;
   int PreviousDetId = -1; 
   for(unsigned int a=0;a<APVsCollOrdered.size();a++)
   {
      stAPVPairGain* APV = APVsCollOrdered[a];
      if(APV==NULL){ printf("Bug\n"); continue; }
      if(APV->DetId != PreviousDetId){
         if(theSiStripVector!=NULL){
	    SiStripApvGain::Range range(theSiStripVector->begin(),theSiStripVector->end());
	    if ( !obj->put(PreviousDetId,range) )  printf("Bug to put detId = %i\n",PreviousDetId);
	 }

 	 theSiStripVector = new std::vector<float>;
         PreviousDetId = APV->DetId;
      }
      printf("%i | %i | %f\n", APV->DetId,APV->APVPairId,APV->Gain);
      theSiStripVector->push_back(APV->Gain);
      theSiStripVector->push_back(APV->Gain);
   }

   if(theSiStripVector!=NULL){
      SiStripApvGain::Range range(theSiStripVector->begin(),theSiStripVector->end());
      if ( !obj->put(PreviousDetId,range) )  printf("Bug to put detId = %i\n",PreviousDetId);
   }

   cout << "DB2" << endl;

   return obj;
}



DEFINE_FWK_MODULE(SiStripGainFromData);
