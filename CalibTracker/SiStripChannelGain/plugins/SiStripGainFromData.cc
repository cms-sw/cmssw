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

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"


#include "TFile.h"
#include "TObjString.h"
#include "TString.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TF1.h"
#include "TROOT.h"

#include <ext/hash_map>



using namespace edm;
using namespace reco;
using namespace std;
using namespace __gnu_cxx;

struct stAPVPairGain{unsigned int Index; int DetId; int APVPairId; int SubDet; float Eta; float R; float Thickness; double MPV; double Gain; double PreviousGain;};

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
      bool                IsFarFromBorder(TrajectoryStateOnSurface trajState, const uint32_t detid, const edm::EventSetup* iSetup);

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
      bool         FirstSetOfConstants;

      std::string  AlgoMode;
      std::string  OutputGains;
      std::string  OutputHistos;
      std::string  TrajToTrackProducer;
      std::string  TrajToTrackLabel;

      vector<string> VInputFiles;


      TH2F*	   Tracks_P_Vs_Eta;
      TH2F*        Tracks_Pt_Vs_Eta;

      TH1F*        NumberOfEntriesByAPVPair;
      TH1F*        HChi2OverNDF;
      TH1F*        HTrackChi2OverNDF;
      TH1F*        HTrackHits;

      TH2F*        MPV_Vs_EtaTIB;
      TH2F*        MPV_Vs_EtaTID;
      TH2F*        MPV_Vs_EtaTOB;
      TH2F*        MPV_Vs_EtaTEC;
      TH2F*        MPV_Vs_EtaTEC1;
      TH2F*        MPV_Vs_EtaTEC2;

      TH2F*        Charge_Vs_PathTIB;
      TH2F*        Charge_Vs_PathTID;
      TH2F*        Charge_Vs_PathTOB;
      TH2F*        Charge_Vs_PathTEC;
      TH2F*        Charge_Vs_PathTEC1;
      TH2F*        Charge_Vs_PathTEC2;

      TH1F*        MPV_Vs_PathTIB; 
      TH1F*        MPV_Vs_PathTID;
      TH1F*        MPV_Vs_PathTOB;
      TH1F*        MPV_Vs_PathTEC;
      TH1F*        MPV_Vs_PathTEC1;
      TH1F*        MPV_Vs_PathTEC2;

      TH2F*        MPV_Vs_Eta;
      TH2F*        MPV_Vs_R;

//      TH2F*        PD_Vs_Eta;
//      TH2F*        PD_Vs_R;

      TH1F*        APV_DetId;
      TH1F*        APV_PairId;
      TH1F*        APV_Eta;
      TH1F*        APV_R;
      TH1F*        APV_SubDet;
      TH2F*        APV_Momentum;
      TH2F*        APV_Charge;
      TH2F*        APV_PathLength;
      TH1F*        APV_PathLengthM;
      TH1F*        APV_MPV;
      TH1F*        APV_Gain;
      TH1F*        APV_Thickness;

      TH1F*        MPVs;
      TH1F*        MPVs320;
      TH1F*        MPVs500;

      TH2F*        MPV_vs_10RplusEta;


      TH1F*        NHighStripInCluster;
      TH2F*        Charge_Vs_PathLength_CS1;
      TH2F*        Charge_Vs_PathLength_CS2;
      TH2F*        Charge_Vs_PathLength_CS3;
      TH2F*        Charge_Vs_PathLength_CS4;
      TH2F*        Charge_Vs_PathLength_CS5;

      TH1F*        MPV_Vs_PathLength_CS1;
      TH1F*        MPV_Vs_PathLength_CS2;
      TH1F*        MPV_Vs_PathLength_CS3;
      TH1F*        MPV_Vs_PathLength_CS4;
      TH1F*        MPV_Vs_PathLength_CS5;

      TH1F*        FWHM_Vs_PathLength_CS1;
      TH1F*        FWHM_Vs_PathLength_CS2;
      TH1F*        FWHM_Vs_PathLength_CS3;
      TH1F*        FWHM_Vs_PathLength_CS4;
      TH1F*        FWHM_Vs_PathLength_CS5;


      TH2F*        Charge_Vs_PathLength;
      TH2F*        Charge_Vs_PathLength320;
      TH2F*        Charge_Vs_PathLength500;

      TH1F*        MPV_Vs_PathLength;
      TH1F*        MPV_Vs_PathLength320;
      TH1F*        MPV_Vs_PathLength500;

      TH1F*        FWHM_Vs_PathLength;
      TH1F*        FWHM_Vs_PathLength320;
      TH1F*        FWHM_Vs_PathLength500;


      TH2F*        Charge_Vs_TransversAngle;
      TH1F*        MPV_Vs_TransversAngle;
      TH2F*        NStrips_Vs_TransversAngle;

      TH2F*        Charge_Vs_Alpha;
      TH1F*        MPV_Vs_Alpha;
      TH2F*        NStrips_Vs_Alpha;

      TH2F*        Charge_Vs_Beta;
      TH1F*        MPV_Vs_Beta;
      TH2F*        NStrips_Vs_Beta;

      TH2F*        MPV_Vs_Error;
      TH2F*        Entries_Vs_Error;

      TH2F*        HitLocalPosition;
      TH2F*        HitLocalPositionBefCut;

      TH1F*        JobInfo;


      TFile*       Output;
      unsigned int NEvent;    
      unsigned int SRun;
      unsigned int SEvent;
      TimeValue_t  STimestamp;
      unsigned int ERun;
      unsigned int EEvent;
      TimeValue_t  ETimestamp;

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

   CheckLocalAngle     = iConfig.getUntrackedParameter<bool>    ("checkLocalAngle"    ,  false);
   MinNrEntries        = iConfig.getUntrackedParameter<unsigned>("minNrEntries"       ,  20);
   MaxMPVError         = iConfig.getUntrackedParameter<double>  ("maxMPVError"        ,  500.0);
   MaxChi2OverNDF      = iConfig.getUntrackedParameter<double>  ("maxChi2OverNDF"     ,  5.0);
   MinTrackMomentum    = iConfig.getUntrackedParameter<double>  ("minTrackMomentum"   ,  3.0);
   MaxTrackMomentum    = iConfig.getUntrackedParameter<double>  ("maxTrackMomentum"   ,  99999.0);
   MinTrackEta         = iConfig.getUntrackedParameter<double>  ("minTrackEta"        , -5.0);
   MaxTrackEta         = iConfig.getUntrackedParameter<double>  ("maxTrackEta"        ,  5.0);
   MaxNrStrips         = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"        ,  2);
   MinTrackHits        = iConfig.getUntrackedParameter<unsigned>("MinTrackHits"       ,  8);
   MaxTrackChiOverNdf  = iConfig.getUntrackedParameter<double>  ("MaxTrackChiOverNdf" ,  3);
   AllowSaturation     = iConfig.getUntrackedParameter<bool>    ("AllowSaturation"    ,  false);
   FirstSetOfConstants = iConfig.getUntrackedParameter<bool>    ("FirstSetOfConstants",  true);

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

   JobInfo                    = new TH1F ("JobInfo" , "JobInfo", 20,0,20);

   APV_DetId                  = new TH1F ("APV_DetId"      , "APV_DetId"      , 36393,0,36392);
   APV_PairId                 = new TH1F ("APV_PairId"     , "APV_PairId"     , 36393,0,36392);
   APV_Eta                    = new TH1F ("APV_Eta"        , "APV_Eta"        , 36393,0,36392);
   APV_R                      = new TH1F ("APV_R"          , "APV_R"          , 36393,0,36392);
   APV_SubDet                 = new TH1F ("APV_SubDet"     , "APV_SubDet"     , 36393,0,36392);
   APV_Momentum               = new TH2F ("APV_Momentum"   , "APV_Momentum"   , 36393,0,36392, 200,0,100);
   APV_Charge                 = new TH2F ("APV_Charge"     , "APV_Charge"     , 36393,0,36392, 1000,0,2000);
   APV_PathLength             = new TH2F ("APV_PathLength" , "APV_PathLength" , 36393,0,36392, 200,0.2,1.4);
   APV_PathLengthM            = new TH1F ("APV_PathLengthM", "APV_PathLengthM", 36393,0,36392);
   APV_MPV                    = new TH1F ("APV_MPV"        , "APV_MPV"        , 36393,0,36392);
   APV_Gain                   = new TH1F ("APV_Gain"       , "APV_Gain"       , 36393,0,36392);
   APV_Thickness              = new TH1F ("APV_Thickness"  , "APV_Thicknes"   , 36393,0,36392);


   Tracks_P_Vs_Eta            = new TH2F ("Tracks_P_Vs_Eta"   , "Tracks_P_Vs_Eta" , 60, 0,3,500,0,100);
   Tracks_Pt_Vs_Eta           = new TH2F ("Tracks_Pt_Vs_Eta"  , "Tracks_Pt_Vs_Eta", 60, 0,3,500,0,100);

   Charge_Vs_PathTIB          = new TH2F ("Charge_Vs_PathTIB" , "Charge_Vs_PathTIB" ,1000,0.2,1.4, 1000,0,2000);
   Charge_Vs_PathTID          = new TH2F ("Charge_Vs_PathTID" , "Charge_Vs_PathTID" ,1000,0.2,1.4, 1000,0,2000);
   Charge_Vs_PathTOB          = new TH2F ("Charge_Vs_PathTOB" , "Charge_Vs_PathTOB" ,1000,0.2,1.4, 1000,0,2000);
   Charge_Vs_PathTEC          = new TH2F ("Charge_Vs_PathTEC" , "Charge_Vs_PathTEC" ,1000,0.2,1.4, 1000,0,2000);
   Charge_Vs_PathTEC1         = new TH2F ("Charge_Vs_PathTEC1", "Charge_Vs_PathTEC1",1000,0.2,1.4, 1000,0,2000);
   Charge_Vs_PathTEC2         = new TH2F ("Charge_Vs_PathTEC2", "Charge_Vs_PathTEC2",1000,0.2,1.4, 1000,0,2000);

   Charge_Vs_PathLength_CS1   = new TH2F ("Charge_Vs_PathLength_CS1", "Charge_Vs_PathLength_CS1"  , 500,0.2,1.4, 2000,0,2000);
   Charge_Vs_PathLength_CS2   = new TH2F ("Charge_Vs_PathLength_CS2", "Charge_Vs_PathLength_CS2"  , 500,0.2,1.4, 2000,0,2000);
   Charge_Vs_PathLength_CS3   = new TH2F ("Charge_Vs_PathLength_CS3", "Charge_Vs_PathLength_CS3"  , 500,0.2,1.4, 2000,0,2000);
   Charge_Vs_PathLength_CS4   = new TH2F ("Charge_Vs_PathLength_CS4", "Charge_Vs_PathLength_CS4"  , 500,0.2,1.4, 2000,0,2000);
   Charge_Vs_PathLength_CS5   = new TH2F ("Charge_Vs_PathLength_CS5", "Charge_Vs_PathLength_CS5"  , 500,0.2,1.4, 2000,0,2000);

   Charge_Vs_PathLength       = new TH2F ("Charge_Vs_PathLength"    , "Charge_Vs_PathLength"  , 500,0.2,1.4, 2000,0,2000);
   Charge_Vs_PathLength320    = new TH2F ("Charge_Vs_PathLength320" , "Charge_Vs_PathLength"  , 500,0.2,1.4, 2000,0,2000);
   Charge_Vs_PathLength500    = new TH2F ("Charge_Vs_PathLength500" , "Charge_Vs_PathLength"  , 500,0.2,1.4, 2000,0,2000);

   Charge_Vs_TransversAngle   = new TH2F ("Charge_Vs_TransversAngle" , "Charge_Vs_TransversAngle" ,220,-20,200, 1000,0,2000);
   Charge_Vs_Alpha            = new TH2F ("Charge_Vs_Alpha"  , "Charge_Vs_Alpha"   ,220 ,-20,200, 1000,0,2000);
   Charge_Vs_Beta             = new TH2F ("Charge_Vs_Beta"   , "Charge_Vs_Beta"    ,220,-20,200, 1000,0,2000);

   NStrips_Vs_TransversAngle  = new TH2F ("NStrips_Vs_TransversAngle", "NStrips_Vs_TransversAngle",220,-20,200, 50,0,10);
   NStrips_Vs_Alpha           = new TH2F ("NStrips_Vs_Alpha" , "NStrips_Vs_Alpha"  ,220 ,-20,200, 50,0,10);
   NStrips_Vs_Beta            = new TH2F ("NStrips_Vs_Beta"  , "NStrips_Vs_Beta"   ,220,-20,200, 50,0,10);
   NHighStripInCluster        = new TH1F ("NHighStripInCluster"     , "NHighStripInCluster"       ,15,0,14);

   HTrackChi2OverNDF          = new TH1F ("TrackChi2OverNDF","TrackChi2OverNDF", 5000, 0,10);
   HTrackHits                 = new TH1F ("TrackHits","TrackHits", 40, 0,40);

   if( strcmp(AlgoMode.c_str(),"MultiJob")!=0 ){

      MPV_Vs_EtaTIB              = new TH2F ("MPV_Vs_EtaTIB"     , "MPV_Vs_EtaTIB" , 50, -3.0, 3.0, 1350, 0, 1350);
      MPV_Vs_EtaTID              = new TH2F ("MPV_Vs_EtaTID"     , "MPV_Vs_EtaTID" , 50, -3.0, 3.0, 1350, 0, 1350);
      MPV_Vs_EtaTOB              = new TH2F ("MPV_Vs_EtaTOB"     , "MPV_Vs_EtaTOB" , 50, -3.0, 3.0, 1350, 0, 1350);
      MPV_Vs_EtaTEC              = new TH2F ("MPV_Vs_EtaTEC"     , "MPV_Vs_EtaTEC" , 50, -3.0, 3.0, 1350, 0, 1350);
      MPV_Vs_EtaTEC1             = new TH2F ("MPV_Vs_EtaTEC1"    , "MPV_Vs_EtaTEC1", 50, -3.0, 3.0, 1350, 0, 1350);
      MPV_Vs_EtaTEC2             = new TH2F ("MPV_Vs_EtaTEC2"    , "MPV_Vs_EtaTEC2", 50, -3.0, 3.0, 1350, 0, 1350);

      MPV_Vs_PathTIB             = new TH1F ("MPV_Vs_PathTIB"    , "MPV_Vs_PathTIB"    ,1000,0.2,1.4);
      MPV_Vs_PathTID             = new TH1F ("MPV_Vs_PathTID"    , "MPV_Vs_PathTID"    ,1000,0.2,1.4);
      MPV_Vs_PathTOB             = new TH1F ("MPV_Vs_PathTOB"    , "MPV_Vs_PathTOB"    ,1000,0.2,1.4);
      MPV_Vs_PathTEC             = new TH1F ("MPV_Vs_PathTEC"    , "MPV_Vs_PathTEC"    ,1000,0.2,1.4);
      MPV_Vs_PathTEC1            = new TH1F ("MPV_Vs_PathTEC1"   , "MPV_Vs_PathTEC1"   ,1000,0.2,1.4);
      MPV_Vs_PathTEC2            = new TH1F ("MPV_Vs_PathTEC2"   , "MPV_Vs_PathTEC2"   ,1000,0.2,1.4);

      MPV_Vs_Eta                 = new TH2F ("MPV_Vs_Eta", "MPV_Vs_Eta", 50, -3.0, 3.0, 1350, 0, 1350);
      MPV_Vs_R                   = new TH2F ("MPV_Vs_R"  , "MPV_Vs_R"  , 150, 0.0, 150.0, 1350, 0, 1350);
   
      MPV_Vs_PathLength_CS1      = new TH1F ("MPV_Vs_PathLength_CS1"   , "MPV_Vs_PathLength_CS1", 500, 0.2, 1.4);
      MPV_Vs_PathLength_CS2      = new TH1F ("MPV_Vs_PathLength_CS2"   , "MPV_Vs_PathLength_CS2", 500, 0.2, 1.4);
      MPV_Vs_PathLength_CS3      = new TH1F ("MPV_Vs_PathLength_CS3"   , "MPV_Vs_PathLength_CS3", 500, 0.2, 1.4);
      MPV_Vs_PathLength_CS4      = new TH1F ("MPV_Vs_PathLength_CS4"   , "MPV_Vs_PathLength_CS4", 500, 0.2, 1.4);
      MPV_Vs_PathLength_CS5      = new TH1F ("MPV_Vs_PathLength_CS5"   , "MPV_Vs_PathLength_CS5", 500, 0.2, 1.4);

      FWHM_Vs_PathLength_CS1     = new TH1F ("FWHM_Vs_PathLength_CS1"  , "FWHM_Vs_PathLength_CS1", 500, 0.2, 1.4);
      FWHM_Vs_PathLength_CS2     = new TH1F ("FWHM_Vs_PathLength_CS2"  , "FWHM_Vs_PathLength_CS2", 500, 0.2, 1.4);
      FWHM_Vs_PathLength_CS3     = new TH1F ("FWHM_Vs_PathLength_CS3"  , "FWHM_Vs_PathLength_CS3", 500, 0.2, 1.4);
      FWHM_Vs_PathLength_CS4     = new TH1F ("FWHM_Vs_PathLength_CS4"  , "FWHM_Vs_PathLength_CS4", 500, 0.2, 1.4);
      FWHM_Vs_PathLength_CS5     = new TH1F ("FWHM_Vs_PathLength_CS5"  , "FWHM_Vs_PathLength_CS5", 500, 0.2, 1.4);

      MPV_Vs_PathLength          = new TH1F ("MPV_Vs_PathLength"       , "MPV_Vs_PathLength"  ,500,0.2,1.4);
      MPV_Vs_PathLength320       = new TH1F ("MPV_Vs_PathLength320"    , "MPV_Vs_PathLength"  ,500,0.2,1.4);
      MPV_Vs_PathLength500       = new TH1F ("MPV_Vs_PathLength500"    , "MPV_Vs_PathLength"  ,500,0.2,1.4);

      FWHM_Vs_PathLength         = new TH1F ("FWHM_Vs_PathLength"      , "FWHM_Vs_PathLength"  ,500,0.2,1.4);
      FWHM_Vs_PathLength320      = new TH1F ("FWHM_Vs_PathLength320"   , "FWHM_Vs_PathLength"  ,500,0.2,1.4);
      FWHM_Vs_PathLength500      = new TH1F ("FWHM_Vs_PathLength500"   , "FWHM_Vs_PathLength"  ,500,0.2,1.4);

      MPV_Vs_TransversAngle      = new TH1F ("MPV_Vs_TransversAngle"    , "MPV_Vs_TransversAngle"    ,220,-20,200);
      MPV_Vs_Alpha               = new TH1F ("MPV_Vs_Alpha"     , "MPV_Vs_Alpha"      ,220 ,-20,200);
      MPV_Vs_Beta                = new TH1F ("MPV_Vs_Beta"      , "MPV_Vs_Beta"       ,220,-20,200);

      MPV_Vs_Error               = new TH2F ("MPV_Vs_Error"   , "MPV_Vs_Error"  ,1350, 0, 1350, 1000,0,100);
      Entries_Vs_Error           = new TH2F ("Entries_Vs_Error","Entries_Vs_Error",2000,0,10000,1000,0,100); 

      NumberOfEntriesByAPVPair   = new TH1F ("NumberOfEntriesByAPVPair", "NumberOfEntriesByAPVPair", 2500, 0,10000);
      HChi2OverNDF               = new TH1F ("Chi2OverNDF","Chi2OverNDF", 5000, 0,10);

      MPVs                       = new TH1F ("MPVs", "MPVs", 600,0,600);
      MPVs320                    = new TH1F ("MPVs320", "MPVs320", 600,0,600);
      MPVs500                    = new TH1F ("MPVs500", "MPVs500", 600,0,600);
 
      MPV_vs_10RplusEta          = new TH2F ("MPV_vs_10RplusEta","MPV_vs_10RplusEta", 48000,0,2400, 800,100,500);
   }

   gROOT->cd();

   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
   vector<GeomDet*> Det = tkGeom->dets();


   edm::ESHandle<SiStripGain> gainHandle;
   if(strcmp(AlgoMode.c_str(),"MultiJob")!=0 && !FirstSetOfConstants){
      iSetup.get<SiStripGainRcd>().get(gainHandle);
      if(!gainHandle.isValid()){printf("\n#####################\n\nERROR --> gainHandle is not valid\n\n#####################\n\n");exit(0);}
   }

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
                APV->PreviousGain  = 1;
                APV->Eta           = Eta;
                APV->R             = R;
                APV->Thickness     = Thick;

                if(!FirstSetOfConstants && gainHandle.isValid()){
                   SiStripApvGain::Range detGainRange = gainHandle->getRange(APV->DetId);
                   APV->PreviousGain = *(detGainRange.first + (2*APV->APVPairId));
                }

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

   NEvent     = 0;
   SRun       = 0;
   SEvent     = 0;
   STimestamp = 0;
   ERun       = 0;
   EEvent     = 0;
   ETimestamp = 0;
}

void 
SiStripGainFromData::algoEndJob() {
   unsigned int I=0;

   if( strcmp(AlgoMode.c_str(),"WriteOnDB")==0 || strcmp(AlgoMode.c_str(),"Merge")==0){
      TFile* file = NULL;
      for(unsigned int f=0;f<VInputFiles.size();f++){
         printf("Loading New Input File : %s\n", VInputFiles[f].c_str());
         file =  new TFile( VInputFiles[f].c_str() ); if(file==NULL || file->IsZombie() ){printf("### Bug With File %s\n### File will be skipped \n",VInputFiles[f].c_str()); continue;}
         APV_Charge               ->Add( (TH1*) file->FindObjectAny("APV_Charge")               , 1);
         APV_Momentum             ->Add( (TH1*) file->FindObjectAny("APV_Momentum")             , 1);
         APV_PathLength           ->Add( (TH1*) file->FindObjectAny("APV_PathLength")           , 1);

         Tracks_P_Vs_Eta          ->Add( (TH1*) file->FindObjectAny("Tracks_P_Vs_Eta")          , 1);
         Tracks_Pt_Vs_Eta         ->Add( (TH1*) file->FindObjectAny("Tracks_Pt_Vs_Eta")         , 1);

         Charge_Vs_PathTIB        ->Add( (TH1*) file->FindObjectAny("Charge_Vs_PathTIB")        , 1);
         Charge_Vs_PathTID        ->Add( (TH1*) file->FindObjectAny("Charge_Vs_PathTID")        , 1);
         Charge_Vs_PathTOB        ->Add( (TH1*) file->FindObjectAny("Charge_Vs_PathTOB")        , 1);
         Charge_Vs_PathTEC        ->Add( (TH1*) file->FindObjectAny("Charge_Vs_PathTEC")        , 1);
         Charge_Vs_PathTEC1       ->Add( (TH1*) file->FindObjectAny("Charge_Vs_PathTEC1")       , 1);
         Charge_Vs_PathTEC2       ->Add( (TH1*) file->FindObjectAny("Charge_Vs_PathTEC2")       , 1);

         HTrackChi2OverNDF        ->Add( (TH1*) file->FindObjectAny("TrackChi2OverNDF")         , 1);
         HTrackHits               ->Add( (TH1*) file->FindObjectAny("TrackHits")                , 1);

         NHighStripInCluster      ->Add( (TH1*) file->FindObjectAny("NHighStripInCluster")      , 1);
         Charge_Vs_PathLength     ->Add( (TH1*) file->FindObjectAny("Charge_Vs_PathLength")     , 1);
         Charge_Vs_PathLength320  ->Add( (TH1*) file->FindObjectAny("Charge_Vs_PathLength320")  , 1);
         Charge_Vs_PathLength500  ->Add( (TH1*) file->FindObjectAny("Charge_Vs_PathLength500")  , 1);
         Charge_Vs_TransversAngle ->Add( (TH1*) file->FindObjectAny("Charge_Vs_TransversAngle") , 1);
         NStrips_Vs_TransversAngle->Add( (TH1*) file->FindObjectAny("NStrips_Vs_TransversAngle"), 1);
         Charge_Vs_Alpha          ->Add( (TH1*) file->FindObjectAny("Charge_Vs_Alpha")          , 1);
         NStrips_Vs_Alpha         ->Add( (TH1*) file->FindObjectAny("NStrips_Vs_Alpha")         , 1);

	 TH1F* JobInfo_tmp = (TH1F*) file->FindObjectAny("JobInfo");
         NEvent                 += (unsigned int) JobInfo_tmp->GetBinContent(JobInfo_tmp->GetXaxis()->FindBin(1));
         unsigned int tmp_SRun   = (unsigned int) JobInfo_tmp->GetBinContent(JobInfo_tmp->GetXaxis()->FindBin(3));
         unsigned int tmp_SEvent = (unsigned int) JobInfo_tmp->GetBinContent(JobInfo_tmp->GetXaxis()->FindBin(4));
         unsigned int tmp_ERun   = (unsigned int) JobInfo_tmp->GetBinContent(JobInfo_tmp->GetXaxis()->FindBin(6));
         unsigned int tmp_EEvent = (unsigned int) JobInfo_tmp->GetBinContent(JobInfo_tmp->GetXaxis()->FindBin(7));

              if(tmp_SRun< SRun){SRun=tmp_SRun; SEvent=tmp_SEvent;}
         else if(tmp_SRun==SRun && tmp_SEvent<SEvent){SEvent=tmp_SEvent;}

              if(tmp_ERun> ERun){ERun=tmp_ERun; EEvent=tmp_EEvent;}
         else if(tmp_ERun==ERun && tmp_EEvent>EEvent){EEvent=tmp_EEvent;}

         printf("Deleting Current Input File\n");
         file->Close();
         delete file;
      }

   }

   JobInfo->Fill(1,NEvent);
   JobInfo->Fill(3,SRun);
   JobInfo->Fill(4,SEvent);
   JobInfo->Fill(6,ERun);
   JobInfo->Fill(7,EEvent);

   if( strcmp(AlgoMode.c_str(),"MultiJob")!=0 ){
      TH1D* Proj = NULL;
      double* FitResults = new double[5];
      I=0;
      for(hash_map<unsigned int, stAPVPairGain*,  hash<unsigned int>, isEqual >::iterator it = APVsColl.begin();it!=APVsColl.end();it++){
      if( I%1825==0 ) printf("Fitting Histograms \t %6.2f%%\n",(100.0*I)/APVsColl.size());I++;
         stAPVPairGain* APV = it->second;

         int bin = APV_Charge->GetXaxis()->FindBin(APV->Index);
         Proj = APV_Charge->ProjectionY(" ",bin,bin,"e");
         if(Proj==NULL)continue;

         getPeakOfLandau(Proj,FitResults);
         APV->MPV = FitResults[0];
         printf("MPV = %f - %f\n",FitResults[0], FitResults[1]);
         if(FitResults[0]!=-0.5 && FitResults[1]<MaxMPVError){
            APV_MPV->Fill(APV->Index,APV->MPV);
            MPVs   ->Fill(APV->MPV);
            if(APV->Thickness<0.04)                 MPVs320->Fill(APV->MPV);
            if(APV->Thickness>0.04)                 MPVs500->Fill(APV->MPV);

            MPV_Vs_R->Fill(APV->R,APV->MPV);
            MPV_Vs_Eta->Fill(APV->Eta,APV->MPV);
            if(APV->SubDet==StripSubdetector::TIB)  MPV_Vs_EtaTIB ->Fill(APV->Eta,APV->MPV);
            if(APV->SubDet==StripSubdetector::TID)  MPV_Vs_EtaTID ->Fill(APV->Eta,APV->MPV);
            if(APV->SubDet==StripSubdetector::TOB)  MPV_Vs_EtaTOB ->Fill(APV->Eta,APV->MPV);
            if(APV->SubDet==StripSubdetector::TEC){ MPV_Vs_EtaTEC ->Fill(APV->Eta,APV->MPV);
            if(APV->Thickness<0.04)		    MPV_Vs_EtaTEC1->Fill(APV->Eta,APV->MPV);
            if(APV->Thickness>0.04)                 MPV_Vs_EtaTEC2->Fill(APV->Eta,APV->MPV);
            }

            double Eta_R = (APV->R*20.0)+APV->Eta;
            MPV_vs_10RplusEta ->Fill(Eta_R,APV->MPV);           
         }

         if(FitResults[0]!=-0.5){
            HChi2OverNDF->Fill(FitResults[4]);
            MPV_Vs_Error->Fill(FitResults[0],FitResults[1]);
            Entries_Vs_Error->Fill(Proj->GetEntries(),FitResults[1]);
         }
         NumberOfEntriesByAPVPair->Fill(Proj->GetEntries());
         delete Proj;


         Proj = APV_PathLength->ProjectionY(" ",bin,bin,"e");
         if(Proj==NULL)continue;

         APV_PathLengthM->SetBinContent(APV->Index, Proj->GetMean(1)      );
         APV_PathLengthM->SetBinError  (APV->Index, Proj->GetMeanError(1) );
         delete Proj;
      }

      double MPVmean = MPVs->GetMean();
      for(hash_map<unsigned int, stAPVPairGain*,  hash<unsigned int>, isEqual >::iterator it = APVsColl.begin();it!=APVsColl.end();it++){

         stAPVPairGain*   APV = it->second;
         if(APV->MPV>0)   APV->Gain = APV->MPV / MPVmean; // APV->MPV;
         else             APV->Gain = 1;    
         if(APV->Gain<=0) APV->Gain = 1;
         APV->Gain *= APV->PreviousGain;

         APV_Gain->Fill(APV->Index,APV->Gain); 
      }

      for(int j=0;j<Charge_Vs_PathLength->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathLength->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_PathLength->SetBinContent (j, FitResults[0]/Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathLength->SetBinError   (j, FitResults[1]/Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j));
         FWHM_Vs_PathLength->SetBinContent(j, FitResults[2]/(FitResults[0]/Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j)) );
         FWHM_Vs_PathLength->SetBinError  (j, FitResults[3]/(FitResults[0]/Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j)) );
         delete Proj;
      }

      for(int j=0;j<Charge_Vs_PathLength320->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathLength320->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_PathLength320->SetBinContent (j, FitResults[0]/Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathLength320->SetBinError   (j, FitResults[1]/Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j));
         FWHM_Vs_PathLength320->SetBinContent(j, FitResults[2]/(FitResults[0]/Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j)) );
         FWHM_Vs_PathLength320->SetBinError  (j, FitResults[3]/(FitResults[0]/Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j)) );
         delete Proj;
      }

      for(int j=0;j<Charge_Vs_PathLength500->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathLength500->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;    
         MPV_Vs_PathLength500->SetBinContent (j, FitResults[0]/Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathLength500->SetBinError   (j, FitResults[1]/Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j));
         FWHM_Vs_PathLength500->SetBinContent(j, FitResults[2]/(FitResults[0]/Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j) ));
         FWHM_Vs_PathLength500->SetBinError  (j, FitResults[3]/(FitResults[0]/Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j) ));
         delete Proj;
      }

      for(int j=0;j<Charge_Vs_PathLength_CS1->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathLength_CS1->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_PathLength_CS1->SetBinContent (j, FitResults[0]/Charge_Vs_PathLength_CS1->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathLength_CS1->SetBinError   (j, FitResults[1]/Charge_Vs_PathLength_CS1->GetXaxis()->GetBinCenter(j));
         FWHM_Vs_PathLength_CS1->SetBinContent(j, FitResults[2]/(FitResults[0]/Charge_Vs_PathLength_CS1->GetXaxis()->GetBinCenter(j) ));
         FWHM_Vs_PathLength_CS1->SetBinError  (j, FitResults[3]/(FitResults[0]/Charge_Vs_PathLength_CS1->GetXaxis()->GetBinCenter(j) ));
         delete Proj;
      }

      for(int j=0;j<Charge_Vs_PathLength_CS2->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathLength_CS2->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_PathLength_CS2->SetBinContent (j, FitResults[0]/Charge_Vs_PathLength_CS2->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathLength_CS2->SetBinError   (j, FitResults[1]/Charge_Vs_PathLength_CS2->GetXaxis()->GetBinCenter(j));
         FWHM_Vs_PathLength_CS2->SetBinContent(j, FitResults[2]/(FitResults[0]/Charge_Vs_PathLength_CS2->GetXaxis()->GetBinCenter(j) ));
         FWHM_Vs_PathLength_CS2->SetBinError  (j, FitResults[3]/(FitResults[0]/Charge_Vs_PathLength_CS2->GetXaxis()->GetBinCenter(j) ));
         delete Proj;
      }

      for(int j=0;j<Charge_Vs_PathLength_CS3->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathLength_CS3->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_PathLength_CS3->SetBinContent (j, FitResults[0]/Charge_Vs_PathLength_CS3->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathLength_CS3->SetBinError   (j, FitResults[1]/Charge_Vs_PathLength_CS3->GetXaxis()->GetBinCenter(j));
         FWHM_Vs_PathLength_CS3->SetBinContent(j, FitResults[2]/(FitResults[0]/Charge_Vs_PathLength_CS3->GetXaxis()->GetBinCenter(j) ));
         FWHM_Vs_PathLength_CS3->SetBinError  (j, FitResults[3]/(FitResults[0]/Charge_Vs_PathLength_CS3->GetXaxis()->GetBinCenter(j) ));
         delete Proj;
      }

      for(int j=0;j<Charge_Vs_PathLength_CS4->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathLength_CS4->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_PathLength_CS4->SetBinContent (j, FitResults[0]/Charge_Vs_PathLength_CS4->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathLength_CS4->SetBinError   (j, FitResults[1]/Charge_Vs_PathLength_CS4->GetXaxis()->GetBinCenter(j));
         FWHM_Vs_PathLength_CS4->SetBinContent(j, FitResults[2]/(FitResults[0]/Charge_Vs_PathLength_CS4->GetXaxis()->GetBinCenter(j) ));
         FWHM_Vs_PathLength_CS4->SetBinError  (j, FitResults[3]/(FitResults[0]/Charge_Vs_PathLength_CS4->GetXaxis()->GetBinCenter(j) ));
         delete Proj;
      }

      for(int j=0;j<Charge_Vs_PathLength_CS5->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathLength_CS5->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_PathLength_CS5->SetBinContent (j, FitResults[0]/Charge_Vs_PathLength_CS5->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathLength_CS5->SetBinError   (j, FitResults[1]/Charge_Vs_PathLength_CS5->GetXaxis()->GetBinCenter(j));
         FWHM_Vs_PathLength_CS5->SetBinContent(j, FitResults[2]/(FitResults[0]/Charge_Vs_PathLength_CS5->GetXaxis()->GetBinCenter(j) ));
         FWHM_Vs_PathLength_CS5->SetBinError  (j, FitResults[3]/(FitResults[0]/Charge_Vs_PathLength_CS5->GetXaxis()->GetBinCenter(j) ));
         delete Proj;
      }



      for(int j=0;j<Charge_Vs_PathTIB->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathTIB->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_PathTIB->SetBinContent(j, FitResults[0]/Charge_Vs_PathTIB->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathTIB->SetBinError  (j, FitResults[1]/Charge_Vs_PathTIB->GetXaxis()->GetBinCenter(j));
         delete Proj;
      }

      for(int j=0;j<Charge_Vs_PathTID->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathTID->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_PathTID->SetBinContent(j, FitResults[0]/Charge_Vs_PathTID->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathTID->SetBinError  (j, FitResults[1]/Charge_Vs_PathTID->GetXaxis()->GetBinCenter(j));
         delete Proj;
      }

      for(int j=0;j<Charge_Vs_PathTOB->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathTOB->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_PathTOB->SetBinContent(j, FitResults[0]/Charge_Vs_PathTOB->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathTOB->SetBinError  (j, FitResults[1]/Charge_Vs_PathTOB->GetXaxis()->GetBinCenter(j));
         delete Proj;
      }

      for(int j=0;j<Charge_Vs_PathTEC->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathTEC->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_PathTEC->SetBinContent(j, FitResults[0]/Charge_Vs_PathTEC->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathTEC->SetBinError  (j, FitResults[1]/Charge_Vs_PathTEC->GetXaxis()->GetBinCenter(j));
         delete Proj;
      }

      for(int j=0;j<Charge_Vs_PathTEC1->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathTEC1->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_PathTEC1->SetBinContent(j, FitResults[0]/Charge_Vs_PathTEC1->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathTEC1->SetBinError  (j, FitResults[1]/Charge_Vs_PathTEC1->GetXaxis()->GetBinCenter(j));
         delete Proj;
      }

      for(int j=0;j<Charge_Vs_PathTEC2->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_PathTEC2->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_PathTEC2->SetBinContent(j, FitResults[0]/Charge_Vs_PathTEC2->GetXaxis()->GetBinCenter(j));
         MPV_Vs_PathTEC2->SetBinError  (j, FitResults[1]/Charge_Vs_PathTEC2->GetXaxis()->GetBinCenter(j));
         delete Proj;
      }


      for(int j=1;j<Charge_Vs_TransversAngle->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_TransversAngle->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_TransversAngle->SetBinContent(j, FitResults[0]);
         MPV_Vs_TransversAngle->SetBinError  (j, FitResults[1]);
         delete Proj;
      }

      for(int j=1;j<Charge_Vs_Alpha->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_Alpha->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_Alpha->SetBinContent(j, FitResults[0]);
         MPV_Vs_Alpha->SetBinError  (j, FitResults[1]);
         delete Proj;
      }

      for(int j=1;j<Charge_Vs_Beta->GetXaxis()->GetNbins();j++){
         Proj      = Charge_Vs_Beta->ProjectionY(" ",j,j,"e");
         getPeakOfLandau(Proj,FitResults); if(FitResults[0] ==-0.5)continue;
         MPV_Vs_Beta->SetBinContent(j, FitResults[0]);
         MPV_Vs_Beta->SetBinError  (j, FitResults[1]);
         delete Proj;
      }

      FILE* Gains = fopen(OutputGains.c_str(),"w");
      fprintf(Gains,"NEvents = %i\n",NEvent);
      fprintf(Gains,"Number of APVs = %i\n",APVsColl.size());
      for(std::vector<stAPVPairGain*>::iterator it = APVsCollOrdered.begin();it!=APVsCollOrdered.end();it++){
         stAPVPairGain* APV = *it;
         fprintf(Gains,"%i | %i | PreviousGain = %7.5f NewGain = %7.5f\n", APV->DetId,APV->APVPairId,APV->PreviousGain,APV->Gain);
      }
      fclose(Gains);

      delete [] FitResults;
      delete Proj;
   }

   Output->SetCompressionLevel(5);
   Output->cd();
   Output->Write();
   Output->Close();
}



void
SiStripGainFromData::algoAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   if( strcmp(AlgoMode.c_str(),"WriteOnDB")==0 ) return;

   if(NEvent==0){
      SRun          = iEvent.id().run();
      SEvent        = iEvent.id().event();
      STimestamp    = iEvent.time().value();
   }
   ERun             = iEvent.id().run();
   EEvent           = iEvent.id().event();
   ETimestamp       = iEvent.time().value();
   NEvent++;

   iEvent_ = &iEvent;

   Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
   iEvent.getByLabel(TrajToTrackProducer, TrajToTrackLabel, trajTrackAssociationHandle);
   const TrajTrackAssociationCollection TrajToTrackMap = *trajTrackAssociationHandle.product();

   for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap.begin(); it!=TrajToTrackMap.end(); ++it) {
      const Track      track = *it->val;   
      const Trajectory traj  = *it->key;


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
//   const vector<uint16_t>& Ampls       = Cluster->amplitudes();
   const vector<uint8_t>&  Ampls       = Cluster->amplitudes();
   uint32_t                DetId       = Cluster->geographicalId();
   int                     FirstStrip  = Cluster->firstStrip();
   int                     APVPairId   = FirstStrip/256;
   stAPVPairGain*          APV         = APVsColl[(DetId<<2) | APVPairId];
   bool                    Saturation  = false;
   bool                    Overlaping  = false;
   int                     Charge      = 0;
   unsigned int            NHighStrip  = 0;

   if(!IsFarFromBorder(trajState,DetId, iSetup))return -1;

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
   if(NHighStrip==1)   Charge_Vs_PathLength_CS1->Fill(path, Charge );
   if(NHighStrip==2)   Charge_Vs_PathLength_CS2->Fill(path, Charge );
   if(NHighStrip==3)   Charge_Vs_PathLength_CS3->Fill(path, Charge );
   if(NHighStrip==4)   Charge_Vs_PathLength_CS4->Fill(path, Charge );
   if(NHighStrip==5)   Charge_Vs_PathLength_CS5->Fill(path, Charge );

 
   APV_Charge    ->Fill(APV->Index,ClusterChargeOverPath);
   APV_Momentum  ->Fill(APV->Index,trajState.globalMomentum().mag());
   APV_PathLength->Fill(APV->Index,path);

   return ClusterChargeOverPath;
}

bool SiStripGainFromData::IsFarFromBorder(TrajectoryStateOnSurface trajState, const uint32_t detid, const edm::EventSetup* iSetup)
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

  if (fabs(HitLocalPos.x())+HitLocalError.xx() >= (HalfWidth  - DistFromBorder) ) return false;//Don't think is really necessary
  if (fabs(HitLocalPos.y())+HitLocalError.xy() >= (HalfLength - DistFromBorder) ) return false;

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
//      printf("%i | %i | %f\n", APV->DetId,APV->APVPairId,APV->Gain);
      printf("%i | %i | PreviousGain = %7.5f NewGain = %7.5f\n", APV->DetId,APV->APVPairId,APV->PreviousGain,APV->Gain);
      theSiStripVector->push_back(APV->Gain);
      theSiStripVector->push_back(APV->Gain);
   }

   if(theSiStripVector!=NULL){
      SiStripApvGain::Range range(theSiStripVector->begin(),theSiStripVector->end());
      if ( !obj->put(PreviousDetId,range) )  printf("Bug to put detId = %i\n",PreviousDetId);
   }

   return obj;
}



DEFINE_FWK_MODULE(SiStripGainFromData);
