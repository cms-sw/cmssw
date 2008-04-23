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
#include "TString.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TF1.h"
#include "TROOT.h"



using namespace edm;
using namespace reco;
using namespace std;

struct stAPVPairGain{int DetId; int APVPairId; double MPV; double Gain; char* Eta_R;};


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
//      double              moduleWidth    (const uint32_t detid, const edm::EventSetup* iSetup);
      double              moduleThickness(const uint32_t detid, const edm::EventSetup* iSetup);
      bool                IsFarFromBorder(LocalPoint HitLocalPos, const uint32_t detid, const edm::EventSetup* iSetup);

      pair<double,double> getPeakOfLandau(TH1* InputHisto, double& maxChiOverNdf);
      pair<double,double> getPeakOfLandau(TH1* InputHisto, double& maxChiOverNdf, double LowRange, double HighRange);

      TObject* FindHisto(const char* Name);
      void  AddHisto(TObject* Histo);
      void PrintHisto();

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


      std::string  OutputGains;
      std::string  OutputHistos;
      std::string  TrajToTrackProducer;
      std::string  TrajToTrackLabel;

      TFile*       Tracks_Output;
      TDirectory*  Landau;
      TDirectory*  LandauLocalAngle;
      TDirectory*  RingCalibration;

      TH1D*	   ChargeDistrib;

      TH2D*	   Tracks_P_Vs_Eta;
      TH2D*        Tracks_Pt_Vs_Eta;

      TH2D*        APVPairPos;       
      TH1D*        NumberOfEntriesByAPVPair;
      TH1D*        HChi2OverNDF;
      TH1D*        HTrackChi2OverNDF;
      TH1D*        HTrackHits;

      TH2D*        MPV_Vs_EtaTID;
      TH2D*        MPV_Vs_EtaTIB;
      TH2D*        MPV_Vs_EtaTEC;
      TH2D*        MPV_Vs_EtaTEC1;
      TH2D*        MPV_Vs_EtaTEC2;
      TH2D*        MPV_Vs_EtaTOB;

      TH2D*        MPV_Vs_Eta;
      TH2D*        MPV_Vs_Eta_Calib;
      TH2D*        MPV_Vs_R;
      TH2D*        MPV_Vs_R_Calib;
      TH2D*        PD_Vs_Eta;
      TH2D*        PD_Vs_R;

      TH2D*        Ring_MPV_Vs_Eta;

      TH1D*        NHighStripInCluster;
      TH2D*        Charge_Vs_PathLength_Sat;
      TH2D*        Charge_Vs_PathLength_NoSat;

      

      TH2D*        Charge_Vs_PathLength;
      TH1D*        MPV_Vs_PathLength;
      TH2D*        Charge_Vs_PathLength320;
      TH1D*        MPV_Vs_PathLength320;
      TH2D*        Charge_Vs_PathLength500;
      TH1D*        MPV_Vs_PathLength500;


      TH2D*        Charge_Vs_TransversAngle;
      TH1D*        MPV_Vs_TransversAngle;

      TH2D*        Charge_Vs_Alpha;
      TH1D*        MPV_Vs_Alpha;
      TH2D*        NStrips_Vs_Alpha;

      TH2D*        Charge_Vs_Beta;
      TH1D*        MPV_Vs_Beta;
      TH2D*        NStrips_Vs_Beta;

      TH2D*        MPV_Vs_LocalAngle_Tick320;
      TH2D*        MPV_Vs_LocalAngle_Tick500;

      TH2D*        MPV_Vs_Error;
      TH2D*        Entries_Vs_Error;
      TH1D*        HMPV;


      TH2D*        HitLocalPosition;
      TH2D*        HitLocalPositionBefCut;

      TFile*       Output;
      TObjArray*   HlistAPVPairsEtaR;
      TObjArray*   HlistAPVPairsEtaR_PD;

      TObjArray*   HISTOS;

      TFile*       Input;

      unsigned int NEvent;     

      map<uint32_t, GlobalVector*> ModulePositionMap;
//      map<const char*, std::vector<const char*> > RingsMap;
//      map<const char*, double> MPVsMap;
//      map<const char*, double > GainsMap;

      std::vector<stAPVPairGain*> GainOfAPVpairs;
};

//SiStripGainFromData::SiStripGainFromData(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripApvGain>::ConditionDBWriter<SiStripApvGain>(iConfig), m_cacheID_(0)
SiStripGainFromData::SiStripGainFromData(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripApvGain>::ConditionDBWriter<SiStripApvGain>(iConfig)
{
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
}


SiStripGainFromData::~SiStripGainFromData()
{ 
}


void
SiStripGainFromData::algoBeginJob(const edm::EventSetup& iSetup)
{
   iSetup_                  = &iSetup;

   TH1::AddDirectory(kTRUE);
   Output                   = new TFile(OutputHistos.c_str(), "RECREATE");
   Landau                   = Output->mkdir("LandauFit");
   if(CheckLocalAngle) LandauLocalAngle = Output->mkdir("LocalAngleFit");
   RingCalibration          = Output->mkdir("RingCalibration");

   ChargeDistrib	    = new TH1D ("ChargeDistrib" , "ChargeDistrib"  ,1800,0,5400); 

   Tracks_P_Vs_Eta          = new TH2D ("Tracks_P_Vs_Eta"   , "Tracks_P_Vs_Eta" , 60, 0,3,500,0,100);
   Tracks_Pt_Vs_Eta         = new TH2D ("Tracks_Pt_Vs_Eta"  , "Tracks_Pt_Vs_Eta", 60, 0,3,500,0,100);

   MPV_Vs_EtaTIB            = new TH2D ("MPV_Vs_EtaTIB", "MPV_Vs_EtaTIB", 50, -3.0, 3.0, 1350, 0, 1350);
   MPV_Vs_EtaTID            = new TH2D ("MPV_Vs_EtaTID", "MPV_Vs_EtaTID", 50, -3.0, 3.0, 1350, 0, 1350);
   MPV_Vs_EtaTOB            = new TH2D ("MPV_Vs_EtaTOB", "MPV_Vs_EtaTOB", 50, -3.0, 3.0, 1350, 0, 1350);
   MPV_Vs_EtaTEC            = new TH2D ("MPV_Vs_EtaTEC", "MPV_Vs_EtaTEC", 50, -3.0, 3.0, 1350, 0, 1350);
   MPV_Vs_EtaTEC1           = new TH2D ("MPV_Vs_EtaTEC1", "MPV_Vs_EtaTEC1", 50, -3.0, 3.0, 1350, 0, 1350);
   MPV_Vs_EtaTEC2           = new TH2D ("MPV_Vs_EtaTEC2", "MPV_Vs_EtaTEC2", 50, -3.0, 3.0, 1350, 0, 1350);

   MPV_Vs_Eta               = new TH2D ("MPV_Vs_Eta", "MPV_Vs_Eta", 50, -3.0, 3.0, 1350, 0, 1350);
   MPV_Vs_Eta_Calib         = new TH2D ("MPV_Vs_Eta_Calib", "MPV_Vs_Eta_Calib", 50, -3.0, 3.0, 1350, 0, 1350);
   MPV_Vs_R                 = new TH2D ("MPV_Vs_R"  , "MPV_Vs_R"  , 150, 0.0, 150.0, 1350, 0, 1350);
   MPV_Vs_R_Calib           = new TH2D ("MPV_Vs_R_Calib"  , "MPV_Vs_R_Calib"  , 150, 0.0, 150.0, 1350, 0, 1350);

   PD_Vs_Eta                = new TH2D ("PD_Vs_Eta", "PD_Vs_Eta", 50, -3.0, 3.0, 500, 0, 100 );
   PD_Vs_R                  = new TH2D ("PD_Vs_R"  , "PD_Vs_R"  , 150, 0.0, 150.0, 500, 0, 100);

   Ring_MPV_Vs_Eta          = new TH2D ("Ring_MPV_Vs_Eta", "Ring_MPV_Vs_Eta", 50, -3.0, 3.0, 1350, 0, 1350);


   NHighStripInCluster        = new TH1D ("NHighStripInCluster" , "NHighStripInCluster"  ,15,0,14);
   Charge_Vs_PathLength_Sat   = new TH2D ("Charge_Vs_PathLength_Sat" , "Charge_Vs_PathLength_Sat"  ,2000,0,2, 1800,0,5400);
   Charge_Vs_PathLength_NoSat = new TH2D ("Charge_Vs_PathLength_NoSat" , "Charge_Vs_PathLength_NoSat"  ,2000,0,2, 1800,0,5400);

   Charge_Vs_PathLength     = new TH2D ("Charge_Vs_PathLength" , "Charge_Vs_PathLength"  ,2000,0,2, 1800,0,5400);
   MPV_Vs_PathLength        = new TH1D ("MPV_Vs_PathLength" , "MPV_Vs_PathLength"  ,2000,0,2);
   Charge_Vs_PathLength320  = new TH2D ("Charge_Vs_PathLength320" , "Charge_Vs_PathLength"  ,2000,0,2, 1800,0,5400);
   MPV_Vs_PathLength320     = new TH1D ("MPV_Vs_PathLength320" , "MPV_Vs_PathLength"  ,2000,0,2);
   Charge_Vs_PathLength500  = new TH2D ("Charge_Vs_PathLength500" , "Charge_Vs_PathLength"  ,2000,0,2, 1800,0,5400);
   MPV_Vs_PathLength500     = new TH1D ("MPV_Vs_PathLength500" , "MPV_Vs_PathLength"  ,2000,0,2);

   Charge_Vs_TransversAngle = new TH2D ("Charge_Vs_TransversAngle" , "Charge_Vs_TransversAngle"  ,361,0,360, 1800,0,5400);
   MPV_Vs_TransversAngle    = new TH1D ("MPV_Vs_TransversAngle" , "MPV_Vs_TransversAngle"  ,361,0,360);

   Charge_Vs_Alpha          = new TH2D ("Charge_Vs_Alpha" , "Charge_Vs_Alpha"  ,720,0,360, 1800,0,5400);
   MPV_Vs_Alpha             = new TH1D ("MPV_Vs_Alpha" , "MPV_Vs_Alpha"  ,720,0,360);
   NStrips_Vs_Alpha         = new TH2D ("NStrips_Vs_Alpha" , "NStrips_Vs_Alpha"  ,720,0,360, 50,0,10);

   Charge_Vs_Beta           = new TH2D ("Charge_Vs_Beta" , "Charge_Vs_Beta"  ,720,0,360, 1800,0,5400);
   MPV_Vs_Beta              = new TH1D ("MPV_Vs_Beta" , "MPV_Vs_Beta"  ,720,0,360);
   NStrips_Vs_Beta          = new TH2D ("NStrips_Vs_Beta" , "NStrips_Vs_Beta"  ,720,0,360, 50,0,10);


   if(CheckLocalAngle) 
   MPV_Vs_LocalAngle_Tick320= new TH2D ("MPV_Vs_LocalAngle_Tick320"   , "MPV_Vs_LocalAngle_Tick320"  ,60,0,180, 1350, 0, 1350);
   if(CheckLocalAngle)
   MPV_Vs_LocalAngle_Tick500= new TH2D ("MPV_Vs_LocalAngle_Tick500"   , "MPV_Vs_LocalAngle_Tick500"  ,60,0,180, 1350, 0, 1350);

   MPV_Vs_Error             = new TH2D ("MPV_Vs_Error"   , "MPV_Vs_Error"  ,1350, 0, 1350, 1000,0,100);
   Entries_Vs_Error         = new TH2D ("Entries_Vs_Error","Entries_Vs_Error",2000,0,2000,1000,0,100); 

   APVPairPos               = new TH2D ("APVPairPos"  , "APVPairPos", 6000, -3,3,1500,0,150);
   APVPairPos->SetMarkerStyle(21);

   NumberOfEntriesByAPVPair = new TH1D ("NumberOfEntriesByAPVPair", "NumberOfEntriesByAPVPair", 2500, 0,2500);
   HChi2OverNDF             = new TH1D ("Chi2OverNDF","Chi2OverNDF", 5000, 0,10);
   HTrackChi2OverNDF        = new TH1D ("TrackChi2OverNDF","TrackChi2OverNDF", 5000, 0,10);
   HTrackHits               = new TH1D ("TrackHits","TrackHits", 40, 0,40);

   HMPV                     = new TH1D ("MPV","MPV", 1350, 0,1350);


   HlistAPVPairsEtaR        = new TObjArray();
   HlistAPVPairsEtaR_PD     = new TObjArray();
   HISTOS                   = new TObjArray();
   
   gROOT->cd();

   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
   vector<GeomDet*> Det = tkGeom->dets();

   for(unsigned int i=0;i<Det.size();i++){
      DetId  Detid  = Det[i]->geographicalId(); 
      int    SubDet = Detid.subdetId();

      if( SubDet == StripSubdetector::TIB ||  SubDet == StripSubdetector::TID ||
          SubDet == StripSubdetector::TOB ||  SubDet == StripSubdetector::TEC  ){

          StripGeomDetUnit* DetUnit     = dynamic_cast<StripGeomDetUnit*> (Det[i]);
	  if(!DetUnit)continue;

          const StripTopology& Topo     = DetUnit->specificTopology();	
          unsigned int         NAPVPair = Topo.nstrips()/256;
	
          ModulePositionMap[Detid.rawId()] = new GlobalVector(DetUnit->position().basicVector());
          double Eta = DetUnit->position().basicVector().eta();
          double R   = DetUnit->position().basicVector().transverse();


          for(unsigned int j=0;j<NAPVPair;j++){
                TString HistoId      = Form("ChargeAPVPair %i %i",Detid.rawId(),j);
                TH1F* PointerToHisto =  new TH1F(HistoId,HistoId, 1800, 0.0, 5400.0);
                if(PointerToHisto   != NULL)  AddHisto(PointerToHisto);

                HistoId              = Form("PD %i %i",Detid.rawId(),j);
                PointerToHisto       =  new TH1F(HistoId,HistoId, 150, 0.0, 50.0);
                if(PointerToHisto   != NULL)  AddHisto( PointerToHisto );

		if(CheckLocalAngle){
                   HistoId              = Form("ChargeVsLocalAngle %i %i",Detid.rawId(),j);
                   TH2F* PointerToHisto2=  new TH2F(HistoId,HistoId, 30, 0, 90, 450, 0.0, 1350.0);
                   if(PointerToHisto2  != NULL)  AddHisto( PointerToHisto2 );
                }

		stAPVPairGain* temp = new stAPVPairGain;
		temp->DetId	    = Detid.rawId();
		temp->APVPairId     = j;
		temp->MPV	    = -1;
		temp->Gain          = -1;
		temp->Eta_R	    = new char[255];
		sprintf(temp->Eta_R,"%+05.2f_%03.0f",Eta,R);
		GainOfAPVpairs.push_back(temp);
          }

          TString HistoId2      = Form("MPVBySame_Eta_R_Module_%+05.2f_%03.0f",Eta,R);

          TH1F* PointerToLayerHisto = (TH1F*) HlistAPVPairsEtaR->FindObject(HistoId2);
	  if(PointerToLayerHisto==NULL){
		  RingCalibration->cd();
	          PointerToLayerHisto     = new TH1F(HistoId2,HistoId2, 1350, 0.0, 1350.0);
	          if(PointerToLayerHisto != NULL)  HlistAPVPairsEtaR->Add( PointerToLayerHisto );

                  HistoId2                = Form("PDBySame_Eta_R_Module_%+05.2f_%03.0f",Eta,R);
                  PointerToLayerHisto     = new TH1F(HistoId2,HistoId2, 150, 0.0, 50.0);
                  if(PointerToLayerHisto != NULL)  HlistAPVPairsEtaR_PD->Add( PointerToLayerHisto );
	           gROOT->cd();

		  APVPairPos->Fill(Eta,R);
	  }
      }
   }

   NEvent = 0;
}

void 
SiStripGainFromData::algoEndJob() {

   int I=0;
   for(map<uint32_t, GlobalVector*>::iterator it = ModulePositionMap.begin();it!=ModulePositionMap.end();it++){

//   printf("Fitting and Storing Histograms \t %6.2f%%\n",(100.0*I)/ModulePositionMap.size());I++;

   for(unsigned int i=0;i<3;i++){

      int detId   = it->first;
      int APVPair = i;
      double eta  = (it->second)->eta();
      double R    = (it->second)->transverse();

      TString HistoName      = Form("ChargeAPVPair %i %i",detId,APVPair);
      TH1F*   PointerToHisto = (TH1F*) FindHisto(HistoName.Data());
      if(PointerToHisto==NULL)continue;

      NumberOfEntriesByAPVPair->Fill(PointerToHisto->GetEntries());

      double chi2overndf = -1;
      pair<double,double> value = getPeakOfLandau(PointerToHisto, chi2overndf);
//     pair<double,double> value = getPeakOfLandau(PointerToHisto, chi2overndf,200,400);

      HMPV->Fill(value.first);

      TString HistoId2          = Form("MPVBySame_Eta_R_Module_%+05.2f_%03.0f",eta,R);
      TH1F* PointerToLayerHisto = (TH1F*) HlistAPVPairsEtaR->FindObject(HistoId2);
      if(PointerToLayerHisto   != NULL)PointerToLayerHisto->Fill(value.first);

      HistoId2                  = Form("PDBySame_Eta_R_Module_%+05.2f_%03.0f",eta,R);
      PointerToLayerHisto       = (TH1F*) HlistAPVPairsEtaR_PD->FindObject(HistoId2);
      TString  tempHistoName    = Form("PD %i %i",detId,APVPair);
      if(PointerToLayerHisto   != NULL)PointerToLayerHisto->Add( (TH1F*) FindHisto( tempHistoName.Data() ),1);
      if(PointerToLayerHisto   == NULL)printf("Cant find : PD %i %i\n",detId,APVPair);

      for(unsigned int a=0;a<GainOfAPVpairs.size();a++){
         if(GainOfAPVpairs[a]==NULL){
		cout << "bug" << endl;
		continue;
	 }

         if(GainOfAPVpairs[a]->DetId == detId && GainOfAPVpairs[a]->APVPairId == APVPair)	
	 GainOfAPVpairs[a]->MPV = value.first;	 
      }

      if(chi2overndf>0)
      HChi2OverNDF->Fill(chi2overndf);

      if(value.first!=-0.5 && value.second<MaxMPVError)
      MPV_Vs_Eta->Fill(eta,value.first);

      if(value.first!=-0.5 && value.second<MaxMPVError){
         DetId temp((uint32_t)(unsigned)detId);

         switch(temp.subdetId()){
	    case StripSubdetector::TIB :
               MPV_Vs_EtaTIB->Fill(eta,value.first);
               break;
            case StripSubdetector::TID :
               MPV_Vs_EtaTID->Fill(eta,value.first);
               break;
            case StripSubdetector::TOB :
               MPV_Vs_EtaTOB->Fill(eta,value.first);
               break;
            case StripSubdetector::TEC :
               MPV_Vs_EtaTEC->Fill(eta,value.first);
	       if(moduleThickness(detId, iSetup_)<0.04){
		  MPV_Vs_EtaTEC1->Fill(eta,value.first);
               }else{
                  MPV_Vs_EtaTEC2->Fill(eta,value.first);
               }
               break;
            default: 
               break;
         }
    }

      if(value.first!=-0.5 && value.second<MaxMPVError)
      MPV_Vs_R->Fill(R,value.first);

      if(value.first!=-0.5)
      MPV_Vs_Error->Fill(value.first, value.second);

      if(value.first!=-0.5)
      Entries_Vs_Error->Fill(PointerToHisto->GetEntries(),value.second);

      if(PointerToHisto->GetEntries()>10){
         Landau->cd();

         DetId temp((uint32_t)(unsigned)detId);
         int SubDet = temp.subdetId();
         
         TString SubdetName;
         if(SubDet==StripSubdetector::TIB)SubdetName = Form("TIB");
         if(SubDet==StripSubdetector::TOB)SubdetName = Form("TOB");
         if(SubDet==StripSubdetector::TID)SubdetName = Form("TID");
         if(SubDet==StripSubdetector::TEC)SubdetName = Form("TEC");

         TDirectory* SubDetDir   = (TDirectory*) Landau->FindObject(SubdetName);
         if(!SubDetDir)SubDetDir = Landau->mkdir(SubdetName);
         SubDetDir->cd();

         TDirectory* EtaDir = (TDirectory*) SubDetDir->FindObject(Form("Eta=%+6.2f",eta));
         if(!EtaDir)EtaDir = SubDetDir->mkdir(Form("Eta=%+6.2f",eta));
         EtaDir->cd();

         TDirectory* RDir = (TDirectory*) EtaDir->FindObject(Form("R=%8.2f",R));
         if(!RDir)   RDir = EtaDir->mkdir(Form("R=%8.2f",R));
         RDir->cd();
                  
         PointerToHisto->Write();
         gROOT->cd();
      }

      if(CheckLocalAngle){
         HistoName              = Form("ChargeVsLocalAngle %i %i",detId,APVPair);
         TH2F*   PointerToHisto2= (TH2F*) FindHisto(HistoName.Data());
         if(PointerToHisto2==NULL){printf("PointerToHisto2 is NULL\n");continue;}
         double ModuleThickness = moduleThickness(detId, iSetup_); 

         for(int j=0;j<PointerToHisto2->GetXaxis()->GetNbins();j++){
            TH1D* temp = PointerToHisto2->ProjectionY(" ",j-1,j,"e");

            double chi2overndfAngle = -1;
            pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
              
            if(tempvalue.first==-0.5)continue;
            if(ModuleThickness<0.04)MPV_Vs_LocalAngle_Tick320->Fill(PointerToHisto2->GetXaxis()->GetBinCenter(j),tempvalue.first);
            if(ModuleThickness>0.04)MPV_Vs_LocalAngle_Tick500->Fill(PointerToHisto2->GetXaxis()->GetBinCenter(j),tempvalue.first);         
         }
      }
   }}

   double MPVmean = HMPV->GetMean();
   for(unsigned int a=0;a<GainOfAPVpairs.size();a++){
      if(GainOfAPVpairs[a]==NULL){
             cout << "bug" << endl;
             continue;
      }

      GainOfAPVpairs[a]->Gain = MPVmean / GainOfAPVpairs[a]->MPV;
   }


   for(int i=0;i<HlistAPVPairsEtaR->GetEntries();i++){

      double MeanMPVvalue =((TH1F*) HlistAPVPairsEtaR->At(i))->GetMean();
      TString HistoName = ((TH1F*)HlistAPVPairsEtaR->At(i))->GetName();

      float eta;
      float R;
      sscanf(HistoName.Data(),"MPVBySame_Eta_R_Module_%f_%f",&eta,&R);
      
      Ring_MPV_Vs_Eta->Fill(eta,MeanMPVvalue);

/*      for(unsigned int a=0;a<GainOfAPVpairs.size();a++){
//         if(GainOfAPVpairs[a]->Eta == eta && GainOfAPVpairs[a]->R == R)
         char temporary[255];sprintf(temporary,"%+05.2f_%03.0f",eta,R);
         if(strcmp(GainOfAPVpairs[a]->Eta_R,temporary) == 0 && GainOfAPVpairs[a]->MPV>0)
         GainOfAPVpairs[a]->Gain = MeanMPVvalue / GainOfAPVpairs[a]->MPV;
      }
*/


      TH1F*  tmp    = (TH1F*)HlistAPVPairsEtaR_PD->FindObject(Form("PDBySame_Eta_R_Module_%+05.2f_%03.0f",eta,R));
      double PDMean = 0;
      if(tmp!=NULL)  PDMean = tmp->GetMean();
      if(tmp==NULL) printf("Can't find : PDBySame_Eta_R_Module_%+05.2f_%03.0f\n",eta,R);

      PD_Vs_Eta->Fill(eta,PDMean);
      PD_Vs_R  ->Fill(R  ,PDMean);      
   }

/*
   for(int j=0;j<Charge_Vs_PathLength->GetXaxis()->GetNbins();j++){
      TH1D* temp = Charge_Vs_PathLength->ProjectionY(" ",j-1,j,"e");

      double chi2overndfAngle = -1;
      pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
      if(tempvalue.first==-0.5)continue;

      printf("%6.2f --> MPV = %8.2f\n",Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j),tempvalue.first/Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j));
//      MPV_Vs_PathLength->Fill(Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j),tempvalue.first/Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength->SetBinContent(j, tempvalue.first /Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength->SetBinError  (j, tempvalue.second/Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j));

   }

   for(int j=0;j<Charge_Vs_PathLength320->GetXaxis()->GetNbins();j++){
      TH1D* temp = Charge_Vs_PathLength320->ProjectionY(" ",j-1,j,"e");

      double chi2overndfAngle = -1;
      pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
      if(tempvalue.first==-0.5)continue;

      printf("%6.2f --> MPV = %8.2f\n",Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j),tempvalue.first/Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j));
//      MPV_Vs_PathLength320->Fill(Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j),tempvalue.first/Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength320->SetBinContent(j, tempvalue.first /Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength320->SetBinError  (j, tempvalue.second/Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j));

   }

   for(int j=0;j<Charge_Vs_PathLength500->GetXaxis()->GetNbins();j++){
      TH1D* temp = Charge_Vs_PathLength500->ProjectionY(" ",j-1,j,"e");

      double chi2overndfAngle = -1;
      pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
      if(tempvalue.first==-0.5)continue;

      printf("%6.2f --> MPV = %8.2f\n",Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j),tempvalue.first/Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j));
//      MPV_Vs_PathLength500->Fill(Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j),tempvalue.first/Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength500->SetBinContent(j, tempvalue.first /Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength500->SetBinError  (j, tempvalue.second/Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j));
   }

   for(int j=1;j<Charge_Vs_TransversAngle->GetXaxis()->GetNbins();j++){
      TH1D* temp = Charge_Vs_TransversAngle->ProjectionY(" ",j-1,j,"e");

      double chi2overndfAngle = -1;
      pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
      if(tempvalue.first==-0.5)continue;

      printf("TransAngle : %6.2f --> MPV = %8.2f\n",Charge_Vs_TransversAngle->GetXaxis()->GetBinCenter(j),tempvalue.first);
//      MPV_Vs_TransversAngle->Fill(Charge_Vs_TransversAngle->GetXaxis()->GetBinCenter(j),tempvalue.first);
      MPV_Vs_TransversAngle->SetBinContent(j, tempvalue.first );
      MPV_Vs_TransversAngle->SetBinError  (j, tempvalue.second);
   }

   for(int j=1;j<Charge_Vs_Alpha->GetXaxis()->GetNbins();j++){
      TH1D* temp = Charge_Vs_Alpha->ProjectionY(" ",j-1,j,"e");

      double chi2overndfAngle = -1;
      pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
      if(tempvalue.first==-0.5)continue;

      printf("Alpha : %6.2f --> MPV = %8.2f\n",Charge_Vs_Alpha->GetXaxis()->GetBinCenter(j),tempvalue.first/Charge_Vs_Alpha->GetXaxis()->GetBinCenter(j));
//      MPV_Vs_Alpha->Fill(Charge_Vs_Alpha->GetXaxis()->GetBinCenter(j),tempvalue.first);
      MPV_Vs_Alpha->SetBinContent(j, tempvalue.first );
      MPV_Vs_Alpha->SetBinError  (j, tempvalue.second);
   }


   for(int j=1;j<Charge_Vs_Beta->GetXaxis()->GetNbins();j++){
      TH1D* temp = Charge_Vs_Beta->ProjectionY(" ",j-1,j,"e");

      double chi2overndfAngle = -1;
      pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
      if(tempvalue.first==-0.5)continue;

      printf("Beta : %6.2f --> MPV = %8.2f\n",Charge_Vs_Beta->GetXaxis()->GetBinCenter(j),tempvalue.first/Charge_Vs_Beta->GetXaxis()->GetBinCenter(j));
//      MPV_Vs_Beta->Fill(Charge_Vs_Beta->GetXaxis()->GetBinCenter(j),tempvalue.first);
      MPV_Vs_Beta->SetBinContent(j, tempvalue.first );
      MPV_Vs_Beta->SetBinError  (j, tempvalue.second);
   }


*/

   printf("NumberOfRings = %i \n",HlistAPVPairsEtaR->GetEntries());

   FILE* Gains = fopen(OutputGains.c_str(),"w");
   fprintf(Gains,"NEvents = %i\n",NEvent);
   fprintf(Gains,"Number of APVs = %i\n",GainOfAPVpairs.size());
   for(unsigned int a=0;a<GainOfAPVpairs.size();a++){
      if(GainOfAPVpairs[a]==NULL){
	     printf("Bug\n");
             fprintf(Gains,"Bug\n");
             continue;
      }

      fprintf(Gains,"%i | %i | %f\n", GainOfAPVpairs[a]->DetId,GainOfAPVpairs[a]->APVPairId,GainOfAPVpairs[a]->Gain);

      float Eta = -99;
      float R   = -99;
      sscanf(GainOfAPVpairs[a]->Eta_R,"%f_%f",&Eta,&R);

      MPV_Vs_Eta_Calib->Fill(Eta,GainOfAPVpairs[a]->MPV*GainOfAPVpairs[a]->Gain);
      MPV_Vs_R_Calib  ->Fill(R  ,GainOfAPVpairs[a]->MPV*GainOfAPVpairs[a]->Gain);
   }

   fclose(Gains);

   Output->cd();
   Output->Write();
   Output->Close();
}



void
SiStripGainFromData::algoAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   NEvent++;
//   printf("-*-*-*-*-*-*- NewEvent -*-*-*-*-*-*- \t\t%i\n",NEvent);

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

      printf("NHits = %i\n",traj.foundHits());

      vector<TrajectoryMeasurement> measurements = traj.measurements();
      HTrackHits->Fill(traj.foundHits());
      if(traj.foundHits()<(int)MinTrackHits)continue;

      //BEGIN TO COMPUTE #MATCHEDRECHIT IN THE TRACK
      int NMatchedHit = 0;
      for(vector<TrajectoryMeasurement>::const_iterator measurement_it = measurements.begin(); measurement_it!=measurements.end(); measurement_it++){
         TrajectoryStateOnSurface trajState = measurement_it->updatedState();
         if( !trajState.isValid() ) continue;
         const TrackingRecHit*         hit               = (*measurement_it->recHit()).hit();
         const SiStripMatchedRecHit2D* sistripmatchedhit = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);
//	 if(sistripmatchedhit)NMatchedHit++;
/**/     NMatchedHit++;

      }
      //END TO COMPUTE #MATCHEDRECHIT IN THE TRACK

      if(NMatchedHit<2){
	 printf("NOT ENOUGH MATCHED RECHITS : %i\n",NMatchedHit);
 	 continue;
      }

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


void SiStripGainFromData::PrintHisto()
{

   for(int i=0;i<=HISTOS->GetLast();i++)
   {
	TObjArray* SubDetArray = (TObjArray*) HISTOS->At(i);
        if(SubDetArray==NULL)continue;
	printf("%s --> Number of Entries = %i\n",SubDetArray->GetName(), SubDetArray->GetEntries());

	for(int j=0;j<=SubDetArray->GetLast();j++)
        {
	   TObjArray* SubSubDetArray = (TObjArray*) SubDetArray->At(j);
	   if(SubSubDetArray==NULL)continue;
           printf("\t%04i : %s --> Number of Entries = %i\n",j,SubSubDetArray->GetName(), SubSubDetArray->GetEntries());	   
        }

   }
}

TObject* SiStripGainFromData::FindHisto(const char* Name)
{

   if(Name==NULL)return NULL;

   char  Txt[255];
   int   detId;
   int   APVPair;
   sscanf( Name, "%s %i %i",Txt,&detId,&APVPair);

   DetId temp((uint32_t)(unsigned)detId);
   int   SubDet = temp.subdetId();
   int   SubSubDet = detId%50;

   TObjArray* SubDetArray = (TObjArray*) HISTOS->FindObject(Form("SubDet%i",SubDet));
   if(SubDetArray==NULL){
      printf("Can't find a TObjArray with Name SubDet%i\n",SubDet);
      return NULL;
   }

   TObjArray* SubSubDetArray = (TObjArray*) SubDetArray->FindObject(Form("SubSubDet%2i",SubSubDet));
   if(SubSubDetArray==NULL){
      printf("Can't find a TObjArray with Name SubSubDet%2i\n",SubSubDet);
      return NULL;
   }

   TObjArray* DetIdArray = (TObjArray*) SubSubDetArray->FindObject(Form("DetId%i",detId));
   if(DetIdArray==NULL){
      printf("Can't find a TObjArray with Name DetId%i\n",detId);
      return NULL;
   }

   return DetIdArray->FindObject(Name);
}



void SiStripGainFromData::AddHisto(TObject* Histo)
{
   if(Histo==NULL)return;

   char  Txt[255];
   int   detId;
   int   APVPair;
   sscanf( Histo->GetName(), "%s %i %i",Txt,&detId,&APVPair);

   DetId temp((uint32_t)(unsigned)detId);
   int   SubDet = temp.subdetId();
   int   SubSubDet = detId%50;

   TObjArray* SubDetArray = (TObjArray*) HISTOS->FindObject(Form("SubDet%i",SubDet));
   if(SubDetArray==NULL){
      SubDetArray = new TObjArray();
      SubDetArray->SetName(Form("SubDet%i",SubDet));
      HISTOS->Add(SubDetArray);
      printf("A new TObjArray has been created for SubDet=%i\n",SubDet);
   }

   TObjArray* SubSubDetArray = (TObjArray*) SubDetArray->FindObject(Form("SubSubDet%2i",SubSubDet));
   if(SubSubDetArray==NULL){
      SubSubDetArray = new TObjArray();
      SubSubDetArray->SetName(Form("SubSubDet%2i",SubSubDet));
      SubDetArray->Add(SubSubDetArray);
//      printf("A new TObjArray has been created for SubSubDet=%2i\n",SubSubDet);
   }

   TObjArray* DetIdArray = (TObjArray*) SubSubDetArray->FindObject(Form("DetId%i",detId));
   if(DetIdArray==NULL){
      DetIdArray = new TObjArray();
      DetIdArray->SetName(Form("DetId%i",detId));
      SubSubDetArray->Add(DetIdArray);
   }

   DetIdArray->Add(Histo);
}

double
SiStripGainFromData::ComputeChargeOverPath(const SiStripRecHit2D* sistripsimplehit,TrajectoryStateOnSurface trajState, const edm::EventSetup* iSetup,  const Track* track, double trajChi2OverN)
{
   printf("ChargeOverPath\n");

   LocalVector          trackDirection = trajState.localDirection();
   double                  cosine      = trackDirection.z()/trackDirection.mag();
   const SiStripCluster*   Cluster     = (sistripsimplehit->cluster()).get();
   uint32_t                DetId       = Cluster->geographicalId();
//   const vector<uint16_t>& Ampls       = Cluster->amplitudes();
   const vector<uint8_t>& Ampls       = Cluster->amplitudes();
// double                  Width       = moduleWidth    (DetId, iSetup);
   double                  Thickness   = moduleThickness(DetId, iSetup);
   int                     FirstStrip  = Cluster->firstStrip();
   int                     APVPairId   = FirstStrip/256;
   bool                    Saturation  = false;
   bool                    Overlaping  = false;
   int                     Charge      = 0;
// double		   TrajP       = trajState.globalMomentum().mag();

   if(!IsFarFromBorder(trajState.localPosition(),DetId, iSetup))return -1;

   if(FirstStrip==0                                 )Overlaping=true;
   if(FirstStrip<=255 && FirstStrip+Ampls.size()>255)Overlaping=true;
   if(FirstStrip<=511 && FirstStrip+Ampls.size()>511)Overlaping=true;
   if(FirstStrip+Ampls.size()==511                  )Overlaping=true;
   if(FirstStrip+Ampls.size()==767                  )Overlaping=true;

   if(Overlaping)return -1;


   for(unsigned int a=0;a<Ampls.size();a++){Charge+=Ampls[a];if(Ampls[a]>=254)Saturation=true;}
   double ClusterChargeOverPath        =  ((double)Charge) * fabs(cosine) / ( 10. * Thickness);

   if(Ampls.size()>MaxNrStrips)return -1;
   if(Saturation && !AllowSaturation)return -1;


   double path = (10.0*Thickness)/fabs(cosine);
   Charge_Vs_PathLength->Fill(path, Charge );

   if(Thickness<0.04) Charge_Vs_PathLength320->Fill(path, Charge );
   if(Thickness>0.04) Charge_Vs_PathLength500->Fill(path, Charge );

   if(path>0.4 && path<0.45)Charge_Vs_TransversAngle->Fill(atan2(trackDirection.y(),trackDirection.x())*(180/3.14159265),Charge/path);

   double alpha = acos(trackDirection.x() / sqrt( pow(trackDirection.x(),2) +  pow(trackDirection.z(),2) ) ) * (180/3.14159265);
   if(path>0.4 && path<0.45)Charge_Vs_Alpha->Fill(alpha,Charge/path);

   double beta = acos(trackDirection.y() / sqrt( pow(trackDirection.x(),2) +  pow(trackDirection.z(),2) ) ) * (180/3.14159265);
   if(path>0.4 && path<0.45)Charge_Vs_Beta->Fill(beta,Charge/path);


   NStrips_Vs_Alpha->Fill(alpha,Ampls.size());
   NStrips_Vs_Beta->Fill(alpha,Ampls.size());


   TString HistoId        = Form("ChargeAPVPair %i %i",DetId,APVPairId);
   TH1F*   PointerToHisto = (TH1F*) FindHisto(HistoId.Data());

   if(PointerToHisto == NULL){
      printf("Histo ChargeAPVPair %i %i does not exist\nExit Program\n",DetId,APVPairId);
      exit(0);
   }
   PointerToHisto->Fill(ClusterChargeOverPath);

   HistoId            = Form("PD %i %i",DetId,APVPairId);
   PointerToHisto     = (TH1F*) FindHisto(HistoId.Data());
   if(PointerToHisto != NULL)PointerToHisto->Fill(trajState.globalMomentum().mag());
   if(PointerToHisto == NULL)printf("can't find : PD %i %i\n",DetId,APVPairId);

   if(CheckLocalAngle){
      HistoId                = Form("ChargeVsLocalAngle %i %i",DetId,APVPairId);
      TH2F*   PointerToHisto2= (TH2F*) FindHisto(HistoId.Data());
      if(PointerToHisto2    != NULL)   PointerToHisto2->Fill(acos(cosine)*(180/3.14159265),ClusterChargeOverPath);
   }


   unsigned int NHighStrip = 0;
   for(unsigned int a=0;a<Ampls.size();a++){if(Ampls[a]>=20)NHighStrip++;}
   NHighStripInCluster->Fill(NHighStrip);

   if(NHighStrip==1)   Charge_Vs_PathLength_Sat  ->Fill(path, Charge );
   if(NHighStrip==2)   Charge_Vs_PathLength_NoSat->Fill(path, Charge );
 
   //Handle< DetSetVector<SiStripCluster> > clusterColl_h;
   //iEvent_->getByLabel("siStripClusters", clusterColl_h);
   //DetSetVector<SiStripCluster> clusterColl = *clusterColl_h.product();
   //DetSet<SiStripCluster> clustersForThisModule = clusterColl[DetId];
   //unsigned int NClusters = clustersForThisModule.size();

   ChargeDistrib->Fill(ClusterChargeOverPath);

   return ClusterChargeOverPath;
}

/*
double SiStripGainFromData::moduleWidth(const uint32_t detid, const edm::EventSetup* iSetup)
{ //dk: copied from A. Giammanco and hacked,  module_width values : 10.49 12.03 6.144 7.14 9.3696
  edm::ESHandle<TrackerGeometry> tkGeom; iSetup->get<TrackerDigiGeometryRecord>().get( tkGeom );     
  double module_width=0.;
  const GeomDetUnit* it = tkGeom->idToDetUnit(DetId(detid));
  if (dynamic_cast<const StripGeomDetUnit*>(it)==0 && dynamic_cast<const PixelGeomDetUnit*>(it)==0) {
    std::cout << "this detID doesn't seem to belong to the Tracker" << std::endl;    
  }else{
    module_width = it->surface().bounds().width();
  }
  return module_width;
}
*/

double SiStripGainFromData::moduleThickness(const uint32_t detid, const edm::EventSetup* iSetup) 
{ //dk: copied from A. Giammanco and hacked
  edm::ESHandle<TrackerGeometry> tkGeom; iSetup->get<TrackerDigiGeometryRecord>().get( tkGeom );
  double module_thickness=0.;
  const GeomDetUnit* it = tkGeom->idToDetUnit(DetId(detid));
  if (dynamic_cast<const StripGeomDetUnit*>(it)==0 && dynamic_cast<const PixelGeomDetUnit*>(it)==0) {
    std::cout << "this detID doesn't seem to belong to the Tracker" << std::endl;
  }else{
    module_thickness = it->surface().bounds().thickness();
  }

  return module_thickness;
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


pair<double,double>  SiStripGainFromData::getPeakOfLandau(TH1* InputHisto, double& chiOverNdf)
{
   return getPeakOfLandau(InputHisto, chiOverNdf, 0, 5400);
}


pair<double,double>  SiStripGainFromData::getPeakOfLandau(TH1* InputHisto, double& chiOverNdf, double LowRange, double HighRange)
{ 
   double adcs           = -0.5; 
   double error          = 0.; 
   double nr_of_entries  = InputHisto->GetEntries();

   if( (unsigned int)nr_of_entries < MinNrEntries){
      return std::make_pair(adcs,error);
   }

   // perform fit with standard landau
   TF1* MyLandau = new TF1("MyLandau","landau",LowRange, HighRange);
   MyLandau->SetParameter("MPV",300);

   InputHisto->Fit("MyLandau","QR WW");
   TF1 * fitfunction = (TF1*) InputHisto->GetListOfFunctions()->First();

   // MPV is parameter 1 (0=constant, 1=MPV, 2=Sigma)
           adcs        = fitfunction->GetParameter("MPV");
           error       = fitfunction->GetParError(1);
    double chi2        = fitfunction->GetChisquare();
    double ndf         = fitfunction->GetNDF();
    double chi2overndf = chi2 / ndf;
    chiOverNdf = chi2 / ndf;

    // if still wrong, give up
    if(adcs<2. || chi2overndf>MaxChi2OverNDF){
       adcs = -0.5; error = 0.;
    }
  
    return std::make_pair(adcs,error);
}




SiStripApvGain* SiStripGainFromData::getNewObject() 
{
  printf("SiStripGainFromData::getNewObject called\n");
/*
   int I=0;
   for(map<uint32_t, GlobalVector*>::iterator it = ModulePositionMap.begin();it!=ModulePositionMap.end();it++){

   printf("Fitting and Storing Histograms \t %6.2f%%\n",(100.0*I)/ModulePositionMap.size());I++;

   for(unsigned int i=0;i<3;i++){

      int detId   = it->first;
      int APVPair = i;
      double eta  = (it->second)->eta();
      double R    = (it->second)->transverse();

      TString HistoName      = Form("ChargeAPVPair %i %i",detId,APVPair);
      TH1F*   PointerToHisto = (TH1F*) FindHisto(HistoName.Data());
      if(PointerToHisto==NULL)continue;

      NumberOfEntriesByAPVPair->Fill(PointerToHisto->GetEntries());

      double chi2overndf = -1;
      pair<double,double> value = getPeakOfLandau(PointerToHisto, chi2overndf);
//     pair<double,double> value = getPeakOfLandau(PointerToHisto, chi2overndf,200,400);


      TString HistoId2          = Form("MPVBySame_Eta_R_Module_%+05.2f_%03.0f",eta,R);
      TH1F* PointerToLayerHisto = (TH1F*) HlistAPVPairsEtaR->FindObject(HistoId2);
      if(PointerToLayerHisto   != NULL)PointerToLayerHisto->Fill(value.first);

      HistoId2                  = Form("PDBySame_Eta_R_Module_%+05.2f_%03.0f",eta,R);
      PointerToLayerHisto       = (TH1F*) HlistAPVPairsEtaR_PD->FindObject(HistoId2);
      TString  tempHistoName    = Form("PD %i %i",detId,APVPair);
      if(PointerToLayerHisto   != NULL)PointerToLayerHisto->Add( (TH1F*) FindHisto( tempHistoName.Data() ),1);
      if(PointerToLayerHisto   == NULL)printf("Cant find : PD %i %i\n",detId,APVPair);

      for(unsigned int a=0;a<GainOfAPVpairs.size();a++){
         if(GainOfAPVpairs[a]==NULL){
		cout << "bug" << endl;
		continue;
	 }

         if(GainOfAPVpairs[a]->DetId == detId && GainOfAPVpairs[a]->APVPairId == APVPair)	
	 GainOfAPVpairs[a]->MPV = value.first;	 
      }

      if(chi2overndf>0)
      HChi2OverNDF->Fill(chi2overndf);

      if(value.first!=-0.5 && value.second<MaxMPVError)
      MPV_Vs_Eta->Fill(eta,value.first);

      if(value.first!=-0.5 && value.second<MaxMPVError){
         DetId temp((uint32_t)(unsigned)detId);

         switch(temp.subdetId()){
	    case StripSubdetector::TIB :
               MPV_Vs_EtaTIB->Fill(eta,value.first);
               break;
            case StripSubdetector::TID :
               MPV_Vs_EtaTID->Fill(eta,value.first);
               break;
            case StripSubdetector::TOB :
               MPV_Vs_EtaTOB->Fill(eta,value.first);
               break;
            case StripSubdetector::TEC :
               MPV_Vs_EtaTEC->Fill(eta,value.first);
	       if(moduleThickness(detId, iSetup_)<0.04){
		  MPV_Vs_EtaTEC1->Fill(eta,value.first);
               }else{
                  MPV_Vs_EtaTEC2->Fill(eta,value.first);
               }
               break;
            default: 
               break;
         }
    }

      if(value.first!=-0.5 && value.second<MaxMPVError)
      MPV_Vs_R->Fill(R,value.first);

      if(value.first!=-0.5)
      MPV_Vs_Error->Fill(value.first, value.second);

      if(value.first!=-0.5)
      Entries_Vs_Error->Fill(PointerToHisto->GetEntries(),value.second);

      if(PointerToHisto->GetEntries()>10){
         Landau->cd();

         DetId temp((uint32_t)(unsigned)detId);
         int SubDet = temp.subdetId();
         
         TString SubdetName;
         if(SubDet==StripSubdetector::TIB)SubdetName = Form("TIB");
         if(SubDet==StripSubdetector::TOB)SubdetName = Form("TOB");
         if(SubDet==StripSubdetector::TID)SubdetName = Form("TID");
         if(SubDet==StripSubdetector::TEC)SubdetName = Form("TEC");

         TDirectory* SubDetDir   = (TDirectory*) Landau->FindObject(SubdetName);
         if(!SubDetDir)SubDetDir = Landau->mkdir(SubdetName);
         SubDetDir->cd();

         TDirectory* EtaDir = (TDirectory*) SubDetDir->FindObject(Form("Eta=%+6.2f",eta));
         if(!EtaDir)EtaDir = SubDetDir->mkdir(Form("Eta=%+6.2f",eta));
         EtaDir->cd();

         TDirectory* RDir = (TDirectory*) EtaDir->FindObject(Form("R=%8.2f",R));
         if(!RDir)   RDir = EtaDir->mkdir(Form("R=%8.2f",R));
         RDir->cd();
                  
         PointerToHisto->Write();
         gROOT->cd();
      }

      if(CheckLocalAngle){
         HistoName              = Form("ChargeVsLocalAngle %i %i",detId,APVPair);
         TH2F*   PointerToHisto2= (TH2F*) FindHisto(HistoName.Data());
         if(PointerToHisto2==NULL){printf("PointerToHisto2 is NULL\n");continue;}
         double ModuleThickness = moduleThickness(detId, iSetup_); 

         for(int j=0;j<PointerToHisto2->GetXaxis()->GetNbins();j++){
            TH1D* temp = PointerToHisto2->ProjectionY(" ",j-1,j,"e");

            double chi2overndfAngle = -1;
            pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
              
            if(tempvalue.first==-0.5)continue;
            if(ModuleThickness<0.04)MPV_Vs_LocalAngle_Tick320->Fill(PointerToHisto2->GetXaxis()->GetBinCenter(j),tempvalue.first);
            if(ModuleThickness>0.04)MPV_Vs_LocalAngle_Tick500->Fill(PointerToHisto2->GetXaxis()->GetBinCenter(j),tempvalue.first);         
         }
      }
   }}


   for(int i=0;i<HlistAPVPairsEtaR->GetEntries();i++){

      double MeanMPVvalue =((TH1F*) HlistAPVPairsEtaR->At(i))->GetMean();
      TString HistoName = ((TH1F*)HlistAPVPairsEtaR->At(i))->GetName();

      float eta;
      float R;
      sscanf(HistoName.Data(),"MPVBySame_Eta_R_Module_%f_%f",&eta,&R);
      
      Ring_MPV_Vs_Eta->Fill(eta,MeanMPVvalue);

      for(unsigned int a=0;a<GainOfAPVpairs.size();a++){
         char temporary[255];sprintf(temporary,"%+05.2f_%03.0f",eta,R);
         if(strcmp(GainOfAPVpairs[a]->Eta_R,temporary) == 0 && GainOfAPVpairs[a]->MPV>0)
         GainOfAPVpairs[a]->Gain = MeanMPVvalue / GainOfAPVpairs[a]->MPV;
      }


      TH1F*  tmp    = (TH1F*)HlistAPVPairsEtaR_PD->FindObject(Form("PDBySame_Eta_R_Module_%+05.2f_%03.0f",eta,R));
      double PDMean = 0;
      if(tmp!=NULL)  PDMean = tmp->GetMean();
      if(tmp==NULL) printf("Can't find : PDBySame_Eta_R_Module_%+05.2f_%03.0f\n",eta,R);

      PD_Vs_Eta->Fill(eta,PDMean);
      PD_Vs_R  ->Fill(R  ,PDMean);      
   }


   for(int j=0;j<Charge_Vs_PathLength->GetXaxis()->GetNbins();j++){
      TH1D* temp = Charge_Vs_PathLength->ProjectionY(" ",j-1,j,"e");

      double chi2overndfAngle = -1;
      pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
      if(tempvalue.first==-0.5)continue;

      printf("%6.2f --> MPV = %8.2f\n",Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j),tempvalue.first/Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength->SetBinContent(j, tempvalue.first /Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength->SetBinError  (j, tempvalue.second/Charge_Vs_PathLength->GetXaxis()->GetBinCenter(j));

   }

   for(int j=0;j<Charge_Vs_PathLength320->GetXaxis()->GetNbins();j++){
      TH1D* temp = Charge_Vs_PathLength320->ProjectionY(" ",j-1,j,"e");

      double chi2overndfAngle = -1;
      pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
      if(tempvalue.first==-0.5)continue;

      printf("%6.2f --> MPV = %8.2f\n",Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j),tempvalue.first/Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength320->SetBinContent(j, tempvalue.first /Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength320->SetBinError  (j, tempvalue.second/Charge_Vs_PathLength320->GetXaxis()->GetBinCenter(j));

   }

   for(int j=0;j<Charge_Vs_PathLength500->GetXaxis()->GetNbins();j++){
      TH1D* temp = Charge_Vs_PathLength500->ProjectionY(" ",j-1,j,"e");

      double chi2overndfAngle = -1;
      pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
      if(tempvalue.first==-0.5)continue;

      printf("%6.2f --> MPV = %8.2f\n",Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j),tempvalue.first/Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength500->SetBinContent(j, tempvalue.first /Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j));
      MPV_Vs_PathLength500->SetBinError  (j, tempvalue.second/Charge_Vs_PathLength500->GetXaxis()->GetBinCenter(j));
   }

   for(int j=1;j<Charge_Vs_TransversAngle->GetXaxis()->GetNbins();j++){
      TH1D* temp = Charge_Vs_TransversAngle->ProjectionY(" ",j-1,j,"e");

      double chi2overndfAngle = -1;
      pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
      if(tempvalue.first==-0.5)continue;

      printf("TransAngle : %6.2f --> MPV = %8.2f\n",Charge_Vs_TransversAngle->GetXaxis()->GetBinCenter(j),tempvalue.first);
      MPV_Vs_TransversAngle->SetBinContent(j, tempvalue.first /Charge_Vs_TransversAngle->GetXaxis()->GetBinCenter(j));
      MPV_Vs_TransversAngle->SetBinError  (j, tempvalue.second/Charge_Vs_TransversAngle->GetXaxis()->GetBinCenter(j));
   }

   for(int j=1;j<Charge_Vs_Alpha->GetXaxis()->GetNbins();j++){
      TH1D* temp = Charge_Vs_Alpha->ProjectionY(" ",j-1,j,"e");

      double chi2overndfAngle = -1;
      pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
      if(tempvalue.first==-0.5)continue;

      printf("Alpha : %6.2f --> MPV = %8.2f\n",Charge_Vs_Alpha->GetXaxis()->GetBinCenter(j),tempvalue.first/Charge_Vs_Alpha->GetXaxis()->GetBinCenter(j));
      MPV_Vs_Alpha->SetBinContent(j, tempvalue.first /Charge_Vs_Alpha->GetXaxis()->GetBinCenter(j));
      MPV_Vs_Alpha->SetBinError  (j, tempvalue.second/Charge_Vs_Alpha->GetXaxis()->GetBinCenter(j));
   }


   for(int j=1;j<Charge_Vs_Beta->GetXaxis()->GetNbins();j++){
      TH1D* temp = Charge_Vs_Beta->ProjectionY(" ",j-1,j,"e");

      double chi2overndfAngle = -1;
      pair<double,double> tempvalue = getPeakOfLandau(temp,chi2overndfAngle);
      if(tempvalue.first==-0.5)continue;

      printf("Beta : %6.2f --> MPV = %8.2f\n",Charge_Vs_Beta->GetXaxis()->GetBinCenter(j),tempvalue.first/Charge_Vs_Beta->GetXaxis()->GetBinCenter(j));
      MPV_Vs_Beta->SetBinContent(j, tempvalue.first /Charge_Vs_Beta->GetXaxis()->GetBinCenter(j));
      MPV_Vs_Beta->SetBinError  (j, tempvalue.second/Charge_Vs_Beta->GetXaxis()->GetBinCenter(j));
   }
*/

   // Save On DB

   SiStripApvGain * obj = new SiStripApvGain();
   std::vector<float>* theSiStripVector = NULL;
   int PreviousDetId = -1; 
   for(unsigned int a=0;a<GainOfAPVpairs.size();a++){
      if(GainOfAPVpairs[a]==NULL){
             printf("Bug\n");
             continue;
      }

      if(GainOfAPVpairs[a]->DetId != PreviousDetId){
        if(theSiStripVector!=NULL){
	   SiStripApvGain::Range range(theSiStripVector->begin(),theSiStripVector->end());
	   if ( !obj->put(PreviousDetId,range) )  printf("Bug to put detId = %i\n",PreviousDetId);
	}

	theSiStripVector = new std::vector<float>;
        PreviousDetId = GainOfAPVpairs[a]->DetId;

//        printf("N ");
      }else{
 //	  printf("  ");
      }



//      printf("%i | %i | %f", GainOfAPVpairs[a]->DetId,GainOfAPVpairs[a]->APVPairId,GainOfAPVpairs[a]->Gain);

      theSiStripVector->push_back(GainOfAPVpairs[a]->Gain);//printf("X");
      theSiStripVector->push_back(GainOfAPVpairs[a]->Gain);//printf("X");
//      printf("\n");

//      theSiStripVector->push_back(GainOfAPVpairs[a]->MPV);
   }

    if(theSiStripVector!=NULL){
       SiStripApvGain::Range range(theSiStripVector->begin(),theSiStripVector->end());
       if ( !obj->put(PreviousDetId,range) )  printf("Bug to put detId = %i\n",PreviousDetId);
    }


//  printf("NAPVPair = %i\n",GainOfAPVpairs.size());

  return obj;
}



DEFINE_FWK_MODULE(SiStripGainFromData);
