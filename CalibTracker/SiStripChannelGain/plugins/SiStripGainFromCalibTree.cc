// Original Author:  Loic QUERTENMONT
//         Created:  Mon Nov  16 08:55:18 CET 2009

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

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
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"

#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"


#include "TFile.h"
#include "TObjString.h"
#include "TString.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TF1.h"
#include "TROOT.h"
#include "TTree.h"
#include "TChain.h"

#include <ext/hash_map>



using namespace edm;
using namespace reco;
using namespace std;
using namespace __gnu_cxx;
using __gnu_cxx::hash_map;
using __gnu_cxx::hash;

struct stAPVGain{
   unsigned int Index; 
   int          Bin;
   unsigned int DetId;
   unsigned int APVId;
   unsigned int SubDet;
   float        x;
   float        y;
   float        z;
   float 	Eta;
   float 	R;
   float 	Phi;
   float  	Thickness;
   double 	FitMPV;
   double 	FitMPVErr;
   double 	FitWidth;
   double 	FitWidthErr;
   double 	FitChi2;
   double 	FitNorm;
   double 	Gain;
   double       CalibGain;
   double 	PreviousGain;
   double 	PreviousGainTick;
   double 	NEntries;
   TH1F*	HCharge;
   TH1F*        HChargeN;
   bool         isMasked;
};

class SiStripGainFromCalibTree : public ConditionDBWriter<SiStripApvGain> {
   public:
      explicit SiStripGainFromCalibTree(const edm::ParameterSet&);
      ~SiStripGainFromCalibTree();


   private:

  
      virtual void algoBeginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
      virtual void algoEndRun  (const edm::Run& run, const edm::EventSetup& iSetup) override;
      virtual void algoBeginJob      (const edm::EventSetup& iSetup) override;
      virtual void algoEndJob        () override;
      virtual void algoAnalyze       (const edm::Event &, const edm::EventSetup &) override;

              void merge(TH2* A, TH2* B); //needed to add histograms with different number of bins
              void algoAnalyzeTheTree();
              void algoComputeMPVandGain();

              bool IsFarFromBorder(TrajectoryStateOnSurface* trajState, const uint32_t detid, const edm::EventSetup* iSetup);
              void getPeakOfLandau(TH1* InputHisto, double* FitResults, double LowRange=50, double HighRange=5400);
              bool IsGoodLandauFit(double* FitResults); 
              void storeOnTree(TFileService* tfs);
              void MakeCalibrationMap();
              bool produceTagFilter();



      SiStripApvGain* getNewObject() override;

      TFileService *tfs;
      DQMStore* dbe;
      bool         harvestingMode;
      double       MinNrEntries;
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
      bool         Validation;
      bool         OldGainRemoving;
      int          CalibrationLevel;

      bool         saveSummary;
      bool         useCalibration;
      string       m_calibrationPath;

      double tagCondition_NClusters;
      double tagCondition_GoodFrac;

      edm::InputTag theTracksLabel;
      std::string  AlgoMode;
      std::string  OutputGains;
      vector<string> VInputFiles;

      MonitorElement*        Charge_Vs_Index;
      MonitorElement*        Charge_Vs_Index_Absolute;
      MonitorElement*        Charge_Vs_PathlengthTIB;
      MonitorElement*        Charge_Vs_PathlengthTOB;
      MonitorElement*        Charge_Vs_PathlengthTIDP;
      MonitorElement*        Charge_Vs_PathlengthTIDM;
      MonitorElement*        Charge_Vs_PathlengthTECP1;
      MonitorElement*        Charge_Vs_PathlengthTECP2;
      MonitorElement*        Charge_Vs_PathlengthTECM1;
      MonitorElement*        Charge_Vs_PathlengthTECM2;

      unsigned int NEvent;    
      unsigned int NTrack;
      unsigned int NClusterStrip;
      unsigned int NClusterPixel;
               int NStripAPVs;
               int NPixelDets;
      unsigned int SRun;
      unsigned int ERun;
      unsigned int GOOD;
      unsigned int BAD;
      unsigned int MASKED;

      edm::EDGetTokenT<edm::View<reco::Track>> tracksToken;
      edm::EDGetTokenT<TrajTrackAssociationCollection> associationsToken;

   private :
      class isEqual{
         public:
		 template <class T> bool operator () (const T& PseudoDetId1, const T& PseudoDetId2) { return PseudoDetId1==PseudoDetId2; }
      };

      std::vector<stAPVGain*> APVsCollOrdered;
      __gnu_cxx::hash_map<unsigned int, stAPVGain*,  __gnu_cxx::hash<unsigned int>, isEqual > APVsColl;
};

void SiStripGainFromCalibTree::merge(TH2* A, TH2* B){
   if(A->GetNbinsX() ==  B->GetNbinsX()){
      A->Add(B);
   }else{
      for(int x=0;x<=B->GetNbinsX()+1; x++){
      for(int y=0;y<=B->GetNbinsY()+1; y++){
         A->SetBinContent(x,y,A->GetBinContent(x,y)+B->GetBinContent(x,y));
      }}
   }
}



SiStripGainFromCalibTree::SiStripGainFromCalibTree(const edm::ParameterSet& iConfig) : ConditionDBWriter<SiStripApvGain>(iConfig)
{
   Charge_Vs_Index     = NULL;//make sure this is initialized to NULL 
   OutputGains         = iConfig.getParameter<std::string>("OutputGains");
   theTracksLabel      = iConfig.getUntrackedParameter<edm::InputTag>("Tracks");

   AlgoMode            = iConfig.getUntrackedParameter<std::string>("AlgoMode", "CalibTree");
   MinNrEntries        = iConfig.getUntrackedParameter<double>  ("minNrEntries"       ,  20);
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
   Validation          = iConfig.getUntrackedParameter<bool>    ("Validation"         ,  false);
   OldGainRemoving     = iConfig.getUntrackedParameter<bool>    ("OldGainRemoving"    ,  false);

   CalibrationLevel    = iConfig.getUntrackedParameter<int>     ("CalibrationLevel"   ,  0);
   VInputFiles         = iConfig.getUntrackedParameter<vector<string> >  ("InputFiles");


   useCalibration      = iConfig.getUntrackedParameter<bool>    ("UseCalibration"     , false);
   m_calibrationPath   = iConfig.getUntrackedParameter<string>  ("calibrationPath");
   harvestingMode      = iConfig.getUntrackedParameter<bool>    ("harvestingMode"     , false);

   tagCondition_NClusters  = iConfig.getUntrackedParameter<double>    ("NClustersForTagProd"     , 2E8);
   tagCondition_GoodFrac   = iConfig.getUntrackedParameter<double>    ("GoodFracForTagProd"     , 0.95);

   saveSummary          = iConfig.getUntrackedParameter<bool>    ("saveSummary"     , false);

   dbe = edm::Service<DQMStore>().operator->();
   dbe->setVerbose(10);

   tracksToken = consumes<edm::View<reco::Track>>(theTracksLabel);
   associationsToken = consumes<TrajTrackAssociationCollection>(theTracksLabel);
}


void SiStripGainFromCalibTree::algoBeginJob(const edm::EventSetup& iSetup)
{
   dbe->setCurrentFolder("AlCaReco/SiStripGains/");
   if(AlgoMode != "PCL" or harvestingMode)dbe->setCurrentFolder("AlCaReco/SiStripGainsHarvesting/");

   Charge_Vs_Index           = dbe->book2D("Charge_Vs_Index"          , "Charge_Vs_Index"          , 88625, 0   , 88624,2000,0,4000);
   Charge_Vs_Index_Absolute  = dbe->book2D("Charge_Vs_Index_Absolute" , "Charge_Vs_Index_Absolute" , 88625, 0   , 88624,1000,0,4000);
//   Charge_Vs_Index           = dbe->book2D("Charge_Vs_Index"          , "Charge_Vs_Index"          , 72785, 0   , 72784,1000,0,2000);
//   Charge_Vs_Index_Absolute  = dbe->book2D("Charge_Vs_Index_Absolute" , "Charge_Vs_Index_Absolute" , 72785, 0   , 72784, 500,0,2000);
   Charge_Vs_PathlengthTIB   = dbe->book2D("Charge_Vs_PathlengthTIB"  , "Charge_Vs_PathlengthTIB"  , 20   , 0.3 , 1.3  , 250,0,2000);
   Charge_Vs_PathlengthTOB   = dbe->book2D("Charge_Vs_PathlengthTOB"  , "Charge_Vs_PathlengthTOB"  , 20   , 0.3 , 1.3  , 250,0,2000);
   Charge_Vs_PathlengthTIDP  = dbe->book2D("Charge_Vs_PathlengthTIDP" , "Charge_Vs_PathlengthTIDP" , 20   , 0.3 , 1.3  , 250,0,2000);
   Charge_Vs_PathlengthTIDM  = dbe->book2D("Charge_Vs_PathlengthTIDM" , "Charge_Vs_PathlengthTIDM" , 20   , 0.3 , 1.3  , 250,0,2000);
   Charge_Vs_PathlengthTECP1 = dbe->book2D("Charge_Vs_PathlengthTECP1", "Charge_Vs_PathlengthTECP1", 20   , 0.3 , 1.3  , 250,0,2000);
   Charge_Vs_PathlengthTECP2 = dbe->book2D("Charge_Vs_PathlengthTECP2", "Charge_Vs_PathlengthTECP2", 20   , 0.3 , 1.3  , 250,0,2000);
   Charge_Vs_PathlengthTECM1 = dbe->book2D("Charge_Vs_PathlengthTECM1", "Charge_Vs_PathlengthTECM1", 20   , 0.3 , 1.3  , 250,0,2000);
   Charge_Vs_PathlengthTECM2 = dbe->book2D("Charge_Vs_PathlengthTECM2", "Charge_Vs_PathlengthTECM2", 20   , 0.3 , 1.3  , 250,0,2000);

   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
   auto const & Det = tkGeom->dets();

   NPixelDets = 0;
   NStripAPVs = 0;
   unsigned int Index=0;
   for(unsigned int i=0;i<Det.size();i++){
      DetId  Detid  = Det[i]->geographicalId(); 
      int    SubDet = Detid.subdetId();

      if( SubDet == StripSubdetector::TIB ||  SubDet == StripSubdetector::TID ||
          SubDet == StripSubdetector::TOB ||  SubDet == StripSubdetector::TEC  ){

          auto DetUnit     = dynamic_cast<const StripGeomDetUnit*> (Det[i]);
	  if(!DetUnit)continue;

          const StripTopology& Topo     = DetUnit->specificTopology();	
          unsigned int         NAPV     = Topo.nstrips()/128;

          for(unsigned int j=0;j<NAPV;j++){
                stAPVGain* APV = new stAPVGain;
                APV->Index         = Index;
                APV->Bin           = -1;
                APV->DetId         = Detid.rawId();
                APV->APVId         = j;
                APV->SubDet        = SubDet;
                APV->FitMPV        = -1;
                APV->FitMPVErr     = -1;
                APV->FitWidth      = -1;
                APV->FitWidthErr   = -1;
                APV->FitChi2       = -1;
                APV->FitNorm       = -1;
                APV->Gain          = -1;
                APV->PreviousGain  = 1;
                APV->PreviousGainTick  = 1;
                APV->x             = DetUnit->position().basicVector().x();
                APV->y             = DetUnit->position().basicVector().y();
                APV->z             = DetUnit->position().basicVector().z();
                APV->Eta           = DetUnit->position().basicVector().eta();
                APV->Phi           = DetUnit->position().basicVector().phi();
                APV->R             = DetUnit->position().basicVector().transverse();
                APV->Thickness     = DetUnit->surface().bounds().thickness();
		APV->NEntries	   = 0;
                APV->isMasked      = false;

                APVsCollOrdered.push_back(APV);
		APVsColl[(APV->DetId<<4) | APV->APVId] = APV;
                Index++;
                NStripAPVs++;
          }
      }
   }

   for(unsigned int i=0;i<Det.size();i++){  //Make two loop such that the Pixel information is added at the end --> make transition simpler
      DetId  Detid  = Det[i]->geographicalId();
      int    SubDet = Detid.subdetId();
      if( SubDet == PixelSubdetector::PixelBarrel || PixelSubdetector::PixelEndcap ){
           auto DetUnit     = dynamic_cast<const PixelGeomDetUnit*> (Det[i]);
           if(!DetUnit) continue;
       
           const PixelTopology& Topo     = DetUnit->specificTopology();
           unsigned int         NROCRow  = Topo.nrows()/(80.);
           unsigned int         NROCCol  = Topo.ncolumns()/(52.);

           for(unsigned int j=0;j<NROCRow;j++){
           for(unsigned int i=0;i<NROCCol;i++){
         
              stAPVGain* APV = new stAPVGain;
              APV->Index         = Index;
              APV->Bin           = -1;
              APV->DetId         = Detid.rawId();
              APV->APVId         = (j<<3 | i);
              APV->SubDet        = SubDet;
              APV->FitMPV        = -1;
              APV->FitMPVErr     = -1;
              APV->FitWidth      = -1;
              APV->FitWidthErr   = -1;
              APV->FitChi2       = -1;
              APV->Gain          = -1;
              APV->PreviousGain  = 1;
              APV->x             = DetUnit->position().basicVector().x();
              APV->y             = DetUnit->position().basicVector().y();
              APV->z             = DetUnit->position().basicVector().z();
              APV->Eta           = DetUnit->position().basicVector().eta();
              APV->Phi           = DetUnit->position().basicVector().phi();
              APV->R             = DetUnit->position().basicVector().transverse();
              APV->Thickness     = DetUnit->surface().bounds().thickness();
              APV->isMasked      = false; //SiPixelQuality_->IsModuleBad(Detid.rawId());
              APV->NEntries      = 0;

              APVsCollOrdered.push_back(APV);
              APVsColl[(APV->DetId<<4) | APV->APVId] = APV;
              Index++;
              NPixelDets++;
           }}
      }
   }


   MakeCalibrationMap();

   NEvent     = 0;
   NTrack     = 0;
   NClusterStrip   = 0;
   NClusterPixel   = 0;
   SRun       = 1<<31;
   ERun       = 0;
   GOOD       = 0;
   BAD        = 0;
   MASKED     = 0;
}



void SiStripGainFromCalibTree::algoBeginRun(const edm::Run& run, const edm::EventSetup& iSetup)
{
   edm::ESHandle<SiStripGain> gainHandle;
   iSetup.get<SiStripGainRcd>().get(gainHandle);
   if(!gainHandle.isValid()){edm::LogError("SiStripGainFromCalibTree")<< "gainHandle is not valid\n"; exit(0);}
 
   edm::ESHandle<SiStripQuality> SiStripQuality_;
   iSetup.get<SiStripQualityRcd>().get(SiStripQuality_);

   for(unsigned int a=0;a<APVsCollOrdered.size();a++){
      stAPVGain* APV = APVsCollOrdered[a];

      APV->isMasked      = SiStripQuality_->IsApvBad(APV->DetId,APV->APVId);
//      if(!FirstSetOfConstants){
         if(gainHandle->getNumberOfTags()!=2){edm::LogError("SiStripGainFromCalibTree")<< "NUMBER OF GAIN TAG IS EXPECTED TO BE 2\n";fflush(stdout);exit(0);};		   
         float newPreviousGain = gainHandle->getApvGain(APV->APVId,gainHandle->getRange(APV->DetId, 1),1);
         if(APV->PreviousGain!=1 and newPreviousGain!=APV->PreviousGain)edm::LogWarning("SiStripGainFromCalibTree")<< "WARNING: ParticleGain in the global tag changed\n";
         APV->PreviousGain = newPreviousGain;

         float newPreviousGainTick = gainHandle->getApvGain(APV->APVId,gainHandle->getRange(APV->DetId, 0),1);
         if(APV->PreviousGainTick!=1 and newPreviousGainTick!=APV->PreviousGainTick)edm::LogWarning("SiStripGainFromCalibTree")<< "WARNING: TickMarkGain in the global tag changed\n";
         APV->PreviousGainTick = newPreviousGainTick;


         //printf("DETID = %7i APVID=%1i Previous Gain=%8.4f (G1) x %8.4f (G2)\n",APV->DetId,APV->APVId,APV->PreviousGainTick, APV->PreviousGain);
         
         

//      }
   }
}

void SiStripGainFromCalibTree::algoEndRun(const edm::Run& run, const edm::EventSetup& iSetup){
   if(AlgoMode == "PCL" && !harvestingMode)return;//nothing to do in that case

   if(AlgoMode == "PCL" and harvestingMode){
     // Load the 2D histograms from the DQM objects
     // When running in AlCaHarvesting mode the histos are already booked and should be just retrieved from
     // DQMStore so that they can be used in the fit
     
//     edm::LogInfo("SiStripGainFromCalibTree")<< "Harvesting " << (dbe->get("AlCaReco/SiStripGains/Charge_Vs_Index"))->getTH2F()->GetEntries() << " more clusters\n";

     merge(Charge_Vs_Index           ->getTH2F(), (dbe->get("AlCaReco/SiStripGains/Charge_Vs_Index"))->getTH2F() );
//     Charge_Vs_Index           ->getTH2F()->Add((dbe->get("AlCaReco/SiStripGains/Charge_Vs_Index"))->getTH2F());
     merge(Charge_Vs_Index_Absolute  ->getTH2F(), (dbe->get("AlCaReco/SiStripGains/Charge_Vs_Index_Absolute"))->getTH2F() );     
//     Charge_Vs_Index_Absolute  ->getTH2F()->Add((dbe->get("AlCaReco/SiStripGains/Charge_Vs_Index_Absolute"))->getTH2F());
     Charge_Vs_PathlengthTIB   ->getTH2F()->Add((dbe->get("AlCaReco/SiStripGains/Charge_Vs_PathlengthTIB"))->getTH2F());
     Charge_Vs_PathlengthTOB   ->getTH2F()->Add((dbe->get("AlCaReco/SiStripGains/Charge_Vs_PathlengthTOB"))->getTH2F());
     Charge_Vs_PathlengthTIDP  ->getTH2F()->Add((dbe->get("AlCaReco/SiStripGains/Charge_Vs_PathlengthTIDP"))->getTH2F());
     Charge_Vs_PathlengthTIDM  ->getTH2F()->Add((dbe->get("AlCaReco/SiStripGains/Charge_Vs_PathlengthTIDM"))->getTH2F());
     Charge_Vs_PathlengthTECP1 ->getTH2F()->Add((dbe->get("AlCaReco/SiStripGains/Charge_Vs_PathlengthTECP1"))->getTH2F());
     Charge_Vs_PathlengthTECP2 ->getTH2F()->Add((dbe->get("AlCaReco/SiStripGains/Charge_Vs_PathlengthTECP2"))->getTH2F());
     Charge_Vs_PathlengthTECM1 ->getTH2F()->Add((dbe->get("AlCaReco/SiStripGains/Charge_Vs_PathlengthTECM1"))->getTH2F());
     Charge_Vs_PathlengthTECM2 ->getTH2F()->Add((dbe->get("AlCaReco/SiStripGains/Charge_Vs_PathlengthTECM2"))->getTH2F());
   }
}
void 
SiStripGainFromCalibTree::algoEndJob() {
   if(AlgoMode == "PCL" && !harvestingMode)return;//nothing to do in that case

   if(AlgoMode == "CalibTree"){
      // Loop on calibTrees to fill the 2D histograms
      algoAnalyzeTheTree();
   }else if(harvestingMode){
     NClusterStrip = Charge_Vs_Index->getTH2F()->Integral(0,NStripAPVs+1, 0, 99999 );
     NClusterPixel = Charge_Vs_Index->getTH2F()->Integral(NStripAPVs+2, NStripAPVs+NPixelDets+2, 0, 99999 );
   }

   // Now that we have the full statistics we can extract the information of the 2D histograms
   algoComputeMPVandGain();
   
   if(AlgoMode != "PCL" or saveSummary){
      //also save the 2D monitor elements to this file as TH2D tfs
      tfs = edm::Service<TFileService>().operator->();
      tfs->make<TH2F> (*Charge_Vs_Index->getTH2F());
      tfs->make<TH2F> (*Charge_Vs_Index_Absolute->getTH2F());
      tfs->make<TH2F> (*Charge_Vs_PathlengthTIB->getTH2F());
      tfs->make<TH2F> (*Charge_Vs_PathlengthTOB->getTH2F());
      tfs->make<TH2F> (*Charge_Vs_PathlengthTIDP->getTH2F());
      tfs->make<TH2F> (*Charge_Vs_PathlengthTIDM->getTH2F());
      tfs->make<TH2F> (*Charge_Vs_PathlengthTECP1->getTH2F());
      tfs->make<TH2F> (*Charge_Vs_PathlengthTECP2->getTH2F());
      tfs->make<TH2F> (*Charge_Vs_PathlengthTECM1->getTH2F());
      tfs->make<TH2F> (*Charge_Vs_PathlengthTECM2->getTH2F());


      storeOnTree(tfs);
   }
}


void SiStripGainFromCalibTree::getPeakOfLandau(TH1* InputHisto, double* FitResults, double LowRange, double HighRange)
{ 
   FitResults[0]         = -0.5;  //MPV
   FitResults[1]         =  0;    //MPV error
   FitResults[2]         = -0.5;  //Width
   FitResults[3]         =  0;    //Width error
   FitResults[4]         = -0.5;  //Fit Chi2/NDF
   FitResults[5]         = 0;     //Normalization

   if( InputHisto->GetEntries() < MinNrEntries)return;

   // perform fit with standard landau
   TF1* MyLandau = new TF1("MyLandau","landau",LowRange, HighRange);
   MyLandau->SetParameter(1,300);
   InputHisto->Fit(MyLandau,"0QR WW");

   // MPV is parameter 1 (0=constant, 1=MPV, 2=Sigma)
   FitResults[0]         = MyLandau->GetParameter(1);  //MPV
   FitResults[1]         = MyLandau->GetParError(1);   //MPV error
   FitResults[2]         = MyLandau->GetParameter(2);  //Width
   FitResults[3]         = MyLandau->GetParError(2);   //Width error
   FitResults[4]         = MyLandau->GetChisquare() / MyLandau->GetNDF();  //Fit Chi2/NDF
   FitResults[5]         = MyLandau->GetParameter(0);

   delete MyLandau;
}

bool SiStripGainFromCalibTree::IsGoodLandauFit(double* FitResults){
   if(FitResults[0] <= 0             )return false;
//   if(FitResults[1] > MaxMPVError   )return false;
//   if(FitResults[4] > MaxChi2OverNDF)return false;
   return true;   
}


void SiStripGainFromCalibTree::algoAnalyzeTheTree()
{
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
      std::vector<bool>*           TrigTech       = 0;    tree->SetBranchAddress(EventPrefix + "TrigTech"       + EventSuffix, &TrigTech      , NULL);

      std::vector<double>*         trackchi2ndof  = 0;    tree->SetBranchAddress(TrackPrefix + "chi2ndof"       + TrackSuffix, &trackchi2ndof , NULL);
      std::vector<float>*          trackp         = 0;    tree->SetBranchAddress(TrackPrefix + "momentum"       + TrackSuffix, &trackp        , NULL);
      std::vector<float>*          trackpt        = 0;    tree->SetBranchAddress(TrackPrefix + "pt"             + TrackSuffix, &trackpt       , NULL);
      std::vector<double>*         tracketa       = 0;    tree->SetBranchAddress(TrackPrefix + "eta"            + TrackSuffix, &tracketa      , NULL);
      std::vector<double>*         trackphi       = 0;    tree->SetBranchAddress(TrackPrefix + "phi"            + TrackSuffix, &trackphi      , NULL);
      std::vector<unsigned int>*   trackhitsvalid = 0;    tree->SetBranchAddress(TrackPrefix + "hitsvalid"      + TrackSuffix, &trackhitsvalid, NULL);

      std::vector<int>*            trackindex     = 0;    tree->SetBranchAddress(CalibPrefix + "trackindex"     + CalibSuffix, &trackindex    , NULL);
      std::vector<unsigned int>*   rawid          = 0;    tree->SetBranchAddress(CalibPrefix + "rawid"          + CalibSuffix, &rawid         , NULL);
      std::vector<float>*          localdirx      = 0;    tree->SetBranchAddress(CalibPrefix + "localdirx"      + CalibSuffix, &localdirx     , NULL);
      std::vector<float>*          localdiry      = 0;    tree->SetBranchAddress(CalibPrefix + "localdiry"      + CalibSuffix, &localdiry     , NULL);
      std::vector<float>*          localdirz      = 0;    tree->SetBranchAddress(CalibPrefix + "localdirz"      + CalibSuffix, &localdirz     , NULL);
      std::vector<unsigned short>* firststrip     = 0;    tree->SetBranchAddress(CalibPrefix + "firststrip"     + CalibSuffix, &firststrip    , NULL);
      std::vector<unsigned short>* nstrips        = 0;    tree->SetBranchAddress(CalibPrefix + "nstrips"        + CalibSuffix, &nstrips       , NULL);
      std::vector<bool>*           saturation     = 0;    tree->SetBranchAddress(CalibPrefix + "saturation"     + CalibSuffix, &saturation    , NULL);
      std::vector<bool>*           overlapping    = 0;    tree->SetBranchAddress(CalibPrefix + "overlapping"    + CalibSuffix, &overlapping   , NULL);
      std::vector<bool>*           farfromedge    = 0;    tree->SetBranchAddress(CalibPrefix + "farfromedge"    + CalibSuffix, &farfromedge   , NULL);
      std::vector<unsigned int>*   charge         = 0;    tree->SetBranchAddress(CalibPrefix + "charge"         + CalibSuffix, &charge        , NULL);
      std::vector<float>*          path           = 0;    tree->SetBranchAddress(CalibPrefix + "path"           + CalibSuffix, &path          , NULL);
      std::vector<float>*          chargeoverpath = 0;    tree->SetBranchAddress(CalibPrefix + "chargeoverpath" + CalibSuffix, &chargeoverpath, NULL);
      std::vector<unsigned char>*  amplitude      = 0;    tree->SetBranchAddress(CalibPrefix + "amplitude"      + CalibSuffix, &amplitude     , NULL);
      std::vector<double>*         gainused       = 0;    tree->SetBranchAddress(CalibPrefix + "gainused"       + CalibSuffix, &gainused      , NULL);


      printf("Number of Events = %i + %i = %i\n",NEvent,(unsigned int)tree->GetEntries(),(unsigned int)(NEvent+tree->GetEntries()));
      printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
      printf("Looping on the Tree          :");
      int TreeStep = tree->GetEntries()/50;if(TreeStep<=1)TreeStep=1;
      for (unsigned int ientry = 0; ientry < tree->GetEntries(); ientry++) {
      if(ientry%TreeStep==0){printf(".");fflush(stdout);}
         tree->GetEntry(ientry);

         if(runnumber<SRun)SRun=runnumber;
         if(runnumber>ERun)ERun=runnumber;

         NEvent++;
         NTrack+=(*trackp).size();

	 unsigned int FirstAmplitude=0;
         for(unsigned int i=0;i<(*chargeoverpath).size();i++){
            FirstAmplitude+=(*nstrips)[i];
            int    TI = (*trackindex)[i];
            //printf("%i - %i - %i %i %i\n", (int)(*rawid)[i], (int)(*firststrip)[i]/128, (int)(*farfromedge)[i], (int)(*overlapping)[i], (int)(*saturation )[i] );
            if((*tracketa      )[TI]  < MinTrackEta        )continue;
            if((*tracketa      )[TI]  > MaxTrackEta        )continue;
            if((*trackp        )[TI]  < MinTrackMomentum   )continue;
            if((*trackp        )[TI]  > MaxTrackMomentum   )continue;
            if((*trackhitsvalid)[TI]  < MinTrackHits       )continue;
            if((*trackchi2ndof )[TI]  > MaxTrackChiOverNdf )continue;

            stAPVGain* APV = APVsColl[((*rawid)[i]<<4) | ((*firststrip)[i]/128)];   //works for both strip and pixel thanks to firstStrip encoding for pixel in the calibTree

            if(APV->SubDet>2 && (*farfromedge)[i]        == false           )continue;
            if(APV->SubDet>2 && (*overlapping)[i]        == true            )continue;
            if(APV->SubDet>2 && (*saturation )[i]        && !AllowSaturation)continue;
            if(APV->SubDet>2 && (*nstrips    )[i]      > MaxNrStrips        )continue;


            //printf("detId=%7i run=%7i event=%9i charge=%5i cs=%3i\n",(*rawid)[i],runnumber,eventnumber,(*charge)[i],(*nstrips)[i]);

            //double trans = atan2((*localdiry)[i],(*localdirx)[i])*(180/3.14159265);
            //double alpha = acos ((*localdirx)[i] / sqrt( pow((*localdirx)[i],2) +  pow((*localdirz)[i],2) ) ) * (180/3.14159265);
            //double beta  = acos ((*localdiry)[i] / sqrt( pow((*localdirx)[i],2) +  pow((*localdirz)[i],2) ) ) * (180/3.14159265);
	  
	    //printf("NStrip = %i : Charge = %i --> Path = %f  --> ChargeOverPath=%f\n",(*nstrips)[i],(*charge)[i],(*path)[i],(*chargeoverpath)[i]);
            //printf("Amplitudes: ");
            //for(unsigned int a=0;a<(*nstrips)[i];a++){printf("%i ",(*amplitude)[FirstAmplitude+a]);}
            //printf("\n");

             if(APV->SubDet>2){NClusterStrip++;}else{NClusterPixel++;}
           
            int Charge = 0;
            if(APV->SubDet>2 && (useCalibration || !FirstSetOfConstants)){
               bool Saturation = false;
               for(unsigned int s=0;s<(*nstrips)[i];s++){
                  int StripCharge =  (*amplitude)[FirstAmplitude-(*nstrips)[i]+s];
                  if(useCalibration && !FirstSetOfConstants){ StripCharge=(int)(StripCharge*(APV->PreviousGain/APV->CalibGain));
                  }else if(useCalibration){                   StripCharge=(int)(StripCharge/APV->CalibGain);
                  }else if(!FirstSetOfConstants){             StripCharge=(int)(StripCharge*APV->PreviousGain);}
                  if(StripCharge>1024){
                     StripCharge = 255;
                     Saturation = true;
                  }else if(StripCharge>254){
                     StripCharge = 254;
                     Saturation = true;
                  }
                  Charge += StripCharge;
               }
               if(Saturation && !AllowSaturation)continue;
            }else if(APV->SubDet>2){
               Charge = (*charge)[i];
            }else{
               Charge = (*charge)[i]/265.0; //expected scale factor between pixel and strip charge               
            }

            //printf("ChargeDifference = %i Vs %i with Gain = %f\n",(*charge)[i],Charge,APV->CalibGain);

            double ClusterChargeOverPath   =  ( (double) Charge )/(*path)[i] ;
            if(APV->SubDet>2){
               if(Validation)     {ClusterChargeOverPath/=(*gainused)[i];}
               if(OldGainRemoving){ClusterChargeOverPath*=(*gainused)[i];}
            }
            Charge_Vs_Index_Absolute->Fill(APV->Index,Charge);   
            Charge_Vs_Index         ->Fill(APV->Index,ClusterChargeOverPath);

	          if(APV->SubDet==StripSubdetector::TIB){ Charge_Vs_PathlengthTIB  ->Fill((*path)[i],Charge); 
            }else if(APV->SubDet==StripSubdetector::TOB){ Charge_Vs_PathlengthTOB  ->Fill((*path)[i],Charge);
            }else if(APV->SubDet==StripSubdetector::TID){
                     if(APV->Eta<0){			  Charge_Vs_PathlengthTIDM ->Fill((*path)[i],Charge);
               }else if(APV->Eta>0){                      Charge_Vs_PathlengthTIDP ->Fill((*path)[i],Charge);
               }
            }else if(APV->SubDet==StripSubdetector::TEC){
                     if(APV->Eta<0){
                        if(APV->Thickness<0.04){          Charge_Vs_PathlengthTECM1->Fill((*path)[i],Charge);
                  }else if(APV->Thickness>0.04){          Charge_Vs_PathlengthTECM2->Fill((*path)[i],Charge);
                  }
               }else if(APV->Eta>0){
                        if(APV->Thickness<0.04){          Charge_Vs_PathlengthTECP1->Fill((*path)[i],Charge);
                  }else if(APV->Thickness>0.04){          Charge_Vs_PathlengthTECP2->Fill((*path)[i],Charge);
                  }
               }
            }

         }// END OF ON-CLUSTER LOOP
      }printf("\n");// END OF EVENT LOOP

   }
}



void SiStripGainFromCalibTree::algoComputeMPVandGain() {
   unsigned int I=0;
   TH1F* Proj = NULL;
   double FitResults[6];
   double MPVmean = 300;

   TH2F *chvsidx = Charge_Vs_Index->getTH2F();


   printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
   printf("Fitting Charge Distribution  :");
   int TreeStep = APVsColl.size()/50;
   for(__gnu_cxx::hash_map<unsigned int, stAPVGain*,  __gnu_cxx::hash<unsigned int>, isEqual >::iterator it = APVsColl.begin();it!=APVsColl.end();it++,I++){
   if(I%TreeStep==0){printf(".");fflush(stdout);}
      stAPVGain* APV = it->second;
      if(APV->Bin<0) APV->Bin = chvsidx->GetXaxis()->FindBin(APV->Index);

      if(APV->isMasked){APV->Gain=APV->PreviousGain; MASKED++; continue;}

      Proj = (TH1F*)(chvsidx->ProjectionY("",chvsidx->GetXaxis()->FindBin(APV->Index),chvsidx->GetXaxis()->FindBin(APV->Index),"e"));
      if(!Proj)continue;

      if(CalibrationLevel==0){
      }else if(CalibrationLevel==1){
         int SecondAPVId = APV->APVId;
         if(SecondAPVId%2==0){    SecondAPVId = SecondAPVId+1; }else{ SecondAPVId = SecondAPVId-1; }
	 stAPVGain* APV2 = APVsColl[(APV->DetId<<4) | SecondAPVId];
         if(APV2->Bin<0) APV2->Bin = chvsidx->GetXaxis()->FindBin(APV2->Index);
         TH1F* Proj2 = (TH1F*)(chvsidx->ProjectionY("",APV2->Bin,APV2->Bin,"e"));
         if(Proj2){Proj->Add(Proj2,1);delete Proj2;}
      }else if(CalibrationLevel==2){
          for(unsigned int i=0;i<16;i++){  //loop up to 6APV for Strip and up to 16 for Pixels
            __gnu_cxx::hash_map<unsigned int, stAPVGain*,  __gnu_cxx::hash<unsigned int>, isEqual >::iterator tmpit;
            tmpit = APVsColl.find((APV->DetId<<4) | i);
            if(tmpit==APVsColl.end())continue;
            stAPVGain* APV2 = tmpit->second;
	    if(APV2->DetId != APV->DetId || APV2->APVId == APV->APVId)continue;            
            if(APV2->Bin<0) APV2->Bin = chvsidx->GetXaxis()->FindBin(APV2->Index);
            TH1F* Proj2 = (TH1F*)(chvsidx->ProjectionY("",APV2->Bin,APV2->Bin,"e"));
            if(Proj2){Proj->Add(Proj2,1);delete Proj2;}
          }          
      }else{
         CalibrationLevel = 0;
         printf("Unknown Calibration Level, will assume %i\n",CalibrationLevel);
      }

      getPeakOfLandau(Proj,FitResults);
      APV->FitMPV      = FitResults[0];
      APV->FitMPVErr   = FitResults[1];
      APV->FitWidth    = FitResults[2];
      APV->FitWidthErr = FitResults[3];
      APV->FitChi2     = FitResults[4];
      APV->FitNorm     = FitResults[5];
      APV->NEntries    = Proj->GetEntries();

      if(IsGoodLandauFit(FitResults)){
          APV->Gain = APV->FitMPV / MPVmean;
          if(APV->SubDet>2)GOOD++;
      }else{
          APV->Gain = APV->PreviousGain;
          if(APV->SubDet>2)BAD++;
      }
      if(APV->Gain<=0)           APV->Gain  = 1;

      //printf("%5i/%5i:  %6i - %1i  %5E Entries --> MPV = %f +- %f\n",I,APVsColl.size(),APV->DetId, APV->APVId, Proj->GetEntries(), FitResults[0], FitResults[1]);fflush(stdout);
      delete Proj;
   }printf("\n");
}


void SiStripGainFromCalibTree::storeOnTree(TFileService* tfs)
{
   unsigned int  tree_Index;
   unsigned int  tree_Bin;
   unsigned int  tree_DetId;
   unsigned char tree_APVId;
   unsigned char tree_SubDet;
   float         tree_x;
   float         tree_y;
   float         tree_z;
   float         tree_Eta;
   float         tree_R;
   float         tree_Phi;
   float         tree_Thickness;
   float         tree_FitMPV;
   float         tree_FitMPVErr;
   float         tree_FitWidth;
   float         tree_FitWidthErr;
   float         tree_FitChi2NDF;
   float         tree_FitNorm;
   double        tree_Gain;
   double        tree_PrevGain;
   double        tree_PrevGainTick;
   double        tree_NEntries;
   bool          tree_isMasked;

   TTree*         MyTree;
   MyTree = tfs->make<TTree> ("APVGain","APVGain");
   MyTree->Branch("Index"             ,&tree_Index      ,"Index/i");
   MyTree->Branch("Bin"               ,&tree_Bin        ,"Bin/i");
   MyTree->Branch("DetId"             ,&tree_DetId      ,"DetId/i");
   MyTree->Branch("APVId"             ,&tree_APVId      ,"APVId/b");
   MyTree->Branch("SubDet"            ,&tree_SubDet     ,"SubDet/b");
   MyTree->Branch("x"                 ,&tree_x          ,"x/F"); 
   MyTree->Branch("y"                 ,&tree_y          ,"y/F");   
   MyTree->Branch("z"                 ,&tree_z          ,"z/F");   
   MyTree->Branch("Eta"               ,&tree_Eta        ,"Eta/F");
   MyTree->Branch("R"                 ,&tree_R          ,"R/F");
   MyTree->Branch("Phi"               ,&tree_Phi        ,"Phi/F");
   MyTree->Branch("Thickness"         ,&tree_Thickness  ,"Thickness/F");
   MyTree->Branch("FitMPV"            ,&tree_FitMPV     ,"FitMPV/F");
   MyTree->Branch("FitMPVErr"         ,&tree_FitMPVErr  ,"FitMPVErr/F");
   MyTree->Branch("FitWidth"          ,&tree_FitWidth   ,"FitWidth/F");
   MyTree->Branch("FitWidthErr"       ,&tree_FitWidthErr,"FitWidthErr/F");
   MyTree->Branch("FitChi2NDF"        ,&tree_FitChi2NDF ,"FitChi2NDF/F");
   MyTree->Branch("FitNorm"           ,&tree_FitNorm    ,"FitNorm/F");
   MyTree->Branch("Gain"              ,&tree_Gain       ,"Gain/D");
   MyTree->Branch("PrevGain"          ,&tree_PrevGain   ,"PrevGain/D");
   MyTree->Branch("PrevGainTick"      ,&tree_PrevGainTick,"PrevGainTick/D");
   MyTree->Branch("NEntries"          ,&tree_NEntries   ,"NEntries/D");
   MyTree->Branch("isMasked"          ,&tree_isMasked   ,"isMasked/O");


   FILE* Gains = stdout;
   fprintf(Gains,"NEvents   = %i\n",NEvent);
   fprintf(Gains,"NTracks   = %i\n",NTrack);
   fprintf(Gains,"NClustersPixel = %i\n",NClusterPixel);
   fprintf(Gains,"NClustersStrip = %i\n",NClusterStrip);
   fprintf(Gains,"Number of Pixel Dets = %lu\n",static_cast<unsigned long>(NPixelDets));
   fprintf(Gains,"Number of Strip APVs = %lu\n",static_cast<unsigned long>(NStripAPVs));
   fprintf(Gains,"GoodFits = %i BadFits = %i ratio = %f%%   (MASKED=%i)\n",GOOD,BAD,(100.0*GOOD)/(GOOD+BAD), MASKED);

   Gains=fopen(OutputGains.c_str(),"w");
   fprintf(Gains,"NEvents   = %i\n",NEvent);
   fprintf(Gains,"NTracks   = %i\n",NTrack);
   fprintf(Gains,"NClustersPixel = %i\n",NClusterPixel);
   fprintf(Gains,"NClustersStrip = %i\n",NClusterStrip);
   fprintf(Gains,"Number of Strip APVs = %lu\n",static_cast<unsigned long>(NStripAPVs));
   fprintf(Gains,"Number of Pixel Dets = %lu\n",static_cast<unsigned long>(NPixelDets));
   fprintf(Gains,"GoodFits = %i BadFits = %i ratio = %f%%   (MASKED=%i)\n",GOOD,BAD,(100.0*GOOD)/(GOOD+BAD), MASKED);

   for(unsigned int a=0;a<APVsCollOrdered.size();a++){
      stAPVGain* APV = APVsCollOrdered[a];
      if(APV==NULL)continue;
//     printf(      "%i | %i | PreviousGain = %7.5f NewGain = %7.5f (#clusters=%8.0f)\n", APV->DetId,APV->APVId,APV->PreviousGain,APV->Gain, APV->NEntries);
      fprintf(Gains,"%i | %i | PreviousGain = %7.5f(tick) x %7.5f(particle) NewGain (particle) = %7.5f (#clusters=%8.0f)\n", APV->DetId,APV->APVId,APV->PreviousGainTick, APV->PreviousGain,APV->Gain, APV->NEntries);

      tree_Index      = APV->Index;
      tree_Bin        = Charge_Vs_Index->getTH2F()->GetXaxis()->FindBin(APV->Index);
      tree_DetId      = APV->DetId;
      tree_APVId      = APV->APVId;
      tree_SubDet     = APV->SubDet;
      tree_x          = APV->x;
      tree_y          = APV->y;
      tree_z          = APV->z;
      tree_Eta        = APV->Eta;
      tree_R          = APV->R;
      tree_Phi        = APV->Phi;
      tree_Thickness  = APV->Thickness;
      tree_FitMPV     = APV->FitMPV;
      tree_FitMPVErr  = APV->FitMPVErr;
      tree_FitWidth   = APV->FitWidth;
      tree_FitWidthErr= APV->FitWidthErr;
      tree_FitChi2NDF = APV->FitChi2;
      tree_FitNorm    = APV->FitNorm;
      tree_Gain       = APV->Gain;
      tree_PrevGain   = APV->PreviousGain;
      tree_PrevGainTick  = APV->PreviousGainTick;
      tree_NEntries   = APV->NEntries;
      tree_isMasked   = APV->isMasked;


     if(tree_DetId==402673324){
         printf("%i | %i : %f --> %f  (%f)\n", tree_DetId, tree_APVId, tree_PrevGain, tree_Gain, tree_NEntries);
      }


      MyTree->Fill();
   }
   if(Gains)fclose(Gains);


}

bool SiStripGainFromCalibTree::produceTagFilter(){
  
   // The goal of this function is to check wether or not there is enough statistics to produce a meaningful tag for the DB or not 
  if(Charge_Vs_Index->getTH2F()->Integral(0,NStripAPVs+1, 0, 99999 ) < tagCondition_NClusters) {
    edm::LogWarning("SiStripGainFromCalibTree")<< "produceTagFilter -> Return false: Statistics is too low : " << Charge_Vs_Index->getTH2F()->Integral() << endl;
    return false;
  }
  if((1.0 * GOOD) / (GOOD+BAD) < tagCondition_GoodFrac) {
    edm::LogWarning("SiStripGainFromCalibTree")<< "produceTagFilter ->  Return false: ratio of GOOD/TOTAL is too low: " << (1.0 * GOOD) / (GOOD+BAD) << endl;
    return false;
  }
  return true; 
}

SiStripApvGain* SiStripGainFromCalibTree::getNewObject() 
{
   SiStripApvGain* obj = new SiStripApvGain();
   if(!harvestingMode) return obj;

   if(!produceTagFilter()){
       edm::LogWarning("SiStripGainFromCalibTree")<< "getNewObject -> will not produce a paylaod because produceTagFilter returned false " << endl;       
       setDoStore(false); 
       return obj;
   }


   std::vector<float>* theSiStripVector = NULL;
   unsigned int PreviousDetId = 0; 
   for(unsigned int a=0;a<APVsCollOrdered.size();a++){
      stAPVGain* APV = APVsCollOrdered[a];
      if(APV==NULL){ printf("Bug\n"); continue; }
      if(APV->SubDet<=2)continue;
      if(APV->DetId != PreviousDetId){
         if(theSiStripVector!=NULL){
	    SiStripApvGain::Range range(theSiStripVector->begin(),theSiStripVector->end());
	    if ( !obj->put(PreviousDetId,range) )  printf("Bug to put detId = %i\n",PreviousDetId);
	 }
 	 theSiStripVector = new std::vector<float>;
         PreviousDetId = APV->DetId;
      }
      theSiStripVector->push_back(APV->Gain);
   }
   if(theSiStripVector!=NULL){
      SiStripApvGain::Range range(theSiStripVector->begin(),theSiStripVector->end());
      if ( !obj->put(PreviousDetId,range) )  printf("Bug to put detId = %i\n",PreviousDetId);
   }

   return obj;
}


SiStripGainFromCalibTree::~SiStripGainFromCalibTree()
{ 
}

void SiStripGainFromCalibTree::MakeCalibrationMap(){
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
       stAPVGain* APV = APVsColl[(tree_DetId<<4) | (unsigned int)tree_APVId];
       APV->CalibGain = tree_Gain;
   }
}

void
SiStripGainFromCalibTree::algoAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // in AlCaHarvesting mode we just need to run the logic in the endJob step
  if(harvestingMode) return;
  
   if(AlgoMode=="CalibTree")return;

   if(NEvent==0){
      SRun          = iEvent.id().run();
   }
   ERun             = iEvent.id().run();
   NEvent++;

  //FROM SHALLOW GAIN
  edm::ESHandle<TrackerGeometry> theTrackerGeometry;         iSetup.get<TrackerDigiGeometryRecord>().get( theTrackerGeometry );  
  edm::ESHandle<SiStripGain> gainHandle;                     iSetup.get<SiStripGainRcd>().get(gainHandle);
  edm::Handle<edm::View<reco::Track> > tracks;               iEvent.getByToken(tracksToken, tracks);
  edm::Handle<TrajTrackAssociationCollection> associations;  iEvent.getByToken(associationsToken, associations);

  for( TrajTrackAssociationCollection::const_iterator association = associations->begin(); association != associations->end(); association++) {
       const Trajectory*  traj  = association->key.get();
       const reco::Track* track = association->val.get();

       //clean on the tracks
       if(fabs(track->eta())          < MinTrackEta        )continue;
       if(fabs(track->eta())          > MaxTrackEta        )continue;
       if(track->p()                  < MinTrackMomentum   )continue;
       if(track->p()                  > MaxTrackMomentum   )continue;
       if(track->numberOfValidHits()  < MinTrackHits       )continue;
       if(track->chi2()/track->ndof() > MaxTrackChiOverNdf )continue;
       NTrack++;

       vector<TrajectoryMeasurement> measurements = traj->measurements();
       for(vector<TrajectoryMeasurement>::const_iterator measurement_it = measurements.begin(); measurement_it!=measurements.end(); measurement_it++){
          TrajectoryStateOnSurface trajState = measurement_it->updatedState();
          if( !trajState.isValid() ) continue;     

          const TrackingRecHit*         hit                 = (*measurement_it->recHit()).hit();
          const SiStripRecHit1D*        sistripsimple1dhit  = dynamic_cast<const SiStripRecHit1D*>(hit);
          const SiStripRecHit2D*        sistripsimplehit    = dynamic_cast<const SiStripRecHit2D*>(hit);
          const SiStripMatchedRecHit2D* sistripmatchedhit   = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);
          const SiPixelRecHit*          sipixelhit          = dynamic_cast<const SiPixelRecHit*>(hit);

          const SiPixelCluster*   PixelCluster = NULL;
          const SiStripCluster*   StripCluster = NULL;
          uint32_t                DetId = 0;

          for(unsigned int h=0;h<2;h++){
            if(!sistripmatchedhit && h==1){
               continue;
            }else if(sistripmatchedhit  && h==0){
               StripCluster = &sistripmatchedhit->monoCluster();
               DetId = sistripmatchedhit->monoId();
            }else if(sistripmatchedhit  && h==1){
               StripCluster = &sistripmatchedhit->stereoCluster();;
               DetId = sistripmatchedhit->stereoId();
            }else if(sistripsimplehit){
               StripCluster = (sistripsimplehit->cluster()).get();
               DetId = sistripsimplehit->geographicalId().rawId();
            }else if(sistripsimple1dhit){
               StripCluster = (sistripsimple1dhit->cluster()).get();
               DetId = sistripsimple1dhit->geographicalId().rawId();
            }else if(sipixelhit && h==0){
               PixelCluster = (sipixelhit->cluster()).get();
               DetId = sipixelhit->geographicalId().rawId();
            }else{
               continue;
            }

            LocalVector             trackDirection = trajState.localDirection();
            double                  cosine         = trackDirection.z()/trackDirection.mag();
            int                     APVId          = 0;
            bool                    Saturation     = false;
            bool                    Overlapping    = false;
            unsigned int            charge         = 0;
            double                  PrevGain       = 1;
            double                  Path           = 0;

            stAPVGain*              APV            = NULL;



            if(StripCluster){
               const auto&          Ampls          = StripCluster->amplitudes();
               int                  FirstStrip     = StripCluster->firstStrip();
               APVId = FirstStrip/128;
               APV   = APVsColl[(DetId<<4) | (FirstStrip/128)];
               Path  = (10.0*APV->Thickness)/fabs(cosine);


               if(gainHandle.isValid()){ 
                  SiStripApvGain::Range detGainRange = gainHandle->getRange(DetId);
                  PrevGain = *(detGainRange.first + APVId);
               }

               for(unsigned int a=0;a<Ampls.size();a++){               
                  charge+=Ampls[a];
                  if(Ampls[a] >=254)Saturation =true;
               }

               if(FirstStrip==0                                  )Overlapping=true;
               if(FirstStrip==128                                )Overlapping=true;
               if(FirstStrip==256                                )Overlapping=true;
               if(FirstStrip==384                                )Overlapping=true;
               if(FirstStrip==512                                )Overlapping=true;
               if(FirstStrip==640                                )Overlapping=true;

               if(FirstStrip<=127 && FirstStrip+Ampls.size()>127)Overlapping=true;
               if(FirstStrip<=255 && FirstStrip+Ampls.size()>255)Overlapping=true;
               if(FirstStrip<=383 && FirstStrip+Ampls.size()>383)Overlapping=true;
               if(FirstStrip<=511 && FirstStrip+Ampls.size()>511)Overlapping=true;
               if(FirstStrip<=639 && FirstStrip+Ampls.size()>639)Overlapping=true;

               if(FirstStrip+Ampls.size()==127                   )Overlapping=true;
               if(FirstStrip+Ampls.size()==255                   )Overlapping=true;
               if(FirstStrip+Ampls.size()==383                   )Overlapping=true;
               if(FirstStrip+Ampls.size()==511                   )Overlapping=true;
               if(FirstStrip+Ampls.size()==639                   )Overlapping=true;
               if(FirstStrip+Ampls.size()==767                   )Overlapping=true;

               //cleaning on the cluster
               if(IsFarFromBorder(&trajState,DetId, &iSetup)  == false           )continue;
               if(Overlapping                                 == true            )continue;
               if(Saturation  && !AllowSaturation                                )continue;
               if(Ampls.size()                              > MaxNrStrips        )continue;
               NClusterStrip++;

               if(useCalibration || !FirstSetOfConstants){
                  bool Saturation = false;
                  charge = 0;
                  for(unsigned int s=0;s<Ampls.size();s++){
                     int StripCharge =  Ampls[s];
                     if(useCalibration && !FirstSetOfConstants){ StripCharge=(int)(StripCharge*(APV->PreviousGain/APV->CalibGain));
                     }else if(useCalibration){                   StripCharge=(int)(StripCharge/APV->CalibGain);
                     }else if(!FirstSetOfConstants){             StripCharge=(int)(StripCharge*APV->PreviousGain);}
                     if(StripCharge>1024){
                        StripCharge = 255;
                        Saturation = true;
                     }else if(StripCharge>254){
                        StripCharge = 254;
                        Saturation = true;
                     }
                     charge += StripCharge;
                  }
                  if(Saturation && !AllowSaturation)continue;
               }
            }else if(PixelCluster){
               const auto&             Ampls     = PixelCluster->pixelADC();
               int                     FirstRow       = PixelCluster->minPixelRow();
               int                     FirstCol       = PixelCluster->minPixelCol();
               APVId = ((FirstRow/80)<<3 | (FirstCol/52));
               APV   = APVsColl[(DetId<<4) | APVId];
               Path  = (10.0*APV->Thickness)/fabs(cosine);

               for(unsigned int a=0;a<Ampls.size();a++){
                  charge+=Ampls[a];
                  if(Ampls[a] >=254)Saturation =true;
               }
               charge/=265.0; //expected scale factor between pixel and strip charge
            }

            //printf("ChargeDifference = %i Vs %i with Gain = %f\n",(*charge)[i],charge,APV->CalibGain);

            double ClusterChargeOverPath   =  ( (double) charge )/Path ;
            if(Validation)     {ClusterChargeOverPath/=PrevGain;}
            if(OldGainRemoving){ClusterChargeOverPath*=PrevGain;}
            Charge_Vs_Index_Absolute->Fill(APV->Index,charge);   
            Charge_Vs_Index         ->Fill(APV->Index,ClusterChargeOverPath);


                  if(APV->SubDet==StripSubdetector::TIB){ Charge_Vs_PathlengthTIB  ->Fill(Path,charge);
            }else if(APV->SubDet==StripSubdetector::TOB){ Charge_Vs_PathlengthTOB  ->Fill(Path,charge);
            }else if(APV->SubDet==StripSubdetector::TID){ 
                     if(APV->Eta<0){                      Charge_Vs_PathlengthTIDM ->Fill(Path,charge);
               }else if(APV->Eta>0){                      Charge_Vs_PathlengthTIDP ->Fill(Path,charge);
               }
            }else if(APV->SubDet==StripSubdetector::TEC){
                     if(APV->Eta<0){
                        if(APV->Thickness<0.04){          Charge_Vs_PathlengthTECM1->Fill(Path,charge);
                  }else if(APV->Thickness>0.04){          Charge_Vs_PathlengthTECM2->Fill(Path,charge);
                  }
               }else if(APV->Eta>0){
                        if(APV->Thickness<0.04){          Charge_Vs_PathlengthTECP1->Fill(Path,charge);
                  }else if(APV->Thickness>0.04){          Charge_Vs_PathlengthTECP2->Fill(Path,charge);
                  }
               }
            }

          }//loop on  clusters
       }//loop on measurements
  }//loop on tracks

}


bool SiStripGainFromCalibTree::IsFarFromBorder(TrajectoryStateOnSurface* trajState, const uint32_t detid, const edm::EventSetup* iSetup)
{ 
  edm::ESHandle<TrackerGeometry> tkGeom; iSetup->get<TrackerDigiGeometryRecord>().get( tkGeom );

  LocalPoint  HitLocalPos   = trajState->localPosition();
  LocalError  HitLocalError = trajState->localError().positionError() ;

  const GeomDetUnit* it = tkGeom->idToDetUnit(DetId(detid));
  if (dynamic_cast<const StripGeomDetUnit*>(it)==0 && dynamic_cast<const PixelGeomDetUnit*>(it)==0) {
     edm::LogWarning("SiStripGainFromCalibTree")<< "this detID doesn't seem to belong to the Tracker" << std::endl;
     return false;
  }

  const BoundPlane plane = it->surface();
  const TrapezoidalPlaneBounds* trapezoidalBounds( dynamic_cast<const TrapezoidalPlaneBounds*>(&(plane.bounds())));
  const RectangularPlaneBounds* rectangularBounds( dynamic_cast<const RectangularPlaneBounds*>(&(plane.bounds())));

  double DistFromBorder = 1.0;    
  double HalfLength     = it->surface().bounds().length() /2.0;

  if(trapezoidalBounds)
  {
     std::array<const float, 4> const & parameters = (*trapezoidalBounds).parameters();
     //std::vector<float> const & parameters = (*trapezoidalBounds).parameters();
     HalfLength     = parameters[3];
  }else if(rectangularBounds){
     HalfLength     = it->surface().bounds().length() /2.0;
  }else{return false;}

  if (fabs(HitLocalPos.y())+HitLocalError.yy() >= (HalfLength - DistFromBorder) ) return false;

  return true;
}



DEFINE_FWK_MODULE(SiStripGainFromCalibTree);
