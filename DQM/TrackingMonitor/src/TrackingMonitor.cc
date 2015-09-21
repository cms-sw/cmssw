/*
 *  See header file for a description of this class.
 *
 *  \author Suchandra Dutta , Giorgia Mila
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/TrackingMonitor/interface/TrackBuildingAnalyzer.h"
#include "DQM/TrackingMonitor/interface/TrackAnalyzer.h"
#include "DQM/TrackingMonitor/interface/VertexMonitor.h"
#include "DQM/TrackingMonitor/interface/TrackingMonitor.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include <string>

// TrackingMonitor 
// ----------------------------------------------------------------------------------//

TrackingMonitor::TrackingMonitor(const edm::ParameterSet& iConfig) 
    : conf_ ( iConfig )
    , theTrackBuildingAnalyzer( new TrackBuildingAnalyzer(conf_) )
    , NumberOfTracks(NULL)
    , NumberOfMeanRecHitsPerTrack(NULL)
    , NumberOfMeanLayersPerTrack(NULL)
				//    , NumberOfGoodTracks(NULL)
    , FractionOfGoodTracks(NULL)
    , NumberOfSeeds(NULL)
    , NumberOfSeeds_lumiFlag(NULL)
    , NumberOfTrackCandidates(NULL)
				//    , NumberOfGoodTrkVsClus(NULL)
    , NumberOfTracksVsLS(NULL)
				//    , NumberOfGoodTracksVsLS(NULL)
    , GoodTracksFractionVsLS(NULL)
				//    , GoodTracksNumberOfRecHitsPerTrackVsLS(NULL)
				// ADD by Mia for PU monitoring
				// vertex plots to be moved in ad hoc class
    , NumberOfTracksVsGoodPVtx(NULL)
    , NumberOfTracksVsPUPVtx(NULL)
    , NumberOfTracksVsBXlumi(NULL)

    , NumberOfTracks_lumiFlag(NULL)
				//    , NumberOfGoodTracks_lumiFlag(NULL)

    , builderName              ( conf_.getParameter<std::string>("TTRHBuilder"))
    , doTrackerSpecific_       ( conf_.getParameter<bool>("doTrackerSpecific") )
    , doLumiAnalysis           ( conf_.getParameter<bool>("doLumiAnalysis"))
    , doProfilesVsLS_          ( conf_.getParameter<bool>("doProfilesVsLS"))
    , doAllPlots               ( conf_.getParameter<bool>("doAllPlots"))
    , doGeneralPropertiesPlots_( conf_.getParameter<bool>("doGeneralPropertiesPlots"))
    , doHitPropertiesPlots_    ( conf_.getParameter<bool>("doHitPropertiesPlots"))
    , doPUmonitoring_          ( conf_.getParameter<bool>("doPUmonitoring") )
    , genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig,consumesCollector(), *this))
    , numSelection_       (conf_.getParameter<std::string>("numCut"))
    , denSelection_       (conf_.getParameter<std::string>("denCut"))
{

  edm::ConsumesCollector c{ consumesCollector() };
  theTrackAnalyzer = new TrackAnalyzer( conf_,c );

  // input tags for collections from the configuration
  bsSrc_ = conf_.getParameter<edm::InputTag>("beamSpot");
  pvSrc_ = conf_.getParameter<edm::InputTag>("primaryVertex");
  bsSrcToken_ = consumes<reco::BeamSpot>(bsSrc_);
  pvSrcToken_ = mayConsume<reco::VertexCollection>(pvSrc_);

  edm::InputTag alltrackProducer = conf_.getParameter<edm::InputTag>("allTrackProducer");
  edm::InputTag trackProducer    = conf_.getParameter<edm::InputTag>("TrackProducer");
  edm::InputTag tcProducer       = conf_.getParameter<edm::InputTag>("TCProducer");
  edm::InputTag seedProducer     = conf_.getParameter<edm::InputTag>("SeedProducer");
  allTrackToken_       = consumes<reco::TrackCollection>(alltrackProducer);
  trackToken_          = consumes<reco::TrackCollection>(trackProducer);
  trackCandidateToken_ = consumes<TrackCandidateCollection>(tcProducer); 
  seedToken_           = consumes<edm::View<TrajectorySeed> >(seedProducer);

  edm::InputTag stripClusterInputTag_ = conf_.getParameter<edm::InputTag>("stripCluster");
  edm::InputTag pixelClusterInputTag_ = conf_.getParameter<edm::InputTag>("pixelCluster");
  stripClustersToken_ = mayConsume<edmNew::DetSetVector<SiStripCluster> > (stripClusterInputTag_);
  pixelClustersToken_ = mayConsume<edmNew::DetSetVector<SiPixelCluster> > (pixelClusterInputTag_);

  doFractionPlot_ = true;
  if (alltrackProducer.label()==trackProducer.label()) doFractionPlot_ = false;
  
  Quality_  = conf_.getParameter<std::string>("Quality");
  AlgoName_ = conf_.getParameter<std::string>("AlgoName");
  
  // get flag from the configuration
  doPlotsVsBXlumi_   = conf_.getParameter<bool>("doPlotsVsBXlumi");   
  if ( doPlotsVsBXlumi_ )
      theLumiDetails_ = new GetLumi( iConfig.getParameter<edm::ParameterSet>("BXlumiSetup"), c );
  doPlotsVsGoodPVtx_ = conf_.getParameter<bool>("doPlotsVsGoodPVtx");
 

  if ( doPUmonitoring_ ) {
    
    std::vector<edm::InputTag> primaryVertexInputTags    = conf_.getParameter<std::vector<edm::InputTag> >("primaryVertexInputTags");
    std::vector<edm::InputTag> selPrimaryVertexInputTags = conf_.getParameter<std::vector<edm::InputTag> >("selPrimaryVertexInputTags");
    std::vector<std::string>   pvLabels                  = conf_.getParameter<std::vector<std::string> >  ("pvLabels");
     
    if (primaryVertexInputTags.size()==pvLabels.size() and primaryVertexInputTags.size()==selPrimaryVertexInputTags.size()) {
      //      for (auto const& tag : primaryVertexInputTags) {
      for (size_t i=0; i<primaryVertexInputTags.size(); i++) {
 	edm::InputTag iPVinputTag    = primaryVertexInputTags[i];
 	edm::InputTag iSelPVinputTag = selPrimaryVertexInputTags[i];
 	std::string   iPVlabel       = pvLabels[i];
	
 	theVertexMonitor.push_back(new VertexMonitor(conf_,iPVinputTag,iSelPVinputTag,iPVlabel,c) );
      }
    }
  }
  
}


TrackingMonitor::~TrackingMonitor() 
{
  if (theTrackAnalyzer)          delete theTrackAnalyzer;
  if (theTrackBuildingAnalyzer)  delete theTrackBuildingAnalyzer;
  if ( doPUmonitoring_ )
    for (size_t i=0; i<theVertexMonitor.size(); i++)
      if (theVertexMonitor[i]) delete theVertexMonitor[i];
  if (genTriggerEventFlag_)      delete genTriggerEventFlag_;
}


void TrackingMonitor::beginJob(void) 
{

    
}

void TrackingMonitor::bookHistograms(DQMStore::IBooker & ibooker,
				     edm::Run const & iRun,
				     edm::EventSetup const & iSetup) 
{
   // parameters from the configuration
   std::string Quality      = conf_.getParameter<std::string>("Quality");
   std::string AlgoName     = conf_.getParameter<std::string>("AlgoName");
   std::string MEFolderName = conf_.getParameter<std::string>("FolderName"); 

   // test for the Quality veriable validity
   if( Quality_ != "") {
     if( Quality_ != "highPurity" && Quality_ != "tight" && Quality_ != "loose") {
       edm::LogWarning("TrackingMonitor")  << "Qualty Name is invalid, using no quality criterea by default";
       Quality_ = "";
     }
   }

   // use the AlgoName and Quality Name
   std::string CategoryName = Quality_ != "" ? AlgoName_ + "_" + Quality_ : AlgoName_;

   // get binning from the configuration
   int    TKNoBin     = conf_.getParameter<int>(   "TkSizeBin");
   double TKNoMin     = conf_.getParameter<double>("TkSizeMin");
   double TKNoMax     = conf_.getParameter<double>("TkSizeMax");

   int    TCNoBin     = conf_.getParameter<int>(   "TCSizeBin");
   double TCNoMin     = conf_.getParameter<double>("TCSizeMin");
   double TCNoMax     = conf_.getParameter<double>("TCSizeMax");

   int    TKNoSeedBin = conf_.getParameter<int>(   "TkSeedSizeBin");
   double TKNoSeedMin = conf_.getParameter<double>("TkSeedSizeMin");
   double TKNoSeedMax = conf_.getParameter<double>("TkSeedSizeMax");

   int    MeanHitBin  = conf_.getParameter<int>(   "MeanHitBin");
   double MeanHitMin  = conf_.getParameter<double>("MeanHitMin");
   double MeanHitMax  = conf_.getParameter<double>("MeanHitMax");

   int    MeanLayBin  = conf_.getParameter<int>(   "MeanLayBin");
   double MeanLayMin  = conf_.getParameter<double>("MeanLayMin");
   double MeanLayMax  = conf_.getParameter<double>("MeanLayMax");

   int LSBin = conf_.getParameter<int>(   "LSBin");
   int LSMin = conf_.getParameter<double>("LSMin");
   int LSMax = conf_.getParameter<double>("LSMax");

   std::string StateName = conf_.getParameter<std::string>("MeasurementState");
   if (
       StateName != "OuterSurface" &&
       StateName != "InnerSurface" &&
       StateName != "ImpactPoint"  &&
       StateName != "default"      &&
       StateName != "All"
       ) {
     // print warning
     edm::LogWarning("TrackingMonitor")  << "State Name is invalid, using 'ImpactPoint' by default";
   }

   ibooker.setCurrentFolder(MEFolderName);

   // book the General Property histograms
   // ---------------------------------------------------------------------------------//

   if (doGeneralPropertiesPlots_ || doAllPlots){
  
     ibooker.setCurrentFolder(MEFolderName+"/GeneralProperties");

     histname = "NumberOfTracks_" + CategoryName;
       // MODIFY by Mia in order to cope w/ high multiplicity
     NumberOfTracks = ibooker.book1D(histname, histname, 3*TKNoBin, TKNoMin, (TKNoMax+0.5)*3.-0.5);
     NumberOfTracks->setAxisTitle("Number of Tracks per Event", 1);
     NumberOfTracks->setAxisTitle("Number of Events", 2);
  
     histname = "NumberOfMeanRecHitsPerTrack_" + CategoryName;
     NumberOfMeanRecHitsPerTrack = ibooker.book1D(histname, histname, MeanHitBin, MeanHitMin, MeanHitMax);
     NumberOfMeanRecHitsPerTrack->setAxisTitle("Mean number of valid RecHits per Track", 1);
     NumberOfMeanRecHitsPerTrack->setAxisTitle("Entries", 2);
  
     histname = "NumberOfMeanLayersPerTrack_" + CategoryName;
     NumberOfMeanLayersPerTrack = ibooker.book1D(histname, histname, MeanLayBin, MeanLayMin, MeanLayMax);
     NumberOfMeanLayersPerTrack->setAxisTitle("Mean number of Layers per Track", 1);
     NumberOfMeanLayersPerTrack->setAxisTitle("Entries", 2);
  
     if (doFractionPlot_) {
       histname = "FractionOfGoodTracks_" + CategoryName;
       FractionOfGoodTracks = ibooker.book1D(histname, histname, 101, -0.005, 1.005);
       FractionOfGoodTracks->setAxisTitle("Fraction of Tracks (w.r.t. generalTracks)", 1);
       FractionOfGoodTracks->setAxisTitle("Entries", 2);
     }
   }

   if ( doLumiAnalysis ) {
     // add by Mia in order to deal with LS transitions
     ibooker.setCurrentFolder(MEFolderName+"/LSanalysis");
  
     histname = "NumberOfTracks_lumiFlag_" + CategoryName;
     NumberOfTracks_lumiFlag = ibooker.book1D(histname, histname, 3*TKNoBin, TKNoMin, (TKNoMax+0.5)*3.-0.5);
     NumberOfTracks_lumiFlag->setAxisTitle("Number of Tracks per Event", 1);
     NumberOfTracks_lumiFlag->setAxisTitle("Number of Events", 2);
  
   }

   // book profile plots vs LS :  
   //---------------------------  


   if ( doProfilesVsLS_ || doAllPlots) {
  
     ibooker.setCurrentFolder(MEFolderName+"/GeneralProperties");
  
     histname = "NumberOfTracksVsLS_"+ CategoryName;
     NumberOfTracksVsLS = ibooker.bookProfile(histname,histname, LSBin,LSMin,LSMax, TKNoMin, (TKNoMax+0.5)*3.-0.5,"");
     NumberOfTracksVsLS->getTH1()->SetCanExtend(TH1::kAllAxes);
     NumberOfTracksVsLS->setAxisTitle("#Lumi section",1);
     NumberOfTracksVsLS->setAxisTitle("Number of  Tracks",2);
  
     histname = "NumberOfRecHitsPerTrackVsLS_" + CategoryName;
     NumberOfRecHitsPerTrackVsLS = ibooker.bookProfile(histname,histname, LSBin,LSMin,LSMax,0.,40.,"");
     NumberOfRecHitsPerTrackVsLS->getTH1()->SetCanExtend(TH1::kAllAxes);
     NumberOfRecHitsPerTrackVsLS->setAxisTitle("#Lumi section",1);
     NumberOfRecHitsPerTrackVsLS->setAxisTitle("Mean number of Valid RecHits per track",2);
  
     if (doFractionPlot_) {
       histname = "GoodTracksFractionVsLS_"+ CategoryName;
       GoodTracksFractionVsLS = ibooker.bookProfile(histname,histname, LSBin,LSMin,LSMax,0,1.1,"");
       GoodTracksFractionVsLS->getTH1()->SetCanExtend(TH1::kAllAxes);
       GoodTracksFractionVsLS->setAxisTitle("#Lumi section",1);
       GoodTracksFractionVsLS->setAxisTitle("Fraction of Good Tracks",2);
     }
   }

   // book PU monitoring plots :  
   //---------------------------  

   if ( doPUmonitoring_ ) {
  
     for (size_t i=0; i<theVertexMonitor.size(); i++)
       theVertexMonitor[i]->initHisto(ibooker);
   }
  
     if ( doPlotsVsGoodPVtx_ ) {
      ibooker.setCurrentFolder(MEFolderName+"/PUmonitoring");
      // get binning from the configuration
       int    GoodPVtxBin   = conf_.getParameter<int>("GoodPVtxBin");
       double GoodPVtxMin   = conf_.getParameter<double>("GoodPVtxMin");
       double GoodPVtxMax   = conf_.getParameter<double>("GoodPVtxMax");
    
       histname = "NumberOfTracksVsGoodPVtx";
       NumberOfTracksVsGoodPVtx = ibooker.bookProfile(histname,histname,GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,TKNoMin, (TKNoMax+0.5)*3.-0.5,"");
       NumberOfTracksVsGoodPVtx->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfTracksVsGoodPVtx->setAxisTitle("Number of PV",1);
       NumberOfTracksVsGoodPVtx->setAxisTitle("Mean number of Tracks per Event",2);

       histname = "NumberOfTracksVsPUPVtx";
       NumberOfTracksVsPUPVtx = ibooker.bookProfile(histname,histname,GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,0., 100.,"");
       NumberOfTracksVsPUPVtx->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfTracksVsPUPVtx->setAxisTitle("Number of PU",1);
       NumberOfTracksVsPUPVtx->setAxisTitle("Mean number of Tracks per PUvtx",2);

     }
  
     if ( doPlotsVsBXlumi_ ) {
       ibooker.setCurrentFolder(MEFolderName+"/PUmonitoring");
       // get binning from the configuration
       edm::ParameterSet BXlumiParameters = conf_.getParameter<edm::ParameterSet>("BXlumiSetup");
       int    BXlumiBin   = BXlumiParameters.getParameter<int>("BXlumiBin");
       double BXlumiMin   = BXlumiParameters.getParameter<double>("BXlumiMin");
       double BXlumiMax   = BXlumiParameters.getParameter<double>("BXlumiMax");
    
       histname = "NumberOfTracksVsBXlumi_"+ CategoryName;
       NumberOfTracksVsBXlumi = ibooker.bookProfile(histname,histname, BXlumiBin,BXlumiMin,BXlumiMax, TKNoMin, TKNoMax,"");
       NumberOfTracksVsBXlumi->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfTracksVsBXlumi->setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
       NumberOfTracksVsBXlumi->setAxisTitle("Mean number of Tracks",2);
    
     }
   

     theTrackAnalyzer->initHisto(ibooker, iSetup);

   // book the Seed Property histograms
   // ---------------------------------------------------------------------------------//

   ibooker.setCurrentFolder(MEFolderName+"/TrackBuilding");

   doAllSeedPlots      = conf_.getParameter<bool>("doSeedParameterHistos");
   doSeedNumberPlot    = conf_.getParameter<bool>("doSeedNumberHisto");
   doSeedLumiAnalysis_ = conf_.getParameter<bool>("doSeedLumiAnalysis");
   doSeedVsClusterPlot = conf_.getParameter<bool>("doSeedVsClusterHisto");
   //    if (doAllPlots) doAllSeedPlots=true;

   runTrackBuildingAnalyzerForSeed=(doAllSeedPlots || conf_.getParameter<bool>("doSeedPTHisto") ||conf_.getParameter<bool>("doSeedETAHisto") || conf_.getParameter<bool>("doSeedPHIHisto") || conf_.getParameter<bool>("doSeedPHIVsETAHisto") || conf_.getParameter<bool>("doSeedThetaHisto") || conf_.getParameter<bool>("doSeedQHisto") || conf_.getParameter<bool>("doSeedDxyHisto") || conf_.getParameter<bool>("doSeedDzHisto") || conf_.getParameter<bool>("doSeedNRecHitsHisto") || conf_.getParameter<bool>("doSeedNVsPhiProf")|| conf_.getParameter<bool>("doSeedNVsEtaProf"));

   edm::InputTag seedProducer   = conf_.getParameter<edm::InputTag>("SeedProducer");

   if (doAllSeedPlots || doSeedNumberPlot){
     ibooker.setCurrentFolder(MEFolderName+"/TrackBuilding");
     histname = "NumberOfSeeds_"+ seedProducer.label() + "_"+ CategoryName;
     NumberOfSeeds = ibooker.book1D(histname, histname, TKNoSeedBin, TKNoSeedMin, TKNoSeedMax);
     NumberOfSeeds->setAxisTitle("Number of Seeds per Event", 1);
     NumberOfSeeds->setAxisTitle("Number of Events", 2);
     
     if ( doSeedLumiAnalysis_ ) {
       ibooker.setCurrentFolder(MEFolderName+"/LSanalysis");
       histname = "NumberOfSeeds_lumiFlag_"+ seedProducer.label() + "_"+ CategoryName;
       NumberOfSeeds_lumiFlag = ibooker.book1D(histname, histname, TKNoSeedBin, TKNoSeedMin, TKNoSeedMax);
       NumberOfSeeds_lumiFlag->setAxisTitle("Number of Seeds per Event", 1);
       NumberOfSeeds_lumiFlag->setAxisTitle("Number of Events", 2);
     }

   }

   if (doAllSeedPlots || doSeedVsClusterPlot){
     ibooker.setCurrentFolder(MEFolderName+"/TrackBuilding");

     ClusterLabels=  conf_.getParameter<std::vector<std::string> >("ClusterLabels");
  
     std::vector<double> histoMin,histoMax;
     std::vector<int> histoBin; //these vectors are for max min and nbins in histograms 
  
     int    NClusPxBin = conf_.getParameter<int>(   "NClusPxBin");
     double NClusPxMin = conf_.getParameter<double>("NClusPxMin");
     double NClusPxMax = conf_.getParameter<double>("NClusPxMax");
  
     int    NClusStrBin = conf_.getParameter<int>(   "NClusStrBin");
     double NClusStrMin = conf_.getParameter<double>("NClusStrMin");
     double NClusStrMax = conf_.getParameter<double>("NClusStrMax");
  
     setMaxMinBin(histoMin,histoMax,histoBin,NClusStrMin,NClusStrMax,NClusStrBin,NClusPxMin,NClusPxMax,NClusPxBin);
  
     for (uint i=0; i<ClusterLabels.size(); i++){
       histname = "SeedsVsClusters_" + seedProducer.label() + "_Vs_" + ClusterLabels[i] + "_" + CategoryName;
       SeedsVsClusters.push_back(dynamic_cast<MonitorElement*>(ibooker.book2D(histname, histname, histoBin[i], histoMin[i], histoMax[i],
										 TKNoSeedBin, TKNoSeedMin, TKNoSeedMax)));
       SeedsVsClusters[i]->setAxisTitle("Number of Clusters", 1);
       SeedsVsClusters[i]->setAxisTitle("Number of Seeds", 2);
     }
   }
  
   doTkCandPlots=conf_.getParameter<bool>("doTrackCandHistos");
  //    if (doAllPlots) doTkCandPlots=true;
  
  if (doTkCandPlots){
    ibooker.setCurrentFolder(MEFolderName+"/TrackBuilding");
    
    edm::InputTag tcProducer     = conf_.getParameter<edm::InputTag>("TCProducer");
    
    histname = "NumberOfTrackCandidates_"+ tcProducer.label() + "_"+ CategoryName;
    NumberOfTrackCandidates = ibooker.book1D(histname, histname, TCNoBin, TCNoMin, TCNoMax);
    NumberOfTrackCandidates->setAxisTitle("Number of Track Candidates per Event", 1);
    NumberOfTrackCandidates->setAxisTitle("Number of Event", 2);
  }
  
  theTrackBuildingAnalyzer->initHisto(ibooker);
  
  
  if (doLumiAnalysis) {
    if ( NumberOfTracks_lumiFlag ) NumberOfTracks_lumiFlag -> setLumiFlag();
    theTrackAnalyzer->setLumiFlag();    
  }

  if(doAllSeedPlots || doSeedNumberPlot){
    if ( doSeedLumiAnalysis_ )
      NumberOfSeeds_lumiFlag->setLumiFlag();
  }
  
  if (doTrackerSpecific_ || doAllPlots) {
    
    ClusterLabels=  conf_.getParameter<std::vector<std::string> >("ClusterLabels");
    
    std::vector<double> histoMin,histoMax;
    std::vector<int> histoBin; //these vectors are for max min and nbins in histograms 
    
    int    NClusStrBin = conf_.getParameter<int>(   "NClusStrBin");
    double NClusStrMin = conf_.getParameter<double>("NClusStrMin");
    double NClusStrMax = conf_.getParameter<double>("NClusStrMax");
    
    int    NClusPxBin = conf_.getParameter<int>(   "NClusPxBin");
    double NClusPxMin = conf_.getParameter<double>("NClusPxMin");
    double NClusPxMax = conf_.getParameter<double>("NClusPxMax");
    
    int    NTrk2DBin     = conf_.getParameter<int>(   "NTrk2DBin");
    double NTrk2DMin     = conf_.getParameter<double>("NTrk2DMin");
    double NTrk2DMax     = conf_.getParameter<double>("NTrk2DMax");
    
    setMaxMinBin(histoMin,histoMax,histoBin,
		 NClusStrMin,NClusStrMax,NClusStrBin,
		 NClusPxMin,  NClusPxMax,  NClusPxBin);
    
    ibooker.setCurrentFolder(MEFolderName+"/HitProperties");
    
    for (uint i=0; i<ClusterLabels.size(); i++){
      
      ibooker.setCurrentFolder(MEFolderName+"/HitProperties");
      histname = "TracksVs" + ClusterLabels[i] + "Cluster_" + CategoryName;
      NumberOfTrkVsClusters.push_back(dynamic_cast<MonitorElement*>(ibooker.book2D(histname, histname,
										      histoBin[i], histoMin[i], histoMax[i],
										      NTrk2DBin,NTrk2DMin,NTrk2DMax
										      )));
      std::string title = "Number of " + ClusterLabels[i] + " Clusters";
      if(ClusterLabels[i].compare("Tot")==0)
	title = "# of Clusters in (Pixel+Strip) Detectors";
      NumberOfTrkVsClusters[i]->setAxisTitle(title, 1);
      NumberOfTrkVsClusters[i]->setAxisTitle("Number of Tracks", 2);
    }
  }
  
  // Initialize the GenericTriggerEventFlag
  if ( genTriggerEventFlag_->on() ) genTriggerEventFlag_->initRun( iRun, iSetup );
  
}

/*
// -- BeginRun
//---------------------------------------------------------------------------------//
void TrackingMonitor::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  
}
*/

// - BeginLumi
// ---------------------------------------------------------------------------------//
void TrackingMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup&  eSetup) {

  if (doLumiAnalysis) {
    if ( NumberOfTracks_lumiFlag ) NumberOfTracks_lumiFlag -> Reset();
    theTrackAnalyzer->doReset();    
  }
  if(doAllSeedPlots || doSeedNumberPlot) {
    if ( doSeedLumiAnalysis_ )
      NumberOfSeeds_lumiFlag->Reset();
  }
}

// -- Analyse
// ---------------------------------------------------------------------------------//
void TrackingMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
  
    // Filter out events if Trigger Filtering is requested
    if (genTriggerEventFlag_->on()&& ! genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

    //  Analyse the tracks
    //  if the collection is empty, do not fill anything
    // ---------------------------------------------------------------------------------//

    // get the track collection
    edm::Handle<reco::TrackCollection> trackHandle;
    iEvent.getByToken(trackToken_, trackHandle);

    //    int numberOfAllTracks = 0;
    int numberOfTracks_den = 0;
    edm::Handle<reco::TrackCollection> allTrackHandle;
    iEvent.getByToken(allTrackToken_,allTrackHandle);
    if (allTrackHandle.isValid()) {
      //      numberOfAllTracks = allTrackHandle->size();
      for (reco::TrackCollection::const_iterator track = allTrackHandle->begin();
	   track!=allTrackHandle->end(); ++track) {
	if ( denSelection_(*track) )
	  numberOfTracks_den++;
      }
    }
      
    edm::Handle< reco::VertexCollection > pvHandle;
    iEvent.getByToken(pvSrcToken_, pvHandle );
    reco::Vertex const * pv0 = nullptr;  
    if (pvHandle.isValid()) {
      pv0 = &pvHandle->front();
      //--- pv fake (the pv collection should have size==1 and the pv==beam spot)
      if (   pv0->isFake() || pv0->tracksSize()==0
      // definition of goodOfflinePrimaryVertex
          || pv0->ndof() < 4. || pv0->z() > 24.)  pv0 = nullptr;
    }


    if (trackHandle.isValid()) {
      
      int numberOfTracks = trackHandle->size();
      int numberOfTracks_num = 0;
      int numberOfTracks_pv0 = 0;

      reco::TrackCollection trackCollection = *trackHandle;
      // calculate the mean # rechits and layers
      int totalRecHits = 0, totalLayers = 0;

      theTrackAnalyzer->setNumberOfGoodVertices(iEvent);
      for (reco::TrackCollection::const_iterator track = trackCollection.begin();
	   track!=trackCollection.end(); ++track) {
	
	if ( numSelection_(*track) ) {
	  numberOfTracks_num++;
          if (pv0 && std::abs(track->dz(pv0->position()))<0.15) ++numberOfTracks_pv0;
        } 

	if ( doProfilesVsLS_ || doAllPlots)
	  NumberOfRecHitsPerTrackVsLS->Fill(static_cast<double>(iEvent.id().luminosityBlock()),track->numberOfValidHits());

	totalRecHits    += track->numberOfValidHits();
	totalLayers     += track->hitPattern().trackerLayersWithMeasurement();
	
	// do analysis per track
	theTrackAnalyzer->analyze(iEvent, iSetup, *track);
      }

      double frac = -1.;
      //      if (numberOfAllTracks > 0) frac = static_cast<double>(numberOfTracks)/static_cast<double>(numberOfAllTracks);
      if (numberOfTracks_den > 0) frac = static_cast<double>(numberOfTracks_num)/static_cast<double>(numberOfTracks_den);
      
      if (doGeneralPropertiesPlots_ || doAllPlots){
	NumberOfTracks       -> Fill(numberOfTracks);
	if (doFractionPlot_)
	  FractionOfGoodTracks -> Fill(frac);

	if( numberOfTracks > 0 ) {
	  double meanRecHits = static_cast<double>(totalRecHits) / static_cast<double>(numberOfTracks);
	  double meanLayers  = static_cast<double>(totalLayers)  / static_cast<double>(numberOfTracks);
	  NumberOfMeanRecHitsPerTrack -> Fill(meanRecHits);
	  NumberOfMeanLayersPerTrack  -> Fill(meanLayers);
	}
      }
      
      if ( doProfilesVsLS_ || doAllPlots) {
	NumberOfTracksVsLS    ->Fill(static_cast<double>(iEvent.id().luminosityBlock()),numberOfTracks);
	if (doFractionPlot_) 
	  GoodTracksFractionVsLS->Fill(static_cast<double>(iEvent.id().luminosityBlock()),frac);
      }
      
      if ( doLumiAnalysis ) {
	NumberOfTracks_lumiFlag       -> Fill(numberOfTracks);
      }
      
      
      //  Analyse the Track Building variables 
      //  if the collection is empty, do not fill anything
      // ---------------------------------------------------------------------------------//
      
      
      // fill the TrackCandidate info
      if (doTkCandPlots) {
	// magnetic field
	edm::ESHandle<MagneticField> theMF;
	iSetup.get<IdealMagneticFieldRecord>().get(theMF);  
	
	// get the candidate collection
	edm::Handle<TrackCandidateCollection> theTCHandle;
	iEvent.getByToken( trackCandidateToken_, theTCHandle );
	const TrackCandidateCollection& theTCCollection = *theTCHandle;
	
	if (theTCHandle.isValid()) {
	  
	  // get the beam spot
	  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
	  iEvent.getByToken(bsSrcToken_, recoBeamSpotHandle );
	  const reco::BeamSpot& bs = *recoBeamSpotHandle;      
	  
	  NumberOfTrackCandidates->Fill(theTCCollection.size());
	  iSetup.get<TransientRecHitRecord>().get(builderName,theTTRHBuilder);
	  for( TrackCandidateCollection::const_iterator cand = theTCCollection.begin(); cand != theTCCollection.end(); ++cand) {
	    
	    theTrackBuildingAnalyzer->analyze(iEvent, iSetup, *cand, bs, theMF, theTTRHBuilder);
	  }
	} else {
	  edm::LogWarning("TrackingMonitor") << "No Track Candidates in the event.  Not filling associated histograms";
	}
      }
      
      //plots for trajectory seeds
      
      if (doAllSeedPlots || doSeedNumberPlot || doSeedVsClusterPlot || runTrackBuildingAnalyzerForSeed) {
	
	// get the seed collection
	edm::Handle<edm::View<TrajectorySeed> > seedHandle;
	iEvent.getByToken(seedToken_, seedHandle );
	const edm::View<TrajectorySeed>& seedCollection = *seedHandle;
	
	// fill the seed info
	if (seedHandle.isValid()) {
	  
	  if(doAllSeedPlots || doSeedNumberPlot) {
	    NumberOfSeeds->Fill(seedCollection.size());
	    if ( doSeedLumiAnalysis_ )
	      NumberOfSeeds_lumiFlag->Fill(seedCollection.size());	           
	  }
	  
	  if(doAllSeedPlots || doSeedVsClusterPlot){
	    
	    std::vector<int> NClus;
	    setNclus(iEvent,NClus);
	    for (uint  i=0; i< ClusterLabels.size(); i++){
	      SeedsVsClusters[i]->Fill(NClus[i],seedCollection.size());
	    }
	  }
	  
	  if (doAllSeedPlots || runTrackBuildingAnalyzerForSeed){
	    
	    //here duplication of mag field and be informations is needed to allow seed and track cand histos to be independent
	    // magnetic field
	    edm::ESHandle<MagneticField> theMF;
	    iSetup.get<IdealMagneticFieldRecord>().get(theMF);  
	    
	    // get the beam spot
	    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
	    iEvent.getByToken(bsSrcToken_, recoBeamSpotHandle );
	    const reco::BeamSpot& bs = *recoBeamSpotHandle;      
	    
	    iSetup.get<TransientRecHitRecord>().get(builderName,theTTRHBuilder);
	    for(size_t i=0; i < seedHandle->size(); ++i) {
	      
	      edm::RefToBase<TrajectorySeed> seed(seedHandle, i);
	      theTrackBuildingAnalyzer->analyze(iEvent, iSetup, *seed, bs, theMF, theTTRHBuilder);
	    }
	  }
	  
	} else {
	  edm::LogWarning("TrackingMonitor") << "No Trajectory seeds in the event.  Not filling associated histograms";
	}
      }
      
      
      if (doTrackerSpecific_ || doAllPlots) {
	
	std::vector<int> NClus;
	setNclus(iEvent,NClus);
	for (uint  i=0; i< ClusterLabels.size(); i++) {
	  NumberOfTrkVsClusters[i]->Fill(NClus[i],numberOfTracks);
	}
      }
      
      if ( doPUmonitoring_ ) {
	
	// do vertex monitoring
	for (size_t i=0; i<theVertexMonitor.size(); i++)
	  theVertexMonitor[i]->analyze(iEvent, iSetup);
      }
      if ( doPlotsVsGoodPVtx_ ) {
	  
	  size_t totalNumGoodPV = 0;
	  if (pvHandle.isValid()) {
	    
	    for (reco::VertexCollection::const_iterator pv = pvHandle->begin();
		 pv != pvHandle->end(); ++pv) {
	      
	      //--- pv fake (the pv collection should have size==1 and the pv==beam spot) 
	      if (pv->isFake() || pv->tracksSize()==0) continue;
	      
	      // definition of goodOfflinePrimaryVertex
	      if (pv->ndof() < 4. || pv->z() > 24.)  continue;
	      totalNumGoodPV++;
	    }
	    
            NumberOfTracksVsGoodPVtx	   -> Fill( totalNumGoodPV, numberOfTracks	);
	    if (totalNumGoodPV>1) NumberOfTracksVsPUPVtx-> Fill( totalNumGoodPV-1, double(numberOfTracks-numberOfTracks_pv0)/double(totalNumGoodPV-1)      );
	  }

	
	if ( doPlotsVsBXlumi_ ) {
	  double bxlumi = theLumiDetails_->getValue(iEvent);
	  NumberOfTracksVsBXlumi       -> Fill( bxlumi, numberOfTracks      );
	}
	
      } // PU monitoring
      
    } // trackHandle is valid
    
}


void TrackingMonitor::endRun(const edm::Run&, const edm::EventSetup&) 
{
}

void TrackingMonitor::setMaxMinBin(std::vector<double> &arrayMin,  std::vector<double> &arrayMax, std::vector<int> &arrayBin, double smin, double smax, int sbin, double pmin, double pmax, int pbin) 
{
  arrayMin.resize(ClusterLabels.size());
  arrayMax.resize(ClusterLabels.size());
  arrayBin.resize(ClusterLabels.size());

  for (uint i=0; i<ClusterLabels.size(); ++i) {

    if     (ClusterLabels[i].compare("Pix")==0  ) {arrayMin[i]=pmin; arrayMax[i]=pmax;      arrayBin[i]=pbin;}
    else if(ClusterLabels[i].compare("Strip")==0) {arrayMin[i]=smin; arrayMax[i]=smax;      arrayBin[i]=sbin;}
    else if(ClusterLabels[i].compare("Tot")==0  ) {arrayMin[i]=smin; arrayMax[i]=smax+pmax; arrayBin[i]=sbin;}
    else {edm::LogWarning("TrackingMonitor")  << "Cluster Label " << ClusterLabels[i] << " not defined, using strip parameters "; 
      arrayMin[i]=smin; arrayMax[i]=smax; arrayBin[i]=sbin;}

  }
    
}

void TrackingMonitor::setNclus(const edm::Event& iEvent,std::vector<int> &arrayNclus) 
{

  int ncluster_pix=-1;
  int ncluster_strip=-1;

  edm::Handle< edmNew::DetSetVector<SiStripCluster> > strip_clusters;
  iEvent.getByToken(stripClustersToken_, strip_clusters );
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> > pixel_clusters;
  iEvent.getByToken(pixelClustersToken_, pixel_clusters );

  if (strip_clusters.isValid() && pixel_clusters.isValid()) {
    ncluster_pix   = (*pixel_clusters).dataSize(); 
    ncluster_strip = (*strip_clusters).dataSize(); 
  }

  arrayNclus.resize(ClusterLabels.size());
  for (uint i=0; i<ClusterLabels.size(); ++i){
    
    if     (ClusterLabels[i].compare("Pix")==0  ) arrayNclus[i]=ncluster_pix ;
    else if(ClusterLabels[i].compare("Strip")==0) arrayNclus[i]=ncluster_strip;
    else if(ClusterLabels[i].compare("Tot")==0  ) arrayNclus[i]=ncluster_pix+ncluster_strip;
    else {edm::LogWarning("TrackingMonitor") << "Cluster Label " << ClusterLabels[i] << " not defined using stri parametrs ";
      arrayNclus[i]=ncluster_strip ;}
  }
    
}
DEFINE_FWK_MODULE(TrackingMonitor);
