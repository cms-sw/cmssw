/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/10/16 10:07:41 $
 *  $Revision: 1.16 $
 *  \author Suchandra Dutta , Giorgia Mila
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h" 
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h" 
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
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include <string>

// TrackingMonitor 
// ----------------------------------------------------------------------------------//

TrackingMonitor::TrackingMonitor(const edm::ParameterSet& iConfig) 
    : dqmStore_( edm::Service<DQMStore>().operator->() )
    , conf_ ( iConfig )
    , theTrackAnalyzer( new TrackAnalyzer(conf_) )
    , theTrackBuildingAnalyzer( new TrackBuildingAnalyzer(conf_) )
    , NumberOfTracks(NULL)
    , NumberOfMeanRecHitsPerTrack(NULL)
    , NumberOfMeanLayersPerTrack(NULL)
    , NumberOfGoodTracks(NULL)
    , FractionOfGoodTracks(NULL)
    , NumberOfSeeds(NULL)
    , NumberOfTrackCandidates(NULL)
				// ADD by Mia
				/*
    , NumberOfPixelClus(NULL)
    , NumberOfStripClus(NULL)
    , RatioOfPixelAndStripClus(NULL)
    , NumberOfTrkVsClus(NULL)
    , NumberOfTrkVsStripClus(NULL)
    , NumberOfTrkVsPixelClus(NULL)
				*/
    , NumberOfGoodTrkVsClus(NULL)
    , NumberOfTracksVsLS(NULL)
    , NumberOfGoodTracksVsLS(NULL)
    , GoodTracksFractionVsLS(NULL)
    , GoodTracksNumberOfRecHitsPerTrackVsLS(NULL)
				// ADD by Mia for PU monitoring
				// vertex plots to be moved in ad hoc class
    , NumberOfTracksVsGoodPVtx(NULL)
    , NumberOfTracksVsBXlumi(NULL)
    , NumberOfGoodTracksVsGoodPVtx(NULL)
    , NumberOfGoodTracksVsBXlumi(NULL)
    , FractionOfGoodTracksVsGoodPVtx(NULL)
    , FractionOfGoodTracksVsBXlumi(NULL)
				// ADD by Mia in order to deal with LS transitions
    , NumberOfTracks_lumiFlag(NULL)
    , NumberOfGoodTracks_lumiFlag(NULL)

    , builderName              ( conf_.getParameter<std::string>("TTRHBuilder"))
    , doTrackerSpecific_       ( conf_.getParameter<bool>("doTrackerSpecific") )
    , doLumiAnalysis           ( conf_.getParameter<bool>("doLumiAnalysis"))
    , doProfilesVsLS_          ( conf_.getParameter<bool>("doProfilesVsLS"))
    , doAllPlots               ( conf_.getParameter<bool>("doAllPlots"))
    , doGeneralPropertiesPlots_( conf_.getParameter<bool>("doGeneralPropertiesPlots"))
    , doHitPropertiesPlots_    ( conf_.getParameter<bool>("doHitPropertiesPlots"))
    , doGoodTrackPlots_        ( conf_.getParameter<bool>("doGoodTrackPlots") )
    , doPUmonitoring_          ( conf_.getParameter<bool>("doPUmonitoring") )
    , genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig))
{

  if ( doPUmonitoring_ ) {

    // get flag from the configuration
    doPlotsVsBXlumi_   = conf_.getParameter<bool>("doPlotsVsBXlumi");
    doPlotsVsGoodPVtx_ = conf_.getParameter<bool>("doPlotsVsGoodPVtx");

    if ( doPlotsVsBXlumi_ )
      theLumiDetails_ = new GetLumi( iConfig.getParameter<edm::ParameterSet>("BXlumiSetup") );

    std::vector<edm::InputTag> primaryVertexInputTags    = conf_.getParameter<std::vector<edm::InputTag> >("primaryVertexInputTags");
    std::vector<edm::InputTag> selPrimaryVertexInputTags = conf_.getParameter<std::vector<edm::InputTag> >("selPrimaryVertexInputTags");
    std::vector<std::string>   pvLabels                  = conf_.getParameter<std::vector<std::string> >  ("pvLabels");

    if (primaryVertexInputTags.size()==pvLabels.size() and primaryVertexInputTags.size()==selPrimaryVertexInputTags.size()) {
      for (size_t i=0; i<primaryVertexInputTags.size(); i++) {
	edm::InputTag iPVinputTag    = primaryVertexInputTags[i];
	edm::InputTag iSelPVinputTag = selPrimaryVertexInputTags[i];
	std::string   iPVlabel       = pvLabels[i];
	
	theVertexMonitor.push_back(new VertexMonitor(conf_,iPVinputTag,iSelPVinputTag,iPVlabel));
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

    // parameters from the configuration
    std::string Quality      = conf_.getParameter<std::string>("Quality");
    std::string AlgoName     = conf_.getParameter<std::string>("AlgoName");
    std::string MEFolderName = conf_.getParameter<std::string>("FolderName"); 

    // test for the Quality veriable validity
    if( Quality != "")
    {
        if( Quality != "highPurity" && Quality != "tight" && Quality != "loose") 
        {
            edm::LogWarning("TrackingMonitor")  << "Qualty Name is invalid, using no quality criterea by default";
            Quality = "";
        }
    }

    // use the AlgoName and Quality Name
    std::string CategoryName = Quality != "" ? AlgoName + "_" + Quality : AlgoName;

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
    if
    (
        StateName != "OuterSurface" &&
        StateName != "InnerSurface" &&
        StateName != "ImpactPoint"  &&
        StateName != "default"      &&
        StateName != "All"
    )
    {
        // print warning
        edm::LogWarning("TrackingMonitor")  << "State Name is invalid, using 'ImpactPoint' by default";
    }

    dqmStore_->setCurrentFolder(MEFolderName);

    // book the General Property histograms
    // ---------------------------------------------------------------------------------//

    if (doGeneralPropertiesPlots_ || doAllPlots){
     
      dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");

      histname = "NumberOfTracks_" + CategoryName;
      // MODIFY by Mia in order to cope w/ high multiplicity
      //      NumberOfTracks = dqmStore_->book1D(histname, histname, TKNoBin, TKNoMin, TKNoMax);
      NumberOfTracks = dqmStore_->book1D(histname, histname, 3*TKNoBin, TKNoMin, (TKNoMax+0.5)*3.-0.5);
      NumberOfTracks->setAxisTitle("Number of Tracks per Event", 1);
      NumberOfTracks->setAxisTitle("Number of Events", 2);
      
      histname = "NumberOfMeanRecHitsPerTrack_" + CategoryName;
      NumberOfMeanRecHitsPerTrack = dqmStore_->book1D(histname, histname, MeanHitBin, MeanHitMin, MeanHitMax);
      NumberOfMeanRecHitsPerTrack->setAxisTitle("Mean number of found RecHits per Track", 1);
      NumberOfMeanRecHitsPerTrack->setAxisTitle("Entries", 2);
      
      histname = "NumberOfMeanLayersPerTrack_" + CategoryName;
      NumberOfMeanLayersPerTrack = dqmStore_->book1D(histname, histname, MeanLayBin, MeanLayMin, MeanLayMax);
      NumberOfMeanLayersPerTrack->setAxisTitle("Mean number of Layers per Track", 1);
      NumberOfMeanLayersPerTrack->setAxisTitle("Entries", 2);
      
      dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties/GoodTracks");

      histname = "NumberOfGoodTracks_" + CategoryName;
      NumberOfGoodTracks = dqmStore_->book1D(histname, histname, TKNoBin, TKNoMin, TKNoMax);
      NumberOfGoodTracks->setAxisTitle("Number of Good Tracks per Event", 1);
      NumberOfGoodTracks->setAxisTitle("Number of Events", 2);
      
      histname = "FractionOfGoodTracks_" + CategoryName;
      FractionOfGoodTracks = dqmStore_->book1D(histname, histname, 101, -0.005, 1.005);
      FractionOfGoodTracks->setAxisTitle("Fraction of High Purity Tracks (Tracks with Pt>1GeV)", 1);
      FractionOfGoodTracks->setAxisTitle("Entries", 2);
    }

    if ( doLumiAnalysis ) {
      // add by Mia in order to deal with LS transitions
      dqmStore_->setCurrentFolder(MEFolderName+"/LSanalysis");

      histname = "NumberOfTracks_lumiFlag_" + CategoryName;
      NumberOfTracks_lumiFlag = dqmStore_->book1D(histname, histname, 3*TKNoBin, TKNoMin, (TKNoMax+0.5)*3.-0.5);
      NumberOfTracks_lumiFlag->setAxisTitle("Number of Tracks per Event", 1);
      NumberOfTracks_lumiFlag->setAxisTitle("Number of Events", 2);

      if ( doGoodTrackPlots_ ) {
	histname = "NumberOfGoodTracks_lumiFlag_" + CategoryName;
	NumberOfGoodTracks_lumiFlag = dqmStore_->book1D(histname, histname, TKNoBin, TKNoMin, TKNoMax);
	NumberOfGoodTracks_lumiFlag->setAxisTitle("Number of Good Tracks per Event", 1);
	NumberOfGoodTracks_lumiFlag->setAxisTitle("Number of Events", 2);
      }
      
    }

    // book profile plots vs LS :  
    //---------------------------  
   
 
    if ( doProfilesVsLS_ || doAllPlots) {

      dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");

      histname = "NumberOfTracksVsLS_"+ CategoryName;
      NumberOfTracksVsLS = dqmStore_->bookProfile(histname,histname, LSBin,LSMin,LSMax, TKNoMin, (TKNoMax+0.5)*3.-0.5,"");
      NumberOfTracksVsLS->getTH1()->SetBit(TH1::kCanRebin);
      NumberOfTracksVsLS->setAxisTitle("#Lumi section",1);
      NumberOfTracksVsLS->setAxisTitle("Number of  Tracks",2);

      if (doGoodTrackPlots_ || doAllPlots ){
	dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties/GoodTracks");

	histname = "NumberOfGoodTracksVsLS_"+ CategoryName;
	NumberOfGoodTracksVsLS = dqmStore_->bookProfile(histname,histname, LSBin,LSMin,LSMax, TKNoMin, TKNoMax,"");
	NumberOfGoodTracksVsLS->getTH1()->SetBit(TH1::kCanRebin);
	NumberOfGoodTracksVsLS->setAxisTitle("#Lumi section",1);
	NumberOfGoodTracksVsLS->setAxisTitle("Number of Good Tracks",2);
	
	histname = "GoodTracksFractionVsLS_"+ CategoryName;
	GoodTracksFractionVsLS = dqmStore_->bookProfile(histname,histname, LSBin,LSMin,LSMax,0,1.1,"");
	GoodTracksFractionVsLS->getTH1()->SetBit(TH1::kCanRebin);
	GoodTracksFractionVsLS->setAxisTitle("#Lumi section",1);
	GoodTracksFractionVsLS->setAxisTitle("Fraction of Good Tracks",2);
	
	histname = "GoodTracksNumberOfRecHitsPerTrackVsLS_" + CategoryName;
	GoodTracksNumberOfRecHitsPerTrackVsLS = dqmStore_->bookProfile(histname,histname, LSBin,LSMin,LSMax,0.,40.,"");
	GoodTracksNumberOfRecHitsPerTrackVsLS->getTH1()->SetBit(TH1::kCanRebin);
	GoodTracksNumberOfRecHitsPerTrackVsLS->setAxisTitle("#Lumi section",1);
	GoodTracksNumberOfRecHitsPerTrackVsLS->setAxisTitle("Mean number of RecHits per Good track",2);
      }
    }

    // book PU monitoring plots :  
    //---------------------------  
   
    if ( doPUmonitoring_ ) {

      for (size_t i=0; i<theVertexMonitor.size(); i++)
	theVertexMonitor[i]->beginJob(dqmStore_);

      dqmStore_->setCurrentFolder(MEFolderName+"/PUmonitoring");
      
      if ( doPlotsVsGoodPVtx_ ) {
	// get binning from the configuration
	int    GoodPVtxBin   = conf_.getParameter<int>("GoodPVtxBin");
	double GoodPVtxMin   = conf_.getParameter<double>("GoodPVtxMin");
	double GoodPVtxMax   = conf_.getParameter<double>("GoodPVtxMax");

	histname = "NumberOfTracksVsGoodPVtx";
	NumberOfTracksVsGoodPVtx = dqmStore_->bookProfile(histname,histname,GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,TKNoMin, (TKNoMax+0.5)*3.-0.5,"");
	NumberOfTracksVsGoodPVtx->getTH1()->SetBit(TH1::kCanRebin);
	NumberOfTracksVsGoodPVtx->setAxisTitle("Number of PV",1);
	NumberOfTracksVsGoodPVtx->setAxisTitle("Mean number of Tracks per Event",2);
      
	histname = "NumberOfGoodTracksVsGoodPVtx";
	NumberOfGoodTracksVsGoodPVtx = dqmStore_->bookProfile(histname,histname,GoodPVtxBin,GoodPVtxMin,GoodPVtxMax, TKNoMin, TKNoMax,"");
	NumberOfGoodTracksVsGoodPVtx->getTH1()->SetBit(TH1::kCanRebin);
	NumberOfGoodTracksVsGoodPVtx->setAxisTitle("Number of PV",1);
	NumberOfGoodTracksVsGoodPVtx->setAxisTitle("Mean number of Good Tracks per Event",2);

	histname = "FractionOfGoodTracksVsGoodPVtx";
	FractionOfGoodTracksVsGoodPVtx = dqmStore_->bookProfile(histname,histname,GoodPVtxBin,GoodPVtxMin,GoodPVtxMax, TKNoMin, TKNoMax,"");
	FractionOfGoodTracksVsGoodPVtx->getTH1()->SetBit(TH1::kCanRebin);
	FractionOfGoodTracksVsGoodPVtx->setAxisTitle("Number of PV",1);
	FractionOfGoodTracksVsGoodPVtx->setAxisTitle("Mean fraction of Good Tracks per Event",2);
      }
      
      if ( doPlotsVsBXlumi_ ) {
	// get binning from the configuration
	edm::ParameterSet BXlumiParameters = conf_.getParameter<edm::ParameterSet>("BXlumiSetup");
	int    BXlumiBin   = BXlumiParameters.getParameter<int>("BXlumiBin");
	double BXlumiMin   = BXlumiParameters.getParameter<double>("BXlumiMin");
	double BXlumiMax   = BXlumiParameters.getParameter<double>("BXlumiMax");
      
	histname = "NumberOfTracksVsBXlumi_"+ CategoryName;
	NumberOfTracksVsBXlumi = dqmStore_->bookProfile(histname,histname, BXlumiBin,BXlumiMin,BXlumiMax, TKNoMin, TKNoMax,"");
	NumberOfTracksVsBXlumi->getTH1()->SetBit(TH1::kCanRebin);
	NumberOfTracksVsBXlumi->setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
	NumberOfTracksVsBXlumi->setAxisTitle("Mean number of Good Tracks",2);

	histname = "NumberOfGoodTracksVsBXlumi_"+ CategoryName;
	NumberOfGoodTracksVsBXlumi = dqmStore_->bookProfile(histname,histname, BXlumiBin,BXlumiMin,BXlumiMax, TKNoMin, TKNoMax,"");
	NumberOfGoodTracksVsBXlumi->getTH1()->SetBit(TH1::kCanRebin);
	NumberOfGoodTracksVsBXlumi->setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
	NumberOfGoodTracksVsBXlumi->setAxisTitle("Mean number of Good Tracks",2);
      
	histname = "FractionOfGoodTracksVsBXlumi_"+ CategoryName;
	FractionOfGoodTracksVsBXlumi = dqmStore_->bookProfile(histname,histname, BXlumiBin,BXlumiMin,BXlumiMax, TKNoMin, TKNoMax,"");
	FractionOfGoodTracksVsBXlumi->getTH1()->SetBit(TH1::kCanRebin);
	FractionOfGoodTracksVsBXlumi->setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
	FractionOfGoodTracksVsBXlumi->setAxisTitle("Mean fraction of Good Tracks",2);
      
      }
    }

    theTrackAnalyzer->beginJob(dqmStore_);

    // book the Seed Property histograms
    // ---------------------------------------------------------------------------------//

    dqmStore_->setCurrentFolder(MEFolderName+"/TrackBuilding");

    doAllSeedPlots=conf_.getParameter<bool>("doSeedParameterHistos");
    doSeedNumberPlot=conf_.getParameter<bool>("doSeedNumberHisto");
    doSeedVsClusterPlot=conf_.getParameter<bool>("doSeedVsClusterHisto");
    //    if (doAllPlots) doAllSeedPlots=true;

    runTrackBuildingAnalyzerForSeed=(doAllSeedPlots || conf_.getParameter<bool>("doSeedPTHisto") ||conf_.getParameter<bool>("doSeedETAHisto") || conf_.getParameter<bool>("doSeedPHIHisto") || conf_.getParameter<bool>("doSeedPHIVsETAHisto") || conf_.getParameter<bool>("doSeedThetaHisto") || conf_.getParameter<bool>("doSeedQHisto") || conf_.getParameter<bool>("doSeedDxyHisto") || conf_.getParameter<bool>("doSeedDzHisto") || conf_.getParameter<bool>("doSeedNRecHitsHisto") || conf_.getParameter<bool>("doSeedNVsPhiProf")|| conf_.getParameter<bool>("doSeedNVsEtaProf"));

    edm::InputTag seedProducer   = conf_.getParameter<edm::InputTag>("SeedProducer");

    if (doAllSeedPlots || doSeedNumberPlot){
      histname = "NumberOfSeeds_"+ seedProducer.label() + "_"+ CategoryName;
      NumberOfSeeds = dqmStore_->book1D(histname, histname, TKNoSeedBin, TKNoSeedMin, TKNoSeedMax);
      NumberOfSeeds->setAxisTitle("Number of Seeds per Event", 1);
      NumberOfSeeds->setAxisTitle("Number of Events", 2);
    }

    if (doAllSeedPlots || doSeedVsClusterPlot){

      ClusterLabels=  conf_.getParameter<std::vector<std::string> >("ClusterLabels");

      std::vector<double> histoMin,histoMax;
      std::vector<int> histoBin; //these vectors are for max min and nbins in histograms 

      int    NClusPxBin  = conf_.getParameter<int>(   "NClusPxBin");
      double NClusPxMin  = conf_.getParameter<double>("NClusPxMin");
      double NClusPxMax = conf_.getParameter<double>("NClusPxMax");
      
      int    NClusStrBin = conf_.getParameter<int>(   "NClusStrBin");
      double NClusStrMin = conf_.getParameter<double>("NClusStrMin");
      double NClusStrMax = conf_.getParameter<double>("NClusStrMax");

      setMaxMinBin(histoMin,histoMax,histoBin,NClusStrMin,NClusStrMax,NClusStrBin,NClusPxMin,NClusPxMax,NClusPxBin);
     
      for (uint i=0; i<ClusterLabels.size(); i++){
	histname = "SeedsVsClusters_" + seedProducer.label() + "_Vs_" + ClusterLabels[i] + "_" + CategoryName;
	SeedsVsClusters.push_back(dynamic_cast<MonitorElement*>(dqmStore_->book2D(histname, histname, histoBin[i], histoMin[i], histoMax[i],
										  TKNoSeedBin, TKNoSeedMin, TKNoSeedMax)));
	SeedsVsClusters[i]->setAxisTitle("Number of Clusters", 1);
	SeedsVsClusters[i]->setAxisTitle("Number of Seeds", 2);
      }
    }
    

    doTkCandPlots=conf_.getParameter<bool>("doTrackCandHistos");
    //    if (doAllPlots) doTkCandPlots=true;

    if (doTkCandPlots){

      edm::InputTag tcProducer     = conf_.getParameter<edm::InputTag>("TCProducer");

      histname = "NumberOfTrackCandidates_"+ tcProducer.label() + "_"+ CategoryName;
      NumberOfTrackCandidates = dqmStore_->book1D(histname, histname, TCNoBin, TCNoMin, TCNoMax);
      NumberOfTrackCandidates->setAxisTitle("Number of Track Candidates per Event", 1);
      NumberOfTrackCandidates->setAxisTitle("Number of Event", 2);
    }

   
    theTrackBuildingAnalyzer->beginJob(dqmStore_);
    

    if (doLumiAnalysis) {
//      if (NumberOfTracks) NumberOfTracks->setLumiFlag();
//      if (NumberOfGoodTracks) NumberOfGoodTracks->setLumiFlag();
//      if (FractionOfGoodTracks) FractionOfGoodTracks->setLumiFlag();
      if ( NumberOfTracks_lumiFlag       ) NumberOfTracks_lumiFlag       -> setLumiFlag();
      if ( NumberOfGoodTracks_lumiFlag   ) NumberOfGoodTracks_lumiFlag   -> setLumiFlag();
      theTrackAnalyzer->setLumiFlag();    
    }

    if (doTrackerSpecific_ || doAllPlots) {

      ClusterLabels=  conf_.getParameter<std::vector<std::string> >("ClusterLabels");

      std::vector<double> histoMin,histoMax;
      std::vector<int> histoBin; //these vectors are for max min and nbins in histograms 

      /*
      int    NClusPxBin  = conf_.getParameter<int>(   "NClusPxBin");
      double NClusPxMin  = conf_.getParameter<double>("NClusPxMin");
      double NClusPxMax = conf_.getParameter<double>("NClusPxMax");
      
      int    NClusStrBin = conf_.getParameter<int>(   "NClusStrBin");
      double NClusStrMin = conf_.getParameter<double>("NClusStrMin");
      double NClusStrMax = conf_.getParameter<double>("NClusStrMax");
      */

      /*
      int    NClus2DTotBin = conf_.getParameter<int>(   "NClus2DTotBin");
      double NClus2DTotMin = conf_.getParameter<double>("NClus2DTotMin");
      double NClus2DTotMax = conf_.getParameter<double>("NClus2DTotMax");
      */

      int    NClusStrBin = conf_.getParameter<int>(   "NClusStrBin");
      double NClusStrMin = conf_.getParameter<double>("NClusStrMin");
      double NClusStrMax = conf_.getParameter<double>("NClusStrMax");

      int    NClusPxBin = conf_.getParameter<int>(   "NClusPxBin");
      double NClusPxMin = conf_.getParameter<double>("NClusPxMin");
      double NClusPxMax = conf_.getParameter<double>("NClusPxMax");

      int    NTrk2DBin     = conf_.getParameter<int>(   "NTrk2DBin");
      double NTrk2DMin     = conf_.getParameter<double>("NTrk2DMin");
      double NTrk2DMax     = conf_.getParameter<double>("NTrk2DMax");

      //      setMaxMinBin(histoMin,histoMax,histoBin,NClusStrMin,NClusStrMax,NClusStrBin,NClusPxMin,NClusPxMax,NClusPxBin);
      setMaxMinBin(histoMin,histoMax,histoBin,
		   NClusStrMin,NClusStrMax,NClusStrBin,
		   NClusPxMin,  NClusPxMax,  NClusPxBin);
     
      /*
      int    NClusPxBin  = conf_.getParameter<int>(   "NClusPxBin");
      double NClusPxMin  = conf_.getParameter<double>("NClusPxMin");
      double NClusPxMax = conf_.getParameter<double>("NClusPxMax");

      int    NClusStrBin = conf_.getParameter<int>(   "NClusStrBin");
      double NClusStrMin = conf_.getParameter<double>("NClusStrMin");
      double NClusStrMax = conf_.getParameter<double>("NClusStrMax");
      */

      dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties");

      for (uint i=0; i<ClusterLabels.size(); i++){

	dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties");
	histname = "TracksVs" + ClusterLabels[i] + "Cluster_" + CategoryName;
	NumberOfTrkVsClusters.push_back(dynamic_cast<MonitorElement*>(dqmStore_->book2D(histname, histname,
											histoBin[i], histoMin[i], histoMax[i],
											NTrk2DBin,NTrk2DMin,NTrk2DMax
											)));
	std::string title = "Number of " + ClusterLabels[i] + " Clusters";
	NumberOfTrkVsClusters[i]->setAxisTitle(title, 1);
	NumberOfTrkVsClusters[i]->setAxisTitle("Number of Seeds", 2);

	if (doGoodTrackPlots_ || doAllPlots ) {

	  dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties/GoodTracks");

	  if(ClusterLabels[i].compare("Tot")==0){
	    histname = "GoodTracksVs" + ClusterLabels[i] + "Cluster_" + CategoryName; 
	    NumberOfGoodTrkVsClus = dqmStore_->book2D(histname,histname,
						      histoBin[i], histoMin[i], histoMax[i],
						      TKNoBin,TKNoMin,TKNoMax
						      );
	    NumberOfGoodTrkVsClus->setAxisTitle("# of Clusters in (Pixel+Strip) Detectors", 1);
	    NumberOfGoodTrkVsClus->setAxisTitle("Number of Good Tracks", 2);
	  }
	}
      }

      /*
      histname = "TracksVsClusters_" + CategoryName; 
      NumberOfTrkVsClus = dqmStore_->book2D(histname,histname,
					    NClus2DTotBin,NClus2DTotMin,NClus2DTotMax,
					    NTrk2DBin,NTrk2DMin,NTrk2DMax
					    );
      NumberOfTrkVsClus->setAxisTitle("# of Clusters in (Pixel+Strip) Detectors", 1);
      NumberOfTrkVsClus->setAxisTitle("Number of Tracks", 2);
      */
    }

    
}

// -- BeginRun
//---------------------------------------------------------------------------------//
void TrackingMonitor::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
 
  // Initialize the GenericTriggerEventFlag
  if ( genTriggerEventFlag_->on() ) genTriggerEventFlag_->initRun( iRun, iSetup );
}

// - BeginLumi
// ---------------------------------------------------------------------------------//
void TrackingMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup&  eSetup) {

  if (doLumiAnalysis) {
//    dqmStore_->softReset(NumberOfTracks);
//    dqmStore_->softReset(NumberOfGoodTracks);
//    dqmStore_->softReset(FractionOfGoodTracks);
//    theTrackAnalyzer->doSoftReset(dqmStore_);    
    if ( NumberOfTracks_lumiFlag       ) NumberOfTracks_lumiFlag       -> Reset();
    if ( doGoodTrackPlots_ )
      if ( NumberOfGoodTracks_lumiFlag   ) NumberOfGoodTracks_lumiFlag   -> Reset();
    theTrackAnalyzer->doReset(dqmStore_);    
  }
}

// -- Analyse
// ---------------------------------------------------------------------------------//
void TrackingMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
    // Filter out events if Trigger Filtering is requested
    if (genTriggerEventFlag_->on()&& ! genTriggerEventFlag_->accept( iEvent, iSetup) ) return;

    // input tags for collections from the configuration
    edm::InputTag trackProducer  = conf_.getParameter<edm::InputTag>("TrackProducer");
    edm::InputTag seedProducer   = conf_.getParameter<edm::InputTag>("SeedProducer");
    edm::InputTag tcProducer     = conf_.getParameter<edm::InputTag>("TCProducer");
    edm::InputTag bsSrc          = conf_.getParameter<edm::InputTag>("beamSpot");
    edm::InputTag pvSrc_         = conf_.getParameter<edm::InputTag>("primaryVertex");
    std::string Quality = conf_.getParameter<std::string>("Quality");
    std::string Algo    = conf_.getParameter<std::string>("AlgoName");

    //  Analyse the tracks
    //  if the collection is empty, do not fill anything
    // ---------------------------------------------------------------------------------//

    // get the track collection
    edm::Handle<reco::TrackCollection> trackHandle;
    iEvent.getByLabel(trackProducer, trackHandle);

    if (trackHandle.isValid()) 
    {

       reco::TrackCollection trackCollection = *trackHandle;
        // calculate the mean # rechits and layers
        int totalNumTracks = 0, totalRecHits = 0, totalLayers = 0;
	int totalNumHPTracks = 0, totalNumPt1Tracks = 0, totalNumHPPt1Tracks = 0;
	double frac = 0.;
        for (reco::TrackCollection::const_iterator track = trackCollection.begin(); track!=trackCollection.end(); ++track) 
        {

	    if( track->quality(reco::TrackBase::highPurity) ) {
	      ++totalNumHPTracks;
	      if ( track->pt() >= 1. ) {
		++totalNumHPPt1Tracks;
		if ( doProfilesVsLS_ || doAllPlots)
		  if ( doGoodTrackPlots_ || doAllPlots ) {
		    GoodTracksNumberOfRecHitsPerTrackVsLS->Fill(static_cast<double>(iEvent.id().luminosityBlock()),track->recHitsSize());
		  }
	      }
	    }
	    
	    if ( track->pt() >= 1. ) ++totalNumPt1Tracks;
	

            if( Quality == "highPurity") 
            {
                if( !track->quality(reco::TrackBase::highPurity) ) continue;
            }
            else if( Quality == "tight") 
            {
                if( !track->quality(reco::TrackBase::tight) ) continue;
            }
            else if( Quality == "loose") 
            {
                if( !track->quality(reco::TrackBase::loose) ) continue;
            }
            
            totalNumTracks++;
            totalRecHits    += track->numberOfValidHits();
            totalLayers     += track->hitPattern().trackerLayersWithMeasurement();

            // do analysis per track
            theTrackAnalyzer->analyze(iEvent, iSetup, *track);
        }

	if (totalNumPt1Tracks > 0) frac = static_cast<double>(totalNumHPPt1Tracks)/static_cast<double>(totalNumPt1Tracks);

	if (doGeneralPropertiesPlots_ || doAllPlots){
	  NumberOfTracks       -> Fill(totalNumTracks);
	  NumberOfGoodTracks   -> Fill(totalNumHPPt1Tracks);
	  FractionOfGoodTracks -> Fill(frac);
	}

	if ( doProfilesVsLS_ || doAllPlots) {
	  NumberOfTracksVsLS->Fill(static_cast<double>(iEvent.id().luminosityBlock()),totalNumTracks);
	  if ( doGoodTrackPlots_ || doAllPlots ) {
	    NumberOfGoodTracksVsLS->Fill(static_cast<double>(iEvent.id().luminosityBlock()),totalNumHPPt1Tracks);
	    GoodTracksFractionVsLS->Fill(static_cast<double>(iEvent.id().luminosityBlock()),frac);
	  }
	}

	if ( doLumiAnalysis ) {
	  NumberOfTracks_lumiFlag       -> Fill(totalNumTracks);
	  if ( doGoodTrackPlots_ )
	    NumberOfGoodTracks_lumiFlag   -> Fill(totalNumHPPt1Tracks);
	}

	if (doGeneralPropertiesPlots_ || doAllPlots){
	  if( totalNumTracks > 0 )
	      {
		double meanRecHits = static_cast<double>(totalRecHits) / static_cast<double>(totalNumTracks);
		double meanLayers  = static_cast<double>(totalLayers)  / static_cast<double>(totalNumTracks);
		NumberOfMeanRecHitsPerTrack -> Fill(meanRecHits);
		NumberOfMeanLayersPerTrack  -> Fill(meanLayers);
	      }
	  }
    
	//  Analyse the Track Building variables 
	//  if the collection is empty, do not fill anything
	// ---------------------------------------------------------------------------------//
	
	    	   	    
	    // fill the TrackCandidate info
	    if (doTkCandPlots) 
	      {
	
		// magnetic field
		edm::ESHandle<MagneticField> theMF;
		iSetup.get<IdealMagneticFieldRecord>().get(theMF);  
		
		// get the beam spot
		edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
		iEvent.getByLabel(bsSrc,recoBeamSpotHandle);
		const reco::BeamSpot& bs = *recoBeamSpotHandle;      
		
		// get the candidate collection
		edm::Handle<TrackCandidateCollection> theTCHandle;
		iEvent.getByLabel(tcProducer, theTCHandle ); 
		const TrackCandidateCollection& theTCCollection = *theTCHandle;

		if (theTCHandle.isValid())
		  {
		    NumberOfTrackCandidates->Fill(theTCCollection.size());
		    iSetup.get<TransientRecHitRecord>().get(builderName,theTTRHBuilder);
		    for( TrackCandidateCollection::const_iterator cand = theTCCollection.begin(); cand != theTCCollection.end(); ++cand)
		      {
			theTrackBuildingAnalyzer->analyze(iEvent, iSetup, *cand, bs, theMF, theTTRHBuilder);
		      }
		  }
		else
		  {
		    edm::LogWarning("TrackingMonitor") << "No Track Candidates in the event.  Not filling associated histograms";
		  }
	      }
    
	    //plots for trajectory seeds

	    if (doAllSeedPlots || doSeedNumberPlot || doSeedVsClusterPlot || runTrackBuildingAnalyzerForSeed){
	    
	    // get the seed collection
	    edm::Handle<edm::View<TrajectorySeed> > seedHandle;
	    iEvent.getByLabel(seedProducer, seedHandle);
	    const edm::View<TrajectorySeed>& seedCollection = *seedHandle;
	    
	    // fill the seed info
	    if (seedHandle.isValid()) 
	      {
		if(doAllSeedPlots || doSeedNumberPlot) NumberOfSeeds->Fill(seedCollection.size());

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
		  iEvent.getByLabel(bsSrc,recoBeamSpotHandle);
		  const reco::BeamSpot& bs = *recoBeamSpotHandle;      
		  
		  iSetup.get<TransientRecHitRecord>().get(builderName,theTTRHBuilder);
		  for(size_t i=0; i < seedHandle->size(); ++i)
		    {
		      edm::RefToBase<TrajectorySeed> seed(seedHandle, i);
		      theTrackBuildingAnalyzer->analyze(iEvent, iSetup, *seed, bs, theMF, theTTRHBuilder);
		    }
		}
		
	      }
	    else
	      {
		edm::LogWarning("TrackingMonitor") << "No Trajectory seeds in the event.  Not filling associated histograms";
	      }
	    }


	 if (doTrackerSpecific_ || doAllPlots) 
          {
	    std::vector<int> NClus;
	    setNclus(iEvent,NClus);
	    for (uint  i=0; i< ClusterLabels.size(); i++){
	      NumberOfTrkVsClusters[i]->Fill(NClus[i],totalNumTracks);
	    }
	    if ( doGoodTrackPlots_ || doAllPlots ) {
	      for (uint  i=0; i< ClusterLabels.size(); i++){
		if(ClusterLabels[i].compare("Tot")==0){
		  NumberOfGoodTrkVsClus->Fill( NClus[i],totalNumHPPt1Tracks);
		}
	      }
	    }

	    /*
	    edm::Handle< edmNew::DetSetVector<SiStripCluster> > strip_clusters;
	    iEvent.getByLabel("siStripClusters", strip_clusters);
	    edm::Handle< edmNew::DetSetVector<SiPixelCluster> > pixel_clusters;
	    iEvent.getByLabel("siPixelClusters", pixel_clusters);
            if (strip_clusters.isValid() && pixel_clusters.isValid()) 
              {
                unsigned int ncluster_pix   = (*pixel_clusters).dataSize(); 
                unsigned int ncluster_strip = (*strip_clusters).dataSize(); 
                double ratio = 0.0;
                if ( ncluster_pix > 0) ratio = atan(ncluster_pix*1.0/ncluster_strip);

		NumberOfStripClus->Fill(ncluster_strip);
		NumberOfPixelClus->Fill(ncluster_pix);
		RatioOfPixelAndStripClus->Fill(ratio);

		NumberOfTrkVsClus->Fill( ncluster_strip+ncluster_pix,totalNumTracks);
	    
		if (doGoodTrackPlots_) {
		  NumberOfGoodTrkVsClus->Fill( ncluster_strip+ncluster_pix,totalNumHPPt1Tracks);
		}
              }
	    */
          }

	 if ( doPUmonitoring_ ) {
	   
	   // do vertex monitoring
	   for (size_t i=0; i<theVertexMonitor.size(); i++)
	     theVertexMonitor[i]->analyze(iEvent, iSetup);
	   
	   if ( doPlotsVsGoodPVtx_ ) {
	     
	     edm::InputTag primaryVertexInputTag = conf_.getParameter<edm::InputTag>("primaryVertex");
	     
	     size_t totalNumGoodPV = 0;
	     edm::Handle< reco::VertexCollection > pvHandle;
	     iEvent.getByLabel(primaryVertexInputTag, pvHandle );
	     if (pvHandle.isValid())
	       {
		 
		 for (reco::VertexCollection::const_iterator pv = pvHandle->begin();
		      pv != pvHandle->end(); ++pv) {
		   
		   //--- pv fake (the pv collection should have size==1 and the pv==beam spot) 
		   if (pv->isFake() || pv->tracksSize()==0) continue;
		   
		   // definition of goodOfflinePrimaryVertex
		   if (pv->ndof() < 4. || pv->z() > 24.)  continue;
		   totalNumGoodPV++;
		 }
		 
		 NumberOfTracksVsGoodPVtx       -> Fill( totalNumGoodPV, totalNumTracks      );
		 NumberOfGoodTracksVsGoodPVtx   -> Fill( totalNumGoodPV, totalNumHPPt1Tracks );
		 FractionOfGoodTracksVsGoodPVtx -> Fill( totalNumGoodPV, frac                );
	       }
	   }
	   
	   if ( doPlotsVsBXlumi_ ) {
	     
	     double bxlumi = theLumiDetails_->getValue(iEvent);
	     
	     NumberOfTracksVsBXlumi       -> Fill( bxlumi, totalNumTracks      );
	     NumberOfGoodTracksVsBXlumi   -> Fill( bxlumi, totalNumHPPt1Tracks );
	     FractionOfGoodTracksVsBXlumi -> Fill( bxlumi, frac                );
	   }
	   
	 }
	 
    }

}


void TrackingMonitor::endRun(const edm::Run&, const edm::EventSetup&) 
{
  if (doLumiAnalysis) {
    /*
    dqmStore_->disableSoftReset(NumberOfTracks);
    dqmStore_->disableSoftReset(NumberOfGoodTracks);
    dqmStore_->disableSoftReset(FractionOfGoodTracks);
    theTrackAnalyzer->undoSoftReset(dqmStore_);    
    */
  }
  
}

void TrackingMonitor::endJob(void) 
{
    bool outputMEsInRootFile   = conf_.getParameter<bool>("OutputMEsInRootFile");
    std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
    if(outputMEsInRootFile)
    {
        dqmStore_->showDirStructure();
        dqmStore_->save(outputFileName);
    }
}

void TrackingMonitor::setMaxMinBin(std::vector<double> &arrayMin,  std::vector<double> &arrayMax, std::vector<int> &arrayBin, double smin, double smax, int sbin, double pmin, double pmax, int pbin) 
{
  arrayMin.resize(ClusterLabels.size());
  arrayMax.resize(ClusterLabels.size());
  arrayBin.resize(ClusterLabels.size());

  for (uint i=0; i<ClusterLabels.size(); ++i){

    if (ClusterLabels[i].compare("Pix")==0) {arrayMin[i]=pmin; arrayMax[i]=pmax; arrayBin[i]=pbin;}
    else if(ClusterLabels[i].compare("Strip")==0){arrayMin[i]=smin; arrayMax[i]=smax; arrayBin[i]=sbin;}
    else if(ClusterLabels[i].compare("Tot")==0){arrayMin[i]=smin; arrayMax[i]=smax+pmax; arrayBin[i]=sbin;}
    else {edm::LogWarning("TrackingMonitor")  << "Cluster Label " << ClusterLabels[i] << " not defined, using strip parameters "; 
      arrayMin[i]=smin; arrayMax[i]=smax; arrayBin[i]=sbin;}

  }
    
}

void TrackingMonitor::setNclus(const edm::Event& iEvent,std::vector<int> &arrayNclus) 
{

  int ncluster_pix=-1;
  int ncluster_strip=-1;

  edm::Handle< edmNew::DetSetVector<SiStripCluster> > strip_clusters;
  iEvent.getByLabel("siStripClusters", strip_clusters);
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> > pixel_clusters;
  iEvent.getByLabel("siPixelClusters", pixel_clusters);

  if (strip_clusters.isValid() && pixel_clusters.isValid()) 
    {
                ncluster_pix   = (*pixel_clusters).dataSize(); 
                ncluster_strip = (*strip_clusters).dataSize(); 
    }

  arrayNclus.resize(ClusterLabels.size());
  for (uint i=0; i<ClusterLabels.size(); ++i){

    if (ClusterLabels[i].compare("Pix")==0) arrayNclus[i]=ncluster_pix ;
    else if(ClusterLabels[i].compare("Strip")==0) arrayNclus[i]=ncluster_strip;
    else if(ClusterLabels[i].compare("Tot")==0)arrayNclus[i]=ncluster_pix+ncluster_strip;
    else {edm::LogWarning("TrackingMonitor") << "Cluster Label " << ClusterLabels[i] << " not defined using stri parametrs ";
      arrayNclus[i]=ncluster_strip ;}
  }
    
}
DEFINE_FWK_MODULE(TrackingMonitor);
