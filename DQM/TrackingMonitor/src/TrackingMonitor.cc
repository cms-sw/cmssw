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
#include "FWCore/Utilities/interface/transform.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/TrackingMonitor/interface/TrackBuildingAnalyzer.h"
#include "DQM/TrackingMonitor/interface/TrackAnalyzer.h"
#include "DQM/TrackingMonitor/interface/VertexMonitor.h"
#include "DQM/TrackingMonitor/interface/TrackingMonitor.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h" 
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include <string>

// TrackingMonitor 
// ----------------------------------------------------------------------------------//

TrackingMonitor::TrackingMonitor(const edm::ParameterSet& iConfig) 
    : confID_ ( iConfig.id() )
    , theTrackBuildingAnalyzer( new TrackBuildingAnalyzer(iConfig) )
    , NumberOfTracks(nullptr)
    , NumberOfMeanRecHitsPerTrack(nullptr)
    , NumberOfMeanLayersPerTrack(nullptr)
				//    , NumberOfGoodTracks(NULL)
    , FractionOfGoodTracks(nullptr)
    , NumberOfTrackingRegions(nullptr)
    , NumberOfSeeds(nullptr)
    , NumberOfSeeds_lumiFlag(nullptr)
    , NumberOfTrackCandidates(nullptr)
    , FractionCandidatesOverSeeds(nullptr)
				//    , NumberOfGoodTrkVsClus(NULL)
    , NumberEventsOfVsLS(nullptr)
    , NumberOfTracksVsLS(nullptr)
				//    , NumberOfGoodTracksVsLS(NULL)
    , GoodTracksFractionVsLS(nullptr)
				//    , GoodTracksNumberOfRecHitsPerTrackVsLS(NULL)
				// ADD by Mia for PU monitoring
				// vertex plots to be moved in ad hoc class
    , NumberOfGoodPVtxVsLS(nullptr)
    , NumberOfGoodPVtxWO0VsLS(nullptr)
    , NumberEventsOfVsBX (nullptr)
    , NumberOfTracksVsBX(nullptr)
    , GoodTracksFractionVsBX(nullptr)
    , NumberOfRecHitsPerTrackVsBX(nullptr)
    , NumberOfGoodPVtxVsBX(nullptr)
    , NumberOfGoodPVtxWO0VsBX(nullptr)
    , NumberOfTracksVsBXlumi(nullptr)
    , NumberOfTracksVsGoodPVtx(nullptr)
    , NumberOfTracksVsPUPVtx(nullptr)
    , NumberEventsOfVsGoodPVtx(nullptr)
    , GoodTracksFractionVsGoodPVtx(nullptr)
    , NumberOfRecHitsPerTrackVsGoodPVtx(nullptr)
    , NumberOfPVtxVsGoodPVtx(nullptr)
    , NumberOfPixelClustersVsGoodPVtx(nullptr)
    , NumberOfStripClustersVsGoodPVtx(nullptr)
    , NumberEventsOfVsLUMI(nullptr)
    , NumberOfTracksVsLUMI(nullptr)
    , GoodTracksFractionVsLUMI(nullptr)
    , NumberOfRecHitsPerTrackVsLUMI(nullptr)
    , NumberOfGoodPVtxVsLUMI(nullptr)
    , NumberOfGoodPVtxWO0VsLUMI(nullptr)
    , NumberOfPixelClustersVsLUMI(nullptr)
    , NumberOfStripClustersVsLUMI(nullptr)
    , NumberOfTracks_lumiFlag(nullptr)
				//    , NumberOfGoodTracks_lumiFlag(NULL)

    , builderName              ( iConfig.getParameter<std::string>("TTRHBuilder"))
    , doTrackerSpecific_       ( iConfig.getParameter<bool>("doTrackerSpecific") )
    , doLumiAnalysis           ( iConfig.getParameter<bool>("doLumiAnalysis"))
    , doProfilesVsLS_          ( iConfig.getParameter<bool>("doProfilesVsLS"))
    , doAllPlots               ( iConfig.getParameter<bool>("doAllPlots"))
    , doGeneralPropertiesPlots_( iConfig.getParameter<bool>("doGeneralPropertiesPlots"))
    , doHitPropertiesPlots_    ( iConfig.getParameter<bool>("doHitPropertiesPlots"))
    , doPUmonitoring_          ( iConfig.getParameter<bool>("doPUmonitoring") )
    , genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("genericTriggerEventPSet"),consumesCollector(), *this))
    , numSelection_       (iConfig.getParameter<std::string>("numCut"))
    , denSelection_       (iConfig.getParameter<std::string>("denCut"))
    , pvNDOF_             ( iConfig.getParameter<int> ("pvNDOF") )
{

  edm::ConsumesCollector c{ consumesCollector() };
  theTrackAnalyzer = new dqm::TrackAnalyzer( iConfig,c );

  // input tags for collections from the configuration
  bsSrc_ = iConfig.getParameter<edm::InputTag>("beamSpot");
  pvSrc_ = iConfig.getParameter<edm::InputTag>("primaryVertex");
  bsSrcToken_ = consumes<reco::BeamSpot>(bsSrc_);
  pvSrcToken_ = mayConsume<reco::VertexCollection>(pvSrc_);

  lumiscalersToken_ = consumes<LumiScalersCollection>(iConfig.getParameter<edm::InputTag>("scal") );

  edm::InputTag alltrackProducer = iConfig.getParameter<edm::InputTag>("allTrackProducer");
  edm::InputTag trackProducer    = iConfig.getParameter<edm::InputTag>("TrackProducer");
  edm::InputTag tcProducer       = iConfig.getParameter<edm::InputTag>("TCProducer");
  edm::InputTag seedProducer     = iConfig.getParameter<edm::InputTag>("SeedProducer");
  allTrackToken_       = consumes<edm::View<reco::Track> >(alltrackProducer);
  trackToken_          = consumes<edm::View<reco::Track> >(trackProducer);
  trackCandidateToken_ = consumes<TrackCandidateCollection>(tcProducer); 
  seedToken_           = consumes<edm::View<TrajectorySeed> >(seedProducer);

  doMVAPlots = iConfig.getParameter<bool>("doMVAPlots");
  if(doMVAPlots) {
    mvaQualityTokens_ = edm::vector_transform(iConfig.getParameter<std::vector<std::string> >("MVAProducers"),
                                              [&](const std::string& tag) {
                                                return std::make_tuple(consumes<MVACollection>(edm::InputTag(tag, "MVAValues")),
                                                                       consumes<QualityMaskCollection>(edm::InputTag(tag, "QualityMasks")));
                                              });
    mvaTrackToken_ = consumes<edm::View<reco::Track> >(iConfig.getParameter<edm::InputTag>("TrackProducerForMVA"));
  }

  doRegionPlots = iConfig.getParameter<bool>("doRegionPlots");
  if(doRegionPlots) {
    regionToken_ = consumes<edm::OwnVector<TrackingRegion> >(iConfig.getParameter<edm::InputTag>("RegionProducer"));
    regionCandidateToken_ = consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("RegionCandidates"));
  }

  edm::InputTag stripClusterInputTag_ = iConfig.getParameter<edm::InputTag>("stripCluster");
  edm::InputTag pixelClusterInputTag_ = iConfig.getParameter<edm::InputTag>("pixelCluster");
  stripClustersToken_ = mayConsume<edmNew::DetSetVector<SiStripCluster> > (stripClusterInputTag_);
  pixelClustersToken_ = mayConsume<edmNew::DetSetVector<SiPixelCluster> > (pixelClusterInputTag_);

  doFractionPlot_ = true;
  if (alltrackProducer.label()==trackProducer.label()) doFractionPlot_ = false;
  
  Quality_  = iConfig.getParameter<std::string>("Quality");
  AlgoName_ = iConfig.getParameter<std::string>("AlgoName");
  
  // get flag from the configuration
  doPlotsVsBXlumi_   = iConfig.getParameter<bool>("doPlotsVsBXlumi");   
  if ( doPlotsVsBXlumi_ )
      theLumiDetails_ = new GetLumi( iConfig.getParameter<edm::ParameterSet>("BXlumiSetup"), c );
  doPlotsVsGoodPVtx_ = iConfig.getParameter<bool>("doPlotsVsGoodPVtx");
  doPlotsVsLUMI_     = iConfig.getParameter<bool>("doPlotsVsLUMI");
  doPlotsVsBX_       = iConfig.getParameter<bool>("doPlotsVsBX");

  if ( doPUmonitoring_ ) {
    
    std::vector<edm::InputTag> primaryVertexInputTags    = iConfig.getParameter<std::vector<edm::InputTag> >("primaryVertexInputTags");
    std::vector<edm::InputTag> selPrimaryVertexInputTags = iConfig.getParameter<std::vector<edm::InputTag> >("selPrimaryVertexInputTags");
    std::vector<std::string>   pvLabels                  = iConfig.getParameter<std::vector<std::string> >  ("pvLabels");
     
    if (primaryVertexInputTags.size()==pvLabels.size() and primaryVertexInputTags.size()==selPrimaryVertexInputTags.size()) {
      for (size_t i=0; i<primaryVertexInputTags.size(); i++) {
 	edm::InputTag iPVinputTag    = primaryVertexInputTags[i];
 	edm::InputTag iSelPVinputTag = selPrimaryVertexInputTags[i];
 	std::string   iPVlabel       = pvLabels[i];
	
 	theVertexMonitor.push_back(new VertexMonitor(iConfig,iPVinputTag,iSelPVinputTag,iPVlabel,c) );
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
   auto const* conf = edm::pset::Registry::instance()->getMapped(confID_);
   assert(conf != nullptr);
   std::string Quality      = conf->getParameter<std::string>("Quality");
   std::string AlgoName     = conf->getParameter<std::string>("AlgoName");
   std::string MEFolderName = conf->getParameter<std::string>("FolderName"); 

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
   int    TKNoBin     = conf->getParameter<int>(   "TkSizeBin");
   double TKNoMin     = conf->getParameter<double>("TkSizeMin");
   double TKNoMax     = conf->getParameter<double>("TkSizeMax");

   int    TCNoBin     = conf->getParameter<int>(   "TCSizeBin");
   double TCNoMin     = conf->getParameter<double>("TCSizeMin");
   double TCNoMax     = conf->getParameter<double>("TCSizeMax");

   int    TKNoSeedBin = conf->getParameter<int>(   "TkSeedSizeBin");
   double TKNoSeedMin = conf->getParameter<double>("TkSeedSizeMin");
   double TKNoSeedMax = conf->getParameter<double>("TkSeedSizeMax");

   int    MeanHitBin  = conf->getParameter<int>(   "MeanHitBin");
   double MeanHitMin  = conf->getParameter<double>("MeanHitMin");
   double MeanHitMax  = conf->getParameter<double>("MeanHitMax");

   int    MeanLayBin  = conf->getParameter<int>(   "MeanLayBin");
   double MeanLayMin  = conf->getParameter<double>("MeanLayMin");
   double MeanLayMax  = conf->getParameter<double>("MeanLayMax");

   int LSBin = conf->getParameter<int>(   "LSBin");
   int LSMin = conf->getParameter<double>("LSMin");
   int LSMax = conf->getParameter<double>("LSMax");

   std::string StateName = conf->getParameter<std::string>("MeasurementState");
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
  
     histname = "NumberEventsVsLS_" + CategoryName;
     NumberEventsOfVsLS = ibooker.book1D(histname,histname, LSBin,LSMin,LSMax);
     NumberEventsOfVsLS->getTH1()->SetCanExtend(TH1::kAllAxes);
     NumberEventsOfVsLS->setAxisTitle("#Lumi section",1);
     NumberEventsOfVsLS->setAxisTitle("Number of events",2);
  
     double GoodPVtxMin   = conf->getParameter<double>("GoodPVtxMin");
     double GoodPVtxMax   = conf->getParameter<double>("GoodPVtxMax");

     histname = "NumberOfGoodPVtxVsLS_" + CategoryName;
     NumberOfGoodPVtxVsLS = ibooker.bookProfile(histname,histname, LSBin,LSMin,LSMax,GoodPVtxMin,GoodPVtxMax,"");
     NumberOfGoodPVtxVsLS->getTH1()->SetCanExtend(TH1::kAllAxes);
     NumberOfGoodPVtxVsLS->setAxisTitle("#Lumi section",1);
     NumberOfGoodPVtxVsLS->setAxisTitle("Mean number of good PV",2);

     histname = "NumberOfGoodPVtxWO0VsLS_" + CategoryName;
     NumberOfGoodPVtxWO0VsLS = ibooker.bookProfile(histname,histname, LSBin,LSMin,LSMax,GoodPVtxMin,GoodPVtxMax,"");
     NumberOfGoodPVtxWO0VsLS->getTH1()->SetCanExtend(TH1::kAllAxes);
     NumberOfGoodPVtxWO0VsLS->setAxisTitle("#Lumi section",1);
     NumberOfGoodPVtxWO0VsLS->setAxisTitle("Mean number of good PV",2);

     if (doFractionPlot_) {
       histname = "GoodTracksFractionVsLS_"+ CategoryName;
       GoodTracksFractionVsLS = ibooker.bookProfile(histname,histname, LSBin,LSMin,LSMax,0,1.1,"");
       GoodTracksFractionVsLS->getTH1()->SetCanExtend(TH1::kAllAxes);
       GoodTracksFractionVsLS->setAxisTitle("#Lumi section",1);
       GoodTracksFractionVsLS->setAxisTitle("Fraction of Good Tracks",2);
     }

     if ( doPlotsVsBX_ || doAllPlots ) {
       ibooker.setCurrentFolder(MEFolderName+"/BXanalysis");
       int BXBin = 3564; double BXMin = 0.5; double BXMax = 3564.5;
       
       histname = "NumberEventsVsBX_" + CategoryName;
       NumberEventsOfVsBX = ibooker.book1D(histname,histname, BXBin,BXMin,BXMax);
       NumberEventsOfVsBX->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberEventsOfVsBX->setAxisTitle("BX",1);
       NumberEventsOfVsBX->setAxisTitle("Number of events",2);
       
       histname = "NumberOfTracksVsBX_"+ CategoryName;
       NumberOfTracksVsBX = ibooker.bookProfile(histname,histname, BXBin,BXMin,BXMax, TKNoMin, (TKNoMax+0.5)*3.-0.5,"");
       NumberOfTracksVsBX->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfTracksVsBX->setAxisTitle("BX",1);
       NumberOfTracksVsBX->setAxisTitle("Number of  Tracks",2);
       
       histname = "NumberOfRecHitsPerTrackVsBX_" + CategoryName;
       NumberOfRecHitsPerTrackVsBX = ibooker.bookProfile(histname,histname, BXBin,BXMin,BXMax,0.,40.,"");
       NumberOfRecHitsPerTrackVsBX->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfRecHitsPerTrackVsBX->setAxisTitle("BX",1);
       NumberOfRecHitsPerTrackVsBX->setAxisTitle("Mean number of Valid RecHits per track",2);
       
       histname = "NumberOfGoodPVtxVsBX_" + CategoryName;
       NumberOfGoodPVtxVsBX = ibooker.bookProfile(histname,histname, BXBin,BXMin,BXMax,GoodPVtxMin,GoodPVtxMax,"");
       NumberOfGoodPVtxVsBX->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfGoodPVtxVsBX->setAxisTitle("BX",1);
       NumberOfGoodPVtxVsBX->setAxisTitle("Mean number of good PV",2);
       
       histname = "NumberOfGoodPVtxWO0VsBX_" + CategoryName;
       NumberOfGoodPVtxWO0VsBX = ibooker.bookProfile(histname,histname, BXBin,BXMin,BXMax,GoodPVtxMin,GoodPVtxMax,"");
       NumberOfGoodPVtxWO0VsBX->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfGoodPVtxWO0VsBX->setAxisTitle("BX",1);
       NumberOfGoodPVtxWO0VsBX->setAxisTitle("Mean number of good PV",2);
       
       if (doFractionPlot_) {
	 histname = "GoodTracksFractionVsBX_"+ CategoryName;
	 GoodTracksFractionVsBX = ibooker.bookProfile(histname,histname, BXBin,BXMin,BXMax,0,1.1,"");
	 GoodTracksFractionVsBX->getTH1()->SetCanExtend(TH1::kAllAxes);
	 GoodTracksFractionVsBX->setAxisTitle("BX",1);
	 GoodTracksFractionVsBX->setAxisTitle("Fraction of Good Tracks",2);
       }
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
       int    GoodPVtxBin   = conf->getParameter<int>("GoodPVtxBin");
       double GoodPVtxMin   = conf->getParameter<double>("GoodPVtxMin");
       double GoodPVtxMax   = conf->getParameter<double>("GoodPVtxMax");
    
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

       histname = "NumberEventsVsGoodPVtx";
       NumberEventsOfVsGoodPVtx = ibooker.book1D(histname,histname,GoodPVtxBin,GoodPVtxMin,GoodPVtxMax);
       NumberEventsOfVsGoodPVtx->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberEventsOfVsGoodPVtx->setAxisTitle("Number of good PV (PU)",1);
       NumberEventsOfVsGoodPVtx->setAxisTitle("Number of events",2);

       if (doFractionPlot_) {
	 histname = "GoodTracksFractionVsGoodPVtx";
	 GoodTracksFractionVsGoodPVtx = ibooker.bookProfile(histname,histname,GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,0., 400.,"");
	 GoodTracksFractionVsGoodPVtx->getTH1()->SetCanExtend(TH1::kAllAxes);
	 GoodTracksFractionVsGoodPVtx->setAxisTitle("Number of good PV (PU)",1);
	 GoodTracksFractionVsGoodPVtx->setAxisTitle("Mean fraction of good tracks",2);
       }

       histname = "NumberOfRecHitsPerTrackVsGoodPVtx";
       NumberOfRecHitsPerTrackVsGoodPVtx = ibooker.bookProfile(histname,histname,GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,0., 100.,"");
       NumberOfRecHitsPerTrackVsGoodPVtx->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfRecHitsPerTrackVsGoodPVtx->setAxisTitle("Number of good PV (PU)",1);
       NumberOfRecHitsPerTrackVsGoodPVtx->setAxisTitle("Mean number of valid rechits per Tracks",2);

       histname = "NumberOfPVtxVsGoodPVtx";
       NumberOfPVtxVsGoodPVtx = ibooker.bookProfile(histname,histname,GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,0., 100.,"");
       NumberOfPVtxVsGoodPVtx->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfPVtxVsGoodPVtx->setAxisTitle("Number of good PV (PU)",1);
       NumberOfPVtxVsGoodPVtx->setAxisTitle("Mean number of vertices",2);

       double NClusPxMin = conf->getParameter<double>("NClusPxMin");
       double NClusPxMax = conf->getParameter<double>("NClusPxMax");
       histname = "NumberOfPixelClustersVsGoodPVtx";
       NumberOfPixelClustersVsGoodPVtx = ibooker.bookProfile(histname,histname,GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,NClusPxMin,NClusPxMax,"");
       NumberOfPixelClustersVsGoodPVtx->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfPixelClustersVsGoodPVtx->setAxisTitle("Number of good PV (PU)",1);
       NumberOfPixelClustersVsGoodPVtx->setAxisTitle("Mean number of pixel clusters",2);
       
       double NClusStrMin = conf->getParameter<double>("NClusStrMin");
       double NClusStrMax = conf->getParameter<double>("NClusStrMax");
       histname = "NumberOfStripClustersVsGoodPVtx";
       NumberOfStripClustersVsGoodPVtx = ibooker.bookProfile(histname,histname,GoodPVtxBin,GoodPVtxMin,GoodPVtxMax,NClusStrMin,NClusStrMax,"");
       NumberOfStripClustersVsGoodPVtx->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfStripClustersVsGoodPVtx->setAxisTitle("Number of good PV (PU)",1);
       NumberOfStripClustersVsGoodPVtx->setAxisTitle("Mean number of strip clusters",2);
  
     }
  

     if ( doPlotsVsLUMI_ || doAllPlots ) {
       ibooker.setCurrentFolder(MEFolderName+"/LUMIanalysis");
       int LUMIBin = conf->getParameter<int>("LUMIBin");
       float LUMIMin = conf->getParameter<double>("LUMIMin");
       float LUMIMax = conf->getParameter<double>("LUMIMax");
       
       histname = "NumberEventsVsLUMI";
       NumberEventsOfVsLUMI = ibooker.book1D(histname,histname,LUMIBin,LUMIMin,LUMIMax);
       NumberEventsOfVsLUMI->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberEventsOfVsLUMI->setAxisTitle("scal lumi [10e30 Hz cm^{-2}]",1);
       NumberEventsOfVsLUMI->setAxisTitle("Number of events",2);
       
       histname = "NumberOfTracksVsLUMI";
       NumberOfTracksVsLUMI = ibooker.bookProfile(histname,histname,LUMIBin,LUMIMin,LUMIMax,0., 400.,"");
       NumberOfTracksVsLUMI->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfTracksVsLUMI->setAxisTitle("scal lumi [10e30 Hz cm^{-2}]",1);
       NumberOfTracksVsLUMI->setAxisTitle("Mean number of vertices",2);
       
       if (doFractionPlot_) {
	 histname = "GoodTracksFractionVsLUMI";
	 GoodTracksFractionVsLUMI = ibooker.bookProfile(histname,histname,LUMIBin,LUMIMin,LUMIMax,0., 100.,"");
	 GoodTracksFractionVsLUMI->getTH1()->SetCanExtend(TH1::kAllAxes);
	 GoodTracksFractionVsLUMI->setAxisTitle("scal lumi [10e30 Hz cm^{-2}]",1);
	 GoodTracksFractionVsLUMI->setAxisTitle("Mean number of vertices",2);
       }

       histname = "NumberOfRecHitsPerTrackVsLUMI";
       NumberOfRecHitsPerTrackVsLUMI = ibooker.bookProfile(histname,histname,LUMIBin,LUMIMin,LUMIMax,0., 20.,"");
       NumberOfRecHitsPerTrackVsLUMI->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfRecHitsPerTrackVsLUMI->setAxisTitle("scal lumi [10e30 Hz cm^{-2}]",1);
       NumberOfRecHitsPerTrackVsLUMI->setAxisTitle("Mean number of vertices",2);

       double GoodPVtxMin   = conf->getParameter<double>("GoodPVtxMin");
       double GoodPVtxMax   = conf->getParameter<double>("GoodPVtxMax");
       
       histname = "NumberOfGoodPVtxVsLUMI";
       NumberOfGoodPVtxVsLUMI = ibooker.bookProfile(histname,histname,LUMIBin,LUMIMin,LUMIMax,GoodPVtxMin,GoodPVtxMax,"");
       NumberOfGoodPVtxVsLUMI->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfGoodPVtxVsLUMI->setAxisTitle("scal lumi [10e30 Hz cm^{-2}]",1);
       NumberOfGoodPVtxVsLUMI->setAxisTitle("Mean number of vertices",2);

       histname = "NumberOfGoodPVtxWO0VsLUMI";
       NumberOfGoodPVtxWO0VsLUMI = ibooker.bookProfile(histname,histname,LUMIBin,LUMIMin,LUMIMax,GoodPVtxMin,GoodPVtxMax,"");
       NumberOfGoodPVtxWO0VsLUMI->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfGoodPVtxWO0VsLUMI->setAxisTitle("scal lumi [10e30 Hz cm^{-2}]",1);
       NumberOfGoodPVtxWO0VsLUMI->setAxisTitle("Mean number of vertices",2);

       double NClusPxMin = conf->getParameter<double>("NClusPxMin");
       double NClusPxMax = conf->getParameter<double>("NClusPxMax");
       histname = "NumberOfPixelClustersVsGoodPVtx";
       NumberOfPixelClustersVsLUMI = ibooker.bookProfile(histname,histname,LUMIBin,LUMIMin,LUMIMax,NClusPxMin,NClusPxMax,"");
       NumberOfPixelClustersVsLUMI->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfPixelClustersVsLUMI->setAxisTitle("scal lumi [10e30 Hz cm^{-2}]",1);
       NumberOfPixelClustersVsLUMI->setAxisTitle("Mean number of pixel clusters",2);
       
       double NClusStrMin = conf->getParameter<double>("NClusStrMin");
       double NClusStrMax = conf->getParameter<double>("NClusStrMax");
       histname = "NumberOfStripClustersVsLUMI";
       NumberOfStripClustersVsLUMI = ibooker.bookProfile(histname,histname,LUMIBin,LUMIMin,LUMIMax,NClusStrMin,NClusStrMax,"");
       NumberOfStripClustersVsLUMI->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfStripClustersVsLUMI->setAxisTitle("scal lumi [10e30 Hz cm^{-2}]",1);
       NumberOfStripClustersVsLUMI->setAxisTitle("Mean number of strip clusters",2);
  
     }
     

     if ( doPlotsVsBXlumi_ ) {
       ibooker.setCurrentFolder(MEFolderName+"/PUmonitoring");
       // get binning from the configuration
       edm::ParameterSet BXlumiParameters = conf->getParameter<edm::ParameterSet>("BXlumiSetup");
       int    BXlumiBin   = BXlumiParameters.getParameter<int>("BXlumiBin");
       double BXlumiMin   = BXlumiParameters.getParameter<double>("BXlumiMin");
       double BXlumiMax   = BXlumiParameters.getParameter<double>("BXlumiMax");
    
       histname = "NumberOfTracksVsBXlumi_"+ CategoryName;
       NumberOfTracksVsBXlumi = ibooker.bookProfile(histname,histname, BXlumiBin,BXlumiMin,BXlumiMax, TKNoMin, TKNoMax,"");
       NumberOfTracksVsBXlumi->getTH1()->SetCanExtend(TH1::kAllAxes);
       NumberOfTracksVsBXlumi->setAxisTitle("lumi BX [10^{30}Hzcm^{-2}]",1);
       NumberOfTracksVsBXlumi->setAxisTitle("Mean number of Tracks",2);
    
     }
   

     theTrackAnalyzer->initHisto(ibooker, iSetup, *conf);

   // book the Seed Property histograms
   // ---------------------------------------------------------------------------------//

   ibooker.setCurrentFolder(MEFolderName+"/TrackBuilding");

   doAllSeedPlots      = conf->getParameter<bool>("doSeedParameterHistos");
   doSeedNumberPlot    = conf->getParameter<bool>("doSeedNumberHisto");
   doSeedLumiAnalysis_ = conf->getParameter<bool>("doSeedLumiAnalysis");
   doSeedVsClusterPlot = conf->getParameter<bool>("doSeedVsClusterHisto");
   //    if (doAllPlots) doAllSeedPlots=true;

   runTrackBuildingAnalyzerForSeed=(doAllSeedPlots || conf->getParameter<bool>("doSeedPTHisto") ||conf->getParameter<bool>("doSeedETAHisto") || conf->getParameter<bool>("doSeedPHIHisto") || conf->getParameter<bool>("doSeedPHIVsETAHisto") || conf->getParameter<bool>("doSeedThetaHisto") || conf->getParameter<bool>("doSeedQHisto") || conf->getParameter<bool>("doSeedDxyHisto") || conf->getParameter<bool>("doSeedDzHisto") || conf->getParameter<bool>("doSeedNRecHitsHisto") || conf->getParameter<bool>("doSeedNVsPhiProf")|| conf->getParameter<bool>("doSeedNVsEtaProf"));

   edm::InputTag seedProducer   = conf->getParameter<edm::InputTag>("SeedProducer");

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

     ClusterLabels=  conf->getParameter<std::vector<std::string> >("ClusterLabels");
  
     std::vector<double> histoMin,histoMax;
     std::vector<int> histoBin; //these vectors are for max min and nbins in histograms 
  
     int    NClusPxBin = conf->getParameter<int>(   "NClusPxBin");
     double NClusPxMin = conf->getParameter<double>("NClusPxMin");
     double NClusPxMax = conf->getParameter<double>("NClusPxMax");
  
     int    NClusStrBin = conf->getParameter<int>(   "NClusStrBin");
     double NClusStrMin = conf->getParameter<double>("NClusStrMin");
     double NClusStrMax = conf->getParameter<double>("NClusStrMax");
  
     setMaxMinBin(histoMin,histoMax,histoBin,NClusStrMin,NClusStrMax,NClusStrBin,NClusPxMin,NClusPxMax,NClusPxBin);
  
     for (uint i=0; i<ClusterLabels.size(); i++){
       histname = "SeedsVsClusters_" + seedProducer.label() + "_Vs_" + ClusterLabels[i] + "_" + CategoryName;
       SeedsVsClusters.push_back(dynamic_cast<MonitorElement*>(ibooker.book2D(histname, histname, histoBin[i], histoMin[i], histoMax[i],
										 TKNoSeedBin, TKNoSeedMin, TKNoSeedMax)));
       SeedsVsClusters[i]->setAxisTitle("Number of Clusters", 1);
       SeedsVsClusters[i]->setAxisTitle("Number of Seeds", 2);
       SeedsVsClusters[i]->getTH2F()->SetCanExtend(TH1::kAllAxes);
     }
   }
  
   if(doRegionPlots) {
     ibooker.setCurrentFolder(MEFolderName+"/TrackBuilding");

     int    regionBin = conf->getParameter<int>(   "RegionSizeBin");
     double regionMin = conf->getParameter<double>("RegionSizeMin");
     double regionMax = conf->getParameter<double>("RegionSizeMax");

     histname = "NumberOfTrackingRegions_"+ seedProducer.label() + "_"+ CategoryName;
     NumberOfTrackingRegions = ibooker.book1D(histname, histname, regionBin, regionMin, regionMax);
     NumberOfTrackingRegions->setAxisTitle("Number of TrackingRegions per Event", 1);
     NumberOfTrackingRegions->setAxisTitle("Number of Events", 2);
   }

   doTkCandPlots=conf->getParameter<bool>("doTrackCandHistos");
  //    if (doAllPlots) doTkCandPlots=true;
  
  if (doTkCandPlots){
    ibooker.setCurrentFolder(MEFolderName+"/TrackBuilding");
    
    edm::InputTag tcProducer     = conf->getParameter<edm::InputTag>("TCProducer");
    
    histname = "NumberOfTrackCandidates_"+ tcProducer.label() + "_"+ CategoryName;
    NumberOfTrackCandidates = ibooker.book1D(histname, histname, TCNoBin, TCNoMin, TCNoMax);
    NumberOfTrackCandidates->setAxisTitle("Number of Track Candidates per Event", 1);
    NumberOfTrackCandidates->setAxisTitle("Number of Event", 2);

    histname = "FractionOfCandOverSeeds_"+ tcProducer.label() + "_"+ CategoryName;
    FractionCandidatesOverSeeds = ibooker.book1D(histname, histname, 101, 0., 1.01);
    FractionCandidatesOverSeeds->setAxisTitle("Number of Track Candidates / Number of Seeds per Event", 1);
    FractionCandidatesOverSeeds->setAxisTitle("Number of Event", 2);

  }
  
  theTrackBuildingAnalyzer->initHisto(ibooker,*conf);
  
  
  if (doLumiAnalysis) {
    if ( NumberOfTracks_lumiFlag ) NumberOfTracks_lumiFlag -> setLumiFlag();
    theTrackAnalyzer->setLumiFlag();    
  }

  if(doAllSeedPlots || doSeedNumberPlot){
    if ( doSeedLumiAnalysis_ )
      NumberOfSeeds_lumiFlag->setLumiFlag();
  }
  
  if (doTrackerSpecific_ || doAllPlots) {
    
    ClusterLabels=  conf->getParameter<std::vector<std::string> >("ClusterLabels");
    
    std::vector<double> histoMin,histoMax;
    std::vector<int> histoBin; //these vectors are for max min and nbins in histograms 
    
    int    NClusStrBin = conf->getParameter<int>(   "NClusStrBin");
    double NClusStrMin = conf->getParameter<double>("NClusStrMin");
    double NClusStrMax = conf->getParameter<double>("NClusStrMax");
    
    int    NClusPxBin = conf->getParameter<int>(   "NClusPxBin");
    double NClusPxMin = conf->getParameter<double>("NClusPxMin");
    double NClusPxMax = conf->getParameter<double>("NClusPxMax");
    
    int    NTrk2DBin     = conf->getParameter<int>(   "NTrk2DBin");
    double NTrk2DMin     = conf->getParameter<double>("NTrk2DMin");
    double NTrk2DMax     = conf->getParameter<double>("NTrk2DMax");
    
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
      if(ClusterLabels[i]=="Tot")
	title = "# of Clusters in (Pixel+Strip) Detectors";
      NumberOfTrkVsClusters[i]->setAxisTitle(title, 1);
      NumberOfTrkVsClusters[i]->setAxisTitle("Number of Tracks", 2);
      NumberOfTrkVsClusters[i]->getTH1()->SetCanExtend(TH1::kXaxis);
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

    float lumi = -1.;
    edm::Handle<LumiScalersCollection> lumiScalers;
    iEvent.getByToken(lumiscalersToken_, lumiScalers);
    if ( lumiScalers.isValid() && !lumiScalers->empty() ) {
      LumiScalersCollection::const_iterator scalit = lumiScalers->begin();
      lumi = scalit->instantLumi();
    } else 
      lumi = -1.;

    if (doPlotsVsLUMI_ || doAllPlots) 
      NumberEventsOfVsLUMI->Fill(lumi);
    
    //  Analyse the tracks
    //  if the collection is empty, do not fill anything
    // ---------------------------------------------------------------------------------//

    size_t bx = iEvent.bunchCrossing();
    if ( doPlotsVsBX_ || doAllPlots )
      NumberEventsOfVsBX->Fill(bx);

    // get the track collection
    edm::Handle<edm::View<reco::Track> > trackHandle;
    iEvent.getByToken(trackToken_, trackHandle);

    int numberOfTracks_den = 0;
    edm::Handle<edm::View<reco::Track> > allTrackHandle;
    iEvent.getByToken(allTrackToken_,allTrackHandle);
    if (allTrackHandle.isValid()) {
      for ( edm::View<reco::Track>::const_iterator track = allTrackHandle->begin();
	    track != allTrackHandle->end(); ++track ) {

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
          || pv0->ndof() < pvNDOF_ || pv0->z() > 24.)  pv0 = nullptr;
    }


    if (trackHandle.isValid()) {
      
      int numberOfTracks = trackHandle->size();
      int numberOfTracks_num = 0;
      int numberOfTracks_pv0 = 0;

      const edm::View<reco::Track>& trackCollection = *trackHandle;
      // calculate the mean # rechits and layers
      int totalRecHits = 0, totalLayers = 0;

      theTrackAnalyzer->setNumberOfGoodVertices(iEvent);
      theTrackAnalyzer->setBX(iEvent);
      theTrackAnalyzer->setLumi(iEvent,iSetup);
      for ( edm::View<reco::Track>::const_iterator track = trackCollection.begin();
	    track != trackCollection.end(); ++track ) {

	if ( doPlotsVsBX_ || doAllPlots )
	  NumberOfRecHitsPerTrackVsBX->Fill(bx,track->numberOfValidHits());
	if ( numSelection_(*track) ) {
	  numberOfTracks_num++;
          if (pv0 && std::abs(track->dz(pv0->position()))<0.15) ++numberOfTracks_pv0;
        } 

	if ( doProfilesVsLS_ || doAllPlots)
	  NumberOfRecHitsPerTrackVsLS->Fill(static_cast<double>(iEvent.id().luminosityBlock()),track->numberOfValidHits());

	if (doPlotsVsLUMI_ || doAllPlots) 
	  NumberOfRecHitsPerTrackVsLUMI->Fill(lumi,track->numberOfValidHits());

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
	if ( doPlotsVsBX_ || doAllPlots )
	  NumberOfTracksVsBX   -> Fill(bx,numberOfTracks);
	if (doPlotsVsLUMI_ || doAllPlots) 
	  NumberOfTracksVsLUMI -> Fill(lumi,numberOfTracks);
	if (doFractionPlot_) {
	  FractionOfGoodTracks     -> Fill(frac);

	  if (doFractionPlot_) {
	    if ( doPlotsVsBX_ || doAllPlots )
	      GoodTracksFractionVsBX   -> Fill(bx,  frac);
	    if (doPlotsVsLUMI_ || doAllPlots) 
	      GoodTracksFractionVsLUMI -> Fill(lumi,frac);
	  }
	}
	if( numberOfTracks > 0 ) {
	  double meanRecHits = static_cast<double>(totalRecHits) / static_cast<double>(numberOfTracks);
	  double meanLayers  = static_cast<double>(totalLayers)  / static_cast<double>(numberOfTracks);
	  NumberOfMeanRecHitsPerTrack -> Fill(meanRecHits);
	  NumberOfMeanLayersPerTrack  -> Fill(meanLayers);
	}
      }
      
      if ( doProfilesVsLS_ || doAllPlots) {
	float nLS = static_cast<double>(iEvent.id().luminosityBlock());
	NumberEventsOfVsLS    ->Fill(nLS);
	NumberOfTracksVsLS    ->Fill(nLS,numberOfTracks);
	if (doFractionPlot_) 
	  GoodTracksFractionVsLS->Fill(nLS,frac);
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

          // get the seed collection
          edm::Handle<edm::View<TrajectorySeed> > seedHandle;
          iEvent.getByToken(seedToken_, seedHandle );
          const edm::View<TrajectorySeed>& seedCollection = *seedHandle;
          if (seedHandle.isValid() && !seedCollection.empty()) 
            FractionCandidatesOverSeeds->Fill(double(theTCCollection.size())/double(seedCollection.size()));

	  iSetup.get<TransientRecHitRecord>().get(builderName,theTTRHBuilder);
	  for( TrackCandidateCollection::const_iterator cand = theTCCollection.begin(); cand != theTCCollection.end(); ++cand) {
	    
	    theTrackBuildingAnalyzer->analyze(iEvent, iSetup, *cand, bs, theMF, theTTRHBuilder);
	  }
	} else {
	  edm::LogWarning("TrackingMonitor") << "No Track Candidates in the event.  Not filling associated histograms";
	}

	if(doMVAPlots) {
	  // Get MVA and quality mask collections
	  std::vector<const MVACollection *> mvaCollections;
	  std::vector<const QualityMaskCollection *> qualityMaskCollections;

	  edm::Handle<edm::View<reco::Track> > htracks;
	  iEvent.getByToken(mvaTrackToken_, htracks);

	  edm::Handle<MVACollection> hmva;
	  edm::Handle<QualityMaskCollection> hqual;
	  for(const auto& tokenTpl: mvaQualityTokens_) {
	    iEvent.getByToken(std::get<0>(tokenTpl), hmva);
	    iEvent.getByToken(std::get<1>(tokenTpl), hqual);

	    mvaCollections.push_back(hmva.product());
	    qualityMaskCollections.push_back(hqual.product());
	  }
	  theTrackBuildingAnalyzer->analyze(*htracks, mvaCollections, qualityMaskCollections);
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
      

      // plots for tracking regions
      if (doRegionPlots) {
        edm::Handle<edm::OwnVector<TrackingRegion> > hregions;
        iEvent.getByToken(regionToken_, hregions);
        NumberOfTrackingRegions->Fill(hregions->size());

        edm::Handle<reco::CandidateView> hcandidates;
        iEvent.getByToken(regionCandidateToken_, hcandidates);
        theTrackBuildingAnalyzer->analyze(*hcandidates);
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
	      if (pv->ndof() < pvNDOF_ || pv->z() > 24.)  continue;
	      totalNumGoodPV++;
	    }
	    
	    NumberEventsOfVsGoodPVtx       -> Fill( float(totalNumGoodPV) );
            NumberOfTracksVsGoodPVtx	   -> Fill( float(totalNumGoodPV), numberOfTracks	);
	    if (totalNumGoodPV>1) NumberOfTracksVsPUPVtx-> Fill( totalNumGoodPV-1, double(numberOfTracks-numberOfTracks_pv0)/double(totalNumGoodPV-1)      );
	    NumberOfPVtxVsGoodPVtx          -> Fill(float(totalNumGoodPV),pvHandle->size());

	    for ( edm::View<reco::Track>::const_iterator track = trackCollection.begin();
		  track != trackCollection.end(); ++track ) {

	      NumberOfRecHitsPerTrackVsGoodPVtx -> Fill(float(totalNumGoodPV), track->numberOfValidHits());
	    }

	    if ( doProfilesVsLS_ || doAllPlots)
	      NumberOfGoodPVtxVsLS->Fill(static_cast<double>(iEvent.id().luminosityBlock()),totalNumGoodPV);
	    if ( doPlotsVsBX_ || doAllPlots )
	      NumberOfGoodPVtxVsBX->Fill(bx,  float(totalNumGoodPV));

	    if (doFractionPlot_)
	      GoodTracksFractionVsGoodPVtx->Fill(float(totalNumGoodPV),frac);

	    if ( doPlotsVsLUMI_ || doAllPlots )	    
	      NumberOfGoodPVtxVsLUMI->Fill(lumi,float(totalNumGoodPV));
	  }
      
	  std::vector<int> NClus;
	  setNclus(iEvent,NClus);
	  for (uint  i=0; i< ClusterLabels.size(); i++){
	    if ( doPlotsVsLUMI_ || doAllPlots )	{
	      if (ClusterLabels[i]  =="Pix") NumberOfPixelClustersVsLUMI->Fill(lumi,NClus[i]);
	      if (ClusterLabels[i]=="Strip") NumberOfStripClustersVsLUMI->Fill(lumi,NClus[i]);
	    }
	    if (ClusterLabels[i]  =="Pix") NumberOfPixelClustersVsGoodPVtx->Fill(float(totalNumGoodPV),NClus[i]);
	    if (ClusterLabels[i]=="Strip") NumberOfStripClustersVsGoodPVtx->Fill(float(totalNumGoodPV),NClus[i]);
	  }
	
	if ( doPlotsVsBXlumi_ ) {
	  double bxlumi = theLumiDetails_->getValue(iEvent);
	  NumberOfTracksVsBXlumi       -> Fill( bxlumi, numberOfTracks      );
	}
	
	if ( doProfilesVsLS_ || doAllPlots ) if ( totalNumGoodPV != 0 ) NumberOfGoodPVtxWO0VsLS  -> Fill(static_cast<double>(iEvent.id().luminosityBlock()),float(totalNumGoodPV));
	if ( doPlotsVsBX_    || doAllPlots ) if ( totalNumGoodPV != 0 ) NumberOfGoodPVtxWO0VsBX  -> Fill(bx,  float(totalNumGoodPV));
	if ( doPlotsVsLUMI_  || doAllPlots ) if ( totalNumGoodPV != 0 ) NumberOfGoodPVtxWO0VsLUMI-> Fill(lumi,float(totalNumGoodPV));

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

    if     (ClusterLabels[i]=="Pix"  ) {arrayMin[i]=pmin; arrayMax[i]=pmax;      arrayBin[i]=pbin;}
    else if(ClusterLabels[i]=="Strip") {arrayMin[i]=smin; arrayMax[i]=smax;      arrayBin[i]=sbin;}
    else if(ClusterLabels[i]=="Tot"  ) {arrayMin[i]=smin; arrayMax[i]=smax+pmax; arrayBin[i]=sbin;}
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
    
    if     (ClusterLabels[i]=="Pix"  ) arrayNclus[i]=ncluster_pix ;
    else if(ClusterLabels[i]=="Strip") arrayNclus[i]=ncluster_strip;
    else if(ClusterLabels[i]=="Tot"  ) arrayNclus[i]=ncluster_pix+ncluster_strip;
    else {edm::LogWarning("TrackingMonitor") << "Cluster Label " << ClusterLabels[i] << " not defined using stri parametrs ";
      arrayNclus[i]=ncluster_strip ;}
  }
    
}
DEFINE_FWK_MODULE(TrackingMonitor);
