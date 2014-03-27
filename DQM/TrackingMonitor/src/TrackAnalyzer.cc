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
#include "DQM/TrackingMonitor/interface/TrackAnalyzer.h"
#include <string>
#include "TMath.h"

TrackAnalyzer::TrackAnalyzer(const edm::ParameterSet& iConfig) 
    : conf_( iConfig )
    , doTrackerSpecific_               ( conf_.getParameter<bool>("doTrackerSpecific") )
    , doAllPlots_                      ( conf_.getParameter<bool>("doAllPlots") )
    , doBSPlots_                       ( conf_.getParameter<bool>("doBeamSpotPlots") )
    , doPVPlots_                       ( conf_.getParameter<bool>("doPrimaryVertexPlots") )
    , doDCAPlots_                      ( conf_.getParameter<bool>("doDCAPlots") )
    , doGeneralPropertiesPlots_        ( conf_.getParameter<bool>("doGeneralPropertiesPlots") )
    , doMeasurementStatePlots_         ( conf_.getParameter<bool>("doMeasurementStatePlots") )
    , doHitPropertiesPlots_            ( conf_.getParameter<bool>("doHitPropertiesPlots") )
    , doRecHitVsPhiVsEtaPerTrack_      ( conf_.getParameter<bool>("doRecHitVsPhiVsEtaPerTrack") )
    , doLayersVsPhiVsEtaPerTrack_      ( conf_.getParameter<bool>("doLayersVsPhiVsEtaPerTrack") )
    , doRecHitsPerTrackProfile_        ( conf_.getParameter<bool>("doRecHitsPerTrackProfile") )
    , doThetaPlots_                    ( conf_.getParameter<bool>("doThetaPlots") )
    , doTrackPxPyPlots_                ( conf_.getParameter<bool>("doTrackPxPyPlots") )
    , doDCAwrtPVPlots_                 ( conf_.getParameter<bool>("doDCAwrtPVPlots") )
    , doDCAwrt000Plots_                ( conf_.getParameter<bool>("doDCAwrt000Plots") )
    , doLumiAnalysis_                  ( conf_.getParameter<bool>("doLumiAnalysis") )
    , doTestPlots_                     ( conf_.getParameter<bool>("doTestPlots") )
{
  initHistos();
  TopFolder_ = conf_.getParameter<std::string>("FolderName"); 

}

TrackAnalyzer::TrackAnalyzer(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC) 
  : TrackAnalyzer(iConfig)
{
  edm::InputTag bsSrc                 = conf_.getParameter<edm::InputTag>("beamSpot");
  edm::InputTag primaryVertexInputTag = conf_.getParameter<edm::InputTag>("primaryVertex");
  beamSpotToken_ = iC.consumes<reco::BeamSpot>(bsSrc);
  pvToken_       = iC.consumes<reco::VertexCollection>(primaryVertexInputTag);
}

void TrackAnalyzer::initHistos()
{
  Chi2 = NULL;
  Chi2Prob = NULL;
  Chi2ProbVsPhi = NULL;
  Chi2ProbVsEta = NULL;
  Chi2oNDF = NULL;
  Chi2oNDFVsEta = NULL;
  Chi2oNDFVsPhi = NULL;
  Chi2oNDFVsTheta = NULL;
  Chi2oNDFVsTheta = NULL;
  Chi2oNDFVsPhi = NULL;
  Chi2oNDFVsEta = NULL;
  	    
  NumberOfRecHitsPerTrack = NULL;
  NumberOfValidRecHitsPerTrack = NULL;
  NumberOfLostRecHitsPerTrack = NULL;

  NumberOfRecHitsPerTrackVsPhi = NULL;
  NumberOfRecHitsPerTrackVsTheta = NULL;
  NumberOfRecHitsPerTrackVsEta = NULL;

  NumberOfRecHitVsPhiVsEtaPerTrack = NULL;

  NumberOfValidRecHitsPerTrackVsPhi = NULL;
  NumberOfValidRecHitsPerTrackVsEta = NULL;

  NumberOfLayersPerTrack = NULL;
  NumberOfLayersVsPhiVsEtaPerTrack = NULL;

  DistanceOfClosestApproach = NULL;
  DistanceOfClosestApproachToBS = NULL;
  DistanceOfClosestApproachVsTheta = NULL;
  DistanceOfClosestApproachVsPhi = NULL;
  DistanceOfClosestApproachToBSVsPhi = NULL;
  DistanceOfClosestApproachVsEta = NULL;
  xPointOfClosestApproach = NULL;
  xPointOfClosestApproachVsZ0wrt000 = NULL;
  xPointOfClosestApproachVsZ0wrtBS = NULL;
  yPointOfClosestApproach = NULL;
  yPointOfClosestApproachVsZ0wrt000 = NULL;
  yPointOfClosestApproachVsZ0wrtBS = NULL;
  zPointOfClosestApproach = NULL;
  zPointOfClosestApproachVsPhi = NULL;
  algorithm = NULL;
    // TESTING
  TESTDistanceOfClosestApproachToBS = NULL;
  TESTDistanceOfClosestApproachToBSVsPhi = NULL;

// by Mia in order to deal w/ LS transitions
  Chi2oNDF_lumiFlag = NULL;
  NumberOfRecHitsPerTrack_lumiFlag = NULL;

}

TrackAnalyzer::~TrackAnalyzer() 
{ 
}

void TrackAnalyzer::beginJob(DQMStore * dqmStore_) 
{
  
}

void TrackAnalyzer::beginRun(DQMStore * dqmStore_) 
{

  bookHistosForHitProperties(dqmStore_);
  bookHistosForBeamSpot(dqmStore_);
  bookHistosForLScertification( dqmStore_);
  
  // book tracker specific related histograms
  // ---------------------------------------------------------------------------------//
  if(doTrackerSpecific_ || doAllPlots_) bookHistosForTrackerSpecific(dqmStore_);
    
  // book state related histograms
  // ---------------------------------------------------------------------------------//
  if (doMeasurementStatePlots_ || doAllPlots_) {

    std::string StateName = conf_.getParameter<std::string>("MeasurementState");
    
    if (StateName == "All") {
      bookHistosForState("OuterSurface", dqmStore_);
      bookHistosForState("InnerSurface", dqmStore_);
      bookHistosForState("ImpactPoint" , dqmStore_);
    } else if (
	       StateName != "OuterSurface" && 
	       StateName != "InnerSurface" && 
	       StateName != "ImpactPoint" &&
	       StateName != "default" 
	       ) {
      bookHistosForState("default", dqmStore_);

    } else {
      bookHistosForState(StateName, dqmStore_);
    }
    
  }
}

void TrackAnalyzer::bookHistosForHitProperties(DQMStore * dqmStore_) {
  
    // parameters from the configuration
    std::string QualName       = conf_.getParameter<std::string>("Quality");
    std::string AlgoName       = conf_.getParameter<std::string>("AlgoName");
    std::string MEBSFolderName = conf_.getParameter<std::string>("BSFolderName"); 

    // use the AlgoName and Quality Name 
    std::string CategoryName = QualName != "" ? AlgoName + "_" + QualName : AlgoName;

    // get binning from the configuration
    int    TKHitBin     = conf_.getParameter<int>(   "RecHitBin");
    double TKHitMin     = conf_.getParameter<double>("RecHitMin");
    double TKHitMax     = conf_.getParameter<double>("RecHitMax");

    int    TKLostBin    = conf_.getParameter<int>(   "RecLostBin");
    double TKLostMin    = conf_.getParameter<double>("RecLostMin");
    double TKLostMax    = conf_.getParameter<double>("RecLostMax");

    int    TKLayBin     = conf_.getParameter<int>(   "RecLayBin");
    double TKLayMin     = conf_.getParameter<double>("RecLayMin");
    double TKLayMax     = conf_.getParameter<double>("RecLayMax");

    int    PhiBin       = conf_.getParameter<int>(   "PhiBin");
    double PhiMin       = conf_.getParameter<double>("PhiMin");
    double PhiMax       = conf_.getParameter<double>("PhiMax");

    int    EtaBin       = conf_.getParameter<int>(   "EtaBin");
    double EtaMin       = conf_.getParameter<double>("EtaMin");
    double EtaMax       = conf_.getParameter<double>("EtaMax");

    int    VXBin        = conf_.getParameter<int>(   "VXBin");
    double VXMin        = conf_.getParameter<double>("VXMin");
    double VXMax        = conf_.getParameter<double>("VXMax");

    int    VYBin        = conf_.getParameter<int>(   "VYBin");
    double VYMin        = conf_.getParameter<double>("VYMin");
    double VYMax        = conf_.getParameter<double>("VYMax");

    int    VZBin        = conf_.getParameter<int>(   "VZBin");
    double VZMin        = conf_.getParameter<double>("VZMin");
    double VZMax        = conf_.getParameter<double>("VZMax");

    dqmStore_->setCurrentFolder(TopFolder_);

    // book the Hit Property histograms
    // ---------------------------------------------------------------------------------//

    TkParameterMEs tkmes;
    if ( doHitPropertiesPlots_ || doAllPlots_ ){

      dqmStore_->setCurrentFolder(TopFolder_+"/HitProperties");
      
      histname = "NumberOfRecHitsPerTrack_";
      NumberOfRecHitsPerTrack = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, TKHitBin, TKHitMin, TKHitMax);
      NumberOfRecHitsPerTrack->setAxisTitle("Number of all RecHits of each Track");
      NumberOfRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfValidRecHitsPerTrack_";
      NumberOfValidRecHitsPerTrack = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, TKHitBin, TKHitMin, TKHitMax);
      NumberOfValidRecHitsPerTrack->setAxisTitle("Number of valid RecHits for each Track");
      NumberOfValidRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfLostRecHitsPerTrack_";
      NumberOfLostRecHitsPerTrack = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, TKLostBin, TKLostMin, TKLostMax);
      NumberOfLostRecHitsPerTrack->setAxisTitle("Number of lost RecHits for each Track");
      NumberOfLostRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfLayersPerTrack_";
      NumberOfLayersPerTrack = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, TKLayBin, TKLayMin, TKLayMax);
      NumberOfLayersPerTrack->setAxisTitle("Number of Layers of each Track", 1);
      NumberOfLayersPerTrack->setAxisTitle("Number of Tracks", 2);
      

      if ( doRecHitVsPhiVsEtaPerTrack_ || doAllPlots_ ){
	
	histname = "NumberOfRecHitVsPhiVsEtaPerTrack_";
	NumberOfRecHitVsPhiVsEtaPerTrack = dqmStore_->bookProfile2D(histname+CategoryName, histname+CategoryName, 
								    EtaBin, EtaMin, EtaMax, PhiBin, PhiMin, PhiMax, 0, 40., "");
	NumberOfRecHitVsPhiVsEtaPerTrack->setAxisTitle("Track #eta ", 1);
	NumberOfRecHitVsPhiVsEtaPerTrack->setAxisTitle("Track #phi ", 2);
      }

      if ( doLayersVsPhiVsEtaPerTrack_ || doAllPlots_ ){
	
	histname = "NumberOfLayersVsPhiVsEtaPerTrack_";
	NumberOfLayersVsPhiVsEtaPerTrack = dqmStore_->bookProfile2D(histname+CategoryName, histname+CategoryName, 
								    EtaBin, EtaMin, EtaMax, PhiBin, PhiMin, PhiMax, 0, 40., "");
	NumberOfLayersVsPhiVsEtaPerTrack->setAxisTitle("Track #eta ", 1);
	NumberOfLayersVsPhiVsEtaPerTrack->setAxisTitle("Track #phi ", 2);
      }
    }

    // book the General Property histograms
    // ---------------------------------------------------------------------------------//
    
    if (doGeneralPropertiesPlots_ || doAllPlots_){
      
      int    Chi2Bin      = conf_.getParameter<int>(   "Chi2Bin");
      double Chi2Min      = conf_.getParameter<double>("Chi2Min");
      double Chi2Max      = conf_.getParameter<double>("Chi2Max");
      
      int    Chi2NDFBin   = conf_.getParameter<int>(   "Chi2NDFBin");
      double Chi2NDFMin   = conf_.getParameter<double>("Chi2NDFMin");
      double Chi2NDFMax   = conf_.getParameter<double>("Chi2NDFMax");
      
      int    Chi2ProbBin  = conf_.getParameter<int>(   "Chi2ProbBin");
      double Chi2ProbMin  = conf_.getParameter<double>("Chi2ProbMin");
      double Chi2ProbMax  = conf_.getParameter<double>("Chi2ProbMax");
    

      dqmStore_->setCurrentFolder(TopFolder_+"/GeneralProperties");

      histname = "Chi2_";
      Chi2 = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, Chi2Bin, Chi2Min, Chi2Max);
      Chi2->setAxisTitle("Track #chi^{2}"  ,1);
      Chi2->setAxisTitle("Number of Tracks",2);

      histname = "Chi2Prob_";
      Chi2Prob = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, Chi2ProbBin, Chi2ProbMin, Chi2ProbMax);
      Chi2Prob->setAxisTitle("Track #chi^{2} probability",1);
      Chi2Prob->setAxisTitle("Number of Tracks"        ,2);
      
      histname = "Chi2oNDF_";
      Chi2oNDF = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, Chi2NDFBin, Chi2NDFMin, Chi2NDFMax);
      Chi2oNDF->setAxisTitle("Track #chi^{2}/ndf",1);
      Chi2oNDF->setAxisTitle("Number of Tracks"  ,2);
      
      if (doDCAPlots_) {
	histname = "xPointOfClosestApproach_";
	xPointOfClosestApproach = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, VXBin, VXMin, VXMax);
	xPointOfClosestApproach->setAxisTitle("x component of Track PCA to beam line (cm)",1);
	xPointOfClosestApproach->setAxisTitle("Number of Tracks",2);
	
	histname = "yPointOfClosestApproach_";
	yPointOfClosestApproach = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, VYBin, VYMin, VYMax);
	yPointOfClosestApproach->setAxisTitle("y component of Track PCA to beam line (cm)",1);
	yPointOfClosestApproach->setAxisTitle("Number of Tracks",2);
	
	histname = "zPointOfClosestApproach_";
	zPointOfClosestApproach = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, VZBin, VZMin, VZMax);
	zPointOfClosestApproach->setAxisTitle("z component of Track PCA to beam line (cm)",1);
	zPointOfClosestApproach->setAxisTitle("Number of Tracks",2);
	
	histname = "xPointOfClosestApproachToPV_";
	xPointOfClosestApproachToPV = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, VXBin, VXMin, VXMax);
	xPointOfClosestApproachToPV->setAxisTitle("x component of Track PCA to pv (cm)",1);
	xPointOfClosestApproachToPV->setAxisTitle("Number of Tracks",2);
	
	histname = "yPointOfClosestApproachToPV_";
	yPointOfClosestApproachToPV = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, VYBin, VYMin, VYMax);
	yPointOfClosestApproachToPV->setAxisTitle("y component of Track PCA to pv line (cm)",1);
	yPointOfClosestApproachToPV->setAxisTitle("Number of Tracks",2);
	
	histname = "zPointOfClosestApproachToPV_";
	zPointOfClosestApproachToPV = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, VZBin, VZMin, VZMax);
	zPointOfClosestApproachToPV->setAxisTitle("z component of Track PCA to pv line (cm)",1);
	zPointOfClosestApproachToPV->setAxisTitle("Number of Tracks",2);
      }

      // See DataFormats/TrackReco/interface/TrackBase.h for track algorithm enum definition
      // http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/DataFormats/TrackReco/interface/TrackBase.h?view=log
      histname = "algorithm_";
      algorithm = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, 32, 0., 32.);
      algorithm->setAxisTitle("Tracking algorithm",1);
      algorithm->setAxisTitle("Number of Tracks",2);
      
    }

}

void TrackAnalyzer::bookHistosForLScertification(DQMStore * dqmStore_) {

    // parameters from the configuration
    std::string QualName       = conf_.getParameter<std::string>("Quality");
    std::string AlgoName       = conf_.getParameter<std::string>("AlgoName");

    // use the AlgoName and Quality Name 
    std::string CategoryName = QualName != "" ? AlgoName + "_" + QualName : AlgoName;


    // book LS analysis related histograms
    // -----------------------------------
    if ( doLumiAnalysis_ ) {

      // get binning from the configuration
      int    TKHitBin     = conf_.getParameter<int>(   "RecHitBin");
      double TKHitMin     = conf_.getParameter<double>("RecHitMin");
      double TKHitMax     = conf_.getParameter<double>("RecHitMax");

      int    Chi2NDFBin   = conf_.getParameter<int>(   "Chi2NDFBin");
      double Chi2NDFMin   = conf_.getParameter<double>("Chi2NDFMin");
      double Chi2NDFMax   = conf_.getParameter<double>("Chi2NDFMax");

      // add by Mia in order to deal w/ LS transitions  
      dqmStore_->setCurrentFolder(TopFolder_+"/LSanalysis");

      histname = "NumberOfRecHitsPerTrack_lumiFlag_";
      NumberOfRecHitsPerTrack_lumiFlag = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, TKHitBin, TKHitMin, TKHitMax);
      NumberOfRecHitsPerTrack_lumiFlag->setAxisTitle("Number of all RecHits of each Track");
      NumberOfRecHitsPerTrack_lumiFlag->setAxisTitle("Number of Tracks", 2);

      histname = "Chi2oNDF_lumiFlag_";
      Chi2oNDF_lumiFlag = dqmStore_->book1D(histname+CategoryName, histname+CategoryName, Chi2NDFBin, Chi2NDFMin, Chi2NDFMax);
      Chi2oNDF_lumiFlag->setAxisTitle("Track #chi^{2}/ndf",1);
      Chi2oNDF_lumiFlag->setAxisTitle("Number of Tracks"  ,2);

    }
}

void TrackAnalyzer::bookHistosForBeamSpot(DQMStore * dqmStore_) {

    // parameters from the configuration
    std::string QualName       = conf_.getParameter<std::string>("Quality");
    std::string AlgoName       = conf_.getParameter<std::string>("AlgoName");

    // use the AlgoName and Quality Name 
    std::string CategoryName = QualName != "" ? AlgoName + "_" + QualName : AlgoName;

    // book the Beam Spot related histograms
    // ---------------------------------------------------------------------------------//
    
    if(doDCAPlots_ || doBSPlots_ || doAllPlots_) {
	
      int    DxyBin       = conf_.getParameter<int>(   "DxyBin");
      double DxyMin       = conf_.getParameter<double>("DxyMin");
      double DxyMax       = conf_.getParameter<double>("DxyMax");
      
      int    PhiBin     = conf_.getParameter<int>(   "PhiBin");
      double PhiMin     = conf_.getParameter<double>("PhiMin");
      double PhiMax     = conf_.getParameter<double>("PhiMax");
      
      int    X0Bin        = conf_.getParameter<int>(   "X0Bin");
      double X0Min        = conf_.getParameter<double>("X0Min");
      double X0Max        = conf_.getParameter<double>("X0Max");
      
      int    Y0Bin        = conf_.getParameter<int>(   "Y0Bin");
      double Y0Min        = conf_.getParameter<double>("Y0Min");
      double Y0Max        = conf_.getParameter<double>("Y0Max");
      
      int    Z0Bin        = conf_.getParameter<int>(   "Z0Bin");
      double Z0Min        = conf_.getParameter<double>("Z0Min");
      double Z0Max        = conf_.getParameter<double>("Z0Max");
      
      int    VZBinProf    = conf_.getParameter<int>(   "VZBinProf");
      double VZMinProf    = conf_.getParameter<double>("VZMinProf");
      double VZMaxProf    = conf_.getParameter<double>("VZMaxProf");
      
      
      dqmStore_->setCurrentFolder(TopFolder_+"/GeneralProperties");
      
      histname = "DistanceOfClosestApproachToBS_";
      DistanceOfClosestApproachToBS = dqmStore_->book1D(histname+CategoryName,histname+CategoryName,DxyBin,DxyMin,DxyMax);
      DistanceOfClosestApproachToBS->setAxisTitle("Track d_{xy} wrt beam spot (cm)",1);
      DistanceOfClosestApproachToBS->setAxisTitle("Number of Tracks",2);
      
      histname = "DistanceOfClosestApproachToBSVsPhi_";
      DistanceOfClosestApproachToBSVsPhi = dqmStore_->bookProfile(histname+CategoryName,histname+CategoryName, PhiBin, PhiMin, PhiMax, DxyBin, DxyMin, DxyMax,"");
      DistanceOfClosestApproachToBSVsPhi->getTH1()->SetBit(TH1::kCanRebin);
      DistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track #phi",1);
      DistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track d_{xy} wrt beam spot (cm)",2);
      
      histname = "xPointOfClosestApproachVsZ0wrt000_";
      xPointOfClosestApproachVsZ0wrt000 = dqmStore_->bookProfile(histname+CategoryName, histname+CategoryName, Z0Bin, Z0Min, Z0Max, X0Bin, X0Min, X0Max,"");
      xPointOfClosestApproachVsZ0wrt000->setAxisTitle("d_{z} (cm)",1);
      xPointOfClosestApproachVsZ0wrt000->setAxisTitle("x component of Track PCA to beam line (cm)",2);
      
      histname = "yPointOfClosestApproachVsZ0wrt000_";
      yPointOfClosestApproachVsZ0wrt000 = dqmStore_->bookProfile(histname+CategoryName, histname+CategoryName, Z0Bin, Z0Min, Z0Max, Y0Bin, Y0Min, Y0Max,"");
      yPointOfClosestApproachVsZ0wrt000->setAxisTitle("d_{z} (cm)",1);
      yPointOfClosestApproachVsZ0wrt000->setAxisTitle("y component of Track PCA to beam line (cm)",2);
      
      histname = "xPointOfClosestApproachVsZ0wrtBS_";
      xPointOfClosestApproachVsZ0wrtBS = dqmStore_->bookProfile(histname+CategoryName, histname+CategoryName, Z0Bin, Z0Min, Z0Max, X0Bin, X0Min, X0Max,"");
      xPointOfClosestApproachVsZ0wrtBS->setAxisTitle("d_{z} w.r.t. Beam Spot  (cm)",1);
      xPointOfClosestApproachVsZ0wrtBS->setAxisTitle("x component of Track PCA to BS (cm)",2);
      
      histname = "yPointOfClosestApproachVsZ0wrtBS_";
      yPointOfClosestApproachVsZ0wrtBS = dqmStore_->bookProfile(histname+CategoryName, histname+CategoryName, Z0Bin, Z0Min, Z0Max, Y0Bin, Y0Min, Y0Max,"");
      yPointOfClosestApproachVsZ0wrtBS->setAxisTitle("d_{z} w.r.t. Beam Spot (cm)",1);
      yPointOfClosestApproachVsZ0wrtBS->setAxisTitle("y component of Track PCA to BS (cm)",2);
      
      histname = "zPointOfClosestApproachVsPhi_";
      zPointOfClosestApproachVsPhi = dqmStore_->bookProfile(histname+CategoryName, histname+CategoryName, PhiBin, PhiMin, PhiMax, VZBinProf, VZMinProf, VZMaxProf, "");
      zPointOfClosestApproachVsPhi->setAxisTitle("Track #phi",1);
      zPointOfClosestApproachVsPhi->setAxisTitle("y component of Track PCA to beam line (cm)",2);
    }
    
    if(doDCAPlots_ || doPVPlots_ || doAllPlots_) {
      
      int    DxyBin       = conf_.getParameter<int>(   "DxyBin");
      double DxyMin       = conf_.getParameter<double>("DxyMin");
      double DxyMax       = conf_.getParameter<double>("DxyMax");
      
      int    PhiBin     = conf_.getParameter<int>(   "PhiBin");
      double PhiMin     = conf_.getParameter<double>("PhiMin");
      double PhiMax     = conf_.getParameter<double>("PhiMax");
      
      int    X0Bin        = conf_.getParameter<int>(   "X0Bin");
      double X0Min        = conf_.getParameter<double>("X0Min");
      double X0Max        = conf_.getParameter<double>("X0Max");
      
      int    Y0Bin        = conf_.getParameter<int>(   "Y0Bin");
      double Y0Min        = conf_.getParameter<double>("Y0Min");
      double Y0Max        = conf_.getParameter<double>("Y0Max");
      
      int    Z0Bin        = conf_.getParameter<int>(   "Z0Bin");
      double Z0Min        = conf_.getParameter<double>("Z0Min");
      double Z0Max        = conf_.getParameter<double>("Z0Max");
      
      dqmStore_->setCurrentFolder(TopFolder_+"/GeneralProperties");
      
      histname = "DistanceOfClosestApproachToPV_";
      DistanceOfClosestApproachToPV = dqmStore_->book1D(histname+CategoryName,histname+CategoryName,DxyBin,DxyMin,DxyMax);
      DistanceOfClosestApproachToPV->setAxisTitle("Track d_{xy} wrt beam spot (cm)",1);
      DistanceOfClosestApproachToPV->setAxisTitle("Number of Tracks",2);
      
      histname = "DistanceOfClosestApproachToPVVsPhi_";
      DistanceOfClosestApproachToPVVsPhi = dqmStore_->bookProfile(histname+CategoryName,histname+CategoryName, PhiBin, PhiMin, PhiMax, DxyBin, DxyMin, DxyMax,"");
      DistanceOfClosestApproachToPVVsPhi->getTH1()->SetBit(TH1::kCanRebin);
      DistanceOfClosestApproachToPVVsPhi->setAxisTitle("Track #phi",1);
      DistanceOfClosestApproachToPVVsPhi->setAxisTitle("Track d_{xy} wrt beam spot (cm)",2);
      
      histname = "xPointOfClosestApproachVsZ0wrtPV_";
      xPointOfClosestApproachVsZ0wrtPV = dqmStore_->bookProfile(histname+CategoryName, histname+CategoryName, Z0Bin, Z0Min, Z0Max, X0Bin, X0Min, X0Max,"");
      xPointOfClosestApproachVsZ0wrtPV->setAxisTitle("d_{z} w.r.t. Beam Spot  (cm)",1);
      xPointOfClosestApproachVsZ0wrtPV->setAxisTitle("x component of Track PCA to PV (cm)",2);
      
      histname = "yPointOfClosestApproachVsZ0wrtPV_";
      yPointOfClosestApproachVsZ0wrtPV = dqmStore_->bookProfile(histname+CategoryName, histname+CategoryName, Z0Bin, Z0Min, Z0Max, Y0Bin, Y0Min, Y0Max,"");
      yPointOfClosestApproachVsZ0wrtPV->setAxisTitle("d_{z} w.r.t. Beam Spot (cm)",1);
      yPointOfClosestApproachVsZ0wrtPV->setAxisTitle("y component of Track PCA to PV (cm)",2);
      
    }

    if (doBSPlots_ || doAllPlots_) {
      if (doTestPlots_) {
	
	int    DxyBin       = conf_.getParameter<int>(   "DxyBin");
	double DxyMin       = conf_.getParameter<double>("DxyMin");
	double DxyMax       = conf_.getParameter<double>("DxyMax");
      
	int    PhiBin     = conf_.getParameter<int>(   "PhiBin");
	double PhiMin     = conf_.getParameter<double>("PhiMin");
	double PhiMax     = conf_.getParameter<double>("PhiMax");

	histname = "TESTDistanceOfClosestApproachToBS_";
	TESTDistanceOfClosestApproachToBS = dqmStore_->book1D(histname+CategoryName,histname+CategoryName,DxyBin,DxyMin,DxyMax);
	TESTDistanceOfClosestApproachToBS->setAxisTitle("Track d_{xy} wrt beam spot (cm)",1);
	TESTDistanceOfClosestApproachToBS->setAxisTitle("Number of Tracks",2);
	
	histname = "TESTDistanceOfClosestApproachToBSVsPhi_";
	TESTDistanceOfClosestApproachToBSVsPhi = dqmStore_->bookProfile(histname+CategoryName,histname+CategoryName, PhiBin, PhiMin, PhiMax, DxyBin, DxyMin, DxyMax,"");
	TESTDistanceOfClosestApproachToBSVsPhi->getTH1()->SetBit(TH1::kCanRebin);
	TESTDistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track #phi",1);
	TESTDistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track d_{xy} wrt beam spot (cm)",2);
	
      }
      
    }
    
    // book the Profile plots for DCA related histograms
    // ---------------------------------------------------------------------------------//
    if(doDCAPlots_ || doAllPlots_) {

      if (doDCAwrt000Plots_) {

	int    EtaBin     = conf_.getParameter<int>(   "EtaBin");
	double EtaMin     = conf_.getParameter<double>("EtaMin");
	double EtaMax     = conf_.getParameter<double>("EtaMax");
	
	int    PhiBin     = conf_.getParameter<int>(   "PhiBin");
	double PhiMin     = conf_.getParameter<double>("PhiMin");
	double PhiMax     = conf_.getParameter<double>("PhiMax");

	int    DxyBin       = conf_.getParameter<int>(   "DxyBin");
	double DxyMin       = conf_.getParameter<double>("DxyMin");
	double DxyMax       = conf_.getParameter<double>("DxyMax");
      
	if (doThetaPlots_) {
	  int    ThetaBin   = conf_.getParameter<int>(   "ThetaBin");
	  double ThetaMin   = conf_.getParameter<double>("ThetaMin");
	  double ThetaMax   = conf_.getParameter<double>("ThetaMax");
	  
	  dqmStore_->setCurrentFolder(TopFolder_+"/GeneralProperties");
	  histname = "DistanceOfClosestApproachVsTheta_";
	  DistanceOfClosestApproachVsTheta = dqmStore_->bookProfile(histname+CategoryName,histname+CategoryName, ThetaBin, ThetaMin, ThetaMax, DxyMin,DxyMax,"");
	  DistanceOfClosestApproachVsTheta->setAxisTitle("Track #theta",1);
	  DistanceOfClosestApproachVsTheta->setAxisTitle("Track d_{xy} wrt (0,0,0) (cm)",2);
	}
	
	histname = "DistanceOfClosestApproachVsEta_";
	DistanceOfClosestApproachVsEta = dqmStore_->bookProfile(histname+CategoryName,histname+CategoryName, EtaBin, EtaMin, EtaMax, DxyMin, DxyMax,"");
	DistanceOfClosestApproachVsEta->setAxisTitle("Track #eta",1);
	DistanceOfClosestApproachVsEta->setAxisTitle("Track d_{xy} wrt (0,0,0) (cm)",2);
	// temporary patch in order to put back those MEs in Muon Workspace
	
	histname = "DistanceOfClosestApproach_";
	DistanceOfClosestApproach = dqmStore_->book1D(histname+CategoryName,histname+CategoryName,DxyBin,DxyMin,DxyMax);
	DistanceOfClosestApproach->setAxisTitle("Track d_{xy} wrt (0,0,0) (cm)",1);
	DistanceOfClosestApproach->setAxisTitle("Number of Tracks",2);
	
	histname = "DistanceOfClosestApproachVsPhi_";
	DistanceOfClosestApproachVsPhi = dqmStore_->bookProfile(histname+CategoryName,histname+CategoryName, PhiBin, PhiMin, PhiMax, DxyMin,DxyMax,"");
	DistanceOfClosestApproachVsPhi->getTH1()->SetBit(TH1::kCanRebin);
	DistanceOfClosestApproachVsPhi->setAxisTitle("Track #phi",1);
	DistanceOfClosestApproachVsPhi->setAxisTitle("Track d_{xy} wrt (0,0,0) (cm)",2);
      }
    }

}

// -- Analyse
// ---------------------------------------------------------------------------------//
void TrackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Track& track)
{
  double phi   = track.phi();
  double eta   = track.eta();

  int nRecHits      = track.hitPattern().numberOfHits();
  int nValidRecHits = track.numberOfValidHits();
  int nLostRecHits  = track.numberOfLostHits();

  double chi2     = track.chi2();
  double chi2prob = TMath::Prob(track.chi2(),(int)track.ndof());
  double chi2oNDF = track.normalizedChi2();
  
  if ( doHitPropertiesPlots_ || doAllPlots_ ){
    // rec hits
    NumberOfRecHitsPerTrack     -> Fill(nRecHits);
    NumberOfValidRecHitsPerTrack-> Fill(nValidRecHits);
    NumberOfLostRecHitsPerTrack -> Fill(nLostRecHits);

    // 2D plots    
    if ( doRecHitVsPhiVsEtaPerTrack_ || doAllPlots_ )
      NumberOfRecHitVsPhiVsEtaPerTrack->Fill(eta,phi,nRecHits);
    
    int nLayers = track.hitPattern().trackerLayersWithMeasurement();    
    // layers
    NumberOfLayersPerTrack->Fill(nLayers);

    // 2D plots    
    if ( doLayersVsPhiVsEtaPerTrack_ || doAllPlots_ )
      NumberOfLayersVsPhiVsEtaPerTrack->Fill(eta,phi,nLayers);
  }
  
  if (doGeneralPropertiesPlots_ || doAllPlots_){
    // fitting
    Chi2     -> Fill(chi2);
    Chi2Prob -> Fill(chi2prob);
    Chi2oNDF -> Fill(chi2oNDF);

    // DCA
    // temporary patch in order to put back those MEs in Muon Workspace 
    if (doDCAPlots_) {
      if (doDCAwrt000Plots_) {
	DistanceOfClosestApproach->Fill(track.dxy());
	DistanceOfClosestApproachVsPhi->Fill(phi, track.dxy());
      }

      // PCA
      xPointOfClosestApproach->Fill(track.referencePoint().x());
      yPointOfClosestApproach->Fill(track.referencePoint().y());
      zPointOfClosestApproach->Fill(track.referencePoint().z());
    }

    // algorithm
    algorithm->Fill(static_cast<double>(track.algo()));
  }

  if ( doLumiAnalysis_ ) {
    NumberOfRecHitsPerTrack_lumiFlag -> Fill(nRecHits);
    Chi2oNDF_lumiFlag                -> Fill(chi2oNDF);
  }

  if(doDCAPlots_ || doBSPlots_ || doAllPlots_) {
    
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByToken(beamSpotToken_,recoBeamSpotHandle);
    reco::BeamSpot bs = *recoBeamSpotHandle;      

    DistanceOfClosestApproachToBS      -> Fill(track.dxy(bs.position()));
    DistanceOfClosestApproachToBSVsPhi -> Fill(track.phi(), track.dxy(bs.position()));
    zPointOfClosestApproachVsPhi       -> Fill(track.phi(), track.vz());
    xPointOfClosestApproachVsZ0wrt000  -> Fill(track.dz(),  track.vx());
    yPointOfClosestApproachVsZ0wrt000  -> Fill(track.dz(),  track.vy());
    xPointOfClosestApproachVsZ0wrtBS   -> Fill(track.dz(bs.position()),(track.vx()-bs.position(track.vz()).x()));
    yPointOfClosestApproachVsZ0wrtBS   -> Fill(track.dz(bs.position()),(track.vy()-bs.position(track.vz()).y()));
    if (doTestPlots_) {
      TESTDistanceOfClosestApproachToBS      -> Fill(track.dxy(bs.position(track.vz())));
      TESTDistanceOfClosestApproachToBSVsPhi -> Fill(track.phi(), track.dxy(bs.position(track.vz())));
    }
  }
  
  if(doDCAPlots_ || doPVPlots_ || doAllPlots_) {
    edm::Handle<reco::VertexCollection> recoPrimaryVerticesHandle;
    iEvent.getByToken(pvToken_,recoPrimaryVerticesHandle);
    if (recoPrimaryVerticesHandle->size() > 0) {
      reco::Vertex pv = recoPrimaryVerticesHandle->at(0);      
    
      DistanceOfClosestApproachToPV      -> Fill(track.dxy(pv.position()));
      DistanceOfClosestApproachToPVVsPhi -> Fill(track.phi(), track.dxy(pv.position()));
      xPointOfClosestApproachVsZ0wrtPV   -> Fill(track.dz(pv.position()),(track.vx()-pv.position().x()));
      yPointOfClosestApproachVsZ0wrtPV   -> Fill(track.dz(pv.position()),(track.vy()-pv.position().y()));
    }
  }

  if(doDCAPlots_ || doAllPlots_) {
    if (doDCAwrt000Plots_) {
      if (doThetaPlots_) {
	DistanceOfClosestApproachVsTheta->Fill(track.theta(), track.d0());
      }
      DistanceOfClosestApproachVsEta->Fill(track.eta(), track.d0());
    }  
    
  }

  //Tracker Specific Histograms
  if(doTrackerSpecific_ || doAllPlots_) {
    fillHistosForTrackerSpecific(track);
  }

  if (doMeasurementStatePlots_ || doAllPlots_){
    std::string StateName = conf_.getParameter<std::string>("MeasurementState");

    if (StateName == "All") {
      fillHistosForState(iSetup, track, std::string("OuterSurface"));
      fillHistosForState(iSetup, track, std::string("InnerSurface"));
      fillHistosForState(iSetup, track, std::string("ImpactPoint"));
    } else if ( 
	       StateName != "OuterSurface" && 
	       StateName != "InnerSurface" && 
	       StateName != "ImpactPoint" &&
	       StateName != "default" 
	       ) {
      fillHistosForState(iSetup, track, std::string("default"));
    } else {
      fillHistosForState(iSetup, track, StateName);
    }
  }
  
  if ( doAllPlots_ ) {
  }

}

// book histograms at differnt measurement points
// ---------------------------------------------------------------------------------//
void TrackAnalyzer::bookHistosForState(std::string sname, DQMStore * dqmStore_) 
{

    // parameters from the configuration
    std::string QualName       = conf_.getParameter<std::string>("Quality");
    std::string AlgoName       = conf_.getParameter<std::string>("AlgoName");

    // use the AlgoName and Quality Name 
    std::string CategoryName = QualName != "" ? AlgoName + "_" + QualName : AlgoName;

    // get binning from the configuration
    double Chi2NDFMin = conf_.getParameter<double>("Chi2NDFMin");
    double Chi2NDFMax = conf_.getParameter<double>("Chi2NDFMax");

    int    RecHitBin   = conf_.getParameter<int>(   "RecHitBin");
    double RecHitMin   = conf_.getParameter<double>("RecHitMin");
    double RecHitMax   = conf_.getParameter<double>("RecHitMax");

    int    RecLayBin   = conf_.getParameter<int>(   "RecHitBin");
    double RecLayMin   = conf_.getParameter<double>("RecHitMin");
    double RecLayMax   = conf_.getParameter<double>("RecHitMax");


    int    PhiBin     = conf_.getParameter<int>(   "PhiBin");
    double PhiMin     = conf_.getParameter<double>("PhiMin");
    double PhiMax     = conf_.getParameter<double>("PhiMax");

    int    EtaBin     = conf_.getParameter<int>(   "EtaBin");
    double EtaMin     = conf_.getParameter<double>("EtaMin");
    double EtaMax     = conf_.getParameter<double>("EtaMax");

    int    ThetaBin   = conf_.getParameter<int>(   "ThetaBin");
    double ThetaMin   = conf_.getParameter<double>("ThetaMin");
    double ThetaMax   = conf_.getParameter<double>("ThetaMax");

    int    TrackQBin  = conf_.getParameter<int>(   "TrackQBin");
    double TrackQMin  = conf_.getParameter<double>("TrackQMin");
    double TrackQMax  = conf_.getParameter<double>("TrackQMax");

    int    TrackPtBin = conf_.getParameter<int>(   "TrackPtBin");
    double TrackPtMin = conf_.getParameter<double>("TrackPtMin");
    double TrackPtMax = conf_.getParameter<double>("TrackPtMax");

    int    TrackPBin  = conf_.getParameter<int>(   "TrackPBin");
    double TrackPMin  = conf_.getParameter<double>("TrackPMin");
    double TrackPMax  = conf_.getParameter<double>("TrackPMax");

    int    TrackPxBin = conf_.getParameter<int>(   "TrackPxBin");
    double TrackPxMin = conf_.getParameter<double>("TrackPxMin");
    double TrackPxMax = conf_.getParameter<double>("TrackPxMax");

    int    TrackPyBin = conf_.getParameter<int>(   "TrackPyBin");
    double TrackPyMin = conf_.getParameter<double>("TrackPyMin");
    double TrackPyMax = conf_.getParameter<double>("TrackPyMax");

    int    TrackPzBin = conf_.getParameter<int>(   "TrackPzBin");
    double TrackPzMin = conf_.getParameter<double>("TrackPzMin");
    double TrackPzMax = conf_.getParameter<double>("TrackPzMax");

    int    ptErrBin   = conf_.getParameter<int>(   "ptErrBin");
    double ptErrMin   = conf_.getParameter<double>("ptErrMin");
    double ptErrMax   = conf_.getParameter<double>("ptErrMax");

    int    pxErrBin   = conf_.getParameter<int>(   "pxErrBin");
    double pxErrMin   = conf_.getParameter<double>("pxErrMin");
    double pxErrMax   = conf_.getParameter<double>("pxErrMax");

    int    pyErrBin   = conf_.getParameter<int>(   "pyErrBin");
    double pyErrMin   = conf_.getParameter<double>("pyErrMin");
    double pyErrMax   = conf_.getParameter<double>("pyErrMax");

    int    pzErrBin   = conf_.getParameter<int>(   "pzErrBin");
    double pzErrMin   = conf_.getParameter<double>("pzErrMin");
    double pzErrMax   = conf_.getParameter<double>("pzErrMax");

    int    pErrBin    = conf_.getParameter<int>(   "pErrBin");
    double pErrMin    = conf_.getParameter<double>("pErrMin");
    double pErrMax    = conf_.getParameter<double>("pErrMax");

    int    phiErrBin  = conf_.getParameter<int>(   "phiErrBin");
    double phiErrMin  = conf_.getParameter<double>("phiErrMin");
    double phiErrMax  = conf_.getParameter<double>("phiErrMax");

    int    etaErrBin  = conf_.getParameter<int>(   "etaErrBin");
    double etaErrMin  = conf_.getParameter<double>("etaErrMin");
    double etaErrMax  = conf_.getParameter<double>("etaErrMax");


    double Chi2ProbMin  = conf_.getParameter<double>("Chi2ProbMin");
    double Chi2ProbMax  = conf_.getParameter<double>("Chi2ProbMax");

    dqmStore_->setCurrentFolder(TopFolder_);

    TkParameterMEs tkmes;

    std::string histTag = (sname == "default") ? CategoryName : sname + "_" + CategoryName;

    if(doAllPlots_) {

      // general properties
      dqmStore_->setCurrentFolder(TopFolder_+"/GeneralProperties");
      
      if (doThetaPlots_) {
	histname = "Chi2oNDFVsTheta_" + histTag;
	tkmes.Chi2oNDFVsTheta = dqmStore_->bookProfile(histname, histname, ThetaBin, ThetaMin, ThetaMax, Chi2NDFMin, Chi2NDFMax,"");
	tkmes.Chi2oNDFVsTheta->setAxisTitle("Track #theta",1);
	tkmes.Chi2oNDFVsTheta->setAxisTitle("Track #chi^{2}/ndf",2);
      }
      histname = "Chi2oNDFVsPhi_" + histTag;
      tkmes.Chi2oNDFVsPhi   = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, Chi2NDFMin, Chi2NDFMax,"");
      tkmes.Chi2oNDFVsPhi->setAxisTitle("Track #phi",1);
      tkmes.Chi2oNDFVsPhi->setAxisTitle("Track #chi^{2}/ndf",2);
      
      histname = "Chi2oNDFVsEta_" + histTag;
      tkmes.Chi2oNDFVsEta   = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, Chi2NDFMin, Chi2NDFMax,"");
      tkmes.Chi2oNDFVsEta->setAxisTitle("Track #eta",1);
      tkmes.Chi2oNDFVsEta->setAxisTitle("Track #chi^{2}/ndf",2);
      
      histname = "Chi2ProbVsPhi_" + histTag;
      tkmes.Chi2ProbVsPhi = dqmStore_->bookProfile(histname+CategoryName, histname+CategoryName, PhiBin, PhiMin, PhiMax, Chi2ProbMin, Chi2ProbMax);
      tkmes.Chi2ProbVsPhi->setAxisTitle("Tracks #phi"  ,1);
      tkmes.Chi2ProbVsPhi->setAxisTitle("Track #chi^{2} probability",2);
      
      histname = "Chi2ProbVsEta_" + histTag;
      tkmes.Chi2ProbVsEta = dqmStore_->bookProfile(histname+CategoryName, histname+CategoryName, EtaBin, EtaMin, EtaMax, Chi2ProbMin, Chi2ProbMax);
      tkmes.Chi2ProbVsEta->setAxisTitle("Tracks #eta"  ,1);
      tkmes.Chi2ProbVsEta->setAxisTitle("Track #chi^{2} probability",2);

    }
    
    // general properties
    dqmStore_->setCurrentFolder(TopFolder_+"/GeneralProperties");

    histname = "TrackP_" + histTag;
    tkmes.TrackP = dqmStore_->book1D(histname, histname, TrackPBin, TrackPMin, TrackPMax);
    tkmes.TrackP->setAxisTitle("Track |p| (GeV/c)", 1);
    tkmes.TrackP->setAxisTitle("Number of Tracks",2);

    histname = "TrackPt_" + histTag;
    tkmes.TrackPt = dqmStore_->book1D(histname, histname, TrackPtBin, TrackPtMin, TrackPtMax);
    tkmes.TrackPt->setAxisTitle("Track p_{T} (GeV/c)", 1);
    tkmes.TrackPt->setAxisTitle("Number of Tracks",2);

    if (doTrackPxPyPlots_) {
      histname = "TrackPx_" + histTag;
      tkmes.TrackPx = dqmStore_->book1D(histname, histname, TrackPxBin, TrackPxMin, TrackPxMax);
      tkmes.TrackPx->setAxisTitle("Track p_{x} (GeV/c)", 1);
      tkmes.TrackPx->setAxisTitle("Number of Tracks",2);

      histname = "TrackPy_" + histTag;
      tkmes.TrackPy = dqmStore_->book1D(histname, histname, TrackPyBin, TrackPyMin, TrackPyMax);
      tkmes.TrackPy->setAxisTitle("Track p_{y} (GeV/c)", 1);
      tkmes.TrackPy->setAxisTitle("Number of Tracks",2);
    }
    histname = "TrackPz_" + histTag;
    tkmes.TrackPz = dqmStore_->book1D(histname, histname, TrackPzBin, TrackPzMin, TrackPzMax);
    tkmes.TrackPz->setAxisTitle("Track p_{z} (GeV/c)", 1);
    tkmes.TrackPz->setAxisTitle("Number of Tracks",2);

    histname = "TrackPhi_" + histTag;
    tkmes.TrackPhi = dqmStore_->book1D(histname, histname, PhiBin, PhiMin, PhiMax);
    tkmes.TrackPhi->setAxisTitle("Track #phi", 1);
    tkmes.TrackPhi->setAxisTitle("Number of Tracks",2);

    histname = "TrackEta_" + histTag;
    tkmes.TrackEta = dqmStore_->book1D(histname, histname, EtaBin, EtaMin, EtaMax);
    tkmes.TrackEta->setAxisTitle("Track #eta", 1);
    tkmes.TrackEta->setAxisTitle("Number of Tracks",2);

    if (doThetaPlots_) {  
      histname = "TrackTheta_" + histTag;
      tkmes.TrackTheta = dqmStore_->book1D(histname, histname, ThetaBin, ThetaMin, ThetaMax);
      tkmes.TrackTheta->setAxisTitle("Track #theta", 1);
      tkmes.TrackTheta->setAxisTitle("Number of Tracks",2);
    }
    histname = "TrackQ_" + histTag;
    tkmes.TrackQ = dqmStore_->book1D(histname, histname, TrackQBin, TrackQMin, TrackQMax);
    tkmes.TrackQ->setAxisTitle("Track Charge", 1);
    tkmes.TrackQ->setAxisTitle("Number of Tracks",2);

    histname = "TrackPErrOverP_" + histTag;
    tkmes.TrackPErr = dqmStore_->book1D(histname, histname, pErrBin, pErrMin, pErrMax);
    tkmes.TrackPErr->setAxisTitle("track error(p)/p", 1);
    tkmes.TrackPErr->setAxisTitle("Number of Tracks",2);

    histname = "TrackPtErrOverPt_" + histTag;
    tkmes.TrackPtErr = dqmStore_->book1D(histname, histname, ptErrBin, ptErrMin, ptErrMax);
    tkmes.TrackPtErr->setAxisTitle("track error(p_{T})/p_{T}", 1);
    tkmes.TrackPtErr->setAxisTitle("Number of Tracks",2);

    histname = "TrackPtErrOverPtVsEta_" + histTag;
    tkmes.TrackPtErrVsEta = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, ptErrMin, ptErrMax);
    tkmes.TrackPtErrVsEta->setAxisTitle("Track #eta",1);
    tkmes.TrackPtErrVsEta->setAxisTitle("track error(p_{T})/p_{T}", 2);

    if (doTrackPxPyPlots_) {
      histname = "TrackPxErrOverPx_" + histTag;
      tkmes.TrackPxErr = dqmStore_->book1D(histname, histname, pxErrBin, pxErrMin, pxErrMax);
      tkmes.TrackPxErr->setAxisTitle("track error(p_{x})/p_{x}", 1);
      tkmes.TrackPxErr->setAxisTitle("Number of Tracks",2);
      
      histname = "TrackPyErrOverPy_" + histTag;
      tkmes.TrackPyErr = dqmStore_->book1D(histname, histname, pyErrBin, pyErrMin, pyErrMax);
      tkmes.TrackPyErr->setAxisTitle("track error(p_{y})/p_{y}", 1);
      tkmes.TrackPyErr->setAxisTitle("Number of Tracks",2);
    }
    histname = "TrackPzErrOverPz_" + histTag;
    tkmes.TrackPzErr = dqmStore_->book1D(histname, histname, pzErrBin, pzErrMin, pzErrMax);
    tkmes.TrackPzErr->setAxisTitle("track error(p_{z})/p_{z}", 1);
    tkmes.TrackPzErr->setAxisTitle("Number of Tracks",2);

    histname = "TrackPhiErr_" + histTag;
    tkmes.TrackPhiErr = dqmStore_->book1D(histname, histname, phiErrBin, phiErrMin, phiErrMax);
    tkmes.TrackPhiErr->setAxisTitle("track error(#phi)");
    tkmes.TrackPhiErr->setAxisTitle("Number of Tracks",2);

    histname = "TrackEtaErr_" + histTag;
    tkmes.TrackEtaErr = dqmStore_->book1D(histname, histname, etaErrBin, etaErrMin, etaErrMax);
    tkmes.TrackEtaErr->setAxisTitle("track error(#eta)");
    tkmes.TrackEtaErr->setAxisTitle("Number of Tracks",2);

    // rec hit profiles
    dqmStore_->setCurrentFolder(TopFolder_+"/GeneralProperties");
    histname = "NumberOfRecHitsPerTrackVsPhi_" + histTag;
    tkmes.NumberOfRecHitsPerTrackVsPhi = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, RecHitBin, RecHitMin, RecHitMax,"");
    tkmes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Track #phi",1);
    tkmes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Number of RecHits in each Track",2);

    if (doThetaPlots_) {
      histname = "NumberOfRecHitsPerTrackVsTheta_" + histTag;
      tkmes.NumberOfRecHitsPerTrackVsTheta = dqmStore_->bookProfile(histname, histname, ThetaBin, ThetaMin, ThetaMax, RecHitBin, RecHitMin, RecHitMax,"");
      tkmes.NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Track #phi",1);
      tkmes.NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Number of RecHits in each Track",2);
    }
    histname = "NumberOfRecHitsPerTrackVsEta_" + histTag;
    tkmes.NumberOfRecHitsPerTrackVsEta = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, RecHitBin, RecHitMin, RecHitMax,"");
    tkmes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Track #eta",1);
    tkmes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Number of RecHits in each Track",2);

    histname = "NumberOfValidRecHitsPerTrackVsPhi_" + histTag;
    tkmes.NumberOfValidRecHitsPerTrackVsPhi = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, RecHitMin, RecHitMax,"");
    tkmes.NumberOfValidRecHitsPerTrackVsPhi->setAxisTitle("Track #phi",1);
    tkmes.NumberOfValidRecHitsPerTrackVsPhi->setAxisTitle("Number of valid RecHits in each Track",2);
    
    //    std::cout << "[TrackAnalyzer::bookHistosForState] histTag: " << histTag << std::endl;
    histname = "NumberOfValidRecHitsPerTrackVsEta_" + histTag;
    tkmes.NumberOfValidRecHitsPerTrackVsEta = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, RecHitMin, RecHitMax,"");
    tkmes.NumberOfValidRecHitsPerTrackVsEta->setAxisTitle("Track #eta",1);
    tkmes.NumberOfValidRecHitsPerTrackVsEta->setAxisTitle("Number of valid RecHits in each Track",2);
    
    //////////////////////////////////////////
    histname = "NumberOfLayersPerTrackVsPhi_" + histTag;
    tkmes.NumberOfLayersPerTrackVsPhi = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, RecLayBin, RecLayMin, RecLayMax,"");
    tkmes.NumberOfLayersPerTrackVsPhi->setAxisTitle("Track #phi",1);
    tkmes.NumberOfLayersPerTrackVsPhi->setAxisTitle("Number of Layers in each Track",2);

    if (doThetaPlots_) {
      histname = "NumberOfLayersPerTrackVsTheta_" + histTag;
      tkmes.NumberOfLayersPerTrackVsTheta = dqmStore_->bookProfile(histname, histname, ThetaBin, ThetaMin, ThetaMax, RecLayBin, RecLayMin, RecLayMax,"");
      tkmes.NumberOfLayersPerTrackVsTheta->setAxisTitle("Track #phi",1);
      tkmes.NumberOfLayersPerTrackVsTheta->setAxisTitle("Number of Layers in each Track",2);
    }
    histname = "NumberOfLayersPerTrackVsEta_" + histTag;
    tkmes.NumberOfLayersPerTrackVsEta = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, RecLayBin, RecLayMin, RecLayMax,"");
    tkmes.NumberOfLayersPerTrackVsEta->setAxisTitle("Track #eta",1);
    tkmes.NumberOfLayersPerTrackVsEta->setAxisTitle("Number of Layers in each Track",2);

    if (doThetaPlots_) {
      histname = "Chi2oNDFVsTheta_" + histTag;
      tkmes.Chi2oNDFVsTheta = dqmStore_->bookProfile(histname, histname, ThetaBin, ThetaMin, ThetaMax, Chi2NDFMin, Chi2NDFMax,"");
      tkmes.Chi2oNDFVsTheta->setAxisTitle("Track #theta",1);
      tkmes.Chi2oNDFVsTheta->setAxisTitle("Track #chi^{2}/ndf",2);
    }
    if (doAllPlots_) {
      histname = "Chi2oNDFVsPhi_" + histTag;
      tkmes.Chi2oNDFVsPhi   = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, Chi2NDFMin, Chi2NDFMax,"");
      tkmes.Chi2oNDFVsPhi->setAxisTitle("Track #phi",1);
      tkmes.Chi2oNDFVsPhi->setAxisTitle("Track #chi^{2}/ndf",2);
      
      histname = "Chi2oNDFVsEta_" + histTag;
      tkmes.Chi2oNDFVsEta   = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, Chi2NDFMin, Chi2NDFMax,"");
      tkmes.Chi2oNDFVsEta->setAxisTitle("Track #eta",1);
      tkmes.Chi2oNDFVsEta->setAxisTitle("Track #chi^{2}/ndf",2);
      
      histname = "Chi2ProbVsPhi_" + histTag;
      tkmes.Chi2ProbVsPhi = dqmStore_->bookProfile(histname+CategoryName, histname+CategoryName, PhiBin, PhiMin, PhiMax, Chi2ProbMin, Chi2ProbMax);
      tkmes.Chi2ProbVsPhi->setAxisTitle("Tracks #phi"  ,1);
      tkmes.Chi2ProbVsPhi->setAxisTitle("Track #chi^{2} probability",2);
      
      histname = "Chi2ProbVsEta_" + histTag;
      tkmes.Chi2ProbVsEta = dqmStore_->bookProfile(histname+CategoryName, histname+CategoryName, EtaBin, EtaMin, EtaMax, Chi2ProbMin, Chi2ProbMax);
      tkmes.Chi2ProbVsEta->setAxisTitle("Tracks #eta"  ,1);
      tkmes.Chi2ProbVsEta->setAxisTitle("Track #chi^{2} probability",2);
    }

    // now put the MEs in the map
    TkParameterMEMap.insert( std::make_pair(sname, tkmes) );

}


// fill histograms at differnt measurement points
// ---------------------------------------------------------------------------------//
void TrackAnalyzer::fillHistosForState(const edm::EventSetup& iSetup, const reco::Track & track, std::string sname) 
{
    //get the kinematic parameters
    double p, px, py, pz, pt, theta, phi, eta, q;
    double pxerror, pyerror, pzerror, pterror, perror, phierror, etaerror;

    if (sname == "default") {

      p     = track.p();
      px    = track.px();
      py    = track.py();
      pz    = track.pz();
      pt    = track.pt();
      phi   = track.phi();
      theta = track.theta();
      eta   = track.eta();
      q     = track.charge();
      
      pterror  = (pt) ? track.ptError()/(pt*pt) : 0.0;
      pxerror  = -1.0;
      pyerror  = -1.0;
      pzerror  = -1.0;
      perror   = -1.0;
      phierror = track.phiError();
      etaerror = track.etaError();
      
    } else {
      
      edm::ESHandle<TransientTrackBuilder> theB;
      iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
      reco::TransientTrack TransTrack = theB->build(track);
      
      TrajectoryStateOnSurface TSOS;

      if      (sname == "OuterSurface")  TSOS = TransTrack.outermostMeasurementState();
      else if (sname == "InnerSurface")  TSOS = TransTrack.innermostMeasurementState();
      else if (sname == "ImpactPoint")   TSOS = TransTrack.impactPointState();

      p     = TSOS.globalMomentum().mag();
      px    = TSOS.globalMomentum().x();
      py    = TSOS.globalMomentum().y();
      pz    = TSOS.globalMomentum().z();
      pt    = TSOS.globalMomentum().perp();
      phi   = TSOS.globalMomentum().phi();
      theta = TSOS.globalMomentum().theta();
      eta   = TSOS.globalMomentum().eta();
      q     = TSOS.charge();

      //get the error of the kinimatic parameters
      AlgebraicSymMatrix66 errors = TSOS.cartesianError().matrix();
      double partialPterror = errors(3,3)*pow(TSOS.globalMomentum().x(),2) + errors(4,4)*pow(TSOS.globalMomentum().y(),2);
      pterror  = sqrt(partialPterror)/TSOS.globalMomentum().perp();
      pxerror  = sqrt(errors(3,3))/TSOS.globalMomentum().x();
      pyerror  = sqrt(errors(4,4))/TSOS.globalMomentum().y();
      pzerror  = sqrt(errors(5,5))/TSOS.globalMomentum().z();
      perror   = sqrt(partialPterror+errors(5,5)*pow(TSOS.globalMomentum().z(),2))/TSOS.globalMomentum().mag();
      phierror = sqrt(TSOS.curvilinearError().matrix()(2,2));
      etaerror = sqrt(TSOS.curvilinearError().matrix()(1,1))*fabs(sin(TSOS.globalMomentum().theta()));

    }

    std::map<std::string, TkParameterMEs>::iterator iPos = TkParameterMEMap.find(sname); 
    if (iPos != TkParameterMEMap.end()) {

      TkParameterMEs tkmes = iPos->second;
      
      // momentum
      tkmes.TrackP->Fill(p);
      if (doTrackPxPyPlots_) {
	tkmes.TrackPx->Fill(px);
	tkmes.TrackPy->Fill(py);
      }
      tkmes.TrackPz->Fill(pz);
      tkmes.TrackPt->Fill(pt);
      
      // angles
      tkmes.TrackPhi->Fill(phi);
      tkmes.TrackEta->Fill(eta);
      if (doThetaPlots_) {
	tkmes.TrackTheta->Fill(theta);
      }
      tkmes.TrackQ->Fill(q);
      
      // errors
      tkmes.TrackPtErr->Fill(pterror);
      tkmes.TrackPtErrVsEta->Fill(eta,pterror);
      if (doTrackPxPyPlots_) {
	tkmes.TrackPxErr->Fill(pxerror);
	tkmes.TrackPyErr->Fill(pyerror);
      }
      tkmes.TrackPzErr->Fill(pzerror);
      tkmes.TrackPErr->Fill(perror);
      tkmes.TrackPhiErr->Fill(phierror);
      tkmes.TrackEtaErr->Fill(etaerror);
      
      int nRecHits      = track.hitPattern().numberOfHits();
      int nValidRecHits = track.numberOfValidHits();
      // rec hits 
      tkmes.NumberOfRecHitsPerTrackVsPhi->Fill(phi,    nRecHits);
      if (doThetaPlots_) {
	tkmes.NumberOfRecHitsPerTrackVsTheta->Fill(theta,nRecHits);
      }
      tkmes.NumberOfRecHitsPerTrackVsEta->Fill(eta,    nRecHits);
      
      tkmes.NumberOfValidRecHitsPerTrackVsPhi->Fill(phi,    nValidRecHits);
      tkmes.NumberOfValidRecHitsPerTrackVsEta->Fill(eta,    nValidRecHits);

      int nLayers = track.hitPattern().stripLayersWithMeasurement();
      // rec layers 
      tkmes.NumberOfLayersPerTrackVsPhi->Fill(phi,     nLayers);
      if (doThetaPlots_) {
	tkmes.NumberOfLayersPerTrackVsTheta->Fill(theta, nLayers);
      }
      tkmes.NumberOfLayersPerTrackVsEta->Fill(eta,     nLayers);
      
      double chi2prob = TMath::Prob(track.chi2(),(int)track.ndof());
      double chi2oNDF = track.normalizedChi2();

      if(doAllPlots_) {
	
	// general properties
	if (doThetaPlots_) {
	  tkmes.Chi2oNDFVsTheta->Fill(theta, chi2oNDF);
	}
	tkmes.Chi2oNDFVsPhi->Fill(phi, chi2oNDF);
	tkmes.Chi2oNDFVsEta->Fill(eta, chi2oNDF);
	tkmes.Chi2ProbVsPhi->Fill(phi, chi2prob);
	tkmes.Chi2ProbVsEta->Fill(eta, chi2prob);
      }
      
    }

}


void TrackAnalyzer::bookHistosForTrackerSpecific(DQMStore * dqmStore_) 
{

    // parameters from the configuration
    std::string QualName     = conf_.getParameter<std::string>("Quality");
    std::string AlgoName     = conf_.getParameter<std::string>("AlgoName");

    // use the AlgoName and Quality Name 
    std::string CategoryName = QualName != "" ? AlgoName + "_" + QualName : AlgoName;

    int    PhiBin     = conf_.getParameter<int>(   "PhiBin");
    double PhiMin     = conf_.getParameter<double>("PhiMin");
    double PhiMax     = conf_.getParameter<double>("PhiMax");

    int    EtaBin     = conf_.getParameter<int>(   "EtaBin");
    double EtaMin     = conf_.getParameter<double>("EtaMin");
    double EtaMax     = conf_.getParameter<double>("EtaMax");

    // book hit property histograms
    // ---------------------------------------------------------------------------------//
    dqmStore_->setCurrentFolder(TopFolder_+"/HitProperties");



    std::vector<std::string> subdetectors = conf_.getParameter<std::vector<std::string> >("subdetectors");
    int detBin = conf_.getParameter<int>("subdetectorBin");

    for ( auto det : subdetectors ) {
      
      // hits properties
      dqmStore_->setCurrentFolder(TopFolder_+"/HitProperties/"+det);
      
      TkRecHitsPerSubDetMEs recHitsPerSubDet_mes;

      recHitsPerSubDet_mes.detectorTag = det;
      int detID = -1;
      if ( det == "TIB" ) detID = StripSubdetector::TIB;
      if ( det == "TOB" ) detID = StripSubdetector::TOB;
      if ( det == "TID" ) detID = StripSubdetector::TID;
      if ( det == "TEC" ) detID = StripSubdetector::TEC;
      if ( det == "PixBarrel" ) detID = PixelSubdetector::PixelBarrel;
      if ( det == "PixEndcap" ) detID = PixelSubdetector::PixelEndcap;
      recHitsPerSubDet_mes.detectorId  = detID;

      histname = "NumberOfRecHitsPerTrack_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrack = dqmStore_->book1D(histname, histname, detBin, -0.5, double(detBin)-0.5);
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrack->setAxisTitle("Number of " + det + " valid RecHits in each Track",1);
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfRecHitsPerTrackVsPhi_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsPhi = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, detBin, -0.5, double(detBin)-0.5,"");
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Track #phi",1);
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Number of " + det + " valid RecHits in each Track",2);

      histname = "NumberOfRecHitsPerTrackVsEta_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsEta = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, detBin, -0.5, double(detBin)-0.5,"");
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Track #eta",1);
      recHitsPerSubDet_mes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Number of " + det + " valid RecHits in each Track",2);

      histname = "NumberOfLayersPerTrack_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfLayersPerTrack = dqmStore_->book1D(histname, histname, detBin, -0.5, double(detBin)-0.5);
      recHitsPerSubDet_mes.NumberOfLayersPerTrack->setAxisTitle("Number of " + det + " valid Layers in each Track",1);
      recHitsPerSubDet_mes.NumberOfLayersPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfLayersPerTrackVsPhi_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsPhi = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, detBin, -0.5, double(detBin)-0.5,"");
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsPhi->setAxisTitle("Track #phi",1);
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsPhi->setAxisTitle("Number of " + det + " valid Layers in each Track",2);

      histname = "NumberOfLayersPerTrackVsEta_" + det + "_" + CategoryName;
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsEta = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, detBin, -0.5, double(detBin)-0.5,"");
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsEta->setAxisTitle("Track #eta",1);
      recHitsPerSubDet_mes.NumberOfLayersPerTrackVsEta->setAxisTitle("Number of " + det + " valid Layers in each Track",2);

      TkRecHitsPerSubDetMEMap.insert(std::pair<std::string,TkRecHitsPerSubDetMEs>(det,recHitsPerSubDet_mes));

      
    }


}


void TrackAnalyzer::fillHistosForTrackerSpecific(const reco::Track & track) 
{
    
  double phi   = track.phi();
  double eta   = track.eta();

  for ( std::map<std::string,TkRecHitsPerSubDetMEs>::iterator it = TkRecHitsPerSubDetMEMap.begin();
       it != TkRecHitsPerSubDetMEMap.end(); it++ ) {

    int nValidLayers  = 0;
    int nValidRecHits = 0;
    int substr = it->second.detectorId;
    switch(substr) {
    case StripSubdetector::TIB :
      nValidLayers  = track.hitPattern().stripTIBLayersWithMeasurement();       // case 0: strip TIB
      nValidRecHits = track.hitPattern().numberOfValidStripTIBHits();           // case 0: strip TIB
      break;
    case StripSubdetector::TID :
      nValidLayers  = track.hitPattern().stripTIDLayersWithMeasurement();       // case 0: strip TID
      nValidRecHits = track.hitPattern().numberOfValidStripTIDHits();           // case 0: strip TID
      break;
    case StripSubdetector::TOB :
      nValidLayers  = track.hitPattern().stripTOBLayersWithMeasurement();       // case 0: strip TOB
      nValidRecHits = track.hitPattern().numberOfValidStripTOBHits();           // case 0: strip TOB
      break;
    case StripSubdetector::TEC :
      nValidLayers  = track.hitPattern().stripTECLayersWithMeasurement();       // case 0: strip TEC
      nValidRecHits = track.hitPattern().numberOfValidStripTECHits();           // case 0: strip TEC
      break;
    case PixelSubdetector::PixelBarrel :
      nValidLayers  = track.hitPattern().pixelBarrelLayersWithMeasurement();    // case 0: pixel PXB
      nValidRecHits = track.hitPattern().numberOfValidPixelBarrelHits();        // case 0: pixel PXB
      break;
    case PixelSubdetector::PixelEndcap :
      nValidLayers  = track.hitPattern().pixelEndcapLayersWithMeasurement();    // case 0: pixel PXF
      nValidRecHits = track.hitPattern().numberOfValidPixelEndcapHits();        // case 0: pixel PXF
      break;
    default :
      break;
    }

    //Fill Layers and RecHits
    it->second.NumberOfRecHitsPerTrack      -> Fill(nValidRecHits); 
    it->second.NumberOfRecHitsPerTrackVsPhi -> Fill(phi,    nValidRecHits);
    it->second.NumberOfRecHitsPerTrackVsEta -> Fill(eta,    nValidRecHits);
    
    it->second.NumberOfLayersPerTrack      -> Fill(nValidLayers);
    it->second.NumberOfLayersPerTrackVsPhi -> Fill(phi,     nValidLayers);
    it->second.NumberOfLayersPerTrackVsEta -> Fill(eta,     nValidLayers);
  }

}
//
// -- Set Lumi Flag
//
void TrackAnalyzer::setLumiFlag() { 

  TkParameterMEs tkmes;
  if ( Chi2oNDF_lumiFlag                ) Chi2oNDF_lumiFlag                -> setLumiFlag();
  if ( NumberOfRecHitsPerTrack_lumiFlag ) NumberOfRecHitsPerTrack_lumiFlag -> setLumiFlag();
}
//
// -- Apply SoftReset 
//
void TrackAnalyzer::doSoftReset(DQMStore * dqmStore_) {
  TkParameterMEs tkmes;
  dqmStore_->softReset(Chi2oNDF);
  dqmStore_->softReset(NumberOfRecHitsPerTrack);
}
//
// -- Apply Reset 
//
void TrackAnalyzer::doReset(DQMStore * dqmStore_) {
  TkParameterMEs tkmes;
  if ( Chi2oNDF_lumiFlag                ) Chi2oNDF_lumiFlag                -> Reset();
  if ( NumberOfRecHitsPerTrack_lumiFlag ) NumberOfRecHitsPerTrack_lumiFlag -> Reset();
}
//
// -- Remove SoftReset
//
void TrackAnalyzer::undoSoftReset(DQMStore * dqmStore_) {
  TkParameterMEs tkmes;
  dqmStore_->disableSoftReset(Chi2oNDF);
  dqmStore_->disableSoftReset(NumberOfRecHitsPerTrack);
}


