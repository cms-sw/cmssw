/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/03/28 22:59:43 $
 *  $Revision: 1.25 $
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
    , doTrackerSpecific_                   ( conf_.getParameter<bool>("doTrackerSpecific") )
    , doAllPlots_                          ( conf_.getParameter<bool>("doAllPlots") )
    , doBSPlots_                           ( conf_.getParameter<bool>("doBeamSpotPlots") )
    , doGoodTrackPlots_                    ( conf_.getParameter<bool>("doGoodTrackPlots") )
    , doDCAPlots_                          ( conf_.getParameter<bool>("doDCAPlots") )
    , doGeneralPropertiesPlots_            ( conf_.getParameter<bool>("doGeneralPropertiesPlots") )
    , doMeasurementStatePlots_             ( conf_.getParameter<bool>("doMeasurementStatePlots") )
    , doHitPropertiesPlots_                ( conf_.getParameter<bool>("doHitPropertiesPlots") )
    , doRecHitVsPhiVsEtaPerTrack_          ( conf_.getParameter<bool>("doRecHitVsPhiVsEtaPerTrack") )
    , doLayersVsPhiVsEtaPerTrack_          ( conf_.getParameter<bool>("doLayersVsPhiVsEtaPerTrack") )
    , doGoodTrackRecHitVsPhiVsEtaPerTrack_ ( conf_.getParameter<bool>("doGoodTrackRecHitVsPhiVsEtaPerTrack") )
    , doGoodTrackLayersVsPhiVsEtaPerTrack_ ( conf_.getParameter<bool>("doGoodTrackLayersVsPhiVsEtaPerTrack") )
    , doGoodTrack2DChi2Plots_              ( conf_.getParameter<bool>("doGoodTrack2DChi2Plots") )
    , doThetaPlots_                        ( conf_.getParameter<bool>("doThetaPlots") )
    , doTrackPxPyPlots_                    ( conf_.getParameter<bool>("doTrackPxPyPlots") )
    , doDCAwrt000Plots_                    ( conf_.getParameter<bool>("doDCAwrt000Plots") )
    , doLumiAnalysis_                      ( conf_.getParameter<bool>("doLumiAnalysis") )
    , doTestPlots_                         ( conf_.getParameter<bool>("doTestPlots") )
    , NumberOfRecHitsPerTrack(NULL)
    , NumberOfRecHitsFoundPerTrack(NULL)
    , NumberOfRecHitsLostPerTrack(NULL)
    , NumberOfLayersPerTrack(NULL)
    , NumberOfRecHitVsPhiVsEtaPerTrack(NULL)
    , NumberOfLayersVsPhiVsEtaPerTrack(NULL)
    , Chi2(NULL)
    , Chi2Prob(NULL)
    , Chi2oNDF(NULL)
    , DistanceOfClosestApproach(NULL)
    , DistanceOfClosestApproachToBS(NULL)
    , DistanceOfClosestApproachVsTheta(NULL)
    , DistanceOfClosestApproachVsPhi(NULL)
    , DistanceOfClosestApproachToBSVsPhi(NULL)
    , DistanceOfClosestApproachVsEta(NULL)
    , xPointOfClosestApproach(NULL)
    , xPointOfClosestApproachVsZ0wrt000(NULL)
    , xPointOfClosestApproachVsZ0wrtBS(NULL)
    , yPointOfClosestApproach(NULL)
    , yPointOfClosestApproachVsZ0wrt000(NULL)
    , yPointOfClosestApproachVsZ0wrtBS(NULL)
    , zPointOfClosestApproach(NULL)
    , zPointOfClosestApproachVsPhi(NULL)
    , algorithm(NULL)
     // TESTING MEs
    , TESTDistanceOfClosestApproachToBS(NULL)
    , TESTDistanceOfClosestApproachToBSVsPhi(NULL)
			    // add by mia in order to deal w/ LS transitions
    , Chi2oNDF_lumiFlag(NULL)
    , NumberOfRecHitsPerTrack_lumiFlag(NULL)
    , GoodTrackChi2oNDF_lumiFlag(NULL)
    , GoodTrackNumberOfRecHitsPerTrack_lumiFlag(NULL)

    , NumberOfTOBRecHitsPerTrack(NULL)
    , NumberOfTOBRecHitsPerTrackVsPhiProfile(NULL)
    , NumberOfTOBRecHitsPerTrackVsEtaProfile(NULL)
    , NumberOfTOBLayersPerTrack(NULL)
    , NumberOfTOBLayersPerTrackVsPhiProfile(NULL)
    , NumberOfTOBLayersPerTrackVsEtaProfile(NULL)

    , NumberOfTIBRecHitsPerTrack(NULL)
    , NumberOfTIBRecHitsPerTrackVsPhiProfile(NULL)
    , NumberOfTIBRecHitsPerTrackVsEtaProfile(NULL)
    , NumberOfTIBLayersPerTrack(NULL)
    , NumberOfTIBLayersPerTrackVsPhiProfile(NULL)
    , NumberOfTIBLayersPerTrackVsEtaProfile(NULL)

    , NumberOfTIDRecHitsPerTrack(NULL)
    , NumberOfTIDRecHitsPerTrackVsPhiProfile(NULL)
    , NumberOfTIDRecHitsPerTrackVsEtaProfile(NULL)
    , NumberOfTIDLayersPerTrack(NULL)
    , NumberOfTIDLayersPerTrackVsPhiProfile(NULL)
    , NumberOfTIDLayersPerTrackVsEtaProfile(NULL)

    , NumberOfTECRecHitsPerTrack(NULL)
    , NumberOfTECRecHitsPerTrackVsPhiProfile(NULL)
    , NumberOfTECRecHitsPerTrackVsEtaProfile(NULL)
    , NumberOfTECLayersPerTrack(NULL)
    , NumberOfTECLayersPerTrackVsPhiProfile(NULL)
    , NumberOfTECLayersPerTrackVsEtaProfile(NULL)

    , NumberOfPixBarrelRecHitsPerTrack(NULL)
    , NumberOfPixBarrelRecHitsPerTrackVsPhiProfile(NULL)
    , NumberOfPixBarrelRecHitsPerTrackVsEtaProfile(NULL)
    , NumberOfPixBarrelLayersPerTrack(NULL)
    , NumberOfPixBarrelLayersPerTrackVsPhiProfile(NULL)
    , NumberOfPixBarrelLayersPerTrackVsEtaProfile(NULL)

    , NumberOfPixEndcapRecHitsPerTrack(NULL)
    , NumberOfPixEndcapRecHitsPerTrackVsPhiProfile(NULL)
    , NumberOfPixEndcapRecHitsPerTrackVsEtaProfile(NULL)
    , NumberOfPixEndcapLayersPerTrack(NULL)
    , NumberOfPixEndcapLayersPerTrackVsPhiProfile(NULL)
    , NumberOfPixEndcapLayersPerTrackVsEtaProfile(NULL)

    , GoodTrackNumberOfRecHitVsPhiVsEtaPerTrack(NULL)
    , GoodTrackNumberOfLayersVsPhiVsEtaPerTrack(NULL)
    , GoodTrackNumberOfRecHitsPerTrackVsPhiProfile(NULL)
    , GoodTrackNumberOfRecHitsPerTrackVsEtaProfile(NULL)
    , GoodTrackNumberOfFoundRecHitsPerTrackVsPhiProfile(NULL)
    , GoodTrackNumberOfFoundRecHitsPerTrackVsEtaProfile(NULL)
    , GoodTrackChi2oNDF(NULL)
    , GoodTrackChi2Prob(NULL)
    , GoodTrackChi2oNDFVsPhi(NULL)
    , GoodTrackChi2ProbVsPhi(NULL)
    , GoodTrackChi2oNDFVsEta(NULL)
    , GoodTrackChi2ProbVsEta(NULL)
    , GoodTrackNumberOfRecHitsPerTrack(NULL)
    , GoodTrackNumberOfFoundRecHitsPerTrack(NULL)
    , GoodTrackAlgorithm(NULL)
{

  //  std::cout << "TrackAnalyzer::TrackAnalyzer() - doGoodTrackPlots_ = "  << doGoodTrackPlots_ << std::endl;

}

TrackAnalyzer::~TrackAnalyzer() 
{ 
}

void TrackAnalyzer::beginJob(DQMStore * dqmStore_) 
{

    // parameters from the configuration
    std::string QualName       = conf_.getParameter<std::string>("Quality");
    std::string AlgoName       = conf_.getParameter<std::string>("AlgoName");
    std::string MEFolderName   = conf_.getParameter<std::string>("FolderName"); 
    std::string MEBSFolderName = conf_.getParameter<std::string>("BSFolderName"); 

    // use the AlgoName and Quality Name 
    std::string CatagoryName = QualName != "" ? AlgoName + "_" + QualName : AlgoName;

    // get binning from the configuration
    double RecHitMin   = conf_.getParameter<double>("RecHitMin");
    double RecHitMax   = conf_.getParameter<double>("RecHitMax");

    int    TKHitBin     = conf_.getParameter<int>(   "RecHitBin");
    double TKHitMin     = conf_.getParameter<double>("RecHitMin");
    double TKHitMax     = conf_.getParameter<double>("RecHitMax");

    int    TKLostBin    = conf_.getParameter<int>(   "RecLostBin");
    double TKLostMin    = conf_.getParameter<double>("RecLostMin");
    double TKLostMax    = conf_.getParameter<double>("RecLostMax");

    int    TKLayBin     = conf_.getParameter<int>(   "RecLayBin");
    double TKLayMin     = conf_.getParameter<double>("RecLayMin");
    double TKLayMax     = conf_.getParameter<double>("RecLayMax");

    int    Chi2Bin      = conf_.getParameter<int>(   "Chi2Bin");
    double Chi2Min      = conf_.getParameter<double>("Chi2Min");
    double Chi2Max      = conf_.getParameter<double>("Chi2Max");

    int    Chi2NDFBin   = conf_.getParameter<int>(   "Chi2NDFBin");
    double Chi2NDFMin   = conf_.getParameter<double>("Chi2NDFMin");
    double Chi2NDFMax   = conf_.getParameter<double>("Chi2NDFMax");

    int    Chi2ProbBin  = conf_.getParameter<int>(   "Chi2ProbBin");
    double Chi2ProbMin  = conf_.getParameter<double>("Chi2ProbMin");
    double Chi2ProbMax  = conf_.getParameter<double>("Chi2ProbMax");

    int    PhiBin       = conf_.getParameter<int>(   "PhiBin");
    double PhiMin       = conf_.getParameter<double>("PhiMin");
    double PhiMax       = conf_.getParameter<double>("PhiMax");

    int    EtaBin       = conf_.getParameter<int>(   "EtaBin");
    double EtaMin       = conf_.getParameter<double>("EtaMin");
    double EtaMax       = conf_.getParameter<double>("EtaMax");

    int    ThetaBin     = conf_.getParameter<int>(   "ThetaBin");
    double ThetaMin     = conf_.getParameter<double>("ThetaMin");
    double ThetaMax     = conf_.getParameter<double>("ThetaMax");

    int    DxyBin       = conf_.getParameter<int>(   "DxyBin");
    double DxyMin       = conf_.getParameter<double>("DxyMin");
    double DxyMax       = conf_.getParameter<double>("DxyMax");

    int    VXBin        = conf_.getParameter<int>(   "VXBin");
    double VXMin        = conf_.getParameter<double>("VXMin");
    double VXMax        = conf_.getParameter<double>("VXMax");

    int    VYBin        = conf_.getParameter<int>(   "VYBin");
    double VYMin        = conf_.getParameter<double>("VYMin");
    double VYMax        = conf_.getParameter<double>("VYMax");

    int    VZBin        = conf_.getParameter<int>(   "VZBin");
    double VZMin        = conf_.getParameter<double>("VZMin");
    double VZMax        = conf_.getParameter<double>("VZMax");

    int    X0Bin        = conf_.getParameter<int>(   "X0Bin");
    double X0Min        = conf_.getParameter<double>("X0Min");
    double X0Max        = conf_.getParameter<double>("X0Max");

    int    Y0Bin        = conf_.getParameter<int>(   "Y0Bin");
    double Y0Min        = conf_.getParameter<double>("Y0Min");
    double Y0Max        = conf_.getParameter<double>("Y0Max");

    int    Z0Bin        = conf_.getParameter<int>(   "Z0Bin");
    double Z0Min        = conf_.getParameter<double>("Z0Min");
    double Z0Max        = conf_.getParameter<double>("Z0Max");

    dqmStore_->setCurrentFolder(MEFolderName);

    // book the Hit Property histograms
    // ---------------------------------------------------------------------------------//

    if ( doHitPropertiesPlots_ || doAllPlots_ ){

      dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties");
      
      histname = "NumberOfRecHitsPerTrack_";
      NumberOfRecHitsPerTrack = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TKHitBin, TKHitMin, TKHitMax);
      NumberOfRecHitsPerTrack->setAxisTitle("Number of all RecHits of each Track");
      NumberOfRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfRecHitsFoundPerTrack_";
      NumberOfRecHitsFoundPerTrack = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TKHitBin, TKHitMin, TKHitMax);
      NumberOfRecHitsFoundPerTrack->setAxisTitle("Number of RecHits found for each Track");
      NumberOfRecHitsFoundPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfRecHitsLostPerTrack_";
      NumberOfRecHitsLostPerTrack = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TKLostBin, TKLostMin, TKLostMax);
      NumberOfRecHitsLostPerTrack->setAxisTitle("Number of RecHits lost for each Track");
      NumberOfRecHitsLostPerTrack->setAxisTitle("Number of Tracks", 2);

      histname = "NumberOfLayersPerTrack_";
      NumberOfLayersPerTrack = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TKLayBin, TKLayMin, TKLayMax);
      NumberOfLayersPerTrack->setAxisTitle("Number of Layers of each Track", 1);
      NumberOfLayersPerTrack->setAxisTitle("Number of Tracks", 2);
      
      if ( doRecHitVsPhiVsEtaPerTrack_ || doAllPlots_ ){
	
	histname = "NumberOfRecHitVsPhiVsEtaPerTrack_";
	NumberOfRecHitVsPhiVsEtaPerTrack = dqmStore_->bookProfile2D(histname+CatagoryName, histname+CatagoryName, 
								    EtaBin, EtaMin, EtaMax, PhiBin, PhiMin, PhiMax, 0, 40., "");
	NumberOfRecHitVsPhiVsEtaPerTrack->setAxisTitle("Track #eta ", 1);
	NumberOfRecHitVsPhiVsEtaPerTrack->setAxisTitle("Track #phi ", 2);
      }
      if ( doLayersVsPhiVsEtaPerTrack_ || doAllPlots_ ){
	
	histname = "NumberOfLayersVsPhiVsEtaPerTrack_";
	NumberOfLayersVsPhiVsEtaPerTrack = dqmStore_->bookProfile2D(histname+CatagoryName, histname+CatagoryName, 
								    EtaBin, EtaMin, EtaMax, PhiBin, PhiMin, PhiMax, 0, 40., "");
	NumberOfLayersVsPhiVsEtaPerTrack->setAxisTitle("Track #eta ", 1);
	NumberOfLayersVsPhiVsEtaPerTrack->setAxisTitle("Track #phi ", 2);
      }
    }


    // book the General Property histograms
    // ---------------------------------------------------------------------------------//

    if (doGeneralPropertiesPlots_ || doAllPlots_){

      dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");

      histname = "Chi2_";
      Chi2 = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, Chi2Bin, Chi2Min, Chi2Max);
      Chi2->setAxisTitle("Track #chi^{2}"  ,1);
      Chi2->setAxisTitle("Number of Tracks",2);

      histname = "Chi2Prob_";
      Chi2Prob = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, Chi2ProbBin, Chi2ProbMin, Chi2ProbMax);
      Chi2Prob->setAxisTitle("Track #chi^{2} probability",1);
      Chi2Prob->setAxisTitle("Number of Tracks"        ,2);
      
      histname = "Chi2oNDF_";
      Chi2oNDF = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, Chi2NDFBin, Chi2NDFMin, Chi2NDFMax);
      Chi2oNDF->setAxisTitle("Track #chi^{2}/ndf",1);
      Chi2oNDF->setAxisTitle("Number of Tracks"  ,2);
      
      histname = "xPointOfClosestApproach_";
      xPointOfClosestApproach = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, VXBin, VXMin, VXMax);
      xPointOfClosestApproach->setAxisTitle("x component of Track PCA to beam line (cm)",1);
      xPointOfClosestApproach->setAxisTitle("Number of Tracks",2);
      
      histname = "yPointOfClosestApproach_";
      yPointOfClosestApproach = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, VYBin, VYMin, VYMax);
      yPointOfClosestApproach->setAxisTitle("y component of Track PCA to beam line (cm)",1);
      yPointOfClosestApproach->setAxisTitle("Number of Tracks",2);

      histname = "zPointOfClosestApproach_";
      zPointOfClosestApproach = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, VZBin, VZMin, VZMax);
      zPointOfClosestApproach->setAxisTitle("z component of Track PCA to beam line (cm)",1);
      zPointOfClosestApproach->setAxisTitle("Number of Tracks",2);
      
      // See DataFormats/TrackReco/interface/TrackBase.h for track algorithm enum definition
      // http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/DataFormats/TrackReco/interface/TrackBase.h?view=log
      histname = "algorithm_";
      algorithm = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, 32, 0., 32.);
      algorithm->setAxisTitle("Tracking algorithm",1);
      algorithm->setAxisTitle("Number of Tracks",2);
      
    }
    
    // book LS analysis related histograms
    // -----------------------------------
    if ( doLumiAnalysis_ ) {
      // add by Mia in order to deal w/ LS transitions  
      dqmStore_->setCurrentFolder(MEFolderName+"/LSanalysis");

      histname = "NumberOfRecHitsPerTrack_lumiFlag_";
      NumberOfRecHitsPerTrack_lumiFlag = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TKHitBin, TKHitMin, TKHitMax);
      NumberOfRecHitsPerTrack_lumiFlag->setAxisTitle("Number of all RecHits of each Track");
      NumberOfRecHitsPerTrack_lumiFlag->setAxisTitle("Number of Tracks", 2);

      histname = "Chi2oNDF_lumiFlag_";
      Chi2oNDF_lumiFlag = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, Chi2NDFBin, Chi2NDFMin, Chi2NDFMax);
      Chi2oNDF_lumiFlag->setAxisTitle("Track #chi^{2}/ndf",1);
      Chi2oNDF_lumiFlag->setAxisTitle("Number of Tracks"  ,2);

      histname = "GoodTrackNumberOfRecHitsPerTrack_lumiFlag_";
      GoodTrackNumberOfRecHitsPerTrack_lumiFlag = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TKHitBin, TKHitMin, TKHitMax);
      GoodTrackNumberOfRecHitsPerTrack_lumiFlag->setAxisTitle("Number of all RecHits of each Good Track");
      GoodTrackNumberOfRecHitsPerTrack_lumiFlag->setAxisTitle("Number of Good Tracks", 2);
      
      histname = "GoodTrackChi2oNDF_lumiFlag_";
      GoodTrackChi2oNDF_lumiFlag = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, Chi2NDFBin, Chi2NDFMin, Chi2NDFMax);
      GoodTrackChi2oNDF_lumiFlag->setAxisTitle("Good Track #chi^{2}/ndf",1);
      GoodTrackChi2oNDF_lumiFlag->setAxisTitle("Number of Good Tracks"  ,2);
    }


    // book the Beam Spot related histograms
    // ---------------------------------------------------------------------------------//
    
    if(doBSPlots_ || doAllPlots_)
      {
	//        dqmStore_->setCurrentFolder(MEBSFolderName);
	dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");

        histname = "DistanceOfClosestApproachToBS_";
        DistanceOfClosestApproachToBS = dqmStore_->book1D(histname+CatagoryName,histname+CatagoryName,DxyBin,DxyMin,DxyMax);
        DistanceOfClosestApproachToBS->setAxisTitle("Track d_{xy} wrt beam spot (cm)",1);
        DistanceOfClosestApproachToBS->setAxisTitle("Number of Tracks",2);
	
        histname = "DistanceOfClosestApproachToBSVsPhi_";
        DistanceOfClosestApproachToBSVsPhi = dqmStore_->bookProfile(histname+CatagoryName,histname+CatagoryName, PhiBin, PhiMin, PhiMax, DxyBin, DxyMin, DxyMax,"");
	DistanceOfClosestApproachToBSVsPhi->getTH1()->SetBit(TH1::kCanRebin);
        DistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track #phi",1);
        DistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track d_{xy} wrt beam spot (cm)",2);
	
        histname = "xPointOfClosestApproachVsZ0wrt000_";
        xPointOfClosestApproachVsZ0wrt000 = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, Z0Bin, Z0Min, Z0Max, X0Bin, X0Min, X0Max,"");
        xPointOfClosestApproachVsZ0wrt000->setAxisTitle("d_{z} (cm)",1);
        xPointOfClosestApproachVsZ0wrt000->setAxisTitle("x component of Track PCA to beam line (cm)",2);
	
        histname = "yPointOfClosestApproachVsZ0wrt000_";
        yPointOfClosestApproachVsZ0wrt000 = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, Z0Bin, Z0Min, Z0Max, Y0Bin, Y0Min, Y0Max,"");
        yPointOfClosestApproachVsZ0wrt000->setAxisTitle("d_{z} (cm)",1);
        yPointOfClosestApproachVsZ0wrt000->setAxisTitle("y component of Track PCA to beam line (cm)",2);

	histname = "xPointOfClosestApproachVsZ0wrtBS_";
	xPointOfClosestApproachVsZ0wrtBS = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, Z0Bin, Z0Min, Z0Max, X0Bin, X0Min, X0Max,"");
	xPointOfClosestApproachVsZ0wrtBS->setAxisTitle("d_{z} w.r.t. Beam Spot  (cm)",1);
	xPointOfClosestApproachVsZ0wrtBS->setAxisTitle("x component of Track PCA to BS (cm)",2);
	
	histname = "yPointOfClosestApproachVsZ0wrtBS_";
	yPointOfClosestApproachVsZ0wrtBS = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, Z0Bin, Z0Min, Z0Max, Y0Bin, Y0Min, Y0Max,"");
	yPointOfClosestApproachVsZ0wrtBS->setAxisTitle("d_{z} w.r.t. Beam Spot (cm)",1);
	yPointOfClosestApproachVsZ0wrtBS->setAxisTitle("y component of Track PCA to BS (cm)",2);

	histname = "zPointOfClosestApproachVsPhi_";
	zPointOfClosestApproachVsPhi = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, PhiBin, PhiMin, PhiMax, VZBin, VZMin, VZMax, "");
	zPointOfClosestApproachVsPhi->setAxisTitle("Track #phi",1);
	zPointOfClosestApproachVsPhi->setAxisTitle("y component of Track PCA to beam line (cm)",2);

	if (doTestPlots_) {
	  
	  histname = "TESTDistanceOfClosestApproachToBS_";
	  TESTDistanceOfClosestApproachToBS = dqmStore_->book1D(histname+CatagoryName,histname+CatagoryName,DxyBin,DxyMin,DxyMax);
	  TESTDistanceOfClosestApproachToBS->setAxisTitle("Track d_{xy} wrt beam spot (cm)",1);
	  TESTDistanceOfClosestApproachToBS->setAxisTitle("Number of Tracks",2);
	  
	  histname = "TESTDistanceOfClosestApproachToBSVsPhi_";
	  TESTDistanceOfClosestApproachToBSVsPhi = dqmStore_->bookProfile(histname+CatagoryName,histname+CatagoryName, PhiBin, PhiMin, PhiMax, DxyBin, DxyMin, DxyMax,"");
	  TESTDistanceOfClosestApproachToBSVsPhi->getTH1()->SetBit(TH1::kCanRebin);
	  TESTDistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track #phi",1);
	  TESTDistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track d_{xy} wrt beam spot (cm)",2);

	}

      }
    
    // book the Profile plots for DCA related histograms
    // ---------------------------------------------------------------------------------//
    if(doDCAPlots_ || doAllPlots_)
      {
	if (doDCAwrt000Plots_) {
	  if (doThetaPlots_) {
	    dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");
	    histname = "DistanceOfClosestApproachVsTheta_";
	    DistanceOfClosestApproachVsTheta = dqmStore_->bookProfile(histname+CatagoryName,histname+CatagoryName, ThetaBin, ThetaMin, ThetaMax, DxyMin,DxyMax,"");
	    DistanceOfClosestApproachVsTheta->setAxisTitle("Track #theta",1);
	    DistanceOfClosestApproachVsTheta->setAxisTitle("Track d_{xy} wrt (0,0,0) (cm)",2);
	  }
	  
	  histname = "DistanceOfClosestApproachVsEta_";
	  DistanceOfClosestApproachVsEta = dqmStore_->bookProfile(histname+CatagoryName,histname+CatagoryName, EtaBin, EtaMin, EtaMax, DxyMin, DxyMax,"");
	  DistanceOfClosestApproachVsEta->setAxisTitle("Track #eta",1);
	  DistanceOfClosestApproachVsEta->setAxisTitle("Track d_{xy} wrt (0,0,0) (cm)",2);
	  // temporary patch in order to put back those MEs in Muon Workspace

	  histname = "DistanceOfClosestApproach_";
	  DistanceOfClosestApproach = dqmStore_->book1D(histname+CatagoryName,histname+CatagoryName,DxyBin,DxyMin,DxyMax);
	  DistanceOfClosestApproach->setAxisTitle("Track d_{xy} wrt (0,0,0) (cm)",1);
	  DistanceOfClosestApproach->setAxisTitle("Number of Tracks",2);
	  
	  histname = "DistanceOfClosestApproachVsPhi_";
	  DistanceOfClosestApproachVsPhi = dqmStore_->bookProfile(histname+CatagoryName,histname+CatagoryName, PhiBin, PhiMin, PhiMax, DxyMin,DxyMax,"");
	  DistanceOfClosestApproachVsPhi->getTH1()->SetBit(TH1::kCanRebin);
	  DistanceOfClosestApproachVsPhi->setAxisTitle("Track #phi",1);
	  DistanceOfClosestApproachVsPhi->setAxisTitle("Track d_{xy} wrt (0,0,0) (cm)",2);
    	}
      }


    // book tracker specific related histograms
    // ---------------------------------------------------------------------------------//
    if(doTrackerSpecific_ || doAllPlots_) 
      {
        doTrackerSpecificInitialization(dqmStore_);
      }
    
    // book state related histograms
    // ---------------------------------------------------------------------------------//
    if (doMeasurementStatePlots_ || doAllPlots_){

      std::string StateName = conf_.getParameter<std::string>("MeasurementState");
      
      if (StateName == "All") 
	{
	  bookHistosForState("OuterSurface", dqmStore_);
	  bookHistosForState("InnerSurface", dqmStore_);
	  bookHistosForState("ImpactPoint" , dqmStore_);
	} 
      else if 
	(   
	 StateName != "OuterSurface" && 
	 StateName != "InnerSurface" && 
	 StateName != "ImpactPoint" &&
	 StateName != "default" 
	 ) 
	{
	  bookHistosForState("default", dqmStore_);
	}
      else
	{
	  bookHistosForState(StateName, dqmStore_);
	}
    }
    
    // book histos for good tracks (HP + Pt>1GeV)
    // ---------------------------------------------------------------------------------//

    if ( doGoodTrackPlots_ || doAllPlots_ ) {

      dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties/GoodTracks");

      histname = "GoodTrackChi2oNDF_";
      GoodTrackChi2oNDF = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, Chi2NDFBin, Chi2NDFMin, Chi2NDFMax);
      GoodTrackChi2oNDF->setAxisTitle("Good Track #chi^{2}/ndf",1);
      GoodTrackChi2oNDF->setAxisTitle("Number of Good Tracks"  ,2);

      histname = "GoodTrackChi2Prob_";
      GoodTrackChi2Prob = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, Chi2ProbBin, Chi2ProbMin, Chi2ProbMax);
      GoodTrackChi2Prob->setAxisTitle("Good Track #chi^{2} probability",1);
      GoodTrackChi2Prob->setAxisTitle("Number of Good Tracks"  ,2);

      histname = "GoodTrackChi2oNDFVsPhi_";
      GoodTrackChi2oNDFVsPhi = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, PhiBin, PhiMin, PhiMax, Chi2NDFMin, Chi2NDFMax);
      GoodTrackChi2oNDFVsPhi->setAxisTitle("Good Tracks #phi"  ,1);
      GoodTrackChi2oNDFVsPhi->setAxisTitle("Good Track #chi^{2}/ndf",2);

      histname = "GoodTrackChi2ProbVsPhi_";
      GoodTrackChi2ProbVsPhi = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, PhiBin, PhiMin, PhiMax, Chi2ProbMin, Chi2ProbMax);
      GoodTrackChi2ProbVsPhi->setAxisTitle("Good Tracks #phi"  ,1);
      GoodTrackChi2ProbVsPhi->setAxisTitle("Good Track #chi^{2} probability",2);

      histname = "GoodTrackChi2oNDFVsEta_";
      GoodTrackChi2oNDFVsEta = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax, Chi2NDFMin, Chi2NDFMax);
      GoodTrackChi2oNDFVsEta->setAxisTitle("Good Tracks #eta"  ,1);
      GoodTrackChi2oNDFVsEta->setAxisTitle("Good Track #chi^{2}/ndf",2);

      histname = "GoodTrackChi2ProbVsEta_";
      GoodTrackChi2ProbVsEta = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax, Chi2ProbMin, Chi2ProbMax);
      GoodTrackChi2ProbVsEta->setAxisTitle("Good Tracks #eta"  ,1);
      GoodTrackChi2ProbVsEta->setAxisTitle("Good Track #chi^{2} probability",2);

      histname = "GoodTrackAlgorithm_";
      GoodTrackAlgorithm = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, 32, 0., 32.);
      GoodTrackAlgorithm->setAxisTitle("Good Track tracking algorithm",1);
      GoodTrackAlgorithm->setAxisTitle("Number of Good Tracks",2);


      dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties/GoodTracks");

      histname = "GoodTrackNumberOfRecHitsPerTrack_";
      GoodTrackNumberOfRecHitsPerTrack = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TKHitBin, TKHitMin, TKHitMax);
      GoodTrackNumberOfRecHitsPerTrack->setAxisTitle("Number of all RecHits of each Good Track");
      GoodTrackNumberOfRecHitsPerTrack->setAxisTitle("Number of Good Tracks", 2);
      
      histname = "GoodTrackNumberOfRecHitsFoundPerTrack_";
      GoodTrackNumberOfFoundRecHitsPerTrack = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TKHitBin, TKHitMin, TKHitMax);
      GoodTrackNumberOfFoundRecHitsPerTrack->setAxisTitle("Number of found RecHits of each Good Track");
      GoodTrackNumberOfFoundRecHitsPerTrack->setAxisTitle("Number of Good Tracks", 2);
      
      if ( doGoodTrackRecHitVsPhiVsEtaPerTrack_ || doAllPlots_ ){
	
	histname = "GoodTrackNumberOfRecHitVsPhiVsEtaPerTrack_";
	GoodTrackNumberOfRecHitVsPhiVsEtaPerTrack = dqmStore_->bookProfile2D(histname+CatagoryName, histname+CatagoryName, 
									     EtaBin, EtaMin, EtaMax, PhiBin, PhiMin, PhiMax, 0, 40., "");
	GoodTrackNumberOfRecHitVsPhiVsEtaPerTrack->setAxisTitle("Good Track #eta ", 1);
	GoodTrackNumberOfRecHitVsPhiVsEtaPerTrack->setAxisTitle("Good Track #phi ", 2);
      }
      
      if ( doGoodTrackLayersVsPhiVsEtaPerTrack_ || doAllPlots_ ){
	
	histname = "GoodTrackNumberOfLayersVsPhiVsEtaPerTrack_";
	GoodTrackNumberOfLayersVsPhiVsEtaPerTrack = dqmStore_->bookProfile2D(histname+CatagoryName, histname+CatagoryName, 
									     EtaBin, EtaMin, EtaMax, PhiBin, PhiMin, PhiMax, 0, 40., "");
	GoodTrackNumberOfLayersVsPhiVsEtaPerTrack->setAxisTitle("Good Track #eta ", 1);
	GoodTrackNumberOfLayersVsPhiVsEtaPerTrack->setAxisTitle("Good Track #phi ", 2);
      }
      // rechits
      histname = "GoodTrackNumberOfRecHitsPerTrackVsPhiProfile_";
      GoodTrackNumberOfRecHitsPerTrackVsPhiProfile = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, PhiBin, PhiMin, PhiMax, RecHitMin, RecHitMax,"");
      GoodTrackNumberOfRecHitsPerTrackVsPhiProfile->setAxisTitle("Good Track #phi",1);
      GoodTrackNumberOfRecHitsPerTrackVsPhiProfile->setAxisTitle("Number of  RecHits of each Track",2);

      histname = "GoodTrackNumberOfRecHitsPerTrackVsEtaProfile_";
      GoodTrackNumberOfRecHitsPerTrackVsEtaProfile = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax, RecHitMin, RecHitMax,"");
      GoodTrackNumberOfRecHitsPerTrackVsEtaProfile->setAxisTitle("Good Track #eta",1);
      GoodTrackNumberOfRecHitsPerTrackVsEtaProfile->setAxisTitle("Number of  RecHits of each Track",2);

      histname = "GoodTrackNumberOfFoundRecHitsPerTrackVsPhiProfile_";
      GoodTrackNumberOfFoundRecHitsPerTrackVsPhiProfile = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, PhiBin, PhiMin, PhiMax, RecHitMin, RecHitMax,"");
      GoodTrackNumberOfFoundRecHitsPerTrackVsPhiProfile->setAxisTitle("Good Track #phi",1);
      GoodTrackNumberOfFoundRecHitsPerTrackVsPhiProfile->setAxisTitle("Number of found RecHits of each Track",2);

      histname = "GoodTrackNumberOfFoundRecHitsPerTrackVsEtaProfile_";
      GoodTrackNumberOfFoundRecHitsPerTrackVsEtaProfile = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax, RecHitMin, RecHitMax,"");
      GoodTrackNumberOfFoundRecHitsPerTrackVsEtaProfile->setAxisTitle("Good Track #eta",1);
      GoodTrackNumberOfFoundRecHitsPerTrackVsEtaProfile->setAxisTitle("Number of found RecHits of each Track",2);

    }
   
}

// -- Analyse
// ---------------------------------------------------------------------------------//
void TrackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Track& track)
{

  if ( doHitPropertiesPlots_ || doAllPlots_ ){
    // rec hits
    // originally it was track.recHitsSize(), but it is exploiting extra
    // therefore it is moved to track.hitPattern().numberOfHits()
    NumberOfRecHitsPerTrack->Fill(track.hitPattern().numberOfHits());
    NumberOfRecHitsFoundPerTrack->Fill(track.numberOfValidHits());
    NumberOfRecHitsLostPerTrack->Fill(track.numberOfLostHits());

    // 2D plots    
    if ( doRecHitVsPhiVsEtaPerTrack_ || doAllPlots_ )
      NumberOfRecHitVsPhiVsEtaPerTrack->Fill(track.eta(),track.phi(),track.hitPattern().numberOfHits());    

    // layers
    NumberOfLayersPerTrack->Fill(track.hitPattern().trackerLayersWithMeasurement());
    // 2D plots    
    if ( doLayersVsPhiVsEtaPerTrack_ || doAllPlots_ )
      NumberOfLayersVsPhiVsEtaPerTrack->Fill(track.eta(),track.phi(),track.hitPattern().trackerLayersWithMeasurement());


  }

  if (doGeneralPropertiesPlots_ || doAllPlots_){
    // fitting
    Chi2->Fill(track.chi2());
    Chi2Prob->Fill(TMath::Prob(track.chi2(),(int)track.ndof()));
    Chi2oNDF->Fill(track.normalizedChi2());

    // DCA
    // temporary patch in order to put back those MEs in Muon Workspace 
    if (doDCAwrt000Plots_) {
      DistanceOfClosestApproach->Fill(track.dxy());
      DistanceOfClosestApproachVsPhi->Fill(track.phi(), track.dxy());
    }
    /*
      http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/DataFormats/TrackReco/interface/TrackBase.h?view=markup
      vertex() is DEPRECATED !!
    xPointOfClosestApproach->Fill(track.vertex().x());
    yPointOfClosestApproach->Fill(track.vertex().y());
    zPointOfClosestApproach->Fill(track.vertex().z());
    */

    // PCA
    xPointOfClosestApproach->Fill(track.referencePoint().x());
    yPointOfClosestApproach->Fill(track.referencePoint().y());
    zPointOfClosestApproach->Fill(track.referencePoint().z());

    // algorithm
    algorithm->Fill(static_cast<double>(track.algo()));

  }

  if ( doLumiAnalysis_ ) {
    NumberOfRecHitsPerTrack_lumiFlag -> Fill(track.hitPattern().numberOfHits());    
    Chi2oNDF_lumiFlag->Fill(track.normalizedChi2());
  }

  if(doBSPlots_ || doAllPlots_)
    {
      edm::InputTag bsSrc = conf_.getParameter< edm::InputTag >("beamSpot");

      edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
        iEvent.getByLabel(bsSrc,recoBeamSpotHandle);
        reco::BeamSpot bs = *recoBeamSpotHandle;      

        DistanceOfClosestApproachToBS->Fill(track.dxy(bs.position()));
        DistanceOfClosestApproachToBSVsPhi->Fill(track.phi(), track.dxy(bs.position()));
	zPointOfClosestApproachVsPhi->Fill(track.phi(),track.vz());
        xPointOfClosestApproachVsZ0wrt000->Fill(track.dz(),track.vx());
        yPointOfClosestApproachVsZ0wrt000->Fill(track.dz(),track.vy());
	xPointOfClosestApproachVsZ0wrtBS->Fill(track.dz(bs.position()),(track.vx()-bs.position(track.vz()).x()));
	yPointOfClosestApproachVsZ0wrtBS->Fill(track.dz(bs.position()),(track.vy()-bs.position(track.vz()).y()));
	if (doTestPlots_) {
	  TESTDistanceOfClosestApproachToBS->Fill(track.dxy(bs.position(track.vz())));
	  TESTDistanceOfClosestApproachToBSVsPhi->Fill(track.phi(), track.dxy(bs.position(track.vz())));
	}
    }

    if(doDCAPlots_ || doAllPlots_)
      {
	if (doDCAwrt000Plots_) {
	  if (doThetaPlots_) {
	    DistanceOfClosestApproachVsTheta->Fill(track.theta(), track.d0());
	  }
	  DistanceOfClosestApproachVsEta->Fill(track.eta(), track.d0());
	}  
      }
    //Tracker Specific Histograms
    if(doTrackerSpecific_ || doAllPlots_) 
      {
        doTrackerSpecificFillHists(track);
      }

    if (doMeasurementStatePlots_ || doAllPlots_){
      std::string StateName = conf_.getParameter<std::string>("MeasurementState");
      if (StateName == "All") 
	{
	  fillHistosForState(iSetup, track, std::string("OuterSurface"));
	  fillHistosForState(iSetup, track, std::string("InnerSurface"));
	  fillHistosForState(iSetup, track, std::string("ImpactPoint"));
	} 
      else if 
	(   
	 StateName != "OuterSurface" && 
	 StateName != "InnerSurface" && 
	 StateName != "ImpactPoint" &&
	 StateName != "default" 
	 ) 
	{
	  fillHistosForState(iSetup, track, std::string("default"));
	}
      else
	{
	  fillHistosForState(iSetup, track, StateName);
	}
    }

    // Good Tracks plots
    if ( track.quality(reco::TrackBase::highPurity) && track.pt() > 1. ) {
      if ( doGoodTrackPlots_ || doAllPlots_ ) {
	GoodTrackChi2oNDF->Fill(track.normalizedChi2());
	GoodTrackChi2Prob->Fill(TMath::Prob(track.chi2(),(int)track.ndof()));
	GoodTrackChi2oNDFVsPhi->Fill(track.phi(),track.normalizedChi2());
	GoodTrackChi2ProbVsPhi->Fill(track.phi(),TMath::Prob(track.chi2(),(int)track.ndof()));
	GoodTrackChi2oNDFVsEta->Fill(track.eta(),track.normalizedChi2());
	GoodTrackChi2ProbVsEta->Fill(track.eta(),TMath::Prob(track.chi2(),(int)track.ndof()));

	// originally it was track.recHitsSize(), but it is exploiting extra
	// therefore it is moved to track.hitPattern().numberOfHits()
	GoodTrackNumberOfRecHitsPerTrack->Fill(track.hitPattern().numberOfHits());
	GoodTrackNumberOfFoundRecHitsPerTrack->Fill(track.numberOfValidHits());
	// algorithm
	GoodTrackAlgorithm->Fill(static_cast<double>(track.algo()));
        GoodTrackNumberOfFoundRecHitsPerTrackVsPhiProfile->Fill(track.phi(),track.numberOfValidHits());
        GoodTrackNumberOfFoundRecHitsPerTrackVsEtaProfile->Fill(track.eta(),track.numberOfValidHits());
        GoodTrackNumberOfRecHitsPerTrackVsPhiProfile->Fill(track.phi(),track.hitPattern().numberOfHits());
        GoodTrackNumberOfRecHitsPerTrackVsEtaProfile->Fill(track.eta(),track.hitPattern().numberOfHits());

	if ( doGoodTrackRecHitVsPhiVsEtaPerTrack_ || doAllPlots_ ) 
	  GoodTrackNumberOfRecHitVsPhiVsEtaPerTrack->Fill(track.eta(),track.phi(),track.hitPattern().numberOfHits());

	if ( doGoodTrackLayersVsPhiVsEtaPerTrack_ || doAllPlots_ ) 
	  GoodTrackNumberOfLayersVsPhiVsEtaPerTrack->Fill(track.eta(),track.phi(),track.hitPattern().trackerLayersWithMeasurement());
      }
      if ( doLumiAnalysis_ ) {
	GoodTrackChi2oNDF_lumiFlag                -> Fill(track.normalizedChi2()); 
	GoodTrackNumberOfRecHitsPerTrack_lumiFlag -> Fill(track.hitPattern().numberOfHits());
      }
    }

}


// book histograms at differnt measurement points
// ---------------------------------------------------------------------------------//
void TrackAnalyzer::bookHistosForState(std::string sname, DQMStore * dqmStore_) 
{


    // parameters from the configuration
    std::string QualName       = conf_.getParameter<std::string>("Quality");
    std::string AlgoName       = conf_.getParameter<std::string>("AlgoName");
    std::string MEFolderName   = conf_.getParameter<std::string>("FolderName"); 

    // use the AlgoName and Quality Name 
    std::string CatagoryName = QualName != "" ? AlgoName + "_" + QualName : AlgoName;

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


    dqmStore_->setCurrentFolder(MEFolderName);

    TkParameterMEs tkmes;

    std::string histTag = (sname == "default") ? CatagoryName : sname + "_" + CatagoryName;

    if(doAllPlots_)
    {

	//	  COMMENTED BEACUSE THERE IS ALREADY THE PROFILE !!! (blablaProfile)
	/*
        // hit properties
        dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties");
        // rechits
        histname = "NumberOfRecHitsPerTrackVsPhi_" + histTag;
        tkmes.NumberOfRecHitsPerTrackVsPhi = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, RecHitMin, RecHitMax,"");
        tkmes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Track #phi",1);
        tkmes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Number of found RecHits of each Track",2);

	if (doThetaPlots_) {
	  histname = "NumberOfRecHitsPerTrackVsTheta_" + histTag;
	  tkmes.NumberOfRecHitsPerTrackVsTheta = dqmStore_->bookProfile(histname, histname, ThetaBin, ThetaMin, ThetaMax, RecHitMin, RecHitMax,"");
	  tkmes.NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Track #theta",1);
	  tkmes.NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Number of found RecHits of each Track",2);
	}
        histname = "NumberOfRecHitsPerTrackVsEta_" + histTag;
        tkmes.NumberOfRecHitsPerTrackVsEta = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, RecHitMin, RecHitMax,"");
        tkmes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Track #eta",1);
        tkmes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Number of found RecHits of each Track",2);

        histname = "NumberOfLayersPerTrackVsPhi_" + histTag;
        tkmes.NumberOfLayersPerTrackVsPhi = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, RecLayMin, RecLayMax,"");
        tkmes.NumberOfLayersPerTrackVsPhi->setAxisTitle("Track #phi",1);
        tkmes.NumberOfLayersPerTrackVsPhi->setAxisTitle("Number of Layers of each Track",2);

	if (doThetaPlots_) {
	  histname = "NumberOfLayersPerTrackVsTheta_" + histTag;
	  tkmes.NumberOfLayersPerTrackVsTheta = dqmStore_->bookProfile(histname, histname, ThetaBin, ThetaMin, ThetaMax, RecLayMin, RecLayMax,"");
	  tkmes.NumberOfLayersPerTrackVsTheta->setAxisTitle("Track #theta",1);
	  tkmes.NumberOfLayersPerTrackVsTheta->setAxisTitle("Number of Layers of each Track",2);
	}
        histname = "NumberOfLayersPerTrackVsEta_" + histTag;
        tkmes.NumberOfLayersPerTrackVsEta = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, RecLayMin, RecLayMax,"");
        tkmes.NumberOfLayersPerTrackVsEta->setAxisTitle("Track #eta",1);
        tkmes.NumberOfLayersPerTrackVsEta->setAxisTitle("Number of Layers of each Track",2);
	*/

        // general properties
        dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");

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
    }

    // general properties
    dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");

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
    tkmes.TrackPErr->setAxisTitle("error(p)/p", 1);
    tkmes.TrackPErr->setAxisTitle("Number of Tracks",2);

    histname = "TrackPtErrOverPt_" + histTag;
    tkmes.TrackPtErr = dqmStore_->book1D(histname, histname, ptErrBin, ptErrMin, ptErrMax);
    tkmes.TrackPtErr->setAxisTitle("error(p_{T})/p_{T}", 1);
    tkmes.TrackPtErr->setAxisTitle("Number of Tracks",2);

    histname = "TrackPtErrOverPtVsEta_" + histTag;
    tkmes.TrackPtErrVsEta = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, ptErrMin, ptErrMax);
    tkmes.TrackPtErrVsEta->setAxisTitle("Track #eta",1);
    tkmes.TrackPtErrVsEta->setAxisTitle("error(p_{T})/p_{T}", 2);

    if (doTrackPxPyPlots_) {
      histname = "TrackPxErrOverPx_" + histTag;
      tkmes.TrackPxErr = dqmStore_->book1D(histname, histname, pxErrBin, pxErrMin, pxErrMax);
      tkmes.TrackPxErr->setAxisTitle("error(p_{x})/p_{x}", 1);
      tkmes.TrackPxErr->setAxisTitle("Number of Tracks",2);
      
      histname = "TrackPyErrOverPy_" + histTag;
      tkmes.TrackPyErr = dqmStore_->book1D(histname, histname, pyErrBin, pyErrMin, pyErrMax);
      tkmes.TrackPyErr->setAxisTitle("error(p_{y})/p_{y}", 1);
      tkmes.TrackPyErr->setAxisTitle("Number of Tracks",2);
    }
    histname = "TrackPzErrOverPz_" + histTag;
    tkmes.TrackPzErr = dqmStore_->book1D(histname, histname, pzErrBin, pzErrMin, pzErrMax);
    tkmes.TrackPzErr->setAxisTitle("error(p_{z})/p_{z}", 1);
    tkmes.TrackPzErr->setAxisTitle("Number of Tracks",2);

    histname = "TrackPhiErr_" + histTag;
    tkmes.TrackPhiErr = dqmStore_->book1D(histname, histname, phiErrBin, phiErrMin, phiErrMax);
    tkmes.TrackPhiErr->setAxisTitle("error(#phi)");
    tkmes.TrackPhiErr->setAxisTitle("Number of Tracks",2);

    histname = "TrackEtaErr_" + histTag;
    tkmes.TrackEtaErr = dqmStore_->book1D(histname, histname, etaErrBin, etaErrMin, etaErrMax);
    tkmes.TrackEtaErr->setAxisTitle("error(#eta)");
    tkmes.TrackEtaErr->setAxisTitle("Number of Tracks",2);

    // rec hit profiles
    histname = "NumberOfRecHitsPerTrackVsPhiProfile_" + histTag;
    tkmes.NumberOfRecHitsPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, RecHitBin, RecHitMin, RecHitMax,"");
    tkmes.NumberOfRecHitsPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    tkmes.NumberOfRecHitsPerTrackVsPhiProfile->setAxisTitle("Number of found RecHits of each Track",2);

    if (doThetaPlots_) {
      histname = "NumberOfRecHitsPerTrackVsThetaProfile_" + histTag;
      tkmes.NumberOfRecHitsPerTrackVsThetaProfile = dqmStore_->bookProfile(histname, histname, ThetaBin, ThetaMin, ThetaMax, RecHitBin, RecHitMin, RecHitMax,"");
      tkmes.NumberOfRecHitsPerTrackVsThetaProfile->setAxisTitle("Track #phi",1);
      tkmes.NumberOfRecHitsPerTrackVsThetaProfile->setAxisTitle("Number of found RecHits of each Track",2);
    }
    histname = "NumberOfRecHitsPerTrackVsEtaProfile_" + histTag;
    tkmes.NumberOfRecHitsPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, RecHitBin, RecHitMin, RecHitMax,"");
    tkmes.NumberOfRecHitsPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    tkmes.NumberOfRecHitsPerTrackVsEtaProfile->setAxisTitle("Number of found RecHits of each Track",2);

    histname = "NumberOfLayersPerTrackVsPhiProfile_" + histTag;
    tkmes.NumberOfLayersPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, RecLayBin, RecLayMin, RecLayMax,"");
    tkmes.NumberOfLayersPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    tkmes.NumberOfLayersPerTrackVsPhiProfile->setAxisTitle("Number of Layers of each Track",2);

    if (doThetaPlots_) {
      histname = "NumberOfLayersPerTrackVsThetaProfile_" + histTag;
      tkmes.NumberOfLayersPerTrackVsThetaProfile = dqmStore_->bookProfile(histname, histname, ThetaBin, ThetaMin, ThetaMax, RecLayBin, RecLayMin, RecLayMax,"");
      tkmes.NumberOfLayersPerTrackVsThetaProfile->setAxisTitle("Track #phi",1);
      tkmes.NumberOfLayersPerTrackVsThetaProfile->setAxisTitle("Number of Layers of each Track",2);
    }
    histname = "NumberOfLayersPerTrackVsEtaProfile_" + histTag;
    tkmes.NumberOfLayersPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, RecLayBin, RecLayMin, RecLayMax,"");
    tkmes.NumberOfLayersPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    tkmes.NumberOfLayersPerTrackVsEtaProfile->setAxisTitle("Number of Layers of each Track",2);

    if ( doGoodTrackPlots_ || doAllPlots_ ) {

      dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties/GoodTracks");

      histname = "GoodTrackPt_" + histTag;
      tkmes.GoodTrackPt = dqmStore_->book1D(histname, histname, TrackPtBin, TrackPtMin, TrackPtMax);
      tkmes.GoodTrackPt->setAxisTitle("Good Track p_{T} (GeV/c)", 1);
      tkmes.GoodTrackPt->setAxisTitle("Number of Tracks",2);
      
      histname = "GoodTrackEta_" + histTag;
      tkmes.GoodTrackEta = dqmStore_->book1D(histname, histname, EtaBin, EtaMin, EtaMax);
      tkmes.GoodTrackEta->setAxisTitle("Good Track #eta", 1);
      tkmes.GoodTrackEta->setAxisTitle("Number of Tracks",2);
      
      histname = "GoodTrackPhi_" + histTag;
      tkmes.GoodTrackPhi = dqmStore_->book1D(histname, histname, PhiBin, PhiMin, PhiMax);
      tkmes.GoodTrackPhi->setAxisTitle("Good Track #phi", 1);
      tkmes.GoodTrackPhi->setAxisTitle("Number of Tracks",2);

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

    bool isHighPurity = track.quality(reco::TrackBase::highPurity);

    if (sname == "default") 
    {
        p     = track.p();
        px    = track.px();
        py    = track.py();
        pz    = track.pz();
        pt    = track.pt();
        phi   = track.phi();
        theta = track.theta();
        eta   = track.eta();
        q     = track.charge();

	//        pterror  = (pt) ? track.ptError()/pt : 0.0;
        pterror  = (pt) ? track.ptError()/(pt*pt) : 0.0;
        pxerror  = -1.0;
        pyerror  = -1.0;
        pzerror  = -1.0;
        perror   = -1.0;
        phierror = track.phiError();
        etaerror = track.etaError();

    } 
    else 
    {
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
    if (iPos != TkParameterMEMap.end()) 
    {
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

        // rec hits 
        tkmes.NumberOfRecHitsPerTrackVsPhiProfile->Fill(phi,    track.hitPattern().numberOfValidHits());
	if (doThetaPlots_) {
	  tkmes.NumberOfRecHitsPerTrackVsThetaProfile->Fill(theta,track.hitPattern().numberOfValidHits());
	}
        tkmes.NumberOfRecHitsPerTrackVsEtaProfile->Fill(eta,    track.hitPattern().numberOfValidHits());


        // rec layers 
        tkmes.NumberOfLayersPerTrackVsPhiProfile->Fill(phi,     track.hitPattern().stripLayersWithMeasurement());
	if (doThetaPlots_) {
	  tkmes.NumberOfLayersPerTrackVsThetaProfile->Fill(theta, track.hitPattern().stripLayersWithMeasurement());
	}
        tkmes.NumberOfLayersPerTrackVsEtaProfile->Fill(eta,     track.hitPattern().stripLayersWithMeasurement());

        if(doAllPlots_)
        {
	  /*
	    // COMMENTED because the same MEs are already fill below !!!!
            // hit related
            tkmes.NumberOfRecHitsPerTrackVsPhi->Fill(phi, track.found());
	    if (doThetaPlots_) {
	      tkmes.NumberOfRecHitsPerTrackVsTheta->Fill(theta, track.found());
	    }
            tkmes.NumberOfRecHitsPerTrackVsEta->Fill(eta, track.found());
	  */

            // general properties
	    if (doThetaPlots_) {
	      tkmes.Chi2oNDFVsTheta->Fill(theta, track.normalizedChi2());
	    }
            tkmes.Chi2oNDFVsPhi->Fill(phi, track.normalizedChi2());
            tkmes.Chi2oNDFVsEta->Fill(eta, track.normalizedChi2());

	    // COMMENTED because there are already those quantity in blablaProfile !!
	    /*
            // rec hits 
            tkmes.NumberOfRecHitsPerTrackVsPhi->Fill(phi,        track.hitPattern().numberOfValidHits());
	    if (doThetaPlots_) {
	      tkmes.NumberOfRecHitsPerTrackVsTheta->Fill(theta,    track.hitPattern().numberOfValidHits());
	    }
            tkmes.NumberOfRecHitsPerTrackVsEta->Fill(eta,        track.hitPattern().numberOfValidHits());
	    */	    

        }

	if ( doGoodTrackPlots_ || doAllPlots_ ) {
	  if ( isHighPurity && pt > 1. ) {
	    tkmes.GoodTrackPt->Fill(pt);
	    tkmes.GoodTrackEta->Fill(eta);
	    tkmes.GoodTrackPhi->Fill(phi);
	  }
	}

    }
}


void TrackAnalyzer::doTrackerSpecificInitialization(DQMStore * dqmStore_) 
{

    // parameters from the configuration
    std::string QualName     = conf_.getParameter<std::string>("Quality");
    std::string AlgoName     = conf_.getParameter<std::string>("AlgoName");
    std::string MEFolderName = conf_.getParameter<std::string>("FolderName"); 

    // use the AlgoName and Quality Name 
    std::string CatagoryName = QualName != "" ? AlgoName + "_" + QualName : AlgoName;

    // get binning from the configuration
    int    TOBHitBin = conf_.getParameter<int>(   "TOBHitBin");
    double TOBHitMin = conf_.getParameter<double>("TOBHitMin");
    double TOBHitMax = conf_.getParameter<double>("TOBHitMax");

    int    TIBHitBin = conf_.getParameter<int>(   "TIBHitBin");
    double TIBHitMin = conf_.getParameter<double>("TIBHitMin");
    double TIBHitMax = conf_.getParameter<double>("TIBHitMax");

    int    TIDHitBin = conf_.getParameter<int>(   "TIDHitBin");
    double TIDHitMin = conf_.getParameter<double>("TIDHitMin");
    double TIDHitMax = conf_.getParameter<double>("TIDHitMax");

    int    TECHitBin = conf_.getParameter<int>(   "TECHitBin");
    double TECHitMin = conf_.getParameter<double>("TECHitMin");
    double TECHitMax = conf_.getParameter<double>("TECHitMax");

    int    PXBHitBin = conf_.getParameter<int>(   "PXBHitBin");
    double PXBHitMin = conf_.getParameter<double>("PXBHitMin");
    double PXBHitMax = conf_.getParameter<double>("PXBHitMax");

    int    PXFHitBin = conf_.getParameter<int>(   "PXFHitBin");
    double PXFHitMin = conf_.getParameter<double>("PXFHitMin");
    double PXFHitMax = conf_.getParameter<double>("PXFHitMax");

    int    TOBLayBin = conf_.getParameter<int>(   "TOBLayBin");
    double TOBLayMin = conf_.getParameter<double>("TOBLayMin");
    double TOBLayMax = conf_.getParameter<double>("TOBLayMax");

    int    TIBLayBin = conf_.getParameter<int>(   "TIBLayBin");
    double TIBLayMin = conf_.getParameter<double>("TIBLayMin");
    double TIBLayMax = conf_.getParameter<double>("TIBLayMax");

    int    TIDLayBin = conf_.getParameter<int>(   "TIDLayBin");
    double TIDLayMin = conf_.getParameter<double>("TIDLayMin");
    double TIDLayMax = conf_.getParameter<double>("TIDLayMax");

    int    TECLayBin = conf_.getParameter<int>(   "TECLayBin");
    double TECLayMin = conf_.getParameter<double>("TECLayMin");
    double TECLayMax = conf_.getParameter<double>("TECLayMax");

    int    PXBLayBin = conf_.getParameter<int>(   "PXBLayBin");
    double PXBLayMin = conf_.getParameter<double>("PXBLayMin");
    double PXBLayMax = conf_.getParameter<double>("PXBLayMax");

    int    PXFLayBin = conf_.getParameter<int>(   "PXFLayBin");
    double PXFLayMin = conf_.getParameter<double>("PXFLayMin");
    double PXFLayMax = conf_.getParameter<double>("PXFLayMax");


    int    PhiBin     = conf_.getParameter<int>(   "PhiBin");
    double PhiMin     = conf_.getParameter<double>("PhiMin");
    double PhiMax     = conf_.getParameter<double>("PhiMax");

    int    EtaBin     = conf_.getParameter<int>(   "EtaBin");
    double EtaMin     = conf_.getParameter<double>("EtaMin");
    double EtaMax     = conf_.getParameter<double>("EtaMax");

    // book hit property histograms
    // ---------------------------------------------------------------------------------//
    dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties");


    TkParameterMEs tkmes;

    // TOB hits properties
    dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties/TOB");

    histname = "NumberOfTOBRecHitsPerTrack_" + CatagoryName;
    NumberOfTOBRecHitsPerTrack = dqmStore_->book1D(histname, histname, TOBHitBin, TOBHitMin, TOBHitMax);
    NumberOfTOBRecHitsPerTrack->setAxisTitle("Number of TOB found RecHits of each Track",1);
    NumberOfTOBRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

    histname = "NumberOfTOBRecHitsPerTrackVsPhiProfile_" + CatagoryName;
    NumberOfTOBRecHitsPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, TOBHitBin, TOBHitMin, TOBHitMax,"");
    NumberOfTOBRecHitsPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    NumberOfTOBRecHitsPerTrackVsPhiProfile->setAxisTitle("Number of TOB found RecHits of each Track",2);

    histname = "NumberOfTOBRecHitsPerTrackVsEtaProfile_" + CatagoryName;
    NumberOfTOBRecHitsPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, TOBHitBin, TOBHitMin, TOBHitMax,"");
    NumberOfTOBRecHitsPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    NumberOfTOBRecHitsPerTrackVsEtaProfile->setAxisTitle("Number of TOB found RecHits of each Track",2);

    histname = "NumberOfTOBLayersPerTrack_" + CatagoryName;
    NumberOfTOBLayersPerTrack = dqmStore_->book1D(histname, histname, TOBLayBin, TOBLayMin, TOBLayMax);
    NumberOfTOBLayersPerTrack->setAxisTitle("Number of TOB Layers of each Track",1);
    NumberOfTOBLayersPerTrack->setAxisTitle("Number of Tracks", 2);

    histname = "NumberOfTOBLayersPerTrackVsPhiProfile_" + CatagoryName;
    NumberOfTOBLayersPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, TOBLayBin, TOBLayMin, TOBLayMax,"");
    NumberOfTOBLayersPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    NumberOfTOBLayersPerTrackVsPhiProfile->setAxisTitle("Number of TOB Layers of each Track",2);

    histname = "NumberOfTOBLayersPerTrackVsEtaProfile_" + CatagoryName;
    NumberOfTOBLayersPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, TOBLayBin, TOBLayMin, TOBLayMax,"");
    NumberOfTOBLayersPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    NumberOfTOBLayersPerTrackVsEtaProfile->setAxisTitle("Number of TOB Layers of each Track",2);


    // TIB hits properties
    dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties/TIB");

    histname = "NumberOfTIBRecHitsPerTrack_" + CatagoryName;
    NumberOfTIBRecHitsPerTrack = dqmStore_->book1D(histname, histname, TIBHitBin, TIBHitMin, TIBHitMax);
    NumberOfTIBRecHitsPerTrack->setAxisTitle("Number of TIB found RecHits of each Track",1);
    NumberOfTIBRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

    histname = "NumberOfTIBRecHitsPerTrackVsPhiProfile_" + CatagoryName;
    NumberOfTIBRecHitsPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, TIBHitBin, TIBHitMin, TIBHitMax,"");
    NumberOfTIBRecHitsPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    NumberOfTIBRecHitsPerTrackVsPhiProfile->setAxisTitle("Number of TIB found RecHits of each Track",2);

    histname = "NumberOfTIBRecHitsPerTrackVsEtaProfile_" + CatagoryName;
    NumberOfTIBRecHitsPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, TIBHitBin, TIBHitMin, TIBHitMax,"");
    NumberOfTIBRecHitsPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    NumberOfTIBRecHitsPerTrackVsEtaProfile->setAxisTitle("Number of TIB found RecHits of each Track",2);

    histname = "NumberOfTIBLayersPerTrack_" + CatagoryName;
    NumberOfTIBLayersPerTrack = dqmStore_->book1D(histname, histname, TIBLayBin, TIBLayMin, TIBLayMax);
    NumberOfTIBLayersPerTrack->setAxisTitle("Number of TIB Layers of each Track",1);
    NumberOfTIBLayersPerTrack->setAxisTitle("Number of Tracks", 2);

    histname = "NumberOfTIBLayersPerTrackVsPhiProfile_" + CatagoryName;
    NumberOfTIBLayersPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, TIBLayBin, TIBLayMin, TIBLayMax,"");
    NumberOfTIBLayersPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    NumberOfTIBLayersPerTrackVsPhiProfile->setAxisTitle("Number of TIB Layers of each Track",2);

    histname = "NumberOfTIBLayersPerTrackVsEtaProfile_" + CatagoryName;
    NumberOfTIBLayersPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, TIBLayBin, TIBLayMin, TIBLayMax,"");
    NumberOfTIBLayersPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    NumberOfTIBLayersPerTrackVsEtaProfile->setAxisTitle("Number of TIB Layers of each Track",2);


    // TID hit properties
    dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties/TID");

    histname = "NumberOfTIDRecHitsPerTrack_" + CatagoryName;
    NumberOfTIDRecHitsPerTrack = dqmStore_->book1D(histname, histname, TIDHitBin, TIDHitMin, TIDHitMax);
    NumberOfTIDRecHitsPerTrack->setAxisTitle("Number of TID found RecHits of each Track",1);
    NumberOfTIDRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

    histname = "NumberOfTIDRecHitsPerTrackVsPhiProfile_" + CatagoryName;
    NumberOfTIDRecHitsPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, TIDHitBin, TIDHitMin, TIDHitMax,"");
    NumberOfTIDRecHitsPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    NumberOfTIDRecHitsPerTrackVsPhiProfile->setAxisTitle("Number of TID found RecHits of each Track",2);

    histname = "NumberOfTIDRecHitsPerTrackVsEtaProfile_" + CatagoryName;
    NumberOfTIDRecHitsPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, TIDHitBin, TIDHitMin, TIDHitMax,"");
    NumberOfTIDRecHitsPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    NumberOfTIDRecHitsPerTrackVsEtaProfile->setAxisTitle("Number of TID found RecHits of each Track",2);

    histname = "NumberOfTIDLayersPerTrack_" + CatagoryName;
    NumberOfTIDLayersPerTrack = dqmStore_->book1D(histname, histname, TIDLayBin, TIDLayMin, TIDLayMax);
    NumberOfTIDLayersPerTrack->setAxisTitle("Number of TID Layers of each Track",1);
    NumberOfTIDLayersPerTrack->setAxisTitle("Number of Tracks", 2);

    histname = "NumberOfTIDLayersPerTrackVsPhiProfile_" + CatagoryName;
    NumberOfTIDLayersPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, TIDLayBin, TIDLayMin, TIDLayMax,"");
    NumberOfTIDLayersPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    NumberOfTIDLayersPerTrackVsPhiProfile->setAxisTitle("Number of TID Layers of each Track",2);

    histname = "NumberOfTIDLayersPerTrackVsEtaProfile_" + CatagoryName;
    NumberOfTIDLayersPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, TIDLayBin, TIDLayMin, TIDLayMax,"");
    NumberOfTIDLayersPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    NumberOfTIDLayersPerTrackVsEtaProfile->setAxisTitle("Number of TID Layers of each Track",2);


    // TEC hits properties    
    dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties/TEC");

    histname = "NumberOfTECRecHitsPerTrack_"+ CatagoryName;
    NumberOfTECRecHitsPerTrack = dqmStore_->book1D(histname, histname, TECHitBin, TECHitMin, TECHitMax);
    NumberOfTECRecHitsPerTrack->setAxisTitle("Number of TEC found RecHits of each Track",1);
    NumberOfTECRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

    histname = "NumberOfTECRecHitsPerTrackVsPhiProfile_" + CatagoryName;
    NumberOfTECRecHitsPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, TECHitBin, TECHitMin, TECHitMax,"");
    NumberOfTECRecHitsPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    NumberOfTECRecHitsPerTrackVsPhiProfile->setAxisTitle("Number of TEC found RecHits of each Track",2);

    histname = "NumberOfTECRecHitsPerTrackVsEtaProfile_" + CatagoryName;
    NumberOfTECRecHitsPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, TECHitBin, TECHitMin, TECHitMax,"");
    NumberOfTECRecHitsPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    NumberOfTECRecHitsPerTrackVsEtaProfile->setAxisTitle("Number of TEC found RecHits of each Track",2);

    histname = "NumberOfTECLayersPerTrack_"+ CatagoryName;
    NumberOfTECLayersPerTrack = dqmStore_->book1D(histname, histname, TECLayBin, TECLayMin, TECLayMax);
    NumberOfTECLayersPerTrack->setAxisTitle("Number of TEC Layers of each Track",1);
    NumberOfTECLayersPerTrack->setAxisTitle("Number of Tracks", 2);

    histname = "NumberOfTECLayersPerTrackVsPhiProfile_" + CatagoryName;
    NumberOfTECLayersPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, TECLayBin, TECLayMin, TECLayMax,"");
    NumberOfTECLayersPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    NumberOfTECLayersPerTrackVsPhiProfile->setAxisTitle("Number of TEC Layers of each Track",2);

    histname = "NumberOfTECLayersPerTrackVsEtaProfile_" + CatagoryName;
    NumberOfTECLayersPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, TECLayBin, TECLayMin, TECLayMax,"");
    NumberOfTECLayersPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    NumberOfTECLayersPerTrackVsEtaProfile->setAxisTitle("Number of TEC Layers of each Track",2);


    // PixBarrel hits properties
    dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties/PixBarrel");

    histname = "NumberOfPixBarrelRecHitsPerTrack_" + CatagoryName;
    NumberOfPixBarrelRecHitsPerTrack = dqmStore_->book1D(histname, histname, PXBHitBin, PXBHitMin, PXBHitMax);
    NumberOfPixBarrelRecHitsPerTrack->setAxisTitle("Number of Pixel Barrel found RecHits of each Track",1);
    NumberOfPixBarrelRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

    histname = "NumberOfPixBarrelRecHitsPerTrackVsPhiProfile_" + CatagoryName;
    NumberOfPixBarrelRecHitsPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, PXBHitBin, PXBHitMin, PXBHitMax,"");
    NumberOfPixBarrelRecHitsPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    NumberOfPixBarrelRecHitsPerTrackVsPhiProfile->setAxisTitle("Number of Pixel Barrel found RecHits of each Track",2);

    histname = "NumberOfPixBarrelRecHitsPerTrackVsEtaProfile_" + CatagoryName;
    NumberOfPixBarrelRecHitsPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, PXBHitBin, PXBHitMin, PXBHitMax,"");
    NumberOfPixBarrelRecHitsPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    NumberOfPixBarrelRecHitsPerTrackVsEtaProfile->setAxisTitle("Number of Pixel Barrel found RecHits of each Track",2);

    histname = "NumberOfPixBarrelLayersPerTrack_" + CatagoryName;
    NumberOfPixBarrelLayersPerTrack = dqmStore_->book1D(histname, histname, PXBLayBin, PXBLayMin, PXBLayMax);
    NumberOfPixBarrelLayersPerTrack->setAxisTitle("Number of Pixel Barrel Layers of each Track",1);
    NumberOfPixBarrelLayersPerTrack->setAxisTitle("Number of Tracks", 2);

    histname = "NumberOfPixBarrelLayersPerTrackVsPhiProfile_" + CatagoryName;
    NumberOfPixBarrelLayersPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, PXBLayBin, PXBLayMin, PXBLayMax,"");
    NumberOfPixBarrelLayersPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    NumberOfPixBarrelLayersPerTrackVsPhiProfile->setAxisTitle("Number of Pixel Barrel Layers of each Track",2);

    histname = "NumberOfPixBarrelLayersPerTrackVsEtaProfile_" + CatagoryName;
    NumberOfPixBarrelLayersPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, PXBLayBin, PXBLayMin, PXBLayMax,"");
    NumberOfPixBarrelLayersPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    NumberOfPixBarrelLayersPerTrackVsEtaProfile->setAxisTitle("Number of Pixel Barrel Layers of each Track",2);


    // PixEndcap hits profiles
    dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties/PixEndcap");

    histname = "NumberOfPixEndcapRecHitsPerTrack_" + CatagoryName;
    NumberOfPixEndcapRecHitsPerTrack = dqmStore_->book1D(histname, histname, PXFHitBin, PXFHitMin, PXFHitMax);
    NumberOfPixEndcapRecHitsPerTrack->setAxisTitle("Number of Pixel Endcap found RecHits of each Track",1);
    NumberOfPixEndcapRecHitsPerTrack->setAxisTitle("Number of Tracks", 2);

    histname = "NumberOfPixEndcapRecHitsPerTrackVsPhiProfile_" + CatagoryName;
    NumberOfPixEndcapRecHitsPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, PXBHitBin, PXBHitMin, PXBHitMax,"");
    NumberOfPixEndcapRecHitsPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    NumberOfPixEndcapRecHitsPerTrackVsPhiProfile->setAxisTitle("Number of Pixel Endcap found RecHits of each Track",2);

    histname = "NumberOfPixEndcapRecHitsPerTrackVsEtaProfile_" + CatagoryName;
    NumberOfPixEndcapRecHitsPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, PXBHitBin, PXBHitMin, PXBHitMax,"");
    NumberOfPixEndcapRecHitsPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    NumberOfPixEndcapRecHitsPerTrackVsEtaProfile->setAxisTitle("Number of Pixel Endcap found RecHits of each Track",2);

    histname = "NumberOfPixEndcapLayersPerTrack_" + CatagoryName;
    NumberOfPixEndcapLayersPerTrack = dqmStore_->book1D(histname, histname, PXFLayBin, PXFLayMin, PXFLayMax);
    NumberOfPixEndcapLayersPerTrack->setAxisTitle("Number of Pixel Endcap Layers of each Track",1);
    NumberOfPixEndcapLayersPerTrack->setAxisTitle("Number of Tracks", 2);

    histname = "NumberOfPixEndcapLayersPerTrackVsPhiProfile_" + CatagoryName;
    NumberOfPixEndcapLayersPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, PXBLayBin, PXBLayMin, PXBLayMax,"");
    NumberOfPixEndcapLayersPerTrackVsPhiProfile->setAxisTitle("Track #phi",1);
    NumberOfPixEndcapLayersPerTrackVsPhiProfile->setAxisTitle("Number of Pixel Endcap Layers of each Track",2);

    histname = "NumberOfPixEndcapLayersPerTrackVsEtaProfile_" + CatagoryName;
    NumberOfPixEndcapLayersPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, PXBLayBin, PXBLayMin, PXBLayMax,"");
    NumberOfPixEndcapLayersPerTrackVsEtaProfile->setAxisTitle("Track #eta",1);
    NumberOfPixEndcapLayersPerTrackVsEtaProfile->setAxisTitle("Number of Pixel Endcap Layers of each Track",2);

}


void TrackAnalyzer::doTrackerSpecificFillHists(const reco::Track & track) 
{
  double phi   = track.phi();
  double eta   = track.eta();

  //Fill TIB Layers and RecHits
  NumberOfTIBRecHitsPerTrack->Fill(track.hitPattern().numberOfValidStripTIBHits()); 
  NumberOfTIBRecHitsPerTrackVsPhiProfile->Fill(phi,    track.hitPattern().numberOfValidStripTIBHits());
  NumberOfTIBRecHitsPerTrackVsEtaProfile->Fill(eta,    track.hitPattern().numberOfValidStripTIBHits());

  NumberOfTIBLayersPerTrack->Fill(track.hitPattern().stripTIBLayersWithMeasurement());
  NumberOfTIBLayersPerTrackVsPhiProfile->Fill(phi,     track.hitPattern().stripTIBLayersWithMeasurement());
  NumberOfTIBLayersPerTrackVsEtaProfile->Fill(eta,     track.hitPattern().stripTIBLayersWithMeasurement());

  //Fill TOB Layers and RecHits
  NumberOfTOBRecHitsPerTrack->Fill(track.hitPattern().numberOfValidStripTOBHits());
  NumberOfTOBRecHitsPerTrackVsPhiProfile->Fill(phi,    track.hitPattern().numberOfValidStripTOBHits());
  NumberOfTOBRecHitsPerTrackVsEtaProfile->Fill(eta,    track.hitPattern().numberOfValidStripTOBHits());

  NumberOfTOBLayersPerTrack->Fill(track.hitPattern().stripTOBLayersWithMeasurement());
  NumberOfTOBLayersPerTrackVsPhiProfile->Fill(phi,     track.hitPattern().stripTOBLayersWithMeasurement());
  NumberOfTOBLayersPerTrackVsEtaProfile->Fill(eta,     track.hitPattern().stripTOBLayersWithMeasurement());
  

  //Fill TID Layers and RecHits
  NumberOfTIDRecHitsPerTrack->Fill(track.hitPattern().numberOfValidStripTIDHits());
  NumberOfTIDRecHitsPerTrackVsPhiProfile->Fill(phi,    track.hitPattern().numberOfValidStripTIDHits());
  NumberOfTIDRecHitsPerTrackVsEtaProfile->Fill(eta,    track.hitPattern().numberOfValidStripTIDHits());

  NumberOfTIDLayersPerTrack->Fill(track.hitPattern().stripTIDLayersWithMeasurement());
  NumberOfTIDLayersPerTrackVsPhiProfile->Fill(phi,     track.hitPattern().stripTIDLayersWithMeasurement());
  NumberOfTIDLayersPerTrackVsEtaProfile->Fill(eta,     track.hitPattern().stripTIDLayersWithMeasurement());

  
  //Fill TEC Layers and RecHits
  NumberOfTECRecHitsPerTrack->Fill(track.hitPattern().numberOfValidStripTECHits());
  NumberOfTECRecHitsPerTrackVsPhiProfile->Fill(phi,     track.hitPattern().numberOfValidStripTECHits());
  NumberOfTECRecHitsPerTrackVsEtaProfile->Fill(eta,     track.hitPattern().numberOfValidStripTECHits());

  NumberOfTECLayersPerTrack->Fill(track.hitPattern().stripTECLayersWithMeasurement());
  NumberOfTECLayersPerTrackVsPhiProfile->Fill(phi,     track.hitPattern().stripTECLayersWithMeasurement());
  NumberOfTECLayersPerTrackVsEtaProfile->Fill(eta,     track.hitPattern().stripTECLayersWithMeasurement());

  //Fill PixBarrel Layers and RecHits
  NumberOfPixBarrelRecHitsPerTrack->Fill(track.hitPattern().numberOfValidPixelBarrelHits());
  NumberOfPixBarrelRecHitsPerTrackVsPhiProfile->Fill(phi,    track.hitPattern().numberOfValidPixelBarrelHits());
  NumberOfPixBarrelRecHitsPerTrackVsEtaProfile->Fill(eta,    track.hitPattern().numberOfValidPixelBarrelHits());

  NumberOfPixBarrelLayersPerTrack->Fill(track.hitPattern().pixelBarrelLayersWithMeasurement());
  NumberOfPixBarrelLayersPerTrackVsPhiProfile->Fill(phi,     track.hitPattern().pixelBarrelLayersWithMeasurement());
  NumberOfPixBarrelLayersPerTrackVsEtaProfile->Fill(eta,     track.hitPattern().pixelBarrelLayersWithMeasurement());

  //Fill PixEndcap Layers and RecHits
  NumberOfPixEndcapRecHitsPerTrack->Fill(track.hitPattern().numberOfValidPixelEndcapHits());
  NumberOfPixEndcapRecHitsPerTrackVsPhiProfile->Fill(phi,    track.hitPattern().numberOfValidPixelEndcapHits());
  NumberOfPixEndcapRecHitsPerTrackVsEtaProfile->Fill(eta,    track.hitPattern().numberOfValidPixelEndcapHits());

  NumberOfPixEndcapLayersPerTrack->Fill(track.hitPattern().pixelEndcapLayersWithMeasurement());
  NumberOfPixEndcapLayersPerTrackVsPhiProfile->Fill(phi,     track.hitPattern().pixelEndcapLayersWithMeasurement());
  NumberOfPixEndcapLayersPerTrackVsEtaProfile->Fill(eta,     track.hitPattern().pixelEndcapLayersWithMeasurement());

}
//
// -- Set Lumi Flag
//
void TrackAnalyzer::setLumiFlag() { 
  /*
  if (Chi2oNDF) Chi2oNDF->setLumiFlag();
  if (NumberOfRecHitsPerTrack) NumberOfRecHitsPerTrack->setLumiFlag();
  if (GoodTrackChi2oNDF) GoodTrackChi2oNDF->setLumiFlag();
  if (GoodTrackNumberOfRecHitsPerTrack) GoodTrackNumberOfRecHitsPerTrack->setLumiFlag();
  */
  if ( Chi2oNDF_lumiFlag                         ) Chi2oNDF_lumiFlag                         -> setLumiFlag();
  if ( NumberOfRecHitsPerTrack_lumiFlag          ) NumberOfRecHitsPerTrack_lumiFlag          -> setLumiFlag();
  if ( GoodTrackChi2oNDF_lumiFlag                ) GoodTrackChi2oNDF_lumiFlag                -> setLumiFlag();
  if ( GoodTrackNumberOfRecHitsPerTrack_lumiFlag ) GoodTrackNumberOfRecHitsPerTrack_lumiFlag -> setLumiFlag();
}
//
// -- Apply SoftReset 
//
void TrackAnalyzer::doSoftReset(DQMStore * dqmStore_) {
  dqmStore_->softReset(Chi2oNDF);
  dqmStore_->softReset(NumberOfRecHitsPerTrack);
  dqmStore_->softReset(GoodTrackChi2oNDF);
  dqmStore_->softReset(GoodTrackNumberOfRecHitsPerTrack);
}
//
// -- Apply Reset 
//
void TrackAnalyzer::doReset(DQMStore * dqmStore_) {
  if ( Chi2oNDF_lumiFlag                         ) Chi2oNDF_lumiFlag                         -> Reset();
  if ( NumberOfRecHitsPerTrack_lumiFlag          ) NumberOfRecHitsPerTrack_lumiFlag          -> Reset();
  if ( GoodTrackChi2oNDF_lumiFlag                ) GoodTrackChi2oNDF_lumiFlag                -> Reset();
  if ( GoodTrackNumberOfRecHitsPerTrack_lumiFlag ) GoodTrackNumberOfRecHitsPerTrack_lumiFlag -> Reset();
}
//
// -- Remove SoftReset
//
void TrackAnalyzer::undoSoftReset(DQMStore * dqmStore_) {
  dqmStore_->disableSoftReset(Chi2oNDF);
  dqmStore_->disableSoftReset(NumberOfRecHitsPerTrack);
  dqmStore_->disableSoftReset(GoodTrackChi2oNDF);
  dqmStore_->disableSoftReset(GoodTrackNumberOfRecHitsPerTrack);
}


