/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/09/14 16:18:39 $
 *  $Revision: 1.4 $
 *  \author Suchandra Dutta , Giorgia Mila
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/TrackingMonitor/interface/TrackAnalyzer.h"
#include <string>
#include "TMath.h"

TrackAnalyzer::TrackAnalyzer(const edm::ParameterSet& iConfig) {
  conf_ = iConfig;
}

TrackAnalyzer::~TrackAnalyzer() { }

void TrackAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dqmStore_) {
  using namespace edm;

  std::string AlgoName     = conf_.getParameter<std::string>("AlgoName");
  std::string MEFolderName = conf_.getParameter<std::string>("FolderName"); 
  std::string MEBSFolderName = conf_.getParameter<std::string>("BSFolderName"); 

  dqmStore_->setCurrentFolder(MEFolderName);

  doTrackerSpecific_ = conf_.getParameter<bool>("doTrackerSpecific");
  doAllPlots_ = conf_.getParameter<bool>("doAllPlots");
  doBSPlots_ = conf_.getParameter<bool>("doBeamSpotPlots");

  int    TKHitBin = conf_.getParameter<int>("RecHitBin");
  double TKHitMin = conf_.getParameter<double>("RecHitMin");
  double TKHitMax = conf_.getParameter<double>("RecHitMax");

  int    TKLostBin = conf_.getParameter<int>("RecLostBin");
  double TKLostMin = conf_.getParameter<double>("RecLostMin");
  double TKLostMax = conf_.getParameter<double>("RecLostMax");

  int    TKLayBin = conf_.getParameter<int>("RecLayBin");
  double TKLayMin = conf_.getParameter<double>("RecLayMin");
  double TKLayMax = conf_.getParameter<double>("RecLayMax");

  int    Chi2Bin  = conf_.getParameter<int>("Chi2Bin");
  double Chi2Min  = conf_.getParameter<double>("Chi2Min");
  double Chi2Max  = conf_.getParameter<double>("Chi2Max");

  int    Chi2ProbBin  = conf_.getParameter<int>("Chi2ProbBin");
  double Chi2ProbMin  = conf_.getParameter<double>("Chi2ProbMin");
  double Chi2ProbMax  = conf_.getParameter<double>("Chi2ProbMax");

  int    PhiBin   = conf_.getParameter<int>("PhiBin");
  double PhiMin   = conf_.getParameter<double>("PhiMin");
  double PhiMax   = conf_.getParameter<double>("PhiMax");

  int    EtaBin   = conf_.getParameter<int>("EtaBin");
  double EtaMin   = conf_.getParameter<double>("EtaMin");
  double EtaMax   = conf_.getParameter<double>("EtaMax");

  int    ThetaBin = conf_.getParameter<int>("ThetaBin");
  double ThetaMin = conf_.getParameter<double>("ThetaMin");
  double ThetaMax = conf_.getParameter<double>("ThetaMax");

  int    D0Bin = conf_.getParameter<int>("D0Bin");
  double D0Min = conf_.getParameter<double>("D0Min");
  double D0Max = conf_.getParameter<double>("D0Max");

  int    VXBin = conf_.getParameter<int>("VXBin");
  double VXMin = conf_.getParameter<double>("VXMin");
  double VXMax = conf_.getParameter<double>("VXMax");

  int    VYBin = conf_.getParameter<int>("VYBin");
  double VYMin = conf_.getParameter<double>("VYMin");
  double VYMax = conf_.getParameter<double>("VYMax");

  int    VZBin = conf_.getParameter<int>("VZBin");
  double VZMin = conf_.getParameter<double>("VZMin");
  double VZMax = conf_.getParameter<double>("VZMax");

  int    X0Bin = conf_.getParameter<int>("X0Bin");
  double X0Min = conf_.getParameter<double>("X0Min");
  double X0Max = conf_.getParameter<double>("X0Max");

  int    Y0Bin = conf_.getParameter<int>("Y0Bin");
  double Y0Min = conf_.getParameter<double>("Y0Min");
  double Y0Max = conf_.getParameter<double>("Y0Max");

  int    Z0Bin = conf_.getParameter<int>("Z0Bin");
  double Z0Min = conf_.getParameter<double>("Z0Min");
  double Z0Max = conf_.getParameter<double>("Z0Max");
  dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties");

  histname = "NumberOfRecHitsPerTrack_";
  NumberOfRecHitsPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsPerTrack->setAxisTitle("Number of RecHits of each track");

  histname = "NumberOfRecHitsFoundPerTrack_";
  NumberOfRecHitsFoundPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKHitBin, TKHitMin, TKHitMax);
  NumberOfRecHitsFoundPerTrack->setAxisTitle("Number of RecHits found for each track");

  histname = "NumberOfRecHitsLostPerTrack_";
  NumberOfRecHitsLostPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKLostBin, TKLostMin, TKLostMax);
  NumberOfRecHitsLostPerTrack->setAxisTitle("Number of RecHits lost for each track");

  histname = "NumberOfLayersPerTrack_";
  NumberOfLayersPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKLayBin, TKLayMin, TKLayMax);
  NumberOfLayersPerTrack->setAxisTitle("Number of Layers of each track");
  dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");

  histname = "Chi2_";
  Chi2 = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, Chi2Bin, Chi2Min, Chi2Max);
  Chi2->setAxisTitle("Chi2 of each track");

  histname = "nChi2Prob_";
  nChi2Prob = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, Chi2ProbBin, Chi2ProbMin, Chi2ProbMax);
  nChi2Prob->setAxisTitle("normalized Chi2 probability");
  
  histname = "Chi2overDoF_";
  Chi2overDoF = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, Chi2Bin, Chi2Min, Chi2Max/10);
  Chi2overDoF->setAxisTitle("Chi2 over nr. of degrees of freedom of each track");

  if(doBSPlots_)
    {
      dqmStore_->setCurrentFolder(MEBSFolderName);
  
      histname = "DistanceOfClosestApproachToBS_";
      DistanceOfClosestApproachToBS = dqmStore_->book1D(histname+AlgoName,histname+AlgoName,D0Bin,D0Min,D0Max);
      DistanceOfClosestApproachToBS->setAxisTitle("Track distance of closest approach to beam spot (cm)");
  
      histname = "DistanceOfClosestApproachToBSVsPhi_";
      DistanceOfClosestApproachToBSVsPhi = dqmStore_->bookProfile(histname+AlgoName,histname+AlgoName, PhiBin, PhiMin, PhiMax, D0Bin,D0Min,D0Max,"");
      DistanceOfClosestApproachToBSVsPhi->getTH1()->SetBit(TH1::kCanRebin);
      DistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track azimuthal angle",1);
      DistanceOfClosestApproachToBSVsPhi->setAxisTitle("Track distance of closest approach to beam spot (cm)",2);

      histname = "xPointOfClosestApproachVsZ0_";
      xPointOfClosestApproachVsZ0 = dqmStore_->bookProfile(histname+AlgoName, histname+AlgoName, Z0Bin, Z0Min, Z0Max, X0Bin, X0Min, X0Max,"");
      xPointOfClosestApproachVsZ0->setAxisTitle("dz (cm)",1);
      xPointOfClosestApproachVsZ0->setAxisTitle("Track distance of closest approach on the x-axis (cm)",2);

      histname = "yPointOfClosestApproachVsZ0_";
      yPointOfClosestApproachVsZ0 = dqmStore_->bookProfile(histname+AlgoName, histname+AlgoName, Z0Bin, Z0Min, Z0Max, Y0Bin, Y0Min, Y0Max,"");
      yPointOfClosestApproachVsZ0->setAxisTitle("dz (cm)",1);
      yPointOfClosestApproachVsZ0->setAxisTitle("Track distance of closest approach on the y-axis (cm)",2);

    }
  dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");
 
  histname = "DistanceOfClosestApproach_";
  DistanceOfClosestApproach = dqmStore_->book1D(histname+AlgoName,histname+AlgoName,D0Bin,D0Min,D0Max);
  DistanceOfClosestApproach->setAxisTitle("Track distance of closest approach (cm)");

  histname = "DistanceOfClosestApproachVsPhi_";
  DistanceOfClosestApproachVsPhi = dqmStore_->bookProfile(histname+AlgoName,histname+AlgoName, PhiBin, PhiMin, PhiMax, D0Bin,D0Min,D0Max,"");
  DistanceOfClosestApproachVsPhi->getTH1()->SetBit(TH1::kCanRebin);
  DistanceOfClosestApproachVsPhi->setAxisTitle("Track azimuthal angle",1);
  DistanceOfClosestApproachVsPhi->setAxisTitle("Track distance of closest approach (cm)",2);
 
  if(doBSPlots_) dqmStore_->setCurrentFolder(MEFolderName);
  
  if(doAllPlots_)
    {
   dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");
     histname = "DistanceOfClosestApproachVsTheta_";
      DistanceOfClosestApproachVsTheta = dqmStore_->book2D(histname+AlgoName,histname+AlgoName, ThetaBin, ThetaMin, ThetaMax, D0Bin,D0Min,D0Max);
      DistanceOfClosestApproachVsTheta->setAxisTitle("Track polar angle",1);
      DistanceOfClosestApproachVsTheta->setAxisTitle("Track distance of closest approach",2);

      histname = "DistanceOfClosestApproachVsEta_";
      DistanceOfClosestApproachVsEta = dqmStore_->book2D(histname+AlgoName,histname+AlgoName, EtaBin, EtaMin, EtaMax, D0Bin,D0Min,D0Max);
      DistanceOfClosestApproachVsEta->setAxisTitle("Track pseudorapidity",1);
      DistanceOfClosestApproachVsEta->setAxisTitle("Track distance of closest approach",2);
    }
 
  histname = "xPointOfClosestApproach_";
  xPointOfClosestApproach = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, VXBin, VXMin, VXMax);
  xPointOfClosestApproach->setAxisTitle("Track distance of closest approach on the x-axis");

  histname = "yPointOfClosestApproach_";
  yPointOfClosestApproach = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, VYBin, VYMin, VYMax);
  yPointOfClosestApproach->setAxisTitle("Track distance of closest approach on the y-axis");

  histname = "zPointOfClosestApproach_";
  zPointOfClosestApproach = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, VZBin, VZMin, VZMax);
  zPointOfClosestApproach->setAxisTitle("Track distance of closest approach on the z-axis");

  if(doTrackerSpecific_) doTrackerSpecificInitialization(dqmStore_);

  std::string StateName = conf_.getParameter<std::string>("MeasurementState");
  if (StateName == "All") {
    bookHistosForState("OuterSurface", dqmStore_);
    bookHistosForState("InnerSurface", dqmStore_);
    bookHistosForState("ImpactPoint", dqmStore_);
  } else {
    bookHistosForState(StateName, dqmStore_);
  }
}
//
// -- Analyse
//
void TrackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Track& track){
   using namespace edm;
  InputTag bsSrc = conf_.getParameter< edm::InputTag >("beamSpot");

 
  NumberOfRecHitsPerTrack->Fill(track.recHitsSize());
  NumberOfRecHitsFoundPerTrack->Fill(track.found());
  NumberOfRecHitsLostPerTrack->Fill(track.lost());
   
  NumberOfLayersPerTrack->Fill(track.hitPattern().trackerLayersWithMeasurement());
  
  Chi2->Fill(track.chi2());
  nChi2Prob->Fill(TMath::Prob(track.chi2(),(int)track.ndof()));
  Chi2overDoF->Fill(track.normalizedChi2());
  
  DistanceOfClosestApproach->Fill(track.d0());
  DistanceOfClosestApproachVsPhi->Fill(track.phi(), track.d0());
  
  if(doBSPlots_){
 Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(bsSrc,recoBeamSpotHandle);
  reco::BeamSpot bs = *recoBeamSpotHandle;      

    DistanceOfClosestApproachToBS->Fill(-track.dxy(bs.position()));
    DistanceOfClosestApproachToBSVsPhi->Fill(track.phi(), -track.dxy(bs.position()));
    xPointOfClosestApproachVsZ0->Fill(track.dz(),track.vx());
    yPointOfClosestApproachVsZ0->Fill(track.dz(),track.vy());
  }
  
  if(doAllPlots_)
    {
      DistanceOfClosestApproachVsTheta->Fill(track.theta(), track.d0());
      DistanceOfClosestApproachVsEta->Fill(track.eta(), track.d0());
    }  
  xPointOfClosestApproach->Fill(track.vertex().x());
  yPointOfClosestApproach->Fill(track.vertex().y());
  zPointOfClosestApproach->Fill(track.vertex().z());

  
  //Tracker Specific Histograms
  
  if(doTrackerSpecific_) doTrackerSpecificFillHists(track);
  
  std::string StateName = conf_.getParameter<std::string>("MeasurementState");
  if (StateName == "All") {
    fillHistosForState(iSetup, track, std::string("OuterSurface"));
    fillHistosForState(iSetup, track, std::string("InnerSurface"));
    fillHistosForState(iSetup, track, std::string("ImpactPoint"));
  } else {
    fillHistosForState(iSetup, track, StateName);
  }
  
}


// 
// book histograms at differnt measurement points
// 
void TrackAnalyzer::bookHistosForState(std::string sname, DQMStore * dqmStore_) {

  std::string AlgoName     = conf_.getParameter<std::string>("AlgoName");
  std::string MEFolderName = conf_.getParameter<std::string>("FolderName"); 

  int    TKHitBin   = conf_.getParameter<int>("RecHitBin");
  double TKHitMin   = conf_.getParameter<double>("RecHitMin");
  double TKHitMax   = conf_.getParameter<double>("RecHitMax");

  int    Chi2Bin    = conf_.getParameter<int>("Chi2Bin");
  double Chi2Min    = conf_.getParameter<double>("Chi2Min");
  double Chi2Max    = conf_.getParameter<double>("Chi2Max");

  int    PhiBin     = conf_.getParameter<int>("PhiBin");
  double PhiMin     = conf_.getParameter<double>("PhiMin");
  double PhiMax     = conf_.getParameter<double>("PhiMax");

  int    EtaBin     = conf_.getParameter<int>("EtaBin");
  double EtaMin     = conf_.getParameter<double>("EtaMin");
  double EtaMax     = conf_.getParameter<double>("EtaMax");

  int    ThetaBin   = conf_.getParameter<int>("ThetaBin");
  double ThetaMin   = conf_.getParameter<double>("ThetaMin");
  double ThetaMax   = conf_.getParameter<double>("ThetaMax");

  int    TrackPtBin = conf_.getParameter<int>("TrackPtBin");
  double TrackPtMin = conf_.getParameter<double>("TrackPtMin");
  double TrackPtMax = conf_.getParameter<double>("TrackPtMax");

  int    TrackPBin = conf_.getParameter<int>("TrackPBin");
  double TrackPMin = conf_.getParameter<double>("TrackPMin");
  double TrackPMax = conf_.getParameter<double>("TrackPMax");

  int    TrackPxBin = conf_.getParameter<int>("TrackPxBin");
  double TrackPxMin = conf_.getParameter<double>("TrackPxMin");
  double TrackPxMax = conf_.getParameter<double>("TrackPxMax");

  int    TrackPyBin = conf_.getParameter<int>("TrackPyBin");
  double TrackPyMin = conf_.getParameter<double>("TrackPyMin");
  double TrackPyMax = conf_.getParameter<double>("TrackPyMax");

  int    TrackPzBin = conf_.getParameter<int>("TrackPzBin");
  double TrackPzMin = conf_.getParameter<double>("TrackPzMin");
  double TrackPzMax = conf_.getParameter<double>("TrackPzMax");

  int    ptErrBin   = conf_.getParameter<int>("ptErrBin");
  double ptErrMin   = conf_.getParameter<double>("ptErrMin");
  double ptErrMax   = conf_.getParameter<double>("ptErrMax");

  int    pxErrBin   = conf_.getParameter<int>("pxErrBin");
  double pxErrMin   = conf_.getParameter<double>("pxErrMin");
  double pxErrMax   = conf_.getParameter<double>("pxErrMax");

  int    pyErrBin   = conf_.getParameter<int>("pyErrBin");
  double pyErrMin   = conf_.getParameter<double>("pyErrMin");
  double pyErrMax   = conf_.getParameter<double>("pyErrMax");

  int    pzErrBin   = conf_.getParameter<int>("pzErrBin");
  double pzErrMin   = conf_.getParameter<double>("pzErrMin");
  double pzErrMax   = conf_.getParameter<double>("pzErrMax");

  int    pErrBin    = conf_.getParameter<int>("pErrBin");
  double pErrMin    = conf_.getParameter<double>("pErrMin");
  double pErrMax    = conf_.getParameter<double>("pErrMax");

  int    phiErrBin  = conf_.getParameter<int>("phiErrBin");
  double phiErrMin  = conf_.getParameter<double>("phiErrMin");
  double phiErrMax  = conf_.getParameter<double>("phiErrMax");

  int    etaErrBin  = conf_.getParameter<int>("etaErrBin");
  double etaErrMin  = conf_.getParameter<double>("etaErrMin");
  double etaErrMax  = conf_.getParameter<double>("etaErrMax");

  std::string htag;

  if (sname == "default") htag = AlgoName;
  else                    htag = sname + "_" + AlgoName; 
  dqmStore_->setCurrentFolder(MEFolderName);

  TkParameterMEs tkmes;
  tkmes.TrackP = 0;
  tkmes.TrackPx = 0;
  tkmes.TrackPy = 0;
  tkmes.TrackPz = 0;
  tkmes.TrackPt = 0;
  tkmes.TrackPxErr = 0;
  tkmes.TrackPyErr = 0;
  tkmes.TrackPzErr = 0;
  tkmes.TrackPtErr = 0;
  tkmes.TrackPErr = 0;
  tkmes.TrackPhi = 0;
  tkmes.TrackEta = 0;
  tkmes.TrackTheta = 0;
  tkmes.TrackPhiErr = 0;
  tkmes.TrackEtaErr = 0;
  tkmes.TrackThetaErr = 0;
  tkmes.NumberOfRecHitsPerTrackVsPhi = 0;
  tkmes.NumberOfRecHitsPerTrackVsTheta = 0;
  tkmes.NumberOfRecHitsPerTrackVsEta = 0;
  tkmes.NumberOfRecHitsPerTrackVsPhiProfile = 0;
  tkmes.NumberOfRecHitsPerTrackVsThetaProfile = 0;
  tkmes.NumberOfRecHitsPerTrackVsEtaProfile = 0;
  tkmes.Chi2overDoFVsTheta = 0;
  tkmes.Chi2overDoFVsPhi = 0;
  tkmes.Chi2overDoFVsEta = 0;

  if(doAllPlots_)
    {
      dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties");

      histname = "NumberOfRecHitsPerTrackVsPhi_" + htag;
      tkmes.NumberOfRecHitsPerTrackVsPhi = dqmStore_->book2D(histname, histname, PhiBin, PhiMin, PhiMax, TKHitBin, TKHitMin, TKHitMax);
      tkmes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Track azimuthal angle",1);
      tkmes.NumberOfRecHitsPerTrackVsPhi->setAxisTitle("Number of RecHits of each track",2);

      histname = "NumberOfRecHitsPerTrackVsTheta_" + htag;
      tkmes.NumberOfRecHitsPerTrackVsTheta = dqmStore_->book2D(histname, histname, ThetaBin, ThetaMin, ThetaMax, TKHitBin, TKHitMin, TKHitMax);
      tkmes.NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Track polar angle",1);
      tkmes.NumberOfRecHitsPerTrackVsTheta->setAxisTitle("Number of RecHits of each track",2);

      histname = "NumberOfRecHitsPerTrackVsEta_" + htag;
      tkmes.NumberOfRecHitsPerTrackVsEta = dqmStore_->book2D(histname, histname, EtaBin, EtaMin, EtaMax, TKHitBin, TKHitMin, TKHitMax);
      tkmes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Track pseudorapidity",1);
      tkmes.NumberOfRecHitsPerTrackVsEta->setAxisTitle("Number of RecHits of each track",2);

      dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");

      histname = "Chi2overDoFVsTheta_" + htag;
      tkmes.Chi2overDoFVsTheta = dqmStore_->book2D(histname, histname, ThetaBin, ThetaMin, ThetaMax, Chi2Bin, Chi2Min, Chi2Max);
      tkmes.Chi2overDoFVsTheta->setAxisTitle("Track polar angle",1);
      tkmes.Chi2overDoFVsTheta->setAxisTitle("Chi2 over nr. of degrees of freedom of each track",2);

      histname = "Chi2overDoFVsPhi_" + htag;
      tkmes.Chi2overDoFVsPhi   = dqmStore_->book2D(histname, histname, PhiBin, PhiMin, PhiMax, Chi2Bin, Chi2Min, Chi2Max);
      tkmes.Chi2overDoFVsPhi->setAxisTitle("Track azimuthal angle",1);
      tkmes.Chi2overDoFVsPhi->setAxisTitle("Chi2 over nr. of degrees of freedom of each track",2);

      histname = "Chi2overDoFVsEta_" + htag;
      tkmes.Chi2overDoFVsEta   = dqmStore_->book2D(histname, histname, EtaBin, EtaMin, EtaMax, Chi2Bin, Chi2Min, Chi2Max);
      tkmes.Chi2overDoFVsEta->setAxisTitle("Track pseudorapidity",1);
      tkmes.Chi2overDoFVsEta->setAxisTitle("Chi2 over nr. of degrees of freedom of each track",2);
    }
      dqmStore_->setCurrentFolder(MEFolderName+"/GeneralProperties");
  histname = "TrackP_" + htag;
  tkmes.TrackP = dqmStore_->book1D(histname, histname, TrackPBin, TrackPMin, TrackPMax);
  tkmes.TrackP->setAxisTitle("Track momentum");

  histname = "TrackPt_" + htag;
  tkmes.TrackPt = dqmStore_->book1D(histname, histname, TrackPtBin, TrackPtMin, TrackPtMax);
  tkmes.TrackPt->setAxisTitle("Transverse track momentum");

  histname = "TrackPx_" + htag;
  tkmes.TrackPx = dqmStore_->book1D(histname, histname, TrackPxBin, TrackPxMin, TrackPxMax);
  tkmes.TrackPx->setAxisTitle("x component of track momentum");

  histname = "TrackPy_" + htag;
  tkmes.TrackPy = dqmStore_->book1D(histname, histname, TrackPyBin, TrackPyMin, TrackPyMax);
  tkmes.TrackPy->setAxisTitle("y component of track momentum");

  histname = "TrackPz_" + htag;
  tkmes.TrackPz = dqmStore_->book1D(histname, histname, TrackPzBin, TrackPzMin, TrackPzMax);
  tkmes.TrackPz->setAxisTitle("z component of track momentum");

  histname = "TrackPhi_" + htag;
  tkmes.TrackPhi = dqmStore_->book1D(histname, histname, PhiBin, PhiMin, PhiMax);
  tkmes.TrackPhi->setAxisTitle("Track azimuthal angle");

  histname = "TrackEta_" + htag;
  tkmes.TrackEta = dqmStore_->book1D(histname, histname, EtaBin, EtaMin, EtaMax);
  tkmes.TrackEta->setAxisTitle("Track pseudorapidity");

  histname = "TrackTheta_" + htag;
  tkmes.TrackTheta = dqmStore_->book1D(histname, histname, ThetaBin, ThetaMin, ThetaMax);
  tkmes.TrackTheta->setAxisTitle("Track polar angle");

  histname = "TrackPtErrOverPt_" + htag;
  tkmes.TrackPtErr = dqmStore_->book1D(histname, histname, ptErrBin, ptErrMin, ptErrMax);
  tkmes.TrackPtErr->setAxisTitle("ptErr/pt");
  
  histname = "TrackPxErrOverPx_" + htag;
  tkmes.TrackPxErr = dqmStore_->book1D(histname, histname, pxErrBin, pxErrMin, pxErrMax);
  tkmes.TrackPxErr->setAxisTitle("pxErr/px");
  
  histname = "TrackPyErrOverPy_" + htag;
  tkmes.TrackPyErr = dqmStore_->book1D(histname, histname, pyErrBin, pyErrMin, pyErrMax);
  tkmes.TrackPyErr->setAxisTitle("pyErr/py");
  
  histname = "TrackPzErrOverPz_" + htag;
  tkmes.TrackPzErr = dqmStore_->book1D(histname, histname, pzErrBin, pzErrMin, pzErrMax);
  tkmes.TrackPzErr->setAxisTitle("pzErr/pz");
  
  histname = "TrackPErrOverP_" + htag;
  tkmes.TrackPErr = dqmStore_->book1D(histname, histname, pErrBin, pErrMin, pErrMax);
  tkmes.TrackPErr->setAxisTitle("pErr/p");
  
  histname = "TrackPhiErr_" + htag;
  tkmes.TrackPhiErr = dqmStore_->book1D(histname, histname, phiErrBin, phiErrMin, phiErrMax);
  tkmes.TrackPhiErr->setAxisTitle("phiErr");
  
  histname = "TrackEtaErr_" + htag;
  tkmes.TrackEtaErr = dqmStore_->book1D(histname, histname, etaErrBin, etaErrMin, etaErrMax);
  tkmes.TrackEtaErr->setAxisTitle("etaErr");

  histname = "NumberOfRecHitsPerTrackVsPhiProfile_" + htag;
  tkmes.NumberOfRecHitsPerTrackVsPhiProfile = dqmStore_->bookProfile(histname, histname, PhiBin, PhiMin, PhiMax, TKHitBin, TKHitMin, TKHitMax);
  tkmes.NumberOfRecHitsPerTrackVsPhiProfile->setAxisTitle("Track azimuthal angle",1);
  tkmes.NumberOfRecHitsPerTrackVsPhiProfile->setAxisTitle("Number of RecHits of each track",2);

  histname = "NumberOfRecHitsPerTrackVsThetaProfile_" + htag;
  tkmes.NumberOfRecHitsPerTrackVsThetaProfile = dqmStore_->bookProfile(histname, histname, ThetaBin, ThetaMin, ThetaMax, TKHitBin, TKHitMin, TKHitMax);
  tkmes.NumberOfRecHitsPerTrackVsThetaProfile->setAxisTitle("Track polar angle",1);
  tkmes.NumberOfRecHitsPerTrackVsThetaProfile->setAxisTitle("Number of RecHits of each track",2);

  histname = "NumberOfRecHitsPerTrackVsEtaProfile_" + htag;
  tkmes.NumberOfRecHitsPerTrackVsEtaProfile = dqmStore_->bookProfile(histname, histname, EtaBin, EtaMin, EtaMax, TKHitBin, TKHitMin, TKHitMax);
  tkmes.NumberOfRecHitsPerTrackVsEtaProfile->setAxisTitle("Pseudorapidity",1);
  tkmes.NumberOfRecHitsPerTrackVsEtaProfile->setAxisTitle("Number of RecHits of each track",2);

  // now put the MEs in the map
  TkParameterMEMap.insert(std::make_pair(sname, tkmes));

}
// 
// fill histograms at differnt measurement points
// 
void TrackAnalyzer::fillHistosForState(const edm::EventSetup& iSetup, const reco::Track & track, std::string sname) {


  //get the kinematic parameters
  double p, px, py, pz, pt, theta, phi, eta;
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
    
    pterror  = (pt) ? track.ptError()/pt : 0.0;
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
    
    tkmes.TrackP->Fill(p);
    tkmes.TrackPx->Fill(px);
    tkmes.TrackPy->Fill(py);
    tkmes.TrackPz->Fill(pz);
    tkmes.TrackPt->Fill(pt);
    
    tkmes.TrackPhi->Fill(phi);
    tkmes.TrackEta->Fill(eta);
    tkmes.TrackTheta->Fill(theta);
    if(doAllPlots_)
      {
	tkmes.NumberOfRecHitsPerTrackVsPhi->Fill(phi, track.found());
	tkmes.NumberOfRecHitsPerTrackVsTheta->Fill(theta, track.found());
	tkmes.NumberOfRecHitsPerTrackVsEta->Fill(eta, track.found());
      }
    tkmes.NumberOfRecHitsPerTrackVsPhiProfile->Fill(phi, track.found());
    tkmes.NumberOfRecHitsPerTrackVsThetaProfile->Fill(theta, track.found());
    tkmes.NumberOfRecHitsPerTrackVsEtaProfile->Fill(eta, track.found());
    if(doAllPlots_)
      {
   
	tkmes.Chi2overDoFVsTheta->Fill(theta, track.normalizedChi2());
	tkmes.Chi2overDoFVsPhi->Fill(phi, track.normalizedChi2());
	tkmes.Chi2overDoFVsEta->Fill(eta, track.normalizedChi2());
      }
    tkmes.TrackPtErr->Fill(pterror);
    tkmes.TrackPxErr->Fill(pxerror);
    tkmes.TrackPyErr->Fill(pyerror);
    tkmes.TrackPzErr->Fill(pzerror);
    tkmes.TrackPErr->Fill(perror);
    tkmes.TrackPhiErr->Fill(phierror);
    tkmes.TrackEtaErr->Fill(etaerror);
  }
}


void TrackAnalyzer::doTrackerSpecificInitialization(DQMStore * dqmStore_) {

  std::string AlgoName     = conf_.getParameter<std::string>("AlgoName");
  std::string MEFolderName = conf_.getParameter<std::string>("FolderName"); 
  int    TKHitBin = conf_.getParameter<int>("RecHitBin");
  double TKHitMin = conf_.getParameter<double>("RecHitMin");
  double TKHitMax = conf_.getParameter<double>("RecHitMax");

  int    TKLayBin = conf_.getParameter<int>("RecLayBin");
  double TKLayMin = conf_.getParameter<double>("RecLayMin");
  double TKLayMax = conf_.getParameter<double>("RecLayMax");
  dqmStore_->setCurrentFolder(MEFolderName+"/HitProperties");

  histname = "NumberOfTOBRecHitsPerTrack_";
  NumberOfTOBRecHitsPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKHitBin, TKHitMin, TKHitMax);
  NumberOfTOBRecHitsPerTrack->setAxisTitle("Number of TOB RecHits of each track");

  histname = "NumberOfTIBRecHitsPerTrack_";
  NumberOfTIBRecHitsPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKHitBin, TKHitMin, TKHitMax);
  NumberOfTIBRecHitsPerTrack->setAxisTitle("Number of TIB RecHits of each track");

  histname = "NumberOfTIDRecHitsPerTrack_";
  NumberOfTIDRecHitsPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKHitBin, TKHitMin, TKHitMax);
  NumberOfTIDRecHitsPerTrack->setAxisTitle("Number of TID RecHits of each track");

  histname = "NumberOfTECRecHitsPerTrack_";
  NumberOfTECRecHitsPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKHitBin, TKHitMin, TKHitMax);
  NumberOfTECRecHitsPerTrack->setAxisTitle("Number of TEC RecHits of each track");

  histname = "NumberOfPixBarrelRecHitsPerTrack_";
  NumberOfPixBarrelRecHitsPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKHitBin, TKHitMin, TKHitMax);
  NumberOfPixBarrelRecHitsPerTrack->setAxisTitle("Number of Pixel Barrel RecHits of each track");

  histname = "NumberOfPixEndcapRecHitsPerTrack_";
  NumberOfPixEndcapRecHitsPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKHitBin, TKHitMin, TKHitMax);
  NumberOfPixEndcapRecHitsPerTrack->setAxisTitle("Number of Pixel Endcap RecHits of each track");

  histname = "NumberOfTOBLayersPerTrack_";
  NumberOfTOBLayersPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKLayBin, TKLayMin, TKLayMax);
  NumberOfTOBLayersPerTrack->setAxisTitle("Number of TOB Layers of each track");

  histname = "NumberOfTIBLayersPerTrack_";
  NumberOfTIBLayersPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKLayBin, TKLayMin, TKLayMax);
  NumberOfTIBLayersPerTrack->setAxisTitle("Number of TIB Layers of each track");

  histname = "NumberOfTIDLayersPerTrack_";
  NumberOfTIDLayersPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKLayBin, TKLayMin, TKLayMax);
  NumberOfTIDLayersPerTrack->setAxisTitle("Number of TID Layers of each track");

  histname = "NumberOfTECLayersPerTrack_";
  NumberOfTECLayersPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKLayBin, TKLayMin, TKLayMax);
  NumberOfTECLayersPerTrack->setAxisTitle("Number of TID Layers of each track");

  histname = "NumberOfPixBarrelLayersPerTrack_";
  NumberOfPixBarrelLayersPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKLayBin, TKLayMin, TKLayMax);
  NumberOfPixBarrelLayersPerTrack->setAxisTitle("Number of Pixel Barrel Layers of each track");

  histname = "NumberOfPixEndcapLayersPerTrack_";
  NumberOfPixEndcapLayersPerTrack = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TKLayBin, TKLayMin, TKLayMax);
  NumberOfPixEndcapLayersPerTrack->setAxisTitle("Number of Pixel Endcap Layers of each track");

}


void TrackAnalyzer::doTrackerSpecificFillHists(const reco::Track & track) {

  //Fill TIB Layers and RecHits
  NumberOfTIBRecHitsPerTrack->Fill(track.hitPattern().numberOfValidStripTIBHits());
  NumberOfTIBLayersPerTrack->Fill(track.hitPattern().stripTIBLayersWithMeasurement());

  //Fill TOB Layers and RecHits
  NumberOfTOBRecHitsPerTrack->Fill(track.hitPattern().numberOfValidStripTOBHits());
  NumberOfTOBLayersPerTrack->Fill(track.hitPattern().stripTOBLayersWithMeasurement());

  //Fill TID Layers and RecHits
  NumberOfTIDRecHitsPerTrack->Fill(track.hitPattern().numberOfValidStripTIDHits());
  NumberOfTIDLayersPerTrack->Fill(track.hitPattern().stripTIDLayersWithMeasurement());

  //Fill TEC Layers and RecHits
  NumberOfTECRecHitsPerTrack->Fill(track.hitPattern().numberOfValidStripTECHits());
  NumberOfTECLayersPerTrack->Fill(track.hitPattern().stripTECLayersWithMeasurement());

  //Fill PixBarrel Layers and RecHits
  NumberOfPixBarrelRecHitsPerTrack->Fill(track.hitPattern().numberOfValidPixelBarrelHits());
  NumberOfPixBarrelLayersPerTrack->Fill(track.hitPattern().pixelBarrelLayersWithMeasurement());

  //Fill PixEndcap Layers and RecHits
  NumberOfPixEndcapRecHitsPerTrack->Fill(track.hitPattern().numberOfValidPixelEndcapHits());
  NumberOfPixEndcapLayersPerTrack->Fill(track.hitPattern().pixelEndcapLayersWithMeasurement());

}



