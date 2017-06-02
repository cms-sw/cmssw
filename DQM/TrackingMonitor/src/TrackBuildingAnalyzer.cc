#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/TrackingMonitor/interface/TrackBuildingAnalyzer.h"
#include <string>
#include "TMath.h"

#include <iostream>

TrackBuildingAnalyzer::TrackBuildingAnalyzer(const edm::ParameterSet& iConfig) 
    : SeedPt(NULL)
    , SeedEta(NULL)
    , SeedPhi(NULL)
    , SeedPhiVsEta(NULL)
    , SeedTheta(NULL)
    , SeedQ(NULL)
    , SeedDxy(NULL)
    , SeedDz(NULL)
    , NumberOfRecHitsPerSeed(NULL)
    , NumberOfRecHitsPerSeedVsPhiProfile(NULL)
    , NumberOfRecHitsPerSeedVsEtaProfile(NULL)
    , stoppingSource(NULL)
    , stoppingSourceVSeta(NULL)
    , stoppingSourceVSphi(NULL)
{
}

TrackBuildingAnalyzer::~TrackBuildingAnalyzer() 
{ 
}

void TrackBuildingAnalyzer::initHisto(DQMStore::IBooker & ibooker, const edm::ParameterSet& iConfig)
{
  
  // parameters from the configuration
  std::string AlgoName       = iConfig.getParameter<std::string>("AlgoName");
  std::string MEFolderName   = iConfig.getParameter<std::string>("FolderName"); 

  //  std::cout << "[TrackBuildingAnalyzer::beginRun] AlgoName: " << AlgoName << std::endl;
  
  // use the AlgoName and Quality Name 
  std::string CatagoryName = AlgoName;
  
  // get binning from the configuration
  int    TrackPtBin = iConfig.getParameter<int>(   "TrackPtBin");
  double TrackPtMin = iConfig.getParameter<double>("TrackPtMin");
  double TrackPtMax = iConfig.getParameter<double>("TrackPtMax");
  
  int    PhiBin     = iConfig.getParameter<int>(   "PhiBin");
  double PhiMin     = iConfig.getParameter<double>("PhiMin");
  double PhiMax     = iConfig.getParameter<double>("PhiMax");
  
  int    EtaBin     = iConfig.getParameter<int>(   "EtaBin");
  double EtaMin     = iConfig.getParameter<double>("EtaMin");
  double EtaMax     = iConfig.getParameter<double>("EtaMax");
  
  int    ThetaBin   = iConfig.getParameter<int>(   "ThetaBin");
  double ThetaMin   = iConfig.getParameter<double>("ThetaMin");
  double ThetaMax   = iConfig.getParameter<double>("ThetaMax");
  
  int    TrackQBin  = iConfig.getParameter<int>(   "TrackQBin");
  double TrackQMin  = iConfig.getParameter<double>("TrackQMin");
  double TrackQMax  = iConfig.getParameter<double>("TrackQMax");
  
  int    SeedDxyBin = iConfig.getParameter<int>(   "SeedDxyBin");
  double SeedDxyMin = iConfig.getParameter<double>("SeedDxyMin");
  double SeedDxyMax = iConfig.getParameter<double>("SeedDxyMax");
  
  int    SeedDzBin  = iConfig.getParameter<int>(   "SeedDzBin");
  double SeedDzMin  = iConfig.getParameter<double>("SeedDzMin");
  double SeedDzMax  = iConfig.getParameter<double>("SeedDzMax");
  
  int    SeedHitBin = iConfig.getParameter<int>(   "SeedHitBin");
  double SeedHitMin = iConfig.getParameter<double>("SeedHitMin");
  double SeedHitMax = iConfig.getParameter<double>("SeedHitMax");
  
  int    TCDxyBin   = iConfig.getParameter<int>(   "TCDxyBin");
  double TCDxyMin   = iConfig.getParameter<double>("TCDxyMin");
  double TCDxyMax   = iConfig.getParameter<double>("TCDxyMax");
  
  int    TCDzBin    = iConfig.getParameter<int>(   "TCDzBin");
  double TCDzMin    = iConfig.getParameter<double>("TCDzMin");
  double TCDzMax    = iConfig.getParameter<double>("TCDzMax");
  
  int    TCHitBin   = iConfig.getParameter<int>(   "TCHitBin");
  double TCHitMin   = iConfig.getParameter<double>("TCHitMin");
  double TCHitMax   = iConfig.getParameter<double>("TCHitMax");
  
  
  edm::InputTag seedProducer   = iConfig.getParameter<edm::InputTag>("SeedProducer");
  edm::InputTag tcProducer     = iConfig.getParameter<edm::InputTag>("TCProducer");
  
  doAllPlots     = iConfig.getParameter<bool>("doAllPlots");
  doAllSeedPlots = iConfig.getParameter<bool>("doSeedParameterHistos");
  doTCPlots      = iConfig.getParameter<bool>("doTrackCandHistos");
  doAllTCPlots   = iConfig.getParameter<bool>("doAllTrackCandHistos");
  doPT           = iConfig.getParameter<bool>("doSeedPTHisto");
  doETA          = iConfig.getParameter<bool>("doSeedETAHisto");
  doPHI          = iConfig.getParameter<bool>("doSeedPHIHisto");
  doPHIVsETA     = iConfig.getParameter<bool>("doSeedPHIVsETAHisto");
  doTheta        = iConfig.getParameter<bool>("doSeedThetaHisto");
  doQ            = iConfig.getParameter<bool>("doSeedQHisto");
  doDxy          = iConfig.getParameter<bool>("doSeedDxyHisto");
  doDz           = iConfig.getParameter<bool>("doSeedDzHisto");
  doNRecHits     = iConfig.getParameter<bool>("doSeedNRecHitsHisto");
  doProfPHI      = iConfig.getParameter<bool>("doSeedNVsPhiProf");
  doProfETA      = iConfig.getParameter<bool>("doSeedNVsEtaProf");
  doStopSource   = iConfig.getParameter<bool>("doStopSource");
  
  //    if (doAllPlots){doAllSeedPlots=true; doTCPlots=true;}
  
  ibooker.setCurrentFolder(MEFolderName);
  
  // book the Seed histograms
  // ---------------------------------------------------------------------------------//
  //  std::cout << "[TrackBuildingAnalyzer::beginRun] MEFolderName: " << MEFolderName << std::endl;
  ibooker.setCurrentFolder(MEFolderName+"/TrackBuilding");

  if (doAllSeedPlots || doPT) {
    histname = "SeedPt_"+seedProducer.label() + "_";
    SeedPt = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, TrackPtBin, TrackPtMin, TrackPtMax);
    SeedPt->setAxisTitle("Seed p_{T} (GeV/c)", 1);
    SeedPt->setAxisTitle("Number of Seeds", 2);
  }
  
  if (doAllSeedPlots || doETA) {
    histname = "SeedEta_"+seedProducer.label() + "_";
    SeedEta = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax);
      SeedEta->setAxisTitle("Seed #eta", 1);
      SeedEta->setAxisTitle("Number of Seeds", 2);
  }
  
  if (doAllSeedPlots || doPHI) {
    histname = "SeedPhi_"+seedProducer.label() + "_";
    SeedPhi = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, PhiBin, PhiMin, PhiMax);
    SeedPhi->setAxisTitle("Seed #phi", 1);
    SeedPhi->setAxisTitle("Number of Seed", 2);
  }
  
  if (doAllSeedPlots || doPHIVsETA) {
    histname = "SeedPhiVsEta_"+seedProducer.label() + "_";
    SeedPhiVsEta = ibooker.book2D(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax, PhiBin, PhiMin, PhiMax);
    SeedPhiVsEta->setAxisTitle("Seed #eta", 1);
    SeedPhiVsEta->setAxisTitle("Seed #phi", 2);
  }
  
  if (doAllSeedPlots || doTheta){
    histname = "SeedTheta_"+seedProducer.label() + "_";
    SeedTheta = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, ThetaBin, ThetaMin, ThetaMax);
    SeedTheta->setAxisTitle("Seed #theta", 1);
    SeedTheta->setAxisTitle("Number of Seeds", 2);
  }
  
  if (doAllSeedPlots || doQ){
    histname = "SeedQ_"+seedProducer.label() + "_";
    SeedQ = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, TrackQBin, TrackQMin, TrackQMax);
    SeedQ->setAxisTitle("Seed Charge", 1);
    SeedQ->setAxisTitle("Number of Seeds",2);
  }
  
  if (doAllSeedPlots || doDxy){
    histname = "SeedDxy_"+seedProducer.label() + "_";
    SeedDxy = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, SeedDxyBin, SeedDxyMin, SeedDxyMax);
    SeedDxy->setAxisTitle("Seed d_{xy} (cm)", 1);
    SeedDxy->setAxisTitle("Number of Seeds",2);
  }
  
  if (doAllSeedPlots || doDz){
    histname = "SeedDz_"+seedProducer.label() + "_";
    SeedDz = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, SeedDzBin, SeedDzMin, SeedDzMax);
    SeedDz->setAxisTitle("Seed d_{z} (cm)", 1);
    SeedDz->setAxisTitle("Number of Seeds",2);
  }
  
  if (doAllSeedPlots || doNRecHits){
    histname = "NumberOfRecHitsPerSeed_"+seedProducer.label() + "_";
    NumberOfRecHitsPerSeed = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, SeedHitBin, SeedHitMin, SeedHitMax);
    NumberOfRecHitsPerSeed->setAxisTitle("Number of RecHits per Seed", 1);
    NumberOfRecHitsPerSeed->setAxisTitle("Number of Seeds",2);
  }
  
  if (doAllSeedPlots || doProfPHI){
    histname = "NumberOfRecHitsPerSeedVsPhiProfile_"+seedProducer.label() + "_";
    NumberOfRecHitsPerSeedVsPhiProfile = ibooker.bookProfile(histname+CatagoryName, histname+CatagoryName, PhiBin, PhiMin, PhiMax, SeedHitBin, SeedHitMin, SeedHitMax,"s");
    NumberOfRecHitsPerSeedVsPhiProfile->setAxisTitle("Seed #phi",1);
    NumberOfRecHitsPerSeedVsPhiProfile->setAxisTitle("Number of RecHits of each Seed",2);
  }
  
  if (doAllSeedPlots || doProfETA){
    histname = "NumberOfRecHitsPerSeedVsEtaProfile_"+seedProducer.label() + "_";
    NumberOfRecHitsPerSeedVsEtaProfile = ibooker.bookProfile(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax, SeedHitBin, SeedHitMin, SeedHitMax,"s");
    NumberOfRecHitsPerSeedVsEtaProfile->setAxisTitle("Seed #eta",1);
    NumberOfRecHitsPerSeedVsEtaProfile->setAxisTitle("Number of RecHits of each Seed",2);
  }

  if (doAllTCPlots || doStopSource) {
    // DataFormats/TrackReco/interface/TrajectoryStopReasons.h
    size_t StopReasonNameSize = sizeof(StopReasonName::StopReasonName)/sizeof(std::string);
    if(StopReasonNameSize != static_cast<unsigned int>(StopReason::SIZE)) {
      throw cms::Exception("Assert") << "StopReason::SIZE is " << static_cast<unsigned int>(StopReason::SIZE)
				     << " but StopReasonName's only for "
				     << StopReasonNameSize
				     << ". Please update DataFormats/TrackReco/interface/TrajectoryStopReasons.h.";
    }
    
    
    histname = "StoppingSource_"+seedProducer.label() + "_";
    stoppingSource = ibooker.book1D(histname+CatagoryName,
                                    histname+CatagoryName,
                                    StopReasonNameSize,
                                    0., double(StopReasonNameSize));
    stoppingSource->setAxisTitle("stopping reason",1);
    stoppingSource->setAxisTitle("Number of Tracks",2);
    
    histname = "StoppingSourceVSeta_"+seedProducer.label() + "_";
    stoppingSourceVSeta = ibooker.book2D(histname+CatagoryName,
                                         histname+CatagoryName,
                                         EtaBin,
                                         EtaMin,
                                         EtaMax,
                                         StopReasonNameSize,
                                         0., double(StopReasonNameSize));
    stoppingSourceVSeta->setAxisTitle("track #eta",1);
    stoppingSourceVSeta->setAxisTitle("stopping reason",2);
    
    histname = "StoppingSourceVSphi_"+seedProducer.label() + "_";
    stoppingSourceVSphi = ibooker.book2D(histname+CatagoryName,
                                         histname+CatagoryName,
                                         PhiBin,
                                         PhiMin,
                                         PhiMax,
                                         StopReasonNameSize,
                                         0., double(StopReasonNameSize));
    stoppingSourceVSphi->setAxisTitle("track #phi",1);
    stoppingSourceVSphi->setAxisTitle("stopping reason",2);
    
    for (size_t ibin=0; ibin<StopReasonNameSize; ibin++) {
      stoppingSource->setBinLabel(ibin+1,StopReasonName::StopReasonName[ibin],1);
      stoppingSourceVSeta->setBinLabel(ibin+1,StopReasonName::StopReasonName[ibin],2);
      stoppingSourceVSphi->setBinLabel(ibin+1,StopReasonName::StopReasonName[ibin],2);
    }
  }
  

  
  // book the TrackCandidate histograms
  // ---------------------------------------------------------------------------------//
  
  if (doTCPlots){
    
    ibooker.setCurrentFolder(MEFolderName+"/TrackBuilding");
    
    histname = "TrackCandPt_"+tcProducer.label() + "_";
    TrackCandPt = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, TrackPtBin, TrackPtMin, TrackPtMax);
    TrackCandPt->setAxisTitle("Track Candidate p_{T} (GeV/c)", 1);
    TrackCandPt->setAxisTitle("Number of Track Candidates", 2);
    
    histname = "TrackCandEta_"+tcProducer.label() + "_";
    TrackCandEta = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax);
    TrackCandEta->setAxisTitle("Track Candidate #eta", 1);
    TrackCandEta->setAxisTitle("Number of Track Candidates", 2);
    
    histname = "TrackCandPhi_"+tcProducer.label() + "_";
    TrackCandPhi = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, PhiBin, PhiMin, PhiMax);
    TrackCandPhi->setAxisTitle("Track Candidate #phi", 1);
    TrackCandPhi->setAxisTitle("Number of Track Candidates", 2);
    
    if (doTheta) {
      histname = "TrackCandTheta_"+tcProducer.label() + "_";
      TrackCandTheta = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, ThetaBin, ThetaMin, ThetaMax);
      TrackCandTheta->setAxisTitle("Track Candidate #theta", 1);
      TrackCandTheta->setAxisTitle("Number of Track Candidates", 2);
    }
    
    if (doAllTCPlots) {
      histname = "TrackCandQ_"+tcProducer.label() + "_";
      TrackCandQ = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, TrackQBin, TrackQMin, TrackQMax);
      TrackCandQ->setAxisTitle("Track Candidate Charge", 1);
      TrackCandQ->setAxisTitle("Number of Track Candidates",2);

      histname = "TrackCandDxy_"+tcProducer.label() + "_";
      TrackCandDxy = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, TCDxyBin, TCDxyMin, TCDxyMax);
      TrackCandDxy->setAxisTitle("Track Candidate d_{xy} (cm)", 1);
      TrackCandDxy->setAxisTitle("Number of Track Candidates",2);
      
      histname = "TrackCandDz_"+tcProducer.label() + "_";
      TrackCandDz = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, TCDzBin, TCDzMin, TCDzMax);
      TrackCandDz->setAxisTitle("Track Candidate d_{z} (cm)", 1);
      TrackCandDz->setAxisTitle("Number of Track Candidates",2);

      histname = "NumberOfRecHitsPerTrackCand_"+tcProducer.label() + "_";
      NumberOfRecHitsPerTrackCand = ibooker.book1D(histname+CatagoryName, histname+CatagoryName, TCHitBin, TCHitMin, TCHitMax);
      NumberOfRecHitsPerTrackCand->setAxisTitle("Number of RecHits per Track Candidate", 1);
      NumberOfRecHitsPerTrackCand->setAxisTitle("Number of Track Candidates",2);
    
      histname = "NumberOfRecHitsPerTrackCandVsPhiProfile_"+tcProducer.label() + "_";
      NumberOfRecHitsPerTrackCandVsPhiProfile = ibooker.bookProfile(histname+CatagoryName, histname+CatagoryName, PhiBin, PhiMin, PhiMax, TCHitBin, TCHitMin, TCHitMax,"s");
      NumberOfRecHitsPerTrackCandVsPhiProfile->setAxisTitle("Track Candidate #phi",1);
      NumberOfRecHitsPerTrackCandVsPhiProfile->setAxisTitle("Number of RecHits of each Track Candidate",2);
      
      histname = "NumberOfRecHitsPerTrackCandVsEtaProfile_"+tcProducer.label() + "_";
      NumberOfRecHitsPerTrackCandVsEtaProfile = ibooker.bookProfile(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax, TCHitBin, TCHitMin, TCHitMax,"s");
      NumberOfRecHitsPerTrackCandVsEtaProfile->setAxisTitle("Track Candidate #eta",1);
      NumberOfRecHitsPerTrackCandVsEtaProfile->setAxisTitle("Number of RecHits of each Track Candidate",2);
    }

    histname = "TrackCandPhiVsEta_"+tcProducer.label() + "_";
    TrackCandPhiVsEta = ibooker.book2D(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax, PhiBin, PhiMin, PhiMax);
    TrackCandPhiVsEta->setAxisTitle("Track Candidate #eta", 1);
    TrackCandPhiVsEta->setAxisTitle("Track Candidate #phi", 2);
  }
  
}

// -- Analyse
// ---------------------------------------------------------------------------------//
void TrackBuildingAnalyzer::analyze
(
    const edm::Event& iEvent,
    const edm::EventSetup& iSetup,
    const TrajectorySeed& candidate,
    const reco::BeamSpot& bs,
    const edm::ESHandle<MagneticField>& theMF,
    const edm::ESHandle<TransientTrackingRecHitBuilder>& theTTRHBuilder
)
{
  TSCBLBuilderNoMaterial tscblBuilder;
  
  //get parameters and errors from the candidate state
  auto const & theG = ((TkTransientTrackingRecHitBuilder const *)(theTTRHBuilder.product()))->geometry();
  auto const & candSS = candidate.startingState();
  TrajectoryStateOnSurface state = trajectoryStateTransform::transientState( candSS, &(theG->idToDet(candSS.detId())->surface()), theMF.product());
  TrajectoryStateClosestToBeamLine tsAtClosestApproachSeed = tscblBuilder(*state.freeState(),bs);//as in TrackProducerAlgorithm
  if(!(tsAtClosestApproachSeed.isValid())) {
    edm::LogVerbatim("TrackBuilding") << "TrajectoryStateClosestToBeamLine not valid";
    return;
  }
  GlobalPoint  v0 = tsAtClosestApproachSeed.trackStateAtPCA().position();
  GlobalVector p = tsAtClosestApproachSeed.trackStateAtPCA().momentum();
  GlobalPoint  v(v0.x()-bs.x0(),v0.y()-bs.y0(),v0.z()-bs.z0());
  
  double pt           = sqrt(state.globalMomentum().perp2());
  double eta          = state.globalPosition().eta();
  double phi          = state.globalPosition().phi();
  double theta        = state.globalPosition().theta();
  //double pm           = sqrt(state.globalMomentum().mag2());
  //double pz           = state.globalMomentum().z();
  //double qoverp       = tsAtClosestApproachSeed.trackStateAtPCA().charge()/p.mag();
  //double theta        = p.theta();
  //double lambda       = M_PI/2-p.theta();
  double numberOfHits = candidate.recHits().second-candidate.recHits().first;
  double dxy          = (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
  double dz           = v.z() - (v.x()*p.x()+v.y()*p.y())/p.perp() * p.z()/p.perp();
  
  // fill the ME's
  if (doAllSeedPlots || doQ)SeedQ->Fill( state.charge() );
  if (doAllSeedPlots || doPT) SeedPt->Fill( pt );
  if (doAllSeedPlots || doETA) SeedEta->Fill( eta );
  if (doAllSeedPlots || doPHI) SeedPhi->Fill( phi );
  if (doAllSeedPlots || doPHIVsETA) SeedPhiVsEta->Fill( eta, phi);
  if (doAllSeedPlots || doTheta) SeedTheta->Fill( theta );
  if (doAllSeedPlots || doDxy) SeedDxy->Fill( dxy );
  if (doAllSeedPlots || doDz) SeedDz->Fill( dz );
  if (doAllSeedPlots || doNRecHits) NumberOfRecHitsPerSeed->Fill( numberOfHits );
  if (doAllSeedPlots || doProfETA) NumberOfRecHitsPerSeedVsEtaProfile->Fill( eta, numberOfHits );
  if (doAllSeedPlots || doProfPHI) NumberOfRecHitsPerSeedVsPhiProfile->Fill( phi, numberOfHits );
  
}

// -- Analyse
// ---------------------------------------------------------------------------------//
void TrackBuildingAnalyzer::analyze
(
    const edm::Event& iEvent,
    const edm::EventSetup& iSetup,
    const TrackCandidate& candidate,
    const reco::BeamSpot& bs,
    const edm::ESHandle<MagneticField>& theMF,
    const edm::ESHandle<TransientTrackingRecHitBuilder>& theTTRHBuilder
)
{
  TSCBLBuilderNoMaterial tscblBuilder;
  
  //get parameters and errors from the candidate state
  auto const & theG = ((TkTransientTrackingRecHitBuilder const *)(theTTRHBuilder.product()))->geometry();
  auto const & candSS = candidate.trajectoryStateOnDet();
  TrajectoryStateOnSurface state = trajectoryStateTransform::transientState( candSS, &(theG->idToDet(candSS.detId())->surface()), theMF.product());
  TrajectoryStateClosestToBeamLine tsAtClosestApproachTrackCand = tscblBuilder(*state.freeState(),bs);//as in TrackProducerAlgorithm
  if(!(tsAtClosestApproachTrackCand.isValid())) {
    edm::LogVerbatim("TrackBuilding") << "TrajectoryStateClosestToBeamLine not valid";
    return;
  }
  GlobalPoint  v0 = tsAtClosestApproachTrackCand.trackStateAtPCA().position();
  GlobalVector p = tsAtClosestApproachTrackCand.trackStateAtPCA().momentum();
  GlobalPoint  v(v0.x()-bs.x0(),v0.y()-bs.y0(),v0.z()-bs.z0());
  
  double pt           = sqrt(state.globalMomentum().perp2());
  double eta          = state.globalPosition().eta();
  double phi          = state.globalPosition().phi();
  double theta        = state.globalPosition().theta();
  //double pm           = sqrt(state.globalMomentum().mag2());
  //double pz           = state.globalMomentum().z();
  //double qoverp       = tsAtClosestApproachTrackCand.trackStateAtPCA().charge()/p.mag();
  //double theta        = p.theta();
  //double lambda       = M_PI/2-p.theta();
  double numberOfHits = candidate.recHits().second-candidate.recHits().first;
  double dxy          = (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
  
  double dz           = v.z() - (v.x()*p.x()+v.y()*p.y())/p.perp() * p.z()/p.perp();

  if (doAllTCPlots || doStopSource) {
    // stopping source
    int max = stoppingSource->getNbinsX();
    double stop = candidate.stopReason() > max ? double(max-1) : static_cast<double>(candidate.stopReason());
    stoppingSource      ->Fill(stop);
    stoppingSourceVSeta ->Fill(eta,stop);
    stoppingSourceVSphi ->Fill(phi,stop);
  }

  if (doTCPlots){
    // fill the ME's
    if (doAllTCPlots) TrackCandQ->Fill( state.charge() );
    TrackCandPt->Fill( pt );
    TrackCandEta->Fill( eta );
    TrackCandPhi->Fill( phi );
    TrackCandPhiVsEta->Fill( eta, phi );
    if (doTheta) TrackCandTheta->Fill( theta );
    if (doAllTCPlots) TrackCandDxy->Fill( dxy );
    if (doAllTCPlots) TrackCandDz->Fill( dz );
    if (doAllTCPlots) NumberOfRecHitsPerTrackCand->Fill( numberOfHits );
    if (doAllTCPlots) NumberOfRecHitsPerTrackCandVsEtaProfile->Fill( eta, numberOfHits );
    if (doAllTCPlots) NumberOfRecHitsPerTrackCandVsPhiProfile->Fill( phi, numberOfHits );
  }
}
