#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
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
    : conf_( iConfig )
    , SeedPt(NULL)
    , SeedEta(NULL)
    , SeedPhi(NULL)
    , SeedTheta(NULL)
    , SeedQ(NULL)
    , SeedDxy(NULL)
    , SeedDz(NULL)
    , NumberOfRecHitsPerSeed(NULL)
    , NumberOfRecHitsPerSeedVsPhiProfile(NULL)
    , NumberOfRecHitsPerSeedVsEtaProfile(NULL)
{
}

TrackBuildingAnalyzer::~TrackBuildingAnalyzer() 
{ 
}

void TrackBuildingAnalyzer::beginJob(DQMStore * dqmStore_) 
{

    // parameters from the configuration
    std::string AlgoName       = conf_.getParameter<std::string>("AlgoName");
    std::string MEFolderName   = conf_.getParameter<std::string>("FolderName"); 

    // use the AlgoName and Quality Name 
    std::string CatagoryName = AlgoName;

    // get binning from the configuration
    int    TrackPtBin = conf_.getParameter<int>(   "TrackPtBin");
    double TrackPtMin = conf_.getParameter<double>("TrackPtMin");
    double TrackPtMax = conf_.getParameter<double>("TrackPtMax");

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

    int    SeedDxyBin = conf_.getParameter<int>(   "SeedDxyBin");
    double SeedDxyMin = conf_.getParameter<double>("SeedDxyMin");
    double SeedDxyMax = conf_.getParameter<double>("SeedDxyMax");

    int    SeedDzBin  = conf_.getParameter<int>(   "SeedDzBin");
    double SeedDzMin  = conf_.getParameter<double>("SeedDzMin");
    double SeedDzMax  = conf_.getParameter<double>("SeedDzMax");

    int    SeedHitBin = conf_.getParameter<int>(   "SeedHitBin");
    double SeedHitMin = conf_.getParameter<double>("SeedHitMin");
    double SeedHitMax = conf_.getParameter<double>("SeedHitMax");

    int    TCDxyBin   = conf_.getParameter<int>(   "TCDxyBin");
    double TCDxyMin   = conf_.getParameter<double>("TCDxyMin");
    double TCDxyMax   = conf_.getParameter<double>("TCDxyMax");

    int    TCDzBin    = conf_.getParameter<int>(   "TCDzBin");
    double TCDzMin    = conf_.getParameter<double>("TCDzMin");
    double TCDzMax    = conf_.getParameter<double>("TCDzMax");

    int    TCHitBin   = conf_.getParameter<int>(   "TCHitBin");
    double TCHitMin   = conf_.getParameter<double>("TCHitMin");
    double TCHitMax   = conf_.getParameter<double>("TCHitMax");

    dqmStore_->setCurrentFolder(MEFolderName);

    // book the Seed histograms
    // ---------------------------------------------------------------------------------//
    dqmStore_->setCurrentFolder(MEFolderName+"/TrackBuilding");

    histname = "SeedPt_";
    SeedPt = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TrackPtBin, TrackPtMin, TrackPtMax);
    SeedPt->setAxisTitle("Seed p_{T} (GeV/c)", 1);
    SeedPt->setAxisTitle("Number of Seeds", 2);

    histname = "SeedEta_";
    SeedEta = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax);
    SeedEta->setAxisTitle("Seed #eta", 1);
    SeedEta->setAxisTitle("Number of Seeds", 2);

    histname = "SeedPhi_";
    SeedPhi = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, PhiBin, PhiMin, PhiMax);
    SeedPhi->setAxisTitle("Seed #phi", 1);
    SeedPhi->setAxisTitle("Number of Seed", 2);

    histname = "SeedTheta_";
    SeedTheta = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, ThetaBin, ThetaMin, ThetaMax);
    SeedTheta->setAxisTitle("Seed #theta", 1);
    SeedTheta->setAxisTitle("Number of Seeds", 2);

    histname = "SeedQ_";
    SeedQ = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TrackQBin, TrackQMin, TrackQMax);
    SeedQ->setAxisTitle("Seed Charge", 1);
    SeedQ->setAxisTitle("Number of Seeds",2);

    histname = "SeedDxy_";
    SeedDxy = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, SeedDxyBin, SeedDxyMin, SeedDxyMax);
    SeedDxy->setAxisTitle("Seed d_{xy} (cm)", 1);
    SeedDxy->setAxisTitle("Number of Seeds",2);

    histname = "SeedDz_";
    SeedDz = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, SeedDzBin, SeedDzMin, SeedDzMax);
    SeedDz->setAxisTitle("Seed d_{z} (cm)", 1);
    SeedDz->setAxisTitle("Number of Seeds",2);

    histname = "NumberOfRecHitsPerSeed_";
    NumberOfRecHitsPerSeed = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, SeedHitBin, SeedHitMin, SeedHitMax);
    NumberOfRecHitsPerSeed->setAxisTitle("Number of RecHits per Seed", 1);
    NumberOfRecHitsPerSeed->setAxisTitle("Number of Seeds",2);

    histname = "NumberOfRecHitsPerSeedVsPhiProfile_";
    NumberOfRecHitsPerSeedVsPhiProfile = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, PhiBin, PhiMin, PhiMax, SeedHitBin, SeedHitMin, SeedHitMax,"s");
    NumberOfRecHitsPerSeedVsPhiProfile->setAxisTitle("Seed #phi",1);
    NumberOfRecHitsPerSeedVsPhiProfile->setAxisTitle("Number of RecHits of each Seed",2);

    histname = "NumberOfRecHitsPerSeedVsEtaProfile_";
    NumberOfRecHitsPerSeedVsEtaProfile = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax, SeedHitBin, SeedHitMin, SeedHitMax,"s");
    NumberOfRecHitsPerSeedVsEtaProfile->setAxisTitle("Seed #eta",1);
    NumberOfRecHitsPerSeedVsEtaProfile->setAxisTitle("Number of RecHits of each Seed",2);

    // book the TrackCandidate histograms
    // ---------------------------------------------------------------------------------//
    dqmStore_->setCurrentFolder(MEFolderName+"/TrackBuilding");

    histname = "TrackCandPt_";
    TrackCandPt = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TrackPtBin, TrackPtMin, TrackPtMax);
    TrackCandPt->setAxisTitle("Track Candidate p_{T} (GeV/c)", 1);
    TrackCandPt->setAxisTitle("Number of Track Candidates", 2);

    histname = "TrackCandEta_";
    TrackCandEta = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax);
    TrackCandEta->setAxisTitle("Track Candidate #eta", 1);
    TrackCandEta->setAxisTitle("Number of Track Candidates", 2);

    histname = "TrackCandPhi_";
    TrackCandPhi = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, PhiBin, PhiMin, PhiMax);
    TrackCandPhi->setAxisTitle("Track Candidate #phi", 1);
    TrackCandPhi->setAxisTitle("Number of Track Candidates", 2);

    histname = "TrackCandTheta_";
    TrackCandTheta = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, ThetaBin, ThetaMin, ThetaMax);
    TrackCandTheta->setAxisTitle("Track Candidate #theta", 1);
    TrackCandTheta->setAxisTitle("Number of Track Candidates", 2);

    histname = "TrackCandQ_";
    TrackCandQ = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TrackQBin, TrackQMin, TrackQMax);
    TrackCandQ->setAxisTitle("Track Candidate Charge", 1);
    TrackCandQ->setAxisTitle("Number of Track Candidates",2);

    histname = "TrackCandDxy_";
    TrackCandDxy = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TCDxyBin, TCDxyMin, TCDxyMax);
    TrackCandDxy->setAxisTitle("Track Candidate d_{xy} (cm)", 1);
    TrackCandDxy->setAxisTitle("Number of Track Candidates",2);

    histname = "TrackCandDz_";
    TrackCandDz = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TCDzBin, TCDzMin, TCDzMax);
    TrackCandDz->setAxisTitle("Track Candidate d_{z} (cm)", 1);
    TrackCandDz->setAxisTitle("Number of Track Candidates",2);

    histname = "NumberOfRecHitsPerTrackCand_";
    NumberOfRecHitsPerTrackCand = dqmStore_->book1D(histname+CatagoryName, histname+CatagoryName, TCHitBin, TCHitMin, TCHitMax);
    NumberOfRecHitsPerTrackCand->setAxisTitle("Number of RecHits per Track Candidate", 1);
    NumberOfRecHitsPerTrackCand->setAxisTitle("Number of Track Candidates",2);

    histname = "NumberOfRecHitsPerTrackCandVsPhiProfile_";
    NumberOfRecHitsPerTrackCandVsPhiProfile = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, PhiBin, PhiMin, PhiMax, TCHitBin, TCHitMin, TCHitMax,"s");
    NumberOfRecHitsPerTrackCandVsPhiProfile->setAxisTitle("Track Candidate #phi",1);
    NumberOfRecHitsPerTrackCandVsPhiProfile->setAxisTitle("Number of RecHits of each Track Candidate",2);

    histname = "NumberOfRecHitsPerTrackCandVsEtaProfile_";
    NumberOfRecHitsPerTrackCandVsEtaProfile = dqmStore_->bookProfile(histname+CatagoryName, histname+CatagoryName, EtaBin, EtaMin, EtaMax, TCHitBin, TCHitMin, TCHitMax,"s");
    NumberOfRecHitsPerTrackCandVsEtaProfile->setAxisTitle("Track Candidate #eta",1);
    NumberOfRecHitsPerTrackCandVsEtaProfile->setAxisTitle("Number of RecHits of each Track Candidate",2);
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
    using namespace edm;
    using std::string;

    TrajectoryStateTransform tsTransform;
    TSCBLBuilderNoMaterial tscblBuilder;

    //get parameters and errors from the candidate state
    TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(&*(candidate.recHits().second-1));
    TrajectoryStateOnSurface state = tsTransform.transientState( candidate.startingState(), recHit->surface(), theMF.product());
    TrajectoryStateClosestToBeamLine tsAtClosestApproachSeed = tscblBuilder(*state.freeState(),bs);//as in TrackProducerAlgorithm
    if(!(tsAtClosestApproachSeed.isValid())) {
        edm::LogVerbatim("TrackBuilding") << "TrajectoryStateClosestToBeamLine not valid";
        return;
    }
    GlobalPoint  v0 = tsAtClosestApproachSeed.trackStateAtPCA().position();
    GlobalVector p = tsAtClosestApproachSeed.trackStateAtPCA().momentum();
    GlobalPoint  v(v0.x()-bs.x0(),v0.y()-bs.y0(),v0.z()-bs.z0());

    double pt           = sqrt(state.globalMomentum().perp2());
    double eta          = state.globalMomentum().eta();
    double phi          = state.globalMomentum().phi();
    double theta        = state.globalMomentum().theta();
    //double pm           = sqrt(state.globalMomentum().mag2());
    //double pz           = state.globalMomentum().z();
    //double qoverp       = tsAtClosestApproachSeed.trackStateAtPCA().charge()/p.mag();
    //double theta        = p.theta();
    //double lambda       = M_PI/2-p.theta();
    double numberOfHits = candidate.recHits().second-candidate.recHits().first;
    double dxy          = (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
    double dz           = v.z() - (v.x()*p.x()+v.y()*p.y())/p.perp() * p.z()/p.perp();

    // fill the ME's
    SeedQ->Fill( state.charge() );
    SeedPt->Fill( pt );
    SeedEta->Fill( eta );
    SeedPhi->Fill( phi );
    SeedTheta->Fill( theta );
    SeedDxy->Fill( dxy );
    SeedDz->Fill( dz );
    NumberOfRecHitsPerSeed->Fill( numberOfHits );
    NumberOfRecHitsPerSeedVsEtaProfile->Fill( eta, numberOfHits );
    NumberOfRecHitsPerSeedVsPhiProfile->Fill( phi, numberOfHits );
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
    using namespace edm;
    using std::string;

    TrajectoryStateTransform tsTransform;
    TSCBLBuilderNoMaterial tscblBuilder;

    //get parameters and errors from the candidate state
    TransientTrackingRecHit::RecHitPointer recHit = theTTRHBuilder->build(&*(candidate.recHits().second-1));
    TrajectoryStateOnSurface state = tsTransform.transientState( candidate.trajectoryStateOnDet(), recHit->surface(), theMF.product());
    TrajectoryStateClosestToBeamLine tsAtClosestApproachTrackCand = tscblBuilder(*state.freeState(),bs);//as in TrackProducerAlgorithm
    if(!(tsAtClosestApproachTrackCand.isValid())) {
        edm::LogVerbatim("TrackBuilding") << "TrajectoryStateClosestToBeamLine not valid";
        return;
    }
    GlobalPoint  v0 = tsAtClosestApproachTrackCand.trackStateAtPCA().position();
    GlobalVector p = tsAtClosestApproachTrackCand.trackStateAtPCA().momentum();
    GlobalPoint  v(v0.x()-bs.x0(),v0.y()-bs.y0(),v0.z()-bs.z0());

    double pt           = sqrt(state.globalMomentum().perp2());
    double eta          = state.globalMomentum().eta();
    double phi          = state.globalMomentum().phi();
    double theta        = state.globalMomentum().theta();
    //double pm           = sqrt(state.globalMomentum().mag2());
    //double pz           = state.globalMomentum().z();
    //double qoverp       = tsAtClosestApproachTrackCand.trackStateAtPCA().charge()/p.mag();
    //double theta        = p.theta();
    //double lambda       = M_PI/2-p.theta();
    double numberOfHits = candidate.recHits().second-candidate.recHits().first;
    double dxy          = (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
    double dz           = v.z() - (v.x()*p.x()+v.y()*p.y())/p.perp() * p.z()/p.perp();

    // fill the ME's
    TrackCandQ->Fill( state.charge() );
    TrackCandPt->Fill( pt );
    TrackCandEta->Fill( eta );
    TrackCandPhi->Fill( phi );
    TrackCandTheta->Fill( theta );
    TrackCandDxy->Fill( dxy );
    TrackCandDz->Fill( dz );
    NumberOfRecHitsPerTrackCand->Fill( numberOfHits );
    NumberOfRecHitsPerTrackCandVsEtaProfile->Fill( eta, numberOfHits );
    NumberOfRecHitsPerTrackCandVsPhiProfile->Fill( phi, numberOfHits );
}
