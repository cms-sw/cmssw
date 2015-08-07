#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "HeavyIonsAnalysis/PhotonAnalysis/src/pfIsoCalculator.h"

#include "HeavyIonsAnalysis/PhotonAnalysis/interface/ggHiNtuplizer.h"
#include "HeavyIonsAnalysis/PhotonAnalysis/interface/GenParticleParentage.h"

using namespace std;

ggHiNtuplizer::ggHiNtuplizer(const edm::ParameterSet& ps)
{

   // class instance configuration
   doGenParticles_         = ps.getParameter<bool>("doGenParticles");
   runOnParticleGun_       = ps.getParameter<bool>("runOnParticleGun");
   useValMapIso_           = ps.getParameter<bool>("useValMapIso");
   genPileupCollection_    = consumes<vector<PileupSummaryInfo> >   (ps.getParameter<edm::InputTag>("pileupCollection"));
   genParticlesCollection_ = consumes<vector<reco::GenParticle> >   (ps.getParameter<edm::InputTag>("genParticleSrc"));
   gsfElectronsCollection_ = consumes<edm::View<reco::GsfElectron> >(ps.getParameter<edm::InputTag>("gsfElectronLabel"));
   recoPhotonsCollection_  = consumes<edm::View<reco::Photon> >     (ps.getParameter<edm::InputTag>("recoPhotonSrc"));
   recoMuonsCollection_    = consumes<edm::View<reco::Muon> >       (ps.getParameter<edm::InputTag>("recoMuonSrc"));
   vtxCollection_          = consumes<vector<reco::Vertex> >        (ps.getParameter<edm::InputTag>("VtxLabel"));
   if(useValMapIso_){
     recoPhotonsHiIso_ = consumes<edm::ValueMap<reco::HIPhotonIsolation> > (ps.getParameter<edm::InputTag>("recoPhotonHiIsolationMap"));
   }

   if(doPfIso_){
     pfCollection_ = consumes<edm::View<reco::PFCandidate> > (ps.getParameter<edm::InputTag>("particleFlowCollection"));
     if(doVsIso_){
       voronoiBkgCalo_ = consumes<edm::ValueMap<reco::VoronoiBackground> > (ps.getParameter<edm::InputTag>("voronoiBackgroundCalo"));
       voronoiBkgPF_ = consumes<edm::ValueMap<reco::VoronoiBackground> > (ps.getParameter<edm::InputTag>("voronoiBackgroundPF"));
     }
   }


   // initialize output TTree
   edm::Service<TFileService> fs;
   tree_ = fs->make<TTree>("EventTree", "Event data");

   tree_->Branch("run",    &run_);
   tree_->Branch("event",  &event_);
   tree_->Branch("lumis",  &lumis_);
   tree_->Branch("isData", &isData_);

   if (doGenParticles_) {
      tree_->Branch("nPUInfo",      &nPUInfo_);
      tree_->Branch("nPU",          &nPU_);
      tree_->Branch("puBX",         &puBX_);
      tree_->Branch("puTrue",       &puTrue_);

      tree_->Branch("nMC",          &nMC_);
      tree_->Branch("mcPID",        &mcPID_);
      tree_->Branch("mcStatus",     &mcStatus_);
      tree_->Branch("mcVtx_x",      &mcVtx_x_);
      tree_->Branch("mcVtx_y",      &mcVtx_y_);
      tree_->Branch("mcVtx_z",      &mcVtx_z_);
      tree_->Branch("mcPt",         &mcPt_);
      tree_->Branch("mcEta",        &mcEta_);
      tree_->Branch("mcPhi",        &mcPhi_);
      tree_->Branch("mcE",          &mcE_);
      tree_->Branch("mcEt",         &mcEt_);
      tree_->Branch("mcMass",       &mcMass_);
      tree_->Branch("mcParentage",  &mcParentage_);
      tree_->Branch("mcMomPID",     &mcMomPID_);
      tree_->Branch("mcMomPt",      &mcMomPt_);
      tree_->Branch("mcMomEta",     &mcMomEta_);
      tree_->Branch("mcMomPhi",     &mcMomPhi_);
      tree_->Branch("mcMomMass",    &mcMomMass_);
      tree_->Branch("mcGMomPID",    &mcGMomPID_);
      tree_->Branch("mcIndex",      &mcIndex_);
      tree_->Branch("mcCalIsoDR03", &mcCalIsoDR03_);
      tree_->Branch("mcCalIsoDR04", &mcCalIsoDR04_);
      tree_->Branch("mcTrkIsoDR03", &mcTrkIsoDR03_);
      tree_->Branch("mcTrkIsoDR04", &mcTrkIsoDR04_);
   }

   tree_->Branch("nEle",                  &nEle_);
   tree_->Branch("eleCharge",             &eleCharge_);
   tree_->Branch("eleChargeConsistent",   &eleChargeConsistent_);
   tree_->Branch("eleEn",                 &eleEn_);
   tree_->Branch("eleD0",                 &eleD0_);
   tree_->Branch("eleDz",                 &eleDz_);
   tree_->Branch("eleD0Err",              &eleD0Err_);
   tree_->Branch("eleDzErr",              &eleDzErr_);
   tree_->Branch("eleTrkPt",              &eleTrkPt_);
   tree_->Branch("eleTrkEta",             &eleTrkEta_);
   tree_->Branch("eleTrkPhi",             &eleTrkPhi_);
   tree_->Branch("eleTrkCharge",          &eleTrkCharge_);
   tree_->Branch("eleTrkChi2",            &eleTrkChi2_);
   tree_->Branch("eleTrkNdof",            &eleTrkNdof_);
   tree_->Branch("eleTrkNormalizedChi2",  &eleTrkNormalizedChi2_);

   tree_->Branch("elePt",                 &elePt_);
   tree_->Branch("eleEta",                &eleEta_);
   tree_->Branch("elePhi",                &elePhi_);
   tree_->Branch("eleSCEn",               &eleSCEn_);
   tree_->Branch("eleESEn",               &eleESEn_);
   tree_->Branch("eleSCEta",              &eleSCEta_);
   tree_->Branch("eleSCPhi",              &eleSCPhi_);
   tree_->Branch("eleSCRawEn",            &eleSCRawEn_);
   tree_->Branch("eleSCEtaWidth",         &eleSCEtaWidth_);
   tree_->Branch("eleSCPhiWidth",         &eleSCPhiWidth_);
   tree_->Branch("eleHoverE",             &eleHoverE_);
   tree_->Branch("eleEoverP",             &eleEoverP_);
   tree_->Branch("eleEoverPInv",          &eleEoverPInv_);
   tree_->Branch("eleBrem",               &eleBrem_);
   tree_->Branch("eledEtaAtVtx",          &eledEtaAtVtx_);
   tree_->Branch("eledPhiAtVtx",          &eledPhiAtVtx_);
   tree_->Branch("eleSigmaIEtaIEta",      &eleSigmaIEtaIEta_);
   tree_->Branch("eleSigmaIEtaIEta_2012", &eleSigmaIEtaIEta_2012_);
   tree_->Branch("eleSigmaIPhiIPhi",      &eleSigmaIPhiIPhi_);
// tree_->Branch("eleConvVeto",           &eleConvVeto_);  // TODO: not available in reco::
   tree_->Branch("eleMissHits",           &eleMissHits_);
   tree_->Branch("eleESEffSigmaRR",       &eleESEffSigmaRR_);
   tree_->Branch("elePFChIso",            &elePFChIso_);
   tree_->Branch("elePFPhoIso",           &elePFPhoIso_);
   tree_->Branch("elePFNeuIso",           &elePFNeuIso_);
   tree_->Branch("elePFPUIso",            &elePFPUIso_);
   tree_->Branch("eleBC1E",               &eleBC1E_);
   tree_->Branch("eleBC1Eta",             &eleBC1Eta_);
   tree_->Branch("eleBC2E",               &eleBC2E_);
   tree_->Branch("eleBC2Eta",             &eleBC2Eta_);

   tree_->Branch("nPho",                  &nPho_);
   tree_->Branch("phoE",                  &phoE_);
   tree_->Branch("phoEt",                 &phoEt_);
   tree_->Branch("phoEta",                &phoEta_);
   tree_->Branch("phoPhi",                &phoPhi_);
   tree_->Branch("phoSCE",                &phoSCE_);
   tree_->Branch("phoSCRawE",             &phoSCRawE_);
   tree_->Branch("phoESEn",               &phoESEn_);
   tree_->Branch("phoSCEta",              &phoSCEta_);
   tree_->Branch("phoSCPhi",              &phoSCPhi_);
   tree_->Branch("phoSCEtaWidth",         &phoSCEtaWidth_);
   tree_->Branch("phoSCPhiWidth",         &phoSCPhiWidth_);
   tree_->Branch("phoSCBrem",             &phoSCBrem_);
   tree_->Branch("phohasPixelSeed",       &phohasPixelSeed_);
// tree_->Branch("phoEleVeto",            &phoEleVeto_);        // TODO: not available in reco::
   tree_->Branch("phoR9",                 &phoR9_);
   tree_->Branch("phoHoverE",             &phoHoverE_);
   tree_->Branch("phoSigmaIEtaIEta",      &phoSigmaIEtaIEta_);  // TODO: not available in reco::
// tree_->Branch("phoSigmaIEtaIPhi",      &phoSigmaIEtaIPhi_);  // TODO: not available in reco::
// tree_->Branch("phoSigmaIPhiIPhi",      &phoSigmaIPhiIPhi_);  // TODO: not available in reco::
   tree_->Branch("phoE1x3",               &phoE1x3_);
   tree_->Branch("phoE2x2",               &phoE2x2_);
   tree_->Branch("phoE2x5Max",            &phoE2x5Max_);
   tree_->Branch("phoE5x5",               &phoE5x5_);
   tree_->Branch("phoESEffSigmaRR",       &phoESEffSigmaRR_);
   tree_->Branch("phoSigmaIEtaIEta_2012", &phoSigmaIEtaIEta_2012_);
   tree_->Branch("phoSigmaIEtaIPhi_2012", &phoSigmaIEtaIPhi_2012_);
   tree_->Branch("phoSigmaIPhiIPhi_2012", &phoSigmaIPhiIPhi_2012_);
   tree_->Branch("phoE1x3_2012",          &phoE1x3_2012_);
   tree_->Branch("phoE2x2_2012",          &phoE2x2_2012_);
   tree_->Branch("phoE2x5Max_2012",       &phoE2x5Max_2012_);
   tree_->Branch("phoE5x5_2012",          &phoE5x5_2012_);
   tree_->Branch("phoBC1E",               &phoBC1E_);
   tree_->Branch("phoBC1Eta",             &phoBC1Eta_);
   tree_->Branch("phoBC2E",               &phoBC2E_);
   tree_->Branch("phoBC2Eta",             &phoBC2Eta_);
   tree_->Branch("pho_ecalClusterIsoR2", &pho_ecalClusterIsoR2_);
   tree_->Branch("pho_ecalClusterIsoR3", &pho_ecalClusterIsoR3_);
   tree_->Branch("pho_ecalClusterIsoR4", &pho_ecalClusterIsoR4_);
   tree_->Branch("pho_ecalClusterIsoR5", &pho_ecalClusterIsoR5_);
   tree_->Branch("pho_hcalRechitIsoR1", &pho_hcalRechitIsoR1_);
   tree_->Branch("pho_hcalRechitIsoR2", &pho_hcalRechitIsoR2_);
   tree_->Branch("pho_hcalRechitIsoR3", &pho_hcalRechitIsoR3_);
   tree_->Branch("pho_hcalRechitIsoR4", &pho_hcalRechitIsoR4_);
   tree_->Branch("pho_hcalRechitIsoR5", &pho_hcalRechitIsoR5_);
   tree_->Branch("pho_trackIsoR1PtCut20", &pho_trackIsoR1PtCut20_);
   tree_->Branch("pho_trackIsoR2PtCut20", &pho_trackIsoR2PtCut20_);
   tree_->Branch("pho_trackIsoR3PtCut20", &pho_trackIsoR3PtCut20_);
   tree_->Branch("pho_trackIsoR4PtCut20", &pho_trackIsoR4PtCut20_);
   tree_->Branch("pho_trackIsoR5PtCut20", &pho_trackIsoR5PtCut20_);
   tree_->Branch("pho_swissCrx", &pho_swissCrx_);
   tree_->Branch("pho_seedTime", &pho_seedTime_);

   if(doPfIso_){
     tree_->Branch("pfcIso1",&pfcIso1);
     tree_->Branch("pfcIso2",&pfcIso2);
     tree_->Branch("pfcIso3",&pfcIso3);
     tree_->Branch("pfcIso4",&pfcIso4);
     tree_->Branch("pfcIso5",&pfcIso5);

     tree_->Branch("pfpIso1",&pfpIso1);
     tree_->Branch("pfpIso2",&pfpIso2);
     tree_->Branch("pfpIso3",&pfpIso3);
     tree_->Branch("pfpIso4",&pfpIso4);
     tree_->Branch("pfpIso5",&pfpIso5);

     tree_->Branch("pfnIso1",&pfnIso1);
     tree_->Branch("pfnIso2",&pfnIso2);
     tree_->Branch("pfnIso3",&pfnIso3);
     tree_->Branch("pfnIso4",&pfnIso4);
     tree_->Branch("pfnIso5",&pfnIso5);

     tree_->Branch("pfsumIso1",&pfsumIso1);
     tree_->Branch("pfsumIso2",&pfsumIso2);
     tree_->Branch("pfsumIso3",&pfsumIso3);
     tree_->Branch("pfsumIso4",&pfsumIso4);
     tree_->Branch("pfsumIso5",&pfsumIso5);

     if(doVsIso_)
     {
       tree_->Branch("pfcVsIso1",&pfcVsIso1);
       tree_->Branch("pfcVsIso2",&pfcVsIso2);
       tree_->Branch("pfcVsIso3",&pfcVsIso3);
       tree_->Branch("pfcVsIso4",&pfcVsIso4);
       tree_->Branch("pfcVsIso5",&pfcVsIso5);
       tree_->Branch("pfcVsIso1th1",&pfcVsIso1th1);
       tree_->Branch("pfcVsIso2th1",&pfcVsIso2th1);
       tree_->Branch("pfcVsIso3th1",&pfcVsIso3th1);
       tree_->Branch("pfcVsIso4th1",&pfcVsIso4th1);
       tree_->Branch("pfcVsIso5th1",&pfcVsIso5th1);
       tree_->Branch("pfcVsIso1th2",&pfcVsIso1th2);
       tree_->Branch("pfcVsIso2th2",&pfcVsIso2th2);
       tree_->Branch("pfcVsIso3th2",&pfcVsIso3th2);
       tree_->Branch("pfcVsIso4th2",&pfcVsIso4th2);
       tree_->Branch("pfcVsIso5th2",&pfcVsIso5th2);

       tree_->Branch("pfnVsIso1",&pfnVsIso1);
       tree_->Branch("pfnVsIso2",&pfnVsIso2);
       tree_->Branch("pfnVsIso3",&pfnVsIso3);
       tree_->Branch("pfnVsIso4",&pfnVsIso4);
       tree_->Branch("pfnVsIso5",&pfnVsIso5);
       tree_->Branch("pfnVsIso1th1",&pfnVsIso1th1);
       tree_->Branch("pfnVsIso2th1",&pfnVsIso2th1);
       tree_->Branch("pfnVsIso3th1",&pfnVsIso3th1);
       tree_->Branch("pfnVsIso4th1",&pfnVsIso4th1);
       tree_->Branch("pfnVsIso5th1",&pfnVsIso5th1);
       tree_->Branch("pfnVsIso1th2",&pfnVsIso1th2);
       tree_->Branch("pfnVsIso2th2",&pfnVsIso2th2);
       tree_->Branch("pfnVsIso3th2",&pfnVsIso3th2);
       tree_->Branch("pfnVsIso4th2",&pfnVsIso4th2);
       tree_->Branch("pfnVsIso5th2",&pfnVsIso5th2);

       tree_->Branch("pfpVsIso1",&pfpVsIso1);
       tree_->Branch("pfpVsIso2",&pfpVsIso2);
       tree_->Branch("pfpVsIso3",&pfpVsIso3);
       tree_->Branch("pfpVsIso4",&pfpVsIso4);
       tree_->Branch("pfpVsIso5",&pfpVsIso5);
       tree_->Branch("pfpVsIso1th1",&pfpVsIso1th1);
       tree_->Branch("pfpVsIso2th1",&pfpVsIso2th1);
       tree_->Branch("pfpVsIso3th1",&pfpVsIso3th1);
       tree_->Branch("pfpVsIso4th1",&pfpVsIso4th1);
       tree_->Branch("pfpVsIso5th1",&pfpVsIso5th1);
       tree_->Branch("pfpVsIso1th2",&pfpVsIso1th2);
       tree_->Branch("pfpVsIso2th2",&pfpVsIso2th2);
       tree_->Branch("pfpVsIso3th2",&pfpVsIso3th2);
       tree_->Branch("pfpVsIso4th2",&pfpVsIso4th2);
       tree_->Branch("pfpVsIso5th2",&pfpVsIso5th2);

       tree_->Branch("pfsumVsIso1",&pfsumVsIso1);
       tree_->Branch("pfsumVsIso2",&pfsumVsIso2);
       tree_->Branch("pfsumVsIso3",&pfsumVsIso3);
       tree_->Branch("pfsumVsIso4",&pfsumVsIso4);
       tree_->Branch("pfsumVsIso5",&pfsumVsIso5);
       tree_->Branch("pfsumVsIso1th1",&pfsumVsIso1th1);
       tree_->Branch("pfsumVsIso2th1",&pfsumVsIso2th1);
       tree_->Branch("pfsumVsIso3th1",&pfsumVsIso3th1);
       tree_->Branch("pfsumVsIso4th1",&pfsumVsIso4th1);
       tree_->Branch("pfsumVsIso5th1",&pfsumVsIso5th1);
       tree_->Branch("pfsumVsIso1th2",&pfsumVsIso1th2);
       tree_->Branch("pfsumVsIso2th2",&pfsumVsIso2th2);
       tree_->Branch("pfsumVsIso3th2",&pfsumVsIso3th2);
       tree_->Branch("pfsumVsIso4th2",&pfsumVsIso4th2);
       tree_->Branch("pfsumVsIso5th2",&pfsumVsIso5th2);


       tree_->Branch("pfVsSubIso1",&pfVsSubIso1);
       tree_->Branch("pfVsSubIso2",&pfVsSubIso2);
       tree_->Branch("pfVsSubIso3",&pfVsSubIso3);
       tree_->Branch("pfVsSubIso4",&pfVsSubIso4);
       tree_->Branch("pfVsSubIso5",&pfVsSubIso5);


       tree_->Branch("towerIso1",&towerIso1);
       tree_->Branch("towerIso2",&towerIso2);
       tree_->Branch("towerIso3",&towerIso3);
       tree_->Branch("towerIso4",&towerIso4);
       tree_->Branch("towerIso5",&towerIso5);
       tree_->Branch("towerVsIso1",&towerVsIso1);
       tree_->Branch("towerVsIso2",&towerVsIso2);
       tree_->Branch("towerVsIso3",&towerVsIso3);
       tree_->Branch("towerVsIso4",&towerVsIso4);
       tree_->Branch("towerVsIso5",&towerVsIso5);
       tree_->Branch("towerVsSubIso1",&towerVsSubIso1);
       tree_->Branch("towerVsSubIso2",&towerVsSubIso2);
       tree_->Branch("towerVsSubIso3",&towerVsSubIso3);
       tree_->Branch("towerVsSubIso4",&towerVsSubIso4);
       tree_->Branch("towerVsSubIso5",&towerVsSubIso5);
     }

   }


   tree_->Branch("nMu",                   &nMu_);
   tree_->Branch("muPt",                  &muPt_);
   tree_->Branch("muEta",                 &muEta_);
   tree_->Branch("muPhi",                 &muPhi_);
   tree_->Branch("muCharge",              &muCharge_);
   tree_->Branch("muType",                &muType_);
   tree_->Branch("muIsGood",              &muIsGood_);
   tree_->Branch("muD0",                  &muD0_);
   tree_->Branch("muDz",                  &muDz_);
   tree_->Branch("muChi2NDF",             &muChi2NDF_);
   tree_->Branch("muInnerD0",             &muInnerD0_);
   tree_->Branch("muInnerDz",             &muInnerDz_);
   tree_->Branch("muTrkLayers",           &muTrkLayers_);
   tree_->Branch("muPixelLayers",         &muPixelLayers_);
   tree_->Branch("muPixelHits",           &muPixelHits_);
   tree_->Branch("muMuonHits",            &muMuonHits_);
   tree_->Branch("muTrkQuality",          &muTrkQuality_);
   tree_->Branch("muStations",            &muStations_);
   tree_->Branch("muIsoTrk",              &muIsoTrk_);
   tree_->Branch("muPFChIso",             &muPFChIso_);
   tree_->Branch("muPFPhoIso",            &muPFPhoIso_);
   tree_->Branch("muPFNeuIso",            &muPFNeuIso_);
   tree_->Branch("muPFPUIso",             &muPFPUIso_);
}

void ggHiNtuplizer::analyze(const edm::Event& e, const edm::EventSetup& es)
{

   // cleanup from previous event
   nPUInfo_ = 0;
   nMC_ = 0;
   nEle_ = 0;
   nPho_ = 0;
   nMu_ = 0;

   nPU_                  .clear();
   puBX_                 .clear();
   puTrue_               .clear();

   mcPID_                .clear();
   mcStatus_             .clear();
   mcVtx_x_              .clear();
   mcVtx_y_              .clear();
   mcVtx_z_              .clear();
   mcPt_                 .clear();
   mcEta_                .clear();
   mcPhi_                .clear();
   mcE_                  .clear();
   mcEt_                 .clear();
   mcMass_               .clear();
   mcParentage_          .clear();
   mcMomPID_             .clear();
   mcMomPt_              .clear();
   mcMomEta_             .clear();
   mcMomPhi_             .clear();
   mcMomMass_            .clear();
   mcGMomPID_            .clear();
   mcIndex_              .clear();
   mcCalIsoDR03_         .clear();
   mcCalIsoDR04_         .clear();
   mcTrkIsoDR03_         .clear();
   mcTrkIsoDR04_         .clear();

   eleCharge_            .clear();
   eleChargeConsistent_  .clear();
   eleEn_                .clear();
   eleD0_                .clear();
   eleDz_                .clear();
   eleD0Err_             .clear();
   eleDzErr_             .clear();
   eleTrkPt_             .clear();
   eleTrkEta_            .clear();
   eleTrkPhi_            .clear();
   eleTrkCharge_         .clear();
   eleTrkChi2_           .clear();
   eleTrkNdof_           .clear();
   eleTrkNormalizedChi2_ .clear();
   elePt_                .clear();
   eleEta_               .clear();
   elePhi_               .clear();
   eleSCEn_              .clear();
   eleESEn_              .clear();
   eleSCEta_             .clear();
   eleSCPhi_             .clear();
   eleSCRawEn_           .clear();
   eleSCEtaWidth_        .clear();
   eleSCPhiWidth_        .clear();
   eleHoverE_            .clear();
   eleEoverP_            .clear();
   eleEoverPInv_         .clear();
   eleBrem_              .clear();
   eledEtaAtVtx_         .clear();
   eledPhiAtVtx_         .clear();
   eleSigmaIEtaIEta_     .clear();
   eleSigmaIEtaIEta_2012_.clear();
   eleSigmaIPhiIPhi_     .clear();
// eleConvVeto_          .clear();  // TODO: not available in reco::
   eleMissHits_          .clear();
   eleESEffSigmaRR_      .clear();
   elePFChIso_           .clear();
   elePFPhoIso_          .clear();
   elePFNeuIso_          .clear();
   elePFPUIso_           .clear();
   eleBC1E_              .clear();
   eleBC1Eta_            .clear();
   eleBC2E_              .clear();
   eleBC2Eta_            .clear();

   phoE_                 .clear();
   phoEt_                .clear();
   phoEta_               .clear();
   phoPhi_               .clear();
   phoSCE_               .clear();
   phoSCRawE_            .clear();
   phoESEn_              .clear();
   phoSCEta_             .clear();
   phoSCPhi_             .clear();
   phoSCEtaWidth_        .clear();
   phoSCPhiWidth_        .clear();
   phoSCBrem_            .clear();
   phohasPixelSeed_      .clear();
// phoEleVeto_           .clear();  // TODO: not available in reco::
   phoR9_                .clear();
   phoHoverE_            .clear();
   phoSigmaIEtaIEta_     .clear();  // TODO: not available in reco::
// phoSigmaIEtaIPhi_     .clear();  // TODO: not available in reco::
// phoSigmaIPhiIPhi_     .clear();  // TODO: not available in reco::
   phoE1x3_              .clear();
   phoE2x2_              .clear();
   phoE2x5Max_           .clear();
   phoE5x5_              .clear();
   phoESEffSigmaRR_      .clear();
   phoSigmaIEtaIEta_2012_.clear();
   phoSigmaIEtaIPhi_2012_.clear();
   phoSigmaIPhiIPhi_2012_.clear();
   phoE1x3_2012_         .clear();
   phoE2x2_2012_         .clear();
   phoE2x5Max_2012_      .clear();
   phoE5x5_2012_         .clear();
   phoBC1E_              .clear();
   phoBC1Eta_            .clear();
   phoBC2E_              .clear();
   phoBC2Eta_            .clear();
   pho_ecalClusterIsoR2_.clear();
   pho_ecalClusterIsoR3_.clear();
   pho_ecalClusterIsoR4_.clear();
   pho_ecalClusterIsoR5_.clear();
   pho_hcalRechitIsoR1_.clear();
   pho_hcalRechitIsoR2_.clear();
   pho_hcalRechitIsoR3_.clear();
   pho_hcalRechitIsoR4_.clear();
   pho_hcalRechitIsoR5_.clear();
   pho_trackIsoR1PtCut20_.clear();
   pho_trackIsoR2PtCut20_.clear();
   pho_trackIsoR3PtCut20_.clear();
   pho_trackIsoR4PtCut20_.clear();
   pho_trackIsoR5PtCut20_.clear();
   pho_swissCrx_.clear();
   pho_seedTime_.clear();

   //photon pf isolation stuff
   pfcIso1.clear();
   pfcIso2.clear();
   pfcIso3.clear();
   pfcIso4.clear();
   pfcIso5.clear();
   pfpIso1.clear();
   pfpIso2.clear();
   pfpIso3.clear();
   pfpIso4.clear();
   pfpIso5.clear();
   pfnIso1.clear();
   pfnIso2.clear();
   pfnIso3.clear();
   pfnIso4.clear();
   pfnIso5.clear();
   pfsumIso1.clear();
   pfsumIso2.clear();
   pfsumIso3.clear();
   pfsumIso4.clear();
   pfsumIso5.clear();
   pfcVsIso1.clear();
   pfcVsIso2.clear();
   pfcVsIso3.clear();
   pfcVsIso4.clear();
   pfcVsIso5.clear();
   pfcVsIso1th1.clear();
   pfcVsIso2th1.clear();
   pfcVsIso3th1.clear();
   pfcVsIso4th1.clear();
   pfcVsIso5th1.clear();
   pfcVsIso1th2.clear();
   pfcVsIso2th2.clear();
   pfcVsIso3th2.clear();
   pfcVsIso4th2.clear();
   pfcVsIso5th2.clear();
   pfnVsIso1.clear();
   pfnVsIso2.clear();
   pfnVsIso3.clear();
   pfnVsIso4.clear();
   pfnVsIso5.clear();
   pfnVsIso1th1.clear();
   pfnVsIso2th1.clear();
   pfnVsIso3th1.clear();
   pfnVsIso4th1.clear();
   pfnVsIso5th1.clear();
   pfnVsIso1th2.clear();
   pfnVsIso2th2.clear();
   pfnVsIso3th2.clear();
   pfnVsIso4th2.clear();
   pfnVsIso5th2.clear();
   pfpVsIso1.clear();
   pfpVsIso2.clear();
   pfpVsIso3.clear();
   pfpVsIso4.clear();
   pfpVsIso5.clear();
   pfpVsIso1th1.clear();
   pfpVsIso2th1.clear();
   pfpVsIso3th1.clear();
   pfpVsIso4th1.clear();
   pfpVsIso5th1.clear();
   pfpVsIso1th2.clear();
   pfpVsIso2th2.clear();
   pfpVsIso3th2.clear();
   pfpVsIso4th2.clear();
   pfpVsIso5th2.clear();
   pfsumVsIso1.clear();
   pfsumVsIso2.clear();
   pfsumVsIso3.clear();
   pfsumVsIso4.clear();
   pfsumVsIso5.clear();
   pfsumVsIso1th1.clear();
   pfsumVsIso2th1.clear();
   pfsumVsIso3th1.clear();
   pfsumVsIso4th1.clear();
   pfsumVsIso5th1.clear();
   pfsumVsIso1th2.clear();
   pfsumVsIso2th2.clear();
   pfsumVsIso3th2.clear();
   pfsumVsIso4th2.clear();
   pfsumVsIso5th2.clear();
   pfVsSubIso1.clear();
   pfVsSubIso2.clear();
   pfVsSubIso3.clear();
   pfVsSubIso4.clear();
   pfVsSubIso5.clear();
   towerIso1.clear();
   towerIso2.clear();
   towerIso3.clear();
   towerIso4.clear();
   towerIso5.clear();
   towerVsIso1.clear();
   towerVsIso2.clear();
   towerVsIso3.clear();
   towerVsIso4.clear();
   towerVsIso5.clear();
   towerVsSubIso1.clear();
   towerVsSubIso2.clear();
   towerVsSubIso3.clear();
   towerVsSubIso4.clear();
   towerVsSubIso5.clear();


   muPt_                 .clear();
   muEta_                .clear();
   muPhi_                .clear();
   muCharge_             .clear();
   muType_               .clear();
   muIsGood_             .clear();
   muD0_                 .clear();
   muDz_                 .clear();
   muChi2NDF_            .clear();
   muInnerD0_            .clear();
   muInnerDz_            .clear();
   muTrkLayers_          .clear();
   muPixelLayers_        .clear();
   muPixelHits_          .clear();
   muMuonHits_           .clear();
   muTrkQuality_         .clear();
   muStations_           .clear();
   muIsoTrk_             .clear();
   muPFChIso_            .clear();
   muPFPhoIso_           .clear();
   muPFNeuIso_           .clear();
   muPFPUIso_            .clear();

   run_    = e.id().run();
   event_  = e.id().event();
   lumis_  = e.luminosityBlock();
   isData_ = e.isRealData();

   // MC truth
   if (doGenParticles_ && !isData_) {
      fillGenPileupInfo(e);
      fillGenParticles(e);
   }

   edm::Handle<vector<reco::Vertex> > vtxHandle;
   e.getByToken(vtxCollection_, vtxHandle);

   // best-known primary vertex coordinates
   math::XYZPoint pv(0, 0, 0);
   for (vector<reco::Vertex>::const_iterator v = vtxHandle->begin(); v != vtxHandle->end(); ++v)
      if (!v->isFake()) {
         pv.SetXYZ(v->x(), v->y(), v->z());
         break;
      }

   fillElectrons(e, es, pv);
   fillPhotons(e, es, pv);
   fillMuons(e, es, pv);

   tree_->Fill();
}

void ggHiNtuplizer::fillGenPileupInfo(const edm::Event& e)
{
   // Fills information about pileup from MC truth.

   edm::Handle<vector<PileupSummaryInfo> > genPileupHandle;
   e.getByToken(genPileupCollection_, genPileupHandle);

   for (vector<PileupSummaryInfo>::const_iterator pu = genPileupHandle->begin(); pu != genPileupHandle->end(); ++pu) {
      nPU_   .push_back(pu->getPU_NumInteractions());
      puTrue_.push_back(pu->getTrueNumInteractions());
      puBX_  .push_back(pu->getBunchCrossing());

      nPUInfo_++;
   }

}

void ggHiNtuplizer::fillGenParticles(const edm::Event& e)
{
   // Fills tree branches with generated particle info.

   edm::Handle<vector<reco::GenParticle> > genParticlesHandle;
   e.getByToken(genParticlesCollection_, genParticlesHandle);

   int genIndex = 0;

   // loop over MC particles
   for (vector<reco::GenParticle>::const_iterator p = genParticlesHandle->begin(); p != genParticlesHandle->end(); ++p) {
      genIndex++;

      // skip all primary particles if not particle gun MC
      if (!runOnParticleGun_ && !p->mother()) continue;

      // stable particles with pT > 5 GeV
      bool isStableFast = (p->status() == 1 && p->pt() > 5.0);

      // stable leptons
      bool isStableLepton = (p->status() == 1 && abs(p->pdgId()) >= 11 && abs(p->pdgId()) <= 16);

      // (unstable) Z, W, H, top, bottom
      bool isHeavy = (p->pdgId() == 23 || abs(p->pdgId()) == 24 || p->pdgId() == 25 ||
                      abs(p->pdgId()) == 6 || abs(p->pdgId()) == 5);

      // reduce size of output root file
      if (!isStableFast && !isStableLepton && !isHeavy)
         continue;

      mcPID_   .push_back(p->pdgId());
      mcStatus_.push_back(p->status());
      mcVtx_x_ .push_back(p->vx());
      mcVtx_y_ .push_back(p->vy());
      mcVtx_z_ .push_back(p->vz());
      mcPt_    .push_back(p->pt());
      mcEta_   .push_back(p->eta());
      mcPhi_   .push_back(p->phi());
      mcE_     .push_back(p->energy());
      mcEt_    .push_back(p->et());
      mcMass_  .push_back(p->mass());

      reco::GenParticleRef partRef = reco::GenParticleRef(
                        genParticlesHandle, p - genParticlesHandle->begin());
      genpartparentage::GenParticleParentage particleHistory(partRef);

      mcParentage_.push_back(particleHistory.hasLeptonParent()*16   +
                             particleHistory.hasBosonParent()*8     +
                             particleHistory.hasNonPromptParent()*4 +
                             particleHistory.hasQCDParent()*2       +
                             particleHistory.hasExoticParent());

      int   momPID  = -999;
      float momPt   = -999;
      float momEta  = -999;
      float momPhi  = -999;
      float momMass = -999;
      int   gmomPID = -999;

      if (particleHistory.hasRealParent()) {
         reco::GenParticleRef momRef = particleHistory.parent();

         // mother
         if (momRef.isNonnull() && momRef.isAvailable()) {
            momPID  = momRef->pdgId();
            momPt   = momRef->pt();
            momEta  = momRef->eta();
            momPhi  = momRef->phi();
            momMass = momRef->mass();

            // granny
            genpartparentage::GenParticleParentage motherParticle(momRef);
            if (motherParticle.hasRealParent()) {
               reco::GenParticleRef granny = motherParticle.parent();
               gmomPID = granny->pdgId();
            }
         }
      }

      mcMomPID_ .push_back(momPID);
      mcMomPt_  .push_back(momPt);
      mcMomEta_ .push_back(momEta);
      mcMomPhi_ .push_back(momPhi);
      mcMomMass_.push_back(momMass);
      mcGMomPID_.push_back(gmomPID);

      mcIndex_  .push_back(genIndex - 1);

      mcCalIsoDR03_.push_back(getGenCalIso(genParticlesHandle, p, 0.3, false, false));
      mcCalIsoDR04_.push_back(getGenCalIso(genParticlesHandle, p, 0.4, false, false));
      mcTrkIsoDR03_.push_back(getGenTrkIso(genParticlesHandle, p, 0.3) );
      mcTrkIsoDR04_.push_back(getGenTrkIso(genParticlesHandle, p, 0.4) );

      nMC_++;

   } // gen-level particles loop

}

float ggHiNtuplizer::getGenCalIso(edm::Handle<vector<reco::GenParticle> > &handle,
                                reco::GenParticleCollection::const_iterator thisPart,
                                float dRMax, bool removeMu, bool removeNu)
{
   // Returns Et sum.

   float etSum = 0;

   for (reco::GenParticleCollection::const_iterator p = handle->begin(); p != handle->end(); ++p) {
      if (p == thisPart) continue;
      if (p->status() != 1) continue;

      // has to come from the same collision
      if (thisPart->collisionId() != p->collisionId())
         continue;

      int pdgCode = abs(p->pdgId());

      // skip muons/neutrinos, if requested
      if (removeMu && pdgCode == 13) continue;
      if (removeNu && (pdgCode == 12 || pdgCode == 14 || pdgCode == 16)) continue;

      // must be within deltaR cone
      float dR = reco::deltaR(thisPart->momentum(), p->momentum());
      if (dR > dRMax) continue;

      etSum += p->et();
   }

   return etSum;
}

float ggHiNtuplizer::getGenTrkIso(edm::Handle<vector<reco::GenParticle> > &handle,
                                reco::GenParticleCollection::const_iterator thisPart, float dRMax)
{
   // Returns pT sum without counting neutral particles.

   float ptSum = 0;

   for (reco::GenParticleCollection::const_iterator p = handle->begin(); p != handle->end(); ++p) {
      if (p == thisPart) continue;
      if (p->status() != 1) continue;
      if (p->charge() == 0) continue;  // do not count neutral particles

      // has to come from the same collision
      if (thisPart->collisionId() != p->collisionId())
         continue;

      // must be within deltaR cone
      float dR = reco::deltaR(thisPart->momentum(), p->momentum());
      if (dR > dRMax) continue;

      ptSum += p->pt();
   }

   return ptSum;
}

void ggHiNtuplizer::fillElectrons(const edm::Event& e, const edm::EventSetup& es, math::XYZPoint& pv)
{
   // Fills tree branches with reco GSF electrons.

   edm::Handle<edm::View<reco::GsfElectron> > gsfElectronsHandle;
   e.getByToken(gsfElectronsCollection_, gsfElectronsHandle);

   // loop over electrons
   for (edm::View<reco::GsfElectron>::const_iterator ele = gsfElectronsHandle->begin(); ele != gsfElectronsHandle->end(); ++ele) {
      eleCharge_           .push_back(ele->charge());
      eleChargeConsistent_ .push_back((int)ele->isGsfCtfScPixChargeConsistent());
      eleEn_               .push_back(ele->energy());
      eleD0_               .push_back(ele->gsfTrack()->dxy(pv));
      eleDz_               .push_back(ele->gsfTrack()->dz(pv));
      eleD0Err_            .push_back(ele->gsfTrack()->dxyError());
      eleDzErr_            .push_back(ele->gsfTrack()->dzError());
      eleTrkPt_            .push_back(ele->gsfTrack()->pt());
      eleTrkEta_           .push_back(ele->gsfTrack()->eta());
      eleTrkPhi_           .push_back(ele->gsfTrack()->phi());
      eleTrkCharge_        .push_back(ele->gsfTrack()->charge());
      eleTrkChi2_          .push_back(ele->gsfTrack()->chi2());
      eleTrkNdof_          .push_back(ele->gsfTrack()->ndof());
      eleTrkNormalizedChi2_.push_back(ele->gsfTrack()->normalizedChi2());
      elePt_               .push_back(ele->pt());
      eleEta_              .push_back(ele->eta());
      elePhi_              .push_back(ele->phi());
      eleSCEn_             .push_back(ele->superCluster()->energy());
      eleESEn_             .push_back(ele->superCluster()->preshowerEnergy());
      eleSCEta_            .push_back(ele->superCluster()->eta());
      eleSCPhi_            .push_back(ele->superCluster()->phi());
      eleSCRawEn_          .push_back(ele->superCluster()->rawEnergy());
      eleSCEtaWidth_       .push_back(ele->superCluster()->etaWidth());
      eleSCPhiWidth_       .push_back(ele->superCluster()->phiWidth());
      eleHoverE_           .push_back(ele->hcalOverEcalBc());
      eleEoverP_           .push_back(ele->eSuperClusterOverP());
      eleEoverPInv_        .push_back(fabs(1./ele->ecalEnergy()-1./ele->trackMomentumAtVtx().R()));
      eleBrem_             .push_back(ele->fbrem());
      eledEtaAtVtx_        .push_back(ele->deltaEtaSuperClusterTrackAtVtx());
      eledPhiAtVtx_        .push_back(ele->deltaPhiSuperClusterTrackAtVtx());
      eleSigmaIEtaIEta_    .push_back(ele->sigmaIetaIeta());
      eleSigmaIPhiIPhi_    .push_back(ele->sigmaIphiIphi());
//    eleConvVeto_         .push_back((int)ele->passConversionVeto()); // TODO: not available in reco::
      eleMissHits_         .push_back(ele->gsfTrack()->numberOfLostHits());
//      eleESEffSigmaRR_     .push_back(lazyTool.eseffsirir(*(ele->superCluster())));

      // full 5x5
      // vector<float> vCovEle = lazyTool_noZS.localCovariances(*(ele->superCluster()->seed()));
      // eleSigmaIEtaIEta_2012_.push_back(isnan(vCovEle[0]) ? 0. : sqrt(vCovEle[0]));
      eleSigmaIEtaIEta_2012_.push_back(ele->full5x5_sigmaIetaIeta() );

      // isolation
      reco::GsfElectron::PflowIsolationVariables pfIso = ele->pfIsolationVariables();
      elePFChIso_          .push_back(pfIso.sumChargedHadronPt);
      elePFPhoIso_         .push_back(pfIso.sumPhotonEt);
      elePFNeuIso_         .push_back(pfIso.sumNeutralHadronEt);
      elePFPUIso_          .push_back(pfIso.sumPUPt);

      // seed
      // eleBC1E_             .push_back(ele->superCluster()->seed()->energy());
      // eleBC1Eta_           .push_back(ele->superCluster()->seed()->eta());

      //parameters of the very first PFCluster
      // reco::CaloCluster_iterator bc = ele->superCluster()->clustersBegin();
      // if (bc != ele->superCluster()->clustersEnd()) {
      //    eleBC2E_  .push_back((*bc)->energy());
      //    eleBC2Eta_.push_back((*bc)->eta());
      // }
      // else {
      //    eleBC2E_  .push_back(-99);
      //    eleBC2Eta_.push_back(-99);
      // }

      nEle_++;

   } // electrons loop
}

void ggHiNtuplizer::fillPhotons(const edm::Event& e, const edm::EventSetup& es, math::XYZPoint& pv)
{
   // Fills tree branches with photons.

  edm::Handle<edm::View<reco::Photon> > recoPhotonsHandle;
  e.getByToken(recoPhotonsCollection_, recoPhotonsHandle);
  edm::Handle<edm::ValueMap<reco::HIPhotonIsolation> > recoPhotonHiIsoHandle;
  edm::ValueMap<reco::HIPhotonIsolation> isoMap;
  if(useValMapIso_){
    e.getByToken(recoPhotonsHiIso_, recoPhotonHiIsoHandle);
    isoMap = * recoPhotonHiIsoHandle;
  }

  //edm::Handle<vector<reco::PFCandidate> > pfHandle;
  //e.getByToken(pfCollection_, pfHandle);
  //edm::Handle<edm::ValueMap<reco::VoronoiBackground> > pfVsMapHandle;
  //e.getByToken(voronoiBkgPF_, pfVsMapHandle);

   // loop over photons
   for (edm::View<reco::Photon>::const_iterator pho = recoPhotonsHandle->begin(); pho != recoPhotonsHandle->end(); ++pho) {
      phoE_             .push_back(pho->energy());
      phoEt_            .push_back(pho->et());
      phoEta_           .push_back(pho->eta());
      phoPhi_           .push_back(pho->phi());
      phoSCE_           .push_back(pho->superCluster()->energy());
      phoSCRawE_        .push_back(pho->superCluster()->rawEnergy());
      phoESEn_          .push_back(pho->superCluster()->preshowerEnergy());
      phoSCEta_         .push_back(pho->superCluster()->eta());
      phoSCPhi_         .push_back(pho->superCluster()->phi());
      phoSCEtaWidth_    .push_back(pho->superCluster()->etaWidth());
      phoSCPhiWidth_    .push_back(pho->superCluster()->phiWidth());
      phoSCBrem_        .push_back(pho->superCluster()->phiWidth()/pho->superCluster()->etaWidth());
      phohasPixelSeed_  .push_back((int)pho->hasPixelSeed());
//    phoEleVeto_       .push_back((int)pho->passElectronVeto());   // TODO: not available in reco::
      phoR9_            .push_back(pho->r9());
      phoHoverE_        .push_back(pho->hadTowOverEm());

      phoSigmaIEtaIEta_ .push_back(pho->sigmaIetaIeta());
      //phoSigmaIEtaIPhi_ .push_back(pho->sep());   // TODO: not available in reco::
      //phoSigmaIPhiIPhi_ .push_back(pho->spp());   // TODO: not available in reco::

      // phoE1x3_          .push_back(lazyTool.e1x3(      *(pho->superCluster()->seed())));
      // phoE2x2_          .push_back(lazyTool.e2x2(      *(pho->superCluster()->seed())));
      // phoE2x5Max_       .push_back(lazyTool.e2x5Max(   *(pho->superCluster()->seed())));
      // phoE5x5_          .push_back(lazyTool.e5x5(      *(pho->superCluster()->seed())));
      // phoESEffSigmaRR_  .push_back(lazyTool.eseffsirir(*(pho->superCluster())));

      // full 5x5
      // vector<float> vCov = lazyTool_noZS.localCovariances(*(pho->superCluster()->seed()));
      // phoSigmaIEtaIEta_2012_ .push_back(isnan(vCov[0]) ? 0. : sqrt(vCov[0]));
      // phoSigmaIEtaIPhi_2012_ .push_back(vCov[1]);
      // phoSigmaIPhiIPhi_2012_ .push_back(isnan(vCov[2]) ? 0. : sqrt(vCov[2]));
      phoSigmaIEtaIEta_2012_.push_back(pho->full5x5_sigmaIetaIeta() );

      // phoE1x3_2012_          .push_back(lazyTool_noZS.e1x3(   *(pho->superCluster()->seed())));
      // phoE2x2_2012_          .push_back(lazyTool_noZS.e2x2(   *(pho->superCluster()->seed())));
      // phoE2x5Max_2012_       .push_back(lazyTool_noZS.e2x5Max(*(pho->superCluster()->seed())));
      // phoE5x5_2012_          .push_back(lazyTool_noZS.e5x5(   *(pho->superCluster()->seed())));

      // seed
      // phoBC1E_     .push_back(pho->superCluster()->seed()->energy());
      // phoBC1Eta_   .push_back(pho->superCluster()->seed()->eta());

      // parameters of the very first PFCluster
      // reco::CaloCluster_iterator bc = pho->superCluster()->clustersBegin();
      // if (bc != pho->superCluster()->clustersEnd()) {
      //    phoBC2E_  .push_back((*bc)->energy());
      //    phoBC2Eta_.push_back((*bc)->eta());
      // }
      // else {
      //    phoBC2E_  .push_back(-99);
      //    phoBC2Eta_.push_back(-99);
      // }

      if(useValMapIso_)
      {
	unsigned int idx = pho - recoPhotonsHandle->begin();
	edm::RefToBase<reco::Photon> photonRef = recoPhotonsHandle->refAt(idx);

	pho_ecalClusterIsoR2_.push_back(isoMap[photonRef].ecalClusterIsoR2());
	pho_ecalClusterIsoR3_.push_back(isoMap[photonRef].ecalClusterIsoR3());
	pho_ecalClusterIsoR4_.push_back(isoMap[photonRef].ecalClusterIsoR4());
	pho_ecalClusterIsoR5_.push_back(isoMap[photonRef].ecalClusterIsoR5());
	pho_hcalRechitIsoR1_.push_back(isoMap[photonRef].hcalRechitIsoR1());
	pho_hcalRechitIsoR2_.push_back(isoMap[photonRef].hcalRechitIsoR2());
	pho_hcalRechitIsoR3_.push_back(isoMap[photonRef].hcalRechitIsoR3());
	pho_hcalRechitIsoR4_.push_back(isoMap[photonRef].hcalRechitIsoR4());
	pho_hcalRechitIsoR5_.push_back(isoMap[photonRef].hcalRechitIsoR5());
	pho_trackIsoR1PtCut20_.push_back(isoMap[photonRef].trackIsoR1PtCut20());
	pho_trackIsoR2PtCut20_.push_back(isoMap[photonRef].trackIsoR2PtCut20());
	pho_trackIsoR3PtCut20_.push_back(isoMap[photonRef].trackIsoR3PtCut20());
	pho_trackIsoR4PtCut20_.push_back(isoMap[photonRef].trackIsoR4PtCut20());
	pho_trackIsoR5PtCut20_.push_back(isoMap[photonRef].trackIsoR5PtCut20());
	pho_swissCrx_.push_back(isoMap[photonRef].swissCrx());
	pho_seedTime_.push_back(isoMap[photonRef].seedTime());
      }

      pfIsoCalculator pfIso(e,es, pfCollection_, voronoiBkgPF_, pv);


      nPho_++;

   } // photons loop
}

void ggHiNtuplizer::fillMuons(const edm::Event& e, const edm::EventSetup& es, math::XYZPoint& pv)
{
   // Fills tree branches with reco muons.

   edm::Handle<edm::View<reco::Muon> > recoMuonsHandle;
   e.getByToken(recoMuonsCollection_, recoMuonsHandle);

   for (edm::View<reco::Muon>::const_iterator mu = recoMuonsHandle->begin(); mu != recoMuonsHandle->end(); ++mu) {
      if (mu->pt() < 5) continue;
      if (!(mu->isPFMuon() || mu->isGlobalMuon() || mu->isTrackerMuon())) continue;

      muPt_    .push_back(mu->pt());
      muEta_   .push_back(mu->eta());
      muPhi_   .push_back(mu->phi());
      muCharge_.push_back(mu->charge());
      muType_  .push_back(mu->type());
      muIsGood_.push_back((int) muon::isGoodMuon(*mu, muon::selectionTypeFromString("TMOneStationTight")));
      muD0_    .push_back(mu->muonBestTrack()->dxy(pv));
      muDz_    .push_back(mu->muonBestTrack()->dz(pv));

      const reco::TrackRef glbMu = mu->globalTrack();
      const reco::TrackRef innMu = mu->innerTrack();

      if (glbMu.isNull()) {
         muChi2NDF_ .push_back(-99);
         muMuonHits_.push_back(-99);
      } else {
         muChi2NDF_.push_back(glbMu->normalizedChi2());
         muMuonHits_.push_back(glbMu->hitPattern().numberOfValidMuonHits());
      }

      if (innMu.isNull()) {
         muInnerD0_     .push_back(-99);
         muInnerDz_     .push_back(-99);
         muTrkLayers_   .push_back(-99);
         muPixelLayers_ .push_back(-99);
         muPixelHits_   .push_back(-99);
         muTrkQuality_  .push_back(-99);
      } else {
         muInnerD0_     .push_back(innMu->dxy(pv));
         muInnerDz_     .push_back(innMu->dz(pv));
         muTrkLayers_   .push_back(innMu->hitPattern().trackerLayersWithMeasurement());
         muPixelLayers_ .push_back(innMu->hitPattern().pixelLayersWithMeasurement());
         muPixelHits_   .push_back(innMu->hitPattern().numberOfValidPixelHits());
         muTrkQuality_  .push_back(innMu->quality(reco::TrackBase::highPurity));
      }

      muStations_ .push_back(mu->numberOfMatchedStations());
      muIsoTrk_   .push_back(mu->isolationR03().sumPt);
      muPFChIso_  .push_back(mu->pfIsolationR04().sumChargedHadronPt);
      muPFPhoIso_ .push_back(mu->pfIsolationR04().sumPhotonEt);
      muPFNeuIso_ .push_back(mu->pfIsolationR04().sumNeutralHadronEt);
      muPFPUIso_  .push_back(mu->pfIsolationR04().sumPUPt);

      nMu_++;
   } // muons loop
}

DEFINE_FWK_MODULE(ggHiNtuplizer);
