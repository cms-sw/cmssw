#include "HeavyIonsAnalysis/EGMAnalysis/plugins/ggHiNtuplizer.h"
#include "HeavyIonsAnalysis/EGMAnalysis/plugins/GenParticleParentage.h"
#include "HeavyIonsAnalysis/EGMAnalysis/interface/pfIsoCalculator.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
//#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "CommonTools/Egamma/interface/ConversionTools.h"
#include "CommonTools/Egamma/interface/EffectiveAreas.h"

using namespace std::placeholders;

ggHiNtuplizer::ggHiNtuplizer(const edm::ParameterSet& ps)
    : effectiveAreas_((ps.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath()) {
  doGenParticles_ = ps.getParameter<bool>("doGenParticles");
  doElectrons_ = ps.getParameter<bool>("doElectrons");
  doPhotons_ = ps.getParameter<bool>("doPhotons");
  doMuons_ = ps.getParameter<bool>("doMuons");

  doEffectiveAreas_ = ps.getParameter<bool>("doEffectiveAreas");

  isParticleGun_ = ps.getParameter<bool>("isParticleGun");
  useValMapIso_ = ps.getParameter<bool>("useValMapIso");
  doPhoEReg_ = ps.getParameter<bool>("doPhoERegression");
  doRecHitsEB_ = ps.getParameter<bool>("doRecHitsEB");
  doRecHitsEE_ = ps.getParameter<bool>("doRecHitsEE");

  if (doRecHitsEB_) {
    recHitsEB_ = consumes<EcalRecHitCollection>(
        ps.getUntrackedParameter<edm::InputTag>("recHitsEB", edm::InputTag("ecalRecHit", "EcalRecHitsEB")));
  }
  if (doRecHitsEE_) {
    recHitsEE_ = consumes<EcalRecHitCollection>(
        ps.getUntrackedParameter<edm::InputTag>("recHitsEE", edm::InputTag("ecalRecHit", "EcalRecHitsEE")));
  }

  doSuperClusters_ = ps.getParameter<bool>("doSuperClusters");
  if (doSuperClusters_) {
    scToken_ = consumes<std::vector<reco::SuperCluster>>(ps.getParameter<edm::InputTag>("superClusters"));
  }

  doPfIso_ = ps.getParameter<bool>("doPfIso");
  if (doPfIso_) {
    isPackedPFCandidate_ = ps.getParameter<bool>("isPackedPFCandidate");

    std::string strPFCandSrc = "particleFlowCollection";
    if (isPackedPFCandidate_) {
      packedPFCandToken_ =
          consumes<edm::View<pat::PackedCandidate>>(ps.getParameter<edm::InputTag>(strPFCandSrc.c_str()));
    } else {
      recoPFCandToken_ = consumes<edm::View<reco::PFCandidate>>(ps.getParameter<edm::InputTag>(strPFCandSrc.c_str()));
    }
  }

  vertexToken_ = consumes<std::vector<reco::Vertex>>(ps.getParameter<edm::InputTag>("vertexSrc"));
  rhoToken_ = consumes<double>(ps.getParameter<edm::InputTag>("rhoSrc"));

  if (doGenParticles_) {
    pileupToken_ = consumes<std::vector<PileupSummaryInfo>>(ps.getParameter<edm::InputTag>("pileupSrc"));

    doPackedGenParticle_ = ps.getParameter<bool>("doPackedGenParticle");
    std::string strGenParticleSrc = "genParticleSrc";
    if (doPackedGenParticle_) {
      packedGenParticlesToken_ =
          consumes<std::vector<pat::PackedGenParticle>>(ps.getParameter<edm::InputTag>(strGenParticleSrc.c_str()));
      signalPackedGenParticlesToken_ =
          consumes<edm::View<pat::PackedGenParticle>>(ps.getParameter<edm::InputTag>("signalGenParticleSrc"));
    } else {
      genParticlesToken_ =
          consumes<std::vector<reco::GenParticle>>(ps.getParameter<edm::InputTag>(strGenParticleSrc.c_str()));
    }
  }

  if (doElectrons_) {
    electronsToken_ = consumes<edm::View<pat::Electron>>(ps.getParameter<edm::InputTag>("electronSrc"));
    beamSpotToken_ = consumes<reco::BeamSpot>(ps.getParameter<edm::InputTag>("beamSpotSrc"));
    conversionsToken_ = consumes<reco::ConversionCollection>(ps.getParameter<edm::InputTag>("conversionsSrc"));
  }

  if (doPhotons_) {
    photonsToken_ = consumes<edm::View<pat::Photon>>(ps.getParameter<edm::InputTag>("photonSrc"));
  }

  if (doMuons_) {
    muonsToken_ = consumes<edm::View<pat::Muon>>(ps.getParameter<edm::InputTag>("muonSrc"));
    muonPtMin_ = ps.getParameter<double>("muonPtMin");
  }

  ttbESToken_ = esConsumes(edm::ESInputTag("", "TransientTrackBuilder"));
  geometryToken_ = esConsumes();

  // initialize output TTree
  edm::Service<TFileService> fs;
  tree_ = fs->make<TTree>("EventTree", "electrons, photons, muons");

  tree_->Branch("run", &run_);
  tree_->Branch("event", &event_);
  tree_->Branch("lumis", &lumis_);

  tree_->Branch("rho", &rho_);

  if (doGenParticles_) {
    tree_->Branch("nPUInfo", &nPUInfo_);
    tree_->Branch("nPU", &nPU_);
    tree_->Branch("puBX", &puBX_);
    tree_->Branch("puTrue", &puTrue_);

    tree_->Branch("nMC", &nMC_);

    tree_->Branch("mcVtx_x", &mcVtx_x_);
    tree_->Branch("mcVtx_y", &mcVtx_y_);
    tree_->Branch("mcVtx_z", &mcVtx_z_);

    tree_->Branch("mcPID", &mcPID_);
    tree_->Branch("mcStatus", &mcStatus_);
    tree_->Branch("mcPt", &mcPt_);
    tree_->Branch("mcEta", &mcEta_);
    tree_->Branch("mcPhi", &mcPhi_);
    tree_->Branch("mcE", &mcE_);
    tree_->Branch("mcEt", &mcEt_);
    tree_->Branch("mcMass", &mcMass_);

    tree_->Branch("mcParentage", &mcParentage_);
    tree_->Branch("mcMomPID", &mcMomPID_);
    tree_->Branch("mcMomKey", &mcMomKey_);
    tree_->Branch("mcMomPt", &mcMomPt_);
    tree_->Branch("mcMomEta", &mcMomEta_);
    tree_->Branch("mcMomPhi", &mcMomPhi_);
    tree_->Branch("mcMomMass", &mcMomMass_);
    tree_->Branch("mcGMomPID", &mcGMomPID_);
    tree_->Branch("mcGMomKey", &mcGMomKey_);
    tree_->Branch("mcGGMomPID", &mcGGMomPID_);
    tree_->Branch("mcGGMomKey", &mcGGMomKey_);
    tree_->Branch("mcIndex", &mcIndex_);
    tree_->Branch("mcSube", &mcSube_);

    tree_->Branch("mcCalIsoDR03", &mcCalIsoDR03_);
    tree_->Branch("mcCalIsoDR04", &mcCalIsoDR04_);
    tree_->Branch("mcTrkIsoDR03", &mcTrkIsoDR03_);
    tree_->Branch("mcTrkIsoDR04", &mcTrkIsoDR04_);
  }

  if (doSuperClusters_) {
    tree_->Branch("nSC", &nSC_);
    tree_->Branch("scE", &scE_);
    tree_->Branch("scRawE", &scRawE_);
    tree_->Branch("scEta", &scEta_);
    tree_->Branch("scPhi", &scPhi_);
  }

  if (doElectrons_) {
    tree_->Branch("nEle", &nEle_);

    tree_->Branch("eleD0", &eleD0_);
    tree_->Branch("eleDz", &eleDz_);
    tree_->Branch("eleD0Err", &eleD0Err_);
    tree_->Branch("eleDzErr", &eleDzErr_);
    tree_->Branch("eleTrkPt", &eleTrkPt_);
    tree_->Branch("eleTrkEta", &eleTrkEta_);
    tree_->Branch("eleTrkPhi", &eleTrkPhi_);
    tree_->Branch("eleTrkCharge", &eleTrkCharge_);
    tree_->Branch("eleTrkPtErr", &eleTrkPtErr_);
    tree_->Branch("eleTrkChi2", &eleTrkChi2_);
    tree_->Branch("eleTrkNdof", &eleTrkNdof_);
    tree_->Branch("eleTrkNormalizedChi2", &eleTrkNormalizedChi2_);
    tree_->Branch("eleTrkValidHits", &eleTrkValidHits_);
    tree_->Branch("eleTrkLayers", &eleTrkLayers_);
    tree_->Branch("eleMissHits", &eleMissHits_);
    tree_->Branch("eleIP3D", &eleIP3D_);
    tree_->Branch("eleIP3DErr", &eleIP3DErr_);

    tree_->Branch("elePt", &elePt_);
    tree_->Branch("eleRawPt", &eleRawPt_);
    tree_->Branch("eleEta", &eleEta_);
    tree_->Branch("elePhi", &elePhi_);
    tree_->Branch("eleCharge", &eleCharge_);
    tree_->Branch("eleEn", &eleEn_);

    tree_->Branch("eleMVAIso", &eleMVAIso_);
    for (auto& w : eleMVAIsoWP_)
      tree_->Branch(("eleMVAIso"+w.first).c_str(), &(w.second));
    tree_->Branch("eleMVAId", &eleMVAId_);
    for (auto& w : eleMVAIdWP_)
      tree_->Branch(("eleMVAId"+w.first).c_str(), &(w.second));
    for (auto& w : eleCutIdWP_)
      tree_->Branch(("eleCutId"+w.first).c_str(), &(w.second));

    tree_->Branch("eleSCEn", &eleSCEn_);
    tree_->Branch("eleESEn", &eleESEn_);
    tree_->Branch("eleSCEta", &eleSCEta_);
    tree_->Branch("eleSCPhi", &eleSCPhi_);
    tree_->Branch("eleSCRawEn", &eleSCRawEn_);
    tree_->Branch("eleSCEtaWidth", &eleSCEtaWidth_);
    tree_->Branch("eleSCPhiWidth", &eleSCPhiWidth_);
    tree_->Branch("eleSCClustersSize", &eleSCClustersSize_);
    tree_->Branch("eleSeedEn", &eleSeedEn_);
    tree_->Branch("eleSeedEta", &eleSeedEta_);
    tree_->Branch("eleSeedPhi", &eleSeedPhi_);

    tree_->Branch("eleHoverE", &eleHoverE_);
    tree_->Branch("eleHoverEBc", &eleHoverEBc_);
    tree_->Branch("eleRawHoverE", &eleRawHoverE_);
    tree_->Branch("eleRawHoverEBc", &eleRawHoverEBc_);
    tree_->Branch("eleEoverP", &eleEoverP_);
    tree_->Branch("eleEoverPInv", &eleEoverPInv_);
    tree_->Branch("eleEcalE", &eleEcalE_);
    tree_->Branch("eleRawEcalE", &eleRawEcalE_);
    tree_->Branch("elePAtVtx", &elePAtVtx_);
    tree_->Branch("elePAtSC", &elePAtSC_);
    tree_->Branch("elePAtCluster", &elePAtCluster_);
    tree_->Branch("elePAtSeed", &elePAtSeed_);
    tree_->Branch("eledEtaAtVtx", &eledEtaAtVtx_);
    tree_->Branch("eledPhiAtVtx", &eledPhiAtVtx_);
    tree_->Branch("eledEtaSeedAtVtx", &eledEtaSeedAtVtx_);
    tree_->Branch("eleSigmaIEtaIEta", &eleSigmaIEtaIEta_);
    tree_->Branch("eleSigmaIPhiIPhi", &eleSigmaIPhiIPhi_);
    tree_->Branch("eleBrem", &eleBrem_);

    tree_->Branch("eleConvVeto", &eleConvVeto_);

    tree_->Branch("eleR9", &eleR9_);
    tree_->Branch("eleE3x3", &eleE3x3_);
    tree_->Branch("eleE5x5", &eleE5x5_);
    tree_->Branch("eleR9Full5x5", &eleR9Full5x5_);
    tree_->Branch("eleE3x3Full5x5", &eleE3x3Full5x5_);
    tree_->Branch("eleE5x5Full5x5", &eleE5x5Full5x5_);
    tree_->Branch("eleSigmaIEtaIEta_2012", &eleSigmaIEtaIEta_2012_);

    tree_->Branch("elePFChIso", &elePFChIso_);
    tree_->Branch("elePFPhoIso", &elePFPhoIso_);
    tree_->Branch("elePFNeuIso", &elePFNeuIso_);
    tree_->Branch("elePFPUIso", &elePFPUIso_);

    if (doEffectiveAreas_) {
      tree_->Branch("elePFRelIsoWithEA", &elePFRelIsoWithEA_);
      tree_->Branch("elePFRelIsoWithDBeta", &elePFRelIsoWithDBeta_);
      tree_->Branch("eleEffAreaTimesRho", &eleEffAreaTimesRho_);
    }

    if (doPfIso_) {
      tree_->Branch("elePFChIso03", &elePFChIso03_);
      tree_->Branch("elePFPhoIso03", &elePFPhoIso03_);
      tree_->Branch("elePFNeuIso03", &elePFNeuIso03_);
      tree_->Branch("elePFChIso04", &elePFChIso04_);
      tree_->Branch("elePFPhoIso04", &elePFPhoIso04_);
      tree_->Branch("elePFNeuIso04", &elePFNeuIso04_);
    }

    tree_->Branch("eleSeedCryEta", &eleSeedCryEta_);
    tree_->Branch("eleSeedCryPhi", &eleSeedCryPhi_);
    tree_->Branch("eleSeedCryIeta", &eleSeedCryIeta_);
    tree_->Branch("eleSeedCryIphi", &eleSeedCryIphi_);

    if (doGenParticles_)
      tree_->Branch("ele_genMatchedIndex", &ele_genMatchedIndex_);
  }

  if (doPhotons_) {
    tree_->Branch("nPho", &nPho_);

    tree_->Branch("phoE", &phoE_);
    tree_->Branch("phoEt", &phoEt_);
    tree_->Branch("phoEta", &phoEta_);
    tree_->Branch("phoPhi", &phoPhi_);

    tree_->Branch("phoEcorrStdEcal", &phoEcorrStdEcal_);
    tree_->Branch("phoEcorrPhoEcal", &phoEcorrPhoEcal_);
    tree_->Branch("phoEcorrRegr1", &phoEcorrRegr1_);
    tree_->Branch("phoEcorrRegr2", &phoEcorrRegr2_);
    tree_->Branch("phoEcorrErrStdEcal", &phoEcorrErrStdEcal_);
    tree_->Branch("phoEcorrErrPhoEcal", &phoEcorrErrPhoEcal_);
    tree_->Branch("phoEcorrErrRegr1", &phoEcorrErrRegr1_);
    tree_->Branch("phoEcorrErrRegr2", &phoEcorrErrRegr2_);

    tree_->Branch("phoSCE", &phoSCE_);
    tree_->Branch("phoSCRawE", &phoSCRawE_);
    tree_->Branch("phoSCEta", &phoSCEta_);
    tree_->Branch("phoSCPhi", &phoSCPhi_);
    tree_->Branch("phoSCEtaWidth", &phoSCEtaWidth_);
    tree_->Branch("phoSCPhiWidth", &phoSCPhiWidth_);
    tree_->Branch("phoSCBrem", &phoSCBrem_);
    tree_->Branch("phoSCnHits", &phoSCnHits_);
    tree_->Branch("phoSCflags", &phoSCflags_);
    tree_->Branch("phoSCinClean", &phoSCinClean_);
    tree_->Branch("phoSCinUnClean", &phoSCinUnClean_);
    tree_->Branch("phoSCnBC", &phoSCnBC_);
    tree_->Branch("phoESEn", &phoESEn_);

    tree_->Branch("phoIsPFPhoton", &phoIsPFPhoton_);
    tree_->Branch("phoIsStandardPhoton", &phoIsStandardPhoton_);
    tree_->Branch("phoHasPixelSeed", &phoHasPixelSeed_);
    tree_->Branch("phoHasConversionTracks", &phoHasConversionTracks_);
    tree_->Branch("phoHadTowerOverEm", &phoHadTowerOverEm_);
    tree_->Branch("phoHoverE", &phoHoverE_);
    tree_->Branch("phoHoverEValid", &phoHoverEValid_);
    tree_->Branch("phoSigmaIEtaIEta", &phoSigmaIEtaIEta_);
    tree_->Branch("phoR9", &phoR9_);
    tree_->Branch("phoE1x5", &phoE1x5_);
    tree_->Branch("phoE2x5", &phoE2x5_);
    tree_->Branch("phoE3x3", &phoE3x3_);
    tree_->Branch("phoE5x5", &phoE5x5_);
    tree_->Branch("phoMaxEnergyXtal", &phoMaxEnergyXtal_);
    tree_->Branch("phoSigmaEtaEta", &phoSigmaEtaEta_);
    tree_->Branch("phoSigmaIEtaIEta_2012", &phoSigmaIEtaIEta_2012_);
    tree_->Branch("phoR9_2012", &phoR9_2012_);
    tree_->Branch("phoE1x5_2012", &phoE1x5_2012_);
    tree_->Branch("phoE2x5_2012", &phoE2x5_2012_);
    tree_->Branch("phoE3x3_2012", &phoE3x3_2012_);
    tree_->Branch("phoE5x5_2012", &phoE5x5_2012_);
    tree_->Branch("phoMaxEnergyXtal_2012", &phoMaxEnergyXtal_2012_);
    tree_->Branch("phoSigmaEtaEta_2012", &phoSigmaEtaEta_2012_);

    if (doPhoEReg_) {
      tree_->Branch("phoHadTowerOverEm1", &phoHadTowerOverEm1_);
      tree_->Branch("phoHadTowerOverEm2", &phoHadTowerOverEm2_);
      tree_->Branch("phoHoverE1", &phoHoverE1_);
      tree_->Branch("phoHoverE2", &phoHoverE2_);

      tree_->Branch("phoSigmaIEtaIPhi", &phoSigmaIEtaIPhi_);
      tree_->Branch("phoSigmaIPhiIPhi", &phoSigmaIPhiIPhi_);
      tree_->Branch("phoR1x5", &phoR1x5_);
      tree_->Branch("phoR2x5", &phoR2x5_);
      tree_->Branch("phoE2nd", &phoE2nd_);
      tree_->Branch("phoETop", &phoETop_);
      tree_->Branch("phoEBottom", &phoEBottom_);
      tree_->Branch("phoELeft", &phoELeft_);
      tree_->Branch("phoERight", &phoERight_);
      tree_->Branch("phoE1x3", &phoE1x3_);
      tree_->Branch("phoE2x2", &phoE2x2_);
      tree_->Branch("phoE2x5Max", &phoE2x5Max_);
      tree_->Branch("phoE2x5Top", &phoE2x5Top_);
      tree_->Branch("phoE2x5Bottom", &phoE2x5Bottom_);
      tree_->Branch("phoE2x5Left", &phoE2x5Left_);
      tree_->Branch("phoE2x5Right", &phoE2x5Right_);
      //tree_->Branch("phoSMMajor",         &phoSMMajor_);
      //tree_->Branch("phoSMMinor",         &phoSMMinor_);
      //tree_->Branch("phoSMAlpha",         &phoSMAlpha_);

      tree_->Branch("phoSigmaIEtaIPhi_2012", &phoSigmaIEtaIPhi_2012_);
      tree_->Branch("phoSigmaIPhiIPhi_2012", &phoSigmaIPhiIPhi_2012_);
      tree_->Branch("phoR1x5_2012", &phoR1x5_2012_);
      tree_->Branch("phoR2x5_2012", &phoR2x5_2012_);
      tree_->Branch("phoE2nd_2012", &phoE2nd_2012_);
      tree_->Branch("phoETop_2012", &phoETop_2012_);
      tree_->Branch("phoEBottom_2012", &phoEBottom_2012_);
      tree_->Branch("phoELeft_2012", &phoELeft_2012_);
      tree_->Branch("phoERight_2012", &phoERight_2012_);
      tree_->Branch("phoE1x3_2012", &phoE1x3_2012_);
      tree_->Branch("phoE2x2_2012", &phoE2x2_2012_);
      tree_->Branch("phoE2x5Max_2012", &phoE2x5Max_2012_);
      tree_->Branch("phoE2x5Top_2012", &phoE2x5Top_2012_);
      tree_->Branch("phoE2x5Bottom_2012", &phoE2x5Bottom_2012_);
      tree_->Branch("phoE2x5Left_2012", &phoE2x5Left_2012_);
      tree_->Branch("phoE2x5Right_2012", &phoE2x5Right_2012_);
      //tree_->Branch("phoSMMajor_2012",       &phoSMMajor_2012_);
      //tree_->Branch("phoSMMinor_2012",       &phoSMMinor_2012_);
      //tree_->Branch("phoSMAlpha_2012",       &phoSMAlpha_2012_);
    }

    tree_->Branch("phoBC1E", &phoBC1E_);
    tree_->Branch("phoBC1Ecorr", &phoBC1Ecorr_);
    tree_->Branch("phoBC1Eta", &phoBC1Eta_);
    tree_->Branch("phoBC1Phi", &phoBC1Phi_);
    tree_->Branch("phoBC1size", &phoBC1size_);
    tree_->Branch("phoBC1flags", &phoBC1flags_);
    tree_->Branch("phoBC1inClean", &phoBC1inClean_);
    tree_->Branch("phoBC1inUnClean", &phoBC1inUnClean_);
    tree_->Branch("phoBC1rawID", &phoBC1rawID_);

    if (useValMapIso_) {
      tree_->Branch("pho_ecalClusterIsoR1", &pho_ecalClusterIsoR1_);
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
    }

    if (doGenParticles_) {
      tree_->Branch("pho_genMatchedIndex", &pho_genMatchedIndex_);
    }

    if (doRecHitsEB_ || doRecHitsEE_) {
      // rechit info
      tree_->Branch("nRH", &nRH_);
      tree_->Branch("rhRawId", &rhRawId_);
      tree_->Branch("rhieta", &rhieta_);
      tree_->Branch("rhiphi", &rhiphi_);
      tree_->Branch("rhix", &rhix_);
      tree_->Branch("rhiy", &rhiy_);
      tree_->Branch("rhE", &rhE_);
      tree_->Branch("rhEt", &rhEt_);
      tree_->Branch("rhEta", &rhEta_);
      tree_->Branch("rhPhi", &rhPhi_);
      tree_->Branch("rhChi2", &rhChi2_);
      tree_->Branch("rhEerror", &rhEerror_);
      tree_->Branch("rhFlags", &rhFlags_);
      tree_->Branch("rhPhoIdx", &rhPhoIdx_);
      tree_->Branch("rhBCIdx", &rhBCIdx_);
    }

    if (doPfIso_) {
      tree_->Branch("pfcIso1", &pfcIso1_);
      tree_->Branch("pfcIso2", &pfcIso2_);
      tree_->Branch("pfcIso3", &pfcIso3_);
      tree_->Branch("pfcIso4", &pfcIso4_);
      tree_->Branch("pfcIso5", &pfcIso5_);

      tree_->Branch("pfpIso1", &pfpIso1_);
      tree_->Branch("pfpIso2", &pfpIso2_);
      tree_->Branch("pfpIso3", &pfpIso3_);
      tree_->Branch("pfpIso4", &pfpIso4_);
      tree_->Branch("pfpIso5", &pfpIso5_);

      tree_->Branch("pfnIso1", &pfnIso1_);
      tree_->Branch("pfnIso2", &pfnIso2_);
      tree_->Branch("pfnIso3", &pfnIso3_);
      tree_->Branch("pfnIso4", &pfnIso4_);
      tree_->Branch("pfnIso5", &pfnIso5_);

      tree_->Branch("pfpIso1subSC", &pfpIso1subSC_);
      tree_->Branch("pfpIso2subSC", &pfpIso2subSC_);
      tree_->Branch("pfpIso3subSC", &pfpIso3subSC_);
      tree_->Branch("pfpIso4subSC", &pfpIso4subSC_);
      tree_->Branch("pfpIso5subSC", &pfpIso5subSC_);

      tree_->Branch("pfcIso1subUE", &pfcIso1subUE_);
      tree_->Branch("pfcIso2subUE", &pfcIso2subUE_);
      tree_->Branch("pfcIso3subUE", &pfcIso3subUE_);
      tree_->Branch("pfcIso4subUE", &pfcIso4subUE_);
      tree_->Branch("pfcIso5subUE", &pfcIso5subUE_);

      tree_->Branch("pfpIso1subUE", &pfpIso1subUE_);
      tree_->Branch("pfpIso2subUE", &pfpIso2subUE_);
      tree_->Branch("pfpIso3subUE", &pfpIso3subUE_);
      tree_->Branch("pfpIso4subUE", &pfpIso4subUE_);
      tree_->Branch("pfpIso5subUE", &pfpIso5subUE_);

      tree_->Branch("pfnIso1subUE", &pfnIso1subUE_);
      tree_->Branch("pfnIso2subUE", &pfnIso2subUE_);
      tree_->Branch("pfnIso3subUE", &pfnIso3subUE_);
      tree_->Branch("pfnIso4subUE", &pfnIso4subUE_);
      tree_->Branch("pfnIso5subUE", &pfnIso5subUE_);

      tree_->Branch("pfpIso1subSCsubUE", &pfpIso1subSCsubUE_);
      tree_->Branch("pfpIso2subSCsubUE", &pfpIso2subSCsubUE_);
      tree_->Branch("pfpIso3subSCsubUE", &pfpIso3subSCsubUE_);
      tree_->Branch("pfpIso4subSCsubUE", &pfpIso4subSCsubUE_);
      tree_->Branch("pfpIso5subSCsubUE", &pfpIso5subSCsubUE_);

      tree_->Branch("pfcIso1pTgt1p0subUE", &pfcIso1pTgt1p0subUE_);
      tree_->Branch("pfcIso2pTgt1p0subUE", &pfcIso2pTgt1p0subUE_);
      tree_->Branch("pfcIso3pTgt1p0subUE", &pfcIso3pTgt1p0subUE_);
      tree_->Branch("pfcIso4pTgt1p0subUE", &pfcIso4pTgt1p0subUE_);
      tree_->Branch("pfcIso5pTgt1p0subUE", &pfcIso5pTgt1p0subUE_);

      tree_->Branch("pfcIso1pTgt2p0subUE", &pfcIso1pTgt2p0subUE_);
      tree_->Branch("pfcIso2pTgt2p0subUE", &pfcIso2pTgt2p0subUE_);
      tree_->Branch("pfcIso3pTgt2p0subUE", &pfcIso3pTgt2p0subUE_);
      tree_->Branch("pfcIso4pTgt2p0subUE", &pfcIso4pTgt2p0subUE_);
      tree_->Branch("pfcIso5pTgt2p0subUE", &pfcIso5pTgt2p0subUE_);

      tree_->Branch("pfcIso1pTgt3p0subUE", &pfcIso1pTgt3p0subUE_);
      tree_->Branch("pfcIso2pTgt3p0subUE", &pfcIso2pTgt3p0subUE_);
      tree_->Branch("pfcIso3pTgt3p0subUE", &pfcIso3pTgt3p0subUE_);
      tree_->Branch("pfcIso4pTgt3p0subUE", &pfcIso4pTgt3p0subUE_);
      tree_->Branch("pfcIso5pTgt3p0subUE", &pfcIso5pTgt3p0subUE_);

      // photon pf isolation UE-subtracted and cone excluded
      tree_->Branch("pfcIso2subUEec", &pfcIso2subUEec_);
      tree_->Branch("pfcIso3subUEec", &pfcIso3subUEec_);
      tree_->Branch("pfcIso4subUEec", &pfcIso4subUEec_);

      tree_->Branch("pfpIso2subUEec", &pfpIso2subUEec_);
      tree_->Branch("pfpIso3subUEec", &pfpIso3subUEec_);
      tree_->Branch("pfpIso4subUEec", &pfpIso4subUEec_);

      tree_->Branch("pfnIso2subUEec", &pfnIso2subUEec_);
      tree_->Branch("pfnIso3subUEec", &pfnIso3subUEec_);
      tree_->Branch("pfnIso4subUEec", &pfnIso4subUEec_);

      tree_->Branch("pfcIso2pTgt2p0subUEec", &pfcIso2pTgt2p0subUEec_);
      tree_->Branch("pfcIso3pTgt2p0subUEec", &pfcIso3pTgt2p0subUEec_);
      tree_->Branch("pfcIso4pTgt2p0subUEec", &pfcIso4pTgt2p0subUEec_);
    }
  }

  if (doMuons_) {
    tree_->Branch("nMu", &nMu_);

    tree_->Branch("muPt", &muPt_);
    tree_->Branch("muEta", &muEta_);
    tree_->Branch("muPhi", &muPhi_);
    tree_->Branch("muL1Eta", &muL1Eta_);
    tree_->Branch("muL1Phi", &muL1Phi_);
    tree_->Branch("muMVAIso", &muMVAIso_);
    for (auto& w : muMVAIsoWP_)
      tree_->Branch(("muMVAIso"+w.first).c_str(), &(w.second));
    tree_->Branch("muCharge", &muCharge_);
    tree_->Branch("muType", &muType_);
    tree_->Branch("muIsGood", &muIsGood_);

    tree_->Branch("muIsGlobal", &muIsGlobal_);
    tree_->Branch("muIsTracker", &muIsTracker_);
    tree_->Branch("muIsPF", &muIsPF_);
    tree_->Branch("muIsSTA", &muIsSTA_);

    tree_->Branch("muD0", &muD0_);
    tree_->Branch("muDz", &muDz_);
    tree_->Branch("muIP3D", &muIP3D_);
    tree_->Branch("muD0Err", &muD0Err_);
    tree_->Branch("muDzErr", &muDzErr_);
    tree_->Branch("muIP3DErr", &muIP3DErr_);
    tree_->Branch("muChi2NDF", &muChi2NDF_);
    tree_->Branch("muInnerD0", &muInnerD0_);
    tree_->Branch("muInnerDz", &muInnerDz_);

    tree_->Branch("muInnerD0Err", &muInnerD0Err_);
    tree_->Branch("muInnerDzErr", &muInnerDzErr_);
    tree_->Branch("muInnerPt", &muInnerPt_);
    tree_->Branch("muInnerPtErr", &muInnerPtErr_);
    tree_->Branch("muInnerEta", &muInnerEta_);

    tree_->Branch("muTrkLayers", &muTrkLayers_);
    tree_->Branch("muPixelLayers", &muPixelLayers_);
    tree_->Branch("muPixelHits", &muPixelHits_);
    tree_->Branch("muMuonHits", &muMuonHits_);
    tree_->Branch("muTrkQuality", &muTrkQuality_);
    tree_->Branch("muStations", &muStations_);
    tree_->Branch("muIsoTrk", &muIsoTrk_);
    tree_->Branch("muPFChIso", &muPFChIso_);
    tree_->Branch("muPFPhoIso", &muPFPhoIso_);
    tree_->Branch("muPFNeuIso", &muPFNeuIso_);
    tree_->Branch("muPFPUIso", &muPFPUIso_);

    tree_->Branch("muSelectors", &muSelectors_);
    tree_->Branch("muIDSoft", &muIDSoft_);
    tree_->Branch("muIDLoose", &muIDLoose_);
    tree_->Branch("muIDMedium", &muIDMedium_);
    tree_->Branch("muIDMediumPrompt", &muIDMediumPrompt_);
    tree_->Branch("muIDTight", &muIDTight_);
    tree_->Branch("muIDGlobalHighPt", &muIDGlobalHighPt_);
    tree_->Branch("muIDTrkHighPt", &muIDTrkHighPt_);
    tree_->Branch("muIDInTime", &muIDInTime_);

    if (doGenParticles_)
      tree_->Branch("mu_genMatchedIndex", &mu_genMatchedIndex_);
  }
}

ggHiNtuplizer::~ggHiNtuplizer() {}

void ggHiNtuplizer::analyze(const edm::Event& e, const edm::EventSetup& es) {
  // cleanup from previous event

  if (doGenParticles_) {
    nPUInfo_ = 0;
    nPU_.clear();
    puBX_.clear();
    puTrue_.clear();

    nMC_ = 0;

    mcVtx_x_.clear();
    mcVtx_y_.clear();
    mcVtx_z_.clear();

    mcPID_.clear();
    mcStatus_.clear();
    mcPt_.clear();
    mcEta_.clear();
    mcPhi_.clear();
    mcE_.clear();
    mcEt_.clear();
    mcMass_.clear();

    mcParentage_.clear();
    mcMomPID_.clear();
    mcMomKey_.clear();
    mcMomPt_.clear();
    mcMomEta_.clear();
    mcMomPhi_.clear();
    mcMomMass_.clear();
    mcGMomPID_.clear();
    mcGMomKey_.clear();
    mcGGMomPID_.clear();
    mcGGMomKey_.clear();
    mcIndex_.clear();
    mcSube_.clear();

    mcCalIsoDR03_.clear();
    mcCalIsoDR04_.clear();
    mcTrkIsoDR03_.clear();
    mcTrkIsoDR04_.clear();
  }

  if (doSuperClusters_) {
    nSC_ = 0;
    scE_.clear();
    scRawE_.clear();
    scEta_.clear();
    scPhi_.clear();
  }

  if (doElectrons_) {
    nEle_ = 0;

    eleD0_.clear();
    eleDz_.clear();
    eleD0Err_.clear();
    eleDzErr_.clear();
    eleTrkPt_.clear();
    eleTrkEta_.clear();
    eleTrkPhi_.clear();
    eleTrkCharge_.clear();
    eleTrkPtErr_.clear();
    eleTrkChi2_.clear();
    eleTrkNdof_.clear();
    eleTrkNormalizedChi2_.clear();
    eleTrkValidHits_.clear();
    eleTrkLayers_.clear();
    eleMissHits_.clear();
    eleIP3D_.clear();
    eleIP3DErr_.clear();

    elePt_.clear();
    eleRawPt_.clear();
    eleEta_.clear();
    elePhi_.clear();
    eleCharge_.clear();
    eleEn_.clear();

    eleMVAIso_.clear();
    for (auto& w : eleMVAIsoWP_)
      w.second.clear();
    eleMVAId_.clear();
    for (auto& w : eleMVAIdWP_)
      w.second.clear();
    for (auto& w : eleCutIdWP_)
      w.second.clear();

    eleSCEn_.clear();
    eleESEn_.clear();
    eleSCEta_.clear();
    eleSCPhi_.clear();
    eleSCRawEn_.clear();
    eleSCEtaWidth_.clear();
    eleSCPhiWidth_.clear();
    eleSCClustersSize_.clear();
    eleSeedEn_.clear();
    eleSeedEta_.clear();
    eleSeedPhi_.clear();

    eleHoverE_.clear();
    eleHoverEBc_.clear();
    eleRawHoverE_.clear();
    eleRawHoverEBc_.clear();
    eleEoverP_.clear();
    eleEoverPInv_.clear();
    eleEcalE_.clear();
    eleRawEcalE_.clear();
    elePAtVtx_.clear();
    elePAtSC_.clear();
    elePAtCluster_.clear();
    elePAtSeed_.clear();
    eleBrem_.clear();
    eledEtaAtVtx_.clear();
    eledPhiAtVtx_.clear();
    eledEtaSeedAtVtx_.clear();
    eleSigmaIEtaIEta_.clear();
    eleSigmaIPhiIPhi_.clear();

    eleConvVeto_.clear();

    elePFChIso_.clear();
    elePFPhoIso_.clear();
    elePFNeuIso_.clear();
    elePFPUIso_.clear();

    elePFRelIsoWithEA_.clear();
    elePFRelIsoWithDBeta_.clear();
    eleEffAreaTimesRho_.clear();

    if (doPfIso_) {
      elePFChIso03_.clear();
      elePFPhoIso03_.clear();
      elePFNeuIso03_.clear();
      elePFChIso04_.clear();
      elePFPhoIso04_.clear();
      elePFNeuIso04_.clear();
    }

    eleR9_.clear();
    eleE3x3_.clear();
    eleE5x5_.clear();
    eleR9Full5x5_.clear();
    eleE3x3Full5x5_.clear();
    eleE5x5Full5x5_.clear();
    eleSigmaIEtaIEta_2012_.clear();

    eleSeedCryEta_.clear();
    eleSeedCryPhi_.clear();
    eleSeedCryIeta_.clear();
    eleSeedCryIphi_.clear();

    ele_genMatchedIndex_.clear();
  }

  if (doPhotons_) {
    nPho_ = 0;

    phoE_.clear();
    phoEt_.clear();
    phoEta_.clear();
    phoPhi_.clear();

    phoEcorrStdEcal_.clear();
    phoEcorrPhoEcal_.clear();
    phoEcorrRegr1_.clear();
    phoEcorrRegr2_.clear();
    phoEcorrErrStdEcal_.clear();
    phoEcorrErrPhoEcal_.clear();
    phoEcorrErrRegr1_.clear();
    phoEcorrErrRegr2_.clear();

    phoSCE_.clear();
    phoSCRawE_.clear();
    phoSCEta_.clear();
    phoSCPhi_.clear();
    phoSCEtaWidth_.clear();
    phoSCPhiWidth_.clear();
    phoSCBrem_.clear();
    phoSCnHits_.clear();
    phoSCflags_.clear();
    phoSCinClean_.clear();
    phoSCinUnClean_.clear();
    phoSCnBC_.clear();
    phoESEn_.clear();

    phoIsPFPhoton_.clear();
    phoIsStandardPhoton_.clear();
    phoHasPixelSeed_.clear();
    phoHasConversionTracks_.clear();
    phoHadTowerOverEm_.clear();
    phoHoverE_.clear();
    phoHoverEValid_.clear();
    phoSigmaIEtaIEta_.clear();
    phoR9_.clear();
    phoE1x5_.clear();
    phoE2x5_.clear();
    phoE3x3_.clear();
    phoE5x5_.clear();
    phoMaxEnergyXtal_.clear();
    phoSigmaEtaEta_.clear();
    phoSigmaIEtaIEta_2012_.clear();
    phoR9_2012_.clear();
    phoE1x5_2012_.clear();
    phoE2x5_2012_.clear();
    phoE3x3_2012_.clear();
    phoE5x5_2012_.clear();
    phoMaxEnergyXtal_2012_.clear();
    phoSigmaEtaEta_2012_.clear();

    if (doPhoEReg_) {
      phoHadTowerOverEm1_.clear();
      phoHadTowerOverEm2_.clear();
      phoHoverE1_.clear();
      phoHoverE2_.clear();

      phoSigmaIEtaIPhi_.clear();
      phoSigmaIPhiIPhi_.clear();
      phoR1x5_.clear();
      phoR2x5_.clear();
      phoE2nd_.clear();
      phoETop_.clear();
      phoEBottom_.clear();
      phoELeft_.clear();
      phoERight_.clear();
      phoE1x3_.clear();
      phoE2x2_.clear();
      phoE2x5Max_.clear();
      phoE2x5Top_.clear();
      phoE2x5Bottom_.clear();
      phoE2x5Left_.clear();
      phoE2x5Right_.clear();
      //phoSMMajor_.clear();
      //phoSMMinor_.clear();
      //phoSMAlpha_.clear();

      phoSigmaIEtaIPhi_2012_.clear();
      phoSigmaIPhiIPhi_2012_.clear();
      phoR1x5_2012_.clear();
      phoR2x5_2012_.clear();
      phoE2nd_2012_.clear();
      phoETop_2012_.clear();
      phoEBottom_2012_.clear();
      phoELeft_2012_.clear();
      phoERight_2012_.clear();
      phoE1x3_2012_.clear();
      phoE2x2_2012_.clear();
      phoE2x5Max_2012_.clear();
      phoE2x5Top_2012_.clear();
      phoE2x5Bottom_2012_.clear();
      phoE2x5Left_2012_.clear();
      phoE2x5Right_2012_.clear();
      //phoSMMajor_2012_.clear();
      //phoSMMinor_2012_.clear();
      //phoSMAlpha_2012_.clear();
    }

    phoBC1E_.clear();
    phoBC1Ecorr_.clear();
    phoBC1Eta_.clear();
    phoBC1Phi_.clear();
    phoBC1size_.clear();
    phoBC1flags_.clear();
    phoBC1inClean_.clear();
    phoBC1inUnClean_.clear();
    phoBC1rawID_.clear();

    if (useValMapIso_) {
      pho_ecalClusterIsoR1_.clear();
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
    }

    pho_genMatchedIndex_.clear();

    // rechit info
    if (doRecHitsEB_ || doRecHitsEE_) {
      nRH_ = 0;
      rhRawId_.clear();
      rhieta_.clear();
      rhiphi_.clear();
      rhix_.clear();
      rhiy_.clear();
      rhE_.clear();
      rhEt_.clear();
      rhEta_.clear();
      rhPhi_.clear();
      rhChi2_.clear();
      rhEerror_.clear();
      rhFlags_.clear();
      rhPhoIdx_.clear();
      rhBCIdx_.clear();
    }

    //photon pf isolation stuff
    if (doPfIso_) {
      pfcIso1_.clear();
      pfcIso2_.clear();
      pfcIso3_.clear();
      pfcIso4_.clear();
      pfcIso5_.clear();
      pfpIso1_.clear();
      pfpIso2_.clear();
      pfpIso3_.clear();
      pfpIso4_.clear();
      pfpIso5_.clear();
      pfnIso1_.clear();
      pfnIso2_.clear();
      pfnIso3_.clear();
      pfnIso4_.clear();
      pfnIso5_.clear();
      pfpIso1subSC_.clear();
      pfpIso2subSC_.clear();
      pfpIso3subSC_.clear();
      pfpIso4subSC_.clear();
      pfpIso5subSC_.clear();

      pfcIso1subUE_.clear();
      pfcIso2subUE_.clear();
      pfcIso3subUE_.clear();
      pfcIso4subUE_.clear();
      pfcIso5subUE_.clear();
      pfpIso1subUE_.clear();
      pfpIso2subUE_.clear();
      pfpIso3subUE_.clear();
      pfpIso4subUE_.clear();
      pfpIso5subUE_.clear();
      pfnIso1subUE_.clear();
      pfnIso2subUE_.clear();
      pfnIso3subUE_.clear();
      pfnIso4subUE_.clear();
      pfnIso5subUE_.clear();
      pfpIso1subSCsubUE_.clear();
      pfpIso2subSCsubUE_.clear();
      pfpIso3subSCsubUE_.clear();
      pfpIso4subSCsubUE_.clear();
      pfpIso5subSCsubUE_.clear();
      pfcIso1pTgt1p0subUE_.clear();
      pfcIso2pTgt1p0subUE_.clear();
      pfcIso3pTgt1p0subUE_.clear();
      pfcIso4pTgt1p0subUE_.clear();
      pfcIso5pTgt1p0subUE_.clear();
      pfcIso1pTgt2p0subUE_.clear();
      pfcIso2pTgt2p0subUE_.clear();
      pfcIso3pTgt2p0subUE_.clear();
      pfcIso4pTgt2p0subUE_.clear();
      pfcIso5pTgt2p0subUE_.clear();
      pfcIso1pTgt3p0subUE_.clear();
      pfcIso2pTgt3p0subUE_.clear();
      pfcIso3pTgt3p0subUE_.clear();
      pfcIso4pTgt3p0subUE_.clear();
      pfcIso5pTgt3p0subUE_.clear();

      pfcIso2subUEec_.clear();
      pfcIso3subUEec_.clear();
      pfcIso4subUEec_.clear();

      pfpIso2subUEec_.clear();
      pfpIso3subUEec_.clear();
      pfpIso4subUEec_.clear();

      pfnIso2subUEec_.clear();
      pfnIso3subUEec_.clear();
      pfnIso4subUEec_.clear();

      pfcIso2pTgt2p0subUEec_.clear();
      pfcIso3pTgt2p0subUEec_.clear();
      pfcIso4pTgt2p0subUEec_.clear();
    }
  }

  if (doMuons_) {
    nMu_ = 0;

    muPt_.clear();
    muEta_.clear();
    muPhi_.clear();
    muL1Eta_.clear();
    muL1Phi_.clear();
    muMVAIso_.clear();
    for (auto& w : muMVAIsoWP_)
      w.second.clear();
    muCharge_.clear();
    muType_.clear();
    muIsGood_.clear();

    muIsGlobal_.clear();
    muIsTracker_.clear();
    muIsPF_.clear();
    muIsSTA_.clear();

    muD0_.clear();
    muDz_.clear();
    muD0Err_.clear();
    muDzErr_.clear();
    muChi2NDF_.clear();
    muInnerD0_.clear();
    muInnerDz_.clear();
    muIP3D_.clear();
    muIP3DErr_.clear();

    muInnerD0Err_.clear();
    muInnerDzErr_.clear();
    muInnerPt_.clear();
    muInnerPtErr_.clear();
    muInnerEta_.clear();

    muTrkLayers_.clear();
    muPixelLayers_.clear();
    muPixelHits_.clear();
    muMuonHits_.clear();
    muTrkQuality_.clear();
    muStations_.clear();
    muIsoTrk_.clear();
    muPFChIso_.clear();
    muPFPhoIso_.clear();
    muPFNeuIso_.clear();
    muPFPUIso_.clear();

    muSelectors_.clear();
    muIDSoft_.clear();
    muIDLoose_.clear();
    muIDMedium_.clear();
    muIDMediumPrompt_.clear();
    muIDTight_.clear();
    muIDGlobalHighPt_.clear();
    muIDTrkHighPt_.clear();
    muIDInTime_.clear();

    mu_genMatchedIndex_.clear();
  }

  run_ = e.id().run();
  event_ = e.id().event();
  lumis_ = e.luminosityBlock();

  edm::Handle<double> rhoH;
  e.getByToken(rhoToken_, rhoH);
  rho_ = *rhoH;

  // MC truth
  if (doGenParticles_) {
    fillGenPileupInfo(e);
    fillGenParticles(e);
  }

  edm::Handle<std::vector<reco::Vertex>> vertices;
  e.getByToken(vertexToken_, vertices);

  // best-known primary vertex coordinates
  reco::Vertex pv(math::XYZPoint(0, 0, -999), math::Error<3>::type());
  for (auto const& vertex : *vertices) {
    if (!vertex.isFake()) {
      pv = vertex;
      break;
    }
  }

  tb = &es.getData(ttbESToken_);

  if (doRecHitsEB_ || doRecHitsEE_ || doElectrons_)
    geo = &es.getData(geometryToken_);

  if (doSuperClusters_)
    fillSC(e);
  if (doElectrons_)
    fillElectrons(e, es, pv);
  if (doPhotons_)
    fillPhotons(e, es, pv);
  if (doMuons_)
    fillMuons(e, es, pv);

  tree_->Fill();
}

void ggHiNtuplizer::fillGenPileupInfo(const edm::Event& e) {
  // Fills information about pileup from MC truth.
  edm::Handle<std::vector<PileupSummaryInfo>> pileup;
  e.getByToken(pileupToken_, pileup);

  for (const auto& pu : *pileup) {
    nPU_.push_back(pu.getPU_NumInteractions());
    puTrue_.push_back(pu.getTrueNumInteractions());
    puBX_.push_back(pu.getBunchCrossing());

    ++nPUInfo_;
  }
}

void ggHiNtuplizer::fillGenParticles(const edm::Event& e) {
  if (doPackedGenParticle_) {
    edm::Handle<std::vector<pat::PackedGenParticle>> packedGenParticles;
    e.getByToken(packedGenParticlesToken_, packedGenParticles);

    edm::Handle<edm::View<pat::PackedGenParticle>> signalPackedGenParticles;
    bool hasSignalPackedGen = e.getByToken(signalPackedGenParticlesToken_, signalPackedGenParticles);

    fillGenCandidates(packedGenParticles, signalPackedGenParticles, hasSignalPackedGen);
  } else {
    // ToDo
    edm::Handle<std::vector<reco::GenParticle>> genParticles;
    e.getByToken(genParticlesToken_, genParticles);

    edm::Handle<edm::View<reco::GenParticle> > dummyHandle;

    fillGenCandidates(genParticles, dummyHandle, false);
  }
}

template <typename T, typename T2>
void ggHiNtuplizer::fillGenCandidates(const edm::Handle<std::vector<T>>& handle,
                                      const edm::Handle<edm::View<T2>>& signalPartHandle,
                                      bool hasSignalPart) {
  for (size_t ig = 0; ig < handle->size(); ig++) {
    const auto& p = edm::Ref<std::vector<T>>(handle, ig);
    // skip all primary particles if not particle gun MC
    //if (!isParticleGun_ && !p->mother())
    if (!isParticleGun_ && p->numberOfMothers() == 0)
      continue;

    // require last copy
    if (!p->statusFlags().isLastCopy())
      continue;

    // stable particles with pT > 5 GeV
    bool isStableFast = (p->status() == 1 && p->pt() > 5.0);

    // stable leptons
    const auto pdgId = std::abs(p->pdgId());
    bool isStableLepton = (p->status() == 1 && pdgId >= 11 && pdgId <= 16);

    // (unstable) Z, W, H, top, bottom
    bool isHeavy = (pdgId == 23 || pdgId == 24 || pdgId == 25 || pdgId == 6 || pdgId == 5);

    // reduce size of output root file
    if (!isStableFast && !isStableLepton && !isHeavy)
      continue;

    mcVtx_x_.push_back(p->vx());
    mcVtx_y_.push_back(p->vy());
    mcVtx_z_.push_back(p->vz());

    mcPID_.push_back(p->pdgId());
    mcStatus_.push_back(p->status());
    mcPt_.push_back(p->pt());
    mcEta_.push_back(p->eta());
    mcPhi_.push_back(p->phi());
    mcE_.push_back(p->energy());
    mcEt_.push_back(p->et());
    mcMass_.push_back(p->mass());

    GenParticleParentage particleHistory(p);
    const auto& partRef = particleHistory.match();
    mcParentage_.push_back((particleHistory.hasLeptonParent() << 4) + (particleHistory.hasBosonParent() << 3) +
                           (particleHistory.hasNonPromptParent() << 2) + (particleHistory.hasQCDParent() << 1) +
                           (particleHistory.hasExoticParent() << 0));

    int momPID = -999;
    int momKey = -1;
    float momPt = -999;
    float momEta = -999;
    float momPhi = -999;
    float momMass = -999;
    int gmomPID = -999;
    int gmomKey = -1;
    int ggmomPID = -999;
    int ggmomKey = -1;

    // mother
    const auto& momRef = GenParticleParentage::findGenMother(partRef);
    if (momRef.isNonnull() && momRef.isAvailable()) {
      momPID = momRef->pdgId();
      momKey = std::abs(momPID) < 1E5 ? (std::abs(momPID) + momRef.key()*1E5) : -1;
      momPt = momRef->pt();
      momEta = momRef->eta();
      momPhi = momRef->phi();
      momMass = momRef->mass();

      // grandmother
      const auto& grandmother = GenParticleParentage::findGenMother(momRef);
      if (grandmother.isNonnull() && grandmother.isAvailable()) {
        gmomPID = grandmother->pdgId();
        gmomKey = std::abs(gmomPID) < 1E5 ? (std::abs(gmomPID) + grandmother.key()*1E5) : -1;

        // grand-grandmother
        const auto& ggmother = GenParticleParentage::findGenMother(grandmother);
        if (ggmother.isNonnull() && ggmother.isAvailable()) {
          ggmomPID = ggmother->pdgId();
          ggmomKey = std::abs(ggmomPID) < 1E5 ? (std::abs(ggmomPID) + ggmother.key()*1E5) : -1;
        }
      }
    }

    mcMomPID_.push_back(momPID);
    mcMomKey_.push_back(momKey);
    mcMomPt_.push_back(momPt);
    mcMomEta_.push_back(momEta);
    mcMomPhi_.push_back(momPhi);
    mcMomMass_.push_back(momMass);
    mcGMomPID_.push_back(gmomPID);
    mcGMomKey_.push_back(gmomKey);
    mcGGMomPID_.push_back(ggmomPID);
    mcGGMomKey_.push_back(ggmomKey);

    mcIndex_.push_back(partRef.key());

    bool isSigPart = false;
    if (doPackedGenParticle_) {
      int tmpSube = -999;
      if (hasSignalPart) {
        for (auto pSig = signalPartHandle->begin(); pSig != signalPartHandle->end(); ++pSig) {
          if (&(*pSig) == &(*p)) {
            tmpSube = 0;
            isSigPart = true;
            break;
          }
        }
        if (!isSigPart) {
          tmpSube = 1;
        }
      }
      mcSube_.push_back(tmpSube);
    } else {
      int tmpCollId = (dynamic_cast<const reco::GenParticle&>(*p)).collisionId();
      isSigPart = (tmpCollId == 0);
      mcSube_.push_back(tmpCollId);
    }

    if (hasSignalPart && isSigPart) {
      mcCalIsoDR03_.push_back(getGenCalIso(signalPartHandle, p, 0.3 * 0.3, false, false));
      mcCalIsoDR04_.push_back(getGenCalIso(signalPartHandle, p, 0.4 * 0.4, false, false));
      mcTrkIsoDR03_.push_back(getGenTrkIso(signalPartHandle, p, 0.3 * 0.3));
      mcTrkIsoDR04_.push_back(getGenTrkIso(signalPartHandle, p, 0.4 * 0.4));
    } else {
      mcCalIsoDR03_.push_back(-999999);
      mcCalIsoDR04_.push_back(-999999);
      mcTrkIsoDR03_.push_back(-999999);
      mcTrkIsoDR04_.push_back(-999999);
    }

    nMC_++;
  }
}

template <typename T, typename T2>
float ggHiNtuplizer::getGenCalIso(
    const edm::Handle<edm::View<T>>& handle, T2 thisPart, float dR2Max, bool removeMu, bool removeNu) {
  float etSum = 0;

  for (auto p = handle->begin(); p != handle->end(); ++p) {
    if (getGenCalIsoPass(&(*p), &(*thisPart), dR2Max, removeMu, removeNu)) {
      etSum += p->et();
    }
  }

  return etSum;
}

bool ggHiNtuplizer::getGenCalIsoPass(
    const reco::Candidate* p, const reco::Candidate* thisPart, float dR2Max, bool removeMu, bool removeNu) {
  if (p == thisPart)
    return false;
  if (p->status() != 1)
    return false;

  // no collisionId info for PackedGenParticle
  if (!doPackedGenParticle_) {
    if ((dynamic_cast<const reco::GenParticle&>(*thisPart)).collisionId() !=
        (dynamic_cast<const reco::GenParticle&>(*p)).collisionId())
      return false;
  }

  int pdgCode = abs(p->pdgId());

  // skip muons/neutrinos, if requested
  if (removeMu && pdgCode == 13)
    return false;
  if (removeNu && (pdgCode == 12 || pdgCode == 14 || pdgCode == 16))
    return false;

  // must be within deltaR cone
  float dR2 = reco::deltaR2(thisPart->momentum(), p->momentum());
  if (dR2 > dR2Max)
    return false;

  return true;
}

template <typename T, typename T2>
float ggHiNtuplizer::getGenTrkIso(const edm::Handle<edm::View<T>>& handle, T2 thisPart, float dR2Max) {
  float ptSum = 0;

  for (auto p = handle->begin(); p != handle->end(); ++p) {
    if (getGenTrkIsoPass(&(*p), &(*thisPart), dR2Max)) {
      ptSum += p->pt();
    }
  }

  return ptSum;
}

bool ggHiNtuplizer::getGenTrkIsoPass(const reco::Candidate* p, const reco::Candidate* thisPart, float dR2Max) {
  if (p == thisPart)
    return false;
  if (p->status() != 1)
    return false;
  // exclude neutral particles
  if (p->charge() == 0)
    return false;

  // no collisionId info for PackedGenParticle
  if (!doPackedGenParticle_) {
    if ((dynamic_cast<const reco::GenParticle&>(*thisPart)).collisionId() !=
        (dynamic_cast<const reco::GenParticle&>(*p)).collisionId())
      return false;
  }

  // must be within deltaR2 cone
  float dR2 = reco::deltaR2(thisPart->momentum(), p->momentum());
  if (dR2 > dR2Max)
    return false;

  return true;
}

void ggHiNtuplizer::fillSC(edm::Event const& e) {
  edm::Handle<std::vector<reco::SuperCluster>> scHandle;
  e.getByToken(scToken_, scHandle);

  for (auto const& sc : *scHandle) {
    scE_.push_back(sc.energy());
    scRawE_.push_back(sc.rawEnergy());
    scEta_.push_back(sc.eta());
    scPhi_.push_back(sc.phi());

    ++nSC_;
  }
}

void ggHiNtuplizer::fillElectrons(const edm::Event& e, const edm::EventSetup& es, reco::Vertex& pv) {
  // Fills tree branches with electrons.
  edm::Handle<edm::View<pat::Electron>> gsfElectrons;
  e.getByToken(electronsToken_, gsfElectrons);

  edm::Handle<reco::ConversionCollection> conversions;
  e.getByToken(conversionsToken_, conversions);

  edm::Handle<reco::BeamSpot> beamSpot;
  e.getByToken(beamSpotToken_, beamSpot);

  // loop over electrons
  for (auto ele = gsfElectrons->begin(); ele != gsfElectrons->end(); ++ele) {
    eleD0_.push_back(ele->gsfTrack()->dxy(pv.position()));
    eleDz_.push_back(ele->gsfTrack()->dz(pv.position()));
    eleD0Err_.push_back(ele->gsfTrack()->dxyError());
    eleDzErr_.push_back(ele->gsfTrack()->dzError());
    eleTrkPt_.push_back(ele->gsfTrack()->pt());
    eleTrkEta_.push_back(ele->gsfTrack()->eta());
    eleTrkPhi_.push_back(ele->gsfTrack()->phi());
    eleTrkCharge_.push_back(ele->gsfTrack()->charge());
    eleTrkPtErr_.push_back(ele->gsfTrack()->ptError());
    eleTrkChi2_.push_back(ele->gsfTrack()->chi2());
    eleTrkNdof_.push_back(ele->gsfTrack()->ndof());
    eleTrkNormalizedChi2_.push_back(ele->gsfTrack()->normalizedChi2());
    eleTrkValidHits_.push_back(ele->gsfTrack()->numberOfValidHits());
    eleTrkLayers_.push_back(ele->gsfTrack()->hitPattern().trackerLayersWithMeasurement());
    eleMissHits_.push_back(ele->gsfTrack()->numberOfLostHits());

    elePt_.push_back(ele->pt());
    eleRawPt_.push_back(ele->hasUserFloat("rawPt") ? ele->userFloat("rawPt") : ele->pt());
    eleEta_.push_back(ele->eta());
    elePhi_.push_back(ele->phi());
    eleCharge_.push_back(ele->charge());
    eleEn_.push_back(ele->energy());

    eleMVAIso_.push_back(ele->hasUserFloat("hiMVAIso") ? ele->userFloat("hiMVAIso") : -99);
    for (auto& w : eleMVAIsoWP_)
      w.second.push_back(ele->hasUserInt("hiMVAIso"+w.first) ? ele->userInt("hiMVAIso"+w.first) : -1);
    eleMVAId_.push_back(ele->hasUserFloat("hiMVAId") ? ele->userFloat("hiMVAId") : -99);
    for (auto& w : eleMVAIdWP_)
      w.second.push_back(ele->hasUserInt("hiMVAId"+w.first) ? ele->userInt("hiMVAId"+w.first) : -1);
    for (auto& w : eleCutIdWP_)
      w.second.push_back(ele->hasUserInt("hiCutId"+w.first) ? ele->userInt("hiCutId"+w.first) : -1);

    eleSCEn_.push_back(ele->superCluster()->energy());
    eleESEn_.push_back(ele->superCluster()->preshowerEnergy());
    eleSCEta_.push_back(ele->superCluster()->eta());
    eleSCPhi_.push_back(ele->superCluster()->phi());
    eleSCRawEn_.push_back(ele->superCluster()->rawEnergy());
    eleSCEtaWidth_.push_back(ele->superCluster()->etaWidth());
    eleSCPhiWidth_.push_back(ele->superCluster()->phiWidth());
    eleSCClustersSize_.push_back(ele->superCluster()->clustersSize());
    eleSeedEn_.push_back(ele->superCluster()->seed()->energy());
    eleSeedEta_.push_back(ele->superCluster()->seed()->eta());
    eleSeedPhi_.push_back(ele->superCluster()->seed()->phi());

    eleHoverE_.push_back(ele->hcalOverEcal());
    eleHoverEBc_.push_back(ele->hcalOverEcalBc());
    eleRawHoverE_.push_back(ele->full5x5_hcalOverEcal());
    eleRawHoverEBc_.push_back(ele->full5x5_hcalOverEcalBc());
    eleEoverP_.push_back(ele->eSuperClusterOverP());
    eleEoverPInv_.push_back(1. / ele->ecalEnergy() - 1. / ele->trackMomentumAtVtx().R());
    eleEcalE_.push_back(ele->ecalEnergy());
    eleRawEcalE_.push_back(ele->hasUserFloat("rawEcalEnergy") ? ele->userFloat("rawEcalEnergy") : ele->ecalEnergy());
    elePAtVtx_.push_back(ele->trackMomentumAtVtx().R());
    elePAtSC_.push_back(ele->trackMomentumAtCalo().R());
    elePAtCluster_.push_back(ele->trackMomentumAtEleClus().R());
    elePAtSeed_.push_back(ele->trackMomentumOut().R());
    eledEtaAtVtx_.push_back(ele->deltaEtaSuperClusterTrackAtVtx());
    eledPhiAtVtx_.push_back(ele->deltaPhiSuperClusterTrackAtVtx());
    eledEtaSeedAtVtx_.push_back(ele->deltaEtaSeedClusterTrackAtVtx());
    eleSigmaIEtaIEta_.push_back(ele->sigmaIetaIeta());
    eleSigmaIPhiIPhi_.push_back(ele->sigmaIphiIphi());
    eleBrem_.push_back(ele->fbrem());

    /* updated in CMSSW_10_6_X */
    bool passConvVeto = !ConversionTools::hasMatchedConversion(*ele, *conversions, beamSpot->position());
    eleConvVeto_.push_back((int)passConvVeto);

    // full 5x5
    eleR9_.push_back(ele->r9());
    eleE3x3_.push_back(ele->r9() * ele->superCluster()->energy());
    eleE5x5_.push_back(ele->e5x5());
    eleR9Full5x5_.push_back(ele->full5x5_r9());
    eleE3x3Full5x5_.push_back(ele->full5x5_r9() * ele->superCluster()->energy());
    eleE5x5Full5x5_.push_back(ele->full5x5_e5x5());
    eleSigmaIEtaIEta_2012_.push_back(ele->full5x5_sigmaIetaIeta());

    // isolation
    reco::GsfElectron::PflowIsolationVariables pfIso = ele->pfIsolationVariables();
    elePFChIso_.push_back(pfIso.sumChargedHadronPt);
    elePFPhoIso_.push_back(pfIso.sumPhotonEt);
    elePFNeuIso_.push_back(pfIso.sumNeutralHadronEt);
    elePFPUIso_.push_back(pfIso.sumPUPt);

    if (doEffectiveAreas_) {
      double area = effectiveAreas_.getEffectiveArea(ele->superCluster()->eta());
      elePFRelIsoWithEA_.push_back(
          (pfIso.sumChargedHadronPt + std::max(0.0, pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt - rho_ * area)) /
          ele->pt());
      elePFRelIsoWithDBeta_.push_back(
          (pfIso.sumChargedHadronPt +
           std::max(0.0, pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt - 0.5 * pfIso.sumPUPt)) /
          ele->pt());
      eleEffAreaTimesRho_.push_back(area * rho_);
    }

    // initialize with unphysical values
    float eleIP3D = -999;
    float eleIP3DErr = -999;
    if (pv.isValid()) {
      // 3D impact parameter
      reco::TransientTrack tt = tb->build(ele->gsfTrack().get());
      eleIP3D = IPTools::absoluteImpactParameter3D(tt, pv).second.value();
      eleIP3DErr = IPTools::absoluteImpactParameter3D(tt, pv).second.error();
    }

    eleIP3D_.push_back(eleIP3D);
    eleIP3DErr_.push_back(eleIP3DErr);

    // calculation on the fly
    if (doPfIso_) {
      pfIsoCalculator pfIsoCal;
      pfIsoCal.setUsePackedCandidates(isPackedPFCandidate_);
      pfIsoCal.setVertex(pv.position());
      if (isPackedPFCandidate_) {
        edm::Handle<edm::View<pat::PackedCandidate>> candidatesPacked;
        e.getByToken(packedPFCandToken_, candidatesPacked);
        pfIsoCal.setCandidatesPacked(candidatesPacked);
      }
      /*
      else {
        edm::Handle<edm::View< reco::PFCandidate >> candidatesReco;
        e.getByToken(recoPFCandToken_, candidatesReco);
        pfIsoCal.setCandidatesReco(candidatesReco);
      }
      */
      if (std::abs(ele->superCluster()->eta()) > 1.566) {
        elePFChIso03_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::h, 0.3, 0.015, 0.));
        elePFChIso04_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::h, 0.4, 0.015, 0.));
        elePFPhoIso03_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::gamma, 0.3, 0.08, 0.));
        elePFPhoIso04_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::gamma, 0.4, 0.08, 0.));
      } else {
        elePFChIso03_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::h, 0.3, 0.0, 0.));
        elePFChIso04_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::h, 0.4, 0.0, 0.));
        elePFPhoIso03_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::gamma, 0.3, 0.0, 0.));
        elePFPhoIso04_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::gamma, 0.4, 0.0, 0.));
      }

      elePFNeuIso03_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::h0, 0.3, 0., 0.));
      elePFNeuIso04_.push_back(pfIsoCal.getPfIso(*ele, reco::PFCandidate::h0, 0.4, 0., 0.));
    }

    // local coordinates
    edm::Ptr<reco::CaloCluster> theseed = ele->superCluster()->seed();
    auto subdetid = theseed->hitsAndFractions().at(0).first.subdetId();

    /* updated in CMSSW_10_4_X */
    //EcalClusterLocal local;

    /* x, y instead of eta, phi in the endcap */
    float eta, phi, thetatilt, phitilt;
    int ieta, iphi;

    if (subdetid == EcalBarrel) {
      //local.localCoordsEB(*theseed, es, eta, phi, ieta, iphi, thetatilt, phitilt);
      egammaTools::localEcalClusterCoordsEB(*theseed, *geo, eta, phi, ieta, iphi, thetatilt, phitilt);
    } else {
      //local.localCoordsEE(*theseed, es, eta, phi, ieta, iphi, thetatilt, phitilt);
      egammaTools::localEcalClusterCoordsEE(*theseed, *geo, eta, phi, ieta, iphi, thetatilt, phitilt);
    }

    eleSeedCryEta_.push_back(eta);
    eleSeedCryPhi_.push_back(phi);
    eleSeedCryIeta_.push_back(ieta);
    eleSeedCryIphi_.push_back(iphi);

    /////////////////////////////// MC matching //////////////////////////
    if (doGenParticles_) {
      float minDeltaR2(0.0225);
      int matchedIndex = -1;
      for (size_t igen = 0; igen < mcEt_.size(); ++igen) {
        if (mcStatus_[igen] != 1 || mcPID_[igen] != -11*ele->charge())
          continue;
        const auto deltaR2 = reco::deltaR2(ele->eta(), ele->phi(), mcEta_[igen], mcPhi_[igen]);
        if (deltaR2 < minDeltaR2 && std::abs(mcPt_[igen] - ele->pt()) < 2*mcPt_[igen]) {
          minDeltaR2 = deltaR2;
          matchedIndex = igen;
        }
      }
      ele_genMatchedIndex_.push_back(matchedIndex);
    }

    ++nEle_;
  }  // electrons loop
}

void ggHiNtuplizer::fillPhotons(const edm::Event& e, const edm::EventSetup& es, reco::Vertex& pv) {
  edm::Handle<edm::View<pat::Photon>> patPhotons;
  e.getByToken(photonsToken_, patPhotons);

  edm::Handle<EcalRecHitCollection> recHitsEBHandle;
  if (doRecHitsEB_) {
    e.getByToken(recHitsEB_, recHitsEBHandle);
  }
  edm::Handle<EcalRecHitCollection> recHitsEEHandle;
  if (doRecHitsEE_) {
    e.getByToken(recHitsEE_, recHitsEEHandle);
  }

  // loop over photons
  for (edm::View<pat::Photon>::const_iterator pho = patPhotons->begin(); pho != patPhotons->end(); ++pho) {
    //const reco::Photon * pho = dynamic_cast<const reco::Photon *>(pPho->originalObject());
    //edm::Ptr<reco::Candidate> ptr(pho.originalObjectRef());

    phoE_.push_back(pho->energy());
    phoEt_.push_back(pho->et());
    phoEta_.push_back(pho->eta());
    phoPhi_.push_back(pho->phi());

    // energies from different types of corrections
    phoEcorrStdEcal_.push_back(pho->getCorrectedEnergy(reco::Photon::P4type::ecal_standard));
    phoEcorrPhoEcal_.push_back(pho->getCorrectedEnergy(reco::Photon::P4type::ecal_photons));
    phoEcorrRegr1_.push_back(pho->getCorrectedEnergy(reco::Photon::P4type::regression1));
    phoEcorrRegr2_.push_back(pho->getCorrectedEnergy(reco::Photon::P4type::regression2));
    // errors for those corrections
    phoEcorrErrStdEcal_.push_back(pho->getCorrectedEnergyError(reco::Photon::P4type::ecal_standard));
    phoEcorrErrPhoEcal_.push_back(pho->getCorrectedEnergyError(reco::Photon::P4type::ecal_photons));
    phoEcorrErrRegr1_.push_back(pho->getCorrectedEnergyError(reco::Photon::P4type::regression1));
    phoEcorrErrRegr2_.push_back(pho->getCorrectedEnergyError(reco::Photon::P4type::regression2));

    // SuperCluster info
    phoSCE_.push_back(pho->superCluster()->energy());
    phoSCRawE_.push_back(pho->superCluster()->rawEnergy());
    phoSCEta_.push_back(pho->superCluster()->eta());
    phoSCPhi_.push_back(pho->superCluster()->phi());
    phoSCEtaWidth_.push_back(pho->superCluster()->etaWidth());
    phoSCPhiWidth_.push_back(pho->superCluster()->phiWidth());
    phoSCBrem_.push_back(pho->superCluster()->phiWidth() / pho->superCluster()->etaWidth());
    phoSCnHits_.push_back(pho->superCluster()->size());
    phoSCflags_.push_back(pho->superCluster()->flags());
    phoSCinClean_.push_back((int)pho->superCluster()->isInClean());
    phoSCinUnClean_.push_back((int)pho->superCluster()->isInUnclean());
    phoSCnBC_.push_back((int)pho->superCluster()->clustersSize());
    phoESEn_.push_back(pho->superCluster()->preshowerEnergy());

    phoIsPFPhoton_.push_back((int)pho->isPFlowPhoton());
    phoIsStandardPhoton_.push_back((int)pho->isStandardPhoton());
    phoHasPixelSeed_.push_back((int)pho->hasPixelSeed());
    phoHasConversionTracks_.push_back((int)pho->hasConversionTracks());
    phoHadTowerOverEm_.push_back(pho->hadTowOverEm());
    phoHoverE_.push_back(pho->hadronicOverEm());
    phoHoverEValid_.push_back(pho->hadronicOverEmValid());
    phoSigmaIEtaIEta_.push_back(pho->sigmaIetaIeta());
    phoR9_.push_back(pho->r9());

    // additional shower shape variables
    phoE3x3_.push_back(pho->e3x3());
    phoE1x5_.push_back(pho->e1x5());
    phoE2x5_.push_back(pho->e2x5());
    phoE5x5_.push_back(pho->e5x5());
    phoMaxEnergyXtal_.push_back(pho->maxEnergyXtal());
    phoSigmaEtaEta_.push_back(pho->sigmaEtaEta());

    if (doPhoEReg_) {
      phoHadTowerOverEm1_.push_back(pho->hadTowOverEm(1));
      phoHadTowerOverEm2_.push_back(pho->hadTowOverEm(2));
      phoHoverE1_.push_back(pho->hadronicOverEm(1));
      phoHoverE2_.push_back(pho->hadronicOverEm(2));

      phoSigmaIEtaIPhi_.push_back(pho->showerShapeVariables().sigmaIetaIphi);
      phoSigmaIPhiIPhi_.push_back(pho->showerShapeVariables().sigmaIphiIphi);
      phoR1x5_.push_back(pho->r1x5());
      phoR2x5_.push_back(pho->r2x5());
      phoE2nd_.push_back(pho->showerShapeVariables().e2nd);
      phoETop_.push_back(pho->showerShapeVariables().eTop);
      phoEBottom_.push_back(pho->showerShapeVariables().eBottom);
      phoELeft_.push_back(pho->showerShapeVariables().eLeft);
      phoERight_.push_back(pho->showerShapeVariables().eRight);
      phoE1x3_.push_back(pho->showerShapeVariables().e1x3);
      phoE2x2_.push_back(pho->showerShapeVariables().e2x2);
      phoE2x5Max_.push_back(pho->showerShapeVariables().e2x5Max);
      phoE2x5Top_.push_back(pho->showerShapeVariables().e2x5Top);
      phoE2x5Bottom_.push_back(pho->showerShapeVariables().e2x5Bottom);
      phoE2x5Left_.push_back(pho->showerShapeVariables().e2x5Left);
      phoE2x5Right_.push_back(pho->showerShapeVariables().e2x5Right);
      //phoSMMajor_.push_back(pho->showerShapeVariables().smMajor);
      //phoSMMinor_.push_back(pho->showerShapeVariables().smMinor);
      //phoSMAlpha_.push_back(pho->showerShapeVariables().smAlpha);

      phoSigmaIEtaIPhi_2012_.push_back(pho->full5x5_showerShapeVariables().sigmaIetaIphi);
      phoSigmaIPhiIPhi_2012_.push_back(pho->full5x5_showerShapeVariables().sigmaIphiIphi);
      phoR1x5_2012_.push_back(pho->full5x5_r1x5());
      phoR2x5_2012_.push_back(pho->full5x5_r2x5());
      phoE2nd_2012_.push_back(pho->full5x5_showerShapeVariables().e2nd);
      phoETop_2012_.push_back(pho->full5x5_showerShapeVariables().eTop);
      phoEBottom_2012_.push_back(pho->full5x5_showerShapeVariables().eBottom);
      phoELeft_2012_.push_back(pho->full5x5_showerShapeVariables().eLeft);
      phoERight_2012_.push_back(pho->full5x5_showerShapeVariables().eRight);
      phoE1x3_2012_.push_back(pho->full5x5_showerShapeVariables().e1x3);
      phoE2x2_2012_.push_back(pho->full5x5_showerShapeVariables().e2x2);
      phoE2x5Max_2012_.push_back(pho->full5x5_showerShapeVariables().e2x5Max);
      phoE2x5Top_2012_.push_back(pho->full5x5_showerShapeVariables().e2x5Top);
      phoE2x5Bottom_2012_.push_back(pho->full5x5_showerShapeVariables().e2x5Bottom);
      phoE2x5Left_2012_.push_back(pho->full5x5_showerShapeVariables().e2x5Left);
      phoE2x5Right_2012_.push_back(pho->full5x5_showerShapeVariables().e2x5Right);
      //phoSMMajor_2012_.push_back(pho->full5x5_showerShapeVariables().smMajor);
      //phoSMMinor_2012_.push_back(pho->full5x5_showerShapeVariables().smMinor);
      //phoSMAlpha_2012_.push_back(pho->full5x5_showerShapeVariables().smAlpha);
    }

    // full 5x5
    phoSigmaIEtaIEta_2012_.push_back(pho->full5x5_sigmaIetaIeta());
    phoR9_2012_.push_back(pho->full5x5_r9());
    phoE1x5_2012_.push_back(pho->full5x5_e1x5());
    phoE2x5_2012_.push_back(pho->full5x5_e2x5());
    phoE3x3_2012_.push_back(pho->full5x5_e3x3());
    phoE5x5_2012_.push_back(pho->full5x5_e5x5());
    phoMaxEnergyXtal_2012_.push_back(pho->full5x5_maxEnergyXtal());
    phoSigmaEtaEta_2012_.push_back(pho->full5x5_sigmaEtaEta());

    // seed BC
    if (pho->superCluster()->seed().isAvailable() && pho->superCluster()->seed().isNonnull()) {
      phoBC1E_.push_back(pho->superCluster()->seed()->energy());
      phoBC1Ecorr_.push_back(pho->superCluster()->seed()->correctedEnergy());
      phoBC1Eta_.push_back(pho->superCluster()->seed()->eta());
      phoBC1Phi_.push_back(pho->superCluster()->seed()->phi());
      phoBC1size_.push_back(pho->superCluster()->seed()->size());
      phoBC1flags_.push_back(pho->superCluster()->seed()->flags());
      phoBC1inClean_.push_back(pho->superCluster()->seed()->isInClean());
      phoBC1inUnClean_.push_back(pho->superCluster()->seed()->isInUnclean());
      phoBC1rawID_.push_back(pho->superCluster()->seed()->seed().rawId());
    } else {
      phoBC1E_.push_back(-999);
      phoBC1Ecorr_.push_back(-999);
      phoBC1Eta_.push_back(-999);
      phoBC1Phi_.push_back(-999);
      phoBC1size_.push_back(-999);
      phoBC1flags_.push_back(-999);
      phoBC1inClean_.push_back(-999);
      phoBC1inUnClean_.push_back(-999);
      phoBC1rawID_.push_back(0);
    }

    bool debugPhoUserDataNames = false;
    if (debugPhoUserDataNames) {
      std::cout << "debugging photon user data names" << std::endl;

      // https://github.com/cms-sw/cmssw/blob/master/DataFormats/PatCandidates/interface/PATObject.h#L334
      const std::vector<std::string> namesTmp = pho->userDataNames();
      int lenNames = namesTmp.size();
      std::cout << "There are " << lenNames << " names." << std::endl;
      for (int iName = 0; iName < lenNames; ++iName) {
        std::cout << "iName = " << iName << " , nameTmp = " << namesTmp[iName].c_str() << std::endl;
        std::cout << "typename = " << pho->userDataObjectType(namesTmp[iName]) << std::endl;
      }
    }

    if (useValMapIso_) {
      const reco::HIPhotonIsolation* hiIso = pho->userData<reco::HIPhotonIsolation>(std::string("photonIsolationHI"));

      pho_ecalClusterIsoR1_.push_back(hiIso->ecalClusterIsoR1());
      pho_ecalClusterIsoR2_.push_back(hiIso->ecalClusterIsoR2());
      pho_ecalClusterIsoR3_.push_back(hiIso->ecalClusterIsoR3());
      pho_ecalClusterIsoR4_.push_back(hiIso->ecalClusterIsoR4());
      pho_ecalClusterIsoR5_.push_back(hiIso->ecalClusterIsoR5());
      pho_hcalRechitIsoR1_.push_back(hiIso->hcalRechitIsoR1());
      pho_hcalRechitIsoR2_.push_back(hiIso->hcalRechitIsoR2());
      pho_hcalRechitIsoR3_.push_back(hiIso->hcalRechitIsoR3());
      pho_hcalRechitIsoR4_.push_back(hiIso->hcalRechitIsoR4());
      pho_hcalRechitIsoR5_.push_back(hiIso->hcalRechitIsoR5());
      pho_trackIsoR1PtCut20_.push_back(hiIso->trackIsoR1PtCut20());
      pho_trackIsoR2PtCut20_.push_back(hiIso->trackIsoR2PtCut20());
      pho_trackIsoR3PtCut20_.push_back(hiIso->trackIsoR3PtCut20());
      pho_trackIsoR4PtCut20_.push_back(hiIso->trackIsoR4PtCut20());
      pho_trackIsoR5PtCut20_.push_back(hiIso->trackIsoR5PtCut20());
      pho_swissCrx_.push_back(hiIso->swissCrx());
      pho_seedTime_.push_back(hiIso->seedTime());
    }

    if (doRecHitsEB_ || doRecHitsEE_) {
      int iBC = 0;
      for (reco::CaloCluster_iterator bc = pho->superCluster()->clustersBegin();
           bc != pho->superCluster()->clustersEnd();
           ++bc) {
        for (const auto& hitBC : (*bc)->hitsAndFractions()) {
          const DetId& rhDetId = hitBC.first;
          edm::Handle<EcalRecHitCollection>* recHitsHandle;

          if (rhDetId.subdetId() == EcalBarrel && doRecHitsEB_) {
            recHitsHandle = (&recHitsEBHandle);
          } else if (rhDetId.subdetId() == EcalEndcap && doRecHitsEE_) {
            recHitsHandle = (&recHitsEEHandle);
          } else
            continue;

          EcalRecHitCollection::const_iterator rhIter = (*recHitsHandle)->find(rhDetId);
          const EcalRecHit& rh = (*rhIter);

          rhRawId_.push_back(rhDetId.rawId());
          if (rhDetId.subdetId() == EcalBarrel) {
            rhieta_.push_back(EBDetId(rhDetId).ieta());
            rhiphi_.push_back(EBDetId(rhDetId).iphi());
            rhix_.push_back(-999);
            rhiy_.push_back(-999);
          } else if (rhDetId.subdetId() == EcalEndcap) {
            rhieta_.push_back(-999);
            rhiphi_.push_back(-999);
            rhix_.push_back(EEDetId(rhDetId).ix());
            rhiy_.push_back(EEDetId(rhDetId).iy());
          } else {
            rhieta_.push_back(-999);
            rhiphi_.push_back(-999);
            rhix_.push_back(-999);
            rhiy_.push_back(-999);
          }
          rhE_.push_back(rh.energy());

          math::XYZPoint posXYZ((geo->getPosition(rhDetId)).x() - pv.x(),
                                (geo->getPosition(rhDetId)).y() - pv.y(),
                                (geo->getPosition(rhDetId)).z() - pv.z());

          rhEt_.push_back(rh.energy() * sin(posXYZ.theta()));
          rhEta_.push_back(posXYZ.eta());
          rhPhi_.push_back(posXYZ.phi());
          rhChi2_.push_back(rh.chi2());
          rhEerror_.push_back(rh.energyError());

          uint32_t flagTmp = 0;
          for (unsigned int iFlag = 0; iFlag < 32; ++iFlag) {
            if (rh.checkFlag(iFlag))
              flagTmp |= 1 << iFlag;
          }
          rhFlags_.push_back(flagTmp);

          rhPhoIdx_.push_back(nPho_);
          rhBCIdx_.push_back(iBC);
          nRH_++;
        }
        iBC++;
      }
    }

    if (doPfIso_) {
      pfIsoCalculator pfIso;
      pfIso.setUsePackedCandidates(isPackedPFCandidate_);
      pfIso.setVertex(pv.position());
      if (isPackedPFCandidate_) {
        edm::Handle<edm::View<pat::PackedCandidate>> candidatesPacked;
        e.getByToken(packedPFCandToken_, candidatesPacked);
        pfIso.setCandidatesPacked(candidatesPacked);
      }
      /*
      else {
        edm::Handle<edm::View< reco::PFCandidate >> candidatesReco;
        e.getByToken(recoPFCandToken_, candidatesReco);
        pfIso.setCandidatesReco(candidatesReco);
      }
      */

      bool debug = false;
      if (debug) {
        // https://github.com/cms-sw/cmssw/blob/master/DataFormats/PatCandidates/interface/Photon.h#L326
        std::cout << "pho numberOfSourceCandidatePtrs = " << pho->numberOfSourceCandidatePtrs() << std::endl;
      }

      // particle flow isolation
      pfcIso1_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso2_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso3_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso4_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso5_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));

      pfpIso1_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso2_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso3_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso4_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso5_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));

      pfnIso1_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h0, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso2_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h0, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso3_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h0, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso4_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h0, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso5_.push_back(pfIso.getPfIso(*pho, reco::PFCandidate::h0, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));

      pfpIso1subSC_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso2subSC_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso3subSC_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso4subSC_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso5subSC_.push_back(
          pfIso.getPfIso(*pho, reco::PFCandidate::gamma, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));

      pfcIso1subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso2subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso3subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso4subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfcIso5subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));

      pfpIso1subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso2subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso3subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso4subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfpIso5subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));

      pfnIso1subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso2subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso3subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso4subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));
      pfnIso5subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removePFcand));

      pfpIso1subSCsubUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.1, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso2subSCsubUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso3subSCsubUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso4subSCsubUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));
      pfpIso5subSCsubUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.5, 0.0, 0.0, 0, pfIsoCalculator::removeSCenergy));

      pfcIso1pTgt1p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.1, 0.0, 1.0, 0, pfIsoCalculator::removePFcand));
      pfcIso2pTgt1p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.2, 0.0, 1.0, 0, pfIsoCalculator::removePFcand));
      pfcIso3pTgt1p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.3, 0.0, 1.0, 0, pfIsoCalculator::removePFcand));
      pfcIso4pTgt1p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.4, 0.0, 1.0, 0, pfIsoCalculator::removePFcand));
      pfcIso5pTgt1p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.5, 0.0, 1.0, 0, pfIsoCalculator::removePFcand));

      pfcIso1pTgt2p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.1, 0.0, 2.0, 0, pfIsoCalculator::removePFcand));
      pfcIso2pTgt2p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.2, 0.0, 2.0, 0, pfIsoCalculator::removePFcand));
      pfcIso3pTgt2p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.3, 0.0, 2.0, 0, pfIsoCalculator::removePFcand));
      pfcIso4pTgt2p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.4, 0.0, 2.0, 0, pfIsoCalculator::removePFcand));
      pfcIso5pTgt2p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.5, 0.0, 2.0, 0, pfIsoCalculator::removePFcand));

      pfcIso1pTgt3p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.1, 0.0, 3.0, 0, pfIsoCalculator::removePFcand));
      pfcIso2pTgt3p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.2, 0.0, 3.0, 0, pfIsoCalculator::removePFcand));
      pfcIso3pTgt3p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.3, 0.0, 3.0, 0, pfIsoCalculator::removePFcand));
      pfcIso4pTgt3p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.4, 0.0, 3.0, 0, pfIsoCalculator::removePFcand));
      pfcIso5pTgt3p0subUE_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.5, 0.0, 3.0, 0, pfIsoCalculator::removePFcand));

      pfcIso2subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));
      pfcIso3subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));
      pfcIso4subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));

      pfpIso2subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));
      pfpIso3subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));
      pfpIso4subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::gamma, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));

      pfnIso2subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.2, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));
      pfnIso3subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.3, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));
      pfnIso4subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h0, 0.4, 0.0, 0.0, 0, pfIsoCalculator::removePFcand, true));

      pfcIso2pTgt2p0subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.2, 0.0, 2.0, 0, pfIsoCalculator::removePFcand, true));
      pfcIso3pTgt2p0subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.3, 0.0, 2.0, 0, pfIsoCalculator::removePFcand, true));
      pfcIso4pTgt2p0subUEec_.push_back(
          pfIso.getPfIsoSubUE(*pho, reco::PFCandidate::h, 0.4, 0.0, 2.0, 0, pfIsoCalculator::removePFcand, true));
    }

    /////////////////////////////// MC matching //////////////////////////
    if (doGenParticles_) {
      constexpr float delta2 = 0.15 * 0.15;

      bool gpTemp = false;
      float currentMaxPt = -1;
      int matchedIndex = -1;

      for (unsigned igen = 0; igen < mcEt_.size(); ++igen) {
        if (mcStatus_[igen] != 1 || mcPID_[igen] != 22)
          continue;
        if (reco::deltaR2(pho->eta(), pho->phi(), mcEta_[igen], mcPhi_[igen]) < delta2 && mcPt_[igen] > currentMaxPt) {
          gpTemp = true;
          currentMaxPt = mcPt_[igen];
          matchedIndex = igen;
        }
      }

      // if no matching photon was found try with other particles
      std::vector<int> otherPdgIds_ = {1, 11};
      if (!gpTemp) {
        currentMaxPt = -1;
        for (unsigned igen = 0; igen < mcEt_.size(); ++igen) {
          if (mcStatus_[igen] != 1 ||
              find(otherPdgIds_.begin(), otherPdgIds_.end(), std::abs(mcPID_[igen])) == otherPdgIds_.end())
            continue;
          if (reco::deltaR2(pho->eta(), pho->phi(), mcEta_[igen], mcPhi_[igen]) < delta2 &&
              mcPt_[igen] > currentMaxPt) {
            gpTemp = true;
            currentMaxPt = mcPt_[igen];
            matchedIndex = igen;
          }
        }
      }

      pho_genMatchedIndex_.push_back(matchedIndex);
    }

    nPho_++;
  }  // photons loop
}

void ggHiNtuplizer::fillMuons(const edm::Event& e, const edm::EventSetup& es, reco::Vertex& pv) {
  // Fills tree branches with muons.
  edm::Handle<edm::View<pat::Muon>> recoMuons;
  e.getByToken(muonsToken_, recoMuons);

  for (const auto& mu : *recoMuons) {
    if (mu.pt() < muonPtMin_)
      continue;
    if (!(mu.isPFMuon() || mu.isGlobalMuon() || mu.isTrackerMuon()))
      continue;

    muPt_.push_back(mu.pt());
    muEta_.push_back(mu.eta());
    muPhi_.push_back(mu.phi());
    muL1Eta_.push_back(mu.hasUserFloat("l1Eta") ? mu.userFloat("l1Eta") : -99);
    muL1Phi_.push_back(mu.hasUserFloat("l1Phi") ? mu.userFloat("l1Phi") : -99);
    muMVAIso_.push_back(mu.hasUserFloat("hiMVAIso") ? mu.userFloat("hiMVAIso") : -99);
    for (auto& w : muMVAIsoWP_)
      w.second.push_back(mu.hasUserInt("hiMVAIso"+w.first) ? mu.userInt("hiMVAIso"+w.first) : -1);
    muCharge_.push_back(mu.charge());
    muType_.push_back(mu.type());
    muIsGood_.push_back(muon::isGoodMuon(mu, muon::selectionTypeFromString("TMOneStationTight")));

    muIsGlobal_.push_back((int)mu.isGlobalMuon());
    muIsTracker_.push_back((int)mu.isTrackerMuon());
    muIsPF_.push_back((int)mu.isPFMuon());
    muIsSTA_.push_back((int)mu.isStandAloneMuon());

    muD0_.push_back(mu.muonBestTrack()->dxy(pv.position()));
    muDz_.push_back(mu.muonBestTrack()->dz(pv.position()));
    muD0Err_.push_back(mu.muonBestTrack()->dxyError());
    muDzErr_.push_back(mu.muonBestTrack()->dzError());

    // initialize with unphysical values
    float muIP3D = -999;
    float muIP3DErr = -999;
    if (pv.isValid()) {
      // 3D impact parameter
      reco::TransientTrack tt = tb->build(mu.muonBestTrack().get());
      muIP3D = IPTools::absoluteImpactParameter3D(tt, pv).second.value();
      muIP3DErr = IPTools::absoluteImpactParameter3D(tt, pv).second.error();
    }
    muIP3D_.push_back(muIP3D);
    muIP3DErr_.push_back(muIP3DErr);

    const reco::TrackRef glbMu = mu.globalTrack();
    const reco::TrackRef innMu = mu.innerTrack();

    if (glbMu.isNull()) {
      muChi2NDF_.push_back(-99);
      muMuonHits_.push_back(-99);
    } else {
      muChi2NDF_.push_back(glbMu->normalizedChi2());
      muMuonHits_.push_back(glbMu->hitPattern().numberOfValidMuonHits());
    }

    if (innMu.isNull()) {
      muInnerD0_.push_back(-99);
      muInnerDz_.push_back(-99);

      muInnerD0Err_.push_back(-99);
      muInnerDzErr_.push_back(-99);
      muInnerPt_.push_back(-99);
      muInnerPtErr_.push_back(-99);
      muInnerEta_.push_back(-99);

      muTrkLayers_.push_back(-99);
      muPixelLayers_.push_back(-99);
      muPixelHits_.push_back(-99);
      muTrkQuality_.push_back(-99);
    } else {
      muInnerD0_.push_back(innMu->dxy(pv.position()));
      muInnerDz_.push_back(innMu->dz(pv.position()));

      muInnerD0Err_.push_back(innMu->dxyError());
      muInnerDzErr_.push_back(innMu->dzError());
      muInnerPt_.push_back(innMu->pt());
      muInnerPtErr_.push_back(innMu->ptError());
      muInnerEta_.push_back(innMu->eta());

      muTrkLayers_.push_back(innMu->hitPattern().trackerLayersWithMeasurement());
      muPixelLayers_.push_back(innMu->hitPattern().pixelLayersWithMeasurement());
      muPixelHits_.push_back(innMu->hitPattern().numberOfValidPixelHits());
      muTrkQuality_.push_back(innMu->quality(reco::TrackBase::highPurity));
    }

    muStations_.push_back(mu.numberOfMatchedStations());
    muIsoTrk_.push_back(mu.isolationR03().sumPt);
    muPFChIso_.push_back(mu.pfIsolationR04().sumChargedHadronPt);
    muPFPhoIso_.push_back(mu.pfIsolationR04().sumPhotonEt);
    muPFNeuIso_.push_back(mu.pfIsolationR04().sumNeutralHadronEt);
    muPFPUIso_.push_back(mu.pfIsolationR04().sumPUPt);

    muSelectors_.push_back(mu.selectors());
    muIDSoft_.push_back(mu.passed(reco::Muon::SoftMvaId));
    muIDLoose_.push_back(mu.passed(reco::Muon::CutBasedIdLoose));
    muIDMedium_.push_back(mu.passed(reco::Muon::CutBasedIdMedium));
    muIDMediumPrompt_.push_back(mu.passed(reco::Muon::CutBasedIdMediumPrompt));
    muIDTight_.push_back(mu.passed(reco::Muon::CutBasedIdTight));
    muIDGlobalHighPt_.push_back(mu.passed(reco::Muon::CutBasedIdGlobalHighPt));
    muIDTrkHighPt_.push_back(mu.passed(reco::Muon::CutBasedIdTrkHighPt));
    muIDInTime_.push_back(mu.passed(reco::Muon::InTimeMuon));

    /////////////////////////////// MC matching //////////////////////////
    if (doGenParticles_) {
      float minDeltaR2(0.0225);
      int matchedIndex = -1;
      for (size_t igen = 0; igen < mcEt_.size(); ++igen) {
        if (mcStatus_[igen] != 1 || mcPID_[igen] != -13*mu.charge())
          continue;
        const auto deltaR2 = reco::deltaR2(mu.eta(), mu.phi(), mcEta_[igen], mcPhi_[igen]);
        if (deltaR2 < minDeltaR2 && std::abs(mcPt_[igen] - mu.pt()) < mcPt_[igen]) {
          minDeltaR2 = deltaR2;
          matchedIndex = igen;
        }
      }
      mu_genMatchedIndex_.push_back(matchedIndex);
    }

    nMu_++;
  }  // muons loop
}

DEFINE_FWK_MODULE(ggHiNtuplizer);
