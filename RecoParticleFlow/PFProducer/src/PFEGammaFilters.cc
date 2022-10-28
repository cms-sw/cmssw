//
// Original Authors: Nicholas Wardle, Florian Beaudette
//
#include "RecoParticleFlow/PFProducer/interface/PFEGammaFilters.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackAlgoTools.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

using namespace std;
using namespace reco;

namespace {

  // Constants defining the ECAL barrel limit
  constexpr float ecalBarrelMaxEtaWithGap = 1.566;
  constexpr float ecalBarrelMaxEtaNoGap = 1.485;

  void readEBEEParams_(const edm::ParameterSet& pset, const std::string& name, std::array<float, 2>& out) {
    const auto& vals = pset.getParameter<std::vector<double>>(name);
    if (vals.size() != 2)
      throw cms::Exception("Configuration") << "Parameter " << name << " does not contain exactly 2 values (EB, EE)\n";
    out[0] = vals[0];
    out[1] = vals[1];
  }

}  // namespace

PFEGammaFilters::PFEGammaFilters(const edm::ParameterSet& cfg)
    : ph_Et_(cfg.getParameter<double>("photon_MinEt")),
      ph_combIso_(cfg.getParameter<double>("photon_combIso")),
      ph_loose_hoe_(cfg.getParameter<double>("photon_HoE")),
      ph_sietaieta_eb_(cfg.getParameter<double>("photon_SigmaiEtaiEta_barrel")),
      ph_sietaieta_ee_(cfg.getParameter<double>("photon_SigmaiEtaiEta_endcap")),
      useElePFidDNN_(cfg.getParameter<bool>("useElePFidDnn")),
      usePhotonPFidDNN_(cfg.getParameter<bool>("usePhotonPFidDnn")),
      useEBModelInGap_(cfg.getParameter<bool>("useEBModelInGap")),
      endcapBoundary_(cfg.getParameter<double>("endcapBoundary")),
      extEtaBoundary_(cfg.getParameter<double>("extEtaBoundary")),
      ele_iso_pt_(cfg.getParameter<double>("electron_iso_pt")),
      ele_iso_mva_eb_(cfg.getParameter<double>("electron_iso_mva_barrel")),
      ele_iso_mva_ee_(cfg.getParameter<double>("electron_iso_mva_endcap")),
      ele_iso_combIso_eb_(cfg.getParameter<double>("electron_iso_combIso_barrel")),
      ele_iso_combIso_ee_(cfg.getParameter<double>("electron_iso_combIso_endcap")),
      ele_noniso_mva_(cfg.getParameter<double>("electron_noniso_mvaCut")),
      ele_missinghits_(cfg.getParameter<unsigned int>("electron_missinghits")),
      ele_ecalDrivenHademPreselCut_(cfg.getParameter<double>("electron_ecalDrivenHademPreselCut")),
      ele_maxElePtForOnlyMVAPresel_(cfg.getParameter<double>("electron_maxElePtForOnlyMVAPresel")) {
  auto const& eleProtectionsForBadHcal = cfg.getParameter<edm::ParameterSet>("electron_protectionsForBadHcal");
  auto const& eleProtectionsForJetMET = cfg.getParameter<edm::ParameterSet>("electron_protectionsForJetMET");
  auto const& phoProtectionsForBadHcal = cfg.getParameter<edm::ParameterSet>("photon_protectionsForBadHcal");
  auto const& phoProtectionsForJetMET = cfg.getParameter<edm::ParameterSet>("photon_protectionsForJetMET");
  auto const& eleDNNIdThresholds = cfg.getParameter<edm::ParameterSet>("electronDnnThresholds");
  auto const& eleDNNBkgIdThresholds = cfg.getParameter<edm::ParameterSet>("electronDnnBkgThresholds");
  auto const& photonDNNIdThresholds = cfg.getParameter<edm::ParameterSet>("photonDnnThresholds");

  pho_sumPtTrackIso_ = phoProtectionsForJetMET.getParameter<double>("sumPtTrackIso");
  pho_sumPtTrackIsoSlope_ = phoProtectionsForJetMET.getParameter<double>("sumPtTrackIsoSlope");

  ele_maxNtracks_ = eleProtectionsForJetMET.getParameter<double>("maxNtracks");
  ele_maxHcalE_ = eleProtectionsForJetMET.getParameter<double>("maxHcalE");
  ele_maxTrackPOverEele_ = eleProtectionsForJetMET.getParameter<double>("maxTrackPOverEele");
  ele_maxE_ = eleProtectionsForJetMET.getParameter<double>("maxE");
  ele_maxEleHcalEOverEcalE_ = eleProtectionsForJetMET.getParameter<double>("maxEleHcalEOverEcalE");
  ele_maxEcalEOverPRes_ = eleProtectionsForJetMET.getParameter<double>("maxEcalEOverPRes");
  ele_maxEeleOverPoutRes_ = eleProtectionsForJetMET.getParameter<double>("maxEeleOverPoutRes");
  ele_maxHcalEOverP_ = eleProtectionsForJetMET.getParameter<double>("maxHcalEOverP");
  ele_maxHcalEOverEcalE_ = eleProtectionsForJetMET.getParameter<double>("maxHcalEOverEcalE");
  ele_maxEcalEOverP_1_ = eleProtectionsForJetMET.getParameter<double>("maxEcalEOverP_1");
  ele_maxEcalEOverP_2_ = eleProtectionsForJetMET.getParameter<double>("maxEcalEOverP_2");
  ele_maxEeleOverPout_ = eleProtectionsForJetMET.getParameter<double>("maxEeleOverPout");
  ele_maxDPhiIN_ = eleProtectionsForJetMET.getParameter<double>("maxDPhiIN");

  ele_dnnLowPtThr_ = eleDNNIdThresholds.getParameter<double>("electronDnnLowPtThr");
  ele_dnnHighPtBarrelThr_ = eleDNNIdThresholds.getParameter<double>("electronDnnHighPtBarrelThr");
  ele_dnnHighPtEndcapThr_ = eleDNNIdThresholds.getParameter<double>("electronDnnHighPtEndcapThr");
  ele_dnnExtEta1Thr_ = eleDNNIdThresholds.getParameter<double>("electronDnnExtEta1Thr");
  ele_dnnExtEta2Thr_ = eleDNNIdThresholds.getParameter<double>("electronDnnExtEta2Thr");

  ele_dnnBkgLowPtThr_ = eleDNNBkgIdThresholds.getParameter<double>("electronDnnBkgLowPtThr");
  ele_dnnBkgHighPtBarrelThr_ = eleDNNBkgIdThresholds.getParameter<double>("electronDnnBkgHighPtBarrelThr");
  ele_dnnBkgHighPtEndcapThr_ = eleDNNBkgIdThresholds.getParameter<double>("electronDnnBkgHighPtEndcapThr");
  ele_dnnBkgExtEta1Thr_ = eleDNNBkgIdThresholds.getParameter<double>("electronDnnBkgExtEta1Thr");
  ele_dnnBkgExtEta2Thr_ = eleDNNBkgIdThresholds.getParameter<double>("electronDnnBkgExtEta2Thr");

  photon_dnnBarrelThr_ = photonDNNIdThresholds.getParameter<double>("photonDnnBarrelThr");
  photon_dnnEndcapThr_ = photonDNNIdThresholds.getParameter<double>("photonDnnEndcapThr");

  readEBEEParams_(eleProtectionsForBadHcal, "full5x5_sigmaIetaIeta", badHcal_full5x5_sigmaIetaIeta_);
  readEBEEParams_(eleProtectionsForBadHcal, "eInvPInv", badHcal_eInvPInv_);
  readEBEEParams_(eleProtectionsForBadHcal, "dEta", badHcal_dEta_);
  readEBEEParams_(eleProtectionsForBadHcal, "dPhi", badHcal_dPhi_);
  badHcal_eleEnable_ = eleProtectionsForBadHcal.getParameter<bool>("enableProtections");

  badHcal_phoTrkSolidConeIso_offs_ = phoProtectionsForBadHcal.getParameter<double>("solidConeTrkIsoOffset");
  badHcal_phoTrkSolidConeIso_slope_ = phoProtectionsForBadHcal.getParameter<double>("solidConeTrkIsoSlope");
  badHcal_phoEnable_ = phoProtectionsForBadHcal.getParameter<bool>("enableProtections");
}

bool PFEGammaFilters::passPhotonSelection(const reco::Photon& photon) const {
  // First simple selection, same as the Run1 to be improved in CMSSW_710

  // Photon ET
  if (photon.pt() < ph_Et_)
    return false;
  bool validHoverE = photon.hadTowOverEmValid();
  if (debug_)
    std::cout << "PFEGammaFilters:: photon pt " << photon.pt() << "   eta, phi " << photon.eta() << ", " << photon.phi()
              << "   isoDr03 "
              << (photon.trkSumPtHollowConeDR03() + photon.ecalRecHitSumEtConeDR03() + photon.hcalTowerSumEtConeDR03())
              << " (cut: " << ph_combIso_ << ")"
              << "   H/E " << photon.hadTowOverEm() << " (valid? " << validHoverE << ", cut: " << ph_loose_hoe_ << ")"
              << "   s(ieie) " << photon.sigmaIetaIeta()
              << " (cut: " << (photon.isEB() ? ph_sietaieta_eb_ : ph_sietaieta_ee_) << ")"
              << "   isoTrkDr03Solid " << (photon.trkSumPtSolidConeDR03()) << " (cut: "
              << (validHoverE || !badHcal_phoEnable_
                      ? -1
                      : badHcal_phoTrkSolidConeIso_offs_ + badHcal_phoTrkSolidConeIso_slope_ * photon.pt())
              << ")" << std::endl;

  if (usePhotonPFidDNN_) {
    // Run3 DNN based PFID
    const auto dnn = photon.pfDNN();
    const auto photEta = std::abs(photon.eta());
    const auto etaThreshold = (useEBModelInGap_) ? ecalBarrelMaxEtaWithGap : ecalBarrelMaxEtaNoGap;
    // using the Barrel model for photons in the EB-EE gap
    if (photEta <= etaThreshold) {
      return dnn > photon_dnnBarrelThr_;
    } else if (photEta > etaThreshold) {
      return dnn > photon_dnnEndcapThr_;
    }
  } else {
    // Run2 cut based PFID
    if (photon.hadTowOverEm() > ph_loose_hoe_)
      return false;
    //Isolation variables in 0.3 cone combined
    if (photon.trkSumPtHollowConeDR03() + photon.ecalRecHitSumEtConeDR03() + photon.hcalTowerSumEtConeDR03() >
        ph_combIso_)
      return false;

    //patch for bad hcal
    if (!validHoverE && badHcal_phoEnable_ &&
        photon.trkSumPtSolidConeDR03() >
            badHcal_phoTrkSolidConeIso_offs_ + badHcal_phoTrkSolidConeIso_slope_ * photon.pt()) {
      return false;
    }

    if (photon.isEB()) {
      if (photon.sigmaIetaIeta() > ph_sietaieta_eb_)
        return false;
    } else {
      if (photon.sigmaIetaIeta() > ph_sietaieta_ee_)
        return false;
    }
  }

  return true;
}

bool PFEGammaFilters::passElectronSelection(const reco::GsfElectron& electron,
                                            const reco::PFCandidate& pfcand,
                                            const int& nVtx) const {
  // First simple selection, same as the Run1 to be improved in CMSSW_710

  bool validHoverE = electron.hcalOverEcalValid();
  if (debug_)
    std::cout << "PFEGammaFilters:: Electron pt " << electron.pt() << " eta, phi " << electron.eta() << ", "
              << electron.phi() << " charge " << electron.charge() << " isoDr03 "
              << (electron.dr03TkSumPt() + electron.dr03EcalRecHitSumEt() + electron.dr03HcalTowerSumEt())
              << " mva_isolated " << electron.mva_Isolated() << " mva_e_pi " << electron.mva_e_pi() << " H/E_valid "
              << validHoverE << " s(ieie) " << electron.full5x5_sigmaIetaIeta() << " H/E " << electron.hcalOverEcal()
              << " 1/e-1/p " << (1.0 - electron.eSuperClusterOverP()) / electron.ecalEnergy() << " deta "
              << std::abs(electron.deltaEtaSeedClusterTrackAtVtx()) << " dphi "
              << std::abs(electron.deltaPhiSuperClusterTrackAtVtx()) << endl;

  bool passEleSelection = false;

  // Electron ET
  const auto electronPt = electron.pt();
  const auto eleEta = std::abs(electron.eta());

  if (useElePFidDNN_) {  // Use DNN for ele pfID >=CMSSW12_1
    const auto dnn_sig = electron.dnn_signal_Isolated() + electron.dnn_signal_nonIsolated();
    const auto dnn_bkg = electron.dnn_bkg_nonIsolated();
    const auto etaThreshold = (useEBModelInGap_) ? ecalBarrelMaxEtaWithGap : ecalBarrelMaxEtaNoGap;
    if (eleEta < endcapBoundary_) {
      if (electronPt > ele_iso_pt_) {
        // using the Barrel model for electron in the EB-EE gap
        if (eleEta <= etaThreshold) {  //high pT barrel
          passEleSelection = (dnn_sig > ele_dnnHighPtBarrelThr_) && (dnn_bkg < ele_dnnBkgHighPtBarrelThr_);
        } else if (eleEta > etaThreshold) {  //high pT endcap (eleEta < 2.5)
          passEleSelection = (dnn_sig > ele_dnnHighPtEndcapThr_) && (dnn_bkg < ele_dnnBkgHighPtEndcapThr_);
        }
      } else {  // pt < ele_iso_pt_ (eleEta < 2.5)
        passEleSelection = (dnn_sig > ele_dnnLowPtThr_) && (dnn_bkg < ele_dnnBkgLowPtThr_);
      }
    } else if ((eleEta >= endcapBoundary_) && (eleEta <= extEtaBoundary_)) {  //First region in extended eta
      passEleSelection = (dnn_sig > ele_dnnExtEta1Thr_) && (dnn_bkg < ele_dnnBkgExtEta1Thr_);
    } else if (eleEta > extEtaBoundary_) {  //Second region in extended eta
      passEleSelection = (dnn_sig > ele_dnnExtEta2Thr_) && (dnn_bkg < ele_dnnBkgExtEta2Thr_);
    }
    // TODO: For the moment do not evaluate further conditions on isolation and HCAL cleaning..
    // To be understood if they are needed
  } else {  // Use legacy MVA for ele pfID < CMSSW_12_1
    if (electronPt > ele_iso_pt_) {
      double isoDr03 = electron.dr03TkSumPt() + electron.dr03EcalRecHitSumEt() + electron.dr03HcalTowerSumEt();
      if (eleEta <= ecalBarrelMaxEtaNoGap && isoDr03 < ele_iso_combIso_eb_) {
        if (electron.mva_Isolated() > ele_iso_mva_eb_)
          passEleSelection = true;
      } else if (eleEta > ecalBarrelMaxEtaNoGap && isoDr03 < ele_iso_combIso_ee_) {
        if (electron.mva_Isolated() > ele_iso_mva_ee_)
          passEleSelection = true;
      }
    }

    if (electron.mva_e_pi() > ele_noniso_mva_) {
      if (validHoverE || !badHcal_eleEnable_) {
        passEleSelection = true;
      } else {
        bool EE = (std::abs(electron.eta()) >
                   ecalBarrelMaxEtaNoGap);  // for prefer consistency with above than with E/gamma for now
        if ((electron.full5x5_sigmaIetaIeta() < badHcal_full5x5_sigmaIetaIeta_[EE]) &&
            (std::abs(1.0 - electron.eSuperClusterOverP()) / electron.ecalEnergy() < badHcal_eInvPInv_[EE]) &&
            (std::abs(electron.deltaEtaSeedClusterTrackAtVtx()) <
             badHcal_dEta_[EE]) &&  // looser in case of misalignment
            (std::abs(electron.deltaPhiSuperClusterTrackAtVtx()) < badHcal_dPhi_[EE])) {
          passEleSelection = true;
        }
      }
    }
  }
  return passEleSelection && passGsfElePreSelWithOnlyConeHadem(electron);
}

bool PFEGammaFilters::isElectron(const reco::GsfElectron& electron) const {
  return electron.gsfTrack()->missingInnerHits() <= ele_missinghits_;
}

bool PFEGammaFilters::isElectronSafeForJetMET(const reco::GsfElectron& electron,
                                              const reco::PFCandidate& pfcand,
                                              const reco::Vertex& primaryVertex,
                                              bool& lockTracks) const {
  bool debugSafeForJetMET = false;
  bool isSafeForJetMET = true;

  //   cout << " ele_maxNtracks_ " <<  ele_maxNtracks_ << endl
  //        << " ele_maxHcalE_ " << ele_maxHcalE_ << endl
  //        << " ele_maxTrackPOverEele_ " << ele_maxTrackPOverEele_ << endl
  //        << " ele_maxE_ " << ele_maxE_ << endl
  //        << " ele_maxEleHcalEOverEcalE_ "<< ele_maxEleHcalEOverEcalE_ << endl
  //        << " ele_maxEcalEOverPRes_ " << ele_maxEcalEOverPRes_ << endl
  //        << " ele_maxEeleOverPoutRes_ "  << ele_maxEeleOverPoutRes_ << endl
  //        << " ele_maxHcalEOverP_ " << ele_maxHcalEOverP_ << endl
  //        << " ele_maxHcalEOverEcalE_ " << ele_maxHcalEOverEcalE_ << endl
  //        << " ele_maxEcalEOverP_1_ " << ele_maxEcalEOverP_1_ << endl
  //        << " ele_maxEcalEOverP_2_ " << ele_maxEcalEOverP_2_ << endl
  //        << " ele_maxEeleOverPout_ "  << ele_maxEeleOverPout_ << endl
  //        << " ele_maxDPhiIN_ " << ele_maxDPhiIN_ << endl;

  // loop on the extra-tracks associated to the electron
  PFCandidateEGammaExtraRef pfcandextra = pfcand.egammaExtraRef();

  unsigned int iextratrack = 0;
  unsigned int itrackHcalLinked = 0;
  float SumExtraKfP = 0.;
  //float Ene_ecalgsf = 0.;

  // problems here: for now get the electron cluster from the gsf electron
  //  const PFCandidate::ElementsInBlocks& eleCluster = pfcandextra->gsfElectronClusterRef();
  // PFCandidate::ElementsInBlocks::const_iterator iegfirst = eleCluster.begin();
  //  float Ene_hcalgsf = pfcandextra->

  float ETtotal = electron.ecalEnergy();

  //NOTE take this from EGammaExtra
  float Ene_ecalgsf = electron.electronCluster()->energy();
  float Ene_hcalgsf = pfcandextra->hadEnergy();
  float HOverHE = Ene_hcalgsf / (Ene_hcalgsf + Ene_ecalgsf);
  float EtotPinMode = electron.eSuperClusterOverP();

  //NOTE take this from EGammaExtra
  float EGsfPoutMode = electron.eEleClusterOverPout();
  float HOverPin = Ene_hcalgsf / electron.gsfTrack()->pMode();
  float dphi_normalsc = electron.deltaPhiSuperClusterTrackAtVtx();

  const PFCandidate::ElementsInBlocks& extraTracks = pfcandextra->extraNonConvTracks();
  for (PFCandidate::ElementsInBlocks::const_iterator itrk = extraTracks.begin(); itrk < extraTracks.end(); ++itrk) {
    const PFBlock& block = *(itrk->first);
    const PFBlock::LinkData& linkData = block.linkData();
    const PFBlockElement& pfele = block.elements()[itrk->second];

    if (debugSafeForJetMET)
      cout << " My track element number " << itrk->second << endl;
    if (pfele.type() == reco::PFBlockElement::TRACK) {
      const reco::TrackRef& trackref = pfele.trackRef();

      bool goodTrack = PFTrackAlgoTools::isGoodForEGM(trackref->algo());
      // iter0, iter1, iter2, iter3 = Algo < 3
      // algo 4,5,6,7

      bool trackIsFromPrimaryVertex = false;
      for (Vertex::trackRef_iterator trackIt = primaryVertex.tracks_begin(); trackIt != primaryVertex.tracks_end();
           ++trackIt) {
        if ((*trackIt).castTo<TrackRef>() == trackref) {
          trackIsFromPrimaryVertex = true;
          break;
        }
      }

      // probably we could now remove the algo request??
      if (goodTrack && trackref->missingInnerHits() == 0 && trackIsFromPrimaryVertex) {
        float p_trk = trackref->p();
        SumExtraKfP += p_trk;
        iextratrack++;
        // Check if these extra tracks are HCAL linked
        std::multimap<double, unsigned int> hcalKfElems;
        block.associatedElements(
            itrk->second, linkData, hcalKfElems, reco::PFBlockElement::HCAL, reco::PFBlock::LINKTEST_ALL);
        if (!hcalKfElems.empty()) {
          itrackHcalLinked++;
        }
        if (debugSafeForJetMET)
          cout << " The ecalGsf cluster is not isolated: >0 KF extra with algo < 3"
               << " Algo " << trackref->algo() << " trackref->missingInnerHits() " << trackref->missingInnerHits()
               << " trackIsFromPrimaryVertex " << trackIsFromPrimaryVertex << endl;
        if (debugSafeForJetMET)
          cout << " My track PT " << trackref->pt() << endl;

      } else {
        if (debugSafeForJetMET)
          cout << " Tracks from PU "
               << " Algo " << trackref->algo() << " trackref->missingInnerHits() " << trackref->missingInnerHits()
               << " trackIsFromPrimaryVertex " << trackIsFromPrimaryVertex << endl;
        if (debugSafeForJetMET)
          cout << " My track PT " << trackref->pt() << endl;
      }
    }
  }
  if (iextratrack > 0) {
    if (iextratrack > ele_maxNtracks_ || Ene_hcalgsf > ele_maxHcalE_ ||
        (SumExtraKfP / Ene_ecalgsf) > ele_maxTrackPOverEele_ ||
        (ETtotal > ele_maxE_ && iextratrack > 1 && (Ene_hcalgsf / Ene_ecalgsf) > ele_maxEleHcalEOverEcalE_)) {
      if (debugSafeForJetMET)
        cout << " *****This electron candidate is discarded: Non isolated  # tracks " << iextratrack << " HOverHE "
             << HOverHE << " SumExtraKfP/Ene_ecalgsf " << SumExtraKfP / Ene_ecalgsf << " SumExtraKfP " << SumExtraKfP
             << " Ene_ecalgsf " << Ene_ecalgsf << " ETtotal " << ETtotal << " Ene_hcalgsf/Ene_ecalgsf "
             << Ene_hcalgsf / Ene_ecalgsf << endl;

      isSafeForJetMET = false;
    }
    // the electron is retained and the kf tracks are not locked
    if ((std::abs(1. - EtotPinMode) < ele_maxEcalEOverPRes_ &&
         (std::abs(electron.eta()) < 1.0 || std::abs(electron.eta()) > 2.0)) ||
        ((EtotPinMode < 1.1 && EtotPinMode > 0.6) &&
         (std::abs(electron.eta()) >= 1.0 && std::abs(electron.eta()) <= 2.0))) {
      if (std::abs(1. - EGsfPoutMode) < ele_maxEeleOverPoutRes_ && (itrackHcalLinked == iextratrack)) {
        lockTracks = false;
        //	lockExtraKf = false;
        if (debugSafeForJetMET)
          cout << " *****This electron is reactivated  # tracks " << iextratrack << " #tracks hcal linked "
               << itrackHcalLinked << " SumExtraKfP/Ene_ecalgsf " << SumExtraKfP / Ene_ecalgsf << " EtotPinMode "
               << EtotPinMode << " EGsfPoutMode " << EGsfPoutMode << " eta gsf " << electron.eta() << endl;
      }
    }
  }

  if (HOverPin > ele_maxHcalEOverP_ && HOverHE > ele_maxHcalEOverEcalE_ && EtotPinMode < ele_maxEcalEOverP_1_) {
    if (debugSafeForJetMET)
      cout << " *****This electron candidate is discarded  HCAL ENERGY "
           << " HOverPin " << HOverPin << " HOverHE " << HOverHE << " EtotPinMode" << EtotPinMode << endl;
    isSafeForJetMET = false;
  }

  // Reject Crazy E/p values... to be understood in the future how to train a
  // BDT in order to avoid to select this bad electron candidates.

  if (EtotPinMode < ele_maxEcalEOverP_2_ && EGsfPoutMode < ele_maxEeleOverPout_) {
    if (debugSafeForJetMET)
      cout << " *****This electron candidate is discarded  Low ETOTPIN "
           << " EtotPinMode " << EtotPinMode << " EGsfPoutMode " << EGsfPoutMode << endl;
    isSafeForJetMET = false;
  }

  // For not-preselected Gsf Tracks ET > 50 GeV, apply dphi preselection
  if (ETtotal > ele_maxE_ && std::abs(dphi_normalsc) > ele_maxDPhiIN_) {
    if (debugSafeForJetMET)
      cout << " *****This electron candidate is discarded  Large ANGLE "
           << " ETtotal " << ETtotal << " EGsfPoutMode " << dphi_normalsc << endl;
    isSafeForJetMET = false;
  }

  return isSafeForJetMET;
}
bool PFEGammaFilters::isPhotonSafeForJetMET(const reco::Photon& photon, const reco::PFCandidate& pfcand) const {
  bool isSafeForJetMET = true;
  bool debugSafeForJetMET = false;

  //   cout << " pho_sumPtTrackIso_ForPhoton " << pho_sumPtTrackIso_
  //        << " pho_sumPtTrackIsoSlope_ForPhoton " << pho_sumPtTrackIsoSlope_ <<  endl;

  float sum_track_pt = 0.;

  PFCandidateEGammaExtraRef pfcandextra = pfcand.egammaExtraRef();
  const PFCandidate::ElementsInBlocks& extraTracks = pfcandextra->extraNonConvTracks();
  for (PFCandidate::ElementsInBlocks::const_iterator itrk = extraTracks.begin(); itrk < extraTracks.end(); ++itrk) {
    const PFBlock& block = *(itrk->first);
    const PFBlockElement& pfele = block.elements()[itrk->second];

    if (pfele.type() == reco::PFBlockElement::TRACK) {
      const reco::TrackRef& trackref = pfele.trackRef();

      if (debugSafeForJetMET)
        cout << "PFEGammaFilters::isPhotonSafeForJetMET photon track:pt " << trackref->pt() << " SingleLegSize "
             << pfcandextra->singleLegConvTrackRefMva().size() << endl;

      //const std::vector<reco::TrackRef>&  mySingleLeg =
      bool singleLegConv = false;
      for (unsigned int iconv = 0; iconv < pfcandextra->singleLegConvTrackRefMva().size(); iconv++) {
        if (debugSafeForJetMET)
          cout << "PFEGammaFilters::SingleLeg track:pt " << (pfcandextra->singleLegConvTrackRefMva()[iconv].first)->pt()
               << endl;

        if (pfcandextra->singleLegConvTrackRefMva()[iconv].first == trackref) {
          singleLegConv = true;
          if (debugSafeForJetMET)
            cout << "PFEGammaFilters::isPhotonSafeForJetMET: SingleLeg conv track " << endl;
          break;
        }
      }
      if (singleLegConv)
        continue;

      sum_track_pt += trackref->pt();
    }
  }

  if (debugSafeForJetMET)
    cout << " PFEGammaFilters::isPhotonSafeForJetMET: SumPt " << sum_track_pt << endl;

  if (sum_track_pt > (pho_sumPtTrackIso_ + pho_sumPtTrackIsoSlope_ * photon.pt())) {
    isSafeForJetMET = false;
    if (debugSafeForJetMET)
      cout << "************************************!!!! PFEGammaFilters::isPhotonSafeForJetMET: Photon Discaded !!! "
           << endl;
  }

  return isSafeForJetMET;
}

//in CMSSW_10_4_0 we changed the electron preselection to be  H/E(cone 0.15) < 0.15
//OR H/E(single tower) < 0.15, with the tower being new.
//However CMS is scared of making any change to the PF content and therefore
//we have to explicitly reject them here
//has to be insync here with GsfElectronAlgo::isPreselected
bool PFEGammaFilters::passGsfElePreSelWithOnlyConeHadem(const reco::GsfElectron& ele) const {
  bool passCutBased = ele.passingCutBasedPreselection();
  if (ele.hadronicOverEm() > ele_ecalDrivenHademPreselCut_)
    passCutBased = false;
  bool passMVA = ele.passingMvaPreselection();
  if (!ele.ecalDrivenSeed()) {
    if (ele.pt() > ele_maxElePtForOnlyMVAPresel_)
      return passMVA && passCutBased;
    else
      return passMVA;
  } else
    return passCutBased || passMVA;
}

void PFEGammaFilters::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
  // Electron selection cuts
  iDesc.add<double>("electron_iso_pt", 10.0);
  iDesc.add<double>("electron_iso_mva_barrel", -0.1875);
  iDesc.add<double>("electron_iso_mva_endcap", -0.1075);
  iDesc.add<double>("electron_iso_combIso_barrel", 10.0);
  iDesc.add<double>("electron_iso_combIso_endcap", 10.0);
  iDesc.add<double>("electron_noniso_mvaCut", -0.1);
  iDesc.add<unsigned int>("electron_missinghits", 1);
  iDesc.add<double>("electron_ecalDrivenHademPreselCut", 0.15);
  iDesc.add<double>("electron_maxElePtForOnlyMVAPresel", 50.0);
  iDesc.add<bool>("useElePFidDnn", false);
  iDesc.add<double>("endcapBoundary", 2.5);
  iDesc.add<double>("extEtaBoundary", 2.65);
  {
    edm::ParameterSetDescription psd;
    psd.add<double>("electronDnnLowPtThr", 0.5);
    psd.add<double>("electronDnnHighPtBarrelThr", 0.5);
    psd.add<double>("electronDnnHighPtEndcapThr", 0.5);
    psd.add<double>("electronDnnExtEta1Thr", 0.5);
    psd.add<double>("electronDnnExtEta2Thr", 0.5);
    iDesc.add<edm::ParameterSetDescription>("electronDnnThresholds", psd);
  }
  {
    edm::ParameterSetDescription psd;
    psd.add<double>("electronDnnBkgLowPtThr", 1);
    psd.add<double>("electronDnnBkgHighPtBarrelThr", 1);
    psd.add<double>("electronDnnBkgHighPtEndcapThr", 1);
    psd.add<double>("electronDnnBkgExtEta1Thr", 1);
    psd.add<double>("electronDnnBkgExtEta2Thr", 1);
    iDesc.add<edm::ParameterSetDescription>("electronDnnBkgThresholds", psd);
  }
  iDesc.add<bool>("usePhotonPFidDnn", false);
  {
    edm::ParameterSetDescription psd;
    psd.add<double>("photonDnnBarrelThr", 0.5);
    psd.add<double>("photonDnnEndcapThr", 0.5);
    iDesc.add<edm::ParameterSetDescription>("photonDnnThresholds", psd);
  }
  // control if the EB DNN models should be used up to eta 1.485 or 1.566
  iDesc.add<bool>("useEBModelInGap", true);
  {
    edm::ParameterSetDescription psd;
    psd.add<double>("maxNtracks", 3.0)->setComment("Max tracks pointing at Ele cluster");
    psd.add<double>("maxHcalE", 10.0);
    psd.add<double>("maxTrackPOverEele", 1.0);
    psd.add<double>("maxE", 50.0)->setComment("Above this maxE, consider dphi(SC,track) cut");
    psd.add<double>("maxEleHcalEOverEcalE", 0.1);
    psd.add<double>("maxEcalEOverPRes", 0.2);
    psd.add<double>("maxEeleOverPoutRes", 0.5);
    psd.add<double>("maxHcalEOverP", 1.0);
    psd.add<double>("maxHcalEOverEcalE", 0.1);
    psd.add<double>("maxEcalEOverP_1", 0.5)->setComment("E(SC)/P cut - pion rejection");
    psd.add<double>("maxEcalEOverP_2", 0.2)->setComment("E(SC)/P cut - weird ele rejection");
    psd.add<double>("maxEeleOverPout", 0.2);
    psd.add<double>("maxDPhiIN", 0.1)->setComment("Above this dphi(SC,track) and maxE, considered not safe");
    iDesc.add<edm::ParameterSetDescription>("electron_protectionsForJetMET", psd);
  }
  {
    edm::ParameterSetDescription psd;
    psd.add<bool>("enableProtections", true);
    psd.add<std::vector<double>>("full5x5_sigmaIetaIeta",  // EB, EE; 94Xv2 cut-based medium id
                                 {0.0106, 0.0387});
    psd.add<std::vector<double>>("eInvPInv", {0.184, 0.0721});
    psd.add<std::vector<double>>("dEta",  // relax factor 2 to be safer against misalignment
                                 {0.0032 * 2, 0.00632 * 2});
    psd.add<std::vector<double>>("dPhi", {0.0547, 0.0394});
    iDesc.add<edm::ParameterSetDescription>("electron_protectionsForBadHcal", psd);
  }

  // Photon selection cuts
  iDesc.add<double>("photon_MinEt", 10.0);
  iDesc.add<double>("photon_combIso", 10.0);
  iDesc.add<double>("photon_HoE", 0.05);
  iDesc.add<double>("photon_SigmaiEtaiEta_barrel", 0.0125);
  iDesc.add<double>("photon_SigmaiEtaiEta_endcap", 0.034);
  {
    edm::ParameterSetDescription psd;
    psd.add<double>("sumPtTrackIso", 4.0);
    psd.add<double>("sumPtTrackIsoSlope", 0.001);
    iDesc.add<edm::ParameterSetDescription>("photon_protectionsForJetMET", psd);
  }
  {
    edm::ParameterSetDescription psd;
    psd.add<double>("solidConeTrkIsoSlope", 0.3);
    psd.add<bool>("enableProtections", true);
    psd.add<double>("solidConeTrkIsoOffset", 10.0);
    iDesc.add<edm::ParameterSetDescription>("photon_protectionsForBadHcal", psd);
  }
}
