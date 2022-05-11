/*
 * \class DeepTauId
 *
 * Tau identification using Deep NN.
 *
 * \author Konstantin Androsov, INFN Pisa
 *         Christian Veelken, Tallinn
 */

#include "RecoTauTag/RecoTau/interface/DeepTauBase.h"
#include "RecoTauTag/RecoTau/interface/DeepTauScaling.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"

#include <fstream>
#include "oneapi/tbb/concurrent_unordered_set.h"

namespace deep_tau {
  constexpr int NumberOfOutputs = 4;
}

namespace {

  struct dnn_inputs_v1 {
    enum vars {
      pt = 0,
      eta,
      mass,
      decayMode,
      chargedIsoPtSum,
      neutralIsoPtSum,
      neutralIsoPtSumWeight,
      photonPtSumOutsideSignalCone,
      puCorrPtSum,
      dxy,
      dxy_sig,
      dz,
      ip3d,
      ip3d_sig,
      hasSecondaryVertex,
      flightLength_r,
      flightLength_dEta,
      flightLength_dPhi,
      flightLength_sig,
      leadChargedHadrCand_pt,
      leadChargedHadrCand_dEta,
      leadChargedHadrCand_dPhi,
      leadChargedHadrCand_mass,
      pt_weighted_deta_strip,
      pt_weighted_dphi_strip,
      pt_weighted_dr_signal,
      pt_weighted_dr_iso,
      leadingTrackNormChi2,
      e_ratio,
      gj_angle_diff,
      n_photons,
      emFraction,
      has_gsf_track,
      inside_ecal_crack,
      gsf_ele_matched,
      gsf_ele_pt,
      gsf_ele_dEta,
      gsf_ele_dPhi,
      gsf_ele_mass,
      gsf_ele_Ee,
      gsf_ele_Egamma,
      gsf_ele_Pin,
      gsf_ele_Pout,
      gsf_ele_EtotOverPin,
      gsf_ele_Eecal,
      gsf_ele_dEta_SeedClusterTrackAtCalo,
      gsf_ele_dPhi_SeedClusterTrackAtCalo,
      gsf_ele_mvaIn_sigmaEtaEta,
      gsf_ele_mvaIn_hadEnergy,
      gsf_ele_mvaIn_deltaEta,
      gsf_ele_Chi2NormGSF,
      gsf_ele_GSFNumHits,
      gsf_ele_GSFTrackResol,
      gsf_ele_GSFTracklnPt,
      gsf_ele_Chi2NormKF,
      gsf_ele_KFNumHits,
      leadChargedCand_etaAtEcalEntrance,
      leadChargedCand_pt,
      leadChargedHadrCand_HoP,
      leadChargedHadrCand_EoP,
      tau_visMass_innerSigCone,
      n_matched_muons,
      muon_pt,
      muon_dEta,
      muon_dPhi,
      muon_n_matches_DT_1,
      muon_n_matches_DT_2,
      muon_n_matches_DT_3,
      muon_n_matches_DT_4,
      muon_n_matches_CSC_1,
      muon_n_matches_CSC_2,
      muon_n_matches_CSC_3,
      muon_n_matches_CSC_4,
      muon_n_hits_DT_2,
      muon_n_hits_DT_3,
      muon_n_hits_DT_4,
      muon_n_hits_CSC_2,
      muon_n_hits_CSC_3,
      muon_n_hits_CSC_4,
      muon_n_hits_RPC_2,
      muon_n_hits_RPC_3,
      muon_n_hits_RPC_4,
      muon_n_stations_with_matches_03,
      muon_n_stations_with_hits_23,
      signalChargedHadrCands_sum_innerSigCone_pt,
      signalChargedHadrCands_sum_innerSigCone_dEta,
      signalChargedHadrCands_sum_innerSigCone_dPhi,
      signalChargedHadrCands_sum_innerSigCone_mass,
      signalChargedHadrCands_sum_outerSigCone_pt,
      signalChargedHadrCands_sum_outerSigCone_dEta,
      signalChargedHadrCands_sum_outerSigCone_dPhi,
      signalChargedHadrCands_sum_outerSigCone_mass,
      signalChargedHadrCands_nTotal_innerSigCone,
      signalChargedHadrCands_nTotal_outerSigCone,
      signalNeutrHadrCands_sum_innerSigCone_pt,
      signalNeutrHadrCands_sum_innerSigCone_dEta,
      signalNeutrHadrCands_sum_innerSigCone_dPhi,
      signalNeutrHadrCands_sum_innerSigCone_mass,
      signalNeutrHadrCands_sum_outerSigCone_pt,
      signalNeutrHadrCands_sum_outerSigCone_dEta,
      signalNeutrHadrCands_sum_outerSigCone_dPhi,
      signalNeutrHadrCands_sum_outerSigCone_mass,
      signalNeutrHadrCands_nTotal_innerSigCone,
      signalNeutrHadrCands_nTotal_outerSigCone,
      signalGammaCands_sum_innerSigCone_pt,
      signalGammaCands_sum_innerSigCone_dEta,
      signalGammaCands_sum_innerSigCone_dPhi,
      signalGammaCands_sum_innerSigCone_mass,
      signalGammaCands_sum_outerSigCone_pt,
      signalGammaCands_sum_outerSigCone_dEta,
      signalGammaCands_sum_outerSigCone_dPhi,
      signalGammaCands_sum_outerSigCone_mass,
      signalGammaCands_nTotal_innerSigCone,
      signalGammaCands_nTotal_outerSigCone,
      isolationChargedHadrCands_sum_pt,
      isolationChargedHadrCands_sum_dEta,
      isolationChargedHadrCands_sum_dPhi,
      isolationChargedHadrCands_sum_mass,
      isolationChargedHadrCands_nTotal,
      isolationNeutrHadrCands_sum_pt,
      isolationNeutrHadrCands_sum_dEta,
      isolationNeutrHadrCands_sum_dPhi,
      isolationNeutrHadrCands_sum_mass,
      isolationNeutrHadrCands_nTotal,
      isolationGammaCands_sum_pt,
      isolationGammaCands_sum_dEta,
      isolationGammaCands_sum_dPhi,
      isolationGammaCands_sum_mass,
      isolationGammaCands_nTotal,
      NumberOfInputs
    };
  };

  namespace dnn_inputs_v2 {
    constexpr int number_of_inner_cell = 11;
    constexpr int number_of_outer_cell = 21;
    constexpr int number_of_conv_features = 64;
    namespace TauBlockInputs {
      enum vars {
        rho = 0,
        tau_pt,
        tau_eta,
        tau_phi,
        tau_mass,
        tau_E_over_pt,
        tau_charge,
        tau_n_charged_prongs,
        tau_n_neutral_prongs,
        chargedIsoPtSum,
        chargedIsoPtSumdR03_over_dR05,
        footprintCorrection,
        neutralIsoPtSum,
        neutralIsoPtSumWeight_over_neutralIsoPtSum,
        neutralIsoPtSumWeightdR03_over_neutralIsoPtSum,
        neutralIsoPtSumdR03_over_dR05,
        photonPtSumOutsideSignalCone,
        puCorrPtSum,
        tau_dxy_pca_x,
        tau_dxy_pca_y,
        tau_dxy_pca_z,
        tau_dxy_valid,
        tau_dxy,
        tau_dxy_sig,
        tau_ip3d_valid,
        tau_ip3d,
        tau_ip3d_sig,
        tau_dz,
        tau_dz_sig_valid,
        tau_dz_sig,
        tau_flightLength_x,
        tau_flightLength_y,
        tau_flightLength_z,
        tau_flightLength_sig,
        tau_pt_weighted_deta_strip,
        tau_pt_weighted_dphi_strip,
        tau_pt_weighted_dr_signal,
        tau_pt_weighted_dr_iso,
        tau_leadingTrackNormChi2,
        tau_e_ratio_valid,
        tau_e_ratio,
        tau_gj_angle_diff_valid,
        tau_gj_angle_diff,
        tau_n_photons,
        tau_emFraction,
        tau_inside_ecal_crack,
        leadChargedCand_etaAtEcalEntrance_minus_tau_eta,
        NumberOfInputs
      };
      std::vector<int> varsToDrop = {
          tau_phi, tau_dxy_pca_x, tau_dxy_pca_y, tau_dxy_pca_z};  // indices of vars to be dropped in the full var enum
    }                                                             // namespace TauBlockInputs

    namespace EgammaBlockInputs {
      enum vars {
        rho = 0,
        tau_pt,
        tau_eta,
        tau_inside_ecal_crack,
        pfCand_ele_valid,
        pfCand_ele_rel_pt,
        pfCand_ele_deta,
        pfCand_ele_dphi,
        pfCand_ele_pvAssociationQuality,
        pfCand_ele_puppiWeight,
        pfCand_ele_charge,
        pfCand_ele_lostInnerHits,
        pfCand_ele_numberOfPixelHits,
        pfCand_ele_vertex_dx,
        pfCand_ele_vertex_dy,
        pfCand_ele_vertex_dz,
        pfCand_ele_vertex_dx_tauFL,
        pfCand_ele_vertex_dy_tauFL,
        pfCand_ele_vertex_dz_tauFL,
        pfCand_ele_hasTrackDetails,
        pfCand_ele_dxy,
        pfCand_ele_dxy_sig,
        pfCand_ele_dz,
        pfCand_ele_dz_sig,
        pfCand_ele_track_chi2_ndof,
        pfCand_ele_track_ndof,
        ele_valid,
        ele_rel_pt,
        ele_deta,
        ele_dphi,
        ele_cc_valid,
        ele_cc_ele_rel_energy,
        ele_cc_gamma_rel_energy,
        ele_cc_n_gamma,
        ele_rel_trackMomentumAtVtx,
        ele_rel_trackMomentumAtCalo,
        ele_rel_trackMomentumOut,
        ele_rel_trackMomentumAtEleClus,
        ele_rel_trackMomentumAtVtxWithConstraint,
        ele_rel_ecalEnergy,
        ele_ecalEnergy_sig,
        ele_eSuperClusterOverP,
        ele_eSeedClusterOverP,
        ele_eSeedClusterOverPout,
        ele_eEleClusterOverPout,
        ele_deltaEtaSuperClusterTrackAtVtx,
        ele_deltaEtaSeedClusterTrackAtCalo,
        ele_deltaEtaEleClusterTrackAtCalo,
        ele_deltaPhiEleClusterTrackAtCalo,
        ele_deltaPhiSuperClusterTrackAtVtx,
        ele_deltaPhiSeedClusterTrackAtCalo,
        ele_mvaInput_earlyBrem,
        ele_mvaInput_lateBrem,
        ele_mvaInput_sigmaEtaEta,
        ele_mvaInput_hadEnergy,
        ele_mvaInput_deltaEta,
        ele_gsfTrack_normalizedChi2,
        ele_gsfTrack_numberOfValidHits,
        ele_rel_gsfTrack_pt,
        ele_gsfTrack_pt_sig,
        ele_has_closestCtfTrack,
        ele_closestCtfTrack_normalizedChi2,
        ele_closestCtfTrack_numberOfValidHits,
        pfCand_gamma_valid,
        pfCand_gamma_rel_pt,
        pfCand_gamma_deta,
        pfCand_gamma_dphi,
        pfCand_gamma_pvAssociationQuality,
        pfCand_gamma_fromPV,
        pfCand_gamma_puppiWeight,
        pfCand_gamma_puppiWeightNoLep,
        pfCand_gamma_lostInnerHits,
        pfCand_gamma_numberOfPixelHits,
        pfCand_gamma_vertex_dx,
        pfCand_gamma_vertex_dy,
        pfCand_gamma_vertex_dz,
        pfCand_gamma_vertex_dx_tauFL,
        pfCand_gamma_vertex_dy_tauFL,
        pfCand_gamma_vertex_dz_tauFL,
        pfCand_gamma_hasTrackDetails,
        pfCand_gamma_dxy,
        pfCand_gamma_dxy_sig,
        pfCand_gamma_dz,
        pfCand_gamma_dz_sig,
        pfCand_gamma_track_chi2_ndof,
        pfCand_gamma_track_ndof,
        NumberOfInputs
      };
    }

    namespace MuonBlockInputs {
      enum vars {
        rho = 0,
        tau_pt,
        tau_eta,
        tau_inside_ecal_crack,
        pfCand_muon_valid,
        pfCand_muon_rel_pt,
        pfCand_muon_deta,
        pfCand_muon_dphi,
        pfCand_muon_pvAssociationQuality,
        pfCand_muon_fromPV,
        pfCand_muon_puppiWeight,
        pfCand_muon_charge,
        pfCand_muon_lostInnerHits,
        pfCand_muon_numberOfPixelHits,
        pfCand_muon_vertex_dx,
        pfCand_muon_vertex_dy,
        pfCand_muon_vertex_dz,
        pfCand_muon_vertex_dx_tauFL,
        pfCand_muon_vertex_dy_tauFL,
        pfCand_muon_vertex_dz_tauFL,
        pfCand_muon_hasTrackDetails,
        pfCand_muon_dxy,
        pfCand_muon_dxy_sig,
        pfCand_muon_dz,
        pfCand_muon_dz_sig,
        pfCand_muon_track_chi2_ndof,
        pfCand_muon_track_ndof,
        muon_valid,
        muon_rel_pt,
        muon_deta,
        muon_dphi,
        muon_dxy,
        muon_dxy_sig,
        muon_normalizedChi2_valid,
        muon_normalizedChi2,
        muon_numberOfValidHits,
        muon_segmentCompatibility,
        muon_caloCompatibility,
        muon_pfEcalEnergy_valid,
        muon_rel_pfEcalEnergy,
        muon_n_matches_DT_1,
        muon_n_matches_DT_2,
        muon_n_matches_DT_3,
        muon_n_matches_DT_4,
        muon_n_matches_CSC_1,
        muon_n_matches_CSC_2,
        muon_n_matches_CSC_3,
        muon_n_matches_CSC_4,
        muon_n_matches_RPC_1,
        muon_n_matches_RPC_2,
        muon_n_matches_RPC_3,
        muon_n_matches_RPC_4,
        muon_n_hits_DT_1,
        muon_n_hits_DT_2,
        muon_n_hits_DT_3,
        muon_n_hits_DT_4,
        muon_n_hits_CSC_1,
        muon_n_hits_CSC_2,
        muon_n_hits_CSC_3,
        muon_n_hits_CSC_4,
        muon_n_hits_RPC_1,
        muon_n_hits_RPC_2,
        muon_n_hits_RPC_3,
        muon_n_hits_RPC_4,
        NumberOfInputs
      };
    }

    namespace HadronBlockInputs {
      enum vars {
        rho = 0,
        tau_pt,
        tau_eta,
        tau_inside_ecal_crack,
        pfCand_chHad_valid,
        pfCand_chHad_rel_pt,
        pfCand_chHad_deta,
        pfCand_chHad_dphi,
        pfCand_chHad_leadChargedHadrCand,
        pfCand_chHad_pvAssociationQuality,
        pfCand_chHad_fromPV,
        pfCand_chHad_puppiWeight,
        pfCand_chHad_puppiWeightNoLep,
        pfCand_chHad_charge,
        pfCand_chHad_lostInnerHits,
        pfCand_chHad_numberOfPixelHits,
        pfCand_chHad_vertex_dx,
        pfCand_chHad_vertex_dy,
        pfCand_chHad_vertex_dz,
        pfCand_chHad_vertex_dx_tauFL,
        pfCand_chHad_vertex_dy_tauFL,
        pfCand_chHad_vertex_dz_tauFL,
        pfCand_chHad_hasTrackDetails,
        pfCand_chHad_dxy,
        pfCand_chHad_dxy_sig,
        pfCand_chHad_dz,
        pfCand_chHad_dz_sig,
        pfCand_chHad_track_chi2_ndof,
        pfCand_chHad_track_ndof,
        pfCand_chHad_hcalFraction,
        pfCand_chHad_rawCaloFraction,
        pfCand_nHad_valid,
        pfCand_nHad_rel_pt,
        pfCand_nHad_deta,
        pfCand_nHad_dphi,
        pfCand_nHad_puppiWeight,
        pfCand_nHad_puppiWeightNoLep,
        pfCand_nHad_hcalFraction,
        NumberOfInputs
      };
    }
  }  // namespace dnn_inputs_v2

  float getTauID(const pat::Tau& tau, const std::string& tauID, float default_value = -999., bool assert_input = true) {
    static tbb::concurrent_unordered_set<std::string> isFirstWarning;
    if (tau.isTauIDAvailable(tauID)) {
      return tau.tauID(tauID);
    } else {
      if (assert_input) {
        throw cms::Exception("DeepTauId")
            << "Exception in <getTauID>: No tauID '" << tauID << "' available in pat::Tau given as function argument.";
      }
      if (isFirstWarning.insert(tauID).second) {
        edm::LogWarning("DeepTauID") << "Warning in <getTauID>: No tauID '" << tauID
                                     << "' available in pat::Tau given as function argument."
                                     << " Using default_value = " << default_value << " instead." << std::endl;
      }
      return default_value;
    }
  }

  struct TauFunc {
    const reco::TauDiscriminatorContainer* basicTauDiscriminatorCollection;
    const reco::TauDiscriminatorContainer* basicTauDiscriminatordR03Collection;
    const edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>*
        pfTauTransverseImpactParameters;

    using BasicDiscr = deep_tau::DeepTauBase::BasicDiscriminator;
    std::map<BasicDiscr, size_t> indexMap;
    std::map<BasicDiscr, size_t> indexMapdR03;

    const float getChargedIsoPtSum(const reco::PFTau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return (*basicTauDiscriminatorCollection)[tau_ref].rawValues.at(indexMap.at(BasicDiscr::ChargedIsoPtSum));
    }
    const float getChargedIsoPtSum(const pat::Tau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return getTauID(tau, "chargedIsoPtSum");
    }
    const float getChargedIsoPtSumdR03(const reco::PFTau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return (*basicTauDiscriminatordR03Collection)[tau_ref].rawValues.at(indexMapdR03.at(BasicDiscr::ChargedIsoPtSum));
    }
    const float getChargedIsoPtSumdR03(const pat::Tau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return getTauID(tau, "chargedIsoPtSumdR03");
    }
    const float getFootprintCorrectiondR03(const reco::PFTau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return (*basicTauDiscriminatordR03Collection)[tau_ref].rawValues.at(
          indexMapdR03.at(BasicDiscr::FootprintCorrection));
    }
    const float getFootprintCorrectiondR03(const pat::Tau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return getTauID(tau, "footprintCorrectiondR03");
    }
    const float getFootprintCorrection(const pat::Tau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return getTauID(tau, "footprintCorrection");
    }
    const float getNeutralIsoPtSum(const reco::PFTau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return (*basicTauDiscriminatorCollection)[tau_ref].rawValues.at(indexMap.at(BasicDiscr::NeutralIsoPtSum));
    }
    const float getNeutralIsoPtSum(const pat::Tau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return getTauID(tau, "neutralIsoPtSum");
    }
    const float getNeutralIsoPtSumdR03(const reco::PFTau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return (*basicTauDiscriminatordR03Collection)[tau_ref].rawValues.at(indexMapdR03.at(BasicDiscr::NeutralIsoPtSum));
    }
    const float getNeutralIsoPtSumdR03(const pat::Tau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return getTauID(tau, "neutralIsoPtSumdR03");
    }
    const float getNeutralIsoPtSumWeight(const reco::PFTau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return (*basicTauDiscriminatorCollection)[tau_ref].rawValues.at(indexMap.at(BasicDiscr::NeutralIsoPtSumWeight));
    }
    const float getNeutralIsoPtSumWeight(const pat::Tau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return getTauID(tau, "neutralIsoPtSumWeight");
    }
    const float getNeutralIsoPtSumdR03Weight(const reco::PFTau& tau,
                                             const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return (*basicTauDiscriminatordR03Collection)[tau_ref].rawValues.at(
          indexMapdR03.at(BasicDiscr::NeutralIsoPtSumWeight));
    }
    const float getNeutralIsoPtSumdR03Weight(const pat::Tau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return getTauID(tau, "neutralIsoPtSumWeightdR03");
    }
    const float getPhotonPtSumOutsideSignalCone(const reco::PFTau& tau,
                                                const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return (*basicTauDiscriminatorCollection)[tau_ref].rawValues.at(
          indexMap.at(BasicDiscr::PhotonPtSumOutsideSignalCone));
    }
    const float getPhotonPtSumOutsideSignalCone(const pat::Tau& tau,
                                                const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return getTauID(tau, "photonPtSumOutsideSignalCone");
    }
    const float getPhotonPtSumOutsideSignalConedR03(const reco::PFTau& tau,
                                                    const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return (*basicTauDiscriminatordR03Collection)[tau_ref].rawValues.at(
          indexMapdR03.at(BasicDiscr::PhotonPtSumOutsideSignalCone));
    }
    const float getPhotonPtSumOutsideSignalConedR03(const pat::Tau& tau,
                                                    const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return getTauID(tau, "photonPtSumOutsideSignalConedR03");
    }
    const float getPuCorrPtSum(const reco::PFTau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return (*basicTauDiscriminatorCollection)[tau_ref].rawValues.at(indexMap.at(BasicDiscr::PUcorrPtSum));
    }
    const float getPuCorrPtSum(const pat::Tau& tau, const edm::RefToBase<reco::BaseTau> tau_ref) const {
      return getTauID(tau, "puCorrPtSum");
    }

    auto getdxyPCA(const reco::PFTau& tau, const size_t tau_index) const {
      return pfTauTransverseImpactParameters->value(tau_index)->dxy_PCA();
    }
    auto getdxyPCA(const pat::Tau& tau, const size_t tau_index) const { return tau.dxy_PCA(); }
    auto getdxy(const reco::PFTau& tau, const size_t tau_index) const {
      return pfTauTransverseImpactParameters->value(tau_index)->dxy();
    }
    auto getdxy(const pat::Tau& tau, const size_t tau_index) const { return tau.dxy(); }
    auto getdxyError(const reco::PFTau& tau, const size_t tau_index) const {
      return pfTauTransverseImpactParameters->value(tau_index)->dxy_error();
    }
    auto getdxyError(const pat::Tau& tau, const size_t tau_index) const { return tau.dxy_error(); }
    auto getdxySig(const reco::PFTau& tau, const size_t tau_index) const {
      return pfTauTransverseImpactParameters->value(tau_index)->dxy_Sig();
    }
    auto getdxySig(const pat::Tau& tau, const size_t tau_index) const { return tau.dxy_Sig(); }
    auto getip3d(const reco::PFTau& tau, const size_t tau_index) const {
      return pfTauTransverseImpactParameters->value(tau_index)->ip3d();
    }
    auto getip3d(const pat::Tau& tau, const size_t tau_index) const { return tau.ip3d(); }
    auto getip3dError(const reco::PFTau& tau, const size_t tau_index) const {
      return pfTauTransverseImpactParameters->value(tau_index)->ip3d_error();
    }
    auto getip3dError(const pat::Tau& tau, const size_t tau_index) const { return tau.ip3d_error(); }
    auto getip3dSig(const reco::PFTau& tau, const size_t tau_index) const {
      return pfTauTransverseImpactParameters->value(tau_index)->ip3d_Sig();
    }
    auto getip3dSig(const pat::Tau& tau, const size_t tau_index) const { return tau.ip3d_Sig(); }
    auto getHasSecondaryVertex(const reco::PFTau& tau, const size_t tau_index) const {
      return pfTauTransverseImpactParameters->value(tau_index)->hasSecondaryVertex();
    }
    auto getHasSecondaryVertex(const pat::Tau& tau, const size_t tau_index) const { return tau.hasSecondaryVertex(); }
    auto getFlightLength(const reco::PFTau& tau, const size_t tau_index) const {
      return pfTauTransverseImpactParameters->value(tau_index)->flightLength();
    }
    auto getFlightLength(const pat::Tau& tau, const size_t tau_index) const { return tau.flightLength(); }
    auto getFlightLengthSig(const reco::PFTau& tau, const size_t tau_index) const {
      return pfTauTransverseImpactParameters->value(tau_index)->flightLengthSig();
    }
    auto getFlightLengthSig(const pat::Tau& tau, const size_t tau_index) const { return tau.flightLengthSig(); }

    auto getLeadingTrackNormChi2(const reco::PFTau& tau) { return reco::tau::lead_track_chi2(tau); }
    auto getLeadingTrackNormChi2(const pat::Tau& tau) { return tau.leadingTrackNormChi2(); }
    auto getEmFraction(const pat::Tau& tau) { return tau.emFraction_MVA(); }
    auto getEmFraction(const reco::PFTau& tau) { return tau.emFraction(); }
    auto getEtaAtEcalEntrance(const pat::Tau& tau) { return tau.etaAtEcalEntranceLeadChargedCand(); }
    auto getEtaAtEcalEntrance(const reco::PFTau& tau) {
      return tau.leadPFChargedHadrCand()->positionAtECALEntrance().eta();
    }
    auto getEcalEnergyLeadingChargedHadr(const reco::PFTau& tau) { return tau.leadPFChargedHadrCand()->ecalEnergy(); }
    auto getEcalEnergyLeadingChargedHadr(const pat::Tau& tau) { return tau.ecalEnergyLeadChargedHadrCand(); }
    auto getHcalEnergyLeadingChargedHadr(const reco::PFTau& tau) { return tau.leadPFChargedHadrCand()->hcalEnergy(); }
    auto getHcalEnergyLeadingChargedHadr(const pat::Tau& tau) { return tau.hcalEnergyLeadChargedHadrCand(); }

    template <typename PreDiscrType>
    bool passPrediscriminants(const PreDiscrType prediscriminants,
                              const size_t andPrediscriminants,
                              const edm::RefToBase<reco::BaseTau> tau_ref) {
      bool passesPrediscriminants = (andPrediscriminants ? 1 : 0);
      // check tau passes prediscriminants
      size_t nPrediscriminants = prediscriminants.size();
      for (size_t iDisc = 0; iDisc < nPrediscriminants; ++iDisc) {
        // current discriminant result for this tau
        double discResult = (*prediscriminants[iDisc].handle)[tau_ref];
        uint8_t thisPasses = (discResult > prediscriminants[iDisc].cut) ? 1 : 0;

        // if we are using the AND option, as soon as one fails,
        // the result is FAIL and we can quit looping.
        // if we are using the OR option as soon as one passes,
        // the result is pass and we can quit looping

        // truth table
        //        |   result (thisPasses)
        //        |     F     |     T
        //-----------------------------------
        // AND(T) | res=fails |  continue
        //        |  break    |
        //-----------------------------------
        // OR (F) |  continue | res=passes
        //        |           |  break

        if (thisPasses ^ andPrediscriminants)  //XOR
        {
          passesPrediscriminants = (andPrediscriminants ? 0 : 1);  //NOR
          break;
        }
      }
      return passesPrediscriminants;
    }
  };

  namespace candFunc {
    auto getTauDz(const reco::PFCandidate& cand) { return cand.bestTrack()->dz(); }
    auto getTauDz(const pat::PackedCandidate& cand) { return cand.dz(); }
    auto getTauDZSigValid(const reco::PFCandidate& cand) {
      return cand.bestTrack() != nullptr && std::isnormal(cand.bestTrack()->dz()) && std::isnormal(cand.dzError()) &&
             cand.dzError() > 0;
    }
    auto getTauDZSigValid(const pat::PackedCandidate& cand) {
      return cand.hasTrackDetails() && std::isnormal(cand.dz()) && std::isnormal(cand.dzError()) && cand.dzError() > 0;
    }
    auto getTauDxy(const reco::PFCandidate& cand) { return cand.bestTrack()->dxy(); }
    auto getTauDxy(const pat::PackedCandidate& cand) { return cand.dxy(); }
    auto getPvAssocationQuality(const reco::PFCandidate& cand) { return 0.7013f; }
    auto getPvAssocationQuality(const pat::PackedCandidate& cand) { return cand.pvAssociationQuality(); }
    auto getPuppiWeight(const reco::PFCandidate& cand, const float aod_value) { return aod_value; }
    auto getPuppiWeight(const pat::PackedCandidate& cand, const float aod_value) { return cand.puppiWeight(); }
    auto getPuppiWeightNoLep(const reco::PFCandidate& cand, const float aod_value) { return aod_value; }
    auto getPuppiWeightNoLep(const pat::PackedCandidate& cand, const float aod_value) {
      return cand.puppiWeightNoLep();
    }
    auto getLostInnerHits(const reco::PFCandidate& cand, float default_value) {
      return cand.bestTrack() != nullptr
                 ? cand.bestTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS)
                 : default_value;
    }
    auto getLostInnerHits(const pat::PackedCandidate& cand, float default_value) { return cand.lostInnerHits(); }
    auto getNumberOfPixelHits(const reco::PFCandidate& cand, float default_value) {
      return cand.bestTrack() != nullptr
                 ? cand.bestTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS)
                 : default_value;
    }
    auto getNumberOfPixelHits(const pat::PackedCandidate& cand, float default_value) {
      return cand.numberOfPixelHits();
    }
    auto getHasTrackDetails(const reco::PFCandidate& cand) { return cand.bestTrack() != nullptr; }
    auto getHasTrackDetails(const pat::PackedCandidate& cand) { return cand.hasTrackDetails(); }
    auto getPseudoTrack(const reco::PFCandidate& cand) { return *cand.bestTrack(); }
    auto getPseudoTrack(const pat::PackedCandidate& cand) { return cand.pseudoTrack(); }
    auto getFromPV(const reco::PFCandidate& cand) { return 0.9994f; }
    auto getFromPV(const pat::PackedCandidate& cand) { return cand.fromPV(); }
    auto getHCalFraction(const reco::PFCandidate& cand, bool disable_hcalFraction_workaround) {
      return cand.rawHcalEnergy() / (cand.rawHcalEnergy() + cand.rawEcalEnergy());
    }
    auto getHCalFraction(const pat::PackedCandidate& cand, bool disable_hcalFraction_workaround) {
      float hcal_fraction = 0.;
      if (disable_hcalFraction_workaround) {
        // CV: use consistent definition for pfCand_chHad_hcalFraction
        //     in DeepTauId.cc code and in TauMLTools/Production/plugins/TauTupleProducer.cc
        hcal_fraction = cand.hcalFraction();
      } else {
        // CV: backwards compatibility with DeepTau training v2p1 used during Run 2
        if (cand.pdgId() == 1 || cand.pdgId() == 130) {
          hcal_fraction = cand.hcalFraction();
        } else if (cand.isIsolatedChargedHadron()) {
          hcal_fraction = cand.rawHcalFraction();
        }
      }
      return hcal_fraction;
    }
    auto getRawCaloFraction(const reco::PFCandidate& cand) {
      return (cand.rawEcalEnergy() + cand.rawHcalEnergy()) / cand.energy();
    }
    auto getRawCaloFraction(const pat::PackedCandidate& cand) { return cand.rawCaloFraction(); }
  };  // namespace candFunc

  template <typename LVector1, typename LVector2>
  float dEta(const LVector1& p4, const LVector2& tau_p4) {
    return static_cast<float>(p4.eta() - tau_p4.eta());
  }

  template <typename LVector1, typename LVector2>
  float dPhi(const LVector1& p4_1, const LVector2& p4_2) {
    return static_cast<float>(reco::deltaPhi(p4_2.phi(), p4_1.phi()));
  }

  struct MuonHitMatchV1 {
    static constexpr int n_muon_stations = 4;

    std::map<int, std::vector<UInt_t>> n_matches, n_hits;
    unsigned n_muons{0};
    const pat::Muon* best_matched_muon{nullptr};
    double deltaR2_best_match{-1};

    MuonHitMatchV1() {
      n_matches[MuonSubdetId::DT].assign(n_muon_stations, 0);
      n_matches[MuonSubdetId::CSC].assign(n_muon_stations, 0);
      n_matches[MuonSubdetId::RPC].assign(n_muon_stations, 0);
      n_hits[MuonSubdetId::DT].assign(n_muon_stations, 0);
      n_hits[MuonSubdetId::CSC].assign(n_muon_stations, 0);
      n_hits[MuonSubdetId::RPC].assign(n_muon_stations, 0);
    }

    void addMatchedMuon(const pat::Muon& muon, reco::BaseTau const& tau) {
      static constexpr int n_stations = 4;

      ++n_muons;
      const double dR2 = reco::deltaR2(tau.p4(), muon.p4());
      if (!best_matched_muon || dR2 < deltaR2_best_match) {
        best_matched_muon = &muon;
        deltaR2_best_match = dR2;
      }

      for (const auto& segment : muon.matches()) {
        if (segment.segmentMatches.empty())
          continue;
        if (n_matches.count(segment.detector()))
          ++n_matches.at(segment.detector()).at(segment.station() - 1);
      }

      if (muon.outerTrack().isNonnull()) {
        const auto& hit_pattern = muon.outerTrack()->hitPattern();
        for (int hit_index = 0; hit_index < hit_pattern.numberOfAllHits(reco::HitPattern::TRACK_HITS); ++hit_index) {
          auto hit_id = hit_pattern.getHitPattern(reco::HitPattern::TRACK_HITS, hit_index);
          if (hit_id == 0)
            break;
          if (hit_pattern.muonHitFilter(hit_id) && (hit_pattern.getHitType(hit_id) == TrackingRecHit::valid ||
                                                    hit_pattern.getHitType(hit_id == TrackingRecHit::bad))) {
            const int station = hit_pattern.getMuonStation(hit_id) - 1;
            if (station > 0 && station < n_stations) {
              std::vector<UInt_t>* muon_n_hits = nullptr;
              if (hit_pattern.muonDTHitFilter(hit_id))
                muon_n_hits = &n_hits.at(MuonSubdetId::DT);
              else if (hit_pattern.muonCSCHitFilter(hit_id))
                muon_n_hits = &n_hits.at(MuonSubdetId::CSC);
              else if (hit_pattern.muonRPCHitFilter(hit_id))
                muon_n_hits = &n_hits.at(MuonSubdetId::RPC);

              if (muon_n_hits)
                ++muon_n_hits->at(station);
            }
          }
        }
      }
    }

    template <typename TauCastType>
    static std::vector<const pat::Muon*> findMatchedMuons(const TauCastType& tau,
                                                          const std::vector<pat::Muon>* muons,
                                                          double deltaR,
                                                          double minPt) {
      const reco::Muon* hadr_cand_muon = nullptr;
      if (tau.leadPFChargedHadrCand().isNonnull() && tau.leadPFChargedHadrCand()->muonRef().isNonnull())
        hadr_cand_muon = tau.leadPFChargedHadrCand()->muonRef().get();
      std::vector<const pat::Muon*> matched_muons;
      const double dR2 = deltaR * deltaR;
      for (const pat::Muon& muon : *muons) {
        const reco::Muon* reco_muon = &muon;
        if (muon.pt() <= minPt)
          continue;
        if (reco_muon == hadr_cand_muon)
          continue;
        if (reco::deltaR2(tau.p4(), muon.p4()) >= dR2)
          continue;
        matched_muons.push_back(&muon);
      }
      return matched_muons;
    }

    template <typename dnn, typename TensorElemGet, typename TauCastType>
    void fillTensor(const TensorElemGet& get, const TauCastType& tau, float default_value) const {
      get(dnn::n_matched_muons) = n_muons;
      get(dnn::muon_pt) = best_matched_muon != nullptr ? best_matched_muon->p4().pt() : default_value;
      get(dnn::muon_dEta) = best_matched_muon != nullptr ? dEta(best_matched_muon->p4(), tau.p4()) : default_value;
      get(dnn::muon_dPhi) = best_matched_muon != nullptr ? dPhi(best_matched_muon->p4(), tau.p4()) : default_value;
      get(dnn::muon_n_matches_DT_1) = n_matches.at(MuonSubdetId::DT).at(0);
      get(dnn::muon_n_matches_DT_2) = n_matches.at(MuonSubdetId::DT).at(1);
      get(dnn::muon_n_matches_DT_3) = n_matches.at(MuonSubdetId::DT).at(2);
      get(dnn::muon_n_matches_DT_4) = n_matches.at(MuonSubdetId::DT).at(3);
      get(dnn::muon_n_matches_CSC_1) = n_matches.at(MuonSubdetId::CSC).at(0);
      get(dnn::muon_n_matches_CSC_2) = n_matches.at(MuonSubdetId::CSC).at(1);
      get(dnn::muon_n_matches_CSC_3) = n_matches.at(MuonSubdetId::CSC).at(2);
      get(dnn::muon_n_matches_CSC_4) = n_matches.at(MuonSubdetId::CSC).at(3);
      get(dnn::muon_n_hits_DT_2) = n_hits.at(MuonSubdetId::DT).at(1);
      get(dnn::muon_n_hits_DT_3) = n_hits.at(MuonSubdetId::DT).at(2);
      get(dnn::muon_n_hits_DT_4) = n_hits.at(MuonSubdetId::DT).at(3);
      get(dnn::muon_n_hits_CSC_2) = n_hits.at(MuonSubdetId::CSC).at(1);
      get(dnn::muon_n_hits_CSC_3) = n_hits.at(MuonSubdetId::CSC).at(2);
      get(dnn::muon_n_hits_CSC_4) = n_hits.at(MuonSubdetId::CSC).at(3);
      get(dnn::muon_n_hits_RPC_2) = n_hits.at(MuonSubdetId::RPC).at(1);
      get(dnn::muon_n_hits_RPC_3) = n_hits.at(MuonSubdetId::RPC).at(2);
      get(dnn::muon_n_hits_RPC_4) = n_hits.at(MuonSubdetId::RPC).at(3);
      get(dnn::muon_n_stations_with_matches_03) = countMuonStationsWithMatches(0, 3);
      get(dnn::muon_n_stations_with_hits_23) = countMuonStationsWithHits(2, 3);
    }

  private:
    unsigned countMuonStationsWithMatches(size_t first_station, size_t last_station) const {
      static const std::map<int, std::vector<bool>> masks = {
          {MuonSubdetId::DT, {false, false, false, false}},
          {MuonSubdetId::CSC, {true, false, false, false}},
          {MuonSubdetId::RPC, {false, false, false, false}},
      };
      unsigned cnt = 0;
      for (unsigned n = first_station; n <= last_station; ++n) {
        for (const auto& match : n_matches) {
          if (!masks.at(match.first).at(n) && match.second.at(n) > 0)
            ++cnt;
        }
      }
      return cnt;
    }

    unsigned countMuonStationsWithHits(size_t first_station, size_t last_station) const {
      static const std::map<int, std::vector<bool>> masks = {
          {MuonSubdetId::DT, {false, false, false, false}},
          {MuonSubdetId::CSC, {false, false, false, false}},
          {MuonSubdetId::RPC, {false, false, false, false}},
      };

      unsigned cnt = 0;
      for (unsigned n = first_station; n <= last_station; ++n) {
        for (const auto& hit : n_hits) {
          if (!masks.at(hit.first).at(n) && hit.second.at(n) > 0)
            ++cnt;
        }
      }
      return cnt;
    }
  };

  struct MuonHitMatchV2 {
    static constexpr size_t n_muon_stations = 4;
    static constexpr int first_station_id = 1;
    static constexpr int last_station_id = first_station_id + n_muon_stations - 1;
    using CountArray = std::array<unsigned, n_muon_stations>;
    using CountMap = std::map<int, CountArray>;

    const std::vector<int>& consideredSubdets() {
      static const std::vector<int> subdets = {MuonSubdetId::DT, MuonSubdetId::CSC, MuonSubdetId::RPC};
      return subdets;
    }

    const std::string& subdetName(int subdet) {
      static const std::map<int, std::string> subdet_names = {
          {MuonSubdetId::DT, "DT"}, {MuonSubdetId::CSC, "CSC"}, {MuonSubdetId::RPC, "RPC"}};
      if (!subdet_names.count(subdet))
        throw cms::Exception("MuonHitMatch") << "Subdet name for subdet id " << subdet << " not found.";
      return subdet_names.at(subdet);
    }

    size_t getStationIndex(int station, bool throw_exception) const {
      if (station < first_station_id || station > last_station_id) {
        if (throw_exception)
          throw cms::Exception("MuonHitMatch") << "Station id is out of range";
        return std::numeric_limits<size_t>::max();
      }
      return static_cast<size_t>(station - 1);
    }

    MuonHitMatchV2(const pat::Muon& muon) {
      for (int subdet : consideredSubdets()) {
        n_matches[subdet].fill(0);
        n_hits[subdet].fill(0);
      }

      countMatches(muon, n_matches);
      countHits(muon, n_hits);
    }

    void countMatches(const pat::Muon& muon, CountMap& n_matches) {
      for (const auto& segment : muon.matches()) {
        if (segment.segmentMatches.empty() && segment.rpcMatches.empty())
          continue;
        if (n_matches.count(segment.detector())) {
          const size_t station_index = getStationIndex(segment.station(), true);
          ++n_matches.at(segment.detector()).at(station_index);
        }
      }
    }

    void countHits(const pat::Muon& muon, CountMap& n_hits) {
      if (muon.outerTrack().isNonnull()) {
        const auto& hit_pattern = muon.outerTrack()->hitPattern();
        for (int hit_index = 0; hit_index < hit_pattern.numberOfAllHits(reco::HitPattern::TRACK_HITS); ++hit_index) {
          auto hit_id = hit_pattern.getHitPattern(reco::HitPattern::TRACK_HITS, hit_index);
          if (hit_id == 0)
            break;
          if (hit_pattern.muonHitFilter(hit_id) && (hit_pattern.getHitType(hit_id) == TrackingRecHit::valid ||
                                                    hit_pattern.getHitType(hit_id) == TrackingRecHit::bad)) {
            const size_t station_index = getStationIndex(hit_pattern.getMuonStation(hit_id), false);
            if (station_index < n_muon_stations) {
              CountArray* muon_n_hits = nullptr;
              if (hit_pattern.muonDTHitFilter(hit_id))
                muon_n_hits = &n_hits.at(MuonSubdetId::DT);
              else if (hit_pattern.muonCSCHitFilter(hit_id))
                muon_n_hits = &n_hits.at(MuonSubdetId::CSC);
              else if (hit_pattern.muonRPCHitFilter(hit_id))
                muon_n_hits = &n_hits.at(MuonSubdetId::RPC);

              if (muon_n_hits)
                ++muon_n_hits->at(station_index);
            }
          }
        }
      }
    }

    unsigned nMatches(int subdet, int station) const {
      if (!n_matches.count(subdet))
        throw cms::Exception("MuonHitMatch") << "Subdet " << subdet << " not found.";
      const size_t station_index = getStationIndex(station, true);
      return n_matches.at(subdet).at(station_index);
    }

    unsigned nHits(int subdet, int station) const {
      if (!n_hits.count(subdet))
        throw cms::Exception("MuonHitMatch") << "Subdet " << subdet << " not found.";
      const size_t station_index = getStationIndex(station, true);
      return n_hits.at(subdet).at(station_index);
    }

    unsigned countMuonStationsWithMatches(int first_station, int last_station) const {
      static const std::map<int, std::vector<bool>> masks = {
          {MuonSubdetId::DT, {false, false, false, false}},
          {MuonSubdetId::CSC, {true, false, false, false}},
          {MuonSubdetId::RPC, {false, false, false, false}},
      };
      const size_t first_station_index = getStationIndex(first_station, true);
      const size_t last_station_index = getStationIndex(last_station, true);
      unsigned cnt = 0;
      for (size_t n = first_station_index; n <= last_station_index; ++n) {
        for (const auto& match : n_matches) {
          if (!masks.at(match.first).at(n) && match.second.at(n) > 0)
            ++cnt;
        }
      }
      return cnt;
    }

    unsigned countMuonStationsWithHits(int first_station, int last_station) const {
      static const std::map<int, std::vector<bool>> masks = {
          {MuonSubdetId::DT, {false, false, false, false}},
          {MuonSubdetId::CSC, {false, false, false, false}},
          {MuonSubdetId::RPC, {false, false, false, false}},
      };

      const size_t first_station_index = getStationIndex(first_station, true);
      const size_t last_station_index = getStationIndex(last_station, true);
      unsigned cnt = 0;
      for (size_t n = first_station_index; n <= last_station_index; ++n) {
        for (const auto& hit : n_hits) {
          if (!masks.at(hit.first).at(n) && hit.second.at(n) > 0)
            ++cnt;
        }
      }
      return cnt;
    }

  private:
    CountMap n_matches, n_hits;
  };

  enum class CellObjectType {
    PfCand_electron,
    PfCand_muon,
    PfCand_chargedHadron,
    PfCand_neutralHadron,
    PfCand_gamma,
    Electron,
    Muon,
    Other
  };

  template <typename Object>
  CellObjectType GetCellObjectType(const Object&);
  template <>
  CellObjectType GetCellObjectType(const pat::Electron&) {
    return CellObjectType::Electron;
  }
  template <>
  CellObjectType GetCellObjectType(const pat::Muon&) {
    return CellObjectType::Muon;
  }

  template <>
  CellObjectType GetCellObjectType(reco::Candidate const& cand) {
    static const std::map<int, CellObjectType> obj_types = {{11, CellObjectType::PfCand_electron},
                                                            {13, CellObjectType::PfCand_muon},
                                                            {22, CellObjectType::PfCand_gamma},
                                                            {130, CellObjectType::PfCand_neutralHadron},
                                                            {211, CellObjectType::PfCand_chargedHadron}};

    auto iter = obj_types.find(std::abs(cand.pdgId()));
    if (iter == obj_types.end())
      return CellObjectType::Other;
    return iter->second;
  }

  using Cell = std::map<CellObjectType, size_t>;
  struct CellIndex {
    int eta, phi;

    bool operator<(const CellIndex& other) const {
      if (eta != other.eta)
        return eta < other.eta;
      return phi < other.phi;
    }
  };

  class CellGrid {
  public:
    using Map = std::map<CellIndex, Cell>;
    using const_iterator = Map::const_iterator;

    CellGrid(unsigned n_cells_eta,
             unsigned n_cells_phi,
             double cell_size_eta,
             double cell_size_phi,
             bool disable_CellIndex_workaround)
        : nCellsEta(n_cells_eta),
          nCellsPhi(n_cells_phi),
          nTotal(nCellsEta * nCellsPhi),
          cellSizeEta(cell_size_eta),
          cellSizePhi(cell_size_phi),
          disable_CellIndex_workaround_(disable_CellIndex_workaround) {
      if (nCellsEta % 2 != 1 || nCellsEta < 1)
        throw cms::Exception("DeepTauId") << "Invalid number of eta cells.";
      if (nCellsPhi % 2 != 1 || nCellsPhi < 1)
        throw cms::Exception("DeepTauId") << "Invalid number of phi cells.";
      if (cellSizeEta <= 0 || cellSizePhi <= 0)
        throw cms::Exception("DeepTauId") << "Invalid cell size.";
    }

    int maxEtaIndex() const { return static_cast<int>((nCellsEta - 1) / 2); }
    int maxPhiIndex() const { return static_cast<int>((nCellsPhi - 1) / 2); }
    double maxDeltaEta() const { return cellSizeEta * (0.5 + maxEtaIndex()); }
    double maxDeltaPhi() const { return cellSizePhi * (0.5 + maxPhiIndex()); }
    int getEtaTensorIndex(const CellIndex& cellIndex) const { return cellIndex.eta + maxEtaIndex(); }
    int getPhiTensorIndex(const CellIndex& cellIndex) const { return cellIndex.phi + maxPhiIndex(); }

    bool tryGetCellIndex(double deltaEta, double deltaPhi, CellIndex& cellIndex) const {
      const auto getCellIndex = [this](double x, double maxX, double size, int& index) {
        const double absX = std::abs(x);
        if (absX > maxX)
          return false;
        double absIndex;
        if (disable_CellIndex_workaround_) {
          // CV: use consistent definition for CellIndex
          //     in DeepTauId.cc code and new DeepTau trainings
          absIndex = std::floor(absX / size + 0.5);
        } else {
          // CV: backwards compatibility with DeepTau training v2p1 used during Run 2
          absIndex = std::floor(std::abs(absX / size - 0.5));
        }
        index = static_cast<int>(std::copysign(absIndex, x));
        return true;
      };

      return getCellIndex(deltaEta, maxDeltaEta(), cellSizeEta, cellIndex.eta) &&
             getCellIndex(deltaPhi, maxDeltaPhi(), cellSizePhi, cellIndex.phi);
    }

    size_t num_valid_cells() const { return cells.size(); }
    Cell& operator[](const CellIndex& cellIndex) { return cells[cellIndex]; }
    const Cell& at(const CellIndex& cellIndex) const { return cells.at(cellIndex); }
    size_t count(const CellIndex& cellIndex) const { return cells.count(cellIndex); }
    const_iterator find(const CellIndex& cellIndex) const { return cells.find(cellIndex); }
    const_iterator begin() const { return cells.begin(); }
    const_iterator end() const { return cells.end(); }

  public:
    const unsigned nCellsEta, nCellsPhi, nTotal;
    const double cellSizeEta, cellSizePhi;

  private:
    std::map<CellIndex, Cell> cells;
    const bool disable_CellIndex_workaround_;
  };
}  // anonymous namespace

using bd = deep_tau::DeepTauBase::BasicDiscriminator;
const std::map<bd, std::string> deep_tau::DeepTauBase::stringFromDiscriminator_{
    {bd::ChargedIsoPtSum, "ChargedIsoPtSum"},
    {bd::NeutralIsoPtSum, "NeutralIsoPtSum"},
    {bd::NeutralIsoPtSumWeight, "NeutralIsoPtSumWeight"},
    {bd::FootprintCorrection, "TauFootprintCorrection"},
    {bd::PhotonPtSumOutsideSignalCone, "PhotonPtSumOutsideSignalCone"},
    {bd::PUcorrPtSum, "PUcorrPtSum"}};
const std::vector<bd> deep_tau::DeepTauBase::requiredBasicDiscriminators_ = {bd::ChargedIsoPtSum,
                                                                             bd::NeutralIsoPtSum,
                                                                             bd::NeutralIsoPtSumWeight,
                                                                             bd::PhotonPtSumOutsideSignalCone,
                                                                             bd::PUcorrPtSum};
const std::vector<bd> deep_tau::DeepTauBase::requiredBasicDiscriminatorsdR03_ = {bd::ChargedIsoPtSum,
                                                                                 bd::NeutralIsoPtSum,
                                                                                 bd::NeutralIsoPtSumWeight,
                                                                                 bd::PhotonPtSumOutsideSignalCone,
                                                                                 bd::FootprintCorrection};

class DeepTauId : public deep_tau::DeepTauBase {
public:
  static constexpr float default_value = -999.;

  static const OutputCollection& GetOutputs() {
    static constexpr size_t e_index = 0, mu_index = 1, tau_index = 2, jet_index = 3;
    static const OutputCollection outputs_ = {
        {"VSe", Output({tau_index}, {e_index, tau_index})},
        {"VSmu", Output({tau_index}, {mu_index, tau_index})},
        {"VSjet", Output({tau_index}, {jet_index, tau_index})},
    };
    return outputs_;
  }

  const std::map<BasicDiscriminator, size_t> matchDiscriminatorIndices(
      edm::Event& event,
      edm::EDGetTokenT<reco::TauDiscriminatorContainer> discriminatorContainerToken,
      std::vector<BasicDiscriminator> requiredDiscr) {
    std::map<std::string, size_t> discrIndexMapStr;
    auto const aHandle = event.getHandle(discriminatorContainerToken);
    auto const aProv = aHandle.provenance();
    if (aProv == nullptr)
      aHandle.whyFailed()->raise();
    const auto& psetsFromProvenance = edm::parameterSet(aProv->stable(), event.processHistory());
    auto const idlist = psetsFromProvenance.getParameter<std::vector<edm::ParameterSet>>("IDdefinitions");
    for (size_t j = 0; j < idlist.size(); ++j) {
      std::string idname = idlist[j].getParameter<std::string>("IDname");
      if (discrIndexMapStr.count(idname)) {
        throw cms::Exception("DeepTauId")
            << "basic discriminator " << idname << " appears more than once in the input.";
      }
      discrIndexMapStr[idname] = j;
    }

    //translate to a map of <BasicDiscriminator, index> and check if all discriminators are present
    std::map<BasicDiscriminator, size_t> discrIndexMap;
    for (size_t i = 0; i < requiredDiscr.size(); i++) {
      if (discrIndexMapStr.find(stringFromDiscriminator_.at(requiredDiscr[i])) == discrIndexMapStr.end())
        throw cms::Exception("DeepTauId") << "Basic Discriminator " << stringFromDiscriminator_.at(requiredDiscr[i])
                                          << " was not provided in the config file.";
      else
        discrIndexMap[requiredDiscr[i]] = discrIndexMapStr[stringFromDiscriminator_.at(requiredDiscr[i])];
    }
    return discrIndexMap;
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("electrons", edm::InputTag("slimmedElectrons"));
    desc.add<edm::InputTag>("muons", edm::InputTag("slimmedMuons"));
    desc.add<edm::InputTag>("taus", edm::InputTag("slimmedTaus"));
    desc.add<edm::InputTag>("pfcands", edm::InputTag("packedPFCandidates"));
    desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
    desc.add<edm::InputTag>("rho", edm::InputTag("fixedGridRhoAll"));
    desc.add<std::vector<std::string>>("graph_file",
                                       {"RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6.pb"});
    desc.add<bool>("mem_mapped", false);
    desc.add<unsigned>("version", 2);
    desc.add<unsigned>("sub_version", 1);
    desc.add<int>("debug_level", 0);
    desc.add<bool>("disable_dxy_pca", false);
    desc.add<bool>("disable_hcalFraction_workaround", false);
    desc.add<bool>("disable_CellIndex_workaround", false);
    desc.add<bool>("save_inputs", false);
    desc.add<bool>("is_online", false);

    desc.add<std::vector<std::string>>("VSeWP", {"-1."});
    desc.add<std::vector<std::string>>("VSmuWP", {"-1."});
    desc.add<std::vector<std::string>>("VSjetWP", {"-1."});

    desc.addUntracked<edm::InputTag>("basicTauDiscriminators", edm::InputTag("basicTauDiscriminators"));
    desc.addUntracked<edm::InputTag>("basicTauDiscriminatorsdR03", edm::InputTag("basicTauDiscriminatorsdR03"));
    desc.add<edm::InputTag>("pfTauTransverseImpactParameters", edm::InputTag("hpsPFTauTransverseImpactParameters"));

    {
      edm::ParameterSetDescription pset_Prediscriminants;
      pset_Prediscriminants.add<std::string>("BooleanOperator", "and");
      {
        edm::ParameterSetDescription psd1;
        psd1.add<double>("cut");
        psd1.add<edm::InputTag>("Producer");
        pset_Prediscriminants.addOptional<edm::ParameterSetDescription>("decayMode", psd1);
      }
      desc.add<edm::ParameterSetDescription>("Prediscriminants", pset_Prediscriminants);
    }

    descriptions.add("DeepTau", desc);
  }

public:
  explicit DeepTauId(const edm::ParameterSet& cfg, const deep_tau::DeepTauCache* cache)
      : DeepTauBase(cfg, GetOutputs(), cache),
        electrons_token_(consumes<std::vector<pat::Electron>>(cfg.getParameter<edm::InputTag>("electrons"))),
        muons_token_(consumes<std::vector<pat::Muon>>(cfg.getParameter<edm::InputTag>("muons"))),
        rho_token_(consumes<double>(cfg.getParameter<edm::InputTag>("rho"))),
        basicTauDiscriminators_inputToken_(consumes<reco::TauDiscriminatorContainer>(
            cfg.getUntrackedParameter<edm::InputTag>("basicTauDiscriminators"))),
        basicTauDiscriminatorsdR03_inputToken_(consumes<reco::TauDiscriminatorContainer>(
            cfg.getUntrackedParameter<edm::InputTag>("basicTauDiscriminatorsdR03"))),
        pfTauTransverseImpactParameters_token_(
            consumes<edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>>(
                cfg.getParameter<edm::InputTag>("pfTauTransverseImpactParameters"))),
        version_(cfg.getParameter<unsigned>("version")),
        sub_version_(cfg.getParameter<unsigned>("sub_version")),
        debug_level(cfg.getParameter<int>("debug_level")),
        disable_dxy_pca_(cfg.getParameter<bool>("disable_dxy_pca")),
        disable_hcalFraction_workaround_(cfg.getParameter<bool>("disable_hcalFraction_workaround")),
        disable_CellIndex_workaround_(cfg.getParameter<bool>("disable_CellIndex_workaround")),
        save_inputs_(cfg.getParameter<bool>("save_inputs")),
        json_file_(nullptr),
        file_counter_(0) {
    if (version_ == 1) {
      input_layer_ = cache_->getGraph().node(0).name();
      output_layer_ = cache_->getGraph().node(cache_->getGraph().node_size() - 1).name();
      const auto& shape = cache_->getGraph().node(0).attr().at("shape").shape();
      if (shape.dim(1).size() != dnn_inputs_v1::NumberOfInputs)
        throw cms::Exception("DeepTauId")
            << "number of inputs does not match the expected inputs for the given version";
    } else if (version_ == 2) {
      using namespace dnn_inputs_v2;
      namespace sc = deep_tau::Scaling;
      tauInputs_indices_.resize(TauBlockInputs::NumberOfInputs);
      std::iota(std::begin(tauInputs_indices_), std::end(tauInputs_indices_), 0);

      if (sub_version_ == 1) {
        tauBlockTensor_ = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT, tensorflow::TensorShape{1, TauBlockInputs::NumberOfInputs});
        scalingParamsMap_ = &sc::scalingParamsMap_v2p1;
      } else if (sub_version_ == 5) {
        std::sort(TauBlockInputs::varsToDrop.begin(), TauBlockInputs::varsToDrop.end());
        for (auto v : TauBlockInputs::varsToDrop) {
          tauInputs_indices_.at(v) = -1;  // set index to -1
          for (std::size_t i = v + 1; i < TauBlockInputs::NumberOfInputs; ++i)
            tauInputs_indices_.at(i) -= 1;  // shift all the following indices by 1
        }
        tauBlockTensor_ = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT,
            tensorflow::TensorShape{1,
                                    static_cast<int>(TauBlockInputs::NumberOfInputs) -
                                        static_cast<int>(TauBlockInputs::varsToDrop.size())});
        scalingParamsMap_ = &sc::scalingParamsMap_v2p5;
      } else
        throw cms::Exception("DeepTauId") << "subversion " << sub_version_ << " is not supported.";

      std::map<std::vector<bool>, std::vector<sc::FeatureT>> GridFeatureTypes_map = {
          {{false}, {sc::FeatureT::TauFlat, sc::FeatureT::GridGlobal}},  // feature types without inner/outer grid split
          {{false, true},
           {sc::FeatureT::PfCand_electron,
            sc::FeatureT::PfCand_muon,  // feature types with inner/outer grid split
            sc::FeatureT::PfCand_chHad,
            sc::FeatureT::PfCand_nHad,
            sc::FeatureT::PfCand_gamma,
            sc::FeatureT::Electron,
            sc::FeatureT::Muon}}};

      // check that sizes of mean/std/lim_min/lim_max vectors are equal between each other
      for (const auto& p : GridFeatureTypes_map) {
        for (auto is_inner : p.first) {
          for (auto featureType : p.second) {
            const sc::ScalingParams& sp = scalingParamsMap_->at(std::make_pair(featureType, is_inner));
            if (!(sp.mean_.size() == sp.std_.size() && sp.mean_.size() == sp.lim_min_.size() &&
                  sp.mean_.size() == sp.lim_max_.size()))
              throw cms::Exception("DeepTauId") << "sizes of scaling parameter vectors do not match between each other";
          }
        }
      }

      for (size_t n = 0; n < 2; ++n) {
        const bool is_inner = n == 0;
        const auto n_cells = is_inner ? number_of_inner_cell : number_of_outer_cell;
        eGammaTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT, tensorflow::TensorShape{1, 1, 1, EgammaBlockInputs::NumberOfInputs});
        muonTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT, tensorflow::TensorShape{1, 1, 1, MuonBlockInputs::NumberOfInputs});
        hadronsTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT, tensorflow::TensorShape{1, 1, 1, HadronBlockInputs::NumberOfInputs});
        convTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT, tensorflow::TensorShape{1, n_cells, n_cells, number_of_conv_features});
        zeroOutputTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT, tensorflow::TensorShape{1, 1, 1, number_of_conv_features});

        eGammaTensor_[is_inner]->flat<float>().setZero();
        muonTensor_[is_inner]->flat<float>().setZero();
        hadronsTensor_[is_inner]->flat<float>().setZero();

        setCellConvFeatures(*zeroOutputTensor_[is_inner], getPartialPredictions(is_inner), 0, 0, 0);
      }
    } else {
      throw cms::Exception("DeepTauId") << "version " << version_ << " is not supported.";
    }
  }

  static std::unique_ptr<deep_tau::DeepTauCache> initializeGlobalCache(const edm::ParameterSet& cfg) {
    return DeepTauBase::initializeGlobalCache(cfg);
  }

  static void globalEndJob(const deep_tau::DeepTauCache* cache_) { return DeepTauBase::globalEndJob(cache_); }

private:
  static constexpr float pi = M_PI;

  template <typename T>
  static float getValue(T value) {
    return std::isnormal(value) ? static_cast<float>(value) : 0.f;
  }

  template <typename T>
  static float getValueLinear(T value, float min_value, float max_value, bool positive) {
    const float fixed_value = getValue(value);
    const float clamped_value = std::clamp(fixed_value, min_value, max_value);
    float transformed_value = (clamped_value - min_value) / (max_value - min_value);
    if (!positive)
      transformed_value = transformed_value * 2 - 1;
    return transformed_value;
  }

  template <typename T>
  static float getValueNorm(T value, float mean, float sigma, float n_sigmas_max = 5) {
    const float fixed_value = getValue(value);
    const float norm_value = (fixed_value - mean) / sigma;
    return std::clamp(norm_value, -n_sigmas_max, n_sigmas_max);
  }

  static bool isAbove(double value, double min) { return std::isnormal(value) && value > min; }

  static bool calculateElectronClusterVarsV2(const pat::Electron& ele,
                                             float& cc_ele_energy,
                                             float& cc_gamma_energy,
                                             int& cc_n_gamma) {
    cc_ele_energy = cc_gamma_energy = 0;
    cc_n_gamma = 0;
    const auto& superCluster = ele.superCluster();
    if (superCluster.isNonnull() && superCluster.isAvailable() && superCluster->clusters().isNonnull() &&
        superCluster->clusters().isAvailable()) {
      for (auto iter = superCluster->clustersBegin(); iter != superCluster->clustersEnd(); ++iter) {
        const float energy = static_cast<float>((*iter)->energy());
        if (iter == superCluster->clustersBegin())
          cc_ele_energy += energy;
        else {
          cc_gamma_energy += energy;
          ++cc_n_gamma;
        }
      }
      return true;
    } else
      return false;
  }

  inline void checkInputs(const tensorflow::Tensor& inputs,
                          const std::string& block_name,
                          int n_inputs,
                          const CellGrid* grid = nullptr) const {
    if (debug_level >= 1) {
      std::cout << "<checkInputs>: block_name = " << block_name << std::endl;
      if (block_name == "input_tau") {
        for (int input_index = 0; input_index < n_inputs; ++input_index) {
          float input = inputs.matrix<float>()(0, input_index);
          if (edm::isNotFinite(input)) {
            throw cms::Exception("DeepTauId")
                << "in the " << block_name
                << ", input is not finite, i.e. infinite or NaN, for input_index = " << input_index;
          }
          if (debug_level >= 2) {
            std::cout << block_name << "[var = " << input_index << "] = " << std::setprecision(5) << std::fixed << input
                      << std::endl;
          }
        }
      } else {
        assert(grid);
        int n_eta, n_phi;
        if (block_name.find("input_inner") != std::string::npos) {
          n_eta = 5;
          n_phi = 5;
        } else if (block_name.find("input_outer") != std::string::npos) {
          n_eta = 10;
          n_phi = 10;
        } else
          assert(0);
        int eta_phi_index = 0;
        for (int eta = -n_eta; eta <= n_eta; ++eta) {
          for (int phi = -n_phi; phi <= n_phi; ++phi) {
            const CellIndex cell_index{eta, phi};
            const auto cell_iter = grid->find(cell_index);
            if (cell_iter != grid->end()) {
              for (int input_index = 0; input_index < n_inputs; ++input_index) {
                float input = inputs.tensor<float, 4>()(eta_phi_index, 0, 0, input_index);
                if (edm::isNotFinite(input)) {
                  throw cms::Exception("DeepTauId")
                      << "in the " << block_name << ", input is not finite, i.e. infinite or NaN, for eta = " << eta
                      << ", phi = " << phi << ", input_index = " << input_index;
                }
                if (debug_level >= 2) {
                  std::cout << block_name << "[eta = " << eta << "][phi = " << phi << "][var = " << input_index
                            << "] = " << std::setprecision(5) << std::fixed << input << std::endl;
                }
              }
              eta_phi_index += 1;
            }
          }
        }
      }
    }
  }

  inline void saveInputs(const tensorflow::Tensor& inputs,
                         const std::string& block_name,
                         int n_inputs,
                         const CellGrid* grid = nullptr) {
    if (debug_level >= 1) {
      std::cout << "<saveInputs>: block_name = " << block_name << std::endl;
    }
    if (!is_first_block_)
      (*json_file_) << ", ";
    (*json_file_) << "\"" << block_name << "\": [";
    if (block_name == "input_tau") {
      for (int input_index = 0; input_index < n_inputs; ++input_index) {
        float input = inputs.matrix<float>()(0, input_index);
        if (input_index != 0)
          (*json_file_) << ", ";
        (*json_file_) << input;
      }
    } else {
      assert(grid);
      int n_eta, n_phi;
      if (block_name.find("input_inner") != std::string::npos) {
        n_eta = 5;
        n_phi = 5;
      } else if (block_name.find("input_outer") != std::string::npos) {
        n_eta = 10;
        n_phi = 10;
      } else
        assert(0);
      int eta_phi_index = 0;
      for (int eta = -n_eta; eta <= n_eta; ++eta) {
        if (eta != -n_eta)
          (*json_file_) << ", ";
        (*json_file_) << "[";
        for (int phi = -n_phi; phi <= n_phi; ++phi) {
          if (phi != -n_phi)
            (*json_file_) << ", ";
          (*json_file_) << "[";
          const CellIndex cell_index{eta, phi};
          const auto cell_iter = grid->find(cell_index);
          for (int input_index = 0; input_index < n_inputs; ++input_index) {
            float input = 0.;
            if (cell_iter != grid->end()) {
              input = inputs.tensor<float, 4>()(eta_phi_index, 0, 0, input_index);
            }
            if (input_index != 0)
              (*json_file_) << ", ";
            (*json_file_) << input;
          }
          if (cell_iter != grid->end()) {
            eta_phi_index += 1;
          }
          (*json_file_) << "]";
        }
        (*json_file_) << "]";
      }
    }
    (*json_file_) << "]";
    is_first_block_ = false;
  }

private:
  tensorflow::Tensor getPredictions(edm::Event& event, edm::Handle<TauCollection> taus) override {
    // Empty dummy vectors
    const std::vector<pat::Electron> electron_collection_default;
    const std::vector<pat::Muon> muon_collection_default;
    const reco::TauDiscriminatorContainer basicTauDiscriminators_default;
    const reco::TauDiscriminatorContainer basicTauDiscriminatorsdR03_default;
    const edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>
        pfTauTransverseImpactParameters_default;

    const std::vector<pat::Electron>* electron_collection;
    const std::vector<pat::Muon>* muon_collection;
    const reco::TauDiscriminatorContainer* basicTauDiscriminators;
    const reco::TauDiscriminatorContainer* basicTauDiscriminatorsdR03;
    const edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>*
        pfTauTransverseImpactParameters;

    if (!is_online_) {
      electron_collection = &event.get(electrons_token_);
      muon_collection = &event.get(muons_token_);
      pfTauTransverseImpactParameters = &pfTauTransverseImpactParameters_default;
      basicTauDiscriminators = &basicTauDiscriminators_default;
      basicTauDiscriminatorsdR03 = &basicTauDiscriminatorsdR03_default;
    } else {
      electron_collection = &electron_collection_default;
      muon_collection = &muon_collection_default;
      pfTauTransverseImpactParameters = &event.get(pfTauTransverseImpactParameters_token_);
      basicTauDiscriminators = &event.get(basicTauDiscriminators_inputToken_);
      basicTauDiscriminatorsdR03 = &event.get(basicTauDiscriminatorsdR03_inputToken_);

      // Get indices for discriminators
      if (!discrIndicesMapped_) {
        basicDiscrIndexMap_ =
            matchDiscriminatorIndices(event, basicTauDiscriminators_inputToken_, requiredBasicDiscriminators_);
        basicDiscrdR03IndexMap_ =
            matchDiscriminatorIndices(event, basicTauDiscriminatorsdR03_inputToken_, requiredBasicDiscriminatorsdR03_);
        discrIndicesMapped_ = true;
      }
    }

    TauFunc tauIDs = {basicTauDiscriminators,
                      basicTauDiscriminatorsdR03,
                      pfTauTransverseImpactParameters,
                      basicDiscrIndexMap_,
                      basicDiscrdR03IndexMap_};

    edm::Handle<edm::View<reco::Candidate>> pfCands;
    event.getByToken(pfcandToken_, pfCands);

    edm::Handle<reco::VertexCollection> vertices;
    event.getByToken(vtxToken_, vertices);

    edm::Handle<double> rho;
    event.getByToken(rho_token_, rho);

    tensorflow::Tensor predictions(tensorflow::DT_FLOAT, {static_cast<int>(taus->size()), deep_tau::NumberOfOutputs});

    for (size_t tau_index = 0; tau_index < taus->size(); ++tau_index) {
      const edm::RefToBase<reco::BaseTau> tauRef = taus->refAt(tau_index);

      std::vector<tensorflow::Tensor> pred_vector;

      bool passesPrediscriminants;
      if (is_online_) {
        passesPrediscriminants = tauIDs.passPrediscriminants<std::vector<TauDiscInfo<reco::PFTauDiscriminator>>>(
            recoPrediscriminants_, andPrediscriminants_, tauRef);
      } else {
        passesPrediscriminants = tauIDs.passPrediscriminants<std::vector<TauDiscInfo<pat::PATTauDiscriminator>>>(
            patPrediscriminants_, andPrediscriminants_, tauRef);
      }

      if (passesPrediscriminants) {
        if (version_ == 1) {
          if (is_online_)
            getPredictionsV1<reco::PFCandidate, reco::PFTau>(
                taus->at(tau_index), tau_index, tauRef, electron_collection, muon_collection, pred_vector, tauIDs);
          else
            getPredictionsV1<pat::PackedCandidate, pat::Tau>(
                taus->at(tau_index), tau_index, tauRef, electron_collection, muon_collection, pred_vector, tauIDs);
        } else if (version_ == 2) {
          if (is_online_) {
            getPredictionsV2<reco::PFCandidate, reco::PFTau>(taus->at(tau_index),
                                                             tau_index,
                                                             tauRef,
                                                             electron_collection,
                                                             muon_collection,
                                                             *pfCands,
                                                             vertices->at(0),
                                                             *rho,
                                                             pred_vector,
                                                             tauIDs);
          } else
            getPredictionsV2<pat::PackedCandidate, pat::Tau>(taus->at(tau_index),
                                                             tau_index,
                                                             tauRef,
                                                             electron_collection,
                                                             muon_collection,
                                                             *pfCands,
                                                             vertices->at(0),
                                                             *rho,
                                                             pred_vector,
                                                             tauIDs);
        } else {
          throw cms::Exception("DeepTauId") << "version " << version_ << " is not supported.";
        }

        for (int k = 0; k < deep_tau::NumberOfOutputs; ++k) {
          const float pred = pred_vector[0].flat<float>()(k);
          if (!(pred >= 0 && pred <= 1))
            throw cms::Exception("DeepTauId")
                << "invalid prediction = " << pred << " for tau_index = " << tau_index << ", pred_index = " << k;
          predictions.matrix<float>()(tau_index, k) = pred;
        }
      } else {
        // This else statement was added as a part of the DeepTau@HLT development. It does not affect the current state
        // of offline DeepTauId code as there the preselection is not used (it was added in the DeepTau@HLT). It returns
        // default values for deepTau score if the preselection failed. Before this statement the values given for this tau
        // were random. k == 2 corresponds to the tau score and all other k values to e, mu and jets. By defining in this way
        // the final score is -1.
        for (int k = 0; k < deep_tau::NumberOfOutputs; ++k) {
          predictions.matrix<float>()(tau_index, k) = (k == 2) ? -1.f : 2.f;
        }
      }
    }
    return predictions;
  }

  template <typename CandidateCastType, typename TauCastType>
  void getPredictionsV1(TauCollection::const_reference& tau,
                        const size_t tau_index,
                        const edm::RefToBase<reco::BaseTau> tau_ref,
                        const std::vector<pat::Electron>* electrons,
                        const std::vector<pat::Muon>* muons,
                        std::vector<tensorflow::Tensor>& pred_vector,
                        TauFunc tau_funcs) {
    const tensorflow::Tensor& inputs = createInputsV1<dnn_inputs_v1, const CandidateCastType>(
        dynamic_cast<const TauCastType&>(tau), tau_index, tau_ref, electrons, muons, tau_funcs);
    tensorflow::run(&(cache_->getSession()), {{input_layer_, inputs}}, {output_layer_}, &pred_vector);
  }

  template <typename CandidateCastType, typename TauCastType>
  void getPredictionsV2(TauCollection::const_reference& tau,
                        const size_t tau_index,
                        const edm::RefToBase<reco::BaseTau> tau_ref,
                        const std::vector<pat::Electron>* electrons,
                        const std::vector<pat::Muon>* muons,
                        const edm::View<reco::Candidate>& pfCands,
                        const reco::Vertex& pv,
                        double rho,
                        std::vector<tensorflow::Tensor>& pred_vector,
                        TauFunc tau_funcs) {
    using namespace dnn_inputs_v2;
    if (debug_level >= 2) {
      std::cout << "<DeepTauId::getPredictionsV2 (moduleLabel = " << moduleDescription().moduleLabel()
                << ")>:" << std::endl;
      std::cout << " tau: pT = " << tau.pt() << ", eta = " << tau.eta() << ", phi = " << tau.phi() << std::endl;
    }
    CellGrid inner_grid(number_of_inner_cell, number_of_inner_cell, 0.02, 0.02, disable_CellIndex_workaround_);
    CellGrid outer_grid(number_of_outer_cell, number_of_outer_cell, 0.05, 0.05, disable_CellIndex_workaround_);
    fillGrids(dynamic_cast<const TauCastType&>(tau), *electrons, inner_grid, outer_grid);
    fillGrids(dynamic_cast<const TauCastType&>(tau), *muons, inner_grid, outer_grid);
    fillGrids(dynamic_cast<const TauCastType&>(tau), pfCands, inner_grid, outer_grid);

    createTauBlockInputs<CandidateCastType>(
        dynamic_cast<const TauCastType&>(tau), tau_index, tau_ref, pv, rho, tau_funcs);
    checkInputs(*tauBlockTensor_, "input_tau", static_cast<int>(tauBlockTensor_->shape().dim_size(1)));
    createConvFeatures<CandidateCastType>(dynamic_cast<const TauCastType&>(tau),
                                          tau_index,
                                          tau_ref,
                                          pv,
                                          rho,
                                          electrons,
                                          muons,
                                          pfCands,
                                          inner_grid,
                                          tau_funcs,
                                          true);
    checkInputs(*eGammaTensor_[true], "input_inner_egamma", EgammaBlockInputs::NumberOfInputs, &inner_grid);
    checkInputs(*muonTensor_[true], "input_inner_muon", MuonBlockInputs::NumberOfInputs, &inner_grid);
    checkInputs(*hadronsTensor_[true], "input_inner_hadrons", HadronBlockInputs::NumberOfInputs, &inner_grid);
    createConvFeatures<CandidateCastType>(dynamic_cast<const TauCastType&>(tau),
                                          tau_index,
                                          tau_ref,
                                          pv,
                                          rho,
                                          electrons,
                                          muons,
                                          pfCands,
                                          outer_grid,
                                          tau_funcs,
                                          false);
    checkInputs(*eGammaTensor_[false], "input_outer_egamma", EgammaBlockInputs::NumberOfInputs, &outer_grid);
    checkInputs(*muonTensor_[false], "input_outer_muon", MuonBlockInputs::NumberOfInputs, &outer_grid);
    checkInputs(*hadronsTensor_[false], "input_outer_hadrons", HadronBlockInputs::NumberOfInputs, &outer_grid);

    if (save_inputs_) {
      std::string json_file_name = "DeepTauId_" + std::to_string(file_counter_) + ".json";
      json_file_ = new std::ofstream(json_file_name.data());
      is_first_block_ = true;
      (*json_file_) << "{";
      saveInputs(*tauBlockTensor_, "input_tau", static_cast<int>(tauBlockTensor_->shape().dim_size(1)));
      saveInputs(
          *eGammaTensor_[true], "input_inner_egamma", dnn_inputs_v2::EgammaBlockInputs::NumberOfInputs, &inner_grid);
      saveInputs(*muonTensor_[true], "input_inner_muon", dnn_inputs_v2::MuonBlockInputs::NumberOfInputs, &inner_grid);
      saveInputs(
          *hadronsTensor_[true], "input_inner_hadrons", dnn_inputs_v2::HadronBlockInputs::NumberOfInputs, &inner_grid);
      saveInputs(
          *eGammaTensor_[false], "input_outer_egamma", dnn_inputs_v2::EgammaBlockInputs::NumberOfInputs, &outer_grid);
      saveInputs(*muonTensor_[false], "input_outer_muon", dnn_inputs_v2::MuonBlockInputs::NumberOfInputs, &outer_grid);
      saveInputs(
          *hadronsTensor_[false], "input_outer_hadrons", dnn_inputs_v2::HadronBlockInputs::NumberOfInputs, &outer_grid);
      (*json_file_) << "}";
      delete json_file_;
      ++file_counter_;
    }

    tensorflow::run(&(cache_->getSession("core")),
                    {{"input_tau", *tauBlockTensor_},
                     {"input_inner", *convTensor_.at(true)},
                     {"input_outer", *convTensor_.at(false)}},
                    {"main_output/Softmax"},
                    &pred_vector);
    if (debug_level >= 1) {
      std::cout << "output = { ";
      for (int idx = 0; idx < deep_tau::NumberOfOutputs; ++idx) {
        if (idx > 0)
          std::cout << ", ";
        std::string label;
        if (idx == 0)
          label = "e";
        else if (idx == 1)
          label = "mu";
        else if (idx == 2)
          label = "tau";
        else if (idx == 3)
          label = "jet";
        else
          assert(0);
        std::cout << label << " = " << pred_vector[0].flat<float>()(idx);
      }
      std::cout << " }" << std::endl;
    }
  }

  template <typename Collection, typename TauCastType>
  void fillGrids(const TauCastType& tau, const Collection& objects, CellGrid& inner_grid, CellGrid& outer_grid) {
    static constexpr double outer_dR2 = 0.25;  //0.5^2
    const double inner_radius = getInnerSignalConeRadius(tau.polarP4().pt());
    const double inner_dR2 = std::pow(inner_radius, 2);

    const auto addObject = [&](size_t n, double deta, double dphi, CellGrid& grid) {
      const auto& obj = objects.at(n);
      const CellObjectType obj_type = GetCellObjectType(obj);
      if (obj_type == CellObjectType::Other)
        return;
      CellIndex cell_index;
      if (grid.tryGetCellIndex(deta, dphi, cell_index)) {
        Cell& cell = grid[cell_index];
        auto iter = cell.find(obj_type);
        if (iter != cell.end()) {
          const auto& prev_obj = objects.at(iter->second);
          if (obj.polarP4().pt() > prev_obj.polarP4().pt())
            iter->second = n;
        } else {
          cell[obj_type] = n;
        }
      }
    };

    for (size_t n = 0; n < objects.size(); ++n) {
      const auto& obj = objects.at(n);
      const double deta = obj.polarP4().eta() - tau.polarP4().eta();
      const double dphi = reco::deltaPhi(obj.polarP4().phi(), tau.polarP4().phi());
      const double dR2 = std::pow(deta, 2) + std::pow(dphi, 2);
      if (dR2 < inner_dR2)
        addObject(n, deta, dphi, inner_grid);
      if (dR2 < outer_dR2)
        addObject(n, deta, dphi, outer_grid);
    }
  }

  tensorflow::Tensor getPartialPredictions(bool is_inner) {
    std::vector<tensorflow::Tensor> pred_vector;
    if (is_inner) {
      tensorflow::run(&(cache_->getSession("inner")),
                      {
                          {"input_inner_egamma", *eGammaTensor_.at(is_inner)},
                          {"input_inner_muon", *muonTensor_.at(is_inner)},
                          {"input_inner_hadrons", *hadronsTensor_.at(is_inner)},
                      },
                      {"inner_all_dropout_4/Identity"},
                      &pred_vector);
    } else {
      tensorflow::run(&(cache_->getSession("outer")),
                      {
                          {"input_outer_egamma", *eGammaTensor_.at(is_inner)},
                          {"input_outer_muon", *muonTensor_.at(is_inner)},
                          {"input_outer_hadrons", *hadronsTensor_.at(is_inner)},
                      },
                      {"outer_all_dropout_4/Identity"},
                      &pred_vector);
    }
    return pred_vector.at(0);
  }

  template <typename CandidateCastType, typename TauCastType>
  void createConvFeatures(const TauCastType& tau,
                          const size_t tau_index,
                          const edm::RefToBase<reco::BaseTau> tau_ref,
                          const reco::Vertex& pv,
                          double rho,
                          const std::vector<pat::Electron>* electrons,
                          const std::vector<pat::Muon>* muons,
                          const edm::View<reco::Candidate>& pfCands,
                          const CellGrid& grid,
                          TauFunc tau_funcs,
                          bool is_inner) {
    if (debug_level >= 2) {
      std::cout << "<DeepTauId::createConvFeatures (is_inner = " << is_inner << ")>:" << std::endl;
    }
    tensorflow::Tensor& convTensor = *convTensor_.at(is_inner);
    eGammaTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape{
            (long long int)grid.num_valid_cells(), 1, 1, dnn_inputs_v2::EgammaBlockInputs::NumberOfInputs});
    muonTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape{
            (long long int)grid.num_valid_cells(), 1, 1, dnn_inputs_v2::MuonBlockInputs::NumberOfInputs});
    hadronsTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape{
            (long long int)grid.num_valid_cells(), 1, 1, dnn_inputs_v2::HadronBlockInputs::NumberOfInputs});

    eGammaTensor_[is_inner]->flat<float>().setZero();
    muonTensor_[is_inner]->flat<float>().setZero();
    hadronsTensor_[is_inner]->flat<float>().setZero();

    unsigned idx = 0;
    for (int eta = -grid.maxEtaIndex(); eta <= grid.maxEtaIndex(); ++eta) {
      for (int phi = -grid.maxPhiIndex(); phi <= grid.maxPhiIndex(); ++phi) {
        if (debug_level >= 2) {
          std::cout << "processing ( eta = " << eta << ", phi = " << phi << " )" << std::endl;
        }
        const CellIndex cell_index{eta, phi};
        const auto cell_iter = grid.find(cell_index);
        if (cell_iter != grid.end()) {
          if (debug_level >= 2) {
            std::cout << " creating inputs for ( eta = " << eta << ", phi = " << phi << " ): idx = " << idx
                      << std::endl;
          }
          const Cell& cell = cell_iter->second;
          createEgammaBlockInputs<CandidateCastType>(
              idx, tau, tau_index, tau_ref, pv, rho, electrons, pfCands, cell, tau_funcs, is_inner);
          createMuonBlockInputs<CandidateCastType>(
              idx, tau, tau_index, tau_ref, pv, rho, muons, pfCands, cell, tau_funcs, is_inner);
          createHadronsBlockInputs<CandidateCastType>(
              idx, tau, tau_index, tau_ref, pv, rho, pfCands, cell, tau_funcs, is_inner);
          idx += 1;
        } else {
          if (debug_level >= 2) {
            std::cout << " skipping creation of inputs, because ( eta = " << eta << ", phi = " << phi
                      << " ) is not in the grid !!" << std::endl;
          }
        }
      }
    }

    const auto predTensor = getPartialPredictions(is_inner);
    idx = 0;
    for (int eta = -grid.maxEtaIndex(); eta <= grid.maxEtaIndex(); ++eta) {
      for (int phi = -grid.maxPhiIndex(); phi <= grid.maxPhiIndex(); ++phi) {
        const CellIndex cell_index{eta, phi};
        const int eta_index = grid.getEtaTensorIndex(cell_index);
        const int phi_index = grid.getPhiTensorIndex(cell_index);

        const auto cell_iter = grid.find(cell_index);
        if (cell_iter != grid.end()) {
          setCellConvFeatures(convTensor, predTensor, idx, eta_index, phi_index);
          idx += 1;
        } else {
          setCellConvFeatures(convTensor, *zeroOutputTensor_[is_inner], 0, eta_index, phi_index);
        }
      }
    }
  }

  void setCellConvFeatures(tensorflow::Tensor& convTensor,
                           const tensorflow::Tensor& features,
                           unsigned batch_idx,
                           int eta_index,
                           int phi_index) {
    for (int n = 0; n < dnn_inputs_v2::number_of_conv_features; ++n) {
      convTensor.tensor<float, 4>()(0, eta_index, phi_index, n) = features.tensor<float, 4>()(batch_idx, 0, 0, n);
    }
  }

  template <typename CandidateCastType, typename TauCastType>
  void createTauBlockInputs(const TauCastType& tau,
                            const size_t& tau_index,
                            const edm::RefToBase<reco::BaseTau> tau_ref,
                            const reco::Vertex& pv,
                            double rho,
                            TauFunc tau_funcs) {
    namespace dnn = dnn_inputs_v2::TauBlockInputs;
    namespace sc = deep_tau::Scaling;
    sc::FeatureT ft = sc::FeatureT::TauFlat;
    const sc::ScalingParams& sp = scalingParamsMap_->at(std::make_pair(ft, false));

    tensorflow::Tensor& inputs = *tauBlockTensor_;
    inputs.flat<float>().setZero();

    const auto& get = [&](int var_index) -> float& {
      return inputs.matrix<float>()(0, tauInputs_indices_.at(var_index));
    };

    auto leadChargedHadrCand = dynamic_cast<const CandidateCastType*>(tau.leadChargedHadrCand().get());

    get(dnn::rho) = sp.scale(rho, tauInputs_indices_[dnn::rho]);
    get(dnn::tau_pt) = sp.scale(tau.polarP4().pt(), tauInputs_indices_[dnn::tau_pt]);
    get(dnn::tau_eta) = sp.scale(tau.polarP4().eta(), tauInputs_indices_[dnn::tau_eta]);
    if (sub_version_ == 1) {
      get(dnn::tau_phi) = getValueLinear(tau.polarP4().phi(), -pi, pi, false);
    }
    get(dnn::tau_mass) = sp.scale(tau.polarP4().mass(), tauInputs_indices_[dnn::tau_mass]);
    get(dnn::tau_E_over_pt) = sp.scale(tau.p4().energy() / tau.p4().pt(), tauInputs_indices_[dnn::tau_E_over_pt]);
    get(dnn::tau_charge) = sp.scale(tau.charge(), tauInputs_indices_[dnn::tau_charge]);
    get(dnn::tau_n_charged_prongs) = sp.scale(tau.decayMode() / 5 + 1, tauInputs_indices_[dnn::tau_n_charged_prongs]);
    get(dnn::tau_n_neutral_prongs) = sp.scale(tau.decayMode() % 5, tauInputs_indices_[dnn::tau_n_neutral_prongs]);
    get(dnn::chargedIsoPtSum) =
        sp.scale(tau_funcs.getChargedIsoPtSum(tau, tau_ref), tauInputs_indices_[dnn::chargedIsoPtSum]);
    get(dnn::chargedIsoPtSumdR03_over_dR05) =
        sp.scale(tau_funcs.getChargedIsoPtSumdR03(tau, tau_ref) / tau_funcs.getChargedIsoPtSum(tau, tau_ref),
                 tauInputs_indices_[dnn::chargedIsoPtSumdR03_over_dR05]);
    if (sub_version_ == 1)
      get(dnn::footprintCorrection) =
          sp.scale(tau_funcs.getFootprintCorrectiondR03(tau, tau_ref), tauInputs_indices_[dnn::footprintCorrection]);
    else if (sub_version_ == 5)
      get(dnn::footprintCorrection) =
          sp.scale(tau_funcs.getFootprintCorrection(tau, tau_ref), tauInputs_indices_[dnn::footprintCorrection]);

    get(dnn::neutralIsoPtSum) =
        sp.scale(tau_funcs.getNeutralIsoPtSum(tau, tau_ref), tauInputs_indices_[dnn::neutralIsoPtSum]);
    get(dnn::neutralIsoPtSumWeight_over_neutralIsoPtSum) =
        sp.scale(tau_funcs.getNeutralIsoPtSumWeight(tau, tau_ref) / tau_funcs.getNeutralIsoPtSum(tau, tau_ref),
                 tauInputs_indices_[dnn::neutralIsoPtSumWeight_over_neutralIsoPtSum]);
    get(dnn::neutralIsoPtSumWeightdR03_over_neutralIsoPtSum) =
        sp.scale(tau_funcs.getNeutralIsoPtSumdR03Weight(tau, tau_ref) / tau_funcs.getNeutralIsoPtSum(tau, tau_ref),
                 tauInputs_indices_[dnn::neutralIsoPtSumWeightdR03_over_neutralIsoPtSum]);
    get(dnn::neutralIsoPtSumdR03_over_dR05) =
        sp.scale(tau_funcs.getNeutralIsoPtSumdR03(tau, tau_ref) / tau_funcs.getNeutralIsoPtSum(tau, tau_ref),
                 tauInputs_indices_[dnn::neutralIsoPtSumdR03_over_dR05]);
    get(dnn::photonPtSumOutsideSignalCone) = sp.scale(tau_funcs.getPhotonPtSumOutsideSignalCone(tau, tau_ref),
                                                      tauInputs_indices_[dnn::photonPtSumOutsideSignalCone]);
    get(dnn::puCorrPtSum) = sp.scale(tau_funcs.getPuCorrPtSum(tau, tau_ref), tauInputs_indices_[dnn::puCorrPtSum]);
    // The global PCA coordinates were used as inputs during the NN training, but it was decided to disable
    // them for the inference, because modeling of dxy_PCA in MC poorly describes the data, and x and y coordinates
    // in data results outside of the expected 5 std. dev. input validity range. On the other hand,
    // these coordinates are strongly era-dependent. Kept as comment to document what NN expects.
    if (sub_version_ == 1) {
      if (!disable_dxy_pca_) {
        auto const pca = tau_funcs.getdxyPCA(tau, tau_index);
        get(dnn::tau_dxy_pca_x) = sp.scale(pca.x(), tauInputs_indices_[dnn::tau_dxy_pca_x]);
        get(dnn::tau_dxy_pca_y) = sp.scale(pca.y(), tauInputs_indices_[dnn::tau_dxy_pca_y]);
        get(dnn::tau_dxy_pca_z) = sp.scale(pca.z(), tauInputs_indices_[dnn::tau_dxy_pca_z]);
      } else {
        get(dnn::tau_dxy_pca_x) = 0;
        get(dnn::tau_dxy_pca_y) = 0;
        get(dnn::tau_dxy_pca_z) = 0;
      }
    }

    const bool tau_dxy_valid =
        isAbove(tau_funcs.getdxy(tau, tau_index), -10) && isAbove(tau_funcs.getdxyError(tau, tau_index), 0);
    if (tau_dxy_valid) {
      get(dnn::tau_dxy_valid) = sp.scale(tau_dxy_valid, tauInputs_indices_[dnn::tau_dxy_valid]);
      get(dnn::tau_dxy) = sp.scale(tau_funcs.getdxy(tau, tau_index), tauInputs_indices_[dnn::tau_dxy]);
      get(dnn::tau_dxy_sig) =
          sp.scale(std::abs(tau_funcs.getdxy(tau, tau_index)) / tau_funcs.getdxyError(tau, tau_index),
                   tauInputs_indices_[dnn::tau_dxy_sig]);
    }
    const bool tau_ip3d_valid =
        isAbove(tau_funcs.getip3d(tau, tau_index), -10) && isAbove(tau_funcs.getip3dError(tau, tau_index), 0);
    if (tau_ip3d_valid) {
      get(dnn::tau_ip3d_valid) = sp.scale(tau_ip3d_valid, tauInputs_indices_[dnn::tau_ip3d_valid]);
      get(dnn::tau_ip3d) = sp.scale(tau_funcs.getip3d(tau, tau_index), tauInputs_indices_[dnn::tau_ip3d]);
      get(dnn::tau_ip3d_sig) =
          sp.scale(std::abs(tau_funcs.getip3d(tau, tau_index)) / tau_funcs.getip3dError(tau, tau_index),
                   tauInputs_indices_[dnn::tau_ip3d_sig]);
    }
    if (leadChargedHadrCand) {
      const bool hasTrackDetails = candFunc::getHasTrackDetails(*leadChargedHadrCand);
      const float tau_dz = (is_online_ && !hasTrackDetails) ? 0 : candFunc::getTauDz(*leadChargedHadrCand);
      get(dnn::tau_dz) = sp.scale(tau_dz, tauInputs_indices_[dnn::tau_dz]);
      get(dnn::tau_dz_sig_valid) =
          sp.scale(candFunc::getTauDZSigValid(*leadChargedHadrCand), tauInputs_indices_[dnn::tau_dz_sig_valid]);
      const double dzError = hasTrackDetails ? leadChargedHadrCand->dzError() : -999.;
      get(dnn::tau_dz_sig) = sp.scale(std::abs(tau_dz) / dzError, tauInputs_indices_[dnn::tau_dz_sig]);
    }
    get(dnn::tau_flightLength_x) =
        sp.scale(tau_funcs.getFlightLength(tau, tau_index).x(), tauInputs_indices_[dnn::tau_flightLength_x]);
    get(dnn::tau_flightLength_y) =
        sp.scale(tau_funcs.getFlightLength(tau, tau_index).y(), tauInputs_indices_[dnn::tau_flightLength_y]);
    get(dnn::tau_flightLength_z) =
        sp.scale(tau_funcs.getFlightLength(tau, tau_index).z(), tauInputs_indices_[dnn::tau_flightLength_z]);
    if (sub_version_ == 1)
      get(dnn::tau_flightLength_sig) = 0.55756444;  //This value is set due to a bug in the training
    else if (sub_version_ == 5)
      get(dnn::tau_flightLength_sig) =
          sp.scale(tau_funcs.getFlightLengthSig(tau, tau_index), tauInputs_indices_[dnn::tau_flightLength_sig]);

    get(dnn::tau_pt_weighted_deta_strip) = sp.scale(reco::tau::pt_weighted_deta_strip(tau, tau.decayMode()),
                                                    tauInputs_indices_[dnn::tau_pt_weighted_deta_strip]);

    get(dnn::tau_pt_weighted_dphi_strip) = sp.scale(reco::tau::pt_weighted_dphi_strip(tau, tau.decayMode()),
                                                    tauInputs_indices_[dnn::tau_pt_weighted_dphi_strip]);
    get(dnn::tau_pt_weighted_dr_signal) = sp.scale(reco::tau::pt_weighted_dr_signal(tau, tau.decayMode()),
                                                   tauInputs_indices_[dnn::tau_pt_weighted_dr_signal]);
    get(dnn::tau_pt_weighted_dr_iso) =
        sp.scale(reco::tau::pt_weighted_dr_iso(tau, tau.decayMode()), tauInputs_indices_[dnn::tau_pt_weighted_dr_iso]);
    get(dnn::tau_leadingTrackNormChi2) =
        sp.scale(tau_funcs.getLeadingTrackNormChi2(tau), tauInputs_indices_[dnn::tau_leadingTrackNormChi2]);
    const auto eratio = reco::tau::eratio(tau);
    const bool tau_e_ratio_valid = std::isnormal(eratio) && eratio > 0.f;
    get(dnn::tau_e_ratio_valid) = sp.scale(tau_e_ratio_valid, tauInputs_indices_[dnn::tau_e_ratio_valid]);
    get(dnn::tau_e_ratio) = tau_e_ratio_valid ? sp.scale(eratio, tauInputs_indices_[dnn::tau_e_ratio]) : 0.f;
    const double gj_angle_diff = calculateGottfriedJacksonAngleDifference(tau, tau_index, tau_funcs);
    const bool tau_gj_angle_diff_valid = (std::isnormal(gj_angle_diff) || gj_angle_diff == 0) && gj_angle_diff >= 0;
    get(dnn::tau_gj_angle_diff_valid) =
        sp.scale(tau_gj_angle_diff_valid, tauInputs_indices_[dnn::tau_gj_angle_diff_valid]);
    get(dnn::tau_gj_angle_diff) =
        tau_gj_angle_diff_valid ? sp.scale(gj_angle_diff, tauInputs_indices_[dnn::tau_gj_angle_diff]) : 0;
    get(dnn::tau_n_photons) = sp.scale(reco::tau::n_photons_total(tau), tauInputs_indices_[dnn::tau_n_photons]);
    get(dnn::tau_emFraction) = sp.scale(tau_funcs.getEmFraction(tau), tauInputs_indices_[dnn::tau_emFraction]);

    get(dnn::tau_inside_ecal_crack) =
        sp.scale(isInEcalCrack(tau.p4().eta()), tauInputs_indices_[dnn::tau_inside_ecal_crack]);
    get(dnn::leadChargedCand_etaAtEcalEntrance_minus_tau_eta) =
        sp.scale(tau_funcs.getEtaAtEcalEntrance(tau) - tau.p4().eta(),
                 tauInputs_indices_[dnn::leadChargedCand_etaAtEcalEntrance_minus_tau_eta]);
  }

  template <typename CandidateCastType, typename TauCastType>
  void createEgammaBlockInputs(unsigned idx,
                               const TauCastType& tau,
                               const size_t tau_index,
                               const edm::RefToBase<reco::BaseTau> tau_ref,
                               const reco::Vertex& pv,
                               double rho,
                               const std::vector<pat::Electron>* electrons,
                               const edm::View<reco::Candidate>& pfCands,
                               const Cell& cell_map,
                               TauFunc tau_funcs,
                               bool is_inner) {
    namespace dnn = dnn_inputs_v2::EgammaBlockInputs;
    namespace sc = deep_tau::Scaling;
    sc::FeatureT ft_global = sc::FeatureT::GridGlobal;
    sc::FeatureT ft_PFe = sc::FeatureT::PfCand_electron;
    sc::FeatureT ft_PFg = sc::FeatureT::PfCand_gamma;
    sc::FeatureT ft_e = sc::FeatureT::Electron;

    // needed to remap indices from scaling vectors to those from dnn_inputs_v2::EgammaBlockInputs
    int PFe_index_offset = scalingParamsMap_->at(std::make_pair(ft_global, false)).mean_.size();
    int e_index_offset = PFe_index_offset + scalingParamsMap_->at(std::make_pair(ft_PFe, false)).mean_.size();
    int PFg_index_offset = e_index_offset + scalingParamsMap_->at(std::make_pair(ft_e, false)).mean_.size();

    // to account for swapped order of PfCand_gamma and Electron blocks for v2p5 training w.r.t. v2p1
    int fill_index_offset_e = 0;
    int fill_index_offset_PFg = 0;
    if (sub_version_ == 5) {
      fill_index_offset_e =
          scalingParamsMap_->at(std::make_pair(ft_PFg, false)).mean_.size();  // size of PF gamma features
      fill_index_offset_PFg =
          -scalingParamsMap_->at(std::make_pair(ft_e, false)).mean_.size();  // size of Electron features
    }

    tensorflow::Tensor& inputs = *eGammaTensor_.at(is_inner);

    const auto& get = [&](int var_index) -> float& { return inputs.tensor<float, 4>()(idx, 0, 0, var_index); };

    const bool valid_index_pf_ele = cell_map.count(CellObjectType::PfCand_electron);
    const bool valid_index_pf_gamma = cell_map.count(CellObjectType::PfCand_gamma);
    const bool valid_index_ele = cell_map.count(CellObjectType::Electron);

    if (!cell_map.empty()) {
      const sc::ScalingParams& sp = scalingParamsMap_->at(std::make_pair(ft_global, false));
      get(dnn::rho) = sp.scale(rho, dnn::rho);
      get(dnn::tau_pt) = sp.scale(tau.polarP4().pt(), dnn::tau_pt);
      get(dnn::tau_eta) = sp.scale(tau.polarP4().eta(), dnn::tau_eta);
      get(dnn::tau_inside_ecal_crack) = sp.scale(isInEcalCrack(tau.polarP4().eta()), dnn::tau_inside_ecal_crack);
    }
    if (valid_index_pf_ele) {
      const sc::ScalingParams& sp = scalingParamsMap_->at(std::make_pair(ft_PFe, is_inner));
      size_t index_pf_ele = cell_map.at(CellObjectType::PfCand_electron);
      const auto& ele_cand = dynamic_cast<const CandidateCastType&>(pfCands.at(index_pf_ele));

      get(dnn::pfCand_ele_valid) = sp.scale(valid_index_pf_ele, dnn::pfCand_ele_valid - PFe_index_offset);
      get(dnn::pfCand_ele_rel_pt) =
          sp.scale(ele_cand.polarP4().pt() / tau.polarP4().pt(), dnn::pfCand_ele_rel_pt - PFe_index_offset);
      get(dnn::pfCand_ele_deta) =
          sp.scale(ele_cand.polarP4().eta() - tau.polarP4().eta(), dnn::pfCand_ele_deta - PFe_index_offset);
      get(dnn::pfCand_ele_dphi) =
          sp.scale(dPhi(tau.polarP4(), ele_cand.polarP4()), dnn::pfCand_ele_dphi - PFe_index_offset);
      get(dnn::pfCand_ele_pvAssociationQuality) = sp.scale<int>(
          candFunc::getPvAssocationQuality(ele_cand), dnn::pfCand_ele_pvAssociationQuality - PFe_index_offset);
      get(dnn::pfCand_ele_puppiWeight) = is_inner ? sp.scale(candFunc::getPuppiWeight(ele_cand, 0.9906834f),
                                                             dnn::pfCand_ele_puppiWeight - PFe_index_offset)
                                                  : sp.scale(candFunc::getPuppiWeight(ele_cand, 0.9669586f),
                                                             dnn::pfCand_ele_puppiWeight - PFe_index_offset);
      get(dnn::pfCand_ele_charge) = sp.scale(ele_cand.charge(), dnn::pfCand_ele_charge - PFe_index_offset);
      get(dnn::pfCand_ele_lostInnerHits) =
          sp.scale<int>(candFunc::getLostInnerHits(ele_cand, 0), dnn::pfCand_ele_lostInnerHits - PFe_index_offset);
      get(dnn::pfCand_ele_numberOfPixelHits) =
          sp.scale(candFunc::getNumberOfPixelHits(ele_cand, 0), dnn::pfCand_ele_numberOfPixelHits - PFe_index_offset);
      get(dnn::pfCand_ele_vertex_dx) =
          sp.scale(ele_cand.vertex().x() - pv.position().x(), dnn::pfCand_ele_vertex_dx - PFe_index_offset);
      get(dnn::pfCand_ele_vertex_dy) =
          sp.scale(ele_cand.vertex().y() - pv.position().y(), dnn::pfCand_ele_vertex_dy - PFe_index_offset);
      get(dnn::pfCand_ele_vertex_dz) =
          sp.scale(ele_cand.vertex().z() - pv.position().z(), dnn::pfCand_ele_vertex_dz - PFe_index_offset);
      get(dnn::pfCand_ele_vertex_dx_tauFL) =
          sp.scale(ele_cand.vertex().x() - pv.position().x() - tau_funcs.getFlightLength(tau, tau_index).x(),
                   dnn::pfCand_ele_vertex_dx_tauFL - PFe_index_offset);
      get(dnn::pfCand_ele_vertex_dy_tauFL) =
          sp.scale(ele_cand.vertex().y() - pv.position().y() - tau_funcs.getFlightLength(tau, tau_index).y(),
                   dnn::pfCand_ele_vertex_dy_tauFL - PFe_index_offset);
      get(dnn::pfCand_ele_vertex_dz_tauFL) =
          sp.scale(ele_cand.vertex().z() - pv.position().z() - tau_funcs.getFlightLength(tau, tau_index).z(),
                   dnn::pfCand_ele_vertex_dz_tauFL - PFe_index_offset);

      const bool hasTrackDetails = candFunc::getHasTrackDetails(ele_cand);
      if (hasTrackDetails) {
        get(dnn::pfCand_ele_hasTrackDetails) =
            sp.scale(hasTrackDetails, dnn::pfCand_ele_hasTrackDetails - PFe_index_offset);
        get(dnn::pfCand_ele_dxy) = sp.scale(candFunc::getTauDxy(ele_cand), dnn::pfCand_ele_dxy - PFe_index_offset);
        get(dnn::pfCand_ele_dxy_sig) = sp.scale(std::abs(candFunc::getTauDxy(ele_cand)) / ele_cand.dxyError(),
                                                dnn::pfCand_ele_dxy_sig - PFe_index_offset);
        get(dnn::pfCand_ele_dz) = sp.scale(candFunc::getTauDz(ele_cand), dnn::pfCand_ele_dz - PFe_index_offset);
        get(dnn::pfCand_ele_dz_sig) = sp.scale(std::abs(candFunc::getTauDz(ele_cand)) / ele_cand.dzError(),
                                               dnn::pfCand_ele_dz_sig - PFe_index_offset);
        get(dnn::pfCand_ele_track_chi2_ndof) =
            candFunc::getPseudoTrack(ele_cand).ndof() > 0
                ? sp.scale(candFunc::getPseudoTrack(ele_cand).chi2() / candFunc::getPseudoTrack(ele_cand).ndof(),
                           dnn::pfCand_ele_track_chi2_ndof - PFe_index_offset)
                : 0;
        get(dnn::pfCand_ele_track_ndof) =
            candFunc::getPseudoTrack(ele_cand).ndof() > 0
                ? sp.scale(candFunc::getPseudoTrack(ele_cand).ndof(), dnn::pfCand_ele_track_ndof - PFe_index_offset)
                : 0;
      }
    }
    if (valid_index_pf_gamma) {
      const sc::ScalingParams& sp = scalingParamsMap_->at(std::make_pair(ft_PFg, is_inner));
      size_t index_pf_gamma = cell_map.at(CellObjectType::PfCand_gamma);
      const auto& gamma_cand = dynamic_cast<const CandidateCastType&>(pfCands.at(index_pf_gamma));

      get(dnn::pfCand_gamma_valid + fill_index_offset_PFg) =
          sp.scale(valid_index_pf_gamma, dnn::pfCand_gamma_valid - PFg_index_offset);
      get(dnn::pfCand_gamma_rel_pt + fill_index_offset_PFg) =
          sp.scale(gamma_cand.polarP4().pt() / tau.polarP4().pt(), dnn::pfCand_gamma_rel_pt - PFg_index_offset);
      get(dnn::pfCand_gamma_deta + fill_index_offset_PFg) =
          sp.scale(gamma_cand.polarP4().eta() - tau.polarP4().eta(), dnn::pfCand_gamma_deta - PFg_index_offset);
      get(dnn::pfCand_gamma_dphi + fill_index_offset_PFg) =
          sp.scale(dPhi(tau.polarP4(), gamma_cand.polarP4()), dnn::pfCand_gamma_dphi - PFg_index_offset);
      get(dnn::pfCand_gamma_pvAssociationQuality + fill_index_offset_PFg) = sp.scale<int>(
          candFunc::getPvAssocationQuality(gamma_cand), dnn::pfCand_gamma_pvAssociationQuality - PFg_index_offset);
      get(dnn::pfCand_gamma_fromPV + fill_index_offset_PFg) =
          sp.scale<int>(candFunc::getFromPV(gamma_cand), dnn::pfCand_gamma_fromPV - PFg_index_offset);
      get(dnn::pfCand_gamma_puppiWeight + fill_index_offset_PFg) =
          is_inner ? sp.scale(candFunc::getPuppiWeight(gamma_cand, 0.9084110f),
                              dnn::pfCand_gamma_puppiWeight - PFg_index_offset)
                   : sp.scale(candFunc::getPuppiWeight(gamma_cand, 0.4211567f),
                              dnn::pfCand_gamma_puppiWeight - PFg_index_offset);
      get(dnn::pfCand_gamma_puppiWeightNoLep + fill_index_offset_PFg) =
          is_inner ? sp.scale(candFunc::getPuppiWeightNoLep(gamma_cand, 0.8857716f),
                              dnn::pfCand_gamma_puppiWeightNoLep - PFg_index_offset)
                   : sp.scale(candFunc::getPuppiWeightNoLep(gamma_cand, 0.3822604f),
                              dnn::pfCand_gamma_puppiWeightNoLep - PFg_index_offset);
      get(dnn::pfCand_gamma_lostInnerHits + fill_index_offset_PFg) =
          sp.scale<int>(candFunc::getLostInnerHits(gamma_cand, 0), dnn::pfCand_gamma_lostInnerHits - PFg_index_offset);
      get(dnn::pfCand_gamma_numberOfPixelHits + fill_index_offset_PFg) = sp.scale(
          candFunc::getNumberOfPixelHits(gamma_cand, 0), dnn::pfCand_gamma_numberOfPixelHits - PFg_index_offset);
      get(dnn::pfCand_gamma_vertex_dx + fill_index_offset_PFg) =
          sp.scale(gamma_cand.vertex().x() - pv.position().x(), dnn::pfCand_gamma_vertex_dx - PFg_index_offset);
      get(dnn::pfCand_gamma_vertex_dy + fill_index_offset_PFg) =
          sp.scale(gamma_cand.vertex().y() - pv.position().y(), dnn::pfCand_gamma_vertex_dy - PFg_index_offset);
      get(dnn::pfCand_gamma_vertex_dz + fill_index_offset_PFg) =
          sp.scale(gamma_cand.vertex().z() - pv.position().z(), dnn::pfCand_gamma_vertex_dz - PFg_index_offset);
      get(dnn::pfCand_gamma_vertex_dx_tauFL + fill_index_offset_PFg) =
          sp.scale(gamma_cand.vertex().x() - pv.position().x() - tau_funcs.getFlightLength(tau, tau_index).x(),
                   dnn::pfCand_gamma_vertex_dx_tauFL - PFg_index_offset);
      get(dnn::pfCand_gamma_vertex_dy_tauFL + fill_index_offset_PFg) =
          sp.scale(gamma_cand.vertex().y() - pv.position().y() - tau_funcs.getFlightLength(tau, tau_index).y(),
                   dnn::pfCand_gamma_vertex_dy_tauFL - PFg_index_offset);
      get(dnn::pfCand_gamma_vertex_dz_tauFL + fill_index_offset_PFg) =
          sp.scale(gamma_cand.vertex().z() - pv.position().z() - tau_funcs.getFlightLength(tau, tau_index).z(),
                   dnn::pfCand_gamma_vertex_dz_tauFL - PFg_index_offset);
      const bool hasTrackDetails = candFunc::getHasTrackDetails(gamma_cand);
      if (hasTrackDetails) {
        get(dnn::pfCand_gamma_hasTrackDetails + fill_index_offset_PFg) =
            sp.scale(hasTrackDetails, dnn::pfCand_gamma_hasTrackDetails - PFg_index_offset);
        get(dnn::pfCand_gamma_dxy + fill_index_offset_PFg) =
            sp.scale(candFunc::getTauDxy(gamma_cand), dnn::pfCand_gamma_dxy - PFg_index_offset);
        get(dnn::pfCand_gamma_dxy_sig + fill_index_offset_PFg) =
            sp.scale(std::abs(candFunc::getTauDxy(gamma_cand)) / gamma_cand.dxyError(),
                     dnn::pfCand_gamma_dxy_sig - PFg_index_offset);
        get(dnn::pfCand_gamma_dz + fill_index_offset_PFg) =
            sp.scale(candFunc::getTauDz(gamma_cand), dnn::pfCand_gamma_dz - PFg_index_offset);
        get(dnn::pfCand_gamma_dz_sig + fill_index_offset_PFg) =
            sp.scale(std::abs(candFunc::getTauDz(gamma_cand)) / gamma_cand.dzError(),
                     dnn::pfCand_gamma_dz_sig - PFg_index_offset);
        get(dnn::pfCand_gamma_track_chi2_ndof + fill_index_offset_PFg) =
            candFunc::getPseudoTrack(gamma_cand).ndof() > 0
                ? sp.scale(candFunc::getPseudoTrack(gamma_cand).chi2() / candFunc::getPseudoTrack(gamma_cand).ndof(),
                           dnn::pfCand_gamma_track_chi2_ndof - PFg_index_offset)
                : 0;
        get(dnn::pfCand_gamma_track_ndof + fill_index_offset_PFg) =
            candFunc::getPseudoTrack(gamma_cand).ndof() > 0
                ? sp.scale(candFunc::getPseudoTrack(gamma_cand).ndof(), dnn::pfCand_gamma_track_ndof - PFg_index_offset)
                : 0;
      }
    }
    if (valid_index_ele) {
      const sc::ScalingParams& sp = scalingParamsMap_->at(std::make_pair(ft_e, is_inner));
      size_t index_ele = cell_map.at(CellObjectType::Electron);
      const auto& ele = electrons->at(index_ele);

      get(dnn::ele_valid + fill_index_offset_e) = sp.scale(valid_index_ele, dnn::ele_valid - e_index_offset);
      get(dnn::ele_rel_pt + fill_index_offset_e) =
          sp.scale(ele.polarP4().pt() / tau.polarP4().pt(), dnn::ele_rel_pt - e_index_offset);
      get(dnn::ele_deta + fill_index_offset_e) =
          sp.scale(ele.polarP4().eta() - tau.polarP4().eta(), dnn::ele_deta - e_index_offset);
      get(dnn::ele_dphi + fill_index_offset_e) =
          sp.scale(dPhi(tau.polarP4(), ele.polarP4()), dnn::ele_dphi - e_index_offset);

      float cc_ele_energy, cc_gamma_energy;
      int cc_n_gamma;
      const bool cc_valid = calculateElectronClusterVarsV2(ele, cc_ele_energy, cc_gamma_energy, cc_n_gamma);
      if (cc_valid) {
        get(dnn::ele_cc_valid + fill_index_offset_e) = sp.scale(cc_valid, dnn::ele_cc_valid - e_index_offset);
        get(dnn::ele_cc_ele_rel_energy + fill_index_offset_e) =
            sp.scale(cc_ele_energy / ele.polarP4().pt(), dnn::ele_cc_ele_rel_energy - e_index_offset);
        get(dnn::ele_cc_gamma_rel_energy + fill_index_offset_e) =
            sp.scale(cc_gamma_energy / cc_ele_energy, dnn::ele_cc_gamma_rel_energy - e_index_offset);
        get(dnn::ele_cc_n_gamma + fill_index_offset_e) = sp.scale(cc_n_gamma, dnn::ele_cc_n_gamma - e_index_offset);
      }
      get(dnn::ele_rel_trackMomentumAtVtx + fill_index_offset_e) =
          sp.scale(ele.trackMomentumAtVtx().R() / ele.polarP4().pt(), dnn::ele_rel_trackMomentumAtVtx - e_index_offset);
      get(dnn::ele_rel_trackMomentumAtCalo + fill_index_offset_e) = sp.scale(
          ele.trackMomentumAtCalo().R() / ele.polarP4().pt(), dnn::ele_rel_trackMomentumAtCalo - e_index_offset);
      get(dnn::ele_rel_trackMomentumOut + fill_index_offset_e) =
          sp.scale(ele.trackMomentumOut().R() / ele.polarP4().pt(), dnn::ele_rel_trackMomentumOut - e_index_offset);
      get(dnn::ele_rel_trackMomentumAtEleClus + fill_index_offset_e) = sp.scale(
          ele.trackMomentumAtEleClus().R() / ele.polarP4().pt(), dnn::ele_rel_trackMomentumAtEleClus - e_index_offset);
      get(dnn::ele_rel_trackMomentumAtVtxWithConstraint + fill_index_offset_e) =
          sp.scale(ele.trackMomentumAtVtxWithConstraint().R() / ele.polarP4().pt(),
                   dnn::ele_rel_trackMomentumAtVtxWithConstraint - e_index_offset);
      get(dnn::ele_rel_ecalEnergy + fill_index_offset_e) =
          sp.scale(ele.ecalEnergy() / ele.polarP4().pt(), dnn::ele_rel_ecalEnergy - e_index_offset);
      get(dnn::ele_ecalEnergy_sig + fill_index_offset_e) =
          sp.scale(ele.ecalEnergy() / ele.ecalEnergyError(), dnn::ele_ecalEnergy_sig - e_index_offset);
      get(dnn::ele_eSuperClusterOverP + fill_index_offset_e) =
          sp.scale(ele.eSuperClusterOverP(), dnn::ele_eSuperClusterOverP - e_index_offset);
      get(dnn::ele_eSeedClusterOverP + fill_index_offset_e) =
          sp.scale(ele.eSeedClusterOverP(), dnn::ele_eSeedClusterOverP - e_index_offset);
      get(dnn::ele_eSeedClusterOverPout + fill_index_offset_e) =
          sp.scale(ele.eSeedClusterOverPout(), dnn::ele_eSeedClusterOverPout - e_index_offset);
      get(dnn::ele_eEleClusterOverPout + fill_index_offset_e) =
          sp.scale(ele.eEleClusterOverPout(), dnn::ele_eEleClusterOverPout - e_index_offset);
      get(dnn::ele_deltaEtaSuperClusterTrackAtVtx + fill_index_offset_e) =
          sp.scale(ele.deltaEtaSuperClusterTrackAtVtx(), dnn::ele_deltaEtaSuperClusterTrackAtVtx - e_index_offset);
      get(dnn::ele_deltaEtaSeedClusterTrackAtCalo + fill_index_offset_e) =
          sp.scale(ele.deltaEtaSeedClusterTrackAtCalo(), dnn::ele_deltaEtaSeedClusterTrackAtCalo - e_index_offset);
      get(dnn::ele_deltaEtaEleClusterTrackAtCalo + fill_index_offset_e) =
          sp.scale(ele.deltaEtaEleClusterTrackAtCalo(), dnn::ele_deltaEtaEleClusterTrackAtCalo - e_index_offset);
      get(dnn::ele_deltaPhiEleClusterTrackAtCalo + fill_index_offset_e) =
          sp.scale(ele.deltaPhiEleClusterTrackAtCalo(), dnn::ele_deltaPhiEleClusterTrackAtCalo - e_index_offset);
      get(dnn::ele_deltaPhiSuperClusterTrackAtVtx + fill_index_offset_e) =
          sp.scale(ele.deltaPhiSuperClusterTrackAtVtx(), dnn::ele_deltaPhiSuperClusterTrackAtVtx - e_index_offset);
      get(dnn::ele_deltaPhiSeedClusterTrackAtCalo + fill_index_offset_e) =
          sp.scale(ele.deltaPhiSeedClusterTrackAtCalo(), dnn::ele_deltaPhiSeedClusterTrackAtCalo - e_index_offset);
      get(dnn::ele_mvaInput_earlyBrem + fill_index_offset_e) =
          sp.scale(ele.mvaInput().earlyBrem, dnn::ele_mvaInput_earlyBrem - e_index_offset);
      get(dnn::ele_mvaInput_lateBrem + fill_index_offset_e) =
          sp.scale(ele.mvaInput().lateBrem, dnn::ele_mvaInput_lateBrem - e_index_offset);
      get(dnn::ele_mvaInput_sigmaEtaEta + fill_index_offset_e) =
          sp.scale(ele.mvaInput().sigmaEtaEta, dnn::ele_mvaInput_sigmaEtaEta - e_index_offset);
      get(dnn::ele_mvaInput_hadEnergy + fill_index_offset_e) =
          sp.scale(ele.mvaInput().hadEnergy, dnn::ele_mvaInput_hadEnergy - e_index_offset);
      get(dnn::ele_mvaInput_deltaEta + fill_index_offset_e) =
          sp.scale(ele.mvaInput().deltaEta, dnn::ele_mvaInput_deltaEta - e_index_offset);
      const auto& gsfTrack = ele.gsfTrack();
      if (gsfTrack.isNonnull()) {
        get(dnn::ele_gsfTrack_normalizedChi2 + fill_index_offset_e) =
            sp.scale(gsfTrack->normalizedChi2(), dnn::ele_gsfTrack_normalizedChi2 - e_index_offset);
        get(dnn::ele_gsfTrack_numberOfValidHits + fill_index_offset_e) =
            sp.scale(gsfTrack->numberOfValidHits(), dnn::ele_gsfTrack_numberOfValidHits - e_index_offset);
        get(dnn::ele_rel_gsfTrack_pt + fill_index_offset_e) =
            sp.scale(gsfTrack->pt() / ele.polarP4().pt(), dnn::ele_rel_gsfTrack_pt - e_index_offset);
        get(dnn::ele_gsfTrack_pt_sig + fill_index_offset_e) =
            sp.scale(gsfTrack->pt() / gsfTrack->ptError(), dnn::ele_gsfTrack_pt_sig - e_index_offset);
      }
      const auto& closestCtfTrack = ele.closestCtfTrackRef();
      const bool has_closestCtfTrack = closestCtfTrack.isNonnull();
      if (has_closestCtfTrack) {
        get(dnn::ele_has_closestCtfTrack + fill_index_offset_e) =
            sp.scale(has_closestCtfTrack, dnn::ele_has_closestCtfTrack - e_index_offset);
        get(dnn::ele_closestCtfTrack_normalizedChi2 + fill_index_offset_e) =
            sp.scale(closestCtfTrack->normalizedChi2(), dnn::ele_closestCtfTrack_normalizedChi2 - e_index_offset);
        get(dnn::ele_closestCtfTrack_numberOfValidHits + fill_index_offset_e) =
            sp.scale(closestCtfTrack->numberOfValidHits(), dnn::ele_closestCtfTrack_numberOfValidHits - e_index_offset);
      }
    }
  }

  template <typename CandidateCastType, typename TauCastType>
  void createMuonBlockInputs(unsigned idx,
                             const TauCastType& tau,
                             const size_t tau_index,
                             const edm::RefToBase<reco::BaseTau> tau_ref,
                             const reco::Vertex& pv,
                             double rho,
                             const std::vector<pat::Muon>* muons,
                             const edm::View<reco::Candidate>& pfCands,
                             const Cell& cell_map,
                             TauFunc tau_funcs,
                             bool is_inner) {
    namespace dnn = dnn_inputs_v2::MuonBlockInputs;
    namespace sc = deep_tau::Scaling;
    sc::FeatureT ft_global = sc::FeatureT::GridGlobal;
    sc::FeatureT ft_PFmu = sc::FeatureT::PfCand_muon;
    sc::FeatureT ft_mu = sc::FeatureT::Muon;

    // needed to remap indices from scaling vectors to those from dnn_inputs_v2::MuonBlockInputs
    int PFmu_index_offset = scalingParamsMap_->at(std::make_pair(ft_global, false)).mean_.size();
    int mu_index_offset = PFmu_index_offset + scalingParamsMap_->at(std::make_pair(ft_PFmu, false)).mean_.size();

    tensorflow::Tensor& inputs = *muonTensor_.at(is_inner);

    const auto& get = [&](int var_index) -> float& { return inputs.tensor<float, 4>()(idx, 0, 0, var_index); };

    const bool valid_index_pf_muon = cell_map.count(CellObjectType::PfCand_muon);
    const bool valid_index_muon = cell_map.count(CellObjectType::Muon);

    if (!cell_map.empty()) {
      const sc::ScalingParams& sp = scalingParamsMap_->at(std::make_pair(ft_global, false));
      get(dnn::rho) = sp.scale(rho, dnn::rho);
      get(dnn::tau_pt) = sp.scale(tau.polarP4().pt(), dnn::tau_pt);
      get(dnn::tau_eta) = sp.scale(tau.polarP4().eta(), dnn::tau_eta);
      get(dnn::tau_inside_ecal_crack) = sp.scale(isInEcalCrack(tau.polarP4().eta()), dnn::tau_inside_ecal_crack);
    }
    if (valid_index_pf_muon) {
      const sc::ScalingParams& sp = scalingParamsMap_->at(std::make_pair(ft_PFmu, is_inner));
      size_t index_pf_muon = cell_map.at(CellObjectType::PfCand_muon);
      const auto& muon_cand = dynamic_cast<const CandidateCastType&>(pfCands.at(index_pf_muon));

      get(dnn::pfCand_muon_valid) = sp.scale(valid_index_pf_muon, dnn::pfCand_muon_valid - PFmu_index_offset);
      get(dnn::pfCand_muon_rel_pt) =
          sp.scale(muon_cand.polarP4().pt() / tau.polarP4().pt(), dnn::pfCand_muon_rel_pt - PFmu_index_offset);
      get(dnn::pfCand_muon_deta) =
          sp.scale(muon_cand.polarP4().eta() - tau.polarP4().eta(), dnn::pfCand_muon_deta - PFmu_index_offset);
      get(dnn::pfCand_muon_dphi) =
          sp.scale(dPhi(tau.polarP4(), muon_cand.polarP4()), dnn::pfCand_muon_dphi - PFmu_index_offset);
      get(dnn::pfCand_muon_pvAssociationQuality) = sp.scale<int>(
          candFunc::getPvAssocationQuality(muon_cand), dnn::pfCand_muon_pvAssociationQuality - PFmu_index_offset);
      get(dnn::pfCand_muon_fromPV) =
          sp.scale<int>(candFunc::getFromPV(muon_cand), dnn::pfCand_muon_fromPV - PFmu_index_offset);
      get(dnn::pfCand_muon_puppiWeight) = is_inner ? sp.scale(candFunc::getPuppiWeight(muon_cand, 0.9786588f),
                                                              dnn::pfCand_muon_puppiWeight - PFmu_index_offset)
                                                   : sp.scale(candFunc::getPuppiWeight(muon_cand, 0.8132477f),
                                                              dnn::pfCand_muon_puppiWeight - PFmu_index_offset);
      get(dnn::pfCand_muon_charge) = sp.scale(muon_cand.charge(), dnn::pfCand_muon_charge - PFmu_index_offset);
      get(dnn::pfCand_muon_lostInnerHits) =
          sp.scale<int>(candFunc::getLostInnerHits(muon_cand, 0), dnn::pfCand_muon_lostInnerHits - PFmu_index_offset);
      get(dnn::pfCand_muon_numberOfPixelHits) = sp.scale(candFunc::getNumberOfPixelHits(muon_cand, 0),
                                                         dnn::pfCand_muon_numberOfPixelHits - PFmu_index_offset);
      get(dnn::pfCand_muon_vertex_dx) =
          sp.scale(muon_cand.vertex().x() - pv.position().x(), dnn::pfCand_muon_vertex_dx - PFmu_index_offset);
      get(dnn::pfCand_muon_vertex_dy) =
          sp.scale(muon_cand.vertex().y() - pv.position().y(), dnn::pfCand_muon_vertex_dy - PFmu_index_offset);
      get(dnn::pfCand_muon_vertex_dz) =
          sp.scale(muon_cand.vertex().z() - pv.position().z(), dnn::pfCand_muon_vertex_dz - PFmu_index_offset);
      get(dnn::pfCand_muon_vertex_dx_tauFL) =
          sp.scale(muon_cand.vertex().x() - pv.position().x() - tau_funcs.getFlightLength(tau, tau_index).x(),
                   dnn::pfCand_muon_vertex_dx_tauFL - PFmu_index_offset);
      get(dnn::pfCand_muon_vertex_dy_tauFL) =
          sp.scale(muon_cand.vertex().y() - pv.position().y() - tau_funcs.getFlightLength(tau, tau_index).y(),
                   dnn::pfCand_muon_vertex_dy_tauFL - PFmu_index_offset);
      get(dnn::pfCand_muon_vertex_dz_tauFL) =
          sp.scale(muon_cand.vertex().z() - pv.position().z() - tau_funcs.getFlightLength(tau, tau_index).z(),
                   dnn::pfCand_muon_vertex_dz_tauFL - PFmu_index_offset);

      const bool hasTrackDetails = candFunc::getHasTrackDetails(muon_cand);
      if (hasTrackDetails) {
        get(dnn::pfCand_muon_hasTrackDetails) =
            sp.scale(hasTrackDetails, dnn::pfCand_muon_hasTrackDetails - PFmu_index_offset);
        get(dnn::pfCand_muon_dxy) = sp.scale(candFunc::getTauDxy(muon_cand), dnn::pfCand_muon_dxy - PFmu_index_offset);
        get(dnn::pfCand_muon_dxy_sig) = sp.scale(std::abs(candFunc::getTauDxy(muon_cand)) / muon_cand.dxyError(),
                                                 dnn::pfCand_muon_dxy_sig - PFmu_index_offset);
        get(dnn::pfCand_muon_dz) = sp.scale(candFunc::getTauDz(muon_cand), dnn::pfCand_muon_dz - PFmu_index_offset);
        get(dnn::pfCand_muon_dz_sig) = sp.scale(std::abs(candFunc::getTauDz(muon_cand)) / muon_cand.dzError(),
                                                dnn::pfCand_muon_dz_sig - PFmu_index_offset);
        get(dnn::pfCand_muon_track_chi2_ndof) =
            candFunc::getPseudoTrack(muon_cand).ndof() > 0
                ? sp.scale(candFunc::getPseudoTrack(muon_cand).chi2() / candFunc::getPseudoTrack(muon_cand).ndof(),
                           dnn::pfCand_muon_track_chi2_ndof - PFmu_index_offset)
                : 0;
        get(dnn::pfCand_muon_track_ndof) =
            candFunc::getPseudoTrack(muon_cand).ndof() > 0
                ? sp.scale(candFunc::getPseudoTrack(muon_cand).ndof(), dnn::pfCand_muon_track_ndof - PFmu_index_offset)
                : 0;
      }
    }
    if (valid_index_muon) {
      const sc::ScalingParams& sp = scalingParamsMap_->at(std::make_pair(ft_mu, is_inner));
      size_t index_muon = cell_map.at(CellObjectType::Muon);
      const auto& muon = muons->at(index_muon);

      get(dnn::muon_valid) = sp.scale(valid_index_muon, dnn::muon_valid - mu_index_offset);
      get(dnn::muon_rel_pt) = sp.scale(muon.polarP4().pt() / tau.polarP4().pt(), dnn::muon_rel_pt - mu_index_offset);
      get(dnn::muon_deta) = sp.scale(muon.polarP4().eta() - tau.polarP4().eta(), dnn::muon_deta - mu_index_offset);
      get(dnn::muon_dphi) = sp.scale(dPhi(tau.polarP4(), muon.polarP4()), dnn::muon_dphi - mu_index_offset);
      get(dnn::muon_dxy) = sp.scale(muon.dB(pat::Muon::PV2D), dnn::muon_dxy - mu_index_offset);
      get(dnn::muon_dxy_sig) =
          sp.scale(std::abs(muon.dB(pat::Muon::PV2D)) / muon.edB(pat::Muon::PV2D), dnn::muon_dxy_sig - mu_index_offset);

      const bool normalizedChi2_valid = muon.globalTrack().isNonnull() && muon.normChi2() >= 0;
      if (normalizedChi2_valid) {
        get(dnn::muon_normalizedChi2_valid) =
            sp.scale(normalizedChi2_valid, dnn::muon_normalizedChi2_valid - mu_index_offset);
        get(dnn::muon_normalizedChi2) = sp.scale(muon.normChi2(), dnn::muon_normalizedChi2 - mu_index_offset);
        if (muon.innerTrack().isNonnull())
          get(dnn::muon_numberOfValidHits) =
              sp.scale(muon.numberOfValidHits(), dnn::muon_numberOfValidHits - mu_index_offset);
      }
      get(dnn::muon_segmentCompatibility) =
          sp.scale(muon.segmentCompatibility(), dnn::muon_segmentCompatibility - mu_index_offset);
      get(dnn::muon_caloCompatibility) =
          sp.scale(muon.caloCompatibility(), dnn::muon_caloCompatibility - mu_index_offset);

      const bool pfEcalEnergy_valid = muon.pfEcalEnergy() >= 0;
      if (pfEcalEnergy_valid) {
        get(dnn::muon_pfEcalEnergy_valid) =
            sp.scale(pfEcalEnergy_valid, dnn::muon_pfEcalEnergy_valid - mu_index_offset);
        get(dnn::muon_rel_pfEcalEnergy) =
            sp.scale(muon.pfEcalEnergy() / muon.polarP4().pt(), dnn::muon_rel_pfEcalEnergy - mu_index_offset);
      }

      MuonHitMatchV2 hit_match(muon);
      static const std::map<int, std::pair<int, int>> muonMatchHitVars = {
          {MuonSubdetId::DT, {dnn::muon_n_matches_DT_1, dnn::muon_n_hits_DT_1}},
          {MuonSubdetId::CSC, {dnn::muon_n_matches_CSC_1, dnn::muon_n_hits_CSC_1}},
          {MuonSubdetId::RPC, {dnn::muon_n_matches_RPC_1, dnn::muon_n_hits_RPC_1}}};

      for (int subdet : hit_match.MuonHitMatchV2::consideredSubdets()) {
        const auto& matchHitVar = muonMatchHitVars.at(subdet);
        for (int station = MuonHitMatchV2::first_station_id; station <= MuonHitMatchV2::last_station_id; ++station) {
          const unsigned n_matches = hit_match.nMatches(subdet, station);
          const unsigned n_hits = hit_match.nHits(subdet, station);
          get(matchHitVar.first + station - 1) = sp.scale(n_matches, matchHitVar.first + station - 1 - mu_index_offset);
          get(matchHitVar.second + station - 1) = sp.scale(n_hits, matchHitVar.second + station - 1 - mu_index_offset);
        }
      }
    }
  }

  template <typename CandidateCastType, typename TauCastType>
  void createHadronsBlockInputs(unsigned idx,
                                const TauCastType& tau,
                                const size_t tau_index,
                                const edm::RefToBase<reco::BaseTau> tau_ref,
                                const reco::Vertex& pv,
                                double rho,
                                const edm::View<reco::Candidate>& pfCands,
                                const Cell& cell_map,
                                TauFunc tau_funcs,
                                bool is_inner) {
    namespace dnn = dnn_inputs_v2::HadronBlockInputs;
    namespace sc = deep_tau::Scaling;
    sc::FeatureT ft_global = sc::FeatureT::GridGlobal;
    sc::FeatureT ft_PFchH = sc::FeatureT::PfCand_chHad;
    sc::FeatureT ft_PFnH = sc::FeatureT::PfCand_nHad;

    // needed to remap indices from scaling vectors to those from dnn_inputs_v2::HadronBlockInputs
    int PFchH_index_offset = scalingParamsMap_->at(std::make_pair(ft_global, false)).mean_.size();
    int PFnH_index_offset = PFchH_index_offset + scalingParamsMap_->at(std::make_pair(ft_PFchH, false)).mean_.size();

    tensorflow::Tensor& inputs = *hadronsTensor_.at(is_inner);

    const auto& get = [&](int var_index) -> float& { return inputs.tensor<float, 4>()(idx, 0, 0, var_index); };

    const bool valid_chH = cell_map.count(CellObjectType::PfCand_chargedHadron);
    const bool valid_nH = cell_map.count(CellObjectType::PfCand_neutralHadron);

    if (!cell_map.empty()) {
      const sc::ScalingParams& sp = scalingParamsMap_->at(std::make_pair(ft_global, false));
      get(dnn::rho) = sp.scale(rho, dnn::rho);
      get(dnn::tau_pt) = sp.scale(tau.polarP4().pt(), dnn::tau_pt);
      get(dnn::tau_eta) = sp.scale(tau.polarP4().eta(), dnn::tau_eta);
      get(dnn::tau_inside_ecal_crack) = sp.scale(isInEcalCrack(tau.polarP4().eta()), dnn::tau_inside_ecal_crack);
    }
    if (valid_chH) {
      const sc::ScalingParams& sp = scalingParamsMap_->at(std::make_pair(ft_PFchH, is_inner));
      size_t index_chH = cell_map.at(CellObjectType::PfCand_chargedHadron);
      const auto& chH_cand = dynamic_cast<const CandidateCastType&>(pfCands.at(index_chH));

      get(dnn::pfCand_chHad_valid) = sp.scale(valid_chH, dnn::pfCand_chHad_valid - PFchH_index_offset);
      get(dnn::pfCand_chHad_rel_pt) =
          sp.scale(chH_cand.polarP4().pt() / tau.polarP4().pt(), dnn::pfCand_chHad_rel_pt - PFchH_index_offset);
      get(dnn::pfCand_chHad_deta) =
          sp.scale(chH_cand.polarP4().eta() - tau.polarP4().eta(), dnn::pfCand_chHad_deta - PFchH_index_offset);
      get(dnn::pfCand_chHad_dphi) =
          sp.scale(dPhi(tau.polarP4(), chH_cand.polarP4()), dnn::pfCand_chHad_dphi - PFchH_index_offset);
      get(dnn::pfCand_chHad_leadChargedHadrCand) =
          sp.scale(&chH_cand == dynamic_cast<const CandidateCastType*>(tau.leadChargedHadrCand().get()),
                   dnn::pfCand_chHad_leadChargedHadrCand - PFchH_index_offset);
      get(dnn::pfCand_chHad_pvAssociationQuality) = sp.scale<int>(
          candFunc::getPvAssocationQuality(chH_cand), dnn::pfCand_chHad_pvAssociationQuality - PFchH_index_offset);
      get(dnn::pfCand_chHad_fromPV) =
          sp.scale<int>(candFunc::getFromPV(chH_cand), dnn::pfCand_chHad_fromPV - PFchH_index_offset);
      const float default_chH_pw_inner = 0.7614090f;
      const float default_chH_pw_outer = 0.1974930f;
      get(dnn::pfCand_chHad_puppiWeight) = is_inner ? sp.scale(candFunc::getPuppiWeight(chH_cand, default_chH_pw_inner),
                                                               dnn::pfCand_chHad_puppiWeight - PFchH_index_offset)
                                                    : sp.scale(candFunc::getPuppiWeight(chH_cand, default_chH_pw_outer),
                                                               dnn::pfCand_chHad_puppiWeight - PFchH_index_offset);
      get(dnn::pfCand_chHad_puppiWeightNoLep) =
          is_inner ? sp.scale(candFunc::getPuppiWeightNoLep(chH_cand, default_chH_pw_inner),
                              dnn::pfCand_chHad_puppiWeightNoLep - PFchH_index_offset)
                   : sp.scale(candFunc::getPuppiWeightNoLep(chH_cand, default_chH_pw_outer),
                              dnn::pfCand_chHad_puppiWeightNoLep - PFchH_index_offset);
      get(dnn::pfCand_chHad_charge) = sp.scale(chH_cand.charge(), dnn::pfCand_chHad_charge - PFchH_index_offset);
      get(dnn::pfCand_chHad_lostInnerHits) =
          sp.scale<int>(candFunc::getLostInnerHits(chH_cand, 0), dnn::pfCand_chHad_lostInnerHits - PFchH_index_offset);
      get(dnn::pfCand_chHad_numberOfPixelHits) = sp.scale(candFunc::getNumberOfPixelHits(chH_cand, 0),
                                                          dnn::pfCand_chHad_numberOfPixelHits - PFchH_index_offset);
      get(dnn::pfCand_chHad_vertex_dx) =
          sp.scale(chH_cand.vertex().x() - pv.position().x(), dnn::pfCand_chHad_vertex_dx - PFchH_index_offset);
      get(dnn::pfCand_chHad_vertex_dy) =
          sp.scale(chH_cand.vertex().y() - pv.position().y(), dnn::pfCand_chHad_vertex_dy - PFchH_index_offset);
      get(dnn::pfCand_chHad_vertex_dz) =
          sp.scale(chH_cand.vertex().z() - pv.position().z(), dnn::pfCand_chHad_vertex_dz - PFchH_index_offset);
      get(dnn::pfCand_chHad_vertex_dx_tauFL) =
          sp.scale(chH_cand.vertex().x() - pv.position().x() - tau_funcs.getFlightLength(tau, tau_index).x(),
                   dnn::pfCand_chHad_vertex_dx_tauFL - PFchH_index_offset);
      get(dnn::pfCand_chHad_vertex_dy_tauFL) =
          sp.scale(chH_cand.vertex().y() - pv.position().y() - tau_funcs.getFlightLength(tau, tau_index).y(),
                   dnn::pfCand_chHad_vertex_dy_tauFL - PFchH_index_offset);
      get(dnn::pfCand_chHad_vertex_dz_tauFL) =
          sp.scale(chH_cand.vertex().z() - pv.position().z() - tau_funcs.getFlightLength(tau, tau_index).z(),
                   dnn::pfCand_chHad_vertex_dz_tauFL - PFchH_index_offset);

      const bool hasTrackDetails = candFunc::getHasTrackDetails(chH_cand);
      if (hasTrackDetails) {
        get(dnn::pfCand_chHad_hasTrackDetails) =
            sp.scale(hasTrackDetails, dnn::pfCand_chHad_hasTrackDetails - PFchH_index_offset);
        get(dnn::pfCand_chHad_dxy) =
            sp.scale(candFunc::getTauDxy(chH_cand), dnn::pfCand_chHad_dxy - PFchH_index_offset);
        get(dnn::pfCand_chHad_dxy_sig) = sp.scale(std::abs(candFunc::getTauDxy(chH_cand)) / chH_cand.dxyError(),
                                                  dnn::pfCand_chHad_dxy_sig - PFchH_index_offset);
        get(dnn::pfCand_chHad_dz) = sp.scale(candFunc::getTauDz(chH_cand), dnn::pfCand_chHad_dz - PFchH_index_offset);
        get(dnn::pfCand_chHad_dz_sig) = sp.scale(std::abs(candFunc::getTauDz(chH_cand)) / chH_cand.dzError(),
                                                 dnn::pfCand_chHad_dz_sig - PFchH_index_offset);
        get(dnn::pfCand_chHad_track_chi2_ndof) =
            candFunc::getPseudoTrack(chH_cand).ndof() > 0
                ? sp.scale(candFunc::getPseudoTrack(chH_cand).chi2() / candFunc::getPseudoTrack(chH_cand).ndof(),
                           dnn::pfCand_chHad_track_chi2_ndof - PFchH_index_offset)
                : 0;
        get(dnn::pfCand_chHad_track_ndof) =
            candFunc::getPseudoTrack(chH_cand).ndof() > 0
                ? sp.scale(candFunc::getPseudoTrack(chH_cand).ndof(), dnn::pfCand_chHad_track_ndof - PFchH_index_offset)
                : 0;
      }
      float hcal_fraction = candFunc::getHCalFraction(chH_cand, disable_hcalFraction_workaround_);
      get(dnn::pfCand_chHad_hcalFraction) =
          sp.scale(hcal_fraction, dnn::pfCand_chHad_hcalFraction - PFchH_index_offset);
      get(dnn::pfCand_chHad_rawCaloFraction) =
          sp.scale(candFunc::getRawCaloFraction(chH_cand), dnn::pfCand_chHad_rawCaloFraction - PFchH_index_offset);
    }
    if (valid_nH) {
      const sc::ScalingParams& sp = scalingParamsMap_->at(std::make_pair(ft_PFnH, is_inner));
      size_t index_nH = cell_map.at(CellObjectType::PfCand_neutralHadron);
      const auto& nH_cand = dynamic_cast<const CandidateCastType&>(pfCands.at(index_nH));

      get(dnn::pfCand_nHad_valid) = sp.scale(valid_nH, dnn::pfCand_nHad_valid - PFnH_index_offset);
      get(dnn::pfCand_nHad_rel_pt) =
          sp.scale(nH_cand.polarP4().pt() / tau.polarP4().pt(), dnn::pfCand_nHad_rel_pt - PFnH_index_offset);
      get(dnn::pfCand_nHad_deta) =
          sp.scale(nH_cand.polarP4().eta() - tau.polarP4().eta(), dnn::pfCand_nHad_deta - PFnH_index_offset);
      get(dnn::pfCand_nHad_dphi) =
          sp.scale(dPhi(tau.polarP4(), nH_cand.polarP4()), dnn::pfCand_nHad_dphi - PFnH_index_offset);
      get(dnn::pfCand_nHad_puppiWeight) = is_inner ? sp.scale(candFunc::getPuppiWeight(nH_cand, 0.9798355f),
                                                              dnn::pfCand_nHad_puppiWeight - PFnH_index_offset)
                                                   : sp.scale(candFunc::getPuppiWeight(nH_cand, 0.7813260f),
                                                              dnn::pfCand_nHad_puppiWeight - PFnH_index_offset);
      get(dnn::pfCand_nHad_puppiWeightNoLep) = is_inner
                                                   ? sp.scale(candFunc::getPuppiWeightNoLep(nH_cand, 0.9046796f),
                                                              dnn::pfCand_nHad_puppiWeightNoLep - PFnH_index_offset)
                                                   : sp.scale(candFunc::getPuppiWeightNoLep(nH_cand, 0.6554860f),
                                                              dnn::pfCand_nHad_puppiWeightNoLep - PFnH_index_offset);
      float hcal_fraction = candFunc::getHCalFraction(nH_cand, disable_hcalFraction_workaround_);
      get(dnn::pfCand_nHad_hcalFraction) = sp.scale(hcal_fraction, dnn::pfCand_nHad_hcalFraction - PFnH_index_offset);
    }
  }

  template <typename dnn, typename CandidateCastType, typename TauCastType>
  tensorflow::Tensor createInputsV1(const TauCastType& tau,
                                    const size_t tau_index,
                                    const edm::RefToBase<reco::BaseTau> tau_ref,
                                    const std::vector<pat::Electron>* electrons,
                                    const std::vector<pat::Muon>* muons,
                                    TauFunc tau_funcs) const {
    static constexpr bool check_all_set = false;
    static constexpr float default_value_for_set_check = -42;

    tensorflow::Tensor inputs(tensorflow::DT_FLOAT, {1, dnn_inputs_v1::NumberOfInputs});
    const auto& get = [&](int var_index) -> float& { return inputs.matrix<float>()(0, var_index); };
    auto leadChargedHadrCand = dynamic_cast<const CandidateCastType*>(tau.leadChargedHadrCand().get());

    if (check_all_set) {
      for (int var_index = 0; var_index < dnn::NumberOfInputs; ++var_index) {
        get(var_index) = default_value_for_set_check;
      }
    }

    get(dnn::pt) = tau.p4().pt();
    get(dnn::eta) = tau.p4().eta();
    get(dnn::mass) = tau.p4().mass();
    get(dnn::decayMode) = tau.decayMode();
    get(dnn::chargedIsoPtSum) = tau_funcs.getChargedIsoPtSum(tau, tau_ref);
    get(dnn::neutralIsoPtSum) = tau_funcs.getNeutralIsoPtSum(tau, tau_ref);
    get(dnn::neutralIsoPtSumWeight) = tau_funcs.getNeutralIsoPtSumWeight(tau, tau_ref);
    get(dnn::photonPtSumOutsideSignalCone) = tau_funcs.getPhotonPtSumOutsideSignalCone(tau, tau_ref);
    get(dnn::puCorrPtSum) = tau_funcs.getPuCorrPtSum(tau, tau_ref);
    get(dnn::dxy) = tau_funcs.getdxy(tau, tau_index);
    get(dnn::dxy_sig) = tau_funcs.getdxySig(tau, tau_index);
    get(dnn::dz) = leadChargedHadrCand ? candFunc::getTauDz(*leadChargedHadrCand) : default_value;
    get(dnn::ip3d) = tau_funcs.getip3d(tau, tau_index);
    get(dnn::ip3d_sig) = tau_funcs.getip3dSig(tau, tau_index);
    get(dnn::hasSecondaryVertex) = tau_funcs.getHasSecondaryVertex(tau, tau_index);
    get(dnn::flightLength_r) = tau_funcs.getFlightLength(tau, tau_index).R();
    get(dnn::flightLength_dEta) = dEta(tau_funcs.getFlightLength(tau, tau_index), tau.p4());
    get(dnn::flightLength_dPhi) = dPhi(tau_funcs.getFlightLength(tau, tau_index), tau.p4());
    get(dnn::flightLength_sig) = tau_funcs.getFlightLengthSig(tau, tau_index);
    get(dnn::leadChargedHadrCand_pt) = leadChargedHadrCand ? leadChargedHadrCand->p4().Pt() : default_value;
    get(dnn::leadChargedHadrCand_dEta) =
        leadChargedHadrCand ? dEta(leadChargedHadrCand->p4(), tau.p4()) : default_value;
    get(dnn::leadChargedHadrCand_dPhi) =
        leadChargedHadrCand ? dPhi(leadChargedHadrCand->p4(), tau.p4()) : default_value;
    get(dnn::leadChargedHadrCand_mass) = leadChargedHadrCand ? leadChargedHadrCand->p4().mass() : default_value;
    get(dnn::pt_weighted_deta_strip) = reco::tau::pt_weighted_deta_strip(tau, tau.decayMode());
    get(dnn::pt_weighted_dphi_strip) = reco::tau::pt_weighted_dphi_strip(tau, tau.decayMode());
    get(dnn::pt_weighted_dr_signal) = reco::tau::pt_weighted_dr_signal(tau, tau.decayMode());
    get(dnn::pt_weighted_dr_iso) = reco::tau::pt_weighted_dr_iso(tau, tau.decayMode());
    get(dnn::leadingTrackNormChi2) = tau_funcs.getLeadingTrackNormChi2(tau);
    get(dnn::e_ratio) = reco::tau::eratio(tau);
    get(dnn::gj_angle_diff) = calculateGottfriedJacksonAngleDifference(tau, tau_index, tau_funcs);
    get(dnn::n_photons) = reco::tau::n_photons_total(tau);
    get(dnn::emFraction) = tau_funcs.getEmFraction(tau);
    get(dnn::has_gsf_track) = leadChargedHadrCand && std::abs(leadChargedHadrCand->pdgId()) == 11;
    get(dnn::inside_ecal_crack) = isInEcalCrack(tau.p4().Eta());
    auto gsf_ele = findMatchedElectron(tau, electrons, 0.3);
    get(dnn::gsf_ele_matched) = gsf_ele != nullptr;
    get(dnn::gsf_ele_pt) = gsf_ele != nullptr ? gsf_ele->p4().Pt() : default_value;
    get(dnn::gsf_ele_dEta) = gsf_ele != nullptr ? dEta(gsf_ele->p4(), tau.p4()) : default_value;
    get(dnn::gsf_ele_dPhi) = gsf_ele != nullptr ? dPhi(gsf_ele->p4(), tau.p4()) : default_value;
    get(dnn::gsf_ele_mass) = gsf_ele != nullptr ? gsf_ele->p4().mass() : default_value;
    calculateElectronClusterVars(gsf_ele, get(dnn::gsf_ele_Ee), get(dnn::gsf_ele_Egamma));
    get(dnn::gsf_ele_Pin) = gsf_ele != nullptr ? gsf_ele->trackMomentumAtVtx().R() : default_value;
    get(dnn::gsf_ele_Pout) = gsf_ele != nullptr ? gsf_ele->trackMomentumOut().R() : default_value;
    get(dnn::gsf_ele_EtotOverPin) = get(dnn::gsf_ele_Pin) > 0
                                        ? (get(dnn::gsf_ele_Ee) + get(dnn::gsf_ele_Egamma)) / get(dnn::gsf_ele_Pin)
                                        : default_value;
    get(dnn::gsf_ele_Eecal) = gsf_ele != nullptr ? gsf_ele->ecalEnergy() : default_value;
    get(dnn::gsf_ele_dEta_SeedClusterTrackAtCalo) =
        gsf_ele != nullptr ? gsf_ele->deltaEtaSeedClusterTrackAtCalo() : default_value;
    get(dnn::gsf_ele_dPhi_SeedClusterTrackAtCalo) =
        gsf_ele != nullptr ? gsf_ele->deltaPhiSeedClusterTrackAtCalo() : default_value;
    get(dnn::gsf_ele_mvaIn_sigmaEtaEta) = gsf_ele != nullptr ? gsf_ele->mvaInput().sigmaEtaEta : default_value;
    get(dnn::gsf_ele_mvaIn_hadEnergy) = gsf_ele != nullptr ? gsf_ele->mvaInput().hadEnergy : default_value;
    get(dnn::gsf_ele_mvaIn_deltaEta) = gsf_ele != nullptr ? gsf_ele->mvaInput().deltaEta : default_value;

    get(dnn::gsf_ele_Chi2NormGSF) = default_value;
    get(dnn::gsf_ele_GSFNumHits) = default_value;
    get(dnn::gsf_ele_GSFTrackResol) = default_value;
    get(dnn::gsf_ele_GSFTracklnPt) = default_value;
    if (gsf_ele != nullptr && gsf_ele->gsfTrack().isNonnull()) {
      get(dnn::gsf_ele_Chi2NormGSF) = gsf_ele->gsfTrack()->normalizedChi2();
      get(dnn::gsf_ele_GSFNumHits) = gsf_ele->gsfTrack()->numberOfValidHits();
      if (gsf_ele->gsfTrack()->pt() > 0) {
        get(dnn::gsf_ele_GSFTrackResol) = gsf_ele->gsfTrack()->ptError() / gsf_ele->gsfTrack()->pt();
        get(dnn::gsf_ele_GSFTracklnPt) = std::log10(gsf_ele->gsfTrack()->pt());
      }
    }

    get(dnn::gsf_ele_Chi2NormKF) = default_value;
    get(dnn::gsf_ele_KFNumHits) = default_value;
    if (gsf_ele != nullptr && gsf_ele->closestCtfTrackRef().isNonnull()) {
      get(dnn::gsf_ele_Chi2NormKF) = gsf_ele->closestCtfTrackRef()->normalizedChi2();
      get(dnn::gsf_ele_KFNumHits) = gsf_ele->closestCtfTrackRef()->numberOfValidHits();
    }
    get(dnn::leadChargedCand_etaAtEcalEntrance) = tau_funcs.getEtaAtEcalEntrance(tau);
    get(dnn::leadChargedCand_pt) = leadChargedHadrCand->pt();

    get(dnn::leadChargedHadrCand_HoP) = default_value;
    get(dnn::leadChargedHadrCand_EoP) = default_value;
    if (leadChargedHadrCand->pt() > 0) {
      get(dnn::leadChargedHadrCand_HoP) = tau_funcs.getEcalEnergyLeadingChargedHadr(tau) / leadChargedHadrCand->pt();
      get(dnn::leadChargedHadrCand_EoP) = tau_funcs.getHcalEnergyLeadingChargedHadr(tau) / leadChargedHadrCand->pt();
    }

    MuonHitMatchV1 muon_hit_match;
    if (tau.leadPFChargedHadrCand().isNonnull() && tau.leadPFChargedHadrCand()->muonRef().isNonnull())
      muon_hit_match.addMatchedMuon(*tau.leadPFChargedHadrCand()->muonRef(), tau);

    auto matched_muons = muon_hit_match.findMatchedMuons(tau, muons, 0.3, 5);
    for (auto muon : matched_muons)
      muon_hit_match.addMatchedMuon(*muon, tau);
    muon_hit_match.fillTensor<dnn>(get, tau, default_value);

    LorentzVectorXYZ signalChargedHadrCands_sumIn, signalChargedHadrCands_sumOut;
    processSignalPFComponents(tau,
                              tau.signalChargedHadrCands(),
                              signalChargedHadrCands_sumIn,
                              signalChargedHadrCands_sumOut,
                              get(dnn::signalChargedHadrCands_sum_innerSigCone_pt),
                              get(dnn::signalChargedHadrCands_sum_innerSigCone_dEta),
                              get(dnn::signalChargedHadrCands_sum_innerSigCone_dPhi),
                              get(dnn::signalChargedHadrCands_sum_innerSigCone_mass),
                              get(dnn::signalChargedHadrCands_sum_outerSigCone_pt),
                              get(dnn::signalChargedHadrCands_sum_outerSigCone_dEta),
                              get(dnn::signalChargedHadrCands_sum_outerSigCone_dPhi),
                              get(dnn::signalChargedHadrCands_sum_outerSigCone_mass),
                              get(dnn::signalChargedHadrCands_nTotal_innerSigCone),
                              get(dnn::signalChargedHadrCands_nTotal_outerSigCone));

    LorentzVectorXYZ signalNeutrHadrCands_sumIn, signalNeutrHadrCands_sumOut;
    processSignalPFComponents(tau,
                              tau.signalNeutrHadrCands(),
                              signalNeutrHadrCands_sumIn,
                              signalNeutrHadrCands_sumOut,
                              get(dnn::signalNeutrHadrCands_sum_innerSigCone_pt),
                              get(dnn::signalNeutrHadrCands_sum_innerSigCone_dEta),
                              get(dnn::signalNeutrHadrCands_sum_innerSigCone_dPhi),
                              get(dnn::signalNeutrHadrCands_sum_innerSigCone_mass),
                              get(dnn::signalNeutrHadrCands_sum_outerSigCone_pt),
                              get(dnn::signalNeutrHadrCands_sum_outerSigCone_dEta),
                              get(dnn::signalNeutrHadrCands_sum_outerSigCone_dPhi),
                              get(dnn::signalNeutrHadrCands_sum_outerSigCone_mass),
                              get(dnn::signalNeutrHadrCands_nTotal_innerSigCone),
                              get(dnn::signalNeutrHadrCands_nTotal_outerSigCone));

    LorentzVectorXYZ signalGammaCands_sumIn, signalGammaCands_sumOut;
    processSignalPFComponents(tau,
                              tau.signalGammaCands(),
                              signalGammaCands_sumIn,
                              signalGammaCands_sumOut,
                              get(dnn::signalGammaCands_sum_innerSigCone_pt),
                              get(dnn::signalGammaCands_sum_innerSigCone_dEta),
                              get(dnn::signalGammaCands_sum_innerSigCone_dPhi),
                              get(dnn::signalGammaCands_sum_innerSigCone_mass),
                              get(dnn::signalGammaCands_sum_outerSigCone_pt),
                              get(dnn::signalGammaCands_sum_outerSigCone_dEta),
                              get(dnn::signalGammaCands_sum_outerSigCone_dPhi),
                              get(dnn::signalGammaCands_sum_outerSigCone_mass),
                              get(dnn::signalGammaCands_nTotal_innerSigCone),
                              get(dnn::signalGammaCands_nTotal_outerSigCone));

    LorentzVectorXYZ isolationChargedHadrCands_sum;
    processIsolationPFComponents(tau,
                                 tau.isolationChargedHadrCands(),
                                 isolationChargedHadrCands_sum,
                                 get(dnn::isolationChargedHadrCands_sum_pt),
                                 get(dnn::isolationChargedHadrCands_sum_dEta),
                                 get(dnn::isolationChargedHadrCands_sum_dPhi),
                                 get(dnn::isolationChargedHadrCands_sum_mass),
                                 get(dnn::isolationChargedHadrCands_nTotal));

    LorentzVectorXYZ isolationNeutrHadrCands_sum;
    processIsolationPFComponents(tau,
                                 tau.isolationNeutrHadrCands(),
                                 isolationNeutrHadrCands_sum,
                                 get(dnn::isolationNeutrHadrCands_sum_pt),
                                 get(dnn::isolationNeutrHadrCands_sum_dEta),
                                 get(dnn::isolationNeutrHadrCands_sum_dPhi),
                                 get(dnn::isolationNeutrHadrCands_sum_mass),
                                 get(dnn::isolationNeutrHadrCands_nTotal));

    LorentzVectorXYZ isolationGammaCands_sum;
    processIsolationPFComponents(tau,
                                 tau.isolationGammaCands(),
                                 isolationGammaCands_sum,
                                 get(dnn::isolationGammaCands_sum_pt),
                                 get(dnn::isolationGammaCands_sum_dEta),
                                 get(dnn::isolationGammaCands_sum_dPhi),
                                 get(dnn::isolationGammaCands_sum_mass),
                                 get(dnn::isolationGammaCands_nTotal));

    get(dnn::tau_visMass_innerSigCone) = (signalGammaCands_sumIn + signalChargedHadrCands_sumIn).mass();

    if (check_all_set) {
      for (int var_index = 0; var_index < dnn::NumberOfInputs; ++var_index) {
        if (get(var_index) == default_value_for_set_check)
          throw cms::Exception("DeepTauId: variable with index = ") << var_index << " is not set.";
      }
    }

    return inputs;
  }

  static void calculateElectronClusterVars(const pat::Electron* ele, float& elecEe, float& elecEgamma) {
    if (ele) {
      elecEe = elecEgamma = 0;
      auto superCluster = ele->superCluster();
      if (superCluster.isNonnull() && superCluster.isAvailable() && superCluster->clusters().isNonnull() &&
          superCluster->clusters().isAvailable()) {
        for (auto iter = superCluster->clustersBegin(); iter != superCluster->clustersEnd(); ++iter) {
          const double energy = (*iter)->energy();
          if (iter == superCluster->clustersBegin())
            elecEe += energy;
          else
            elecEgamma += energy;
        }
      }
    } else {
      elecEe = elecEgamma = default_value;
    }
  }

  template <typename CandidateCollection, typename TauCastType>
  static void processSignalPFComponents(const TauCastType& tau,
                                        const CandidateCollection& candidates,
                                        LorentzVectorXYZ& p4_inner,
                                        LorentzVectorXYZ& p4_outer,
                                        float& pt_inner,
                                        float& dEta_inner,
                                        float& dPhi_inner,
                                        float& m_inner,
                                        float& pt_outer,
                                        float& dEta_outer,
                                        float& dPhi_outer,
                                        float& m_outer,
                                        float& n_inner,
                                        float& n_outer) {
    p4_inner = LorentzVectorXYZ(0, 0, 0, 0);
    p4_outer = LorentzVectorXYZ(0, 0, 0, 0);
    n_inner = 0;
    n_outer = 0;

    const double innerSigCone_radius = getInnerSignalConeRadius(tau.pt());
    for (const auto& cand : candidates) {
      const double dR = reco::deltaR(cand->p4(), tau.leadChargedHadrCand()->p4());
      const bool isInside_innerSigCone = dR < innerSigCone_radius;
      if (isInside_innerSigCone) {
        p4_inner += cand->p4();
        ++n_inner;
      } else {
        p4_outer += cand->p4();
        ++n_outer;
      }
    }

    pt_inner = n_inner != 0 ? p4_inner.Pt() : default_value;
    dEta_inner = n_inner != 0 ? dEta(p4_inner, tau.p4()) : default_value;
    dPhi_inner = n_inner != 0 ? dPhi(p4_inner, tau.p4()) : default_value;
    m_inner = n_inner != 0 ? p4_inner.mass() : default_value;

    pt_outer = n_outer != 0 ? p4_outer.Pt() : default_value;
    dEta_outer = n_outer != 0 ? dEta(p4_outer, tau.p4()) : default_value;
    dPhi_outer = n_outer != 0 ? dPhi(p4_outer, tau.p4()) : default_value;
    m_outer = n_outer != 0 ? p4_outer.mass() : default_value;
  }

  template <typename CandidateCollection, typename TauCastType>
  static void processIsolationPFComponents(const TauCastType& tau,
                                           const CandidateCollection& candidates,
                                           LorentzVectorXYZ& p4,
                                           float& pt,
                                           float& d_eta,
                                           float& d_phi,
                                           float& m,
                                           float& n) {
    p4 = LorentzVectorXYZ(0, 0, 0, 0);
    n = 0;

    for (const auto& cand : candidates) {
      p4 += cand->p4();
      ++n;
    }

    pt = n != 0 ? p4.Pt() : default_value;
    d_eta = n != 0 ? dEta(p4, tau.p4()) : default_value;
    d_phi = n != 0 ? dPhi(p4, tau.p4()) : default_value;
    m = n != 0 ? p4.mass() : default_value;
  }

  static double getInnerSignalConeRadius(double pt) {
    static constexpr double min_pt = 30., min_radius = 0.05, cone_opening_coef = 3.;
    // This is equivalent of the original formula (std::max(std::min(0.1, 3.0/pt), 0.05)
    return std::max(cone_opening_coef / std::max(pt, min_pt), min_radius);
  }

  // Copied from https://github.com/cms-sw/cmssw/blob/CMSSW_9_4_X/RecoTauTag/RecoTau/plugins/PATTauDiscriminationByMVAIsolationRun2.cc#L218
  template <typename TauCastType>
  static bool calculateGottfriedJacksonAngleDifference(const TauCastType& tau,
                                                       const size_t tau_index,
                                                       double& gj_diff,
                                                       TauFunc tau_funcs) {
    if (tau_funcs.getHasSecondaryVertex(tau, tau_index)) {
      static constexpr double mTau = 1.77682;
      const double mAOne = tau.p4().M();
      const double pAOneMag = tau.p();
      const double argumentThetaGJmax = (std::pow(mTau, 2) - std::pow(mAOne, 2)) / (2 * mTau * pAOneMag);
      const double argumentThetaGJmeasured = tau.p4().Vect().Dot(tau_funcs.getFlightLength(tau, tau_index)) /
                                             (pAOneMag * tau_funcs.getFlightLength(tau, tau_index).R());
      if (std::abs(argumentThetaGJmax) <= 1. && std::abs(argumentThetaGJmeasured) <= 1.) {
        double thetaGJmax = std::asin(argumentThetaGJmax);
        double thetaGJmeasured = std::acos(argumentThetaGJmeasured);
        gj_diff = thetaGJmeasured - thetaGJmax;
        return true;
      }
    }
    return false;
  }

  template <typename TauCastType>
  static float calculateGottfriedJacksonAngleDifference(const TauCastType& tau,
                                                        const size_t tau_index,
                                                        TauFunc tau_funcs) {
    double gj_diff;
    if (calculateGottfriedJacksonAngleDifference(tau, tau_index, gj_diff, tau_funcs))
      return static_cast<float>(gj_diff);
    return default_value;
  }

  static bool isInEcalCrack(double eta) {
    const double abs_eta = std::abs(eta);
    return abs_eta > 1.46 && abs_eta < 1.558;
  }

  template <typename TauCastType>
  static const pat::Electron* findMatchedElectron(const TauCastType& tau,
                                                  const std::vector<pat::Electron>* electrons,
                                                  double deltaR) {
    const double dR2 = deltaR * deltaR;
    const pat::Electron* matched_ele = nullptr;
    for (const auto& ele : *electrons) {
      if (reco::deltaR2(tau.p4(), ele.p4()) < dR2 && (!matched_ele || matched_ele->pt() < ele.pt())) {
        matched_ele = &ele;
      }
    }
    return matched_ele;
  }

private:
  edm::EDGetTokenT<std::vector<pat::Electron>> electrons_token_;
  edm::EDGetTokenT<std::vector<pat::Muon>> muons_token_;
  edm::EDGetTokenT<double> rho_token_;
  edm::EDGetTokenT<reco::TauDiscriminatorContainer> basicTauDiscriminators_inputToken_;
  edm::EDGetTokenT<reco::TauDiscriminatorContainer> basicTauDiscriminatorsdR03_inputToken_;
  edm::EDGetTokenT<edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>>
      pfTauTransverseImpactParameters_token_;
  std::string input_layer_, output_layer_;
  const unsigned version_;
  const unsigned sub_version_;
  const int debug_level;
  const bool disable_dxy_pca_;
  const bool disable_hcalFraction_workaround_;
  const bool disable_CellIndex_workaround_;
  std::unique_ptr<tensorflow::Tensor> tauBlockTensor_;
  std::array<std::unique_ptr<tensorflow::Tensor>, 2> eGammaTensor_, muonTensor_, hadronsTensor_, convTensor_,
      zeroOutputTensor_;
  const std::map<std::pair<deep_tau::Scaling::FeatureT, bool>, deep_tau::Scaling::ScalingParams>* scalingParamsMap_;
  const bool save_inputs_;
  std::ofstream* json_file_;
  bool is_first_block_;
  int file_counter_;
  std::vector<int> tauInputs_indices_;

  //boolean to check if discriminator indices are already mapped
  bool discrIndicesMapped_ = false;
  std::map<BasicDiscriminator, size_t> basicDiscrIndexMap_;
  std::map<BasicDiscriminator, size_t> basicDiscrdR03IndexMap_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeepTauId);
