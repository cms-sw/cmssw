#ifndef DeepTauIdBase_H
#define DeepTauIdBase_H

/*
 * \class DeepTauIdBase
 *
 * Base class for DeepTauId producers: DeepTauId and DeepTauIdSONIC
 *
 */

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/TauReco/interface/TauDiscriminatorContainer.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/PatCandidates/interface/PATTauDiscriminator.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"
#include "FWCore/Common/interface/Provenance.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauClusterVariables.h"
#include "RecoTauTag/RecoTau/interface/DeepTauScaling.h"
#include "RecoTauTag/RecoTau/interface/TauWPThreshold.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "tensorflow/core/util/memmapped_file_system.h"

#include <Math/VectorUtil.h>
#include <map>
#include <fstream>
#include "oneapi/tbb/concurrent_unordered_set.h"

namespace deep_tau {
  enum BasicDiscriminator {
    ChargedIsoPtSum,
    NeutralIsoPtSum,
    NeutralIsoPtSumWeight,
    FootprintCorrection,
    PhotonPtSumOutsideSignalCone,
    PUcorrPtSum
  };

  constexpr int NumberOfOutputs = 4;
}  // namespace deep_tau

namespace {

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
      inline std::vector<int> varsToDrop = {
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

  inline float getTauID(const pat::Tau& tau,
                        const std::string& tauID,
                        float default_value = -999.,
                        bool assert_input = true) {
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

    using BasicDiscr = deep_tau::BasicDiscriminator;
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
    inline auto getTauDz(const reco::PFCandidate& cand) { return cand.bestTrack()->dz(); }
    inline auto getTauDz(const pat::PackedCandidate& cand) { return cand.dz(); }
    inline auto getTauDZSigValid(const reco::PFCandidate& cand) {
      return cand.bestTrack() != nullptr && std::isnormal(cand.bestTrack()->dz()) && std::isnormal(cand.dzError()) &&
             cand.dzError() > 0;
    }
    inline auto getTauDZSigValid(const pat::PackedCandidate& cand) {
      return cand.hasTrackDetails() && std::isnormal(cand.dz()) && std::isnormal(cand.dzError()) && cand.dzError() > 0;
    }
    inline auto getTauDxy(const reco::PFCandidate& cand) { return cand.bestTrack()->dxy(); }
    inline auto getTauDxy(const pat::PackedCandidate& cand) { return cand.dxy(); }
    inline auto getPvAssocationQuality(const reco::PFCandidate& cand) { return 0.7013f; }
    inline auto getPvAssocationQuality(const pat::PackedCandidate& cand) { return cand.pvAssociationQuality(); }
    inline auto getPuppiWeight(const reco::PFCandidate& cand, const float aod_value) { return aod_value; }
    inline auto getPuppiWeight(const pat::PackedCandidate& cand, const float aod_value) { return cand.puppiWeight(); }
    inline auto getPuppiWeightNoLep(const reco::PFCandidate& cand, const float aod_value) { return aod_value; }
    inline auto getPuppiWeightNoLep(const pat::PackedCandidate& cand, const float aod_value) {
      return cand.puppiWeightNoLep();
    }
    inline auto getLostInnerHits(const reco::PFCandidate& cand, float default_value) {
      return cand.bestTrack() != nullptr
                 ? cand.bestTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS)
                 : default_value;
    }
    inline auto getLostInnerHits(const pat::PackedCandidate& cand, float default_value) { return cand.lostInnerHits(); }
    inline auto getNumberOfPixelHits(const reco::PFCandidate& cand, float default_value) {
      return cand.bestTrack() != nullptr
                 ? cand.bestTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS)
                 : default_value;
    }
    inline auto getNumberOfPixelHits(const pat::PackedCandidate& cand, float default_value) {
      return cand.numberOfPixelHits();
    }
    inline auto getHasTrackDetails(const reco::PFCandidate& cand) { return cand.bestTrack() != nullptr; }
    inline auto getHasTrackDetails(const pat::PackedCandidate& cand) { return cand.hasTrackDetails(); }
    inline auto getPseudoTrack(const reco::PFCandidate& cand) { return *cand.bestTrack(); }
    inline auto getPseudoTrack(const pat::PackedCandidate& cand) { return cand.pseudoTrack(); }
    inline auto getFromPV(const reco::PFCandidate& cand) { return 0.9994f; }
    inline auto getFromPV(const pat::PackedCandidate& cand) { return cand.fromPV(); }
    inline auto getHCalFraction(const reco::PFCandidate& cand, bool disable_hcalFraction_workaround) {
      return cand.rawHcalEnergy() / (cand.rawHcalEnergy() + cand.rawEcalEnergy());
    }
    inline auto getHCalFraction(const pat::PackedCandidate& cand, bool disable_hcalFraction_workaround) {
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
    inline auto getRawCaloFraction(const reco::PFCandidate& cand) {
      return (cand.rawEcalEnergy() + cand.rawHcalEnergy()) / cand.energy();
    }
    inline auto getRawCaloFraction(const pat::PackedCandidate& cand) { return cand.rawCaloFraction(); }
  };  // namespace candFunc

  template <typename LVector1, typename LVector2>
  float dEta(const LVector1& p4, const LVector2& tau_p4) {
    return static_cast<float>(p4.eta() - tau_p4.eta());
  }

  template <typename LVector1, typename LVector2>
  float dPhi(const LVector1& p4_1, const LVector2& p4_2) {
    return static_cast<float>(reco::deltaPhi(p4_2.phi(), p4_1.phi()));
  }

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
  inline CellObjectType GetCellObjectType(const Object&);
  template <>
  inline CellObjectType GetCellObjectType(const pat::Electron&) {
    return CellObjectType::Electron;
  }
  template <>
  inline CellObjectType GetCellObjectType(const pat::Muon&) {
    return CellObjectType::Muon;
  }

  template <>
  inline CellObjectType GetCellObjectType(reco::Candidate const& cand) {
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

template <class Producer>
class DeepTauIdBase : public Producer {
public:
  using TauDiscriminator = reco::TauDiscriminatorContainer;
  using TauCollection = edm::View<reco::BaseTau>;
  using CandidateCollection = edm::View<reco::Candidate>;
  using TauRef = edm::Ref<TauCollection>;
  using TauRefProd = edm::RefProd<TauCollection>;
  using ElectronCollection = pat::ElectronCollection;
  using MuonCollection = pat::MuonCollection;
  using LorentzVectorXYZ = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;
  using Cutter = tau::TauWPThreshold;
  using CutterPtr = std::unique_ptr<Cutter>;
  using WPList = std::vector<CutterPtr>;

  struct IDOutput {
    std::vector<size_t> num_, den_;

    IDOutput(const std::vector<size_t>& num, const std::vector<size_t>& den) : num_(num), den_(den) {}

    // for direct inference, read the output tensorflow::Tensor
    float read_value(const tensorflow::Tensor& pred, size_t tau_index, size_t elem) const {
      return pred.matrix<float>()(tau_index, elem);
    }
    // for SONIC, read the output vector
    float read_value(const std::vector<std::vector<float>>& pred, size_t tau_index, size_t elem) const {
      return pred.at(tau_index).at(elem);
    }

    template <typename PredType>
    std::unique_ptr<TauDiscriminator> get_value(const edm::Handle<TauCollection>& taus,
                                                const PredType& pred,
                                                const WPList* working_points,
                                                bool is_online) const {
      std::vector<reco::SingleTauDiscriminatorContainer> outputbuffer(taus->size());

      for (size_t tau_index = 0; tau_index < taus->size(); ++tau_index) {
        float x = 0;
        for (size_t num_elem : num_)
          x += read_value(pred, tau_index, num_elem);
        if (x != 0 && !den_.empty()) {
          float den_val = 0;
          for (size_t den_elem : den_)
            den_val += read_value(pred, tau_index, den_elem);
          x = den_val != 0 ? x / den_val : std::numeric_limits<float>::max();
        }
        outputbuffer[tau_index].rawValues.push_back(x);
        if (working_points) {
          for (const auto& wp : *working_points) {
            const bool pass = x > (*wp)(taus->at(tau_index), is_online);
            outputbuffer[tau_index].workingPoints.push_back(pass);
          }
        }
      }
      std::unique_ptr<TauDiscriminator> output = std::make_unique<TauDiscriminator>();
      reco::TauDiscriminatorContainer::Filler filler(*output);
      filler.insert(taus, outputbuffer.begin(), outputbuffer.end());
      filler.fill();
      return output;
    }
  };

  using IDOutputCollection = std::map<std::string, IDOutput>;

  static constexpr float default_value = -999.;

  static const IDOutputCollection& GetIDOutputs() {
    static constexpr size_t e_index = 0, mu_index = 1, tau_index = 2, jet_index = 3;
    static const IDOutputCollection idoutputs_ = {
        {"VSe", IDOutput({tau_index}, {e_index, tau_index})},
        {"VSmu", IDOutput({tau_index}, {mu_index, tau_index})},
        {"VSjet", IDOutput({tau_index}, {jet_index, tau_index})},
    };
    return idoutputs_;
  }

  using BasicDiscriminator = deep_tau::BasicDiscriminator;

  const std::map<BasicDiscriminator, size_t> matchDiscriminatorIndices(
      edm::Event const& event,
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

  static void fillDescriptionsHelper(edm::ParameterSetDescription& desc) {
    desc.add<edm::InputTag>("electrons", edm::InputTag("slimmedElectrons"));
    desc.add<edm::InputTag>("muons", edm::InputTag("slimmedMuons"));
    desc.add<edm::InputTag>("taus", edm::InputTag("slimmedTaus"));
    desc.add<edm::InputTag>("pfcands", edm::InputTag("packedPFCandidates"));
    desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
    desc.add<edm::InputTag>("rho", edm::InputTag("fixedGridRhoAll"));
    desc.add<bool>("mem_mapped", false);
    desc.add<unsigned>("year", 2017);
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
  }

public:
  DeepTauIdBase(const edm::ParameterSet& cfg)
      : Producer(cfg),
        tausToken_(this->template consumes<TauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
        pfcandToken_(this->template consumes<CandidateCollection>(cfg.getParameter<edm::InputTag>("pfcands"))),
        vtxToken_(this->template consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices"))),
        is_online_(cfg.getParameter<bool>("is_online")),
        idoutputs_(GetIDOutputs()),
        electrons_token_(
            this->template consumes<std::vector<pat::Electron>>(cfg.getParameter<edm::InputTag>("electrons"))),
        muons_token_(this->template consumes<std::vector<pat::Muon>>(cfg.getParameter<edm::InputTag>("muons"))),
        rho_token_(this->template consumes<double>(cfg.getParameter<edm::InputTag>("rho"))),
        basicTauDiscriminators_inputToken_(this->template consumes<reco::TauDiscriminatorContainer>(
            cfg.getUntrackedParameter<edm::InputTag>("basicTauDiscriminators"))),
        basicTauDiscriminatorsdR03_inputToken_(this->template consumes<reco::TauDiscriminatorContainer>(
            cfg.getUntrackedParameter<edm::InputTag>("basicTauDiscriminatorsdR03"))),
        pfTauTransverseImpactParameters_token_(
            this->template consumes<
                edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>>(
                cfg.getParameter<edm::InputTag>("pfTauTransverseImpactParameters"))),
        year_(cfg.getParameter<unsigned>("year")),
        version_(cfg.getParameter<unsigned>("version")),
        sub_version_(cfg.getParameter<unsigned>("sub_version")),
        debug_level(cfg.getParameter<int>("debug_level")),
        disable_dxy_pca_(cfg.getParameter<bool>("disable_dxy_pca")),
        disable_hcalFraction_workaround_(cfg.getParameter<bool>("disable_hcalFraction_workaround")),
        disable_CellIndex_workaround_(cfg.getParameter<bool>("disable_CellIndex_workaround")),
        save_inputs_(cfg.getParameter<bool>("save_inputs")),
        json_file_(nullptr),
        file_counter_(0) {
    for (const auto& output_desc : idoutputs_) {
      this->template produces<TauDiscriminator>(output_desc.first);
      const auto& cut_list = cfg.getParameter<std::vector<std::string>>(output_desc.first + "WP");
      for (const std::string& cut_str : cut_list) {
        workingPoints_[output_desc.first].push_back(std::make_unique<Cutter>(cut_str));
      }
    }

    // prediscriminant operator
    // require the tau to pass the following prediscriminants
    const edm::ParameterSet& prediscriminantConfig = cfg.getParameter<edm::ParameterSet>("Prediscriminants");

    // determine boolean operator used on the prediscriminants
    std::string pdBoolOperator = prediscriminantConfig.getParameter<std::string>("BooleanOperator");
    // convert string to lowercase
    transform(pdBoolOperator.begin(), pdBoolOperator.end(), pdBoolOperator.begin(), ::tolower);

    if (pdBoolOperator == "and") {
      andPrediscriminants_ = 0x1;  //use chars instead of bools so we can do a bitwise trick later
    } else if (pdBoolOperator == "or") {
      andPrediscriminants_ = 0x0;
    } else {
      throw cms::Exception("TauDiscriminationProducerBase")
          << "PrediscriminantBooleanOperator defined incorrectly, options are: AND,OR";
    }

    // get the list of prediscriminants
    std::vector<std::string> prediscriminantsNames =
        prediscriminantConfig.getParameterNamesForType<edm::ParameterSet>();

    for (auto const& iDisc : prediscriminantsNames) {
      const edm::ParameterSet& iPredisc = prediscriminantConfig.getParameter<edm::ParameterSet>(iDisc);
      const edm::InputTag& label = iPredisc.getParameter<edm::InputTag>("Producer");
      double cut = iPredisc.getParameter<double>("cut");

      if (is_online_) {
        TauDiscInfo<reco::PFTauDiscriminator> thisDiscriminator;
        thisDiscriminator.label = label;
        thisDiscriminator.cut = cut;
        thisDiscriminator.disc_token = this->template consumes<reco::PFTauDiscriminator>(label);
        recoPrediscriminants_.push_back(thisDiscriminator);
      } else {
        TauDiscInfo<pat::PATTauDiscriminator> thisDiscriminator;
        thisDiscriminator.label = label;
        thisDiscriminator.cut = cut;
        thisDiscriminator.disc_token = this->template consumes<pat::PATTauDiscriminator>(label);
        patPrediscriminants_.push_back(thisDiscriminator);
      }
    }
    if (version_ == 2) {
      using namespace dnn_inputs_v2;
      namespace sc = deep_tau::Scaling;
      tauInputs_indices_.resize(TauBlockInputs::NumberOfInputs);
      std::iota(std::begin(tauInputs_indices_), std::end(tauInputs_indices_), 0);

      if (sub_version_ == 1) {
        scalingParamsMap_ = &sc::scalingParamsMap_v2p1;
      } else if (sub_version_ == 5) {
        std::sort(TauBlockInputs::varsToDrop.begin(), TauBlockInputs::varsToDrop.end());
        for (auto v : TauBlockInputs::varsToDrop) {
          tauInputs_indices_.at(v) = TauBlockInputs::NumberOfInputs - TauBlockInputs::varsToDrop.size();
          for (std::size_t i = v + 1; i < TauBlockInputs::NumberOfInputs; ++i)
            tauInputs_indices_.at(i) -= 1;  // shift all the following indices by 1
        }
        if (year_ == 2026) {
          scalingParamsMap_ = &sc::scalingParamsMap_PhaseIIv2p5;
        } else {
          scalingParamsMap_ = &sc::scalingParamsMap_v2p5;
        }
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
    } else {
      throw cms::Exception("DeepTauId") << "version " << version_ << " is not supported.";
    }
  }

  template <typename ConsumeType>
  struct TauDiscInfo {
    edm::InputTag label;
    edm::Handle<ConsumeType> handle;
    edm::EDGetTokenT<ConsumeType> disc_token;
    double cut;
    void fill(const edm::Event& evt) { evt.getByToken(disc_token, handle); }
  };

  // select boolean operation on prediscriminants (and = 0x01, or = 0x00)
  uint8_t andPrediscriminants_;
  std::vector<TauDiscInfo<pat::PATTauDiscriminator>> patPrediscriminants_;
  std::vector<TauDiscInfo<reco::PFTauDiscriminator>> recoPrediscriminants_;

protected:
  static constexpr float pi = M_PI;

  template <typename PredType>
  void createOutputs(edm::Event& event, const PredType& pred, edm::Handle<TauCollection> taus) {
    for (const auto& output_desc : idoutputs_) {
      const WPList* working_points = nullptr;
      if (workingPoints_.find(output_desc.first) != workingPoints_.end()) {
        working_points = &workingPoints_.at(output_desc.first);
      }
      auto result = output_desc.second.get_value(taus, pred, working_points, is_online_);
      event.put(std::move(result), output_desc.first);
    }
  }

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

protected:
  // load prediscriminators
  void loadPrediscriminants(edm::Event const& event, edm::Handle<TauCollection> const& taus) {
    edm::ProductID tauProductID = taus.id();
    size_t nPrediscriminants =
        patPrediscriminants_.empty() ? recoPrediscriminants_.size() : patPrediscriminants_.size();
    for (size_t iDisc = 0; iDisc < nPrediscriminants; ++iDisc) {
      edm::ProductID discKeyId;
      if (is_online_) {
        recoPrediscriminants_[iDisc].fill(event);
        discKeyId = recoPrediscriminants_[iDisc].handle->keyProduct().id();
      } else {
        patPrediscriminants_[iDisc].fill(event);
        discKeyId = patPrediscriminants_[iDisc].handle->keyProduct().id();
      }

      // Check to make sure the product is correct for the discriminator.
      // If not, throw a more informative exception.
      if (tauProductID != discKeyId) {
        throw cms::Exception("MisconfiguredPrediscriminant")
            << "The tau collection has product ID: " << tauProductID
            << " but the pre-discriminator is keyed with product ID: " << discKeyId << std::endl;
      }
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

  template <typename CandidateCastType, typename TauCastType, typename TauBlockType>
  void createTauBlockInputs(const TauCastType& tau,
                            const size_t& tau_index,
                            const edm::RefToBase<reco::BaseTau> tau_ref,
                            const reco::Vertex& pv,
                            double rho,
                            TauFunc tau_funcs,
                            TauBlockType& tauBlockInputs) {
    namespace dnn = dnn_inputs_v2::TauBlockInputs;
    namespace sc = deep_tau::Scaling;
    namespace candFunc = candFunc;
    sc::FeatureT ft = sc::FeatureT::TauFlat;
    const sc::ScalingParams& sp = scalingParamsMap_->at(std::make_pair(ft, false));

    const auto& get = [&](int var_index) -> float& {
      if constexpr (std::is_same_v<TauBlockType, std::vector<float>::iterator>) {
        return *(tauBlockInputs + tauInputs_indices_.at(var_index));
      } else {
        return ((tensorflow::Tensor)tauBlockInputs).matrix<float>()(0, tauInputs_indices_.at(var_index));
      }
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

  template <typename CandidateCastType, typename TauCastType, typename EgammaBlockType>
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
                               bool is_inner,
                               EgammaBlockType& egammaBlockInputs) {
    namespace dnn = dnn_inputs_v2::EgammaBlockInputs;
    namespace sc = deep_tau::Scaling;
    namespace candFunc = candFunc;
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

    const auto& get = [&](int var_index) -> float& {
      if constexpr (std::is_same_v<EgammaBlockType, std::vector<float>::iterator>) {
        return *(egammaBlockInputs + var_index);
      } else {
        return ((tensorflow::Tensor)egammaBlockInputs).tensor<float, 4>()(idx, 0, 0, var_index);
      }
    };

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
      const bool mva_valid =
          (ele.mvaInput().earlyBrem > -2) ||
          (year_ !=
           2026);  // Known issue that input can be invalid in Phase2 samples (early/lateBrem==-2, hadEnergy==0, sigmaEtaEta/deltaEta==3.40282e+38). Unknown if also in Run2/3, so don't change there
      if (mva_valid) {
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
      }
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

  template <typename CandidateCastType, typename TauCastType, typename MuonBlockType>
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
                             bool is_inner,
                             MuonBlockType& muonBlockInputs) {
    namespace dnn = dnn_inputs_v2::MuonBlockInputs;
    namespace sc = deep_tau::Scaling;
    namespace candFunc = candFunc;
    using MuonHitMatchV2 = MuonHitMatchV2;
    sc::FeatureT ft_global = sc::FeatureT::GridGlobal;
    sc::FeatureT ft_PFmu = sc::FeatureT::PfCand_muon;
    sc::FeatureT ft_mu = sc::FeatureT::Muon;

    // needed to remap indices from scaling vectors to those from dnn_inputs_v2::MuonBlockInputs
    int PFmu_index_offset = scalingParamsMap_->at(std::make_pair(ft_global, false)).mean_.size();
    int mu_index_offset = PFmu_index_offset + scalingParamsMap_->at(std::make_pair(ft_PFmu, false)).mean_.size();

    const auto& get = [&](int var_index) -> float& {
      if constexpr (std::is_same_v<MuonBlockType, std::vector<float>::iterator>) {
        return *(muonBlockInputs + var_index);
      } else {
        return ((tensorflow::Tensor)muonBlockInputs).tensor<float, 4>()(idx, 0, 0, var_index);
      }
    };

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

  template <typename CandidateCastType, typename TauCastType, typename HadronBlockType>
  void createHadronsBlockInputs(unsigned idx,
                                const TauCastType& tau,
                                const size_t tau_index,
                                const edm::RefToBase<reco::BaseTau> tau_ref,
                                const reco::Vertex& pv,
                                double rho,
                                const edm::View<reco::Candidate>& pfCands,
                                const Cell& cell_map,
                                TauFunc tau_funcs,
                                bool is_inner,
                                HadronBlockType& hadronBlockInputs) {
    namespace dnn = dnn_inputs_v2::HadronBlockInputs;
    namespace sc = deep_tau::Scaling;
    namespace candFunc = candFunc;
    sc::FeatureT ft_global = sc::FeatureT::GridGlobal;
    sc::FeatureT ft_PFchH = sc::FeatureT::PfCand_chHad;
    sc::FeatureT ft_PFnH = sc::FeatureT::PfCand_nHad;

    // needed to remap indices from scaling vectors to those from dnn_inputs_v2::HadronBlockInputs
    int PFchH_index_offset = scalingParamsMap_->at(std::make_pair(ft_global, false)).mean_.size();
    int PFnH_index_offset = PFchH_index_offset + scalingParamsMap_->at(std::make_pair(ft_PFchH, false)).mean_.size();

    const auto& get = [&](int var_index) -> float& {
      if constexpr (std::is_same_v<HadronBlockType, std::vector<float>::iterator>) {
        return *(hadronBlockInputs + var_index);
      } else {
        return ((tensorflow::Tensor)hadronBlockInputs).tensor<float, 4>()(idx, 0, 0, var_index);
      }
    };

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

protected:
  edm::EDGetTokenT<TauCollection> tausToken_;
  edm::EDGetTokenT<CandidateCollection> pfcandToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  std::map<std::string, WPList> workingPoints_;
  const bool is_online_;
  IDOutputCollection idoutputs_;

  const std::map<BasicDiscriminator, std::string> stringFromDiscriminator_{
      {BasicDiscriminator::ChargedIsoPtSum, "ChargedIsoPtSum"},
      {BasicDiscriminator::NeutralIsoPtSum, "NeutralIsoPtSum"},
      {BasicDiscriminator::NeutralIsoPtSumWeight, "NeutralIsoPtSumWeight"},
      {BasicDiscriminator::FootprintCorrection, "TauFootprintCorrection"},
      {BasicDiscriminator::PhotonPtSumOutsideSignalCone, "PhotonPtSumOutsideSignalCone"},
      {BasicDiscriminator::PUcorrPtSum, "PUcorrPtSum"}};
  const std::vector<BasicDiscriminator> requiredBasicDiscriminators_{BasicDiscriminator::ChargedIsoPtSum,
                                                                     BasicDiscriminator::NeutralIsoPtSum,
                                                                     BasicDiscriminator::NeutralIsoPtSumWeight,
                                                                     BasicDiscriminator::PhotonPtSumOutsideSignalCone,
                                                                     BasicDiscriminator::PUcorrPtSum};
  const std::vector<BasicDiscriminator> requiredBasicDiscriminatorsdR03_{
      BasicDiscriminator::ChargedIsoPtSum,
      BasicDiscriminator::NeutralIsoPtSum,
      BasicDiscriminator::NeutralIsoPtSumWeight,
      BasicDiscriminator::PhotonPtSumOutsideSignalCone,
      BasicDiscriminator::FootprintCorrection};

  edm::EDGetTokenT<std::vector<pat::Electron>> electrons_token_;
  edm::EDGetTokenT<std::vector<pat::Muon>> muons_token_;
  edm::EDGetTokenT<double> rho_token_;
  edm::EDGetTokenT<reco::TauDiscriminatorContainer> basicTauDiscriminators_inputToken_;
  edm::EDGetTokenT<reco::TauDiscriminatorContainer> basicTauDiscriminatorsdR03_inputToken_;
  edm::EDGetTokenT<edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>>
      pfTauTransverseImpactParameters_token_;
  std::string input_layer_, output_layer_;
  const unsigned year_;
  const unsigned version_;
  const unsigned sub_version_;
  const int debug_level;
  const bool disable_dxy_pca_;
  const bool disable_hcalFraction_workaround_;
  const bool disable_CellIndex_workaround_;
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

#endif
