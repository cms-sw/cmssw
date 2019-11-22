/*
 * \class DeepTauId
 *
 * Tau identification using Deep NN.
 *
 * \author Konstantin Androsov, INFN Pisa
 */

#include "RecoTauTag/RecoTau/interface/DeepTauBase.h"
#include "FWCore/Utilities/interface/isFinite.h"

namespace deep_tau {
  constexpr int NumberOfOutputs = 4;
}

namespace {

  struct dnn_inputs_2017v1 {
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

  namespace dnn_inputs_2017_v2 {
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
    }

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
  }  // namespace dnn_inputs_2017_v2

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

    void addMatchedMuon(const pat::Muon& muon, const pat::Tau& tau) {
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

    static std::vector<const pat::Muon*> findMatchedMuons(const pat::Tau& tau,
                                                          const pat::MuonCollection& muons,
                                                          double deltaR,
                                                          double minPt) {
      const reco::Muon* hadr_cand_muon = nullptr;
      if (tau.leadPFChargedHadrCand().isNonnull() && tau.leadPFChargedHadrCand()->muonRef().isNonnull())
        hadr_cand_muon = tau.leadPFChargedHadrCand()->muonRef().get();
      std::vector<const pat::Muon*> matched_muons;
      const double dR2 = deltaR * deltaR;
      for (const pat::Muon& muon : muons) {
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

    template <typename dnn, typename TensorElemGet>
    void fillTensor(const TensorElemGet& get, const pat::Tau& tau, float default_value) const {
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
  CellObjectType GetCellObjectType(const pat::PackedCandidate& cand) {
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

    CellGrid(unsigned n_cells_eta, unsigned n_cells_phi, double cell_size_eta, double cell_size_phi)
        : nCellsEta(n_cells_eta),
          nCellsPhi(n_cells_phi),
          nTotal(nCellsEta * nCellsPhi),
          cellSizeEta(cell_size_eta),
          cellSizePhi(cell_size_phi) {
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
      static auto getCellIndex = [](double x, double maxX, double size, int& index) {
        const double absX = std::abs(x);
        if (absX > maxX)
          return false;
        const double absIndex = std::floor(std::abs(absX / size - 0.5));
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
  };

}  // anonymous namespace

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
    desc.add<int>("debug_level", 0);
    desc.add<bool>("disable_dxy_pca", false);

    edm::ParameterSetDescription descWP;
    descWP.add<std::string>("VVVLoose", "0");
    descWP.add<std::string>("VVLoose", "0");
    descWP.add<std::string>("VLoose", "0");
    descWP.add<std::string>("Loose", "0");
    descWP.add<std::string>("Medium", "0");
    descWP.add<std::string>("Tight", "0");
    descWP.add<std::string>("VTight", "0");
    descWP.add<std::string>("VVTight", "0");
    descWP.add<std::string>("VVVTight", "0");
    desc.add<edm::ParameterSetDescription>("VSeWP", descWP);
    desc.add<edm::ParameterSetDescription>("VSmuWP", descWP);
    desc.add<edm::ParameterSetDescription>("VSjetWP", descWP);
    descriptions.add("DeepTau", desc);
  }

public:
  explicit DeepTauId(const edm::ParameterSet& cfg, const deep_tau::DeepTauCache* cache)
      : DeepTauBase(cfg, GetOutputs(), cache),
        electrons_token_(consumes<ElectronCollection>(cfg.getParameter<edm::InputTag>("electrons"))),
        muons_token_(consumes<MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))),
        rho_token_(consumes<double>(cfg.getParameter<edm::InputTag>("rho"))),
        version(cfg.getParameter<unsigned>("version")),
        debug_level(cfg.getParameter<int>("debug_level")),
        disable_dxy_pca_(cfg.getParameter<bool>("disable_dxy_pca")) {
    if (version == 1) {
      input_layer_ = cache_->getGraph().node(0).name();
      output_layer_ = cache_->getGraph().node(cache_->getGraph().node_size() - 1).name();
      const auto& shape = cache_->getGraph().node(0).attr().at("shape").shape();
      if (shape.dim(1).size() != dnn_inputs_2017v1::NumberOfInputs)
        throw cms::Exception("DeepTauId")
            << "number of inputs does not match the expected inputs for the given version";
    } else if (version == 2) {
      tauBlockTensor_ = std::make_unique<tensorflow::Tensor>(
          tensorflow::DT_FLOAT, tensorflow::TensorShape{1, dnn_inputs_2017_v2::TauBlockInputs::NumberOfInputs});
      for (size_t n = 0; n < 2; ++n) {
        const bool is_inner = n == 0;
        const auto n_cells =
            is_inner ? dnn_inputs_2017_v2::number_of_inner_cell : dnn_inputs_2017_v2::number_of_outer_cell;
        eGammaTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT,
            tensorflow::TensorShape{1, 1, 1, dnn_inputs_2017_v2::EgammaBlockInputs::NumberOfInputs});
        muonTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT,
            tensorflow::TensorShape{1, 1, 1, dnn_inputs_2017_v2::MuonBlockInputs::NumberOfInputs});
        hadronsTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT,
            tensorflow::TensorShape{1, 1, 1, dnn_inputs_2017_v2::HadronBlockInputs::NumberOfInputs});
        convTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT,
            tensorflow::TensorShape{1, n_cells, n_cells, dnn_inputs_2017_v2::number_of_conv_features});
        zeroOutputTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT, tensorflow::TensorShape{1, 1, 1, dnn_inputs_2017_v2::number_of_conv_features});

        eGammaTensor_[is_inner]->flat<float>().setZero();
        muonTensor_[is_inner]->flat<float>().setZero();
        hadronsTensor_[is_inner]->flat<float>().setZero();

        setCellConvFeatures(*zeroOutputTensor_[is_inner], getPartialPredictions(is_inner), 0, 0, 0);
      }
    } else {
      throw cms::Exception("DeepTauId") << "version " << version << " is not supported.";
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

  inline void checkInputs(
      const tensorflow::Tensor& inputs, const char* block_name, int n_inputs, int n_eta = 1, int n_phi = 1) const {
    if (debug_level >= 1) {
      for (int eta = 0; eta < n_eta; ++eta) {
        for (int phi = 0; phi < n_phi; phi++) {
          for (int k = 0; k < n_inputs; ++k) {
            const float input =
                n_eta == 1 && n_phi == 1 ? inputs.matrix<float>()(0, k) : inputs.tensor<float, 4>()(0, eta, phi, k);
            if (edm::isNotFinite(input))
              throw cms::Exception("DeepTauId")
                  << "in the " << block_name << ", input is not finite, i.e. infinite or NaN, for eta_index = " << n_eta
                  << ", phi_index = " << n_phi << ", input_index = " << k;
            if (debug_level >= 2)
              std::cout << block_name << "," << eta << "," << phi << "," << k << "," << std::setprecision(5)
                        << std::fixed << input << '\n';
          }
        }
      }
    }
  }

private:
  tensorflow::Tensor getPredictions(edm::Event& event,
                                    const edm::EventSetup& es,
                                    edm::Handle<TauCollection> taus) override {
    edm::Handle<pat::ElectronCollection> electrons;
    event.getByToken(electrons_token_, electrons);

    edm::Handle<pat::MuonCollection> muons;
    event.getByToken(muons_token_, muons);

    edm::Handle<pat::PackedCandidateCollection> pfCands;
    event.getByToken(pfcandToken_, pfCands);

    edm::Handle<reco::VertexCollection> vertices;
    event.getByToken(vtxToken_, vertices);

    edm::Handle<double> rho;
    event.getByToken(rho_token_, rho);

    tensorflow::Tensor predictions(tensorflow::DT_FLOAT, {static_cast<int>(taus->size()), deep_tau::NumberOfOutputs});
    for (size_t tau_index = 0; tau_index < taus->size(); ++tau_index) {
      std::vector<tensorflow::Tensor> pred_vector;
      if (version == 1)
        getPredictionsV1(taus->at(tau_index), *electrons, *muons, pred_vector);
      else if (version == 2)
        getPredictionsV2(taus->at(tau_index), *electrons, *muons, *pfCands, vertices->at(0), *rho, pred_vector);
      else
        throw cms::Exception("DeepTauId") << "version " << version << " is not supported.";
      for (int k = 0; k < deep_tau::NumberOfOutputs; ++k) {
        const float pred = pred_vector[0].flat<float>()(k);
        if (!(pred >= 0 && pred <= 1))
          throw cms::Exception("DeepTauId")
              << "invalid prediction = " << pred << " for tau_index = " << tau_index << ", pred_index = " << k;
        predictions.matrix<float>()(tau_index, k) = pred;
      }
    }
    return predictions;
  }

  void getPredictionsV1(const TauType& tau,
                        const pat::ElectronCollection& electrons,
                        const pat::MuonCollection& muons,
                        std::vector<tensorflow::Tensor>& pred_vector) {
    const tensorflow::Tensor& inputs = createInputsV1<dnn_inputs_2017v1>(tau, electrons, muons);
    tensorflow::run(&(cache_->getSession()), {{input_layer_, inputs}}, {output_layer_}, &pred_vector);
  }

  void getPredictionsV2(const TauType& tau,
                        const pat::ElectronCollection& electrons,
                        const pat::MuonCollection& muons,
                        const pat::PackedCandidateCollection& pfCands,
                        const reco::Vertex& pv,
                        double rho,
                        std::vector<tensorflow::Tensor>& pred_vector) {
    CellGrid inner_grid(dnn_inputs_2017_v2::number_of_inner_cell, dnn_inputs_2017_v2::number_of_inner_cell, 0.02, 0.02);
    CellGrid outer_grid(dnn_inputs_2017_v2::number_of_outer_cell, dnn_inputs_2017_v2::number_of_outer_cell, 0.05, 0.05);
    fillGrids(tau, electrons, inner_grid, outer_grid);
    fillGrids(tau, muons, inner_grid, outer_grid);
    fillGrids(tau, pfCands, inner_grid, outer_grid);

    createTauBlockInputs(tau, pv, rho);
    createConvFeatures(tau, pv, rho, electrons, muons, pfCands, inner_grid, true);
    createConvFeatures(tau, pv, rho, electrons, muons, pfCands, outer_grid, false);

    tensorflow::run(&(cache_->getSession("core")),
                    {{"input_tau", *tauBlockTensor_},
                     {"input_inner", *convTensor_.at(true)},
                     {"input_outer", *convTensor_.at(false)}},
                    {"main_output/Softmax"},
                    &pred_vector);
  }

  template <typename Collection>
  void fillGrids(const TauType& tau, const Collection& objects, CellGrid& inner_grid, CellGrid& outer_grid) {
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

  void createConvFeatures(const TauType& tau,
                          const reco::Vertex& pv,
                          double rho,
                          const pat::ElectronCollection& electrons,
                          const pat::MuonCollection& muons,
                          const pat::PackedCandidateCollection& pfCands,
                          const CellGrid& grid,
                          bool is_inner) {
    tensorflow::Tensor& convTensor = *convTensor_.at(is_inner);

    eGammaTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape{
            (long long int)grid.num_valid_cells(), 1, 1, dnn_inputs_2017_v2::EgammaBlockInputs::NumberOfInputs});
    muonTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape{
            (long long int)grid.num_valid_cells(), 1, 1, dnn_inputs_2017_v2::MuonBlockInputs::NumberOfInputs});
    hadronsTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape{
            (long long int)grid.num_valid_cells(), 1, 1, dnn_inputs_2017_v2::HadronBlockInputs::NumberOfInputs});

    eGammaTensor_[is_inner]->flat<float>().setZero();
    muonTensor_[is_inner]->flat<float>().setZero();
    hadronsTensor_[is_inner]->flat<float>().setZero();

    unsigned idx = 0;
    for (int eta = -grid.maxEtaIndex(); eta <= grid.maxEtaIndex(); ++eta) {
      for (int phi = -grid.maxPhiIndex(); phi <= grid.maxPhiIndex(); ++phi) {
        const CellIndex cell_index{eta, phi};
        const auto cell_iter = grid.find(cell_index);
        if (cell_iter != grid.end()) {
          const Cell& cell = cell_iter->second;
          createEgammaBlockInputs(idx, tau, pv, rho, electrons, pfCands, cell, is_inner);
          createMuonBlockInputs(idx, tau, pv, rho, muons, pfCands, cell, is_inner);
          createHadronsBlockInputs(idx, tau, pv, rho, pfCands, cell, is_inner);
          idx += 1;
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
    for (int n = 0; n < dnn_inputs_2017_v2::number_of_conv_features; ++n)
      convTensor.tensor<float, 4>()(0, eta_index, phi_index, n) = features.tensor<float, 4>()(batch_idx, 0, 0, n);
  }

  void createTauBlockInputs(const TauType& tau, const reco::Vertex& pv, double rho) {
    namespace dnn = dnn_inputs_2017_v2::TauBlockInputs;

    tensorflow::Tensor& inputs = *tauBlockTensor_;
    inputs.flat<float>().setZero();

    const auto& get = [&](int var_index) -> float& { return inputs.matrix<float>()(0, var_index); };

    auto leadChargedHadrCand = dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get());

    get(dnn::rho) = getValueNorm(rho, 21.49f, 9.713f);
    get(dnn::tau_pt) = getValueLinear(tau.polarP4().pt(), 20.f, 1000.f, true);
    get(dnn::tau_eta) = getValueLinear(tau.polarP4().eta(), -2.3f, 2.3f, false);
    get(dnn::tau_phi) = getValueLinear(tau.polarP4().phi(), -pi, pi, false);
    get(dnn::tau_mass) = getValueNorm(tau.polarP4().mass(), 0.6669f, 0.6553f);
    get(dnn::tau_E_over_pt) = getValueLinear(tau.p4().energy() / tau.p4().pt(), 1.f, 5.2f, true);
    get(dnn::tau_charge) = getValue(tau.charge());
    get(dnn::tau_n_charged_prongs) = getValueLinear(tau.decayMode() / 5 + 1, 1, 3, true);
    get(dnn::tau_n_neutral_prongs) = getValueLinear(tau.decayMode() % 5, 0, 2, true);
    get(dnn::chargedIsoPtSum) = getValueNorm(tau.tauID("chargedIsoPtSum"), 47.78f, 123.5f);
    get(dnn::chargedIsoPtSumdR03_over_dR05) = getValue(tau.tauID("chargedIsoPtSumdR03") / tau.tauID("chargedIsoPtSum"));
    get(dnn::footprintCorrection) = getValueNorm(tau.tauID("footprintCorrectiondR03"), 9.029f, 26.42f);
    get(dnn::neutralIsoPtSum) = getValueNorm(tau.tauID("neutralIsoPtSum"), 57.59f, 155.3f);
    get(dnn::neutralIsoPtSumWeight_over_neutralIsoPtSum) =
        getValue(tau.tauID("neutralIsoPtSumWeight") / tau.tauID("neutralIsoPtSum"));
    get(dnn::neutralIsoPtSumWeightdR03_over_neutralIsoPtSum) =
        getValue(tau.tauID("neutralIsoPtSumWeightdR03") / tau.tauID("neutralIsoPtSum"));
    get(dnn::neutralIsoPtSumdR03_over_dR05) = getValue(tau.tauID("neutralIsoPtSumdR03") / tau.tauID("neutralIsoPtSum"));
    get(dnn::photonPtSumOutsideSignalCone) =
        getValueNorm(tau.tauID("photonPtSumOutsideSignalConedR03"), 1.731f, 6.846f);
    get(dnn::puCorrPtSum) = getValueNorm(tau.tauID("puCorrPtSum"), 22.38f, 16.34f);
    // The global PCA coordinates were used as inputs during the NN training, but it was decided to disable
    // them for the inference, because modeling of dxy_PCA in MC poorly describes the data, and x and y coordinates
    // in data results outside of the expected 5 std. dev. input validity range. On the other hand,
    // these coordinates are strongly era-dependent. Kept as comment to document what NN expects.
    if (!disable_dxy_pca_) {
      get(dnn::tau_dxy_pca_x) = getValueNorm(tau.dxy_PCA().x(), -0.0241f, 0.0074f);
      get(dnn::tau_dxy_pca_y) = getValueNorm(tau.dxy_PCA().y(), 0.0675f, 0.0128f);
      get(dnn::tau_dxy_pca_z) = getValueNorm(tau.dxy_PCA().z(), 0.7973f, 3.456f);
    } else {
      get(dnn::tau_dxy_pca_x) = 0;
      get(dnn::tau_dxy_pca_y) = 0;
      get(dnn::tau_dxy_pca_z) = 0;
    }

    const bool tau_dxy_valid =
        std::isnormal(tau.dxy()) && tau.dxy() > -10 && std::isnormal(tau.dxy_error()) && tau.dxy_error() > 0;
    if (tau_dxy_valid) {
      get(dnn::tau_dxy_valid) = tau_dxy_valid;
      get(dnn::tau_dxy) = getValueNorm(tau.dxy(), 0.0018f, 0.0085f);
      get(dnn::tau_dxy_sig) = getValueNorm(std::abs(tau.dxy()) / tau.dxy_error(), 2.26f, 4.191f);
    }
    const bool tau_ip3d_valid =
        std::isnormal(tau.ip3d()) && tau.ip3d() > -10 && std::isnormal(tau.ip3d_error()) && tau.ip3d_error() > 0;
    if (tau_ip3d_valid) {
      get(dnn::tau_ip3d_valid) = tau_ip3d_valid;
      get(dnn::tau_ip3d) = getValueNorm(tau.ip3d(), 0.0026f, 0.0114f);
      get(dnn::tau_ip3d_sig) = getValueNorm(std::abs(tau.ip3d()) / tau.ip3d_error(), 2.928f, 4.466f);
    }
    if (leadChargedHadrCand) {
      get(dnn::tau_dz) = getValueNorm(leadChargedHadrCand->dz(), 0.f, 0.0190f);
      const bool tau_dz_sig_valid = leadChargedHadrCand->hasTrackDetails() &&
                                    std::isnormal(leadChargedHadrCand->dz()) &&
                                    std::isnormal(leadChargedHadrCand->dzError()) && leadChargedHadrCand->dzError() > 0;
      get(dnn::tau_dz_sig_valid) = tau_dz_sig_valid;
      const double dzError = leadChargedHadrCand->hasTrackDetails() ? leadChargedHadrCand->dzError() : default_value;
      get(dnn::tau_dz_sig) = getValueNorm(std::abs(leadChargedHadrCand->dz()) / dzError, 4.717f, 11.78f);
    }
    get(dnn::tau_flightLength_x) = getValueNorm(tau.flightLength().x(), -0.0003f, 0.7362f);
    get(dnn::tau_flightLength_y) = getValueNorm(tau.flightLength().y(), -0.0009f, 0.7354f);
    get(dnn::tau_flightLength_z) = getValueNorm(tau.flightLength().z(), -0.0022f, 1.993f);
    // get(dnn::tau_flightLength_sig) = getValueNorm(tau.flightLengthSig(), -4.78f, 9.573f);
    get(dnn::tau_flightLength_sig) = 0.55756444;  //This value is set due to a bug in the training
    get(dnn::tau_pt_weighted_deta_strip) =
        getValueLinear(reco::tau::pt_weighted_deta_strip(tau, tau.decayMode()), 0, 1, true);

    get(dnn::tau_pt_weighted_dphi_strip) =
        getValueLinear(reco::tau::pt_weighted_dphi_strip(tau, tau.decayMode()), 0, 1, true);
    get(dnn::tau_pt_weighted_dr_signal) =
        getValueNorm(reco::tau::pt_weighted_dr_signal(tau, tau.decayMode()), 0.0052f, 0.01433f);
    get(dnn::tau_pt_weighted_dr_iso) = getValueLinear(reco::tau::pt_weighted_dr_iso(tau, tau.decayMode()), 0, 1, true);
    get(dnn::tau_leadingTrackNormChi2) = getValueNorm(tau.leadingTrackNormChi2(), 1.538f, 4.401f);
    const auto eratio = reco::tau::eratio(tau);
    const bool tau_e_ratio_valid = std::isnormal(eratio) && eratio > 0.f;
    get(dnn::tau_e_ratio_valid) = tau_e_ratio_valid;
    get(dnn::tau_e_ratio) = tau_e_ratio_valid ? getValueLinear(eratio, 0, 1, true) : 0.f;
    const double gj_angle_diff = calculateGottfriedJacksonAngleDifference(tau);
    const bool tau_gj_angle_diff_valid = (std::isnormal(gj_angle_diff) || gj_angle_diff == 0) && gj_angle_diff >= 0;
    get(dnn::tau_gj_angle_diff_valid) = tau_gj_angle_diff_valid;
    get(dnn::tau_gj_angle_diff) = tau_gj_angle_diff_valid ? getValueLinear(gj_angle_diff, 0, pi, true) : 0;
    get(dnn::tau_n_photons) = getValueNorm(reco::tau::n_photons_total(tau), 2.95f, 3.927f);
    get(dnn::tau_emFraction) = getValueLinear(tau.emFraction_MVA(), -1, 1, false);
    get(dnn::tau_inside_ecal_crack) = getValue(isInEcalCrack(tau.p4().eta()));
    get(dnn::leadChargedCand_etaAtEcalEntrance_minus_tau_eta) =
        getValueNorm(tau.etaAtEcalEntranceLeadChargedCand() - tau.p4().eta(), 0.0042f, 0.0323f);

    checkInputs(inputs, "tau_block", dnn::NumberOfInputs);
  }

  void createEgammaBlockInputs(unsigned idx,
                               const TauType& tau,
                               const reco::Vertex& pv,
                               double rho,
                               const pat::ElectronCollection& electrons,
                               const pat::PackedCandidateCollection& pfCands,
                               const Cell& cell_map,
                               bool is_inner) {
    namespace dnn = dnn_inputs_2017_v2::EgammaBlockInputs;

    tensorflow::Tensor& inputs = *eGammaTensor_.at(is_inner);

    const auto& get = [&](int var_index) -> float& { return inputs.tensor<float, 4>()(idx, 0, 0, var_index); };

    const bool valid_index_pf_ele = cell_map.count(CellObjectType::PfCand_electron);
    const bool valid_index_pf_gamma = cell_map.count(CellObjectType::PfCand_gamma);
    const bool valid_index_ele = cell_map.count(CellObjectType::Electron);

    if (!cell_map.empty()) {
      get(dnn::rho) = getValueNorm(rho, 21.49f, 9.713f);
      get(dnn::tau_pt) = getValueLinear(tau.polarP4().pt(), 20.f, 1000.f, true);
      get(dnn::tau_eta) = getValueLinear(tau.polarP4().eta(), -2.3f, 2.3f, false);
      get(dnn::tau_inside_ecal_crack) = getValue(isInEcalCrack(tau.polarP4().eta()));
    }
    if (valid_index_pf_ele) {
      size_t index_pf_ele = cell_map.at(CellObjectType::PfCand_electron);

      get(dnn::pfCand_ele_valid) = valid_index_pf_ele;
      get(dnn::pfCand_ele_rel_pt) = getValueNorm(pfCands.at(index_pf_ele).polarP4().pt() / tau.polarP4().pt(),
                                                 is_inner ? 0.9792f : 0.304f,
                                                 is_inner ? 0.5383f : 1.845f);
      get(dnn::pfCand_ele_deta) = getValueLinear(pfCands.at(index_pf_ele).polarP4().eta() - tau.polarP4().eta(),
                                                 is_inner ? -0.1f : -0.5f,
                                                 is_inner ? 0.1f : 0.5f,
                                                 false);
      get(dnn::pfCand_ele_dphi) = getValueLinear(dPhi(tau.polarP4(), pfCands.at(index_pf_ele).polarP4()),
                                                 is_inner ? -0.1f : -0.5f,
                                                 is_inner ? 0.1f : 0.5f,
                                                 false);
      get(dnn::pfCand_ele_pvAssociationQuality) =
          getValueLinear<int>(pfCands.at(index_pf_ele).pvAssociationQuality(), 0, 7, true);
      get(dnn::pfCand_ele_puppiWeight) = getValue(pfCands.at(index_pf_ele).puppiWeight());
      get(dnn::pfCand_ele_charge) = getValue(pfCands.at(index_pf_ele).charge());
      get(dnn::pfCand_ele_lostInnerHits) = getValue<int>(pfCands.at(index_pf_ele).lostInnerHits());
      get(dnn::pfCand_ele_numberOfPixelHits) =
          getValueLinear(pfCands.at(index_pf_ele).numberOfPixelHits(), 0, 10, true);
      get(dnn::pfCand_ele_vertex_dx) =
          getValueNorm(pfCands.at(index_pf_ele).vertex().x() - pv.position().x(), 0.f, 0.1221f);
      get(dnn::pfCand_ele_vertex_dy) =
          getValueNorm(pfCands.at(index_pf_ele).vertex().y() - pv.position().y(), 0.f, 0.1226f);
      get(dnn::pfCand_ele_vertex_dz) =
          getValueNorm(pfCands.at(index_pf_ele).vertex().z() - pv.position().z(), 0.001f, 1.024f);
      get(dnn::pfCand_ele_vertex_dx_tauFL) = getValueNorm(
          pfCands.at(index_pf_ele).vertex().x() - pv.position().x() - tau.flightLength().x(), 0.f, 0.3411f);
      get(dnn::pfCand_ele_vertex_dy_tauFL) = getValueNorm(
          pfCands.at(index_pf_ele).vertex().y() - pv.position().y() - tau.flightLength().y(), 0.0003f, 0.3385f);
      get(dnn::pfCand_ele_vertex_dz_tauFL) =
          getValueNorm(pfCands.at(index_pf_ele).vertex().z() - pv.position().z() - tau.flightLength().z(), 0.f, 1.307f);

      const bool hasTrackDetails = pfCands.at(index_pf_ele).hasTrackDetails();
      if (hasTrackDetails) {
        get(dnn::pfCand_ele_hasTrackDetails) = hasTrackDetails;
        get(dnn::pfCand_ele_dxy) = getValueNorm(pfCands.at(index_pf_ele).dxy(), 0.f, 0.171f);
        get(dnn::pfCand_ele_dxy_sig) =
            getValueNorm(std::abs(pfCands.at(index_pf_ele).dxy()) / pfCands.at(index_pf_ele).dxyError(), 1.634f, 6.45f);
        get(dnn::pfCand_ele_dz) = getValueNorm(pfCands.at(index_pf_ele).dz(), 0.001f, 1.02f);
        get(dnn::pfCand_ele_dz_sig) =
            getValueNorm(std::abs(pfCands.at(index_pf_ele).dz()) / pfCands.at(index_pf_ele).dzError(), 24.56f, 210.4f);
        get(dnn::pfCand_ele_track_chi2_ndof) =
            getValueNorm(pfCands.at(index_pf_ele).pseudoTrack().chi2() / pfCands.at(index_pf_ele).pseudoTrack().ndof(),
                         2.272f,
                         8.439f);
        get(dnn::pfCand_ele_track_ndof) = getValueNorm(pfCands.at(index_pf_ele).pseudoTrack().ndof(), 15.18f, 3.203f);
      }
    }
    if (valid_index_pf_gamma) {
      size_t index_pf_gamma = cell_map.at(CellObjectType::PfCand_gamma);
      get(dnn::pfCand_gamma_valid) = valid_index_pf_gamma;
      get(dnn::pfCand_gamma_rel_pt) = getValueNorm(pfCands.at(index_pf_gamma).polarP4().pt() / tau.polarP4().pt(),
                                                   is_inner ? 0.6048f : 0.02576f,
                                                   is_inner ? 1.669f : 0.3833f);
      get(dnn::pfCand_gamma_deta) = getValueLinear(pfCands.at(index_pf_gamma).polarP4().eta() - tau.polarP4().eta(),
                                                   is_inner ? -0.1f : -0.5f,
                                                   is_inner ? 0.1f : 0.5f,
                                                   false);
      get(dnn::pfCand_gamma_dphi) = getValueLinear(dPhi(tau.polarP4(), pfCands.at(index_pf_gamma).polarP4()),
                                                   is_inner ? -0.1f : -0.5f,
                                                   is_inner ? 0.1f : 0.5f,
                                                   false);
      get(dnn::pfCand_gamma_pvAssociationQuality) =
          getValueLinear<int>(pfCands.at(index_pf_gamma).pvAssociationQuality(), 0, 7, true);
      get(dnn::pfCand_gamma_fromPV) = getValueLinear<int>(pfCands.at(index_pf_gamma).fromPV(), 0, 3, true);
      get(dnn::pfCand_gamma_puppiWeight) = getValue(pfCands.at(index_pf_gamma).puppiWeight());
      get(dnn::pfCand_gamma_puppiWeightNoLep) = getValue(pfCands.at(index_pf_gamma).puppiWeightNoLep());
      get(dnn::pfCand_gamma_lostInnerHits) = getValue<int>(pfCands.at(index_pf_gamma).lostInnerHits());
      get(dnn::pfCand_gamma_numberOfPixelHits) =
          getValueLinear(pfCands.at(index_pf_gamma).numberOfPixelHits(), 0, 7, true);
      get(dnn::pfCand_gamma_vertex_dx) =
          getValueNorm(pfCands.at(index_pf_gamma).vertex().x() - pv.position().x(), 0.f, 0.0067f);
      get(dnn::pfCand_gamma_vertex_dy) =
          getValueNorm(pfCands.at(index_pf_gamma).vertex().y() - pv.position().y(), 0.f, 0.0069f);
      get(dnn::pfCand_gamma_vertex_dz) =
          getValueNorm(pfCands.at(index_pf_gamma).vertex().z() - pv.position().z(), 0.f, 0.0578f);
      get(dnn::pfCand_gamma_vertex_dx_tauFL) = getValueNorm(
          pfCands.at(index_pf_gamma).vertex().x() - pv.position().x() - tau.flightLength().x(), 0.001f, 0.9565f);
      get(dnn::pfCand_gamma_vertex_dy_tauFL) = getValueNorm(
          pfCands.at(index_pf_gamma).vertex().y() - pv.position().y() - tau.flightLength().y(), 0.0008f, 0.9592f);
      get(dnn::pfCand_gamma_vertex_dz_tauFL) = getValueNorm(
          pfCands.at(index_pf_gamma).vertex().z() - pv.position().z() - tau.flightLength().z(), 0.0038f, 2.154f);

      const bool hasTrackDetails = pfCands.at(index_pf_gamma).hasTrackDetails();
      if (hasTrackDetails) {
        get(dnn::pfCand_gamma_hasTrackDetails) = hasTrackDetails;
        get(dnn::pfCand_gamma_dxy) = getValueNorm(pfCands.at(index_pf_gamma).dxy(), 0.0004f, 0.882f);
        get(dnn::pfCand_gamma_dxy_sig) = getValueNorm(
            std::abs(pfCands.at(index_pf_gamma).dxy()) / pfCands.at(index_pf_gamma).dxyError(), 4.271f, 63.78f);
        get(dnn::pfCand_gamma_dz) = getValueNorm(pfCands.at(index_pf_gamma).dz(), 0.0071f, 5.285f);
        get(dnn::pfCand_gamma_dz_sig) = getValueNorm(
            std::abs(pfCands.at(index_pf_gamma).dz()) / pfCands.at(index_pf_gamma).dzError(), 162.1f, 622.4f);
        get(dnn::pfCand_gamma_track_chi2_ndof) = pfCands.at(index_pf_gamma).pseudoTrack().ndof() > 0
                                                     ? getValueNorm(pfCands.at(index_pf_gamma).pseudoTrack().chi2() /
                                                                        pfCands.at(index_pf_gamma).pseudoTrack().ndof(),
                                                                    4.268f,
                                                                    15.47f)
                                                     : 0;
        get(dnn::pfCand_gamma_track_ndof) =
            pfCands.at(index_pf_gamma).pseudoTrack().ndof() > 0
                ? getValueNorm(pfCands.at(index_pf_gamma).pseudoTrack().ndof(), 12.25f, 4.774f)
                : 0;
      }
    }
    if (valid_index_ele) {
      size_t index_ele = cell_map.at(CellObjectType::Electron);

      get(dnn::ele_valid) = valid_index_ele;
      get(dnn::ele_rel_pt) = getValueNorm(electrons.at(index_ele).polarP4().pt() / tau.polarP4().pt(),
                                          is_inner ? 1.067f : 0.5111f,
                                          is_inner ? 1.521f : 2.765f);
      get(dnn::ele_deta) = getValueLinear(electrons.at(index_ele).polarP4().eta() - tau.polarP4().eta(),
                                          is_inner ? -0.1f : -0.5f,
                                          is_inner ? 0.1f : 0.5f,
                                          false);
      get(dnn::ele_dphi) = getValueLinear(dPhi(tau.polarP4(), electrons.at(index_ele).polarP4()),
                                          is_inner ? -0.1f : -0.5f,
                                          is_inner ? 0.1f : 0.5f,
                                          false);

      float cc_ele_energy, cc_gamma_energy;
      int cc_n_gamma;
      const bool cc_valid =
          calculateElectronClusterVarsV2(electrons.at(index_ele), cc_ele_energy, cc_gamma_energy, cc_n_gamma);
      if (cc_valid) {
        get(dnn::ele_cc_valid) = cc_valid;
        get(dnn::ele_cc_ele_rel_energy) =
            getValueNorm(cc_ele_energy / electrons.at(index_ele).polarP4().pt(), 1.729f, 1.644f);
        get(dnn::ele_cc_gamma_rel_energy) = getValueNorm(cc_gamma_energy / cc_ele_energy, 0.1439f, 0.3284f);
        get(dnn::ele_cc_n_gamma) = getValueNorm(cc_n_gamma, 1.794f, 2.079f);
      }
      get(dnn::ele_rel_trackMomentumAtVtx) = getValueNorm(
          electrons.at(index_ele).trackMomentumAtVtx().R() / electrons.at(index_ele).polarP4().pt(), 1.531f, 1.424f);
      get(dnn::ele_rel_trackMomentumAtCalo) = getValueNorm(
          electrons.at(index_ele).trackMomentumAtCalo().R() / electrons.at(index_ele).polarP4().pt(), 1.531f, 1.424f);
      get(dnn::ele_rel_trackMomentumOut) = getValueNorm(
          electrons.at(index_ele).trackMomentumOut().R() / electrons.at(index_ele).polarP4().pt(), 0.7735f, 0.935f);
      get(dnn::ele_rel_trackMomentumAtEleClus) =
          getValueNorm(electrons.at(index_ele).trackMomentumAtEleClus().R() / electrons.at(index_ele).polarP4().pt(),
                       0.7735f,
                       0.935f);
      get(dnn::ele_rel_trackMomentumAtVtxWithConstraint) = getValueNorm(
          electrons.at(index_ele).trackMomentumAtVtxWithConstraint().R() / electrons.at(index_ele).polarP4().pt(),
          1.625f,
          1.581f);
      get(dnn::ele_rel_ecalEnergy) =
          getValueNorm(electrons.at(index_ele).ecalEnergy() / electrons.at(index_ele).polarP4().pt(), 1.993f, 1.308f);
      get(dnn::ele_ecalEnergy_sig) = getValueNorm(
          electrons.at(index_ele).ecalEnergy() / electrons.at(index_ele).ecalEnergyError(), 70.25f, 58.16f);
      get(dnn::ele_eSuperClusterOverP) = getValueNorm(electrons.at(index_ele).eSuperClusterOverP(), 2.432f, 15.13f);
      get(dnn::ele_eSeedClusterOverP) = getValueNorm(electrons.at(index_ele).eSeedClusterOverP(), 2.034f, 13.96f);
      get(dnn::ele_eSeedClusterOverPout) = getValueNorm(electrons.at(index_ele).eSeedClusterOverPout(), 6.64f, 36.8f);
      get(dnn::ele_eEleClusterOverPout) = getValueNorm(electrons.at(index_ele).eEleClusterOverPout(), 4.183f, 20.63f);
      get(dnn::ele_deltaEtaSuperClusterTrackAtVtx) =
          getValueNorm(electrons.at(index_ele).deltaEtaSuperClusterTrackAtVtx(), 0.f, 0.0363f);
      get(dnn::ele_deltaEtaSeedClusterTrackAtCalo) =
          getValueNorm(electrons.at(index_ele).deltaEtaSeedClusterTrackAtCalo(), -0.0001f, 0.0512f);
      get(dnn::ele_deltaEtaEleClusterTrackAtCalo) =
          getValueNorm(electrons.at(index_ele).deltaEtaEleClusterTrackAtCalo(), -0.0001f, 0.0541f);
      get(dnn::ele_deltaPhiEleClusterTrackAtCalo) =
          getValueNorm(electrons.at(index_ele).deltaPhiEleClusterTrackAtCalo(), 0.0002f, 0.0553f);
      get(dnn::ele_deltaPhiSuperClusterTrackAtVtx) =
          getValueNorm(electrons.at(index_ele).deltaPhiSuperClusterTrackAtVtx(), 0.0001f, 0.0523f);
      get(dnn::ele_deltaPhiSeedClusterTrackAtCalo) =
          getValueNorm(electrons.at(index_ele).deltaPhiSeedClusterTrackAtCalo(), 0.0004f, 0.0777f);
      get(dnn::ele_mvaInput_earlyBrem) = getValue(electrons.at(index_ele).mvaInput().earlyBrem);
      get(dnn::ele_mvaInput_lateBrem) = getValue(electrons.at(index_ele).mvaInput().lateBrem);
      get(dnn::ele_mvaInput_sigmaEtaEta) =
          getValueNorm(electrons.at(index_ele).mvaInput().sigmaEtaEta, 0.0008f, 0.0052f);
      get(dnn::ele_mvaInput_hadEnergy) = getValueNorm(electrons.at(index_ele).mvaInput().hadEnergy, 14.04f, 69.48f);
      get(dnn::ele_mvaInput_deltaEta) = getValueNorm(electrons.at(index_ele).mvaInput().deltaEta, 0.0099f, 0.0851f);

      const auto& gsfTrack = electrons.at(index_ele).gsfTrack();
      if (gsfTrack.isNonnull()) {
        get(dnn::ele_gsfTrack_normalizedChi2) = getValueNorm(gsfTrack->normalizedChi2(), 3.049f, 10.39f);
        get(dnn::ele_gsfTrack_numberOfValidHits) = getValueNorm(gsfTrack->numberOfValidHits(), 16.52f, 2.806f);
        get(dnn::ele_rel_gsfTrack_pt) =
            getValueNorm(gsfTrack->pt() / electrons.at(index_ele).polarP4().pt(), 1.355f, 16.81f);
        get(dnn::ele_gsfTrack_pt_sig) = getValueNorm(gsfTrack->pt() / gsfTrack->ptError(), 5.046f, 3.119f);
      }
      const auto& closestCtfTrack = electrons.at(index_ele).closestCtfTrackRef();
      const bool has_closestCtfTrack = closestCtfTrack.isNonnull();
      if (has_closestCtfTrack) {
        get(dnn::ele_has_closestCtfTrack) = has_closestCtfTrack;
        get(dnn::ele_closestCtfTrack_normalizedChi2) = getValueNorm(closestCtfTrack->normalizedChi2(), 2.411f, 6.98f);
        get(dnn::ele_closestCtfTrack_numberOfValidHits) =
            getValueNorm(closestCtfTrack->numberOfValidHits(), 15.16f, 5.26f);
      }
    }
    checkInputs(inputs, is_inner ? "egamma_inner_block" : "egamma_outer_block", dnn::NumberOfInputs);
  }

  void createMuonBlockInputs(unsigned idx,
                             const TauType& tau,
                             const reco::Vertex& pv,
                             double rho,
                             const pat::MuonCollection& muons,
                             const pat::PackedCandidateCollection& pfCands,
                             const Cell& cell_map,
                             bool is_inner) {
    namespace dnn = dnn_inputs_2017_v2::MuonBlockInputs;

    tensorflow::Tensor& inputs = *muonTensor_.at(is_inner);

    const auto& get = [&](int var_index) -> float& { return inputs.tensor<float, 4>()(idx, 0, 0, var_index); };

    const bool valid_index_pf_muon = cell_map.count(CellObjectType::PfCand_muon);
    const bool valid_index_muon = cell_map.count(CellObjectType::Muon);

    if (!cell_map.empty()) {
      get(dnn::rho) = getValueNorm(rho, 21.49f, 9.713f);
      get(dnn::tau_pt) = getValueLinear(tau.polarP4().pt(), 20.f, 1000.f, true);
      get(dnn::tau_eta) = getValueLinear(tau.polarP4().eta(), -2.3f, 2.3f, false);
      get(dnn::tau_inside_ecal_crack) = getValue(isInEcalCrack(tau.polarP4().eta()));
    }
    if (valid_index_pf_muon) {
      size_t index_pf_muon = cell_map.at(CellObjectType::PfCand_muon);

      get(dnn::pfCand_muon_valid) = valid_index_pf_muon;
      get(dnn::pfCand_muon_rel_pt) = getValueNorm(pfCands.at(index_pf_muon).polarP4().pt() / tau.polarP4().pt(),
                                                  is_inner ? 0.9509f : 0.0861f,
                                                  is_inner ? 0.4294f : 0.4065f);
      get(dnn::pfCand_muon_deta) = getValueLinear(pfCands.at(index_pf_muon).polarP4().eta() - tau.polarP4().eta(),
                                                  is_inner ? -0.1f : -0.5f,
                                                  is_inner ? 0.1f : 0.5f,
                                                  false);
      get(dnn::pfCand_muon_dphi) = getValueLinear(dPhi(tau.polarP4(), pfCands.at(index_pf_muon).polarP4()),
                                                  is_inner ? -0.1f : -0.5f,
                                                  is_inner ? 0.1f : 0.5f,
                                                  false);
      get(dnn::pfCand_muon_pvAssociationQuality) =
          getValueLinear<int>(pfCands.at(index_pf_muon).pvAssociationQuality(), 0, 7, true);
      get(dnn::pfCand_muon_fromPV) = getValueLinear<int>(pfCands.at(index_pf_muon).fromPV(), 0, 3, true);
      get(dnn::pfCand_muon_puppiWeight) = getValue(pfCands.at(index_pf_muon).puppiWeight());
      get(dnn::pfCand_muon_charge) = getValue(pfCands.at(index_pf_muon).charge());
      get(dnn::pfCand_muon_lostInnerHits) = getValue<int>(pfCands.at(index_pf_muon).lostInnerHits());
      get(dnn::pfCand_muon_numberOfPixelHits) =
          getValueLinear(pfCands.at(index_pf_muon).numberOfPixelHits(), 0, 11, true);
      get(dnn::pfCand_muon_vertex_dx) =
          getValueNorm(pfCands.at(index_pf_muon).vertex().x() - pv.position().x(), -0.0007f, 0.6869f);
      get(dnn::pfCand_muon_vertex_dy) =
          getValueNorm(pfCands.at(index_pf_muon).vertex().y() - pv.position().y(), 0.0001f, 0.6784f);
      get(dnn::pfCand_muon_vertex_dz) =
          getValueNorm(pfCands.at(index_pf_muon).vertex().z() - pv.position().z(), -0.0117f, 4.097f);
      get(dnn::pfCand_muon_vertex_dx_tauFL) = getValueNorm(
          pfCands.at(index_pf_muon).vertex().x() - pv.position().x() - tau.flightLength().x(), -0.0001f, 0.8642f);
      get(dnn::pfCand_muon_vertex_dy_tauFL) = getValueNorm(
          pfCands.at(index_pf_muon).vertex().y() - pv.position().y() - tau.flightLength().y(), 0.0004f, 0.8561f);
      get(dnn::pfCand_muon_vertex_dz_tauFL) = getValueNorm(
          pfCands.at(index_pf_muon).vertex().z() - pv.position().z() - tau.flightLength().z(), -0.0118f, 4.405f);

      const bool hasTrackDetails = pfCands.at(index_pf_muon).hasTrackDetails();
      if (hasTrackDetails) {
        get(dnn::pfCand_muon_hasTrackDetails) = hasTrackDetails;
        get(dnn::pfCand_muon_dxy) = getValueNorm(pfCands.at(index_pf_muon).dxy(), -0.0045f, 0.9655f);
        get(dnn::pfCand_muon_dxy_sig) = getValueNorm(
            std::abs(pfCands.at(index_pf_muon).dxy()) / pfCands.at(index_pf_muon).dxyError(), 4.575f, 42.36f);
        get(dnn::pfCand_muon_dz) = getValueNorm(pfCands.at(index_pf_muon).dz(), -0.0117f, 4.097f);
        get(dnn::pfCand_muon_dz_sig) = getValueNorm(
            std::abs(pfCands.at(index_pf_muon).dz()) / pfCands.at(index_pf_muon).dzError(), 80.37f, 343.3f);
        get(dnn::pfCand_muon_track_chi2_ndof) = getValueNorm(
            pfCands.at(index_pf_muon).pseudoTrack().chi2() / pfCands.at(index_pf_muon).pseudoTrack().ndof(),
            0.69f,
            1.711f);
        get(dnn::pfCand_muon_track_ndof) = getValueNorm(pfCands.at(index_pf_muon).pseudoTrack().ndof(), 17.5f, 5.11f);
      }
    }
    if (valid_index_muon) {
      size_t index_muon = cell_map.at(CellObjectType::Muon);

      get(dnn::muon_valid) = valid_index_muon;
      get(dnn::muon_rel_pt) = getValueNorm(muons.at(index_muon).polarP4().pt() / tau.polarP4().pt(),
                                           is_inner ? 0.7966f : 0.2678f,
                                           is_inner ? 3.402f : 3.592f);
      get(dnn::muon_deta) = getValueLinear(muons.at(index_muon).polarP4().eta() - tau.polarP4().eta(),
                                           is_inner ? -0.1f : -0.5f,
                                           is_inner ? 0.1f : 0.5f,
                                           false);
      get(dnn::muon_dphi) = getValueLinear(
          dPhi(tau.polarP4(), muons.at(index_muon).polarP4()), is_inner ? -0.1f : -0.5f, is_inner ? 0.1f : 0.5f, false);
      get(dnn::muon_dxy) = getValueNorm(muons.at(index_muon).dB(pat::Muon::PV2D), 0.0019f, 1.039f);
      get(dnn::muon_dxy_sig) =
          getValueNorm(std::abs(muons.at(index_muon).dB(pat::Muon::PV2D)) / muons.at(index_muon).edB(pat::Muon::PV2D),
                       8.98f,
                       71.17f);

      const bool normalizedChi2_valid =
          muons.at(index_muon).globalTrack().isNonnull() && muons.at(index_muon).normChi2() >= 0;
      if (normalizedChi2_valid) {
        get(dnn::muon_normalizedChi2_valid) = normalizedChi2_valid;
        get(dnn::muon_normalizedChi2) = getValueNorm(muons.at(index_muon).normChi2(), 21.52f, 265.8f);
        if (muons.at(index_muon).innerTrack().isNonnull())
          get(dnn::muon_numberOfValidHits) = getValueNorm(muons.at(index_muon).numberOfValidHits(), 21.84f, 10.59f);
      }
      get(dnn::muon_segmentCompatibility) = getValue(muons.at(index_muon).segmentCompatibility());
      get(dnn::muon_caloCompatibility) = getValue(muons.at(index_muon).caloCompatibility());

      const bool pfEcalEnergy_valid = muons.at(index_muon).pfEcalEnergy() >= 0;
      if (pfEcalEnergy_valid) {
        get(dnn::muon_pfEcalEnergy_valid) = pfEcalEnergy_valid;
        get(dnn::muon_rel_pfEcalEnergy) =
            getValueNorm(muons.at(index_muon).pfEcalEnergy() / muons.at(index_muon).polarP4().pt(), 0.2273f, 0.4865f);
      }

      MuonHitMatchV2 hit_match(muons.at(index_muon));
      static const std::map<int, std::pair<int, int>> muonMatchHitVars = {
          {MuonSubdetId::DT, {dnn::muon_n_matches_DT_1, dnn::muon_n_hits_DT_1}},
          {MuonSubdetId::CSC, {dnn::muon_n_matches_CSC_1, dnn::muon_n_hits_CSC_1}},
          {MuonSubdetId::RPC, {dnn::muon_n_matches_RPC_1, dnn::muon_n_hits_RPC_1}}};

      static const std::map<int, std::vector<float>> muonMatchVarLimits = {
          {MuonSubdetId::DT, {2, 2, 2, 2}}, {MuonSubdetId::CSC, {6, 2, 2, 2}}, {MuonSubdetId::RPC, {7, 6, 4, 4}}};

      static const std::map<int, std::vector<float>> muonHitVarLimits = {{MuonSubdetId::DT, {12, 12, 12, 8}},
                                                                         {MuonSubdetId::CSC, {24, 12, 12, 12}},
                                                                         {MuonSubdetId::RPC, {4, 4, 2, 2}}};

      for (int subdet : hit_match.MuonHitMatchV2::consideredSubdets()) {
        const auto& matchHitVar = muonMatchHitVars.at(subdet);
        const auto& matchLimits = muonMatchVarLimits.at(subdet);
        const auto& hitLimits = muonHitVarLimits.at(subdet);
        for (int station = MuonHitMatchV2::first_station_id; station <= MuonHitMatchV2::last_station_id; ++station) {
          const unsigned n_matches = hit_match.nMatches(subdet, station);
          const unsigned n_hits = hit_match.nHits(subdet, station);
          get(matchHitVar.first + station - 1) = getValueLinear(n_matches, 0, matchLimits.at(station - 1), true);
          get(matchHitVar.second + station - 1) = getValueLinear(n_hits, 0, hitLimits.at(station - 1), true);
        }
      }
    }
    checkInputs(inputs, is_inner ? "muon_inner_block" : "muon_outer_block", dnn::NumberOfInputs);
  }

  void createHadronsBlockInputs(unsigned idx,
                                const TauType& tau,
                                const reco::Vertex& pv,
                                double rho,
                                const pat::PackedCandidateCollection& pfCands,
                                const Cell& cell_map,
                                bool is_inner) {
    namespace dnn = dnn_inputs_2017_v2::HadronBlockInputs;

    tensorflow::Tensor& inputs = *hadronsTensor_.at(is_inner);

    const auto& get = [&](int var_index) -> float& { return inputs.tensor<float, 4>()(idx, 0, 0, var_index); };

    const bool valid_chH = cell_map.count(CellObjectType::PfCand_chargedHadron);
    const bool valid_nH = cell_map.count(CellObjectType::PfCand_neutralHadron);

    if (!cell_map.empty()) {
      get(dnn::rho) = getValueNorm(rho, 21.49f, 9.713f);
      get(dnn::tau_pt) = getValueLinear(tau.polarP4().pt(), 20.f, 1000.f, true);
      get(dnn::tau_eta) = getValueLinear(tau.polarP4().eta(), -2.3f, 2.3f, false);
      get(dnn::tau_inside_ecal_crack) = getValue(isInEcalCrack(tau.polarP4().eta()));
    }
    if (valid_chH) {
      size_t index_chH = cell_map.at(CellObjectType::PfCand_chargedHadron);

      get(dnn::pfCand_chHad_valid) = valid_chH;
      get(dnn::pfCand_chHad_rel_pt) = getValueNorm(pfCands.at(index_chH).polarP4().pt() / tau.polarP4().pt(),
                                                   is_inner ? 0.2564f : 0.0194f,
                                                   is_inner ? 0.8607f : 0.1865f);
      get(dnn::pfCand_chHad_deta) = getValueLinear(pfCands.at(index_chH).polarP4().eta() - tau.polarP4().eta(),
                                                   is_inner ? -0.1f : -0.5f,
                                                   is_inner ? 0.1f : 0.5f,
                                                   false);
      get(dnn::pfCand_chHad_dphi) = getValueLinear(dPhi(tau.polarP4(), pfCands.at(index_chH).polarP4()),
                                                   is_inner ? -0.1f : -0.5f,
                                                   is_inner ? 0.1f : 0.5f,
                                                   false);
      get(dnn::pfCand_chHad_leadChargedHadrCand) = getValue(
          &pfCands.at(index_chH) == dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get()));
      get(dnn::pfCand_chHad_pvAssociationQuality) =
          getValueLinear<int>(pfCands.at(index_chH).pvAssociationQuality(), 0, 7, true);
      get(dnn::pfCand_chHad_fromPV) = getValueLinear<int>(pfCands.at(index_chH).fromPV(), 0, 3, true);
      get(dnn::pfCand_chHad_puppiWeight) = getValue(pfCands.at(index_chH).puppiWeight());
      get(dnn::pfCand_chHad_puppiWeightNoLep) = getValue(pfCands.at(index_chH).puppiWeightNoLep());
      get(dnn::pfCand_chHad_charge) = getValue(pfCands.at(index_chH).charge());
      get(dnn::pfCand_chHad_lostInnerHits) = getValue<int>(pfCands.at(index_chH).lostInnerHits());
      get(dnn::pfCand_chHad_numberOfPixelHits) = getValueLinear(pfCands.at(index_chH).numberOfPixelHits(), 0, 12, true);
      get(dnn::pfCand_chHad_vertex_dx) =
          getValueNorm(pfCands.at(index_chH).vertex().x() - pv.position().x(), 0.0005f, 1.735f);
      get(dnn::pfCand_chHad_vertex_dy) =
          getValueNorm(pfCands.at(index_chH).vertex().y() - pv.position().y(), -0.0008f, 1.752f);
      get(dnn::pfCand_chHad_vertex_dz) =
          getValueNorm(pfCands.at(index_chH).vertex().z() - pv.position().z(), -0.0201f, 8.333f);
      get(dnn::pfCand_chHad_vertex_dx_tauFL) = getValueNorm(
          pfCands.at(index_chH).vertex().x() - pv.position().x() - tau.flightLength().x(), -0.0014f, 1.93f);
      get(dnn::pfCand_chHad_vertex_dy_tauFL) = getValueNorm(
          pfCands.at(index_chH).vertex().y() - pv.position().y() - tau.flightLength().y(), 0.0022f, 1.948f);
      get(dnn::pfCand_chHad_vertex_dz_tauFL) = getValueNorm(
          pfCands.at(index_chH).vertex().z() - pv.position().z() - tau.flightLength().z(), -0.0138f, 8.622f);

      const bool hasTrackDetails = pfCands.at(index_chH).hasTrackDetails();
      if (hasTrackDetails) {
        get(dnn::pfCand_chHad_hasTrackDetails) = hasTrackDetails;
        get(dnn::pfCand_chHad_dxy) = getValueNorm(pfCands.at(index_chH).dxy(), -0.012f, 2.386f);
        get(dnn::pfCand_chHad_dxy_sig) =
            getValueNorm(std::abs(pfCands.at(index_chH).dxy()) / pfCands.at(index_chH).dxyError(), 6.417f, 36.28f);
        get(dnn::pfCand_chHad_dz) = getValueNorm(pfCands.at(index_chH).dz(), -0.0246f, 7.618f);
        get(dnn::pfCand_chHad_dz_sig) =
            getValueNorm(std::abs(pfCands.at(index_chH).dz()) / pfCands.at(index_chH).dzError(), 301.3f, 491.1f);
        get(dnn::pfCand_chHad_track_chi2_ndof) =
            pfCands.at(index_chH).pseudoTrack().ndof() > 0
                ? getValueNorm(pfCands.at(index_chH).pseudoTrack().chi2() / pfCands.at(index_chH).pseudoTrack().ndof(),
                               0.7876f,
                               3.694f)
                : 0;
        get(dnn::pfCand_chHad_track_ndof) =
            pfCands.at(index_chH).pseudoTrack().ndof() > 0
                ? getValueNorm(pfCands.at(index_chH).pseudoTrack().ndof(), 13.92f, 6.581f)
                : 0;
      }
      float hcal_fraction = 0.;
      if (pfCands.at(index_chH).pdgId() == 1 || pfCands.at(index_chH).pdgId() == 130) {
        hcal_fraction = pfCands.at(index_chH).hcalFraction();
      } else if (pfCands.at(index_chH).isIsolatedChargedHadron()) {
        hcal_fraction = pfCands.at(index_chH).rawHcalFraction();
      }
      get(dnn::pfCand_chHad_hcalFraction) = getValue(hcal_fraction);
      get(dnn::pfCand_chHad_rawCaloFraction) = getValueLinear(pfCands.at(index_chH).rawCaloFraction(), 0.f, 2.6f, true);
    }
    if (valid_nH) {
      size_t index_nH = cell_map.at(CellObjectType::PfCand_neutralHadron);

      get(dnn::pfCand_nHad_valid) = valid_nH;
      get(dnn::pfCand_nHad_rel_pt) = getValueNorm(pfCands.at(index_nH).polarP4().pt() / tau.polarP4().pt(),
                                                  is_inner ? 0.3163f : 0.0502f,
                                                  is_inner ? 0.2769f : 0.4266f);
      get(dnn::pfCand_nHad_deta) = getValueLinear(pfCands.at(index_nH).polarP4().eta() - tau.polarP4().eta(),
                                                  is_inner ? -0.1f : -0.5f,
                                                  is_inner ? 0.1f : 0.5f,
                                                  false);
      get(dnn::pfCand_nHad_dphi) = getValueLinear(
          dPhi(tau.polarP4(), pfCands.at(index_nH).polarP4()), is_inner ? -0.1f : -0.5f, is_inner ? 0.1f : 0.5f, false);
      get(dnn::pfCand_nHad_puppiWeight) = getValue(pfCands.at(index_nH).puppiWeight());
      get(dnn::pfCand_nHad_puppiWeightNoLep) = getValue(pfCands.at(index_nH).puppiWeightNoLep());
      float hcal_fraction = 0.;
      if (pfCands.at(index_nH).pdgId() == 1 || pfCands.at(index_nH).pdgId() == 130) {
        hcal_fraction = pfCands.at(index_nH).hcalFraction();
      } else if (pfCands.at(index_nH).isIsolatedChargedHadron()) {
        hcal_fraction = pfCands.at(index_nH).rawHcalFraction();
      }
      get(dnn::pfCand_nHad_hcalFraction) = getValue(hcal_fraction);
    }
    checkInputs(inputs, is_inner ? "hadron_inner_block" : "hadron_outer_block", dnn::NumberOfInputs);
  }

  template <typename dnn>
  tensorflow::Tensor createInputsV1(const TauType& tau,
                                    const ElectronCollection& electrons,
                                    const MuonCollection& muons) const {
    static constexpr bool check_all_set = false;
    static constexpr float default_value_for_set_check = -42;

    tensorflow::Tensor inputs(tensorflow::DT_FLOAT, {1, dnn_inputs_2017v1::NumberOfInputs});
    const auto& get = [&](int var_index) -> float& { return inputs.matrix<float>()(0, var_index); };
    auto leadChargedHadrCand = dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get());

    if (check_all_set) {
      for (int var_index = 0; var_index < dnn::NumberOfInputs; ++var_index) {
        get(var_index) = default_value_for_set_check;
      }
    }

    get(dnn::pt) = tau.p4().pt();
    get(dnn::eta) = tau.p4().eta();
    get(dnn::mass) = tau.p4().mass();
    get(dnn::decayMode) = tau.decayMode();
    get(dnn::chargedIsoPtSum) = tau.tauID("chargedIsoPtSum");
    get(dnn::neutralIsoPtSum) = tau.tauID("neutralIsoPtSum");
    get(dnn::neutralIsoPtSumWeight) = tau.tauID("neutralIsoPtSumWeight");
    get(dnn::photonPtSumOutsideSignalCone) = tau.tauID("photonPtSumOutsideSignalCone");
    get(dnn::puCorrPtSum) = tau.tauID("puCorrPtSum");
    get(dnn::dxy) = tau.dxy();
    get(dnn::dxy_sig) = tau.dxy_Sig();
    get(dnn::dz) = leadChargedHadrCand ? leadChargedHadrCand->dz() : default_value;
    get(dnn::ip3d) = tau.ip3d();
    get(dnn::ip3d_sig) = tau.ip3d_Sig();
    get(dnn::hasSecondaryVertex) = tau.hasSecondaryVertex();
    get(dnn::flightLength_r) = tau.flightLength().R();
    get(dnn::flightLength_dEta) = dEta(tau.flightLength(), tau.p4());
    get(dnn::flightLength_dPhi) = dPhi(tau.flightLength(), tau.p4());
    get(dnn::flightLength_sig) = tau.flightLengthSig();
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
    get(dnn::leadingTrackNormChi2) = tau.leadingTrackNormChi2();
    get(dnn::e_ratio) = reco::tau::eratio(tau);
    get(dnn::gj_angle_diff) = calculateGottfriedJacksonAngleDifference(tau);
    get(dnn::n_photons) = reco::tau::n_photons_total(tau);
    get(dnn::emFraction) = tau.emFraction_MVA();
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
    get(dnn::leadChargedCand_etaAtEcalEntrance) = tau.etaAtEcalEntranceLeadChargedCand();
    get(dnn::leadChargedCand_pt) = tau.ptLeadChargedCand();

    get(dnn::leadChargedHadrCand_HoP) = default_value;
    get(dnn::leadChargedHadrCand_EoP) = default_value;
    if (tau.leadChargedHadrCand()->pt() > 0) {
      get(dnn::leadChargedHadrCand_HoP) = tau.hcalEnergyLeadChargedHadrCand() / tau.leadChargedHadrCand()->pt();
      get(dnn::leadChargedHadrCand_EoP) = tau.ecalEnergyLeadChargedHadrCand() / tau.leadChargedHadrCand()->pt();
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

  template <typename CandidateCollection>
  static void processSignalPFComponents(const pat::Tau& tau,
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

  template <typename CandidateCollection>
  static void processIsolationPFComponents(const pat::Tau& tau,
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
  static bool calculateGottfriedJacksonAngleDifference(const pat::Tau& tau, double& gj_diff) {
    if (tau.hasSecondaryVertex()) {
      static constexpr double mTau = 1.77682;
      const double mAOne = tau.p4().M();
      const double pAOneMag = tau.p();
      const double argumentThetaGJmax = (std::pow(mTau, 2) - std::pow(mAOne, 2)) / (2 * mTau * pAOneMag);
      const double argumentThetaGJmeasured =
          tau.p4().Vect().Dot(tau.flightLength()) / (pAOneMag * tau.flightLength().R());
      if (std::abs(argumentThetaGJmax) <= 1. && std::abs(argumentThetaGJmeasured) <= 1.) {
        double thetaGJmax = std::asin(argumentThetaGJmax);
        double thetaGJmeasured = std::acos(argumentThetaGJmeasured);
        gj_diff = thetaGJmeasured - thetaGJmax;
        return true;
      }
    }
    return false;
  }

  static float calculateGottfriedJacksonAngleDifference(const pat::Tau& tau) {
    double gj_diff;
    if (calculateGottfriedJacksonAngleDifference(tau, gj_diff))
      return static_cast<float>(gj_diff);
    return default_value;
  }

  static bool isInEcalCrack(double eta) {
    const double abs_eta = std::abs(eta);
    return abs_eta > 1.46 && abs_eta < 1.558;
  }

  static const pat::Electron* findMatchedElectron(const pat::Tau& tau,
                                                  const pat::ElectronCollection& electrons,
                                                  double deltaR) {
    const double dR2 = deltaR * deltaR;
    const pat::Electron* matched_ele = nullptr;
    for (const auto& ele : electrons) {
      if (reco::deltaR2(tau.p4(), ele.p4()) < dR2 && (!matched_ele || matched_ele->pt() < ele.pt())) {
        matched_ele = &ele;
      }
    }
    return matched_ele;
  }

private:
  edm::EDGetTokenT<ElectronCollection> electrons_token_;
  edm::EDGetTokenT<MuonCollection> muons_token_;
  edm::EDGetTokenT<double> rho_token_;
  std::string input_layer_, output_layer_;
  const unsigned version;
  const int debug_level;
  const bool disable_dxy_pca_;
  std::unique_ptr<tensorflow::Tensor> tauBlockTensor_;
  std::array<std::unique_ptr<tensorflow::Tensor>, 2> eGammaTensor_, muonTensor_, hadronsTensor_, convTensor_,
      zeroOutputTensor_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeepTauId);
