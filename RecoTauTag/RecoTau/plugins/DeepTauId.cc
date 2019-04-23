/*
 * \class DeepTauId
 *
 * Tau identification using Deep NN.
 *
 * \author Konstantin Androsov, INFN Pisa
 */

#include "RecoTauTag/RecoTau/interface/DeepTauBase.h"

namespace {

struct dnn_inputs_2017v1 {
    enum vars {
        pt = 0, eta, mass, decayMode, chargedIsoPtSum, neutralIsoPtSum, neutralIsoPtSumWeight,
        photonPtSumOutsideSignalCone, puCorrPtSum,
        dxy, dxy_sig, dz, ip3d, ip3d_sig,
        hasSecondaryVertex, flightLength_r, flightLength_dEta, flightLength_dPhi,
        flightLength_sig, leadChargedHadrCand_pt, leadChargedHadrCand_dEta,
        leadChargedHadrCand_dPhi, leadChargedHadrCand_mass, pt_weighted_deta_strip,
        pt_weighted_dphi_strip, pt_weighted_dr_signal, pt_weighted_dr_iso,
        leadingTrackNormChi2, e_ratio, gj_angle_diff, n_photons, emFraction,
        has_gsf_track, inside_ecal_crack,
        gsf_ele_matched, gsf_ele_pt, gsf_ele_dEta, gsf_ele_dPhi, gsf_ele_mass, gsf_ele_Ee,
        gsf_ele_Egamma, gsf_ele_Pin, gsf_ele_Pout, gsf_ele_EtotOverPin, gsf_ele_Eecal,
        gsf_ele_dEta_SeedClusterTrackAtCalo, gsf_ele_dPhi_SeedClusterTrackAtCalo, gsf_ele_mvaIn_sigmaEtaEta,
        gsf_ele_mvaIn_hadEnergy,
        gsf_ele_mvaIn_deltaEta, gsf_ele_Chi2NormGSF, gsf_ele_GSFNumHits, gsf_ele_GSFTrackResol,
        gsf_ele_GSFTracklnPt, gsf_ele_Chi2NormKF, gsf_ele_KFNumHits,
        leadChargedCand_etaAtEcalEntrance, leadChargedCand_pt, leadChargedHadrCand_HoP,
        leadChargedHadrCand_EoP, tau_visMass_innerSigCone,
        n_matched_muons, muon_pt, muon_dEta, muon_dPhi,
        muon_n_matches_DT_1, muon_n_matches_DT_2, muon_n_matches_DT_3, muon_n_matches_DT_4,
        muon_n_matches_CSC_1, muon_n_matches_CSC_2, muon_n_matches_CSC_3, muon_n_matches_CSC_4,
        muon_n_hits_DT_2, muon_n_hits_DT_3, muon_n_hits_DT_4,
        muon_n_hits_CSC_2, muon_n_hits_CSC_3, muon_n_hits_CSC_4,
        muon_n_hits_RPC_2, muon_n_hits_RPC_3, muon_n_hits_RPC_4,
        muon_n_stations_with_matches_03, muon_n_stations_with_hits_23,
        signalChargedHadrCands_sum_innerSigCone_pt, signalChargedHadrCands_sum_innerSigCone_dEta,
        signalChargedHadrCands_sum_innerSigCone_dPhi, signalChargedHadrCands_sum_innerSigCone_mass,
        signalChargedHadrCands_sum_outerSigCone_pt, signalChargedHadrCands_sum_outerSigCone_dEta,
        signalChargedHadrCands_sum_outerSigCone_dPhi, signalChargedHadrCands_sum_outerSigCone_mass,
        signalChargedHadrCands_nTotal_innerSigCone, signalChargedHadrCands_nTotal_outerSigCone,
        signalNeutrHadrCands_sum_innerSigCone_pt, signalNeutrHadrCands_sum_innerSigCone_dEta,
        signalNeutrHadrCands_sum_innerSigCone_dPhi, signalNeutrHadrCands_sum_innerSigCone_mass,
        signalNeutrHadrCands_sum_outerSigCone_pt, signalNeutrHadrCands_sum_outerSigCone_dEta,
        signalNeutrHadrCands_sum_outerSigCone_dPhi, signalNeutrHadrCands_sum_outerSigCone_mass,
        signalNeutrHadrCands_nTotal_innerSigCone, signalNeutrHadrCands_nTotal_outerSigCone,
        signalGammaCands_sum_innerSigCone_pt, signalGammaCands_sum_innerSigCone_dEta,
        signalGammaCands_sum_innerSigCone_dPhi, signalGammaCands_sum_innerSigCone_mass,
        signalGammaCands_sum_outerSigCone_pt, signalGammaCands_sum_outerSigCone_dEta,
        signalGammaCands_sum_outerSigCone_dPhi, signalGammaCands_sum_outerSigCone_mass,
        signalGammaCands_nTotal_innerSigCone, signalGammaCands_nTotal_outerSigCone,
        isolationChargedHadrCands_sum_pt, isolationChargedHadrCands_sum_dEta,
        isolationChargedHadrCands_sum_dPhi, isolationChargedHadrCands_sum_mass,
        isolationChargedHadrCands_nTotal,
        isolationNeutrHadrCands_sum_pt, isolationNeutrHadrCands_sum_dEta,
        isolationNeutrHadrCands_sum_dPhi, isolationNeutrHadrCands_sum_mass,
        isolationNeutrHadrCands_nTotal,
        isolationGammaCands_sum_pt, isolationGammaCands_sum_dEta,
        isolationGammaCands_sum_dPhi, isolationGammaCands_sum_mass,
        isolationGammaCands_nTotal,
        NumberOfInputs
    };

    static constexpr int NumberOfOutputs = 4;
};

namespace dnn_inputs_2017_v2 {
namespace TauBlockInputs {
    enum vars {
        rho = 0, tau_pt, tau_eta, tau_phi, tau_mass, tau_E_over_pt, tau_charge, tau_n_charged_prongs,
        tau_n_neutral_prongs, chargedIsoPtSum, chargedIsoPtSumdR03_over_dR05, footprintCorrection,
        neutralIsoPtSum, neutralIsoPtSumWeight_over_neutralIsoPtSum, neutralIsoPtSumWeightdR03_over_neutralIsoPtSum,
        neutralIsoPtSumdR03_over_dR05, photonPtSumOutsideSignalCone, puCorrPtSum,
        tau_dxy_pca_x, tau_dxy_pca_y, tau_dxy_pca_z, tau_dxy_valid, tau_dxy, tau_dxy_sig,
        tau_ip3d_valid, tau_ip3d, tau_ip3d_sig, tau_dz, tau_dz_sig_valid, tau_dz_sig,
        tau_flightLength_x, tau_flightLength_y, tau_flightLength_z, tau_flightLength_sig,
        tau_pt_weighted_deta_strip, tau_pt_weighted_dphi_strip, tau_pt_weighted_dr_signal,
        tau_pt_weighted_dr_iso, tau_leadingTrackNormChi2, tau_e_ratio_valid, tau_e_ratio,
        tau_gj_angle_diff_valid, tau_gj_angle_diff, tau_n_photons, tau_emFraction,
        tau_inside_ecal_crack, leadChargedCand_etaAtEcalEntrance_minus_tau_eta, NumberOfInputs
    };
}

namespace EgammaBlockInputs {
    enum vars {rho = 0, tau_pt, tau_eta, tau_inside_ecal_crack, pfCand_ele_valid, pfCand_ele_rel_pt,
        pfCand_ele_deta, pfCand_ele_dphi, pfCand_ele_pvAssociationQuality, pfCand_ele_puppiWeight,
        pfCand_ele_charge, pfCand_ele_lostInnerHits, pfCand_ele_numberOfPixelHits, pfCand_ele_vertex_dx,
        pfCand_ele_vertex_dy, pfCand_ele_vertex_dz, pfCand_ele_vertex_dx_tauFL, pfCand_ele_vertex_dy_tauFL,
        pfCand_ele_vertex_dz_tauFL, pfCand_ele_hasTrackDetails, pfCand_ele_dxy, pfCand_ele_dxy_sig,
        pfCand_ele_dz, pfCand_ele_dz_sig, pfCand_ele_track_chi2_ndof, pfCand_ele_track_ndof,
        pfCand_gamma_valid, pfCand_gamma_rel_pt, pfCand_gamma_deta, pfCand_gamma_dphi,
        pfCand_gamma_pvAssociationQuality, pfCand_gamma_fromPV, pfCand_gamma_puppiWeight,
        pfCand_gamma_puppiWeightNoLep, pfCand_gamma_lostInnerHits, pfCand_gamma_numberOfPixelHits,
        pfCand_gamma_vertex_dx, pfCand_gamma_vertex_dy, pfCand_gamma_vertex_dz, pfCand_gamma_vertex_dx_tauFL,
        pfCand_gamma_vertex_dy_tauFL, pfCand_gamma_vertex_dz_tauFL, pfCand_gamma_hasTrackDetails,
        pfCand_gamma_dxy, pfCand_gamma_dxy_sig, pfCand_gamma_dz, pfCand_gamma_dz_sig,
        pfCand_gamma_track_chi2_ndof, pfCand_gamma_track_ndof, ele_valid, ele_rel_pt, ele_deta, ele_dphi,
        ele_cc_valid, ele_cc_ele_rel_energy, ele_cc_gamma_rel_energy, ele_cc_n_gamma,
        ele_rel_trackMomentumAtVtx, ele_rel_trackMomentumAtCalo, ele_rel_trackMomentumOut,
        ele_rel_trackMomentumAtEleClus, ele_rel_trackMomentumAtVtxWithConstraint,
        ele_rel_ecalEnergy, ele_ecalEnergy_sig, ele_eSuperClusterOverP,
        ele_eSeedClusterOverP, ele_eSeedClusterOverPout, ele_eEleClusterOverPout,
        ele_deltaEtaSuperClusterTrackAtVtx, ele_deltaEtaSeedClusterTrackAtCalo,
        ele_deltaEtaEleClusterTrackAtCalo, ele_deltaPhiEleClusterTrackAtCalo,
        ele_deltaPhiSuperClusterTrackAtVtx, ele_deltaPhiSeedClusterTrackAtCalo,
        ele_mvaInput_earlyBrem, ele_mvaInput_lateBrem, ele_mvaInput_sigmaEtaEta,
        ele_mvaInput_hadEnergy, ele_mvaInput_deltaEta, ele_gsfTrack_normalizedChi2,
        ele_gsfTrack_numberOfValidHits, ele_rel_gsfTrack_pt, ele_gsfTrack_pt_sig,
        ele_has_closestCtfTrack, ele_closestCtfTrack_normalizedChi2, ele_closestCtfTrack_numberOfValidHits,
        NumberOfInputs
    };
}

namespace MuonBlockInputs {
    enum vars {rho = 0, tau_pt, tau_eta, tau_inside_ecal_crack};
}

namespace HadronBlockInputs {
    enum vars {rho = 0, tau_pt, tau_eta, tau_inside_ecal_crack, pfCand_chHad_valid,
        pfCand_chHad_rel_pt, pfCand_chHad_deta, pfCand_chHad_dphi, pfCand_chHad_leadChargedHadrCand,
        pfCand_chHad_pvAssociationQuality, pfCand_chHad_fromPV, pfCand_chHad_puppiWeight,
        pfCand_chHad_puppiWeightNoLep, pfCand_chHad_charge, pfCand_chHad_lostInnerHits,
        pfCand_chHad_numberOfPixelHits, pfCand_chHad_vertex_dx, pfCand_chHad_vertex_dy,
        pfCand_chHad_vertex_dz, pfCand_chHad_vertex_dx_tauFL, pfCand_chHad_vertex_dy_tauFL,
        pfCand_chHad_vertex_dz_tauFL, pfCand_chHad_hasTrackDetails, pfCand_chHad_dxy,
        pfCand_chHad_dxy_sig, pfCand_chHad_dz, pfCand_chHad_dz_sig, pfCand_chHad_track_chi2_ndof,
        pfCand_chHad_track_ndof, pfCand_chHad_hcalFraction, pfCand_chHad_rawCaloFraction,
        pfCand_nHad_valid, pfCand_nHad_rel_pt, pfCand_nHad_deta, pfCand_nHad_dphi,
        pfCand_nHad_puppiWeight, pfCand_nHad_puppiWeightNoLep, pfCand_nHad_hcalFraction, NumberOfInputs};
}

    static constexpr int NumberOfOutputs = 4;
}

template<typename LVector1, typename LVector2>
float dEta(const LVector1& p4, const LVector2& tau_p4)
{
    return static_cast<float>(p4.eta() - tau_p4.eta());
}

template<typename LVector1, typename LVector2>
float dPhi(const LVector1& p4, const LVector2& tau_p4)
{
    return static_cast<float>(ROOT::Math::VectorUtil::DeltaPhi(p4, tau_p4));
}

namespace MuonSubdetId {
  enum { DT = 1, CSC = 2, RPC = 3, GEM = 4, ME0 = 5 };
}

struct MuonHitMatchV1 {
    static constexpr int n_muon_stations = 4;

    std::map<int, std::vector<UInt_t>> n_matches, n_hits;
    unsigned n_muons{0};
    const pat::Muon* best_matched_muon{nullptr};
    double deltaR2_best_match{-1};

    MuonHitMatchV1()
    {
        n_matches[MuonSubdetId::DT].assign(n_muon_stations, 0);
        n_matches[MuonSubdetId::CSC].assign(n_muon_stations, 0);
        n_matches[MuonSubdetId::RPC].assign(n_muon_stations, 0);
        n_hits[MuonSubdetId::DT].assign(n_muon_stations, 0);
        n_hits[MuonSubdetId::CSC].assign(n_muon_stations, 0);
        n_hits[MuonSubdetId::RPC].assign(n_muon_stations, 0);
    }

    void addMatchedMuon(const pat::Muon& muon, const pat::Tau& tau)
    {
        static constexpr int n_stations = 4;

        ++n_muons;
        const double dR2 = reco::deltaR2(tau.p4(), muon.p4());
        if(!best_matched_muon || dR2 < deltaR2_best_match) {
            best_matched_muon = &muon;
            deltaR2_best_match = dR2;
        }

        for(const auto& segment : muon.matches()) {
            if(segment.segmentMatches.empty()) continue;
            if(n_matches.count(segment.detector()))
                ++n_matches.at(segment.detector()).at(segment.station() - 1);
        }

        if(muon.outerTrack().isNonnull()) {
            const auto& hit_pattern = muon.outerTrack()->hitPattern();
            for(int hit_index = 0; hit_index < hit_pattern.numberOfAllHits(reco::HitPattern::TRACK_HITS); ++hit_index) {
                auto hit_id = hit_pattern.getHitPattern(reco::HitPattern::TRACK_HITS, hit_index);
                if(hit_id == 0) break;
                if(hit_pattern.muonHitFilter(hit_id) && (hit_pattern.getHitType(hit_id) == TrackingRecHit::valid
                                                         || hit_pattern.getHitType(hit_id == TrackingRecHit::bad))) {
                    const int station = hit_pattern.getMuonStation(hit_id) - 1;
                    if(station > 0 && station < n_stations) {
                        std::vector<UInt_t>* muon_n_hits = nullptr;
                        if(hit_pattern.muonDTHitFilter(hit_id))
                            muon_n_hits = &n_hits.at(MuonSubdetId::DT);
                        else if(hit_pattern.muonCSCHitFilter(hit_id))
                            muon_n_hits = &n_hits.at(MuonSubdetId::CSC);
                        else if(hit_pattern.muonRPCHitFilter(hit_id))
                            muon_n_hits = &n_hits.at(MuonSubdetId::RPC);

                        if(muon_n_hits)
                            ++muon_n_hits->at(station);
                    }
                }
            }
        }
    }

    static std::vector<const pat::Muon*> findMatchedMuons(const pat::Tau& tau, const pat::MuonCollection& muons,
                                                           double deltaR, double minPt)
    {
        const reco::Muon* hadr_cand_muon = nullptr;
        if(tau.leadPFChargedHadrCand().isNonnull() && tau.leadPFChargedHadrCand()->muonRef().isNonnull())
            hadr_cand_muon = tau.leadPFChargedHadrCand()->muonRef().get();
        std::vector<const pat::Muon*> matched_muons;
        const double dR2 = deltaR*deltaR;
        for(const pat::Muon& muon : muons) {
            const reco::Muon* reco_muon = &muon;
            if(muon.pt() <= minPt) continue;
            if(reco_muon == hadr_cand_muon) continue;
            if(reco::deltaR2(tau.p4(), muon.p4()) >= dR2) continue;
            matched_muons.push_back(&muon);
        }
        return matched_muons;
    }

    template<typename dnn, typename TensorElemGet>
    void fillTensor(const TensorElemGet& get, const pat::Tau& tau, float default_value) const
    {
        get(dnn::n_matched_muons) = n_muons;
        get(dnn::muon_pt) = best_matched_muon != nullptr ? best_matched_muon->p4().pt() : default_value;
        get(dnn::muon_dEta) = best_matched_muon != nullptr
                ? dEta(best_matched_muon->p4(), tau.p4()) : default_value;
        get(dnn::muon_dPhi) = best_matched_muon != nullptr
                ? dPhi(best_matched_muon->p4(), tau.p4()) : default_value;
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
    unsigned countMuonStationsWithMatches(size_t first_station, size_t last_station) const
    {
        static const std::map<int, std::vector<bool>> masks = {
            { MuonSubdetId::DT, { false, false, false, false } },
            { MuonSubdetId::CSC, { true, false, false, false } },
            { MuonSubdetId::RPC, { false, false, false, false } },
        };
        unsigned cnt = 0;
        for(unsigned n = first_station; n <= last_station; ++n) {
            for(const auto& match : n_matches) {
                if(!masks.at(match.first).at(n) && match.second.at(n) > 0) ++cnt;
            }
        }
        return cnt;
    }

    unsigned countMuonStationsWithHits(size_t first_station, size_t last_station) const
    {
        static const std::map<int, std::vector<bool>> masks = {
            { MuonSubdetId::DT, { false, false, false, false } },
            { MuonSubdetId::CSC, { false, false, false, false } },
            { MuonSubdetId::RPC, { false, false, false, false } },
        };

        unsigned cnt = 0;
        for(unsigned n = first_station; n <= last_station; ++n) {
            for(const auto& hit : n_hits) {
                if(!masks.at(hit.first).at(n) && hit.second.at(n) > 0) ++cnt;
            }
        }
        return cnt;
    }
};


enum class CellObjectType { PfCand_electron, PfCand_muon, PfCand_chargedHadron, PfCand_neutralHadron,
                            PfCand_gamma, Electron, Muon };

template<typename Object>
CellObjectType GetCellObjectType(const Object&);
template<>
CellObjectType GetCellObjectType(const pat::Electron&) { return CellObjectType::Electron; }
template<>
CellObjectType GetCellObjectType(const pat::Muon&) { return CellObjectType::Muon; }

template<>
CellObjectType GetCellObjectType(const pat::PackedCandidate& cand)
{
    static const std::map<int, CellObjectType> obj_types = {
        { 11, CellObjectType::PfCand_electron },
        { 13, CellObjectType::PfCand_muon },
        { 22, CellObjectType::PfCand_gamma },
        { 130, CellObjectType::PfCand_neutralHadron },
        { 211, CellObjectType::PfCand_chargedHadron }
    };

    auto iter = obj_types.find(std::abs(cand.pdgId()));
    if(iter == obj_types.end())
        throw cms::Exception("DeepTauId") << "Unknown object pdg id = " << cand.pdgId();
    return iter->second;
}


using Cell = std::map<CellObjectType, size_t>;
struct CellIndex {
    int eta, phi;

    bool operator<(const CellIndex& other) const
    {
        if(eta != other.eta) return eta < other.eta;
        return phi < other.phi;
    }
};

class CellGrid {
public:
    using Map = std::map<CellIndex, Cell>;
    using const_iterator = Map::const_iterator;

    CellGrid(unsigned n_cells_eta, unsigned n_cells_phi, double cell_size_eta, double cell_size_phi) :
        nCellsEta(n_cells_eta), nCellsPhi(n_cells_phi), nTotal(nCellsEta * nCellsPhi),
        cellSizeEta(cell_size_eta), cellSizePhi(cell_size_phi)
    {
        if(nCellsEta % 2 != 1 || nCellsEta < 1)
            throw cms::Exception("DeepTauId") << "Invalid number of eta cells.";
        if(nCellsPhi % 2 != 1 || nCellsPhi < 1)
            throw cms::Exception("DeepTauId") << "Invalid number of phi cells.";
        if(cellSizeEta <= 0 || cellSizePhi <= 0)
            throw cms::Exception("DeepTauId") << "Invalid cell size.";
    }

    int maxEtaIndex() const { return static_cast<int>((nCellsEta - 1) / 2); }
    int maxPhiIndex() const { return static_cast<int>((nCellsPhi - 1) / 2); }
    double maxDeltaEta() const { return cellSizeEta * (0.5 + maxEtaIndex()); }
    double maxDeltaPhi() const { return cellSizePhi * (0.5 + maxPhiIndex()); }
    int getEtaTensorIndex(const CellIndex& cellIndex) const { return cellIndex.eta + maxEtaIndex(); }
    int getPhiTensorIndex(const CellIndex& cellIndex) const { return cellIndex.phi + maxPhiIndex(); }

    bool TryGetCellIndex(double deltaEta, double deltaPhi, CellIndex& cellIndex) const
    {
        static auto getCellIndex = [](double x, double maxX, double size, int& index) {
            const double absX = std::abs(x);
            if(absX > maxX) return false;
            const double absIndex = std::floor(std::abs(absX / size - 0.5));
            index = static_cast<int>(std::copysign(absIndex, x));
            return true;
        };

        return getCellIndex(deltaEta, maxDeltaEta(), cellSizeEta, cellIndex.eta)
               && getCellIndex(deltaPhi, maxDeltaPhi(), cellSizePhi, cellIndex.phi);
    }

    Cell& operator[](const CellIndex& cellIndex) { return cells[cellIndex]; }
    const Cell& at(const CellIndex& cellIndex) const { return cells.at(cellIndex); }
    const_iterator begin() const { return cells.begin(); }
    const_iterator end() const { return cells.end(); }

public:
    const unsigned nCellsEta, nCellsPhi, nTotal;
    const double cellSizeEta, cellSizePhi;

private:
    std::map<CellIndex, Cell> cells;
};

} // anonymous namespace

class DeepTauId : public deep_tau::DeepTauBase {
public:

    static constexpr float default_value = -999.;

    static const OutputCollection& GetOutputs()
    {
        static constexpr size_t e_index = 0, mu_index = 1, tau_index = 2, jet_index = 3;
        static const OutputCollection outputs_ = {
            { "VSe", Output({tau_index}, {e_index, tau_index}) },
            { "VSmu", Output({tau_index}, {mu_index, tau_index}) },
            { "VSjet", Output({tau_index}, {jet_index, tau_index}) },
        };
        return outputs_;
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions)
    {
        edm::ParameterSetDescription desc;
        desc.add<edm::InputTag>("electrons", edm::InputTag("slimmedElectrons"));
        desc.add<edm::InputTag>("muons", edm::InputTag("slimmedMuons"));
        desc.add<edm::InputTag>("taus", edm::InputTag("slimmedTaus"));
        desc.add<edm::InputTag>("pfcands", edm::InputTag("packedPFCandidates"));
        desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
        desc.add<edm::InputTag>("rho", edm::InputTag("fixedGridRhoAll"));
        desc.add<std::string>("graph_file", "RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2.pb");
        desc.add<bool>("mem_mapped", false);
        desc.add<unsigned>("version", 2);

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
    explicit DeepTauId(const edm::ParameterSet& cfg, const deep_tau::DeepTauCache* cache) :
        DeepTauBase(cfg, GetOutputs(), cache),
        electrons_token_(consumes<ElectronCollection>(cfg.getParameter<edm::InputTag>("electrons"))),
        muons_token_(consumes<MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))),
        rho_token_(consumes<double>(cfg.getParameter<edm::InputTag>("rho"))),
        version(cfg.getParameter<unsigned>("version"))
    {
        if(version == 1) {
            input_layer_ = cache_->getGraph().node(0).name();
            output_layer_ = cache_->getGraph().node(cache_->getGraph().node_size() - 1).name();
            const auto& shape = cache_->getGraph().node(0).attr().at("shape").shape();
            if(shape.dim(1).size() != dnn_inputs_2017v1::NumberOfInputs)
                throw cms::Exception("DeepTauId") << "number of inputs does not match the expected inputs for the given version";
        } else if(version == 2) {

        } else {
            throw cms::Exception("DeepTauId") << "version " << version << " is not supported.";
        }
    }

    static std::unique_ptr<deep_tau::DeepTauCache> initializeGlobalCache(const edm::ParameterSet& cfg)
    {
        return DeepTauBase::initializeGlobalCache(cfg);
    }

    static void globalEndJob(const deep_tau::DeepTauCache* cache_)
    {
        return DeepTauBase::globalEndJob(cache_);
    }
private:
    static constexpr float pi = boost::math::constants::pi<float>();

    template<typename T>
    static float GetValue(T value)
    {
        return std::isnormal(value) ? static_cast<float>(value) : 0.f;
    }

    template<typename T>
    static float GetValueLinear(T value, float min_value, float max_value, bool positive)
    {
        const float fixed_value = GetValue(value);
        const float clamped_value = std::clamp(fixed_value, min_value, max_value);
        float transformed_value = (clamped_value - min_value) / (max_value - min_value);
        if(!positive)
            transformed_value = transformed_value * 2 - 1;
        return transformed_value;
    }

    template<typename T>
    static float GetValueNorm(T value, float mean, float sigma, float n_sigmas_max = 5)
    {
        const float fixed_value = GetValue(value);
        const float norm_value = (fixed_value - mean) / sigma;
        return std::clamp(norm_value, -n_sigmas_max, n_sigmas_max);
    }

private:
    tensorflow::Tensor getPredictions(edm::Event& event, const edm::EventSetup& es,
                                      edm::Handle<TauCollection> taus) override
    {
        edm::Handle<pat::ElectronCollection> electrons;
        event.getByToken(electrons_token_, electrons);

        edm::Handle<pat::MuonCollection> muons;
        event.getByToken(muons_token_, muons);

        edm::Handle<pat::PackedCandidateCollection> pfCands;
        event.getByToken(pfcand_token_, pfCands);

        edm::Handle<reco::VertexCollection> vertices;
        event.getByToken(vtx_token_, vertices);

        edm::Handle<double> rho;
        event.getByToken(rho_token_, rho);

        tensorflow::Tensor predictions(tensorflow::DT_FLOAT, { static_cast<int>(taus->size()),
                                       dnn_inputs_2017v1::NumberOfOutputs});
        for(size_t tau_index = 0; tau_index < taus->size(); ++tau_index) {
            std::vector<tensorflow::Tensor> pred_vector;
            if(version == 1)
                getPredictionsV1(taus->at(tau_index), *electrons, *muons, pred_vector);
            else if(version == 2)
                getPredictionsV2(taus->at(tau_index), *electrons, *muons, *pfCands, vertices->at(0), *rho, pred_vector);
            else
                throw cms::Exception("DeepTauId") << "version " << version << " is not supported.";
            for(int k = 0; k < dnn_inputs_2017v1::NumberOfOutputs; ++k)
                predictions.matrix<float>()(tau_index, k) = pred_vector[0].flat<float>()(k);
        }
        return predictions;
    }

    void getPredictionsV1(const TauType& tau, const pat::ElectronCollection& electrons,
                          const pat::MuonCollection& muons, std::vector<tensorflow::Tensor>& pred_vector)
    {
        const tensorflow::Tensor& inputs = createInputsV1<dnn_inputs_2017v1>(tau, electrons, muons);
        tensorflow::run(&(cache_->getSession()), { { input_layer_, inputs } }, { output_layer_ }, &pred_vector);
    }

    void getPredictionsV2(const TauType& tau, const pat::ElectronCollection& electrons,
                          const pat::MuonCollection& muons, const pat::PackedCandidateCollection& pfCands,
                          const reco::Vertex& pv, double rho, std::vector<tensorflow::Tensor>& pred_vector)
    {
        CellGrid inner_grid(11, 11, 0.02, 0.02);
        CellGrid outer_grid(21, 21, 0.05, 0.05);
        fillGrids(tau, electrons, inner_grid, outer_grid);
        fillGrids(tau, muons, inner_grid, outer_grid);
        fillGrids(tau, pfCands, inner_grid, outer_grid);

        const auto input_tau = createTauBlockInputs(tau, pv, rho);
        const auto input_inner_egamma = createEgammaBlockInputs(tau, pv, rho, electrons, pfCands, inner_grid, true);
        const auto input_inner_muon = createMuonBlockInputs(tau, pv, rho, muons, pfCands, inner_grid, true);
        const auto input_inner_hadrons = createHadronsBlockInputs(tau, pv, rho, pfCands, inner_grid, true);
        const auto input_outer_egamma = createEgammaBlockInputs(tau, pv, rho, electrons, pfCands, outer_grid, false);
        const auto input_outer_muon = createMuonBlockInputs(tau, pv, rho, muons, pfCands, outer_grid, false);
        const auto input_outer_hadrons = createHadronsBlockInputs(tau, pv, rho, pfCands, outer_grid, false);

        tensorflow::run(&(cache_->getSession()),
            { { "input_tau", input_tau },
              { "input_inner_egamma", input_inner_egamma}, { "input_outer_egamma", input_outer_egamma },
              { "input_inner_muon", input_inner_muon }, { "input_outer_muon", input_outer_muon },
              { "input_inner_hadrons", input_inner_hadrons }, { "input_outer_hadrons", input_outer_hadrons } },
            { "main_output" }, &pred_vector);
    }

    template<typename Collection>
    void fillGrids(const TauType& tau, const Collection& objects, CellGrid& inner_grid, CellGrid& outer_grid)
    {
        static constexpr double outer_dR2 = std::pow(0.5, 2);
        const double inner_radius = getInnerSignalConeRadius(tau.polarP4().pt());
        const double inner_dR2 = std::pow(inner_radius, 2);

        const auto addObject = [&](size_t n, double deta, double dphi, CellGrid& grid) {
            const auto& obj = objects.at(n);
            const CellObjectType obj_type = GetCellObjectType(obj);
            CellIndex cell_index;
            if(grid.TryGetCellIndex(deta, dphi, cell_index)) {
                Cell& cell = grid[cell_index];
                auto iter = cell.find(obj_type);
                if(iter != cell.end()) {
                     const auto& prev_obj = objects.at(iter->second);
                     if(obj.polarP4().pt() > prev_obj.polarP4().pt())
                        iter->second = n;
                } else {
                    cell[obj_type] = n;
                }
            }
        };

        for(size_t n = 0; n < objects.size(); ++n) {
            const auto& obj = objects.at(n);
            const double deta = obj.polarP4().eta() - tau.polarP4().eta();
            const double dphi = ROOT::Math::VectorUtil::DeltaPhi(tau.polarP4(), obj.polarP4());
            const double dR2 = std::pow(deta, 2) + std::pow(dphi, 2);
            if(dR2 < inner_dR2)
                addObject(n, deta, dphi, inner_grid);
            if(dR2 < outer_dR2)
                addObject(n, deta, dphi, outer_grid);
        }
    }


    tensorflow::Tensor createTauBlockInputs(const TauType& tau, const reco::Vertex& pv, double rho)
    {
        using namespace dnn_inputs_2017_v2;
        using namespace TauBlockInputs;

        tensorflow::Tensor inputs(tensorflow::DT_FLOAT, { 1,NumberOfInputs});
        const auto& get = [&](int var_index) -> float& { return inputs.matrix<float>()(0, var_index); };
        auto leadChargedHadrCand = dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get());

        get(rho) = GetValueNorm(rho, 21.49f, 9.713f);
        get(tau_pt) =  GetValueLinear(tau.polarP4().pt(), 20.f, 1000.f, true);
        get(tau_eta) = GetValueLinear(tau.polarP4().eta(), -2.3f, 2.3f, false);
        get(tau_phi) = GetValueLinear(tau.polarP4().phi(), -pi, pi, false);
        get(tau_mass) = GetValueNorm(tau.polarP4().mass(), 0.6669f, 0.6553f);
        get(tau_E_over_pt) = GetValueLinear(tau.p4().energy() / tau.p4().pt(), 1.f, 5.2f, true);
        get(tau_charge) = GetValue(tau.charge());
        get(tau_n_charged_prongs) = GetValueLinear(tau.decayMode() / 5 + 1, 1, 3, true);
        get(tau_n_neutral_prongs) = GetValueLinear(tau.decayMode() % 5, 0, 2, true);
        get(chargedIsoPtSum) = GetValueNorm(tau.tauID("chargedIsoPtSum"), 47.78f, 123.5f);
        get(chargedIsoPtSumdR03_over_dR05) = GetValue(tau.tauID("chargedIsoPtSumdR03") /
                                                                          tau.tauID("chargedIsoPtSum"));
        get(footprintCorrection) = GetValueNorm(tau.tauID("footprintCorrectiondR03"),9.029f, 26.42f);
        get(neutralIsoPtSum) = GetValueNorm(tau.tauID("neutralIsoPtSum"), 57.59f, 155.3f);
        get(neutralIsoPtSumWeight_over_neutralIsoPtSum) =
            GetValue(tau.tauID("neutralIsoPtSumWeight") / tau.tauID("neutralIsoPtSum"));
        get(neutralIsoPtSumWeightdR03_over_neutralIsoPtSum) =
            GetValue(tau.tauID("neutralIsoPtSumWeightdR03") / tau.tauID("neutralIsoPtSum"));
        get(neutralIsoPtSumdR03_over_dR05) = GetValue(tau.tauID("neutralIsoPtSumdR03") /
                                                                          tau.tauID("neutralIsoPtSum"));
        get(photonPtSumOutsideSignalCone) = GetValueNorm(tau.tauID("photonPtSumOutsideSignalConedR03"),
                                                                             1.731f, 6.846f);
        get(puCorrPtSum) = GetValueNorm(tau.tauID("puCorrPtSum"), 22.38f, 16.34f);
        get(tau_dxy_pca_x) = GetValueNorm(tau.dxy_PCA().x(), -0.0241f, 0.0074f);
        get(tau_dxy_pca_y) = GetValueNorm(tau.dxy_PCA().y(),0.0675f, 0.0128f);
        get(tau_dxy_pca_z) = GetValueNorm(tau.dxy_PCA().z(), 0.7973f, 3.456f);
        const bool tau_dxy_valid = std::isnormal(tau.dxy()) && tau.dxy() > - 10 && std::isnormal(tau.dxy_error())
                                   && tau.dxy_error() > 0;
        get(tau_dxy_valid) = tau_dxy_valid;
        get(tau_dxy) = tau_dxy_valid ? GetValueNorm(tau.dxy(), 0.0018f, 0.0085f) : 0.f;
        get(tau_dxy_sig) = tau_dxy_valid
                                             ? GetValueNorm(std::abs(tau.dxy())/tau.dxy_error(), 2.26f, 4.191f) : 0.f;
       const bool tau_ip3d_valid = std::isnormal(tau.ip3d()) && tau.ip3d() > - 10 && std::isnormal(tau.ip3d_error())
                                   && tau.ip3d_error() > 0;
        get(tau_ip3d_valid) = tau_ip3d_valid;
        get(tau_ip3d) = tau_ip3d_valid ? GetValueNorm(tau.ip3d(), 0.0026f, 0.0114f) : 0.f;
        get(tau_ip3d_sig) = tau_ip3d_valid ? GetValueNorm(std::abs(tau.ip3d()) / tau.ip3d_error(),
                                                2.928f, 4.466f) : 0.f ;
        get(tau_dz) = GetValueNorm(leadChargedHadrCand->dz(), 0.f, 0.0190f);
        const bool tau_dz_sig_valid = std::isnormal(leadChargedHadrCand->dz()) && std::isnormal(leadChargedHadrCand->dzError())
                                      && leadChargedHadrCand->dzError() > 0;
        get(tau_dz_sig_valid) = tau_dz_sig_valid;
        get(tau_dz_sig) = GetValueNorm(std::abs(leadChargedHadrCand->dz()) /
                                                           leadChargedHadrCand->dzError(), 4.717f, 11.78f);
        get(tau_flightLength_x) = GetValueNorm(tau.flightLength().x(), -0.0003f, 0.7362f);
        get(tau_flightLength_y) = GetValueNorm(tau.flightLength().y(), -0.0009f, 0.7354f);
        get(tau_flightLength_z) = GetValueNorm(tau.flightLength().z(), -0.0022f, 1.993f);
        get(tau_flightLength_sig) = GetValueNorm(tau.flightLengthSig(), -4.78f, 9.573f);
        get(tau_pt_weighted_deta_strip) = GetValueLinear(reco::tau::pt_weighted_deta_strip(tau,
                                                                             tau.decayMode()), 0, 1, true);
        get(tau_pt_weighted_dphi_strip) = GetValueLinear(reco::tau::pt_weighted_dphi_strip(tau,
                                                                             tau.decayMode()), 0, 1, true);
        get(tau_pt_weighted_dr_signal) = GetValueNorm(reco::tau::pt_weighted_dr_signal(tau,
                                                                          tau.decayMode()), 0.0052f, 0.01433f);
        get(tau_pt_weighted_dr_iso) = GetValueLinear(reco::tau::pt_weighted_dr_iso(tau,
                                                                         tau.decayMode()), 0, 1, true);
        get(tau_leadingTrackNormChi2) = GetValueNorm(tau.leadingTrackNormChi2(), 1.538f, 4.401f);
        const bool tau_e_ratio_valid = std::isnormal(reco::tau::eratio(tau)) && reco::tau::eratio(tau) > 0.f;
        get(tau_e_ratio_valid) = tau_e_ratio_valid;
        get(tau_e_ratio) = tau_e_ratio_valid ? GetValueLinear(reco::tau::eratio(tau), 0, 1, true)
                                                                 : 0.f;
        const double gj_angle_diff = calculateGottfriedJacksonAngleDifference(tau);
        const bool tau_gj_angle_diff_valid = (std::isnormal(gj_angle_diff) || gj_angle_diff == 0) && gj_angle_diff >= 0;
        get(tau_gj_angle_diff_valid) = tau_gj_angle_diff_valid;
        get(tau_gj_angle_diff) = tau_gj_angle_diff_valid ?
                                                     GetValueLinear(calculateGottfriedJacksonAngleDifference(tau),
                                                                    0, pi, true) : 0;;
        get(tau_n_photons) = GetValueNorm(reco::tau::n_photons_total(tau), 2.95f, 3.927f);
        get(tau_emFraction) = GetValueLinear(tau.emFraction_MVA(), -1, 1, false);
        get(tau_inside_ecal_crack) = GetValue(isInEcalCrack(tau.p4().eta()));
        get(leadChargedCand_etaAtEcalEntrance_minus_tau_eta) =
            GetValueNorm(tau.etaAtEcalEntranceLeadChargedCand() - tau.p4().eta(), 0.0042f, 0.0323f);

        return inputs;
    }

    tensorflow::Tensor createEgammaBlockInputs(const TauType& tau, const reco::Vertex& pv, double rho,
                                                 const pat::ElectronCollection& electrons,
                                                 const pat::PackedCandidateCollection& pfCands,
                                                 const CellGrid& grid, bool is_inner)
    {
        return tensorflow::Tensor();
    }

    tensorflow::Tensor createMuonBlockInputs(const TauType& tau, const reco::Vertex& pv, double rho,
                                             const pat::MuonCollection& electrons,
                                             const pat::PackedCandidateCollection& pfCands,
                                             const CellGrid& grid, bool is_inner)
    {
        return tensorflow::Tensor();
    }

    tensorflow::Tensor createHadronsBlockInputs(const TauType& tau, const reco::Vertex& pv, double rho,
                                                const pat::PackedCandidateCollection& pfCands,
                                                const CellGrid& grid, bool is_inner)
    {
        using namespace dnn_inputs_2017_v2;
        using namespace HadronBlockInputs;

        tensorflow::Tensor inputs(tensorflow::DT_FLOAT, {1, grid.nCellsEta, grid.nCellsPhi, NumberOfInputs});
        inputs.flat<float>().setZero();

        for(const auto& cell : grid) {
            int eta_index = grid.getEtaTensorIndex(cell.first);
            int phi_index = grid.getPhiTensorIndex(cell.first);

            const auto& get = [&](int var_index) -> float& {
                return inputs.tensor<float,4>()(0,eta_index,phi_index,var_index);
            };

            const auto& cell_map = cell.second;
            size_t index_chH, index_nH;

            const bool valid_chH = cell_map.count(CellObjectType::PfCand_chargedHadron);
            const bool valid_nH = cell_map.count(CellObjectType::PfCand_neutralHadron);

            if(valid_chH)
                index_chH = cell_map.at(CellObjectType::PfCand_chargedHadron);
            if(cell_map.count(CellObjectType::PfCand_neutralHadron))
                index_nH = cell_map.at(CellObjectType::PfCand_neutralHadron);

            if(valid_chH && valid_nH){
                get(rho) = GetValueNorm(rho, 21.49f, 9.713f);
                get(tau_pt) =  GetValueLinear(tau.polarP4().pt(), 20.f, 1000.f, true);
                get(tau_eta) = GetValueLinear(tau.polarP4().eta(), -2.3f, 2.3f, false);
                get(tau_inside_ecal_crack) = GetValue(isInEcalCrack(tau.p4().eta()));
            }
            if(valid_chH){
                get(pfCand_chHad_valid) = valid_chH;
                get(pfCand_chHad_rel_pt) = GetValueNorm(pfCands.at(index_chH).polarP4().pt() / tau.p4().pt(),
                    is_inner ? 0.2564f : 0.0194f, is_inner ? 0.8607f : 0.1865f);
                get(pfCand_chHad_deta) = GetValueLinear(pfCands.at(index_chH).polarP4().eta() - tau.p4().eta(),
                    is_inner ? -0.1f : -0.5f, is_inner ? 0.1f : 0.5f, false);
                get(pfCand_chHad_dphi) = GetValueLinear(dPhi(tau.p4(),pfCands.at(index_chH).polarP4()),
                    is_inner ? -0.1f : -0.5f, is_inner ? 0.1f : 0.5f, false);
                get(pfCand_chHad_leadChargedHadrCand) = GetValue(&pfCands.at(index_chH) ==
                    dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get()));
                get(pfCand_chHad_pvAssociationQuality) =
                    GetValueLinear(static_cast<float>(pfCands.at(index_chH).pvAssociationQuality()), 0, 7, true);
                get(pfCand_chHad_fromPV) = GetValueLinear(static_cast<float>(pfCands.at(index_chH).fromPV()), 0, 3, true);
                get(pfCand_chHad_puppiWeight) = GetValue(static_cast<float>(pfCands.at(index_chH).puppiWeight()));
                get(pfCand_chHad_puppiWeightNoLep) = GetValue(static_cast<float>(pfCands.at(index_chH).puppiWeightNoLep()));
                get(pfCand_chHad_charge) =  GetValue(static_cast<float>(pfCands.at(index_chH).charge()));
                get(pfCand_chHad_lostInnerHits) = GetValue(static_cast<float>(pfCands.at(index_chH).lostInnerHits()));
                get(pfCand_chHad_numberOfPixelHits) =
                    GetValueLinear(static_cast<float>(pfCands.at(index_chH).numberOfPixelHits()), 0, 12, true);
                get(pfCand_chHad_vertex_dx) =
                    GetValueNorm(static_cast<float>(pfCands.at(index_chH).vertex().x() - pv.position().x()), 0.0005f, 1.735f);
                get(pfCand_chHad_vertex_dy) =
                    GetValueNorm(static_cast<float>(pfCands.at(index_chH).vertex().x() - pv.position().y()), -0.0008f, 1.752f);
                get(pfCand_chHad_vertex_dz) =
                    GetValueNorm(static_cast<float>(pfCands.at(index_chH).vertex().z() - pv.position().z()), -0.0201f, 8.333f);
                get(pfCand_chHad_vertex_dx_tauFL) = GetValueNorm(pfCands.at(index_chH).vertex().x() - pv.position().x() -
                    tau.flightLength().x(), -0.0014f, 1.93f);
                get(pfCand_chHad_vertex_dy_tauFL) = GetValueNorm(pfCands.at(index_chH).vertex().y() - pv.position().y() -
                    tau.flightLength().y(), 0.0022f, 1.948f);
                get(pfCand_chHad_vertex_dz_tauFL) = GetValueNorm(pfCands.at(index_chH).vertex().z() - pv.position().z() -
                    tau.flightLength().z(), -0.0138f, 8.622f);

                const bool hasTrackDetails = pfCands.at(index_chH).hasTrackDetails() == 1;
                if(hasTrackDetails){
                    get(pfCand_chHad_hasTrackDetails) = hasTrackDetails;
                    get(pfCand_chHad_dxy) = GetValueNorm(std::abs(pfCands.at(index_chH).dxy()) /
                        pfCands.at(index_chH).dxyError(), 6.417f, 36.28f);
                    get(pfCand_chHad_dxy_sig) = GetValueNorm(std::abs(pfCands.at(index_chH).dxy()) /
                        pfCands.at(index_chH).dxyError(), 6.417f, 36.28f);
                    get(pfCand_chHad_dz) =  GetValueNorm(std::abs(pfCands.at(index_chH).dz()) /
                        pfCands.at(index_chH).dzError(), 301.3f, 491.1f);
                    get(pfCand_chHad_dz_sig) =  GetValueNorm(std::abs(pfCands.at(index_chH).dz()) /
                        pfCands.at(index_chH).dzError(), 301.3f, 491.1f);
                    get(pfCand_chHad_track_chi2_ndof) = static_cast<float>(pfCands.at(index_chH).pseudoTrack().ndof()) > 0 ?
                        GetValueNorm(static_cast<float>(pfCands.at(index_chH).pseudoTrack().chi2()) /
                        pfCands.at(index_chH).pseudoTrack().ndof(), 0.7876f, 3.694f) : 0;
                    get(pfCand_chHad_track_ndof) = static_cast<float>(pfCands.at(index_chH).pseudoTrack().ndof()) > 0 ?
                        GetValueNorm(static_cast<float>(pfCands.at(index_chH).pseudoTrack().ndof()), 13.92f, 6.581f) : 0;
                }
                get(pfCand_chHad_hcalFraction) = GetValue(pfCands.at(index_chH).hcalFraction());
                get(pfCand_chHad_rawCaloFraction) = GetValueLinear(pfCands.at(index_chH).rawCaloFraction(), 0.f, 2.6f, true);
            }
            if(valid_nH){
                get(pfCand_nHad_valid) = valid_nH;
                get(pfCand_nHad_rel_pt) = GetValueNorm(pfCands.at(index_nH).polarP4().pt() / tau.polarP4().pt(),
                    is_inner ? 0.3163f : 0.0502f, is_inner ? 0.2769f : 0.4266f);
                get(pfCand_nHad_deta) = GetValueLinear(pfCands.at(index_nH).polarP4().eta() - tau.polarP4().eta(),
                    is_inner ? -0.1f : -0.5f, is_inner ? 0.1f : 0.5f, false);
                get(pfCand_nHad_dphi) = GetValueLinear(dPhi(tau.polarP4(),pfCands.at(index_nH).polarP4()),
                    is_inner ? -0.1f : -0.5f, is_inner ? 0.1f : 0.5f, false);
                get(pfCand_nHad_puppiWeight) = GetValue(pfCands.at(index_nH).puppiWeight());
                get(pfCand_nHad_puppiWeightNoLep) = GetValue(pfCands.at(index_nH).puppiWeightNoLep());
                get(pfCand_nHad_hcalFraction) = GetValue(pfCands.at(index_nH).hcalFraction());
            }
        }
        return inputs;
    }

    template<typename dnn>
    tensorflow::Tensor createInputsV1(const TauType& tau, const ElectronCollection& electrons,
                                      const MuonCollection& muons) const
    {
        static constexpr bool check_all_set = false;
        static constexpr float default_value_for_set_check = -42;
        static const TauIdMVAAuxiliaries clusterVariables;

        tensorflow::Tensor inputs(tensorflow::DT_FLOAT, { 1, dnn_inputs_2017v1::NumberOfInputs});
        const auto& get = [&](int var_index) -> float& { return inputs.matrix<float>()(0, var_index); };
        auto leadChargedHadrCand = dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get());

        if(check_all_set) {
            for(int var_index = 0; var_index < dnn::NumberOfInputs; ++var_index) {
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
        get(dnn::leadChargedHadrCand_dEta) = leadChargedHadrCand
                ? dEta(leadChargedHadrCand->p4(), tau.p4()) : default_value;
        get(dnn::leadChargedHadrCand_dPhi) = leadChargedHadrCand
                ? dPhi(leadChargedHadrCand->p4(), tau.p4()) : default_value;
        get(dnn::leadChargedHadrCand_mass) = leadChargedHadrCand
                ? leadChargedHadrCand->p4().mass() : default_value;
        get(dnn::pt_weighted_deta_strip) = clusterVariables.tau_pt_weighted_deta_strip(tau, tau.decayMode());
        get(dnn::pt_weighted_dphi_strip) = clusterVariables.tau_pt_weighted_dphi_strip(tau, tau.decayMode());
        get(dnn::pt_weighted_dr_signal) = clusterVariables.tau_pt_weighted_dr_signal(tau, tau.decayMode());
        get(dnn::pt_weighted_dr_iso) = clusterVariables.tau_pt_weighted_dr_iso(tau, tau.decayMode());
        get(dnn::leadingTrackNormChi2) = tau.leadingTrackNormChi2();
        get(dnn::e_ratio) = clusterVariables.tau_Eratio(tau);
        get(dnn::gj_angle_diff) = calculateGottfriedJacksonAngleDifference(tau);
        get(dnn::n_photons) = clusterVariables.tau_n_photons_total(tau);
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
        get(dnn::gsf_ele_dEta_SeedClusterTrackAtCalo) = gsf_ele != nullptr
                ? gsf_ele->deltaEtaSeedClusterTrackAtCalo() : default_value;
        get(dnn::gsf_ele_dPhi_SeedClusterTrackAtCalo) = gsf_ele != nullptr
                ? gsf_ele->deltaPhiSeedClusterTrackAtCalo() : default_value;
        get(dnn::gsf_ele_mvaIn_sigmaEtaEta) = gsf_ele != nullptr
                ? gsf_ele->mvaInput().sigmaEtaEta : default_value;
        get(dnn::gsf_ele_mvaIn_hadEnergy) = gsf_ele != nullptr ? gsf_ele->mvaInput().hadEnergy : default_value;
        get(dnn::gsf_ele_mvaIn_deltaEta) = gsf_ele != nullptr ? gsf_ele->mvaInput().deltaEta : default_value;

        get(dnn::gsf_ele_Chi2NormGSF) = default_value;
        get(dnn::gsf_ele_GSFNumHits) = default_value;
        get(dnn::gsf_ele_GSFTrackResol) = default_value;
        get(dnn::gsf_ele_GSFTracklnPt) = default_value;
        if(gsf_ele != nullptr && gsf_ele->gsfTrack().isNonnull()) {
            get(dnn::gsf_ele_Chi2NormGSF) = gsf_ele->gsfTrack()->normalizedChi2();
            get(dnn::gsf_ele_GSFNumHits) = gsf_ele->gsfTrack()->numberOfValidHits();
            if(gsf_ele->gsfTrack()->pt() > 0) {
                get(dnn::gsf_ele_GSFTrackResol) = gsf_ele->gsfTrack()->ptError() / gsf_ele->gsfTrack()->pt();
                get(dnn::gsf_ele_GSFTracklnPt) = std::log10(gsf_ele->gsfTrack()->pt());
            }
        }

        get(dnn::gsf_ele_Chi2NormKF) = default_value;
        get(dnn::gsf_ele_KFNumHits) = default_value;
        if(gsf_ele != nullptr && gsf_ele->closestCtfTrackRef().isNonnull()) {
            get(dnn::gsf_ele_Chi2NormKF) = gsf_ele->closestCtfTrackRef()->normalizedChi2();
            get(dnn::gsf_ele_KFNumHits) = gsf_ele->closestCtfTrackRef()->numberOfValidHits();
        }
        get(dnn::leadChargedCand_etaAtEcalEntrance) = tau.etaAtEcalEntranceLeadChargedCand();
        get(dnn::leadChargedCand_pt) = tau.ptLeadChargedCand();

        get(dnn::leadChargedHadrCand_HoP) = default_value;
        get(dnn::leadChargedHadrCand_EoP) = default_value;
        if(tau.leadChargedHadrCand()->pt() > 0) {
            get(dnn::leadChargedHadrCand_HoP) = tau.hcalEnergyLeadChargedHadrCand()
                                                    / tau.leadChargedHadrCand()->pt();
            get(dnn::leadChargedHadrCand_EoP) = tau.ecalEnergyLeadChargedHadrCand()
                                                    / tau.leadChargedHadrCand()->pt();
        }

        MuonHitMatchV1 muon_hit_match;
        if(tau.leadPFChargedHadrCand().isNonnull() && tau.leadPFChargedHadrCand()->muonRef().isNonnull())
            muon_hit_match.addMatchedMuon(*tau.leadPFChargedHadrCand()->muonRef(), tau);

        auto matched_muons = muon_hit_match.findMatchedMuons(tau, muons, 0.3, 5);
        for(auto muon : matched_muons)
            muon_hit_match.addMatchedMuon(*muon, tau);
        muon_hit_match.fillTensor<dnn>(get, tau, default_value);

        LorentzVectorXYZ signalChargedHadrCands_sumIn, signalChargedHadrCands_sumOut;
        processSignalPFComponents(tau, tau.signalChargedHadrCands(),
                                  signalChargedHadrCands_sumIn, signalChargedHadrCands_sumOut,
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
        processSignalPFComponents(tau, tau.signalNeutrHadrCands(),
                                  signalNeutrHadrCands_sumIn, signalNeutrHadrCands_sumOut,
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
        processSignalPFComponents(tau, tau.signalGammaCands(),
                                  signalGammaCands_sumIn, signalGammaCands_sumOut,
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
        processIsolationPFComponents(tau, tau.isolationChargedHadrCands(), isolationChargedHadrCands_sum,
                                     get(dnn::isolationChargedHadrCands_sum_pt),
                                     get(dnn::isolationChargedHadrCands_sum_dEta),
                                     get(dnn::isolationChargedHadrCands_sum_dPhi),
                                     get(dnn::isolationChargedHadrCands_sum_mass),
                                     get(dnn::isolationChargedHadrCands_nTotal));

        LorentzVectorXYZ isolationNeutrHadrCands_sum;
        processIsolationPFComponents(tau, tau.isolationNeutrHadrCands(), isolationNeutrHadrCands_sum,
                                     get(dnn::isolationNeutrHadrCands_sum_pt),
                                     get(dnn::isolationNeutrHadrCands_sum_dEta),
                                     get(dnn::isolationNeutrHadrCands_sum_dPhi),
                                     get(dnn::isolationNeutrHadrCands_sum_mass),
                                     get(dnn::isolationNeutrHadrCands_nTotal));

        LorentzVectorXYZ isolationGammaCands_sum;
        processIsolationPFComponents(tau, tau.isolationGammaCands(), isolationGammaCands_sum,
                                     get(dnn::isolationGammaCands_sum_pt),
                                     get(dnn::isolationGammaCands_sum_dEta),
                                     get(dnn::isolationGammaCands_sum_dPhi),
                                     get(dnn::isolationGammaCands_sum_mass),
                                     get(dnn::isolationGammaCands_nTotal));

        get(dnn::tau_visMass_innerSigCone) = (signalGammaCands_sumIn + signalChargedHadrCands_sumIn).mass();

        if(check_all_set) {
            for(int var_index = 0; var_index < dnn::NumberOfInputs; ++var_index) {
                if(get(var_index) == default_value_for_set_check)
                    throw cms::Exception("DeepTauId: variable with index = ") << var_index << " is not set.";
            }
        }

        return inputs;
    }

    static void calculateElectronClusterVars(const pat::Electron* ele, float& elecEe, float& elecEgamma)
    {
        if(ele) {
            elecEe = elecEgamma = 0;
            auto superCluster = ele->superCluster();
            if(superCluster.isNonnull() && superCluster.isAvailable() && superCluster->clusters().isNonnull()
                    && superCluster->clusters().isAvailable()) {
                for(auto iter = superCluster->clustersBegin(); iter != superCluster->clustersEnd(); ++iter) {
                    const double energy = (*iter)->energy();
                    if(iter == superCluster->clustersBegin()) elecEe += energy;
                    else elecEgamma += energy;
                }
            }
        } else {
            elecEe = elecEgamma = default_value;
        }
    }

    template<typename CandidateCollection>
    static void processSignalPFComponents(const pat::Tau& tau, const CandidateCollection& candidates,
                                          LorentzVectorXYZ& p4_inner, LorentzVectorXYZ& p4_outer,
                                          float& pt_inner, float& dEta_inner, float& dPhi_inner, float& m_inner,
                                          float& pt_outer, float& dEta_outer, float& dPhi_outer, float& m_outer,
                                          float& n_inner, float& n_outer)
    {
        p4_inner = LorentzVectorXYZ(0, 0, 0, 0);
        p4_outer = LorentzVectorXYZ(0, 0, 0, 0);
        n_inner = 0;
        n_outer = 0;

        const double innerSigCone_radius = getInnerSignalConeRadius(tau.pt());
        for(const auto& cand : candidates) {
            const double dR = reco::deltaR(cand->p4(), tau.leadChargedHadrCand()->p4());
            const bool isInside_innerSigCone = dR < innerSigCone_radius;
            if(isInside_innerSigCone) {
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

    template<typename CandidateCollection>
    static void processIsolationPFComponents(const pat::Tau& tau, const CandidateCollection& candidates,
                                             LorentzVectorXYZ& p4, float& pt, float& d_eta, float& d_phi, float& m,
                                             float& n)
    {
        p4 = LorentzVectorXYZ(0, 0, 0, 0);
        n = 0;

        for(const auto& cand : candidates) {
            p4 += cand->p4();
            ++n;
        }

        pt = n != 0 ? p4.Pt() : default_value;
        d_eta = n != 0 ? dEta(p4, tau.p4()) : default_value;
        d_phi = n != 0 ? dPhi(p4, tau.p4()) : default_value;
        m = n != 0 ? p4.mass() : default_value;
    }

    static double getInnerSignalConeRadius(double pt)
    {
        static constexpr double min_pt = 30., min_radius = 0.05, cone_opening_coef = 3.;
        // This is equivalent of the original formula (std::max(std::min(0.1, 3.0/pt), 0.05)
        return std::max(cone_opening_coef / std::max(pt, min_pt), min_radius);
    }

    // Copied from https://github.com/cms-sw/cmssw/blob/CMSSW_9_4_X/RecoTauTag/RecoTau/plugins/PATTauDiscriminationByMVAIsolationRun2.cc#L218
    static float calculateGottfriedJacksonAngleDifference(const pat::Tau& tau)
    {
        if(tau.decayMode() == 10) {
            static constexpr double mTau = 1.77682;
            const double mAOne = tau.p4().M();
            const double pAOneMag = tau.p();
            const double argumentThetaGJmax = (std::pow(mTau,2) - std::pow(mAOne,2) ) / ( 2 * mTau * pAOneMag );
            const double argumentThetaGJmeasured = tau.p4().Vect().Dot(tau.flightLength())
                    / ( pAOneMag * tau.flightLength().R() );
            if ( std::abs(argumentThetaGJmax) <= 1. && std::abs(argumentThetaGJmeasured) <= 1. ) {
                double thetaGJmax = std::asin( argumentThetaGJmax );
                double thetaGJmeasured = std::acos( argumentThetaGJmeasured );
                return thetaGJmeasured - thetaGJmax;
            }
        }
        return default_value;
    }

    static bool isInEcalCrack(double eta)
    {
        const double abs_eta = std::abs(eta);
        return abs_eta > 1.46 && abs_eta < 1.558;
    }

    static const pat::Electron* findMatchedElectron(const pat::Tau& tau, const pat::ElectronCollection& electrons,
                                                    double deltaR)
    {
        const double dR2 = deltaR*deltaR;
        const pat::Electron* matched_ele = nullptr;
        for(const auto& ele : electrons) {
            if(reco::deltaR2(tau.p4(), ele.p4()) < dR2 && (!matched_ele || matched_ele->pt() < ele.pt())) {
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
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeepTauId);
