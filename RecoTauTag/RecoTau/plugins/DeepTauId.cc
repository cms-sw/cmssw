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

struct MuonHitMatch {
    static constexpr int n_muon_stations = 4;

    std::map<int, std::vector<UInt_t>> n_matches, n_hits;
    unsigned n_muons{0};
    const pat::Muon* best_matched_muon{nullptr};
    double deltaR2_best_match{-1};

    MuonHitMatch()
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
        desc.add<std::string>("graph_file", "RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v1_20L1024N_quantized.pb");
        desc.add<bool>("mem_mapped", false);

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
        descriptions.add("DeepTau2017v1", desc);
    }

public:
    explicit DeepTauId(const edm::ParameterSet& cfg, const deep_tau::DeepTauCache* cache) :
        DeepTauBase(cfg, GetOutputs(), cache),
        electrons_token(consumes<ElectronCollection>(cfg.getParameter<edm::InputTag>("electrons"))),
        muons_token(consumes<MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))),
        input_layer(cache_->getGraph().node(0).name()),
        output_layer(cache_->getGraph().node(cache_->getGraph().node_size() - 1).name())
    {
        const auto& shape = cache_->getGraph().node(0).attr().at("shape").shape();
        if(shape.dim(1).size() != dnn_inputs_2017v1::NumberOfInputs)
            throw cms::Exception("DeepTauId") << "number of inputs does not match the expected inputs for the given version";

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
    virtual tensorflow::Tensor getPredictions(edm::Event& event, const edm::EventSetup& es,
                                              edm::Handle<TauCollection> taus) override
    {
        edm::Handle<pat::ElectronCollection> electrons;
        event.getByToken(electrons_token, electrons);

        edm::Handle<pat::MuonCollection> muons;
        event.getByToken(muons_token, muons);

        tensorflow::Tensor predictions(tensorflow::DT_FLOAT, { static_cast<int>(taus->size()),
                                       dnn_inputs_2017v1::NumberOfOutputs});
        for(size_t tau_index = 0; tau_index < taus->size(); ++tau_index) {
            const tensorflow::Tensor& inputs = createInputs<dnn_inputs_2017v1>(taus->at(tau_index), *electrons, *muons);
            std::vector<tensorflow::Tensor> pred_vector;
            tensorflow::run(&(cache_->getSession()), { { input_layer, inputs } }, { output_layer }, &pred_vector);
            for(int k = 0; k < dnn_inputs_2017v1::NumberOfOutputs; ++k)
                predictions.matrix<float>()(tau_index, k) = pred_vector[0].flat<float>()(k);
        }
        return predictions;
    }

    template<typename dnn>
    tensorflow::Tensor createInputs(const TauType& tau, const ElectronCollection& electrons,
                                    const MuonCollection& muons) const
    {
        static constexpr bool check_all_set = false;
        static constexpr float default_value_for_set_check = -42;

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

        MuonHitMatch muon_hit_match;
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
    edm::EDGetTokenT<ElectronCollection> electrons_token;
    edm::EDGetTokenT<MuonCollection> muons_token;
    std::string input_layer, output_layer;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeepTauId);
