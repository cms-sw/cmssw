/*
 * \class DeepTauId
 *
 * Tau identification using Deep NN
 *
 * \author Konstantin Androsov, INFN Pisa
 */

#include <Math/VectorUtil.h>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/PATTauDiscriminator.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauClusterVariables.h"
#include "DataFormats/Math/interface/deltaR.h"

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

    void AddMatchedMuon(const pat::Muon& muon, const pat::Tau& tau)
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
            for(int hit_index = 0; hit_index < hit_pattern.numberOfAllHits(reco::HitPattern::TRACK_HITS);
                ++hit_index) {
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

    static std::vector<const pat::Muon*> FindMatchedMuons(const pat::Tau& tau, const pat::MuonCollection& muons,
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
    void FillTensor(const TensorElemGet& get, const pat::Tau& tau, float default_value) const
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
        get(dnn::muon_n_stations_with_matches_03) = CountMuonStationsWithMatches(0, 3);
        get(dnn::muon_n_stations_with_hits_23) = CountMuonStationsWithHits(2, 3);
    }

private:
    unsigned CountMuonStationsWithMatches(size_t first_station, size_t last_station) const
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

    unsigned CountMuonStationsWithHits(size_t first_station, size_t last_station) const
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


class DeepTauId : public edm::stream::EDProducer<> {
public:
    using TauType = pat::Tau;
    using TauDiscriminator = pat::PATTauDiscriminator;
    using TauCollection = std::vector<TauType>;
    using TauRef = edm::Ref<TauCollection>;
    using TauRefProd = edm::RefProd<TauCollection>;
    using ElectronCollection = pat::ElectronCollection;
    using MuonCollection = pat::MuonCollection;
    using LorentzVectorXYZ = ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double>>;
    using GraphPtr = std::shared_ptr<tensorflow::GraphDef>;

    static constexpr float default_value = -999.;

    struct Output {
        std::vector<size_t> num, den;

        Output(const std::vector<size_t>& _num, const std::vector<size_t>& _den) : num(_num), den(_den) {}

        std::unique_ptr<TauDiscriminator> get_value(const edm::Handle<TauCollection>& taus,
                                                    const tensorflow::Tensor& pred) const
        {
            auto output = std::make_unique<TauDiscriminator>(TauRefProd(taus));
            for(size_t tau_index = 0; tau_index < taus->size(); ++tau_index) {
                float x = 0;
                for(size_t num_elem : num)
                    x += pred.matrix<float>()(tau_index, num_elem);
                if(x != 0) {
                    float den_val = 0;
                    for(size_t den_elem : den)
                        den_val += pred.matrix<float>()(tau_index, den_elem);
                    x = den_val != 0 ? x / den_val : std::numeric_limits<float>::max();
                }
                output->setValue(tau_index, x);
            }
            return output;
        }
    };

    using OutputCollection = std::map<std::string, Output>;

    static const OutputCollection& GetOutputs()
    {
        static constexpr size_t e_index = 0, mu_index = 1, tau_index = 2, jet_index = 3;
        static const OutputCollection outputs = {
            { "tauVSe", Output({tau_index}, {e_index, tau_index}) },
            { "tauVSmu", Output({tau_index}, {mu_index, tau_index}) },
            { "tauVSjet", Output({tau_index}, {jet_index, tau_index}) },
            { "tauVSall", Output({tau_index}, {e_index, mu_index, jet_index, tau_index}) }
        };
        return outputs;
    }

public:
    explicit DeepTauId(const edm::ParameterSet& cfg) :
        electrons_token(consumes<ElectronCollection>(cfg.getParameter<edm::InputTag>("electrons"))),
        muons_token(consumes<MuonCollection>(cfg.getParameter<edm::InputTag>("muons"))),
        taus_token(consumes<TauCollection>(cfg.getParameter<edm::InputTag>("taus"))),
        graph(tensorflow::loadGraphDef(edm::FileInPath(cfg.getParameter<std::string>("graph_file")).fullPath())),
        session(tensorflow::createSession(graph.get())),
        input_layer(graph->node(0).name()),
        output_layer(graph->node(graph->node_size() - 1).name())
    {
        for(const auto& output_desc : GetOutputs())
            produces<TauDiscriminator>(output_desc.first);
    }

    virtual ~DeepTauId() override
    {
        tensorflow::closeSession(session);
    }

    virtual void produce(edm::Event& event, const edm::EventSetup& es) override
    {
        edm::Handle<pat::ElectronCollection> electrons;
        event.getByToken(electrons_token, electrons);

        edm::Handle<pat::MuonCollection> muons;
        event.getByToken(muons_token, muons);

        edm::Handle<TauCollection> taus;
        event.getByToken(taus_token, taus);

        const tensorflow::Tensor& inputs = CreateInputs<dnn_inputs_2017v1>(*taus, *electrons, *muons);
        std::vector<tensorflow::Tensor> pred_vector;
        tensorflow::run(session, { { input_layer, inputs } }, { output_layer }, &pred_vector);
        const tensorflow::Tensor& pred = pred_vector.at(0);

        for(const auto& output_desc : GetOutputs())
            event.put(output_desc.second.get_value(taus, pred), output_desc.first);
    }

private:
    template<typename dnn_inputs>
    tensorflow::Tensor CreateInputs(const TauCollection& taus, const ElectronCollection& electrons,
                                    const MuonCollection& muons) const
    {
        tensorflow::Tensor inputs(tensorflow::DT_FLOAT, { static_cast<int>(taus.size()), dnn_inputs::NumberOfInputs});
        for(size_t tau_index = 0; tau_index < taus.size(); ++tau_index)
            SetInputs<dnn_inputs>(taus, tau_index, inputs, electrons, muons);
        return inputs;
    }

    template<typename dnn>
    void SetInputs(const TauCollection& taus, size_t tau_index, tensorflow::Tensor& inputs,
                   const ElectronCollection& electrons, const MuonCollection& muons) const
    {

        static constexpr bool check_all_set = false;
        static constexpr float magic_number = -42;
        const auto& get = [&](int var_index) -> float& { return inputs.matrix<float>()(tau_index, var_index); };
        const TauType& tau = taus.at(tau_index);
        auto leadChargedHadrCand = dynamic_cast<const pat::PackedCandidate*>(tau.leadChargedHadrCand().get());

        if(check_all_set) {
            for(int var_index = 0; var_index < dnn::NumberOfInputs; ++var_index) {
                get(var_index) = magic_number;
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
        get(dnn::gj_angle_diff) = CalculateGottfriedJacksonAngleDifference(tau);
        get(dnn::n_photons) = clusterVariables.tau_n_photons_total(tau);
        get(dnn::emFraction) = tau.emFraction_MVA();
        get(dnn::has_gsf_track) = leadChargedHadrCand && std::abs(leadChargedHadrCand->pdgId()) == 11;
        get(dnn::inside_ecal_crack) = IsInEcalCrack(tau.p4().Eta());
        auto gsf_ele = FindMatchedElectron(tau, electrons, 0.3);
        get(dnn::gsf_ele_matched) = gsf_ele != nullptr;
        get(dnn::gsf_ele_pt) = gsf_ele != nullptr ? gsf_ele->p4().Pt() : default_value;
        get(dnn::gsf_ele_dEta) = gsf_ele != nullptr ? dEta(gsf_ele->p4(), tau.p4()) : default_value;
        get(dnn::gsf_ele_dPhi) = gsf_ele != nullptr ? dPhi(gsf_ele->p4(), tau.p4()) : default_value;
        get(dnn::gsf_ele_mass) = gsf_ele != nullptr ? gsf_ele->p4().mass() : default_value;
        CalculateElectronClusterVars(gsf_ele, get(dnn::gsf_ele_Ee), get(dnn::gsf_ele_Egamma));
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
            muon_hit_match.AddMatchedMuon(*tau.leadPFChargedHadrCand()->muonRef(), tau);

        auto matched_muons = muon_hit_match.FindMatchedMuons(tau, muons, 0.3, 5);
        for(auto muon : matched_muons)
            muon_hit_match.AddMatchedMuon(*muon, tau);
        muon_hit_match.FillTensor<dnn>(get, tau, default_value);

        LorentzVectorXYZ signalChargedHadrCands_sumIn, signalChargedHadrCands_sumOut;
        ProcessSignalPFComponents(tau, tau.signalChargedHadrCands(),
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
        ProcessSignalPFComponents(tau, tau.signalNeutrHadrCands(),
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
        ProcessSignalPFComponents(tau, tau.signalGammaCands(),
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
        ProcessIsolationPFComponents(tau, tau.isolationChargedHadrCands(), isolationChargedHadrCands_sum,
                                     get(dnn::isolationChargedHadrCands_sum_pt),
                                     get(dnn::isolationChargedHadrCands_sum_dEta),
                                     get(dnn::isolationChargedHadrCands_sum_dPhi),
                                     get(dnn::isolationChargedHadrCands_sum_mass),
                                     get(dnn::isolationChargedHadrCands_nTotal));

        LorentzVectorXYZ isolationNeutrHadrCands_sum;
        ProcessIsolationPFComponents(tau, tau.isolationNeutrHadrCands(), isolationNeutrHadrCands_sum,
                                     get(dnn::isolationNeutrHadrCands_sum_pt),
                                     get(dnn::isolationNeutrHadrCands_sum_dEta),
                                     get(dnn::isolationNeutrHadrCands_sum_dPhi),
                                     get(dnn::isolationNeutrHadrCands_sum_mass),
                                     get(dnn::isolationNeutrHadrCands_nTotal));

        LorentzVectorXYZ isolationGammaCands_sum;
        ProcessIsolationPFComponents(tau, tau.isolationGammaCands(), isolationGammaCands_sum,
                                     get(dnn::isolationGammaCands_sum_pt),
                                     get(dnn::isolationGammaCands_sum_dEta),
                                     get(dnn::isolationGammaCands_sum_dPhi),
                                     get(dnn::isolationGammaCands_sum_mass),
                                     get(dnn::isolationGammaCands_nTotal));

        get(dnn::tau_visMass_innerSigCone) = (signalGammaCands_sumIn + signalChargedHadrCands_sumIn).mass();

        if(check_all_set) {
            for(int var_index = 0; var_index < dnn::NumberOfInputs; ++var_index) {
                if(get(var_index) == magic_number)
                    throw cms::Exception("DeepTauId: variable with index = ") << var_index << " is not set.";
            }
        }
    }

    static void CalculateElectronClusterVars(const pat::Electron* ele, float& elecEe, float& elecEgamma)
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
    static void ProcessSignalPFComponents(const pat::Tau& tau, const CandidateCollection& candidates,
                                          LorentzVectorXYZ& p4_inner, LorentzVectorXYZ& p4_outer,
                                          float& pt_inner, float& dEta_inner, float& dPhi_inner, float& m_inner,
                                          float& pt_outer, float& dEta_outer, float& dPhi_outer, float& m_outer,
                                          float& n_inner, float& n_outer)
    {
        p4_inner = LorentzVectorXYZ(0, 0, 0, 0);
        p4_outer = LorentzVectorXYZ(0, 0, 0, 0);
        n_inner = 0;
        n_outer = 0;

        const double innerSigCone_radius = GetInnerSignalConeRadius(tau.pt());
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
    static void ProcessIsolationPFComponents(const pat::Tau& tau, const CandidateCollection& candidates,
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

    static double GetInnerSignalConeRadius(double pt)
    {
        return std::max(.05, std::min(.1, 3./std::max(1., pt)));
    }

    // Copied from https://github.com/cms-sw/cmssw/blob/CMSSW_9_4_X/RecoTauTag/RecoTau/plugins/PATTauDiscriminationByMVAIsolationRun2.cc#L218
    static float CalculateGottfriedJacksonAngleDifference(const pat::Tau& tau)
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

    static bool IsInEcalCrack(double eta)
    {
        const double abs_eta = std::abs(eta);
        return abs_eta > 1.46 && abs_eta < 1.558;
    }

    static const pat::Electron* FindMatchedElectron(const pat::Tau& tau, const pat::ElectronCollection& electrons,
                                                    double deltaR)
    {
        const double dR2 = deltaR*deltaR;
        const pat::Electron* matched_ele = nullptr;
        for(const auto& ele : electrons) {
	  if(reco::deltaR2(tau.p4(), ele.p4()) < dR2 &&
	       (!matched_ele || matched_ele->pt() < ele.pt())) {
	      matched_ele = &ele;
            }
        }
        return matched_ele;
    }

private:
    edm::EDGetTokenT<ElectronCollection> electrons_token;
    edm::EDGetTokenT<MuonCollection> muons_token;
    edm::EDGetTokenT<TauCollection> taus_token;
    GraphPtr graph;
    tensorflow::Session* session;
    std::string input_layer, output_layer;
    TauIdMVAAuxiliaries clusterVariables;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeepTauId);
