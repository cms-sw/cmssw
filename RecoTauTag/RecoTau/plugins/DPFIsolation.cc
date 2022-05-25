/*
 * \class DPFIsolation
 *
 * Deep ParticleFlow tau isolation using Deep NN.
 *
 * \author Owen Colegrove, UCSB
 */

#include "RecoTauTag/RecoTau/interface/DeepTauBase.h"

namespace {
  inline int getPFCandidateIndex(const edm::Handle<edm::View<reco::Candidate>>& pfcands,
                                 const reco::CandidatePtr& cptr) {
    for (unsigned int i = 0; i < pfcands->size(); ++i) {
      if (reco::CandidatePtr(pfcands, i) == cptr)
        return i;
    }
    return -1;
  }
}  // anonymous namespace

class DPFIsolation : public deep_tau::DeepTauBase {
public:
  static const OutputCollection& GetOutputs() {
    const size_t tau_index = 0;
    static const OutputCollection outputs_ = {{"VSall", Output({tau_index}, {})}};
    return outputs_;
  };

  static unsigned getNumberOfParticles(unsigned graphVersion) {
    static const std::map<unsigned, unsigned> nparticles{{0, 60}, {1, 36}};
    return nparticles.at(graphVersion);
  }

  static unsigned GetNumberOfFeatures(unsigned graphVersion) {
    static const std::map<unsigned, unsigned> nfeatures{{0, 47}, {1, 51}};
    return nfeatures.at(graphVersion);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("pfcands", edm::InputTag("packedPFCandidates"));
    desc.add<edm::InputTag>("taus", edm::InputTag("slimmedTaus"));
    desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
    desc.add<std::vector<std::string>>("graph_file",
                                       {"RecoTauTag/TrainingFiles/data/DPFTauId/DPFIsolation_2017v0_quantized.pb"});
    desc.add<unsigned>("version", 0);
    desc.add<bool>("mem_mapped", false);
    desc.add<bool>("is_online", false);

    //pre-discriminants
    edm::ParameterSetDescription pset_Prediscriminants;
    pset_Prediscriminants.add<std::string>("BooleanOperator", "and");
    edm::ParameterSetDescription psd1;
    psd1.add<double>("cut");
    psd1.add<edm::InputTag>("Producer");
    pset_Prediscriminants.addOptional<edm::ParameterSetDescription>("decayMode", psd1);
    desc.add<edm::ParameterSetDescription>("Prediscriminants", pset_Prediscriminants);

    desc.add<std::vector<std::string>>("VSallWP", {"0"});
    descriptions.add("DPFTau2016v0", desc);
  }

  explicit DPFIsolation(const edm::ParameterSet& cfg, const deep_tau::DeepTauCache* cache)
      : DeepTauBase(cfg, GetOutputs(), cache), graphVersion(cfg.getParameter<unsigned>("version")) {
    const auto& shape = cache_->getGraph().node(0).attr().at("shape").shape();

    if (!(graphVersion == 1 || graphVersion == 0))
      throw cms::Exception("DPFIsolation") << "unknown version of the graph file.";

    if (!(shape.dim(1).size() == getNumberOfParticles(graphVersion) &&
          shape.dim(2).size() == GetNumberOfFeatures(graphVersion)))
      throw cms::Exception("DPFIsolation")
          << "number of inputs does not match the expected inputs for the given version";

    if (is_online_) {
      throw cms::Exception("DPFIsolation") << "Online version based on reco objects in not implemented. Use offline "
                                              "version on top of miniAOD with pat objects.";
    }
  }

private:
  tensorflow::Tensor getPredictions(edm::Event& event, edm::Handle<TauCollection> taus) override {
    edm::Handle<edm::View<reco::Candidate>> pfcands;
    event.getByToken(pfcandToken_, pfcands);

    edm::Handle<reco::VertexCollection> vertices;
    event.getByToken(vtxToken_, vertices);

    tensorflow::Tensor tensor(
        tensorflow::DT_FLOAT,
        {1, static_cast<int>(getNumberOfParticles(graphVersion)), static_cast<int>(GetNumberOfFeatures(graphVersion))});

    tensorflow::Tensor predictions(tensorflow::DT_FLOAT, {static_cast<int>(taus->size()), 1});

    std::vector<tensorflow::Tensor> outputs_;

    float pfCandPt, pfCandPz, pfCandPtRel, pfCandPzRel, pfCandDr, pfCandDEta, pfCandDPhi, pfCandEta, pfCandDz,
        pfCandDzErr, pfCandD0, pfCandD0D0, pfCandD0Dz, pfCandD0Dphi, pfCandPuppiWeight, pfCandPixHits, pfCandHits,
        pfCandLostInnerHits, pfCandPdgID, pfCandCharge, pfCandFromPV, pfCandVtxQuality, pfCandHighPurityTrk,
        pfCandTauIndMatch, pfCandDzSig, pfCandD0Sig, pfCandD0Err, pfCandPtRelPtRel, pfCandDzDz, pfCandDVx_1,
        pfCandDVy_1, pfCandDVz_1, pfCandD_1;
    float pvx = !vertices->empty() ? (*vertices)[0].x() : -1;
    float pvy = !vertices->empty() ? (*vertices)[0].y() : -1;
    float pvz = !vertices->empty() ? (*vertices)[0].z() : -1;

    bool pfCandIsBarrel;

    // These variables define ranges further used for standardization
    static constexpr float pfCandPt_max = 500.f;
    static constexpr float pfCandPz_max = 1000.f;
    static constexpr float pfCandPtRel_max = 1.f;
    static constexpr float pfCandPzRel_max = 100.f;
    static constexpr float pfCandPtRelPtRel_max = 1.f;
    static constexpr float pfCandD0_max = 5.f;
    static constexpr float pfCandDz_max = 5.f;
    static constexpr float pfCandDVx_y_z_1_max = 0.05f;
    static constexpr float pfCandD_1_max = 0.1f;
    static constexpr float pfCandD0_z_Err_max = 1.f;
    static constexpr float pfCandDzSig_max = 3.f;
    static constexpr float pfCandD0Sig_max = 1.f;
    static constexpr float pfCandDr_max = 0.5f;
    static constexpr float pfCandEta_max = 2.75f;
    static constexpr float pfCandDEta_max = 0.5f;
    static constexpr float pfCandDPhi_max = 0.5f;
    static constexpr float pfCandPixHits_max = 7.f;
    static constexpr float pfCandHits_max = 30.f;

    for (size_t tau_index = 0; tau_index < taus->size(); tau_index++) {
      pat::Tau tau = taus->at(tau_index);
      bool isGoodTau = false;
      const float lepRecoPt = tau.pt();
      const float lepRecoPz = std::abs(tau.pz());
      const float lepRecoEta = tau.eta();
      const float lepRecoPhi = tau.phi();

      if (lepRecoPt >= 30 && std::abs(lepRecoEta) < 2.3 && tau.isTauIDAvailable("againstMuonLoose3") &&
          tau.isTauIDAvailable("againstElectronVLooseMVA6")) {
        isGoodTau = (tau.tauID("againstElectronVLooseMVA6") && tau.tauID("againstMuonLoose3"));
      }

      if (!isGoodTau) {
        predictions.matrix<float>()(tau_index, 0) = -1;
        continue;
      }

      std::vector<unsigned int> signalCandidateInds;

      for (const auto& c : tau.signalCands())
        signalCandidateInds.push_back(getPFCandidateIndex(pfcands, c));

      // Use of setZero results in warnings in eigen library during compilation.
      //tensor.flat<float>().setZero();
      const unsigned n_inputs = getNumberOfParticles(graphVersion) * GetNumberOfFeatures(graphVersion);
      for (unsigned input_idx = 0; input_idx < n_inputs; ++input_idx)
        tensor.flat<float>()(input_idx) = 0;

      unsigned int iPF = 0;
      const unsigned max_iPF = getNumberOfParticles(graphVersion);

      std::vector<unsigned int> sorted_inds(pfcands->size());
      std::size_t n = 0;
      std::generate(std::begin(sorted_inds), std::end(sorted_inds), [&] { return n++; });

      std::sort(std::begin(sorted_inds), std::end(sorted_inds), [&](int i1, int i2) {
        return pfcands->at(i1).pt() > pfcands->at(i2).pt();
      });

      for (size_t pf_index = 0; pf_index < pfcands->size() && iPF < max_iPF; pf_index++) {
        const pat::PackedCandidate& p = static_cast<const pat::PackedCandidate&>(pfcands->at(sorted_inds.at(pf_index)));
        float deltaR_tau_p = deltaR(p.p4(), tau.p4());

        if (p.pt() < 0.5)
          continue;
        if (deltaR_tau_p > 0.5)
          continue;
        if (p.fromPV() < 1 && p.charge() != 0)
          continue;
        pfCandPt = p.pt();
        pfCandPtRel = p.pt() / lepRecoPt;

        pfCandDr = deltaR_tau_p;
        pfCandDEta = std::abs(lepRecoEta - p.eta());
        pfCandDPhi = std::abs(deltaPhi(lepRecoPhi, p.phi()));
        pfCandEta = p.eta();
        pfCandIsBarrel = (std::abs(pfCandEta) < 1.4);
        pfCandPz = std::abs(std::sinh(pfCandEta) * pfCandPt);
        pfCandPzRel = pfCandPz / lepRecoPz;
        pfCandPdgID = std::abs(p.pdgId());
        pfCandCharge = p.charge();
        pfCandDVx_1 = p.vx() - pvx;
        pfCandDVy_1 = p.vy() - pvy;
        pfCandDVz_1 = p.vz() - pvz;

        pfCandD_1 = std::sqrt(pfCandDVx_1 * pfCandDVx_1 + pfCandDVy_1 * pfCandDVy_1 + pfCandDVz_1 * pfCandDVz_1);

        if (pfCandCharge != 0 and p.hasTrackDetails()) {
          pfCandDz = p.dz();
          pfCandDzErr = p.dzError();
          pfCandDzSig = (std::abs(p.dz()) + 0.000001) / (p.dzError() + 0.00001);
          pfCandD0 = p.dxy();
          pfCandD0Err = p.dxyError();
          pfCandD0Sig = (std::abs(p.dxy()) + 0.000001) / (p.dxyError() + 0.00001);
          pfCandPixHits = p.numberOfPixelHits();
          pfCandHits = p.numberOfHits();
          pfCandLostInnerHits = p.lostInnerHits();
        } else {
          float disp = 1;
          int psudorand = p.pt() * 1000000;
          if (psudorand % 2 == 0)
            disp = -1;
          pfCandDz = 5 * disp;
          pfCandDzErr = 0;
          pfCandDzSig = 0;
          pfCandD0 = 5 * disp;
          pfCandD0Err = 0;
          pfCandD0Sig = 0;
          pfCandPixHits = 0;
          pfCandHits = 0;
          pfCandLostInnerHits = 2.;
          pfCandDVx_1 = 1;
          pfCandDVy_1 = 1;
          pfCandDVz_1 = 1;
          pfCandD_1 = 1;
        }

        pfCandPuppiWeight = p.puppiWeight();
        pfCandFromPV = p.fromPV();
        pfCandVtxQuality = p.pvAssociationQuality();
        pfCandHighPurityTrk = p.trackHighPurity();
        float pfCandTauIndMatch_temp = 0;

        for (auto i : signalCandidateInds) {
          if (i == sorted_inds.at(pf_index))
            pfCandTauIndMatch_temp = 1;
        }

        pfCandTauIndMatch = pfCandTauIndMatch_temp;
        pfCandPtRelPtRel = pfCandPtRel * pfCandPtRel;
        pfCandPt = std::min(pfCandPt, pfCandPt_max);
        pfCandPt = pfCandPt / pfCandPt_max;

        pfCandPz = std::min(pfCandPz, pfCandPz_max);
        pfCandPz = pfCandPz / pfCandPz_max;

        pfCandPtRel = std::min(pfCandPtRel, pfCandPtRel_max);
        pfCandPzRel = std::min(pfCandPzRel, pfCandPzRel_max);
        pfCandPzRel = pfCandPzRel / pfCandPzRel_max;
        pfCandDr = pfCandDr / pfCandDr_max;
        pfCandEta = pfCandEta / pfCandEta_max;
        pfCandDEta = pfCandDEta / pfCandDEta_max;
        pfCandDPhi = pfCandDPhi / pfCandDPhi_max;
        pfCandPixHits = pfCandPixHits / pfCandPixHits_max;
        pfCandHits = pfCandHits / pfCandHits_max;

        pfCandPtRelPtRel = std::min(pfCandPtRelPtRel, pfCandPtRelPtRel_max);

        pfCandD0 = std::clamp(pfCandD0, -pfCandD0_max, pfCandD0_max);
        pfCandD0 = pfCandD0 / pfCandD0_max;

        pfCandDz = std::clamp(pfCandDz, -pfCandDz_max, pfCandDz_max);
        pfCandDz = pfCandDz / pfCandDz_max;

        pfCandD0Err = std::min(pfCandD0Err, pfCandD0_z_Err_max);
        pfCandDzErr = std::min(pfCandDzErr, pfCandD0_z_Err_max);
        pfCandDzSig = std::min(pfCandDzSig, pfCandDzSig_max);
        pfCandDzSig = pfCandDzSig / pfCandDzSig_max;

        pfCandD0Sig = std::min(pfCandD0Sig, pfCandD0Sig_max);
        pfCandD0D0 = pfCandD0 * pfCandD0;
        pfCandDzDz = pfCandDz * pfCandDz;
        pfCandD0Dz = pfCandD0 * pfCandDz;
        pfCandD0Dphi = pfCandD0 * pfCandDPhi;

        pfCandDVx_1 = std::clamp(pfCandDVx_1, -pfCandDVx_y_z_1_max, pfCandDVx_y_z_1_max);
        pfCandDVx_1 = pfCandDVx_1 / pfCandDVx_y_z_1_max;

        pfCandDVy_1 = std::clamp(pfCandDVy_1, -pfCandDVx_y_z_1_max, pfCandDVx_y_z_1_max);
        pfCandDVy_1 = pfCandDVy_1 / pfCandDVx_y_z_1_max;

        pfCandDVz_1 = std::clamp(pfCandDVz_1, -pfCandDVx_y_z_1_max, pfCandDVx_y_z_1_max);
        pfCandDVz_1 = pfCandDVz_1 / pfCandDVx_y_z_1_max;

        pfCandD_1 = std::clamp(pfCandD_1, -pfCandD_1_max, pfCandD_1_max);
        pfCandD_1 = pfCandD_1 / pfCandD_1_max;

        if (graphVersion == 0) {
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 0) = pfCandPt;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 1) = pfCandPz;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 2) = pfCandPtRel;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 3) = pfCandPzRel;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 4) = pfCandDr;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 5) = pfCandDEta;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 6) = pfCandDPhi;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 7) = pfCandEta;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 8) = pfCandDz;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 9) = pfCandDzSig;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 10) = pfCandD0;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 11) = pfCandD0Sig;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 12) = pfCandDzErr;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 13) = pfCandD0Err;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 14) = pfCandD0D0;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 15) = pfCandCharge == 0;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 16) = pfCandCharge == 1;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 17) = pfCandCharge == -1;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 18) = pfCandPdgID > 22;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 19) = pfCandPdgID == 22;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 20) = pfCandDzDz;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 21) = pfCandD0Dz;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 22) = pfCandD0Dphi;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 23) = pfCandPtRelPtRel;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 24) = pfCandPixHits;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 25) = pfCandHits;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 26) = pfCandLostInnerHits == -1;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 27) = pfCandLostInnerHits == 0;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 28) = pfCandLostInnerHits == 1;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 29) = pfCandLostInnerHits == 2;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 30) = pfCandPuppiWeight;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 31) = (pfCandVtxQuality == 1);
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 32) = (pfCandVtxQuality == 5);
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 33) = (pfCandVtxQuality == 6);
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 34) = (pfCandVtxQuality == 7);
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 35) = (pfCandFromPV == 1);
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 36) = (pfCandFromPV == 2);
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 37) = (pfCandFromPV == 3);
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 38) = pfCandIsBarrel;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 39) = pfCandHighPurityTrk;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 40) = pfCandPdgID == 1;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 41) = pfCandPdgID == 2;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 42) = pfCandPdgID == 11;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 43) = pfCandPdgID == 13;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 44) = pfCandPdgID == 130;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 45) = pfCandPdgID == 211;
          tensor.tensor<float, 3>()(0, 60 - 1 - iPF, 46) = pfCandTauIndMatch;
        }

        if (graphVersion == 1) {
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 0) = pfCandPt;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 1) = pfCandPz;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 2) = pfCandPtRel;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 3) = pfCandPzRel;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 4) = pfCandDr;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 5) = pfCandDEta;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 6) = pfCandDPhi;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 7) = pfCandEta;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 8) = pfCandDz;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 9) = pfCandDzSig;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 10) = pfCandD0;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 11) = pfCandD0Sig;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 12) = pfCandDzErr;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 13) = pfCandD0Err;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 14) = pfCandD0D0;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 15) = pfCandCharge == 0;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 16) = pfCandCharge == 1;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 17) = pfCandCharge == -1;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 18) = pfCandPdgID > 22;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 19) = pfCandPdgID == 22;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 20) = pfCandDVx_1;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 21) = pfCandDVy_1;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 22) = pfCandDVz_1;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 23) = pfCandD_1;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 24) = pfCandDzDz;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 25) = pfCandD0Dz;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 26) = pfCandD0Dphi;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 27) = pfCandPtRelPtRel;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 28) = pfCandPixHits;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 29) = pfCandHits;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 30) = pfCandLostInnerHits == -1;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 31) = pfCandLostInnerHits == 0;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 32) = pfCandLostInnerHits == 1;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 33) = pfCandLostInnerHits == 2;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 34) = pfCandPuppiWeight;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 35) = (pfCandVtxQuality == 1);
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 36) = (pfCandVtxQuality == 5);
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 37) = (pfCandVtxQuality == 6);
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 38) = (pfCandVtxQuality == 7);
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 39) = (pfCandFromPV == 1);
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 40) = (pfCandFromPV == 2);
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 41) = (pfCandFromPV == 3);
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 42) = pfCandIsBarrel;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 43) = pfCandHighPurityTrk;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 44) = pfCandPdgID == 1;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 45) = pfCandPdgID == 2;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 46) = pfCandPdgID == 11;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 47) = pfCandPdgID == 13;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 48) = pfCandPdgID == 130;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 49) = pfCandPdgID == 211;
          tensor.tensor<float, 3>()(0, 36 - 1 - iPF, 50) = pfCandTauIndMatch;
        }
        iPF++;
      }
      tensorflow::run(&(cache_->getSession()), {{"input_1", tensor}}, {"output_node0"}, &outputs_);
      predictions.matrix<float>()(tau_index, 0) = outputs_[0].flat<float>()(0);
    }
    return predictions;
  }

private:
  unsigned graphVersion;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DPFIsolation);
