#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "RecoBTag/FeatureTools/interface/TrackInfoBuilder.h"

using namespace btagbtvdeep;

#include <string>

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

// add tag info and a way to go back to the jet reference
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/UnifiedParticleTransformerAK4TagInfo.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

// To store the gen info to get the truth flavour of the jet
#include "DataFormats/PatCandidates/interface/Jet.h"

template <typename T>
class JetTaggerTableProducer : public edm::stream::EDProducer<> {
public:
  explicit JetTaggerTableProducer(const edm::ParameterSet &);
  ~JetTaggerTableProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  const std::string nameDeepJet_;
  const std::string idx_nameDeepJet_;
  const unsigned int n_cpf_ = 29;
  const unsigned int n_npf_ = 25;
  const unsigned int n_sv_ = 12;
  const unsigned int n_lt_ = 5;

  edm::EDGetTokenT<edm::View<T>> jet_token_;

  typedef std::vector<reco::UnifiedParticleTransformerAK4TagInfo> TagInfoCollection;
  const edm::EDGetTokenT<TagInfoCollection> tag_info_src_;

  constexpr static bool usePhysForLightAndUndefined = false;
};

//
// constructors and destructor
//
template <typename T>
JetTaggerTableProducer<T>::JetTaggerTableProducer(const edm::ParameterSet &iConfig)
    : nameDeepJet_(iConfig.getParameter<std::string>("nameDeepJet")),
      idx_nameDeepJet_(iConfig.getParameter<std::string>("idx_nameDeepJet")),
      n_cpf_(iConfig.getParameter<unsigned int>("n_cpf")),
      n_npf_(iConfig.getParameter<unsigned int>("n_npf")),
      n_sv_(iConfig.getParameter<unsigned int>("n_sv")),
      n_lt_(iConfig.getParameter<unsigned int>("n_lt")),
      jet_token_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("jets"))),
      tag_info_src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("tagInfo_src"))) {
  produces<nanoaod::FlatTable>(nameDeepJet_);
}

template <typename T>
JetTaggerTableProducer<T>::~JetTaggerTableProducer() {}

template <typename T>
void JetTaggerTableProducer<T>::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // elements in all these collections must have the same order!

  // only necessary to explicitly check correct matching of jets

  auto jets = iEvent.getHandle(jet_token_);

  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(tag_info_src_, tag_infos);

  unsigned nJets = jets->size();

  std::vector<int> jet_N_CPFCands(nJets);
  std::vector<int> jet_N_NPFCands(nJets);
  std::vector<int> jet_N_SVs(nJets);
  std::vector<int> jet_N_LTs(nJets);

  // should default to 0 if less than nCpf cpf with information
  std::vector<std::vector<float>> Cpfcan_BtagPf_trackEtaRel_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_BtagPf_trackPtRel_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_BtagPf_trackPPar_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_BtagPf_trackDeltaR_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_BtagPf_trackPParRatio_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_BtagPf_trackSip2dVal_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_BtagPf_trackSip2dSig_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_BtagPf_trackSip3dVal_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_BtagPf_trackSip3dSig_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_BtagPf_trackJetDistVal_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_ptrel_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_drminsv_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<int>> Cpfcan_VTX_ass_nCpf(n_cpf_, std::vector<int>(nJets));
  std::vector<std::vector<float>> Cpfcan_puppiw_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_chi2_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<int>> Cpfcan_quality_nCpf(n_cpf_, std::vector<int>(nJets));
  std::vector<std::vector<float>> Cpfcan_charge_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_dz_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_btagPf_trackDecayLen_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_HadFrac_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_CaloFrac_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_pdgID_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_lostInnerHits_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_numberOfPixelHits_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_numberOfStripHits_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_px_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_py_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_pz_nCpf(n_cpf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Cpfcan_e_nCpf(n_cpf_, std::vector<float>(nJets));

  // should default to 0 if less than nNpf npf with information
  std::vector<std::vector<float>> Npfcan_ptrel_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Npfcan_etarel_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Npfcan_phirel_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Npfcan_deltaR_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<int>> Npfcan_isGamma_nNpf(n_npf_, std::vector<int>(nJets));
  std::vector<std::vector<float>> Npfcan_HadFrac_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Npfcan_drminsv_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Npfcan_puppiw_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Npfcan_px_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Npfcan_py_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Npfcan_pz_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Npfcan_e_nNpf(n_npf_, std::vector<float>(nJets));

  // should default to 0 if less than nSv SVs with information
  std::vector<std::vector<float>> sv_pt_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_deltaR_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_mass_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_etarel_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_phirel_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<int>> sv_ntracks_nSV(n_sv_, std::vector<int>(nJets));
  std::vector<std::vector<float>> sv_chi2_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_normchi2_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_dxy_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_dxysig_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_d3d_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_d3dsig_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_costhetasvpv_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_enratio_nSV(n_sv_, std::vector<float>(nJets));
#ifdef JTTP_NEED_SV_PE  // disabled by default to better save space
  std::vector<std::vector<float>> sv_px_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_py_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_pz_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_e_nSV(n_sv_, std::vector<float>(nJets));
#else  /* JTTP_NEED_SV_PE */
  std::vector<std::vector<float>> sv_eta_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_phi_nSV(n_sv_, std::vector<float>(nJets));
#endif /* JTTP_NEED_SV_PE */

  // should default to 0 if less than nLT LTs with information
  std::vector<std::vector<float>> lt_btagPf_trackEtaRel_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_btagPf_trackPtRel_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_btagPf_trackPPar_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_btagPf_trackDeltaR_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_btagPf_trackPParRatio_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_btagPf_trackSip2dVal_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_btagPf_trackSip2dSig_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_btagPf_trackSip3dVal_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_btagPf_trackSip3dSig_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_btagPf_trackJetDistVal_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_drminsv_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_charge_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_puppiw_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_chi2_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_quality_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_lostInnerHits_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_numberOfPixelHits_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_numberOfStripHits_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_pt_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_eta_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_phi_nLT(n_lt_, std::vector<float>(nJets));
  std::vector<std::vector<float>> lt_e_nLT(n_lt_, std::vector<float>(nJets));

  if (!tag_infos->empty()) {
    for (unsigned i_jet = 0; i_jet < nJets; ++i_jet) {
      // jet loop reads tag info instead of constituent info

      const auto &taginfo = (*tag_infos)[i_jet];
      const auto &features = taginfo.features();

      // jet.pt and jet.eta as well as other jet variables (ShallowTagInfo) already included (via DeepCSV)
      // number of elements in different collections
      jet_N_CPFCands[i_jet] = features.c_pf_features.size();
      jet_N_NPFCands[i_jet] = features.n_pf_features.size();
      jet_N_SVs[i_jet] = features.sv_features.size();
      jet_N_LTs[i_jet] = features.lt_features.size();

      std::vector<const btagbtvdeep::ChargedCandidateFeatures *> ranked_c_pf_features;
      ranked_c_pf_features.reserve(features.c_pf_features.size());
      for (auto &c_pf : features.c_pf_features)
        ranked_c_pf_features.push_back(&c_pf);

      std::vector<const btagbtvdeep::NeutralCandidateFeatures *> ranked_n_pf_features;
      ranked_n_pf_features.reserve(features.n_pf_features.size());
      for (auto &n_pf : features.n_pf_features)
        ranked_n_pf_features.push_back(&n_pf);

      std::vector<const btagbtvdeep::SecondaryVertexFeatures *> ranked_sv_features;
      ranked_sv_features.reserve(features.sv_features.size());
      for (auto &sv : features.sv_features)
        ranked_sv_features.push_back(&sv);

      std::vector<const btagbtvdeep::LostTracksFeatures *> ranked_lt_features;
      ranked_lt_features.reserve(features.lt_features.size());
      for (auto &lt : features.lt_features)
        ranked_lt_features.push_back(&lt);

      auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t)n_cpf_);
      auto max_n_pf_n = std::min(features.n_pf_features.size(), (std::size_t)n_npf_);
      auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)n_sv_);
      auto max_lt_n = std::min(features.lt_features.size(), (std::size_t)n_lt_);

      auto c_pf_cmp = [](const btagbtvdeep::ChargedCandidateFeatures *a,
                         const btagbtvdeep::ChargedCandidateFeatures *b) { return a->pt > b->pt; };
      //auto n_pf_cmp = [](const btagbtvdeep::NeutralCandidateFeatures *a, const btagbtvdeep::NeutralCandidateFeatures *b)
      //{ return a->pt > b->pt; };
      auto sv_cmp = [](const btagbtvdeep::SecondaryVertexFeatures *a, const btagbtvdeep::SecondaryVertexFeatures *b) {
        return a->pt > b->pt;
      };
      //auto lt_cmp = [](const btagbtvdeep::LostTracksFeatures *a, const btagbtvdeep::LostTracksFeatures *b)
      //{ return a->pt > b->pt; };

      auto c_pf_cmp_ip = [](const btagbtvdeep::ChargedCandidateFeatures *a,
                            const btagbtvdeep::ChargedCandidateFeatures *b) {
        return fabs(a->btagPf_trackSip3dVal) > fabs(b->btagPf_trackSip3dVal);
      };
      auto sv_cmp_ip = [](const btagbtvdeep::SecondaryVertexFeatures *a,
                          const btagbtvdeep::SecondaryVertexFeatures *b) { return fabs(a->d3d) > fabs(b->d3d); };

      // c_pf
      if (n_cpf_ == 2) {
        // 0: highest pT
        // 1: highest IP
        if (!ranked_c_pf_features.empty()) {
          auto highest_pT = *std::min_element(ranked_c_pf_features.begin(), ranked_c_pf_features.end(), c_pf_cmp);
          auto highest_IP = *std::min_element(ranked_c_pf_features.begin(), ranked_c_pf_features.end(), c_pf_cmp_ip);
          ranked_c_pf_features = {highest_pT, highest_IP};
        }
      } else {
        // highest pT
        //std::nth_element(ranked_c_pf_features.begin(),
        //    ranked_c_pf_features.begin() + max_c_pf_n, ranked_c_pf_features.end(), c_pf_cmp);
        //std::sort(ranked_c_pf_features.begin(), ranked_c_pf_features.end(), c_pf_cmp);
      }

      // n_pf
      {
        // highest pT
        //std::nth_element(ranked_n_pf_features.begin(),
        //    ranked_n_pf_features.begin() + max_n_pf_n, ranked_n_pf_features.end(), n_pf_cmp);
        //std::sort(ranked_n_pf_features.begin(), ranked_n_pf_features.end(), n_pf_cmp);
      }

      // sv
      if (n_sv_ == 2) {
        // 0: highest pT
        // 1: highest IP
        if (ranked_sv_features.size() >= 2) {
          auto highest_pT = *std::min_element(ranked_sv_features.begin(), ranked_sv_features.end(), sv_cmp);
          auto highest_IP = *std::min_element(ranked_sv_features.begin(), ranked_sv_features.end(), sv_cmp_ip);
          if (highest_IP == highest_pT) {  // 1 overlaps with 0: try 2nd highest IP
            std::nth_element(
                ranked_sv_features.begin(), next(ranked_sv_features.begin()), ranked_sv_features.end(), sv_cmp_ip);
            for (size_t isv = 0; isv < 2; ++isv) {  // At most one of the both overlaps with 0.
              highest_IP = ranked_sv_features[isv];
              if (highest_IP != highest_pT)
                break;
            }
          }
          ranked_sv_features.clear();
          ranked_sv_features = {highest_pT, highest_IP};
        }
      } else {
        // highest pT
        //std::nth_element(ranked_sv_features.begin(),
        //    ranked_sv_features.begin() + max_sv_n, ranked_sv_features.end(), sv_cmp);
        //std::sort(ranked_sv_features.begin(), ranked_sv_features.end(), sv_cmp);
      }

      // lt
      {
        // highest pT
        //std::nth_element(ranked_lt_features.begin(),
        //    ranked_lt_features.begin() + max_lt_n, ranked_lt_features.end(), lt_cmp);
        //std::sort(ranked_lt_features.begin(), ranked_lt_features.end(), lt_cmp);
      }

      // c_pf candidates
      for (std::size_t c_pf_n = 0; c_pf_n < max_c_pf_n; c_pf_n++) {
        const auto &c_pf_features = *ranked_c_pf_features.at(c_pf_n);
        Cpfcan_BtagPf_trackEtaRel_nCpf[c_pf_n][i_jet] = c_pf_features.btagPf_trackEtaRel;
        Cpfcan_BtagPf_trackPtRel_nCpf[c_pf_n][i_jet] = c_pf_features.btagPf_trackPtRel;
        Cpfcan_BtagPf_trackPPar_nCpf[c_pf_n][i_jet] = c_pf_features.btagPf_trackPPar;
        Cpfcan_BtagPf_trackDeltaR_nCpf[c_pf_n][i_jet] = c_pf_features.btagPf_trackDeltaR;
        Cpfcan_BtagPf_trackPParRatio_nCpf[c_pf_n][i_jet] = c_pf_features.btagPf_trackPParRatio;
        Cpfcan_BtagPf_trackSip2dVal_nCpf[c_pf_n][i_jet] = c_pf_features.btagPf_trackSip2dVal;
        Cpfcan_BtagPf_trackSip2dSig_nCpf[c_pf_n][i_jet] = c_pf_features.btagPf_trackSip2dSig;
        Cpfcan_BtagPf_trackSip3dVal_nCpf[c_pf_n][i_jet] = c_pf_features.btagPf_trackSip3dVal;
        Cpfcan_BtagPf_trackSip3dSig_nCpf[c_pf_n][i_jet] = c_pf_features.btagPf_trackSip3dSig;
        Cpfcan_BtagPf_trackJetDistVal_nCpf[c_pf_n][i_jet] = c_pf_features.btagPf_trackJetDistVal;
        Cpfcan_ptrel_nCpf[c_pf_n][i_jet] = c_pf_features.ptrel;
        Cpfcan_drminsv_nCpf[c_pf_n][i_jet] = c_pf_features.drminsv;
        Cpfcan_VTX_ass_nCpf[c_pf_n][i_jet] = c_pf_features.vtx_ass;
        Cpfcan_puppiw_nCpf[c_pf_n][i_jet] = c_pf_features.puppiw;
        Cpfcan_chi2_nCpf[c_pf_n][i_jet] = c_pf_features.chi2;
        Cpfcan_quality_nCpf[c_pf_n][i_jet] = c_pf_features.quality;
        Cpfcan_charge_nCpf[c_pf_n][i_jet] = c_pf_features.charge;
        Cpfcan_dz_nCpf[c_pf_n][i_jet] = c_pf_features.dz;
        Cpfcan_btagPf_trackDecayLen_nCpf[c_pf_n][i_jet] = c_pf_features.btagPf_trackDecayLen;
        Cpfcan_HadFrac_nCpf[c_pf_n][i_jet] = c_pf_features.HadFrac;
        Cpfcan_CaloFrac_nCpf[c_pf_n][i_jet] = c_pf_features.CaloFrac;
        Cpfcan_pdgID_nCpf[c_pf_n][i_jet] = c_pf_features.pdgID;
        Cpfcan_lostInnerHits_nCpf[c_pf_n][i_jet] = c_pf_features.lostInnerHits;
        Cpfcan_numberOfPixelHits_nCpf[c_pf_n][i_jet] = c_pf_features.numberOfPixelHits;
        Cpfcan_numberOfStripHits_nCpf[c_pf_n][i_jet] = c_pf_features.numberOfStripHits;
        Cpfcan_px_nCpf[c_pf_n][i_jet] = c_pf_features.px;
        Cpfcan_py_nCpf[c_pf_n][i_jet] = c_pf_features.py;
        Cpfcan_pz_nCpf[c_pf_n][i_jet] = c_pf_features.pz;
        Cpfcan_e_nCpf[c_pf_n][i_jet] = c_pf_features.e;
      }

      // n_pf candidates
      for (std::size_t n_pf_n = 0; n_pf_n < max_n_pf_n; n_pf_n++) {
        const auto &n_pf_features = *ranked_n_pf_features.at(n_pf_n);
        Npfcan_ptrel_nNpf[n_pf_n][i_jet] = n_pf_features.ptrel;
        Npfcan_etarel_nNpf[n_pf_n][i_jet] = n_pf_features.etarel;
        Npfcan_phirel_nNpf[n_pf_n][i_jet] = n_pf_features.phirel;
        Npfcan_deltaR_nNpf[n_pf_n][i_jet] = n_pf_features.deltaR;
        Npfcan_isGamma_nNpf[n_pf_n][i_jet] = n_pf_features.isGamma;
        Npfcan_HadFrac_nNpf[n_pf_n][i_jet] = n_pf_features.hadFrac;
        Npfcan_drminsv_nNpf[n_pf_n][i_jet] = n_pf_features.drminsv;
        Npfcan_puppiw_nNpf[n_pf_n][i_jet] = n_pf_features.puppiw;
        Npfcan_px_nNpf[n_pf_n][i_jet] = n_pf_features.px;
        Npfcan_py_nNpf[n_pf_n][i_jet] = n_pf_features.py;
        Npfcan_pz_nNpf[n_pf_n][i_jet] = n_pf_features.pz;
        Npfcan_e_nNpf[n_pf_n][i_jet] = n_pf_features.e;
      }

      // sv candidates
      for (std::size_t sv_n = 0; sv_n < max_sv_n; sv_n++) {
        const auto &sv_features = *ranked_sv_features.at(sv_n);
        sv_pt_nSV[sv_n][i_jet] = sv_features.pt;
        sv_deltaR_nSV[sv_n][i_jet] = sv_features.deltaR;
        sv_mass_nSV[sv_n][i_jet] = sv_features.mass;
        sv_etarel_nSV[sv_n][i_jet] = sv_features.etarel;
        sv_phirel_nSV[sv_n][i_jet] = sv_features.phirel;
        sv_ntracks_nSV[sv_n][i_jet] = sv_features.ntracks;
        sv_chi2_nSV[sv_n][i_jet] = sv_features.chi2;
        sv_normchi2_nSV[sv_n][i_jet] = sv_features.normchi2;
        sv_dxy_nSV[sv_n][i_jet] = sv_features.dxy;
        sv_dxysig_nSV[sv_n][i_jet] = sv_features.dxysig;
        sv_d3d_nSV[sv_n][i_jet] = sv_features.d3d;
        sv_d3dsig_nSV[sv_n][i_jet] = sv_features.d3dsig;
        sv_costhetasvpv_nSV[sv_n][i_jet] = sv_features.costhetasvpv;
        sv_enratio_nSV[sv_n][i_jet] = sv_features.enratio;
#ifdef JTTP_NEED_SV_PE
        sv_px_nSV[sv_n][i_jet] = sv_features.px;
        sv_py_nSV[sv_n][i_jet] = sv_features.py;
        sv_pz_nSV[sv_n][i_jet] = sv_features.pz;
        sv_e_nSV[sv_n][i_jet] = sv_features.e;
#else  /* JTTP_NEED_SV_PE */
        sv_eta_nSV[sv_n][i_jet] = sv_features.eta;
        sv_phi_nSV[sv_n][i_jet] = sv_features.phi;
#endif /* JTTP_NEED_SV_PE */
      }

      // lt candidates
      for (std::size_t lt_n = 0; lt_n < max_lt_n; lt_n++) {
        const auto &lt_features = *ranked_lt_features.at(lt_n);
        lt_btagPf_trackEtaRel_nLT[lt_n][i_jet] = lt_features.btagPf_trackEtaRel;
        lt_btagPf_trackPtRel_nLT[lt_n][i_jet] = lt_features.btagPf_trackPtRel;
        lt_btagPf_trackPPar_nLT[lt_n][i_jet] = lt_features.btagPf_trackPPar;
        lt_btagPf_trackDeltaR_nLT[lt_n][i_jet] = lt_features.btagPf_trackDeltaR;
        lt_btagPf_trackPParRatio_nLT[lt_n][i_jet] = lt_features.btagPf_trackPParRatio;
        lt_btagPf_trackSip2dVal_nLT[lt_n][i_jet] = lt_features.btagPf_trackSip2dVal;
        lt_btagPf_trackSip2dSig_nLT[lt_n][i_jet] = lt_features.btagPf_trackSip2dSig;
        lt_btagPf_trackSip3dVal_nLT[lt_n][i_jet] = lt_features.btagPf_trackSip3dVal;
        lt_btagPf_trackSip3dSig_nLT[lt_n][i_jet] = lt_features.btagPf_trackSip3dSig;
        lt_btagPf_trackJetDistVal_nLT[lt_n][i_jet] = lt_features.btagPf_trackJetDistVal;
        lt_drminsv_nLT[lt_n][i_jet] = lt_features.drminsv;
        lt_charge_nLT[lt_n][i_jet] = lt_features.charge;
        lt_puppiw_nLT[lt_n][i_jet] = lt_features.puppiw;
        lt_chi2_nLT[lt_n][i_jet] = lt_features.chi2;
        lt_quality_nLT[lt_n][i_jet] = lt_features.quality;
        lt_lostInnerHits_nLT[lt_n][i_jet] = lt_features.lostInnerHits;
        lt_numberOfPixelHits_nLT[lt_n][i_jet] = lt_features.numberOfPixelHits;
        lt_numberOfStripHits_nLT[lt_n][i_jet] = lt_features.numberOfStripHits;
        lt_pt_nLT[lt_n][i_jet] = lt_features.pt;
        lt_eta_nLT[lt_n][i_jet] = lt_features.eta;
        lt_phi_nLT[lt_n][i_jet] = lt_features.phi;
        lt_e_nLT[lt_n][i_jet] = lt_features.e;
      }
    }
  }

  // DeepJetInputs table
  auto djTable = std::make_unique<nanoaod::FlatTable>(jet_N_CPFCands.size(), nameDeepJet_, false, true);

  djTable->addColumn<int>("DeepJet_nCpfcand", jet_N_CPFCands, "Number of charged PF candidates in the jet");
  djTable->addColumn<int>("DeepJet_nNpfcand", jet_N_NPFCands, "Number of neutral PF candidates in the jet");
  djTable->addColumn<int>("DeepJet_nsv", jet_N_SVs, "Number of secondary vertices in the jet");
  djTable->addColumn<int>("DeepJet_nlt", jet_N_LTs, "Number of lost tracks in the jet");

  // ============================================================== Cpfs ===================================================================
  for (unsigned int p = 0; p < n_cpf_; p++) {
    auto s = std::to_string(p);

    djTable->addColumn<float>("DeepJet_Cpfcan_BtagPf_trackDeltaR_" + s,
                              Cpfcan_BtagPf_trackDeltaR_nCpf[p],
                              "track pseudoangular distance from the jet axis for the " + s + ". cpf",
                              10);
    djTable->addColumn<float>("DeepJet_Cpfcan_BtagPf_trackEtaRel_" + s,
                              Cpfcan_BtagPf_trackEtaRel_nCpf[p],
                              "track pseudorapidity, relative to the jet axis for the " + s + ". cpf",
                              10);
    djTable->addColumn<float>("DeepJet_Cpfcan_BtagPf_trackJetDistVal_" + s,
                              Cpfcan_BtagPf_trackJetDistVal_nCpf[p],
                              "minimum track approach distance to jet axis for the " + s + ". cpf",
                              10);
    djTable->addColumn<float>("DeepJet_Cpfcan_BtagPf_trackPPar_" + s,
                              Cpfcan_BtagPf_trackPPar_nCpf[p],
                              "dot product of the jet and track momentum for the " + s + ". cpf",
                              10);
    djTable->addColumn<float>(
        "DeepJet_Cpfcan_BtagPf_trackPParRatio_" + s,
        Cpfcan_BtagPf_trackPParRatio_nCpf[p],
        "dot product of the jet and track momentum divided by the magnitude of the jet momentum for the " + s + ". cpf",
        10);
    djTable->addColumn<float>("DeepJet_Cpfcan_BtagPf_trackPtRel_" + s,
                              Cpfcan_BtagPf_trackPtRel_nCpf[p],
                              "track transverse momentum, relative to the jet axis for the " + s + ". cpf",
                              10);
    djTable->addColumn<float>("DeepJet_Cpfcan_BtagPf_trackSip2dSig_" + s,
                              Cpfcan_BtagPf_trackSip2dSig_nCpf[p],
                              "track 2D signed impact parameter significance for the " + s + ". cpf",
                              10);
    djTable->addColumn<float>("DeepJet_Cpfcan_BtagPf_trackSip3dSig_" + s,
                              Cpfcan_BtagPf_trackSip3dSig_nCpf[p],
                              "track 3D signed impact parameter significance for the " + s + ". cpf",
                              10);
    djTable->addColumn<float>("DeepJet_Cpfcan_BtagPf_trackSip2dVal_" + s,
                              Cpfcan_BtagPf_trackSip2dVal_nCpf[p],
                              "track 2D signed impact parameter for the " + s + ". cpf",
                              10);
    djTable->addColumn<float>("DeepJet_Cpfcan_BtagPf_trackSip3dVal_" + s,
                              Cpfcan_BtagPf_trackSip3dVal_nCpf[p],
                              "track 3D signed impact parameter for the " + s + ". cpf",
                              10);
    djTable->addColumn<float>("DeepJet_Cpfcan_ptrel_" + s,
                              Cpfcan_ptrel_nCpf[p],
                              "fraction of the jet momentum carried by the track for the " + s + ". cpf",
                              10);
    djTable->addColumn<float>("DeepJet_Cpfcan_drminsv_" + s,
                              Cpfcan_drminsv_nCpf[p],
                              "track pseudoangular distance from the closest secondary vertex of the " + s + ". cpf",
                              10);
    djTable->addColumn<int>(
        "DeepJet_Cpfcan_VTX_ass_" + s,
        Cpfcan_VTX_ass_nCpf[p],
        "integer flag that indicates whether the track was used in the primary vertex fit for the " + s + ". cpf",
        10);
    djTable->addColumn<float>("DeepJet_Cpfcan_puppiw_" + s,
                              Cpfcan_puppiw_nCpf[p],
                              "charged candidate PUPPI weight of the " + s + ". cpf",
                              10);
    djTable->addColumn<float>(
        "DeepJet_Cpfcan_chi2_" + s, Cpfcan_chi2_nCpf[p], "chi2 of the charged track fit for the " + s + ". cpf", 10);
    djTable->addColumn<int>(
        "DeepJet_Cpfcan_quality_" + s,
        Cpfcan_quality_nCpf[p],
        "integer flag which indicates the quality of the fitted track, based on number of detector hits used for the "
        "reconstruction as well as the overall chi2 of the charged track fit for the " +
            s + ". cpf",
        10);
    djTable->addColumn<float>("DeepJet_Cpfcan_charge_" + s, Cpfcan_charge_nCpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Cpfcan_dz_" + s, Cpfcan_dz_nCpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Cpfcan_btagPf_trackDecayLen_" + s, Cpfcan_btagPf_trackDecayLen_nCpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Cpfcan_HadFrac_" + s, Cpfcan_HadFrac_nCpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Cpfcan_CaloFrac_" + s, Cpfcan_CaloFrac_nCpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Cpfcan_pdgID_" + s, Cpfcan_pdgID_nCpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Cpfcan_lostInnerHits_" + s, Cpfcan_lostInnerHits_nCpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Cpfcan_numberOfPixelHits_" + s, Cpfcan_numberOfPixelHits_nCpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Cpfcan_numberOfStripHits_" + s, Cpfcan_numberOfStripHits_nCpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Cpfcan_px_" + s, Cpfcan_px_nCpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Cpfcan_py_" + s, Cpfcan_py_nCpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Cpfcan_pz_" + s, Cpfcan_pz_nCpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Cpfcan_e_" + s, Cpfcan_e_nCpf[p], "", 10);
  }

  // ============================================================== Npfs ===================================================================
  for (unsigned int p = 0; p < n_npf_; p++) {
    auto s = std::to_string(p);

    djTable->addColumn<float>("DeepJet_Npfcan_ptrel_" + s,
                              Npfcan_ptrel_nNpf[p],
                              "fraction of the jet momentum carried by the neutral candidate for the " + s + ". npf",
                              10);
    djTable->addColumn<float>("DeepJetExtra_Npfcan_etarel_" + s,
                              Npfcan_etarel_nNpf[p],
                              "pseudorapidity relative to parent jet for the " + s + ". npf",
                              10);
    djTable->addColumn<float>(
        "DeepJetExtra_Npfcan_phirel_" + s, Npfcan_phirel_nNpf[p], "DeltaPhi(npf, jet) for the " + s + ". npf", 10);
    djTable->addColumn<float>(
        "DeepJet_Npfcan_deltaR_" + s,
        Npfcan_deltaR_nNpf[p],
        "pseudoangular distance between the neutral candidate and the jet axis for the " + s + ". npf",
        10);
    djTable->addColumn<int>("DeepJet_Npfcan_isGamma_" + s,
                            Npfcan_isGamma_nNpf[p],
                            "integer flag indicating whether the neutral candidate is a photon for the " + s + ". npf",
                            10);
    djTable->addColumn<float>(
        "DeepJet_Npfcan_HadFrac_" + s,
        Npfcan_HadFrac_nNpf[p],
        "fraction of the neutral candidate energy deposited in the hadronic calorimeter for the " + s + ". npf",
        10);
    djTable->addColumn<float>(
        "DeepJet_Npfcan_drminsv_" + s,
        Npfcan_drminsv_nNpf[p],
        "pseudoangular distance between the neutral candidate and the closest secondary vertex for the " + s + ". npf",
        10);
    djTable->addColumn<float>("DeepJet_Npfcan_puppiw_" + s,
                              Npfcan_puppiw_nNpf[p],
                              "neutral candidate PUPPI weight for the " + s + ". npf",
                              10);
    djTable->addColumn<float>("DeepJet_Npfcan_px_" + s, Npfcan_px_nNpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Npfcan_py_" + s, Npfcan_py_nNpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Npfcan_pz_" + s, Npfcan_pz_nNpf[p], "", 10);
    djTable->addColumn<float>("DeepJet_Npfcan_e_" + s, Npfcan_e_nNpf[p], "", 10);
  }

  // ============================================================== SVs ===================================================================
  for (unsigned int p = 0; p < n_sv_; p++) {
    auto s = std::to_string(p);

    djTable->addColumn<float>("DeepJet_sv_pt_" + s, sv_pt_nSV[p], "SV pt of the " + s + ". SV", 10);
    djTable->addColumn<float>("DeepJet_sv_deltaR_" + s,
                              sv_deltaR_nSV[p],
                              "pseudoangular distance between jet axis and the " + s + ". SV direction",
                              10);
    djTable->addColumn<float>("DeepJet_sv_mass_" + s, sv_mass_nSV[p], "SV mass of the " + s + ". SV", 10);
    djTable->addColumn<float>("DeepJetExtra_sv_etarel_" + s,
                              sv_etarel_nSV[p],
                              "pseudorapidity relative to parent jet for the " + s + ". SV",
                              10);
    djTable->addColumn<float>(
        "DeepJetExtra_sv_phirel_" + s, sv_phirel_nSV[p], "DeltaPhi(sv, jet) for the " + s + ". SV", 10);
    djTable->addColumn<float>(
        "DeepJet_sv_ntracks_" + s, sv_ntracks_nSV[p], "Number of tracks asociated to the " + s + ". SV", 10);
    djTable->addColumn<float>("DeepJet_sv_chi2_" + s, sv_chi2_nSV[p], "chi2 of the " + s + ". SV", 10);
    djTable->addColumn<float>("DeepJet_sv_normchi2_" + s, sv_normchi2_nSV[p], "chi2/dof of the " + s + ". SV", 10);
    djTable->addColumn<float>(
        "DeepJet_sv_dxy_" + s, sv_dxy_nSV[p], "2D impact parameter (flight distance) value of the " + s + ". SV", 10);
    djTable->addColumn<float>("DeepJet_sv_dxysig_" + s,
                              sv_dxysig_nSV[p],
                              "2D impact parameter (flight distance) significance of the " + s + ". SV",
                              10);
    djTable->addColumn<float>(
        "DeepJet_sv_d3d_" + s, sv_d3d_nSV[p], "3D impact parameter (flight distance) value of the " + s + ". SV", 10);
    djTable->addColumn<float>("DeepJet_sv_d3dsig_" + s,
                              sv_d3dsig_nSV[p],
                              "3D impact parameter (flight distance) significance of the " + s + ". SV",
                              10);
    djTable->addColumn<float>("DeepJet_sv_costhetasvpv_" + s,
                              sv_costhetasvpv_nSV[p],
                              "cosine of the angle between the " + s +
                                  ". SV flight direction and the direction of the " + s + ". SV momentum",
                              10);
    djTable->addColumn<float>(
        "DeepJet_sv_enratio_" + s, sv_enratio_nSV[p], "ratio of the " + s + ". SV energy ratio to the jet energy", 10);
#ifdef JTTP_NEED_SV_PE
    djTable->addColumn<float>("DeepJet_sv_px_" + s, sv_px_nSV[p], "", 10);
    djTable->addColumn<float>("DeepJet_sv_py_" + s, sv_py_nSV[p], "", 10);
    djTable->addColumn<float>("DeepJet_sv_pz_" + s, sv_pz_nSV[p], "", 10);
    djTable->addColumn<float>("DeepJet_sv_e_" + s, sv_e_nSV[p], "", 10);
#else  /* JTTP_NEED_SV_PE */
    djTable->addColumn<float>("DeepJet_sv_eta_" + s, sv_eta_nSV[p], "", 10);
    djTable->addColumn<float>("DeepJet_sv_phi_" + s, sv_phi_nSV[p], "", 10);
#endif /* JTTP_NEED_SV_PE */
  }

  // ============================================================== LTs ===================================================================
  for (unsigned int p = 0; p < n_lt_; p++) {
    auto s = std::to_string(p);

    djTable->addColumn<float>("DeepJet_lt_btagPf_trackEtaRel_" + s, lt_btagPf_trackEtaRel_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_btagPf_trackPtRel_" + s, lt_btagPf_trackPtRel_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_btagPf_trackPPar_" + s, lt_btagPf_trackPPar_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_btagPf_trackDeltaR_" + s, lt_btagPf_trackDeltaR_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_btagPf_trackPParRatio_" + s, lt_btagPf_trackPParRatio_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_btagPf_trackSip2dVal_" + s, lt_btagPf_trackSip2dVal_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_btagPf_trackSip2dSig_" + s, lt_btagPf_trackSip2dSig_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_btagPf_trackSip3dVal_" + s, lt_btagPf_trackSip3dVal_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_btagPf_trackSip3dSig_" + s, lt_btagPf_trackSip3dSig_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_btagPf_trackJetDistVal_" + s, lt_btagPf_trackJetDistVal_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_drminsv_" + s, lt_drminsv_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_charge_" + s, lt_charge_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_puppiw_" + s, lt_puppiw_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_chi2_" + s, lt_chi2_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_quality_" + s, lt_quality_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_lostInnerHits_" + s, lt_lostInnerHits_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_numberOfPixelHits_" + s, lt_numberOfPixelHits_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_numberOfStripHits_" + s, lt_numberOfStripHits_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_pt_" + s, lt_pt_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_eta_" + s, lt_eta_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_phi_" + s, lt_phi_nLT[p], "", 10);
    djTable->addColumn<float>("DeepJet_lt_e_" + s, lt_e_nLT[p], "", 10);
  }

  iEvent.put(std::move(djTable), nameDeepJet_);
}

template <typename T>
void JetTaggerTableProducer<T>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameDeepJet", "Jet");
  desc.add<std::string>("idx_nameDeepJet", "djIdx");

  desc.add<unsigned int>("n_cpf", 2);
  desc.add<unsigned int>("n_npf", 2);
  desc.add<unsigned int>("n_sv", 2);
  desc.add<unsigned int>("n_lt", 2);
  desc.add<edm::InputTag>("jets", edm::InputTag("slimmedJetsPuppi"));
  desc.add<edm::InputTag>("tagInfo_src", edm::InputTag("pfUnifiedParticleTransformerAK4TagInfosPuppiWithDeepInfo"));
  descriptions.addWithDefaultLabel(desc);
}

typedef JetTaggerTableProducer<pat::Jet> PatJetTaggerTableProducer;

DEFINE_FWK_MODULE(PatJetTaggerTableProducer);
