#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "RecoBTag/FeatureTools/interface/TrackInfoBuilder.h"

using namespace btagbtvdeep;

#include "DataFormats/Math/interface/deltaR.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include <string>

// add tag info and a way to go back to the jet reference
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"

// To store the gen info to get the truth flavour of the jet
#include "DataFormats/PatCandidates/interface/Jet.h"

template <typename T>
class DeepJetTableProducer : public edm::stream::EDProducer<> {
public:
  explicit DeepJetTableProducer(const edm::ParameterSet&);
  ~DeepJetTableProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const std::string nameDeepJet_;
  const std::string idx_nameDeepJet_;
  const unsigned int n_cpf_ = 25;
  const unsigned int n_npf_ = 25;
  const unsigned int n_sv_ = 12;

  edm::EDGetTokenT<edm::View<T>> jet_token_;

  typedef std::vector<reco::DeepFlavourTagInfo> TagInfoCollection;
  const edm::EDGetTokenT<TagInfoCollection> tag_info_src_;

  constexpr static bool usePhysForLightAndUndefined = false;
};

//
// constructors and destructor
//
template <typename T>
DeepJetTableProducer<T>::DeepJetTableProducer(const edm::ParameterSet& iConfig)
    : nameDeepJet_(iConfig.getParameter<std::string>("nameDeepJet")),
      idx_nameDeepJet_(iConfig.getParameter<std::string>("idx_nameDeepJet")),
      n_cpf_(iConfig.getParameter<unsigned int>("n_cpf")),
      n_npf_(iConfig.getParameter<unsigned int>("n_npf")),
      n_sv_(iConfig.getParameter<unsigned int>("n_sv")),
      jet_token_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("jets"))),
      tag_info_src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("tagInfo_src"))) {
  produces<nanoaod::FlatTable>(nameDeepJet_);
}

template <typename T>
DeepJetTableProducer<T>::~DeepJetTableProducer() {}

template <typename T>
void DeepJetTableProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // elements in all these collections must have the same order!

  // only necessary to explicitly check correct matching of jets
  // std::vector<int> jetIdx_dj;

  auto jets = iEvent.getHandle(jet_token_);

  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(tag_info_src_, tag_infos);

  unsigned nJets = jets->size();

  std::vector<int> jet_N_CPFCands(nJets);
  std::vector<int> jet_N_NPFCands(nJets);
  std::vector<int> jet_N_PVs(nJets);
  std::vector<int> jet_N_SVs(nJets);

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

  // should default to 0 if less than nNpf npf with information
  std::vector<std::vector<float>> Npfcan_ptrel_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Npfcan_deltaR_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<int>> Npfcan_isGamma_nNpf(n_npf_, std::vector<int>(nJets));
  std::vector<std::vector<float>> Npfcan_HadFrac_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Npfcan_drminsv_nNpf(n_npf_, std::vector<float>(nJets));
  std::vector<std::vector<float>> Npfcan_puppiw_nNpf(n_npf_, std::vector<float>(nJets));
  /*
  // only after dataformat updated as well
  std::vector<std::vector<float>> Npfcan_etarel_nNpf(n_npf_, std::vector<float>(nJets)); 
  std::vector<std::vector<float>> Npfcan_phirel_nNpf(n_npf_, std::vector<float>(nJets));
  */

  // should default to 0 if less than nSv SVs with information
  std::vector<std::vector<float>> sv_mass_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_pt_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<int>> sv_ntracks_nSV(n_sv_, std::vector<int>(nJets));
  std::vector<std::vector<float>> sv_chi2_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_normchi2_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_dxy_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_dxysig_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_d3d_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_d3dsig_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_costhetasvpv_nSV(n_sv_, std::vector<float>(nJets));
  /*
  // only after dataformat updated as well
  std::vector<std::vector<float>> sv_etarel_nSV(n_sv_, std::vector<float>(nJets)); 
  std::vector<std::vector<float>> sv_phirel_nSV(n_sv_, std::vector<float>(nJets));
  */
  std::vector<std::vector<float>> sv_deltaR_nSV(n_sv_, std::vector<float>(nJets));
  std::vector<std::vector<float>> sv_enratio_nSV(n_sv_, std::vector<float>(nJets));

  if (!tag_infos->empty()) {
    for (unsigned i_jet = 0; i_jet < nJets; ++i_jet) {
      // jet loop reads tag info instead of constituent info

      const auto& taginfo = (*tag_infos)[i_jet];
      const auto& features = taginfo.features();

      // jet.pt and jet.eta as well as other jet variables (ShallowTagInfo) already included (via DeepCSV)
      // number of elements in different collections
      jet_N_CPFCands[i_jet] = features.c_pf_features.size();
      jet_N_NPFCands[i_jet] = features.n_pf_features.size();
      jet_N_SVs[i_jet] = features.sv_features.size();
      jet_N_PVs[i_jet] = features.npv;

      // c_pf candidates
      auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t)n_cpf_);
      for (std::size_t c_pf_n = 0; c_pf_n < max_c_pf_n; c_pf_n++) {
        const auto& c_pf_features = features.c_pf_features.at(c_pf_n);
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
      }

      // n_pf candidates
      auto max_n_pf_n = std::min(features.n_pf_features.size(), (std::size_t)n_npf_);
      for (std::size_t n_pf_n = 0; n_pf_n < max_n_pf_n; n_pf_n++) {
        const auto& n_pf_features = features.n_pf_features.at(n_pf_n);
        Npfcan_ptrel_nNpf[n_pf_n][i_jet] = n_pf_features.ptrel;
        Npfcan_deltaR_nNpf[n_pf_n][i_jet] = n_pf_features.deltaR;
        Npfcan_isGamma_nNpf[n_pf_n][i_jet] = n_pf_features.isGamma;
        Npfcan_HadFrac_nNpf[n_pf_n][i_jet] = n_pf_features.hadFrac;
        Npfcan_drminsv_nNpf[n_pf_n][i_jet] = n_pf_features.drminsv;
        Npfcan_puppiw_nNpf[n_pf_n][i_jet] = n_pf_features.puppiw;
        /*
            // only after dataformat updated as well
            Npfcan_etarel_nNpf[n_pf_n][i_jet] = n_pf_features.etarel;
            Npfcan_phirel_nNpf[n_pf_n][i_jet] = n_pf_features.phirel;
            */
      }

      // sv candidates
      auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)n_sv_);
      for (std::size_t sv_n = 0; sv_n < max_sv_n; sv_n++) {
        const auto& sv_features = features.sv_features.at(sv_n);
        sv_pt_nSV[sv_n][i_jet] = sv_features.pt;
        sv_deltaR_nSV[sv_n][i_jet] = sv_features.deltaR;
        sv_mass_nSV[sv_n][i_jet] = sv_features.mass;
        sv_ntracks_nSV[sv_n][i_jet] = sv_features.ntracks;
        sv_chi2_nSV[sv_n][i_jet] = sv_features.chi2;
        sv_normchi2_nSV[sv_n][i_jet] = sv_features.normchi2;
        sv_dxy_nSV[sv_n][i_jet] = sv_features.dxy;
        sv_dxysig_nSV[sv_n][i_jet] = sv_features.dxysig;
        sv_d3d_nSV[sv_n][i_jet] = sv_features.d3d;
        sv_d3dsig_nSV[sv_n][i_jet] = sv_features.d3dsig;
        sv_costhetasvpv_nSV[sv_n][i_jet] = sv_features.costhetasvpv;
        sv_enratio_nSV[sv_n][i_jet] = sv_features.enratio;
        /*
            // only after dataformat updated as well
            sv_etarel_nSV[sv_n][i_jet] = sv_features.etarel;
            sv_phirel_nSV[sv_n][i_jet] = sv_features.phirel;
            */
      }
    }
  }

  // DeepJetInputs table
  auto djTable = std::make_unique<nanoaod::FlatTable>(jet_N_CPFCands.size(), nameDeepJet_, false, true);
  //djTable->addColumn<int>("DeepJet_jetIdx", jetIdx_dj, "Index of the parent jet", );

  djTable->addColumn<int>("DeepJet_nCpfcand", jet_N_CPFCands, "Number of charged PF candidates in the jet");
  djTable->addColumn<int>("DeepJet_nNpfcand", jet_N_NPFCands, "Number of neutral PF candidates in the jet");
  djTable->addColumn<int>("DeepJet_nsv", jet_N_SVs, "Number of secondary vertices in the jet");
  djTable->addColumn<int>("DeepJet_npv", jet_N_PVs, "Number of primary vertices");

  // ============================================================== Cpfs ===================================================================
  for (unsigned int p = 0; p < n_cpf_; p++) {
    auto s = std::to_string(p);

    djTable->addColumn<float>("DeepJet_Cpfcan_puppiw_" + s,
                              Cpfcan_puppiw_nCpf[p],
                              "charged candidate PUPPI weight of the " + s + ". cpf",
                              10);
    djTable->addColumn<int>(
        "DeepJet_Cpfcan_VTX_ass_" + s,
        Cpfcan_VTX_ass_nCpf[p],
        "integer flag that indicates whether the track was used in the primary vertex fit for the " + s + ". cpf",
        10);
    djTable->addColumn<float>("DeepJet_Cpfcan_drminsv_" + s,
                              Cpfcan_drminsv_nCpf[p],
                              "track pseudoangular distance from the closest secondary vertex of the " + s + ". cpf",
                              10);
    djTable->addColumn<float>("DeepJet_Cpfcan_ptrel_" + s,
                              Cpfcan_ptrel_nCpf[p],
                              "fraction of the jet momentum carried by the track for the " + s + ". cpf",
                              10);
    djTable->addColumn<int>(
        "DeepJet_Cpfcan_quality_" + s,
        Cpfcan_quality_nCpf[p],
        "integer flag which indicates the quality of the fitted track, based on number of detector hits used for the "
        "reconstruction as well as the overall chi2 of the charged track fit for the " +
            s + ". cpf",
        10);
    djTable->addColumn<float>(
        "DeepJet_Cpfcan_chi2_" + s, Cpfcan_chi2_nCpf[p], "chi2 of the charged track fit for the " + s + ". cpf", 10);

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
  }

  // ============================================================== Npfs ===================================================================
  for (unsigned int p = 0; p < n_npf_; p++) {
    auto s = std::to_string(p);

    djTable->addColumn<float>("DeepJet_Npfcan_puppiw_" + s,
                              Npfcan_puppiw_nNpf[p],
                              "neutral candidate PUPPI weight for the " + s + ". npf",
                              10);
    djTable->addColumn<float>(
        "DeepJet_Npfcan_deltaR_" + s,
        Npfcan_deltaR_nNpf[p],
        "pseudoangular distance between the neutral candidate and the jet axis for the " + s + ". npf",
        10);
    djTable->addColumn<float>(
        "DeepJet_Npfcan_drminsv_" + s,
        Npfcan_drminsv_nNpf[p],
        "pseudoangular distance between the neutral candidate and the closest secondary vertex for the " + s + ". npf",
        10);
    djTable->addColumn<float>(
        "DeepJet_Npfcan_HadFrac_" + s,
        Npfcan_HadFrac_nNpf[p],
        "fraction of the neutral candidate energy deposited in the hadronic calorimeter for the " + s + ". npf",
        10);
    djTable->addColumn<float>("DeepJet_Npfcan_ptrel_" + s,
                              Npfcan_ptrel_nNpf[p],
                              "fraction of the jet momentum carried by the neutral candidate for the " + s + ". npf",
                              10);
    djTable->addColumn<int>("DeepJet_Npfcan_isGamma_" + s,
                            Npfcan_isGamma_nNpf[p],
                            "integer flag indicating whether the neutral candidate is a photon for the " + s + ". npf",
                            10);
    /*
      // only after dataformat updated as well
      djTable->addColumn<float>("DeepJetExtra_Npfcan_etarel_" + s,
                                Npfcan_etarel_nNpf[p], 
                                "pseudorapidity relative to parent jet for the " + s + ". npf",
                                10);
      djTable->addColumn<float>("DeepJetExtra_Npfcan_phirel_" + s,
                                Npfcan_phirel_nNpf[p],
                                "DeltaPhi(npf, jet) for the " + s + ". npf",
                                10);
      */
  }

  // ============================================================== SVs ===================================================================
  for (unsigned int p = 0; p < n_sv_; p++) {
    auto s = std::to_string(p);

    djTable->addColumn<float>("DeepJet_sv_mass_" + s, sv_mass_nSV[p], "SV mass of the " + s + ". SV", 10);
    djTable->addColumn<float>("DeepJet_sv_pt_" + s, sv_pt_nSV[p], "SV pt of the " + s + ". SV", 10);
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
    /*
      // only after dataformat updated as well
      djTable->addColumn<float>("DeepJetExtra_sv_etarel_" + s,
                                sv_etarel_nSV[p], 
                                "pseudorapidity relative to parent jet for the " + s + ". SV",
                                10);
      djTable->addColumn<float>("DeepJetExtra_sv_phirel_" + s,
                                sv_phirel_nSV[p],
                                "DeltaPhi(sv, jet) for the " + s + ". SV",
                                10);
      */
    djTable->addColumn<float>("DeepJet_sv_deltaR_" + s,
                              sv_deltaR_nSV[p],
                              "pseudoangular distance between jet axis and the " + s + ". SV direction",
                              10);
    djTable->addColumn<float>(
        "DeepJet_sv_enratio_" + s, sv_enratio_nSV[p], "ratio of the " + s + ". SV energy ratio to the jet energy", 10);
  }

  iEvent.put(std::move(djTable), nameDeepJet_);
}

template <typename T>
void DeepJetTableProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameDeepJet", "Jet");
  desc.add<std::string>("idx_nameDeepJet", "djIdx");

  desc.add<unsigned int>("n_cpf", 3);
  desc.add<unsigned int>("n_npf", 3);
  desc.add<unsigned int>("n_sv", 4);
  desc.add<edm::InputTag>("jets", edm::InputTag("slimmedJetsPuppi"));
  desc.add<edm::InputTag>("tagInfo_src", edm::InputTag("pfDeepFlavourTagInfosPuppiWithDeepInfo"));
  descriptions.addWithDefaultLabel(desc);
}

typedef DeepJetTableProducer<pat::Jet> PatJetDeepJetTableProducer;

DEFINE_FWK_MODULE(PatJetDeepJetTableProducer);
