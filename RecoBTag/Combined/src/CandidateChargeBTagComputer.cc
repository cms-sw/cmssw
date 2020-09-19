#include <memory>

#include "RecoBTag/Combined/interface/CandidateChargeBTagComputer.h"

CandidateChargeBTagComputer::Tokens::Tokens(const edm::ParameterSet& parameters, edm::ESConsumesCollector&& cc) {
  if (parameters.getParameter<bool>("useCondDB")) {
    gbrForest_ = cc.consumes(edm::ESInputTag{"", parameters.getParameter<std::string>("gbrForestLabel")});
  }
}

CandidateChargeBTagComputer::CandidateChargeBTagComputer(const edm::ParameterSet& parameters, Tokens tokens)
    : weightFile_(parameters.getParameter<edm::FileInPath>("weightFile")),
      useAdaBoost_(parameters.getParameter<bool>("useAdaBoost")),
      jetChargeExp_(parameters.getParameter<double>("jetChargeExp")),
      svChargeExp_(parameters.getParameter<double>("svChargeExp")),
      tokens_{tokens} {
  uses(0, "pfImpactParameterTagInfos");
  uses(1, "pfInclusiveSecondaryVertexFinderCvsLTagInfos");
  uses(2, "softPFMuonsTagInfos");
  uses(3, "softPFElectronsTagInfos");

  mvaID = std::make_unique<TMVAEvaluator>();
}

CandidateChargeBTagComputer::~CandidateChargeBTagComputer() {}

void CandidateChargeBTagComputer::initialize(const JetTagComputerRecord& record) {
  // Saving MVA variable names;
  // names and order need to be the same as in the training
  std::vector<std::string> variables({"tr_ch_inc",
                                      "sv_ch",
                                      "mu_sip2d",
                                      /*"mu_sip3d",*/ "mu_delR",
                                      "mu_ptrel",
                                      "mu_pfrac",
                                      "mu_char",
                                      /*"el_sip2d","el_sip3d",*/ "el_delR",
                                      "el_ptrel",
                                      "el_pfrac",
                                      "el_mva",
                                      "el_char",
                                      "pt1_ch/j_pt",
                                      "pt2_ch/j_pt",
                                      "pt3_ch/j_pt"});
  std::vector<std::string> spectators(0);

  if (tokens_.gbrForest_.isInitialized()) {
    mvaID->initializeGBRForest(&record.get(tokens_.gbrForest_), variables, spectators, useAdaBoost_);
  } else
    mvaID->initialize("Color:Silent:Error", "BDT", weightFile_.fullPath(), variables, spectators, true, useAdaBoost_);
}

float CandidateChargeBTagComputer::discriminator(const TagInfoHelper& tagInfo) const {
  // get TagInfos
  const reco::CandIPTagInfo& ip_info = tagInfo.get<reco::CandIPTagInfo>(0);
  const reco::CandSecondaryVertexTagInfo& sv_info = tagInfo.get<reco::CandSecondaryVertexTagInfo>(1);
  const reco::CandSoftLeptonTagInfo& sm_info = tagInfo.get<reco::CandSoftLeptonTagInfo>(2);
  const reco::CandSoftLeptonTagInfo& se_info = tagInfo.get<reco::CandSoftLeptonTagInfo>(3);

  size_t n_ip_info = ip_info.jet()->getJetConstituents().size();
  size_t n_sv_info = sv_info.nVertices();
  size_t n_sm_info = sm_info.leptons();
  size_t n_se_info = se_info.leptons();

  // default discriminator value
  float value = -10.;

  // if no tag info is present, MVA not computed and default discriminator value returned
  if (n_ip_info + n_sv_info + n_sm_info + n_se_info > 0) {
    // default variable values
    float tr_ch_inc = 0;
    float pt_ratio1_ch = 0;
    float pt_ratio2_ch = 0;
    float pt_ratio3_ch = 0;

    float sv_ch = 0;

    float mu_sip2d = 0;
    //float mu_sip3d = 0;
    float mu_delR = 0;
    float mu_ptrel = 0;
    float mu_pfrac = 0;
    int mu_char = 0;

    //float el_sip2d = 0;
    //float el_sip3d = 0;
    float el_delR = 0;
    float el_ptrel = 0;
    float el_pfrac = 0;
    float el_mva = 0;
    int el_char = 0;

    // compute jet-charge
    float tr_ch_num = 0;
    float tr_ch_den = 0;

    // loop over tracks associated to the jet
    for (size_t i_ip = 0; i_ip < n_ip_info; ++i_ip) {
      const reco::Candidate* trackPtr = ip_info.jet()->getJetConstituents().at(i_ip).get();
      if (trackPtr->charge() == 0)
        continue;

      float tr_ch_weight = pow(trackPtr->pt(), jetChargeExp_);
      tr_ch_num += tr_ch_weight * trackPtr->charge();
      tr_ch_den += tr_ch_weight;

      // find the three higher-pt tracks
      if (fabs(pt_ratio1_ch) < trackPtr->pt()) {
        pt_ratio3_ch = pt_ratio2_ch;
        pt_ratio2_ch = pt_ratio1_ch;
        pt_ratio1_ch = trackPtr->pt() * trackPtr->charge();
      } else if (fabs(pt_ratio2_ch) < trackPtr->pt()) {
        pt_ratio3_ch = pt_ratio2_ch;
        pt_ratio2_ch = trackPtr->pt() * trackPtr->charge();
      } else if (fabs(pt_ratio3_ch) < trackPtr->pt()) {
        pt_ratio3_ch = trackPtr->pt() * trackPtr->charge();
      }
    }
    if (n_ip_info > 0) {
      float jet_pt = ip_info.jet()->pt();
      if (jet_pt > 0) {
        pt_ratio1_ch = pt_ratio1_ch / jet_pt;
        pt_ratio2_ch = pt_ratio2_ch / jet_pt;
        pt_ratio3_ch = pt_ratio3_ch / jet_pt;
      }
    }

    if (tr_ch_den > 0)
      tr_ch_inc = tr_ch_num / tr_ch_den;

    // compute secondary vertex charge
    if (n_sv_info > 0) {
      float jet_pt = sv_info.jet()->pt();

      float sv_ch_num = 0;
      float sv_ch_den = 0;

      // find the selected secondary vertex with highest invariant mass
      int vtx_idx = -1;
      float max_mass = 0;
      for (size_t i_vtx = 0; i_vtx < n_sv_info; ++i_vtx) {
        float sv_mass = sv_info.secondaryVertex(i_vtx).p4().mass();
        float sv_chi2 = sv_info.secondaryVertex(i_vtx).vertexNormalizedChi2();
        float sv_pfrac = sv_info.secondaryVertex(i_vtx).pt() / jet_pt;
        float sv_L = sv_info.flightDistance(i_vtx).value();
        float sv_sL = sv_info.flightDistance(i_vtx).significance();
        float delEta = sv_info.secondaryVertex(i_vtx).momentum().eta() - sv_info.flightDirection(i_vtx).eta();
        float delPhi = sv_info.secondaryVertex(i_vtx).momentum().phi() - sv_info.flightDirection(i_vtx).phi();
        if (fabs(delPhi) > M_PI)
          delPhi = 2 * M_PI - fabs(delPhi);
        float sv_delR = sqrt(delEta * delEta + delPhi * delPhi);

        if (sv_mass > max_mass && sv_mass > 1.4 && sv_chi2 < 3 && sv_chi2 > 0 && sv_pfrac > 0.25 && sv_L < 2.5 &&
            sv_sL > 4 && sv_delR < 0.1) {
          max_mass = sv_mass;
          vtx_idx = i_vtx;
        }
      }

      if (vtx_idx >= 0) {
        // loop over tracks associated to the vertex
        size_t n_sv_tracks = sv_info.vertexTracks(vtx_idx).size();
        for (size_t i_tr = 0; i_tr < n_sv_tracks; ++i_tr) {
          const reco::CandidatePtr trackRef = sv_info.vertexTracks(vtx_idx)[i_tr];
          const reco::Track* trackPtr = reco::btag::toTrack(trackRef);

          float sv_ch_weight = pow(trackPtr->pt(), svChargeExp_);
          sv_ch_num += sv_ch_weight * trackPtr->charge();
          sv_ch_den += sv_ch_weight;
        }

        if (sv_ch_den > 0)
          sv_ch = sv_ch_num / sv_ch_den;
      }
    }

    // fill soft muon variables
    if (n_sm_info > 0) {
      // find the muon with higher transverse momentum
      int lep_idx = 0;
      float max_pt = 0;
      for (size_t i_lep = 0; i_lep < n_sm_info; ++i_lep) {
        float lep_pt = sm_info.lepton(0)->pt();
        if (lep_pt > max_pt) {
          max_pt = lep_pt;
          lep_idx = i_lep;
        }
      }

      mu_sip2d = sm_info.properties(lep_idx).sip2d;
      //mu_sip3d = sm_info.properties(lep_idx).sip3d;
      mu_delR = sm_info.properties(lep_idx).deltaR;
      mu_ptrel = sm_info.properties(lep_idx).ptRel;
      mu_pfrac = sm_info.properties(lep_idx).ratio;
      mu_char = sm_info.properties(lep_idx).charge;
    }

    // fill soft electron variables
    if (n_se_info > 0) {
      // find the electron with higher transverse momentum
      int lep_idx = 0;
      float max_pt = 0;
      for (size_t i_lep = 0; i_lep < n_se_info; ++i_lep) {
        float lep_pt = se_info.lepton(0)->pt();
        if (lep_pt > max_pt) {
          max_pt = lep_pt;
          lep_idx = i_lep;
        }
      }

      //el_sip2d = se_info.properties(lep_idx).sip2d;
      //el_sip3d = se_info.properties(lep_idx).sip3d;
      el_delR = se_info.properties(lep_idx).deltaR;
      el_ptrel = se_info.properties(lep_idx).ptRel;
      el_pfrac = se_info.properties(lep_idx).ratio;
      el_mva = se_info.properties(lep_idx).elec_mva;
      el_char = se_info.properties(lep_idx).charge;
    }

    std::map<std::string, float> inputs;
    inputs["tr_ch_inc"] = tr_ch_inc;
    inputs["pt1_ch/j_pt"] = pt_ratio1_ch;
    inputs["pt2_ch/j_pt"] = pt_ratio2_ch;
    inputs["pt3_ch/j_pt"] = pt_ratio3_ch;
    inputs["sv_ch"] = sv_ch;
    inputs["mu_sip2d"] = mu_sip2d;
    //inputs["mu_sip3d"] = mu_sip3d;
    inputs["mu_delR"] = mu_delR;
    inputs["mu_ptrel"] = mu_ptrel;
    inputs["mu_pfrac"] = mu_pfrac;
    inputs["mu_char"] = mu_char;
    //inputs["el_sip2d"] = el_sip2d;
    //inputs["el_sip3d"] = el_sip3d;
    inputs["el_delR"] = el_delR;
    inputs["el_ptrel"] = el_ptrel;
    inputs["el_pfrac"] = el_pfrac;
    inputs["el_mva"] = el_mva;
    inputs["el_char"] = el_char;

    // evaluate the MVA
    value = mvaID->evaluate(inputs);
  }

  // return the final discriminator value
  return value;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void CandidateChargeBTagComputer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("useCondDB", false);
  desc.add<std::string>("gbrForestLabel", "");
  desc.add<edm::FileInPath>("weightFile", edm::FileInPath());
  desc.add<bool>("useAdaBoost", true);
  desc.add<double>("jetChargeExp", 0.8);
  desc.add<double>("svChargeExp", 0.5);
  descriptions.addDefault(desc);
}
