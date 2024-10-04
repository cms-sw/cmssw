// -*- C++ -*-
//
// Package:    PhysicsTools/PatAlgos
// Class:      LeptonTagInfoCollectionProducer
//
// Original Author:  Sergio Sanchez Cruz
//         Created:  Mon, 15 May 2023 08:32:03 GMT
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "DataFormats/BTauReco/interface/DeepBoostedJetFeatures.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"

using namespace btagbtvdeep;

template <typename LeptonType>
class LeptonTagInfoCollectionProducer : public edm::stream::EDProducer<> {
public:
  explicit LeptonTagInfoCollectionProducer(const edm::ParameterSet& iConfig);
  ~LeptonTagInfoCollectionProducer() override {};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using LeptonTagInfoCollection = DeepBoostedJetFeaturesCollection;

  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  void fill_lepton_features(const LeptonType&, DeepBoostedJetFeatures&);
  void fill_lepton_extfeatures(const edm::RefToBase<LeptonType>&, DeepBoostedJetFeatures&, edm::Event&);
  void fill_pf_features(const LeptonType&, DeepBoostedJetFeatures&);
  void fill_sv_features(const LeptonType&, DeepBoostedJetFeatures&);

  template <typename VarType>
  using VarWithName = std::pair<std::string, StringObjectFunction<VarType, true>>;
  template <typename VarType>
  void parse_vars_into(const edm::ParameterSet& varsPSet, std::vector<std::unique_ptr<VarWithName<VarType>>>& vars) {
    for (const std::string& vname : varsPSet.getParameterNamesForType<std::string>()) {
      const std::string& func = varsPSet.getParameter<std::string>(vname);
      vars.push_back(std::make_unique<VarWithName<VarType>>(vname, StringObjectFunction<VarType, true>(func)));
    }
  }

  template <typename VarType>
  using ExtVarWithName = std::pair<std::string, edm::EDGetTokenT<edm::ValueMap<VarType>>>;
  template <typename VarType>
  void parse_extvars_into(const edm::ParameterSet& varsPSet,
                          std::vector<std::unique_ptr<ExtVarWithName<VarType>>>& vars) {
    for (const std::string& vname : varsPSet.getParameterNamesForType<edm::InputTag>()) {
      vars.push_back(std::make_unique<ExtVarWithName<VarType>>(
          vname, consumes<edm::ValueMap<VarType>>(varsPSet.getParameter<edm::InputTag>(vname))));
    }
  }

  edm::EDGetTokenT<edm::View<LeptonType>> src_token_;
  edm::EDGetTokenT<pat::PackedCandidateCollection> pf_token_;
  edm::EDGetTokenT<reco::VertexCompositePtrCandidateCollection> sv_token_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> pv_token_;

  edm::ParameterSet lepton_varsPSet_;
  edm::ParameterSet lepton_varsExtPSet_;
  edm::ParameterSet pf_varsPSet_;
  edm::ParameterSet sv_varsPSet_;

  std::vector<std::unique_ptr<VarWithName<LeptonType>>> lepton_vars_;
  std::vector<std::unique_ptr<VarWithName<pat::PackedCandidate>>> pf_vars_;
  std::vector<std::unique_ptr<VarWithName<reco::VertexCompositePtrCandidate>>> sv_vars_;
  edm::Handle<reco::VertexCompositePtrCandidateCollection> svs_;
  edm::Handle<pat::PackedCandidateCollection> pfs_;
  edm::Handle<std::vector<reco::Vertex>> pvs_;
  std::vector<std::unique_ptr<ExtVarWithName<float>>> extLepton_vars_;
};

template <typename LeptonType>
LeptonTagInfoCollectionProducer<LeptonType>::LeptonTagInfoCollectionProducer(const edm::ParameterSet& iConfig)
    : src_token_(consumes<edm::View<LeptonType>>(iConfig.getParameter<edm::InputTag>("src"))),
      pf_token_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandidates"))),
      sv_token_(consumes<reco::VertexCompositePtrCandidateCollection>(
          iConfig.getParameter<edm::InputTag>("secondary_vertices"))),
      pv_token_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("pvSrc"))),
      lepton_varsPSet_(iConfig.getParameter<edm::ParameterSet>("leptonVars")),
      lepton_varsExtPSet_(iConfig.getParameter<edm::ParameterSet>("leptonVarsExt")),
      pf_varsPSet_(iConfig.getParameter<edm::ParameterSet>("pfVars")),
      sv_varsPSet_(iConfig.getParameter<edm::ParameterSet>("svVars")) {
  parse_vars_into(lepton_varsPSet_, lepton_vars_);
  parse_vars_into(pf_varsPSet_, pf_vars_);
  parse_vars_into(sv_varsPSet_, sv_vars_);
  parse_extvars_into(lepton_varsExtPSet_, extLepton_vars_);

  produces<LeptonTagInfoCollection>();
}

template <typename LeptonType>
void LeptonTagInfoCollectionProducer<LeptonType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("slimmedMuons"));
  desc.add<edm::InputTag>("pfCandidates", edm::InputTag("packedPFCandidates"));
  desc.add<edm::InputTag>("secondary_vertices", edm::InputTag("slimmedSecondaryVertices"));
  desc.add<edm::InputTag>("pvSrc", edm::InputTag("offlineSlimmedPrimaryVertices"));

  for (auto&& what : {"leptonVars", "pfVars", "svVars"}) {
    edm::ParameterSetDescription descNested;
    descNested.addWildcard<std::string>("*");
    desc.add<edm::ParameterSetDescription>(what, descNested);
  }

  for (auto&& what : {"leptonVarsExt"}) {
    edm::ParameterSetDescription descNested;
    descNested.addWildcard<edm::InputTag>("*");
    desc.add<edm::ParameterSetDescription>(what, descNested);
  }

  std::string modname;
  if (typeid(LeptonType) == typeid(pat::Muon))
    modname += "muon";
  else if (typeid(LeptonType) == typeid(pat::Electron))
    modname += "electron";
  modname += "TagInfos";
  descriptions.add(modname, desc);
}

template <typename LeptonType>
void LeptonTagInfoCollectionProducer<LeptonType>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto src = iEvent.getHandle(src_token_);
  iEvent.getByToken(sv_token_, svs_);
  iEvent.getByToken(pv_token_, pvs_);
  iEvent.getByToken(pf_token_, pfs_);

  auto output_info = std::make_unique<LeptonTagInfoCollection>();

  if (pvs_->empty()) {
    // produce empty TagInfos in case no primary vertex
    iEvent.put(std::move(output_info));
    return;
  }

  for (size_t ilep = 0; ilep < src->size(); ilep++) {
    const auto& lep = (*src)[ilep];
    edm::RefToBase<LeptonType> lep_ref(src, ilep);
    DeepBoostedJetFeatures features;
    fill_lepton_features(lep, features);
    fill_lepton_extfeatures(lep_ref, features, iEvent);  // fixme
    fill_pf_features(lep, features);
    fill_sv_features(lep, features);

    output_info->emplace_back(features);
  }
  iEvent.put(std::move(output_info));
}

template <typename LeptonType>
void LeptonTagInfoCollectionProducer<LeptonType>::fill_lepton_features(const LeptonType& lep,
                                                                       DeepBoostedJetFeatures& features) {
  for (auto& var : lepton_vars_) {
    features.add(var->first);
    features.reserve(var->first, 1);
    features.fill(var->first, var->second(lep));
  }
}

template <typename LeptonType>
void LeptonTagInfoCollectionProducer<LeptonType>::fill_lepton_extfeatures(const edm::RefToBase<LeptonType>& lep,
                                                                          DeepBoostedJetFeatures& features,
                                                                          edm::Event& iEvent) {
  for (auto& var : extLepton_vars_) {
    edm::Handle<edm::ValueMap<float>> vmap;
    iEvent.getByToken(var->second, vmap);

    features.add(var->first);
    features.reserve(var->first, 1);
    features.fill(var->first, (*vmap)[lep]);
  }
}

template <typename LeptonType>
void LeptonTagInfoCollectionProducer<LeptonType>::fill_pf_features(const LeptonType& lep,
                                                                   DeepBoostedJetFeatures& features) {
  pat::PackedCandidateCollection pfcands;
  for (size_t ipf = 0; ipf < pfs_->size(); ++ipf) {
    if (reco::deltaR(pfs_->at(ipf), lep) < 0.4)
      pfcands.push_back(pfs_->at(ipf));
  }

  for (auto& var : pf_vars_) {
    features.add(var->first);
    features.reserve(var->first, pfcands.size());
    for (const auto& cand : pfcands) {
      features.fill(var->first, var->second(cand));
    }
  }

  // afaik these need to be hardcoded because I cannot put userFloats to pat::packedCandidates
  features.add("PF_phi_rel");
  features.reserve("PF_phi_rel", pfcands.size());
  features.add("PF_eta_rel");
  features.reserve("PF_eta_rel", pfcands.size());
  features.add("PF_dR_lep");
  features.reserve("PF_dR_lep", pfcands.size());
  features.add("PF_pt_rel_log");
  features.reserve("PF_pt_rel_log", pfcands.size());

  for (const auto& cand : pfcands) {
    features.fill("PF_phi_rel", reco::deltaPhi(lep.phi(), cand.phi()));
    features.fill("PF_eta_rel", lep.eta() - cand.eta());
    features.fill("PF_dR_lep", reco::deltaR(lep, cand));
    features.fill("PF_pt_rel_log", log(cand.pt() / lep.pt()));
  }
}

template <typename LeptonType>
void LeptonTagInfoCollectionProducer<LeptonType>::fill_sv_features(const LeptonType& lep,
                                                                   DeepBoostedJetFeatures& features) {
  reco::VertexCompositePtrCandidateCollection selectedSVs;
  for (size_t isv = 0; isv < svs_->size(); ++isv) {
    if (reco::deltaR(lep, svs_->at(isv)) < 0.4) {
      selectedSVs.push_back(svs_->at(isv));
    }
  }

  for (auto& var : sv_vars_) {
    features.add(var->first);
    features.reserve(var->first, selectedSVs.size());
    for (auto& sv : selectedSVs)
      features.fill(var->first, var->second(sv));
  }

  // afaik these need to be hardcoded
  const auto& PV0 = pvs_->front();
  VertexDistance3D vdist;
  VertexDistanceXY vdistXY;

  features.add("SV_dlenSig");
  features.reserve("SV_dlenSig", selectedSVs.size());
  features.add("SV_dxy");
  features.reserve("SV_dxy", selectedSVs.size());
  features.add("SV_eta_rel");
  features.reserve("SV_eta_rel", selectedSVs.size());
  features.add("SV_phi_rel");
  features.reserve("SV_phi_rel", selectedSVs.size());
  features.add("SV_dR_lep");
  features.reserve("SV_dR_lep", selectedSVs.size());
  features.add("SV_pt_rel");
  features.reserve("SV_pt_rel", selectedSVs.size());
  features.add("SV_cospAngle");
  features.reserve("SV_cospAngle", selectedSVs.size());
  features.add("SV_d3d");
  features.reserve("SV_d3d", selectedSVs.size());

  for (auto& sv : selectedSVs) {
    Measurement1D dl =
        vdist.distance(PV0, VertexState(RecoVertex::convertPos(sv.position()), RecoVertex::convertError(sv.error())));
    features.fill("SV_d3d", dl.value());
    features.fill("SV_dlenSig", dl.significance());
    Measurement1D d2d =
        vdistXY.distance(PV0, VertexState(RecoVertex::convertPos(sv.position()), RecoVertex::convertError(sv.error())));
    features.fill("SV_dxy", d2d.value());
    features.fill("SV_phi_rel", reco::deltaPhi(lep.phi(), sv.phi()));
    features.fill("SV_eta_rel", lep.eta() - sv.eta());
    features.fill("SV_dR_lep", reco::deltaR(sv, lep));
    features.fill("SV_pt_rel", sv.pt() / lep.pt());
    double dx = (PV0.x() - sv.vx()), dy = (PV0.y() - sv.vy()), dz = (PV0.z() - sv.vz());
    double pdotv = (dx * sv.px() + dy * sv.py() + dz * sv.pz()) / sv.p() / sqrt(dx * dx + dy * dy + dz * dz);
    features.fill("SV_cospAngle", pdotv);
  }
}

typedef LeptonTagInfoCollectionProducer<pat::Muon> MuonTagInfoCollectionProducer;
typedef LeptonTagInfoCollectionProducer<pat::Electron> ElectronTagInfoCollectionProducer;

DEFINE_FWK_MODULE(MuonTagInfoCollectionProducer);
DEFINE_FWK_MODULE(ElectronTagInfoCollectionProducer);
