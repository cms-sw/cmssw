// -*- C++ -*-
//
// Package:    PhysicsTools/PatAlgos
// Class:      LeptonTagInfoCollectionProducer
//
/**\class LeptonTagInfoCollectionProducer LeptonTagInfoCollectionProducer.cc PhysicsTools/PatAlgos/plugins/PNETLeptonProducer.cc


*/
//
// Original Author:  Sergio Sanchez Cruz
//         Created:  Mon, 15 May 2023 08:32:03 GMT
//
//

#include "PhysicsTools/PatAlgos/interface/LeptonTagInfoCollectionProducer.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "PhysicsTools/NanoAOD/interface/MatchingUtils.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"

namespace pat {
  template <typename T>
  LeptonTagInfoCollectionProducer<T>::LeptonTagInfoCollectionProducer(const edm::ParameterSet& iConfig)
      : src_token_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("src"))),
        pf_token_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandidates"))),
        sv_token_(consumes<reco::VertexCompositePtrCandidateCollection>(
            iConfig.getParameter<edm::InputTag>("secondary_vertices"))),
        pv_token_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("pvSrc"))),
        lepton_varsPSet_(iConfig.getParameter<edm::ParameterSet>("leptonVars")),
        lepton_varsExtPSet_(iConfig.getParameter<edm::ParameterSet>("leptonVarsExt")),
        pf_varsPSet_(iConfig.getParameter<edm::ParameterSet>("pfVars")),
        sv_varsPSet_(iConfig.getParameter<edm::ParameterSet>("svVars")) {
    produces<LeptonTagInfoCollection<T>>();
    parse_vars_into(lepton_varsPSet_, lepton_vars_);
    parse_vars_into(pf_varsPSet_, pf_vars_);
    parse_vars_into(sv_varsPSet_, sv_vars_);
    parse_extvars_into(lepton_varsExtPSet_, extLepton_vars_);
  }

  template <typename T>
  void LeptonTagInfoCollectionProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    auto src = iEvent.getHandle(src_token_);
    iEvent.getByToken(sv_token_, svs_);
    iEvent.getByToken(pv_token_, pvs_);
    iEvent.getByToken(pf_token_, pfs_);

    auto output_info = std::make_unique<LeptonTagInfoCollection<T>>();

    for (size_t ilep = 0; ilep < src->size(); ilep++) {
      const auto& lep = (*src)[ilep];
      edm::RefToBase<T> lep_ref(src, ilep);
      btagbtvdeep::DeepBoostedJetFeatures features;
      fill_lepton_features(lep, features);
      fill_lepton_extfeatures(lep_ref, features, iEvent);
      fill_pf_features(lep, features);
      fill_sv_features(lep, features);

      output_info->emplace_back(features, lep_ref);
    }
    iEvent.put(std::move(output_info));
  }

  template <typename T>
  template <typename T2>
  void LeptonTagInfoCollectionProducer<T>::parse_vars_into(const edm::ParameterSet& varsPSet,
                                                           std::vector<std::unique_ptr<varWithName<T2>>>& vars) {
    for (const std::string& vname : varsPSet.getParameterNamesForType<std::string>()) {
      const std::string& func = varsPSet.getParameter<std::string>(vname);
      vars.push_back(std::make_unique<varWithName<T2>>(vname, StringObjectFunction<T2, true>(func)));
    }
  }

  template <typename T>
  template <typename T2>
  void LeptonTagInfoCollectionProducer<T>::parse_extvars_into(const edm::ParameterSet& varsPSet,
                                                              std::vector<std::unique_ptr<extVarWithName<T2>>>& vars) {
    for (const std::string& vname : varsPSet.getParameterNamesForType<edm::InputTag>()) {
      vars.push_back(std::make_unique<extVarWithName<float>>(
          vname, consumes<edm::ValueMap<float>>(varsPSet.getParameter<edm::InputTag>(vname))));
    }
  }

  template <typename T>
  void LeptonTagInfoCollectionProducer<T>::fill_lepton_features(const T& lep,
                                                                btagbtvdeep::DeepBoostedJetFeatures& features) {
    for (auto& var : lepton_vars_) {
      features.add(var->first);
      features.reserve(var->first, 1);
      features.fill(var->first, var->second(lep));
    }
  }

  template <typename T>
  void LeptonTagInfoCollectionProducer<T>::fill_lepton_extfeatures(const edm::RefToBase<T>& lep,
                                                                   btagbtvdeep::DeepBoostedJetFeatures& features,
                                                                   edm::Event& iEvent) {
    for (auto& var : extLepton_vars_) {
      edm::Handle<edm::ValueMap<float>> vmap;
      iEvent.getByToken(var->second, vmap);

      features.add(var->first);
      features.reserve(var->first, 1);
      features.fill(var->first, (*vmap)[lep]);
    }
  }

  template <typename T>
  void LeptonTagInfoCollectionProducer<T>::fill_pf_features(const T& lep,
                                                            btagbtvdeep::DeepBoostedJetFeatures& features) {
    pat::PackedCandidateCollection pfcands;
    for (size_t ipf = 0; ipf < pfs_->size(); ++ipf) {
      if (deltaR(pfs_->at(ipf), lep) < 0.4)
        pfcands.push_back(pfs_->at(ipf));
    }

    for (auto& var : pf_vars_) {
      features.add(var->first);
      features.reserve(var->first, pfcands.size());
      for (const auto& _d : pfcands) {
        features.fill(var->first, var->second(_d));
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

    for (const auto& _d : pfcands) {
      features.fill("PF_phi_rel", deltaPhi(lep.phi(), _d.phi()));
      features.fill("PF_eta_rel", lep.eta() - _d.eta());
      features.fill("PF_dR_lep", deltaR(lep, _d));
      features.fill("PF_pt_rel_log", log(_d.pt() / lep.pt()));
    }
  }

  template <typename T>
  void LeptonTagInfoCollectionProducer<T>::fill_sv_features(const T& lep,
                                                            btagbtvdeep::DeepBoostedJetFeatures& features) {
    reco::VertexCompositePtrCandidateCollection selectedSVs;
    for (size_t isv = 0; isv < svs_->size(); ++isv) {
      if (deltaR(lep, svs_->at(isv)) < 0.4) {
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
      Measurement1D d2d = vdistXY.distance(
          PV0, VertexState(RecoVertex::convertPos(sv.position()), RecoVertex::convertError(sv.error())));
      features.fill("SV_dxy", d2d.value());
      features.fill("SV_phi_rel", deltaPhi(lep.phi(), sv.phi()));
      features.fill("SV_eta_rel", lep.eta() - sv.eta());
      features.fill("SV_dR_lep", deltaR(sv, lep));
      features.fill("SV_pt_rel", sv.pt() / lep.pt());
      double dx = (PV0.x() - sv.vx()), dy = (PV0.y() - sv.vy()), dz = (PV0.z() - sv.vz());
      double pdotv = (dx * sv.px() + dy * sv.py() + dz * sv.pz()) / sv.p() / sqrt(dx * dx + dy * dy + dz * dz);
      features.fill("SV_cospAngle", pdotv);
    }
  }

  typedef LeptonTagInfoCollectionProducer<Muon> MuonInfoCollectionProducer;
  typedef LeptonTagInfoCollectionProducer<Electron> ElectronInfoCollectionProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
  DEFINE_FWK_MODULE(MuonInfoCollectionProducer);
  DEFINE_FWK_MODULE(ElectronInfoCollectionProducer);

}  // namespace pat
