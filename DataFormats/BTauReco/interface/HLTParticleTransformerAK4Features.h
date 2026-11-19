#ifndef DataFormats_BTauReco_HLTParticleTransformerAK4Features_h
#define DataFormats_BTauReco_HLTParticleTransformerAK4Features_h

#include <vector>

class HLTGlobalFeatures {
public:
  float jet_pt;
  float jet_eta;
  float jet_phi;
  float jet_energy;
};

class HLTCpfCandidateFeatures {
public:
  float jet_pfcand_deta;
  float jet_pfcand_dphi;
  float jet_pfcand_pt_log;
  float jet_pfcand_energy_log;
  float jet_pfcand_charge;
  float jet_pfcand_frompv;
  float jet_pfcand_nlostinnerhits;
  float jet_pfcand_track_chi2;
  float jet_pfcand_track_qual;
  float jet_pfcand_dz;
  float jet_pfcand_dzsig;
  float jet_pfcand_dxy;
  float jet_pfcand_dxysig;
  float jet_pfcand_etarel;
  float jet_pfcand_pperp_ratio;
  float jet_pfcand_ppara_ratio;
  float jet_pfcand_trackjet_d3d;
  float jet_pfcand_trackjet_d3dsig;
  float jet_pfcand_trackjet_dist;
  float jet_pfcand_trackjet_decayL;
  float jet_pfcand_npixhits;
  float jet_pfcand_nstriphits;
  float jet_pfcand_highpurity;
  float jet_pfcand_id;

  float jet_pfcand_pt;
  float jet_pfcand_eta;
  float jet_pfcand_phi;
  float jet_pfcand_energy;
};

class HLTVtxFeatures {
public:
  float jet_sv_deta;
  float jet_sv_dphi;
  float jet_sv_pt_log;
  float jet_sv_mass;
  float jet_sv_ntrack;
  float jet_sv_chi2;
  float jet_sv_dxy;
  float jet_sv_dxysig;
  float jet_sv_d3d;
  float jet_sv_d3dsig;
  float jet_sv_costhetasvpv;
  float jet_sv_enratio;

  float jet_sv_pt;
  float jet_sv_eta;
  float jet_sv_phi;
  float jet_sv_energy;
};

namespace btagbtvdeep {

  class HLTParticleTransformerAK4Features {
  public:
    bool is_filled = true;
    HLTGlobalFeatures global_features;
    std::vector<HLTCpfCandidateFeatures> cpf_candidates;
    std::vector<HLTVtxFeatures> vtx_features;
  };

}  // namespace btagbtvdeep

#endif
