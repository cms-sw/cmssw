#include "RecoBTag/FeatureTools/interface/tensor_fillers.h"
#include <cassert>

namespace btagbtvdeep {

  // Note on setting tensor values:
  // Instead of using the more convenient tensor.matrix (etc) methods,
  // we can exploit that in the following methods values are set along
  // the innermost (= last) axis. Those values are stored contiguously in
  // the memory, so it is most performant to get the pointer to the first
  // value and use pointer arithmetic to iterate through the next pointers.

  void jet_tensor_filler(float *ptr, const btagbtvdeep::DeepFlavourFeatures &features, unsigned feature_dims) {
    // jet variables
    const float *start = ptr;
    const auto &jet_features = features.jet_features;
    *(ptr++) = jet_features.pt;
    *(ptr++) = jet_features.eta;
    // number of elements in different collections
    *(ptr++) = features.c_pf_features.size();
    *(ptr++) = features.n_pf_features.size();
    *(ptr++) = features.sv_features.size();
    *(ptr++) = features.npv;
    // variables from ShallowTagInfo
    const auto &tag_info_features = features.tag_info_features;
    *(ptr++) = tag_info_features.trackSumJetEtRatio;
    *(ptr++) = tag_info_features.trackSumJetDeltaR;
    *(ptr++) = tag_info_features.vertexCategory;
    *(ptr++) = tag_info_features.trackSip2dValAboveCharm;
    *(ptr++) = tag_info_features.trackSip2dSigAboveCharm;
    *(ptr++) = tag_info_features.trackSip3dValAboveCharm;
    *(ptr++) = tag_info_features.trackSip3dSigAboveCharm;
    *(ptr++) = tag_info_features.jetNSelectedTracks;
    *(ptr++) = tag_info_features.jetNTracksEtaRel;
    assert(start + feature_dims == ptr);
  }

  void jet4vec_tensor_filler(float *ptr, const btagbtvdeep::DeepFlavourFeatures &features, unsigned feature_dims) {
    // jet 4 vector variables
    const float *start = ptr;
    const auto &jet_features = features.jet_features;
    *(ptr++) = jet_features.pt;
    *(ptr++) = jet_features.eta;
    *(ptr++) = jet_features.phi;
    *(ptr++) = jet_features.mass;
    assert(start + feature_dims == ptr);
  }

  void db_tensor_filler(float *ptr, const btagbtvdeep::DeepDoubleXFeatures &features, unsigned feature_dims) {
    // variables from BoostedDoubleSVTagInfo
    const float *start = ptr;
    const auto &tag_info_features = features.tag_info_features;
    *(ptr++) = tag_info_features.jetNTracks;
    *(ptr++) = tag_info_features.jetNSecondaryVertices;
    *(ptr++) = tag_info_features.tau1_trackEtaRel_0;
    *(ptr++) = tag_info_features.tau1_trackEtaRel_1;
    *(ptr++) = tag_info_features.tau1_trackEtaRel_2;
    *(ptr++) = tag_info_features.tau2_trackEtaRel_0;
    *(ptr++) = tag_info_features.tau2_trackEtaRel_1;
    *(ptr++) = tag_info_features.tau2_trackEtaRel_2;
    *(ptr++) = tag_info_features.tau1_flightDistance2dSig;
    *(ptr++) = tag_info_features.tau2_flightDistance2dSig;
    *(ptr++) = tag_info_features.tau1_vertexDeltaR;
    // Note: this variable is not used in the 27-input BDT
    //    *(ptr++) = tag_info_features.tau2_vertexDeltaR;
    *(ptr++) = tag_info_features.tau1_vertexEnergyRatio;
    *(ptr++) = tag_info_features.tau2_vertexEnergyRatio;
    *(ptr++) = tag_info_features.tau1_vertexMass;
    *(ptr++) = tag_info_features.tau2_vertexMass;
    *(ptr++) = tag_info_features.trackSip2dSigAboveBottom_0;
    *(ptr++) = tag_info_features.trackSip2dSigAboveBottom_1;
    *(ptr++) = tag_info_features.trackSip2dSigAboveCharm;
    *(ptr++) = tag_info_features.trackSip3dSig_0;
    *(ptr++) = tag_info_features.tau1_trackSip3dSig_0;
    *(ptr++) = tag_info_features.tau1_trackSip3dSig_1;
    *(ptr++) = tag_info_features.trackSip3dSig_1;
    *(ptr++) = tag_info_features.tau2_trackSip3dSig_0;
    *(ptr++) = tag_info_features.tau2_trackSip3dSig_1;
    *(ptr++) = tag_info_features.trackSip3dSig_2;
    *(ptr++) = tag_info_features.trackSip3dSig_3;
    *(ptr++) = tag_info_features.z_ratio;
    assert(start + feature_dims == ptr);
  }

  void c_pf_tensor_filler(float *ptr,
                          std::size_t max_c_pf_n,
                          const std::vector<btagbtvdeep::ChargedCandidateFeatures> &c_pf_features_vec,
                          unsigned feature_dims) {
    for (std::size_t i = 0; i < max_c_pf_n; ++i) {
      const auto &c_pf_features = c_pf_features_vec.at(i);
      const float *start = ptr;
      *(ptr++) = c_pf_features.btagPf_trackEtaRel;
      *(ptr++) = c_pf_features.btagPf_trackPtRel;
      *(ptr++) = c_pf_features.btagPf_trackPPar;
      *(ptr++) = c_pf_features.btagPf_trackDeltaR;
      *(ptr++) = c_pf_features.btagPf_trackPParRatio;
      *(ptr++) = c_pf_features.btagPf_trackSip2dVal;
      *(ptr++) = c_pf_features.btagPf_trackSip2dSig;
      *(ptr++) = c_pf_features.btagPf_trackSip3dVal;
      *(ptr++) = c_pf_features.btagPf_trackSip3dSig;
      *(ptr++) = c_pf_features.btagPf_trackJetDistVal;
      *(ptr++) = c_pf_features.ptrel;
      *(ptr++) = c_pf_features.drminsv;
      *(ptr++) = c_pf_features.vtx_ass;
      *(ptr++) = c_pf_features.puppiw;
      *(ptr++) = c_pf_features.chi2;
      *(ptr++) = c_pf_features.quality;
      assert(start + feature_dims == ptr);
    }
  }

  void c_pf_reduced_tensor_filler(float *ptr,
                                  std::size_t max_c_pf_n,
                                  const std::vector<btagbtvdeep::ChargedCandidateFeatures> &c_pf_features_vec,
                                  unsigned feature_dims) {
    for (std::size_t i = 0; i < max_c_pf_n; ++i) {
      const auto &c_pf_features = c_pf_features_vec.at(i);
      const float *start = ptr;
      *(ptr++) = c_pf_features.btagPf_trackEtaRel;
      *(ptr++) = c_pf_features.btagPf_trackPtRatio;
      *(ptr++) = c_pf_features.btagPf_trackPParRatio;
      *(ptr++) = c_pf_features.btagPf_trackSip2dVal;
      *(ptr++) = c_pf_features.btagPf_trackSip2dSig;
      *(ptr++) = c_pf_features.btagPf_trackSip3dVal;
      *(ptr++) = c_pf_features.btagPf_trackSip3dSig;
      *(ptr++) = c_pf_features.btagPf_trackJetDistVal;
      assert(start + feature_dims == ptr);
    }
  }

  void n_pf_tensor_filler(float *ptr,
                          std::size_t max_n_pf_n,
                          const std::vector<btagbtvdeep::NeutralCandidateFeatures> &n_pf_features_vec,
                          unsigned feature_dims) {
    for (std::size_t i = 0; i < max_n_pf_n; ++i) {
      const auto &n_pf_features = n_pf_features_vec.at(i);
      const float *start = ptr;
      *(ptr++) = n_pf_features.ptrel;
      *(ptr++) = n_pf_features.deltaR;
      *(ptr++) = n_pf_features.isGamma;
      *(ptr++) = n_pf_features.hadFrac;
      *(ptr++) = n_pf_features.drminsv;
      *(ptr++) = n_pf_features.puppiw;
      assert(start + feature_dims == ptr);
    }
  }

  void sv_tensor_filler(float *ptr,
                        std::size_t max_sv_n,
                        const std::vector<btagbtvdeep::SecondaryVertexFeatures> &sv_features_vec,
                        unsigned feature_dims) {
    for (std::size_t i = 0; i < max_sv_n; ++i) {
      const auto &sv_features = sv_features_vec.at(i);
      const float *start = ptr;
      *(ptr++) = sv_features.pt;
      *(ptr++) = sv_features.deltaR;
      *(ptr++) = sv_features.mass;
      *(ptr++) = sv_features.ntracks;
      *(ptr++) = sv_features.chi2;
      *(ptr++) = sv_features.normchi2;
      *(ptr++) = sv_features.dxy;
      *(ptr++) = sv_features.dxysig;
      *(ptr++) = sv_features.d3d;
      *(ptr++) = sv_features.d3dsig;
      *(ptr++) = sv_features.costhetasvpv;
      *(ptr++) = sv_features.enratio;
      assert(start + feature_dims == ptr);
    }
  }

  void sv_reduced_tensor_filler(float *ptr,
                                std::size_t max_sv_n,
                                const std::vector<btagbtvdeep::SecondaryVertexFeatures> &sv_features_vec,
                                unsigned feature_dims) {
    for (std::size_t i = 0; i < max_sv_n; ++i) {
      const auto &sv_features = sv_features_vec.at(i);
      const float *start = ptr;
      *(ptr++) = sv_features.d3d;
      *(ptr++) = sv_features.d3dsig;
      assert(start + feature_dims == ptr);
    }
  }

  void seed_tensor_filler(float *ptr, const btagbtvdeep::SeedingTrackFeatures &seed_features, unsigned feature_dims) {
    const float *start = ptr;
    *(ptr++) = seed_features.pt;
    *(ptr++) = seed_features.eta;
    *(ptr++) = seed_features.phi;
    *(ptr++) = seed_features.mass;
    *(ptr++) = seed_features.dz;
    *(ptr++) = seed_features.dxy;
    *(ptr++) = seed_features.ip3D;
    *(ptr++) = seed_features.sip3D;
    *(ptr++) = seed_features.ip2D;
    *(ptr++) = seed_features.sip2D;
    *(ptr++) = seed_features.signedIp3D;
    *(ptr++) = seed_features.signedSip3D;
    *(ptr++) = seed_features.signedIp2D;
    *(ptr++) = seed_features.signedSip2D;
    *(ptr++) = seed_features.trackProbability3D;
    *(ptr++) = seed_features.trackProbability2D;
    *(ptr++) = seed_features.chi2reduced;
    *(ptr++) = seed_features.nPixelHits;
    *(ptr++) = seed_features.nHits;
    *(ptr++) = seed_features.jetAxisDistance;
    *(ptr++) = seed_features.jetAxisDlength;
    assert(start + feature_dims == ptr);
  }

  void neighbourTracks_tensor_filler(float *ptr,
                                     const btagbtvdeep::SeedingTrackFeatures &seed_features,
                                     unsigned feature_dims) {
    const auto &neighbourTracks_features = seed_features.nearTracks;
    for (unsigned int t_i = 0; t_i < neighbourTracks_features.size(); t_i++) {
      const float *start = ptr;
      *(ptr++) = neighbourTracks_features[t_i].pt;
      *(ptr++) = neighbourTracks_features[t_i].eta;
      *(ptr++) = neighbourTracks_features[t_i].phi;
      *(ptr++) = neighbourTracks_features[t_i].dz;
      *(ptr++) = neighbourTracks_features[t_i].dxy;
      *(ptr++) = neighbourTracks_features[t_i].mass;
      *(ptr++) = neighbourTracks_features[t_i].ip3D;
      *(ptr++) = neighbourTracks_features[t_i].sip3D;
      *(ptr++) = neighbourTracks_features[t_i].ip2D;
      *(ptr++) = neighbourTracks_features[t_i].sip2D;
      *(ptr++) = neighbourTracks_features[t_i].distPCA;
      *(ptr++) = neighbourTracks_features[t_i].dsigPCA;
      *(ptr++) = neighbourTracks_features[t_i].x_PCAonSeed;
      *(ptr++) = neighbourTracks_features[t_i].y_PCAonSeed;
      *(ptr++) = neighbourTracks_features[t_i].z_PCAonSeed;
      *(ptr++) = neighbourTracks_features[t_i].xerr_PCAonSeed;
      *(ptr++) = neighbourTracks_features[t_i].yerr_PCAonSeed;
      *(ptr++) = neighbourTracks_features[t_i].zerr_PCAonSeed;
      *(ptr++) = neighbourTracks_features[t_i].x_PCAonTrack;
      *(ptr++) = neighbourTracks_features[t_i].y_PCAonTrack;
      *(ptr++) = neighbourTracks_features[t_i].z_PCAonTrack;
      *(ptr++) = neighbourTracks_features[t_i].xerr_PCAonTrack;
      *(ptr++) = neighbourTracks_features[t_i].yerr_PCAonTrack;
      *(ptr++) = neighbourTracks_features[t_i].zerr_PCAonTrack;
      *(ptr++) = neighbourTracks_features[t_i].dotprodTrack;
      *(ptr++) = neighbourTracks_features[t_i].dotprodSeed;
      *(ptr++) = neighbourTracks_features[t_i].dotprodTrackSeed2D;
      *(ptr++) = neighbourTracks_features[t_i].dotprodTrackSeed3D;
      *(ptr++) = neighbourTracks_features[t_i].dotprodTrackSeedVectors2D;
      *(ptr++) = neighbourTracks_features[t_i].dotprodTrackSeedVectors3D;
      *(ptr++) = neighbourTracks_features[t_i].pvd_PCAonSeed;
      *(ptr++) = neighbourTracks_features[t_i].pvd_PCAonTrack;
      *(ptr++) = neighbourTracks_features[t_i].dist_PCAjetAxis;
      *(ptr++) = neighbourTracks_features[t_i].dotprod_PCAjetMomenta;
      *(ptr++) = neighbourTracks_features[t_i].deta_PCAjetDirs;
      *(ptr++) = neighbourTracks_features[t_i].dphi_PCAjetDirs;
      assert(start + feature_dims == ptr);
    }
  }

}  // namespace btagbtvdeep
