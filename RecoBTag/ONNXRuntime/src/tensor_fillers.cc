#include "RecoBTag/ONNXRuntime/interface/tensor_fillers.h"

namespace btagbtvdeep {

  void jet_tensor_filler(float*& ptr, const btagbtvdeep::DeepFlavourFeatures& features) {
    // jet variables
    const auto& jet_features = features.jet_features;
    *ptr = jet_features.pt;
    *(++ptr) = jet_features.eta;
    // number of elements in different collections
    *(++ptr) = features.c_pf_features.size();
    *(++ptr) = features.n_pf_features.size();
    *(++ptr) = features.sv_features.size();
    *(++ptr) = features.npv;
    // variables from ShallowTagInfo
    const auto& tag_info_features = features.tag_info_features;
    *(++ptr) = tag_info_features.trackSumJetEtRatio;
    *(++ptr) = tag_info_features.trackSumJetDeltaR;
    *(++ptr) = tag_info_features.vertexCategory;
    *(++ptr) = tag_info_features.trackSip2dValAboveCharm;
    *(++ptr) = tag_info_features.trackSip2dSigAboveCharm;
    *(++ptr) = tag_info_features.trackSip3dValAboveCharm;
    *(++ptr) = tag_info_features.trackSip3dSigAboveCharm;
    *(++ptr) = tag_info_features.jetNSelectedTracks;
    *(++ptr) = tag_info_features.jetNTracksEtaRel;
  }

  void cpf_tensor_filler(float*& ptr, const btagbtvdeep::ChargedCandidateFeatures& c_pf_features) {
    *ptr = c_pf_features.btagPf_trackEtaRel;
    *(++ptr) = c_pf_features.btagPf_trackPtRel;
    *(++ptr) = c_pf_features.btagPf_trackPPar;
    *(++ptr) = c_pf_features.btagPf_trackDeltaR;
    *(++ptr) = c_pf_features.btagPf_trackPParRatio;
    *(++ptr) = c_pf_features.btagPf_trackSip2dVal;
    *(++ptr) = c_pf_features.btagPf_trackSip2dSig;
    *(++ptr) = c_pf_features.btagPf_trackSip3dVal;
    *(++ptr) = c_pf_features.btagPf_trackSip3dSig;
    *(++ptr) = c_pf_features.btagPf_trackJetDistVal;
    *(++ptr) = c_pf_features.ptrel;
    *(++ptr) = c_pf_features.drminsv;
    *(++ptr) = c_pf_features.vtx_ass;
    *(++ptr) = c_pf_features.puppiw;
    *(++ptr) = c_pf_features.chi2;
    *(++ptr) = c_pf_features.quality;
  }

  void npf_tensor_filler(float*& ptr, const btagbtvdeep::NeutralCandidateFeatures& n_pf_features) {
    *ptr = n_pf_features.ptrel;
    *(++ptr) = n_pf_features.deltaR;
    *(++ptr) = n_pf_features.isGamma;
    *(++ptr) = n_pf_features.hadFrac;
    *(++ptr) = n_pf_features.drminsv;
    *(++ptr) = n_pf_features.puppiw;
  }

  void sv_tensor_filler(float*& ptr, const btagbtvdeep::SecondaryVertexFeatures& sv_features) {
    *ptr = sv_features.pt;
    *(++ptr) = sv_features.deltaR;
    *(++ptr) = sv_features.mass;
    *(++ptr) = sv_features.ntracks;
    *(++ptr) = sv_features.chi2;
    *(++ptr) = sv_features.normchi2;
    *(++ptr) = sv_features.dxy;
    *(++ptr) = sv_features.dxysig;
    *(++ptr) = sv_features.d3d;
    *(++ptr) = sv_features.d3dsig;
    *(++ptr) = sv_features.costhetasvpv;
    *(++ptr) = sv_features.enratio;
  }

  void jet4vec_tensor_filler(float*& ptr, const btagbtvdeep::JetFeatures& jet_features) {
    *ptr = jet_features.pt;
    *(++ptr) = jet_features.eta;
    *(++ptr) = jet_features.phi;
    *(++ptr) = jet_features.mass;
  }

  void seedTrack_tensor_filler(float*& ptr, const btagbtvdeep::SeedingTrackFeatures& seed_features) {
    *ptr = seed_features.pt;
    *(++ptr) = seed_features.eta;
    *(++ptr) = seed_features.phi;
    *(++ptr) = seed_features.mass;
    *(++ptr) = seed_features.dz;
    *(++ptr) = seed_features.dxy;
    *(++ptr) = seed_features.ip3D;
    *(++ptr) = seed_features.sip3D;
    *(++ptr) = seed_features.ip2D;
    *(++ptr) = seed_features.sip2D;
    *(++ptr) = seed_features.signedIp3D;
    *(++ptr) = seed_features.signedSip3D;
    *(++ptr) = seed_features.signedIp2D;
    *(++ptr) = seed_features.signedSip2D;
    *(++ptr) = seed_features.trackProbability3D;
    *(++ptr) = seed_features.trackProbability2D;
    *(++ptr) = seed_features.chi2reduced;
    *(++ptr) = seed_features.nPixelHits;
    *(++ptr) = seed_features.nHits;
    *(++ptr) = seed_features.jetAxisDistance;
    *(++ptr) = seed_features.jetAxisDlength;
  }

  void neighbourTrack_tensor_filler(float*& ptr, const btagbtvdeep::TrackPairFeatures& neighbourTrack_features) {
    *ptr = neighbourTrack_features.pt;
    *(++ptr) = neighbourTrack_features.eta;
    *(++ptr) = neighbourTrack_features.phi;
    *(++ptr) = neighbourTrack_features.dz;
    *(++ptr) = neighbourTrack_features.dxy;
    *(++ptr) = neighbourTrack_features.mass;
    *(++ptr) = neighbourTrack_features.ip3D;
    *(++ptr) = neighbourTrack_features.sip3D;
    *(++ptr) = neighbourTrack_features.ip2D;
    *(++ptr) = neighbourTrack_features.sip2D;
    *(++ptr) = neighbourTrack_features.distPCA;
    *(++ptr) = neighbourTrack_features.dsigPCA;
    *(++ptr) = neighbourTrack_features.x_PCAonSeed;
    *(++ptr) = neighbourTrack_features.y_PCAonSeed;
    *(++ptr) = neighbourTrack_features.z_PCAonSeed;
    *(++ptr) = neighbourTrack_features.xerr_PCAonSeed;
    *(++ptr) = neighbourTrack_features.yerr_PCAonSeed;
    *(++ptr) = neighbourTrack_features.zerr_PCAonSeed;
    *(++ptr) = neighbourTrack_features.x_PCAonTrack;
    *(++ptr) = neighbourTrack_features.y_PCAonTrack;
    *(++ptr) = neighbourTrack_features.z_PCAonTrack;
    *(++ptr) = neighbourTrack_features.xerr_PCAonTrack;
    *(++ptr) = neighbourTrack_features.yerr_PCAonTrack;
    *(++ptr) = neighbourTrack_features.zerr_PCAonTrack;
    *(++ptr) = neighbourTrack_features.dotprodTrack;
    *(++ptr) = neighbourTrack_features.dotprodSeed;
    *(++ptr) = neighbourTrack_features.dotprodTrackSeed2D;
    *(++ptr) = neighbourTrack_features.dotprodTrackSeed3D;
    *(++ptr) = neighbourTrack_features.dotprodTrackSeedVectors2D;
    *(++ptr) = neighbourTrack_features.dotprodTrackSeedVectors3D;
    *(++ptr) = neighbourTrack_features.pvd_PCAonSeed;
    *(++ptr) = neighbourTrack_features.pvd_PCAonTrack;
    *(++ptr) = neighbourTrack_features.dist_PCAjetAxis;
    *(++ptr) = neighbourTrack_features.dotprod_PCAjetMomenta;
    *(++ptr) = neighbourTrack_features.deta_PCAjetDirs;
    *(++ptr) = neighbourTrack_features.dphi_PCAjetDirs;
  }

}  // namespace btagbtvdeep
