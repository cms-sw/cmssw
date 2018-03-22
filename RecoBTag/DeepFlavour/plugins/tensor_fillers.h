#ifndef RecoBTag_DeepFlavour_tensor_fillers_h
#define RecoBTag_DeepFlavour_tensor_fillers_h

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"

namespace btagbtvdeep {

  // Note on setting tensor values:
  // Instead of using the more convenient tensor.matrix (etc) methods,
  // we can exploit that in the following methods values are set along
  // the innermost (= last) axis. Those values are stored contiguously in
  // the memory, so it is most performant to get the pointer to the first
  // value and use pointer arithmetic to iterate through the next pointers.

  void jet_tensor_filler(tensorflow::Tensor & tensor,
                         std::size_t jet_n,
                         const btagbtvdeep::DeepFlavourFeatures & features) {

    float* ptr = &tensor.matrix<float>()(jet_n, 0);

    // jet variables
    const auto & jet_features = features.jet_features;
    *ptr     = jet_features.pt;
    *(++ptr) = jet_features.eta;
    // number of elements in different collections
    *(++ptr) = features.c_pf_features.size();
    *(++ptr) = features.n_pf_features.size();
    *(++ptr) = features.sv_features.size();
    *(++ptr) = features.npv;
    // variables from ShallowTagInfo
    const auto & tag_info_features = features.tag_info_features;
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

  void c_pf_tensor_filler(tensorflow::Tensor & tensor,
                          std::size_t jet_n,
                          std::size_t c_pf_n,
                          const btagbtvdeep::ChargedCandidateFeatures & c_pf_features) {

    float* ptr = &tensor.tensor<float, 3>()(jet_n, c_pf_n, 0);

    *ptr     = c_pf_features.btagPf_trackEtaRel;
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

  void n_pf_tensor_filler(tensorflow::Tensor & tensor,
                          std::size_t jet_n,
                          std::size_t n_pf_n,
                          const btagbtvdeep::NeutralCandidateFeatures & n_pf_features) {

    float* ptr = &tensor.tensor<float, 3>()(jet_n, n_pf_n, 0);

    *ptr     = n_pf_features.ptrel;
    *(++ptr) = n_pf_features.deltaR;
    *(++ptr) = n_pf_features.isGamma;
    *(++ptr) = n_pf_features.hadFrac;
    *(++ptr) = n_pf_features.drminsv;
    *(++ptr) = n_pf_features.puppiw;

  }

  void sv_tensor_filler(tensorflow::Tensor & tensor,
                          std::size_t jet_n,
                          std::size_t sv_n,
                          const btagbtvdeep::SecondaryVertexFeatures & sv_features) {

    float* ptr = &tensor.tensor<float, 3>()(jet_n, sv_n, 0);

    *ptr     = sv_features.pt;
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

}

#endif
