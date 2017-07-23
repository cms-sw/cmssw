#ifndef RecoBTag_DeepFlavour_tensor_fillers_h
#define RecoBTag_DeepFlavour_tensor_fillers_h

#include "DNN/Tensorflow/interface/Tensor.h"
#include "DataFormats/DeepFormats/interface/DeepFlavourTagInfo.h"

namespace deep {

  void jet_tensor_filler(dnn::tf::Tensor * tensor,
                         std::size_t jet_n,
                         const deep::DeepFlavourFeatures & features) {

    // jet variables
    const auto & jet_features = features.jet_features;
    tensor->setValue(jet_n, 0, jet_features.pt);
    tensor->setValue(jet_n, 1, jet_features.eta);
    // number of elements in different collections
    tensor->setValue(jet_n, 2, (float) features.c_pf_features.size());
    tensor->setValue(jet_n, 3, (float) features.n_pf_features.size());
    tensor->setValue(jet_n, 4, (float) features.sv_features.size());
    tensor->setValue(jet_n, 5, (float) features.npv);
    // variables from ShallowTagInfo
    const auto & tag_info_features = features.tag_info_features;
    tensor->setValue(jet_n, 6, tag_info_features.trackSumJetEtRatio);
    tensor->setValue(jet_n, 7, tag_info_features.trackSumJetDeltaR);
    tensor->setValue(jet_n, 8, tag_info_features.vertexCategory);
    tensor->setValue(jet_n, 9, tag_info_features.trackSip2dValAboveCharm);
    tensor->setValue(jet_n, 10, tag_info_features.trackSip2dSigAboveCharm);
    tensor->setValue(jet_n, 11, tag_info_features.trackSip3dValAboveCharm);
    tensor->setValue(jet_n, 12, tag_info_features.trackSip3dSigAboveCharm);
    tensor->setValue(jet_n, 13, tag_info_features.jetNSelectedTracks);
    tensor->setValue(jet_n, 14, tag_info_features.jetNTracksEtaRel);

  } 

  void c_pf_tensor_filler(dnn::tf::Tensor * tensor,
                          std::size_t jet_n,
                          std::size_t c_pf_n,
                          const deep::ChargedCandidateFeatures & c_pf_features) {

    tensor->setValue(jet_n, c_pf_n, 0, c_pf_features.BtagPf_trackEtaRel);
    tensor->setValue(jet_n, c_pf_n, 1, c_pf_features.BtagPf_trackPtRel);
    tensor->setValue(jet_n, c_pf_n, 2, c_pf_features.BtagPf_trackPPar);
    tensor->setValue(jet_n, c_pf_n, 3, c_pf_features.BtagPf_trackDeltaR);
    tensor->setValue(jet_n, c_pf_n, 4, c_pf_features.BtagPf_trackPParRatio);
    tensor->setValue(jet_n, c_pf_n, 5, c_pf_features.BtagPf_trackSip2dVal);
    tensor->setValue(jet_n, c_pf_n, 6, c_pf_features.BtagPf_trackSip2dSig);
    tensor->setValue(jet_n, c_pf_n, 7, c_pf_features.BtagPf_trackSip3dVal);
    tensor->setValue(jet_n, c_pf_n, 8, c_pf_features.BtagPf_trackSip3dSig);
    tensor->setValue(jet_n, c_pf_n, 9, c_pf_features.BtagPf_trackJetDistVal);
    tensor->setValue(jet_n, c_pf_n, 10, c_pf_features.ptrel);
    tensor->setValue(jet_n, c_pf_n, 11, c_pf_features.drminsv);
    tensor->setValue(jet_n, c_pf_n, 12, c_pf_features.VTX_ass);
    tensor->setValue(jet_n, c_pf_n, 13, c_pf_features.puppiw);
    tensor->setValue(jet_n, c_pf_n, 14, c_pf_features.chi2);
    tensor->setValue(jet_n, c_pf_n, 15, c_pf_features.quality);

  }

  void n_pf_tensor_filler(dnn::tf::Tensor * tensor,
                          std::size_t jet_n,
                          std::size_t n_pf_n,
                          const deep::NeutralCandidateFeatures & n_pf_features) {

    tensor->setValue(jet_n, n_pf_n, 0, n_pf_features.ptrel);
    tensor->setValue(jet_n, n_pf_n, 1, n_pf_features.deltaR);
    tensor->setValue(jet_n, n_pf_n, 2, n_pf_features.isGamma);
    tensor->setValue(jet_n, n_pf_n, 3, n_pf_features.HadFrac);
    tensor->setValue(jet_n, n_pf_n, 4, n_pf_features.drminsv);
    tensor->setValue(jet_n, n_pf_n, 5, n_pf_features.puppiw);

  }

  void sv_tensor_filler(dnn::tf::Tensor * tensor,
                          std::size_t jet_n,
                          std::size_t sv_n,
                          const deep::SecondaryVertexFeatures & sv_features) {

    tensor->setValue(jet_n, sv_n, 0, sv_features.pt);
    tensor->setValue(jet_n, sv_n, 1, sv_features.deltaR);
    tensor->setValue(jet_n, sv_n, 2, sv_features.mass);
    tensor->setValue(jet_n, sv_n, 3, sv_features.ntracks);
    tensor->setValue(jet_n, sv_n, 4, sv_features.chi2);
    tensor->setValue(jet_n, sv_n, 5, sv_features.normchi2);
    tensor->setValue(jet_n, sv_n, 6, sv_features.dxy);
    tensor->setValue(jet_n, sv_n, 7, sv_features.dxysig);
    tensor->setValue(jet_n, sv_n, 8, sv_features.d3d);
    tensor->setValue(jet_n, sv_n, 9, sv_features.d3dsig);
    tensor->setValue(jet_n, sv_n, 10, sv_features.costhetasvpv);
    tensor->setValue(jet_n, sv_n, 11, sv_features.enratio);

  }


}

#endif

