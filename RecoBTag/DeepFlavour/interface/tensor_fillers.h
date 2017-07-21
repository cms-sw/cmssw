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
    tensor->setValue(jet_n, 8, tag_info_features.trackSumJetDeltaR);
    tensor->setValue(jet_n, 9, tag_info_features.vertexCategory);
    tensor->setValue(jet_n, 10, tag_info_features.trackSip2dValAboveCharm);
    tensor->setValue(jet_n, 11, tag_info_features.trackSip2dSigAboveCharm);
    tensor->setValue(jet_n, 12, tag_info_features.trackSip3dValAboveCharm);
    tensor->setValue(jet_n, 13, tag_info_features.trackSip3dSigAboveCharm);
    tensor->setValue(jet_n, 14, tag_info_features.jetNTracksEtaRel);

  } 
 


}

#endif

