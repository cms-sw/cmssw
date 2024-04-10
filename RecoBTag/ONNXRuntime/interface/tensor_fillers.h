#ifndef RecoBTag_ONNXRuntime_tensor_fillers_h
#define RecoBTag_ONNXRuntime_tensor_fillers_h

#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"
#include "RecoBTag/ONNXRuntime/interface/tensor_configs.h"

namespace btagbtvdeep {

  void jet_tensor_filler(float*& ptr, const btagbtvdeep::DeepFlavourFeatures& features);

  void cpf_tensor_filler(float*& ptr, const btagbtvdeep::ChargedCandidateFeatures& c_pf_features);

  void npf_tensor_filler(float*& ptr, const btagbtvdeep::NeutralCandidateFeatures& n_pf_features);

  void sv_tensor_filler(float*& ptr, const btagbtvdeep::SecondaryVertexFeatures& sv_features);

  void jet4vec_tensor_filler(float*& ptr, const btagbtvdeep::JetFeatures& jet_features);

  void seedTrack_tensor_filler(float*& ptr, const btagbtvdeep::SeedingTrackFeatures& seed_features);

  void neighbourTrack_tensor_filler(float*& ptr, const btagbtvdeep::TrackPairFeatures& neighbourTrack_features);
 
  std::vector<float> inputs_parT(const btagbtvdeep::ChargedCandidateFeatures& c_pf_features, parT::InputIndexes idx);

  std::vector<float> inputs_parT(const btagbtvdeep::NeutralCandidateFeatures& n_pf_features, parT::InputIndexes idx);

  std::vector<float> inputs_parT(const btagbtvdeep::SecondaryVertexFeatures& sv_features, parT::InputIndexes idx);

  template<class parT_features>
  void parT_tensor_filler(float*& ptr, parT::InputIndexes idx , const parT_features pf) {
    std::vector<float> inputs;
    inputs = inputs_parT(pf, idx);
    for (unsigned int i = 0; i < inputs.size(); i++) {
      *ptr = inputs[i];
      ++ptr;
    }
    if (inputs.size() > 0) --ptr;
  }

  template<class parT_features>
  void parT_tensor_filler(std::vector<float>& vdata, parT::InputIndexes idx , const parT_features pf) {
    std::vector<float> inputs;
    inputs = inputs_parT(pf, idx);
    vdata.insert(vdata.end(), inputs.begin(), inputs.end());
  }

}  // namespace btagbtvdeep

#endif
