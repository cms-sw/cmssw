#ifndef RecoBTag_ONNXRuntime_tensor_fillers_h
#define RecoBTag_ONNXRuntime_tensor_fillers_h

#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"
#include "DataFormats/BTauReco/interface/ParticleTransformerAK4Features.h"
#include "DataFormats/BTauReco/interface/UnifiedParticleTransformerAK4Features.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "RecoBTag/ONNXRuntime/interface/tensor_configs.h"

namespace btagbtvdeep {

  void jet_tensor_filler(float*& ptr, const btagbtvdeep::DeepFlavourFeatures& features);

  void cpf_tensor_filler(float*& ptr, const btagbtvdeep::ChargedCandidateFeatures& c_pf_features);

  void npf_tensor_filler(float*& ptr, const btagbtvdeep::NeutralCandidateFeatures& n_pf_features);

  void sv_tensor_filler(float*& ptr, const btagbtvdeep::SecondaryVertexFeatures& sv_features);

  void jet4vec_tensor_filler(float*& ptr, const btagbtvdeep::JetFeatures& jet_features);

  void seedTrack_tensor_filler(float*& ptr, const btagbtvdeep::SeedingTrackFeatures& seed_features);

  void neighbourTrack_tensor_filler(float*& ptr, const btagbtvdeep::TrackPairFeatures& neighbourTrack_features);

  std::vector<float> inputs_parT(const btagbtvdeep::ChargedCandidateFeatures& c_pf_features,
                                 parT::InputFeatures ifeature);

  std::vector<float> inputs_parT(const btagbtvdeep::NeutralCandidateFeatures& n_pf_features,
                                 parT::InputFeatures ifeature);

  std::vector<float> inputs_parT(const btagbtvdeep::SecondaryVertexFeatures& sv_features, parT::InputFeatures ifeature);

  std::vector<float> inputs_UparT(const btagbtvdeep::ChargedCandidateFeatures& c_pf_features,
                                  UparT::InputFeatures ifeature);

  std::vector<float> inputs_UparT(const btagbtvdeep::LostTracksFeatures& lt_features, UparT::InputFeatures ifeature);

  std::vector<float> inputs_UparT(const btagbtvdeep::NeutralCandidateFeatures& n_pf_features,
                                  UparT::InputFeatures ifeature);

  std::vector<float> inputs_UparT(const btagbtvdeep::SecondaryVertexFeatures& sv_features,
                                  UparT::InputFeatures ifeature);

  template <class parT_features>
  void parT_tensor_filler(cms::Ort::FloatArrays& data,
                          const parT::InputFeatures ifeature,
                          const std::vector<parT_features>& features,
                          const unsigned int max_n,
                          const float*& start,
                          unsigned offset) {
    float* ptr = nullptr;
    for (std::size_t n = 0; n < max_n; n++) {
      const auto& f = features.at(n);
      ptr = &data[ifeature][offset + n * parT::N_InputFeatures.at(ifeature)];
      start = ptr;
      const std::vector<float>& inputs = inputs_parT(f, ifeature);
      for (unsigned int i = 0; i < inputs.size(); i++) {
        *ptr = inputs[i];
        ++ptr;
      }
      if (!inputs.empty())
        --ptr;
      assert(start + parT::N_InputFeatures.at(ifeature) - 1 == ptr);
    }
  }

  template <class parT_features>
  void parT_tensor_filler(std::vector<float>& vdata,
                          const parT::InputFeatures ifeature,
                          const std::vector<parT_features>& features,
                          const unsigned int target_n) {
    unsigned int n =
        std::clamp((unsigned int)features.size(), (unsigned int)0, (unsigned int)parT::N_AcceptedFeatures.at(ifeature));
    for (unsigned int count = 0; count < n; count++) {
      const std::vector<float>& inputs = inputs_parT(features.at(count), ifeature);
      vdata.insert(vdata.end(), inputs.begin(), inputs.end());
    }
    unsigned int n_features = parT::N_InputFeatures.at(ifeature);
    if (n < target_n)
      vdata.insert(vdata.end(), (target_n - n) * n_features, 0);  // Add 0 to unfilled part as padding value
  }

  template <class UparT_features>
  void UparT_tensor_filler(cms::Ort::FloatArrays& data,
                           const UparT::InputFeatures ifeature,
                           const std::vector<UparT_features>& features,
                           const unsigned int max_n,
                           const float*& start,
                           unsigned offset) {
    float* ptr = nullptr;
    for (std::size_t n = 0; n < max_n; n++) {
      const auto& f = features.at(n);
      ptr = &data[ifeature][offset + n * UparT::N_InputFeatures.at(ifeature)];
      start = ptr;
      const std::vector<float>& inputs = inputs_UparT(f, ifeature);
      for (unsigned int i = 0; i < inputs.size(); i++) {
        *ptr = inputs[i];
        ++ptr;
      }
      if (!inputs.empty())
        --ptr;
      assert(start + UparT::N_InputFeatures.at(ifeature) - 1 == ptr);
    }
  }

  template <class UparT_features>
  void UparT_tensor_filler(std::vector<float>& vdata,
                           const UparT::InputFeatures ifeature,
                           const std::vector<UparT_features>& features,
                           const unsigned int target_n) {
    unsigned int n = std::clamp(
        (unsigned int)features.size(), (unsigned int)0, (unsigned int)UparT::N_AcceptedFeatures.at(ifeature));
    for (unsigned int count = 0; count < n; count++) {
      const std::vector<float>& inputs = inputs_UparT(features.at(count), ifeature);
      vdata.insert(vdata.end(), inputs.begin(), inputs.end());
    }
    unsigned int n_features = UparT::N_InputFeatures.at(ifeature);
    if (n < target_n)
      vdata.insert(vdata.end(), (target_n - n) * n_features, 0);  // Add 0 to unfilled part as padding value
  }
}  // namespace btagbtvdeep

#endif
