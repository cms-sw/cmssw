#ifndef RecoBTag_FeatureTools_tensor_fillers_h
#define RecoBTag_FeatureTools_tensor_fillers_h

#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepDoubleXTagInfo.h"

namespace btagbtvdeep {

  void jet_tensor_filler(float* ptr, const btagbtvdeep::DeepFlavourFeatures& features, unsigned feature_dims);

  void jet4vec_tensor_filler(float* ptr, const btagbtvdeep::DeepFlavourFeatures& features, unsigned feature_dims);

  void db_tensor_filler(float* ptr, const btagbtvdeep::DeepDoubleXFeatures& features, unsigned feature_dims);

  void c_pf_tensor_filler(float* ptr,
                          std::size_t max_c_pf_n,
                          const std::vector<btagbtvdeep::ChargedCandidateFeatures>& c_pf_features_vec,
                          unsigned feature_dims);

  void c_pf_reduced_tensor_filler(float* ptr,
                                  std::size_t max_c_pf_n,
                                  const std::vector<btagbtvdeep::ChargedCandidateFeatures>& c_pf_features_vec,
                                  unsigned feature_dims);

  void n_pf_tensor_filler(float* ptr,
                          std::size_t max_n_pf_n,
                          const std::vector<btagbtvdeep::NeutralCandidateFeatures>& n_pf_features_vec,
                          unsigned feature_dims);

  void sv_tensor_filler(float* ptr,
                        std::size_t max_sv_n,
                        const std::vector<btagbtvdeep::SecondaryVertexFeatures>& sv_features_vec,
                        unsigned feature_dims);

  void sv_reduced_tensor_filler(float* ptr,
                                std::size_t max_sv_n,
                                const std::vector<btagbtvdeep::SecondaryVertexFeatures>& sv_features_vec,
                                unsigned feature_dims);

  void seed_tensor_filler(float* ptr, const btagbtvdeep::SeedingTrackFeatures& seed_features, unsigned feature_dims);

  void neighbourTracks_tensor_filler(float* ptr,
                                     const btagbtvdeep::SeedingTrackFeatures& seed_features,
                                     unsigned feature_dims);

}  // namespace btagbtvdeep

#endif
