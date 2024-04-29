#ifndef RecoBTag_ONNXRuntime_tensor_configs_h
#define RecoBTag_ONNXRuntime_tensor_configs_h

#include <map>
namespace deepflavour {

  constexpr unsigned n_features_global = 15;

  constexpr unsigned n_cpf = 25;
  constexpr unsigned n_features_cpf = 16;

  constexpr unsigned n_npf = 25;
  constexpr unsigned n_features_npf = 6;

  constexpr unsigned n_sv = 4;
  constexpr unsigned n_features_sv = 12;

}  // namespace deepflavour

namespace deepvertex {

  constexpr unsigned n_features_global = 4;

  constexpr unsigned n_seed = 10;
  constexpr unsigned n_features_seed = 21;

  constexpr unsigned n_neighbor = 20;
  constexpr unsigned n_features_neighbor = 36;

}  // namespace deepvertex

namespace parT {

  enum InputFeatures {
    kChargedCandidates=0,
    kNeutralCandidates=1,
    kVertices=2,
    kChargedCandidates4Vec=3,
    kNeutralCandidates4Vec=4,
    kVertices4Vec=5
  };
  
  const std::map<unsigned int, InputFeatures> InputIndexes{
    {0, kChargedCandidates},
    {1, kNeutralCandidates},
    {2, kVertices},
    {3, kChargedCandidates4Vec},
    {4, kNeutralCandidates4Vec},
    {5, kVertices4Vec}
  };

  constexpr unsigned n_cpf_accept = 25;
  constexpr unsigned n_npf_accept = 25;
  constexpr unsigned n_sv_accept = 5; 

  const std::map<InputFeatures, unsigned int> N_InputFeatures{
    {kChargedCandidates, 16},
    {kNeutralCandidates, 8},
    {kVertices, 14},
    {kChargedCandidates4Vec, 4},
    {kNeutralCandidates4Vec, 4},
    {kVertices4Vec, 4}
  };

  const std::map<InputFeatures, unsigned int> N_AcceptedFeatures{
    {kChargedCandidates, n_cpf_accept},
    {kNeutralCandidates, n_npf_accept},
    {kVertices, n_sv_accept},
    {kChargedCandidates4Vec, n_cpf_accept},
    {kNeutralCandidates4Vec, n_npf_accept},
    {kVertices4Vec, n_sv_accept}
  };

} // namespace parT

#endif
