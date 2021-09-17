#ifndef RecoBTag_ONNXRuntime_tensor_configs_h
#define RecoBTag_ONNXRuntime_tensor_configs_h

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

#endif
