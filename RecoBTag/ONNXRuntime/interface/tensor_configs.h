#ifndef RecoBTag_ONNXRuntime_tensor_configs_h
#define RecoBTag_ONNXRuntime_tensor_configs_h

#include <array>
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
    kBegin = 0,
    kChargedCandidates = kBegin,
    kNeutralCandidates = 1,
    kVertices = 2,
    kChargedCandidates4Vec = 3,
    kNeutralCandidates4Vec = 4,
    kVertices4Vec = 5,
    kEnd = 6
  };

  inline constexpr unsigned n_cpf_accept = 25;
  inline constexpr unsigned n_npf_accept = 25;
  inline constexpr unsigned n_sv_accept = 5;

  constexpr std::array<unsigned int, kEnd> N_InputFeatures{{
      16,  // kChargedCandidates
      8,   // kNeutralCandidates
      14,  // kVertices
      4,   // kChargedCandidates4Vec
      4,   // kNeutralCandidates4Vec
      4,   // kVertices4Vec
  }};

  constexpr std::array<unsigned int, kEnd> N_AcceptedFeatures{{
      n_cpf_accept,  // kChargedCandidates
      n_npf_accept,  // kNeutralCandidates
      n_sv_accept,   // kVertices
      n_cpf_accept,  // kChargedCandidates4Vec
      n_npf_accept,  // kNeutralCandidates4Vec
      n_sv_accept,   // kVertices4Vec
  }};

}  // namespace parT

namespace UparT {

  enum InputFeatures {
    kBegin = 0,
    kChargedCandidates = kBegin,
    kLostTracks = 1,
    kNeutralCandidates = 2,
    kVertices = 3,
    kChargedCandidates4Vec = 4,
    kLostTracks4Vec = 5,
    kNeutralCandidates4Vec = 6,
    kVertices4Vec = 7,
    kEnd = 8
  };

  inline constexpr unsigned n_cpf_accept = 29;
  inline constexpr unsigned n_lt_accept = 5;
  inline constexpr unsigned n_npf_accept = 25;
  inline constexpr unsigned n_sv_accept = 5;

  constexpr std::array<unsigned int, kEnd> N_InputFeatures{{
      25,  // kChargedCandidates
      18,  // kLostTracks
      8,   // kNeutralCandidates
      14,  // kVertices
      4,   // kChargedCandidates4Vec
      4,   // kLostTracks4Vec
      4,   // kNeutralCandidates4Vec
      4,   // kVertices4Vec
  }};

  constexpr std::array<unsigned int, kEnd> N_AcceptedFeatures{{
      n_cpf_accept,  // kChargedCandidates
      n_lt_accept,   // kLostTracks
      n_npf_accept,  // kNeutralCandidates
      n_sv_accept,   // kVertices
      n_cpf_accept,  // kChargedCandidates4Vec
      n_lt_accept,   // kLostTracks4Vec
      n_npf_accept,  // kNeutralCandidates4Vec
      n_sv_accept,   // kVertices4Vec
  }};

}  // namespace UparT
#endif
