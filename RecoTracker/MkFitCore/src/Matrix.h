#ifndef RecoTracker_MkFitCore_src_Matrix_h
#define RecoTracker_MkFitCore_src_Matrix_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/MatrixSTypes.h"
#include "RecoTracker/MkFitCore/interface/MathInlineCore.h"

//==============================================================================

// Matriplex dimensions and typedefs

// Provide fast_xyzz() Matriplex methods and operators using VDT.
#define MPLEX_VDT
// Define the following to have fast_xyzz() functions actually call std:: stuff.
// #define MPLEX_VDT_USE_STD

#include "Matriplex/MatriplexSym.h"

#ifndef MPT_SIZE
#if defined(__AVX512F__)
#define MPT_SIZE 16
#elif defined(__AVX__) || defined(__AVX2__)
#define MPT_SIZE 8
#elif defined(__SSE3__)
#define MPT_SIZE 4
#else
#define MPT_SIZE 8
#endif
#endif

namespace mkfit {

  constexpr Matriplex::idx_t NN = MPT_SIZE;  // "Length" of MPlex.

  constexpr Matriplex::idx_t LL = 6;  // Dimension of large/long  MPlex entities
  constexpr Matriplex::idx_t HH = 3;  // Dimension of small/short MPlex entities

  typedef Matriplex::Matriplex<float, LL, LL, NN> MPlexLL;
  typedef Matriplex::Matriplex<float, LL, 1, NN> MPlexLV;
  typedef Matriplex::MatriplexSym<float, LL, NN> MPlexLS;

  typedef Matriplex::Matriplex<float, HH, HH, NN> MPlexHH;
  typedef Matriplex::Matriplex<float, HH, 1, NN> MPlexHV;
  typedef Matriplex::MatriplexSym<float, HH, NN> MPlexHS;

  typedef Matriplex::Matriplex<float, 5, 1, NN> MPlex5V;
  typedef Matriplex::MatriplexSym<float, 5, NN> MPlex5S;

  typedef Matriplex::Matriplex<float, 5, 5, NN> MPlex55;
  typedef Matriplex::Matriplex<float, 5, 6, NN> MPlex56;
  typedef Matriplex::Matriplex<float, 6, 5, NN> MPlex65;

  typedef Matriplex::Matriplex<float, 2, 2, NN> MPlex22;
  typedef Matriplex::Matriplex<float, 2, 1, NN> MPlex2V;
  typedef Matriplex::MatriplexSym<float, 2, NN> MPlex2S;

  typedef Matriplex::Matriplex<float, LL, HH, NN> MPlexLH;
  typedef Matriplex::Matriplex<float, HH, LL, NN> MPlexHL;

  typedef Matriplex::Matriplex<float, 5, 2, NN> MPlex52;
  typedef Matriplex::Matriplex<float, LL, 2, NN> MPlexL2;
  typedef Matriplex::Matriplex<float, HH, 2, NN> MPlexH2;
  typedef Matriplex::Matriplex<float, 2, HH, NN> MPlex2H;

  typedef Matriplex::Matriplex<float, 3, 1, NN> MPlex3V;
  typedef Matriplex::Matriplex<float, 4, 1, NN> MPlex4V;

  typedef Matriplex::Matriplex<float, 1, 1, NN> MPlexQF;
  typedef Matriplex::Matriplex<int, 1, 1, NN> MPlexQI;
  typedef Matriplex::Matriplex<unsigned int, 1, 1, NN> MPlexQUI;
  typedef Matriplex::Matriplex<short, 1, 1, NN> MPlexQH;
  typedef Matriplex::Matriplex<unsigned short, 1, 1, NN> MPlexQUH;

  typedef Matriplex::Matriplex<bool, 1, 1, NN> MPlexQB;

  // Copies lane 0 into the lanes from n_filled up, so that a partial batch can be computed at full
  // width: a loop over the fill compiles into a vector body and a remainder that round differently,
  // so an item's result would depend on how many items its batch held. The free lanes then hold
  // valid inputs (material lookup, no NaN, no uninitialised reads); their results are not used.
  template <typename... MP>
  void pad_free_lanes(int n_filled, MP&... mp) {
    for (int i = n_filled; i < NN; ++i)
      (mp.copy(i, 0), ...);
  }

}  // end namespace mkfit

#endif
