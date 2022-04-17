#ifndef RecoTracker_MkFitCore_src_Matrix_h
#define RecoTracker_MkFitCore_src_Matrix_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/MatrixSTypes.h"

namespace mkfit {

  inline float hipo(float x, float y) { return std::sqrt(x * x + y * y); }

  inline float hipo_sqr(float x, float y) { return x * x + y * y; }

  inline void sincos4(const float x, float& sin, float& cos) {
    // Had this writen with explicit division by factorial.
    // The *whole* fitting test ran like 2.5% slower on MIC, sigh.

    const float x2 = x * x;
    cos = 1.f - 0.5f * x2 + 0.04166667f * x2 * x2;
    sin = x - 0.16666667f * x * x2;
  }
}  // end namespace mkfit

//==============================================================================

// Matriplex dimensions and typedefs

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

  typedef Matriplex::Matriplex<float, 2, 2, NN> MPlex22;
  typedef Matriplex::Matriplex<float, 2, 1, NN> MPlex2V;
  typedef Matriplex::MatriplexSym<float, 2, NN> MPlex2S;

  typedef Matriplex::Matriplex<float, LL, HH, NN> MPlexLH;
  typedef Matriplex::Matriplex<float, HH, LL, NN> MPlexHL;

  typedef Matriplex::Matriplex<float, LL, 2, NN> MPlexL2;

  typedef Matriplex::Matriplex<float, 1, 1, NN> MPlexQF;
  typedef Matriplex::Matriplex<int, 1, 1, NN> MPlexQI;
  typedef Matriplex::Matriplex<unsigned int, 1, 1, NN> MPlexQUI;

  typedef Matriplex::Matriplex<bool, 1, 1, NN> MPlexQB;

}  // end namespace mkfit

#endif
