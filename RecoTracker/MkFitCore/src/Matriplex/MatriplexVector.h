#ifndef RecoTracker_MkFitCore_src_Matriplex_MatriplexVector_h
#define RecoTracker_MkFitCore_src_Matriplex_MatriplexVector_h

#include "Matriplex.h"

#include <vector>
#include <cassert>

namespace Matriplex {

  //------------------------------------------------------------------------------

  template <class MP>
  class MatriplexVector {
    MP* fV;
    const idx_t fN;

    typedef typename MP::value_type T;

  public:
    MatriplexVector(idx_t n) : fN(n) { fV = (MP*)std::aligned_alloc(64, sizeof(MP) * fN); }

    ~MatriplexVector() { std::free(fV); }

    idx_t size() const { return fN; }

    MP& mplex(int i) { return fV[i]; }
    MP& operator[](int i) { return fV[i]; }

    const MP& mplex(int i) const { return fV[i]; }
    const MP& operator[](int i) const { return fV[i]; }

    void setVal(T v) {
      for (idx_t i = 0; i < kTotSize; ++i) {
        fArray[i] = v;
      }
    }

    T& At(idx_t n, idx_t i, idx_t j) { return fV[n / fN].At(n % fN, i, j); }

    T& operator()(idx_t n, idx_t i, idx_t j) { return fV[n / fN].At(n % fN, i, j); }

    void copyIn(idx_t n, T* arr) { fV[n / fN].copyIn(n % fN, arr); }
    void copyOut(idx_t n, T* arr) { fV[n / fN].copyOut(n % fN, arr); }
  };

  template <class MP>
  using MPlexVec = MatriplexVector<MP>;

  //==============================================================================

  template <typename T, idx_t D1, idx_t D2, idx_t D3, idx_t N>
  void multiply(const MPlexVec<MPlex<T, D1, D2, N>>& A,
                const MPlexVec<MPlex<T, D2, D3, N>>& B,
                MPlexVec<MPlex<T, D1, D3, N>>& C,
                int n_to_process = 0) {
    assert(A.size() == B.size());
    assert(A.size() == C.size());

    const int np = n_to_process ? n_to_process : A.size();

    for (int i = 0; i < np; ++i) {
      multiply(A[i], B[i], C[i]);
    }
  }

  template <typename T, idx_t D1, idx_t D2, idx_t D3, idx_t N>
  void multiplyGeneral(const MPlexVec<MPlex<T, D1, D2, N>>& A,
                       const MPlexVec<MPlex<T, D2, D3, N>>& B,
                       MPlexVec<MPlex<T, D1, D3, N>>& C,
                       int n_to_process = 0) {
    assert(A.size() == B.size());
    assert(A.size() == C.size());

    const int np = n_to_process ? n_to_process : A.size();

    for (int i = 0; i < np; ++i) {
      multiplyGeneral(A[i], B[i], C[i]);
    }
  }

  template <typename T, idx_t D1, idx_t D2, idx_t D3, idx_t N>
  void multiply3in(MPlexVec<MPlex<T, D1, D2, N>>& A,
                   MPlexVec<MPlex<T, D2, D3, N>>& B,
                   MPlexVec<MPlex<T, D1, D3, N>>& C,
                   int n_to_process = 0) {
    assert(A.size() == B.size());
    assert(A.size() == C.size());

    const int np = n_to_process ? n_to_process : A.size();

    for (int i = 0; i < np; ++i) {
      multiply(A[i], B[i], C[i]);
      multiply(B[i], C[i], A[i]);
      multiply(C[i], A[i], B[i]);
    }
  }

  template <typename T, idx_t D, idx_t N>
  void multiply(const MPlexVec<MPlexSym<T, D, N>>& A,
                const MPlexVec<MPlexSym<T, D, N>>& B,
                MPlexVec<MPlex<T, D, D, N>>& C,
                int n_to_process = 0) {
    assert(A.size() == B.size());
    assert(A.size() == C.size());

    const int np = n_to_process ? n_to_process : A.size();

    for (int i = 0; i < np; ++i) {
      multiply(A[i], B[i], C[i]);
    }
  }

  //==============================================================================

  template <typename T, idx_t D, idx_t N>
  void invertCramer(MPlexVec<MPlex<T, D, D, N>>& A, int n_to_process = 0) {
    const int np = n_to_process ? n_to_process : A.size();

    for (int i = 0; i < np; ++i) {
      invertCramer(A[i]);
    }
  }

  template <typename T, idx_t D, idx_t N>
  void invertCholesky(MPlexVec<MPlex<T, D, D, N>>& A, int n_to_process = 0) {
    const int np = n_to_process ? n_to_process : A.size();

    for (int i = 0; i < np; ++i) {
      invertCholesky(A[i]);
    }
  }

  template <typename T, idx_t D, idx_t N>
  void invertCramerSym(MPlexVec<MPlexSym<T, D, N>>& A, int n_to_process = 0) {
    const int np = n_to_process ? n_to_process : A.size();

    for (int i = 0; i < np; ++i) {
      invertCramerSym(A[i]);
    }
  }

  template <typename T, idx_t D, idx_t N>
  void invertCholeskySym(MPlexVec<MPlexSym<T, D, N>>& A, int n_to_process = 0) {
    const int np = n_to_process ? n_to_process : A.size();

    for (int i = 0; i < np; ++i) {
      invertCholeskySym(A[i]);
    }
  }

}  // namespace Matriplex

#endif
