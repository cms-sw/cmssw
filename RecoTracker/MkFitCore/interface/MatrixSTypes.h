#ifndef RecoTracker_MkFitCore_interface_MatrixSTypes_h
#define RecoTracker_MkFitCore_interface_MatrixSTypes_h

#include "Math/SMatrix.h"

namespace mkfit {

  typedef ROOT::Math::SMatrix<float, 6, 6, ROOT::Math::MatRepSym<float, 6> > SMatrixSym66;
  typedef ROOT::Math::SMatrix<float, 6> SMatrix66;
  typedef ROOT::Math::SVector<float, 6> SVector6;

  typedef ROOT::Math::SMatrix<float, 3> SMatrix33;
  typedef ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepSym<float, 3> > SMatrixSym33;
  typedef ROOT::Math::SVector<float, 3> SVector3;

  typedef ROOT::Math::SMatrix<float, 2> SMatrix22;
  typedef ROOT::Math::SMatrix<float, 2, 2, ROOT::Math::MatRepSym<float, 2> > SMatrixSym22;
  typedef ROOT::Math::SVector<float, 2> SVector2;

  typedef ROOT::Math::SMatrix<float, 3, 6> SMatrix36;
  typedef ROOT::Math::SMatrix<float, 6, 3> SMatrix63;

  typedef ROOT::Math::SMatrix<float, 2, 6> SMatrix26;
  typedef ROOT::Math::SMatrix<float, 6, 2> SMatrix62;

  template <typename Matrix>
  inline void diagonalOnly(Matrix& m) {
    for (int r = 0; r < m.kRows; r++) {
      for (int c = 0; c < m.kCols; c++) {
        if (r != c)
          m[r][c] = 0.f;
      }
    }
  }

  template <typename Matrix>
  void dumpMatrix(Matrix m) {
    for (int r = 0; r < m.kRows; ++r) {
      for (int c = 0; c < m.kCols; ++c) {
        std::cout << std::setw(12) << m.At(r, c) << " ";
      }
      std::cout << std::endl;
    }
  }

}  // namespace mkfit

#endif
