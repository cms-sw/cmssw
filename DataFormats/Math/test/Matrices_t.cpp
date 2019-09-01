// test of Matrices

#define SMATRIX_USE_CONSTEXPR
#include "DataFormats/Math/interface/invertPosDefMatrix.h"
#include "Math/SMatrix.h"
#include "Math/SVector.h"
#include <memory>
#include "FWCore/Utilities/interface/HRRealTime.h"
#include <iostream>

typedef ROOT::Math::SMatrix<double, 2, 2, ROOT::Math::MatRepSym<double, 2> > Matrix2;
typedef ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3> > Matrix3;

void finvert(Matrix2& mm) {
  auto m = mm.Array();

  auto c0 = 1 / m[0];
  auto c1 = m[1] * m[1] * c0;
  auto c2 = 1 / (m[2] - c1);

  auto li21 = c1 * c0 * c2;
  m[0] = li21 + c0;
  m[1] = -m[1] * c0 * c2;
  m[2] = c2;
}

void st() {}
void en() {}

int main(int argc, char* argv[]) {
  double v[3] = {1., -0.2, 0.5};
  Matrix2& m = *(new Matrix2(v, 3));

  std::cout << m << std::endl;
  invertPosDefMatrix(m);
  std::cout << m << std::endl;
  invertPosDefMatrix(m);
  std::cout << m << std::endl;
  m.Invert();
  std::cout << m << std::endl;
  m.Invert();
  std::cout << m << std::endl;

  finvert(m);
  std::cout << m << std::endl;
  finvert(m);
  std::cout << m << std::endl;

  if (argc > 1) {
    {
      edm::HRTimeType s = edm::hrRealTime();
      st();
      invertPosDefMatrix(m);
      en();
      edm::HRTimeType e = edm::hrRealTime();
      std::cout << e - s << std::endl;
    }

    {
      edm::HRTimeType s = edm::hrRealTime();
      st();
      invertPosDefMatrix(m);
      en();
      edm::HRTimeType e = edm::hrRealTime();
      std::cout << e - s << std::endl;
    }
  } else {
    {
      edm::HRTimeType s = edm::hrRealTime();
      st();
      m.Invert();
      en();
      edm::HRTimeType e = edm::hrRealTime();
      std::cout << e - s << std::endl;
    }

    {
      edm::HRTimeType s = edm::hrRealTime();
      st();
      m.Invert();
      en();
      edm::HRTimeType e = edm::hrRealTime();
      std::cout << e - s << std::endl;
    }
  }
  return 0;
}
