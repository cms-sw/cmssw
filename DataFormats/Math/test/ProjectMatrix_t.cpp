#include "DataFormats/Math/interface/ProjectMatrix.h"
#include <functional>

int main() {
  typedef double T;
  typedef ROOT::Math::SMatrix<T, 5, 5, ROOT::Math::MatRepSym<T, 5> > SMat55;
  typedef ROOT::Math::SMatrix<T, 2, 2, ROOT::Math::MatRepSym<T, 2> > SMatDD;
  typedef ROOT::Math::SMatrix<T, 5, 5> SMatNN;
  typedef ROOT::Math::SMatrix<T, 5, 2> SMatND;
  typedef ROOT::Math::SMatrix<T, 2, 5> SMatDN;

  double v[3] = {1., -0.5, 2.};
  SMatDD S(v, 3);

  std::cout << S << std::endl;
  std::cout << std::endl;

  {
    SMatDN H;
    H(0, 3) = 1;
    H(1, 4) = 1;
    SMatND K = ROOT::Math::Transpose(H) * S;

    SMatNN V = K * H;

    std::cout << H << std::endl;
    std::cout << K << std::endl;
    std::cout << V << std::endl;
    std::cout << std::endl;
  }
  {
    ProjectMatrix<double, 5, 2> H;
    H.index[0] = 3;
    H.index[1] = 4;
    SMatND K = H.project(S);

    SMatNN V = H.project(K);

    std::cout << H.matrix() << std::endl;
    std::cout << K << std::endl;
    std::cout << V << std::endl;
    std::cout << std::endl;
  }

  {
    SMatDN HH;
    HH(0, 3) = 1;
    HH(1, 4) = 1;
    ProjectMatrix<double, 5, 2> H;
    H.fromH(HH);
    SMatND K = H.project(S);

    SMatNN V = H.project(K);

    std::cout << H.matrix() << std::endl;
    std::cout << K << std::endl;
    std::cout << V << std::endl;
    std::cout << std::endl;
  }

  return 0;
}
