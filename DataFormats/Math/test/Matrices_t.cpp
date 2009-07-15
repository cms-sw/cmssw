// test of Matrices

#include "DataFormats/Math/interface/invertPosDefMatrix.h"
#include "Math/SMatrix.h"
#include "Math/SVector.h"
#include <memory>
#include "FWCore/Utilities/interface/HRRealTime.h"
#include<iostream>

typedef ROOT::Math::SMatrix<double,2,2,ROOT::Math::MatRepSym<double,2> > Matrix2;
typedef ROOT::Math::SMatrix<double,3,3,ROOT::Math::MatRepSym<double,3> > Matrix3;






int main() {

  double v[3] = {1.,-0.2,0.5};
  Matrix2 m(v,3);

  std::cout << m << std::endl;

  edm::HRTimeType s= edm::hrRealTime();
  invertPosDefMatrix(m);
  edm::HRTimeType e = edm::hrRealTime();
  std::cout << e-s << std::endl;
  
  std::cout << m << std::endl;


  return 0;


}
