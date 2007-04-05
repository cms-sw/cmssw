#ifndef DataFormats_Math_EcalWeight
#define DataFormats_Math_EcalWeight
#include "Math/SMatrix.h"
#include "Math/SVector.h"

namespace math {
  struct EcalWeightMatrix
  {
    typedef ROOT::Math::SMatrix<double,3,10> type;
  };
  
  struct EcalChi2WeightMatrix
  {
    typedef ROOT::Math::SMatrix<double,10,10,ROOT::Math::MatRepStd<double,10,10> > type;
  };
}

#endif
