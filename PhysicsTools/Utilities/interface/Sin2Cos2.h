#ifndef PhysicsTools_Utilities_SimplifySin2Cos2_h
#define PhysicsTools_Utilities_SimplifySin2Cos2_h

#include "PhysicsTools/Utilities/interface/Sin.h"
#include "PhysicsTools/Utilities/interface/Cos.h"
#include "PhysicsTools/Utilities/interface/Power.h"
#include "PhysicsTools/Utilities/interface/Numerical.h"

#include "PhysicsTools/Utilities/interface/Simplify_begin.h"

namespace funct {

  TEMPL(T1) struct Sin2 {
    typedef POWER(SIN(A), NUM(2)) type;
  };

  TEMPL(T1) struct Cos2 {
    typedef POWER(COS(A), NUM(2)) type;
  };

}  // namespace funct

#include "PhysicsTools/Utilities/interface/Simplify_end.h"

#endif
