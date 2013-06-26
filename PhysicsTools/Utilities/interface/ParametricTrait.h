#ifndef PhysicsTools_Utilities_ParametricTrait_h
#define PhysicsTools_Utilities_ParametricTrait_h

#include "PhysicsTools/Utilities/interface/Functions.h"
#include "PhysicsTools/Utilities/interface/Fraction.h"
#include "PhysicsTools/Utilities/interface/Operations.h"

namespace funct {

  template<typename F> 
  struct Parametric { 
    static const int value = 1; 
  };

  template<int n> 
  struct Parametric<Numerical<n> > { 
    static const int value = 0; 
  };

 template<int n, int m> 
 struct Parametric<FractionStruct<n, m> > { 
    static const int value = 0; 
 };
 
#define NON_PARAMETRIC( FUN ) \
template<> struct Parametric<FUN> { \
  static const int value = 0; \
}

#define NON_PARAMETRIC_UNARY(FUN) \
template<typename A> \
struct Parametric<FUN<A> > { \
  static const int value = Parametric<A>::value; \
}
 
 NON_PARAMETRIC_UNARY(AbsStruct);
 NON_PARAMETRIC_UNARY(SgnStruct);
 NON_PARAMETRIC_UNARY(ExpStruct);
 NON_PARAMETRIC_UNARY(LogStruct);
 NON_PARAMETRIC_UNARY(SinStruct);
 NON_PARAMETRIC_UNARY(CosStruct);
 NON_PARAMETRIC_UNARY(TanStruct);
 NON_PARAMETRIC_UNARY(MinusStruct);
 
#define NON_PARAMETRIC_BINARY(FUN) \
template<typename A, typename B> \
struct Parametric<FUN<A, B> > { \
  static const int value = Parametric<A>::value || Parametric<A>::value; \
}

 NON_PARAMETRIC_BINARY(SumStruct);
 NON_PARAMETRIC_BINARY(ProductStruct);
 NON_PARAMETRIC_BINARY(RatioStruct);
 NON_PARAMETRIC_BINARY(PowerStruct);
 
#undef NON_PARAMETRIC_UNARY
#undef NON_PARAMETRIC_BINARY

}


#endif
