#include <cmath>
namespace {
 // not a generic solution (wrong for N negative for instance)
  template<int N> 
    struct PowN {
      template<typename T>
      static T op(T t) { return PowN<N/2>::op(t)*PowN<(N+1)/2>::op(t);}
    };
  template<> 
    struct PowN<0> {
    template<typename T>
      static T op(T t) { return T(1);}
  };
  template<>
    struct PowN<1> {
    template<typename T>
      static T op(T t) { return t;}
  };
  template<>
    struct PowN<2> {
    template<typename T>
      static T op(T t) { return t*t;}
  };
  
  template<typename T>
    T powN(T t, int n) {
    switch(n) {
    case 4: return PowN<4>::op(t); // the only one that matters
    case 3: return PowN<3>::op(t); // and this
    case 8: return PowN<8>::op(t); // used in conversion????
    case 2: return PowN<2>::op(t);
    case 5: return PowN<5>::op(t);
    case 6: return PowN<6>::op(t);
    case 7: return PowN<7>::op(t);
    case 0: return PowN<0>::op(t);
    case 1: return PowN<1>::op(t);
    default : return std::pow(t,T(n)); 
    }
  }
}

