#ifndef PhysicsTools_Utilities_SimplifyNumerical_h
#define PhysicsTools_Utilities_SimplifyNumerical_h
#include "PhysicsTools/Utilities/interface/Numerical.h"
#include "PhysicsTools/Utilities/interface/Operations.h"

namespace funct {

  template<int n, int m>
    struct Sum<Numerical<n>, Numerical<m> > { 
      typedef Numerical<n + m>  type; 
      inline static type combine(const Numerical<n>&, const Numerical<m>&) 
      { return num<n + m>(); }
    };

  template<int n, int m>
    struct Difference<Numerical<n>, Numerical<m> > { 
      typedef Numerical<n - m> type; 
      inline static type combine(const Numerical<n>&, const Numerical<m>&) 
      { return num<n - m>(); }
    };

template<int n>
  struct Minus<Numerical<n> > { 
    typedef Numerical<- n> type; 
    inline static type operate(const Numerical<n>&) 
    { return num<- n>(); }
  };

 template<int n, int m>
   struct Product<Numerical<n>, Numerical<m> > { 
     typedef Numerical<n * m> type; 
     inline static type combine(const Numerical<n>&, const Numerical<m>&) 
     { return num<n * m>(); }
   };

 template<int n>
  struct Ratio<Numerical<n>, Numerical<1> > { 
    typedef Numerical<n> type; 
    inline static type combine(const Numerical<n>&, const Numerical<1>&) 
    { return num<n>(); }
  };

}

#endif
