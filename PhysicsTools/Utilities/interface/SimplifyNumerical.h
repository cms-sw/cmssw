#ifndef PhysicsTools_Utilities_SimplifyNumerical_h
#define PhysicsTools_Utilities_SimplifyNumerical_h
#include "PhysicsTools/Utilities/interface/Fraction.h"
#include "PhysicsTools/Utilities/interface/Numerical.h"
#include "PhysicsTools/Utilities/interface/Operations.h"

namespace funct {

  template<int n, int m>
    struct Sum<Numerical<n>, Numerical<m> > { 
      typedef Numerical<n + m>  type; 
      inline static type combine(const Numerical<n>&, const Numerical<m>&) 
      { return type(); }
    };

  template<int n, int m>
    struct Difference<Numerical<n>, Numerical<m> > { 
      typedef Numerical<n - m> type; 
      inline static type combine(const Numerical<n>&, const Numerical<m>&) 
      { return type(); }
    };

template<int n>
  struct Minus<Numerical<n> > { 
    typedef Numerical<- n> type; 
    inline static type operate(const Numerical<n>&) 
    { return type(); }
  };

 template<int n, int m>
   struct Product<Numerical<n>, Numerical<m> > { 
     typedef Numerical<n * m> type; 
     inline static type combine(const Numerical<n>&, const Numerical<m>&) 
     { return type(); }
   };

 template<int n>
  struct Ratio<Numerical<n>, Numerical<1> > { 
    typedef Numerical<n> type; 
    inline static type combine(const Numerical<n>&, const Numerical<1>&) 
    { return type(); }
  };

 // n ^ m = n * n ^ ( m - 1 )
 template<int n, int m, bool posM = (m > 0)>
 struct NumPower {
    typedef Numerical<n * NumPower<n, m-1>::type::value> type; 
   inline static type combine(const Numerical<n>&, const Numerical<m>&) 
   { return type(); }
  };

 // 1 ^ m = 1
 template<int m, bool posM>
 struct NumPower<1, m, posM>  {
   typedef Numerical<1> type; 
   inline static type combine(const Numerical<1>&, const Numerical<m>&) 
   { return type(); }
  };

 // n ^ 1 = n
 template<int n>
 struct NumPower<n, 1, true>  {
   typedef Numerical<n> type; 
   inline static type combine(const Numerical<n>&, const Numerical<1>&) 
   { return type(); }
  };

 // n ^ 0 = 1
 template<int n>
 struct NumPower<n, 0, true>  {
   typedef Numerical<1> type; 
   inline static type combine(const Numerical<n>&, const Numerical<0>&) 
   { return type(); }
  };

 // n ^ (-m) = 1 / n ^ m
 template<int n, int m>
 struct NumPower<n, m, false> {
    typedef typename Fraction<1, NumPower<n, -m>::type::value>::type type; 
   inline static type combine(const Numerical<n>&, const Numerical<m>&) 
   { return type(); }
  };

 template<int n, int m>
   struct Power<Numerical<n>, Numerical<m> > : public NumPower<n, m> {
 };


}

#endif
