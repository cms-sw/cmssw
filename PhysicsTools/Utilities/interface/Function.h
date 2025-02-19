#ifndef PhysicsTools_Utilities_Function_h
#define PhysicsTools_Utilities_Function_h
#include "PhysicsTools/Utilities/interface/Variables.h"
#include "PhysicsTools/Utilities/interface/Expression.h"
#include <boost/mpl/for_each.hpp>

namespace funct {

  struct null_var;

  template<typename X1 = null_var, typename X2 = null_var, typename X3 = null_var> 
  struct Function {
    template<typename F> 
    Function(const F& f) : _f(f) { }
    double operator()(typename X1::type x1, 
		      typename X2::type x2, 
		      typename X3::type x3) const { 
     X1::set(x1); X2::set(x2); X3::set(x3); return _f(); 
    }
    std::ostream& print(std::ostream& cout) const { return _f.print(cout); }
  private:
    Expression _f;
  };

  template<typename X1, typename X2> 
  struct Function<X1, X2, null_var> {
    template<typename F>  
    Function(const F& f) : _f(f) { }
    double operator()(typename X1::type x1, 
		      typename X2::type x2) const { 
      X1::set(x1); X2::set(x2); return _f(); 
    }
    std::ostream& print(std::ostream& cout) const { return _f.print(cout); }
  private:
    Expression _f;
  };

  template<typename X1> 
  struct Function<X1, null_var, null_var> {
    template<typename F> 
    Function(const F& f) : _f(f) { }
    double operator()(typename X1::type x1) const { X1::set(x1); return _f(); }
    std::ostream& print(std::ostream& cout) const { return _f.print(cout); }
  private:
    Expression _f;
  };

  template<typename X1, typename X2, typename X3>
  std::ostream& operator<<(std::ostream& cout, const Function<X1, X2, X3>& f) { 
    return f.print(cout); 
  }
 
}


#endif
