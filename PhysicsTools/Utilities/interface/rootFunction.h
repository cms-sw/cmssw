#ifndef PhysicsTools_Utilities_rootFunction_h
#define PhysicsTools_Utilities_rootFunction_h
#include "PhysicsTools/Utilities/interface/RootFunctionHelper.h"

namespace root {
  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
  function(F& f) {
    return helper::RootFunctionHelper<F>::fun(f);
  }

  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
  function(F& f, 
	   const funct::Parameter & p0) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    helper::RootFunctionHelper<F>::addParameter(p0);
    return fun;
  }

  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    helper::RootFunctionHelper<F>::addParameter(p0);
    helper::RootFunctionHelper<F>::addParameter(p1);
    return fun;
  }

  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    helper::RootFunctionHelper<F>::addParameter(p0);
    helper::RootFunctionHelper<F>::addParameter(p1);
    helper::RootFunctionHelper<F>::addParameter(p2);
    return fun;
  }

  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2,
	     const funct::Parameter & p3) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    helper::RootFunctionHelper<F>::addParameter(p0);
    helper::RootFunctionHelper<F>::addParameter(p1);
    helper::RootFunctionHelper<F>::addParameter(p2);
    helper::RootFunctionHelper<F>::addParameter(p3);
    return fun;
  }

  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2,
	     const funct::Parameter & p3,
	     const funct::Parameter & p4) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    helper::RootFunctionHelper<F>::addParameter(p0);
    helper::RootFunctionHelper<F>::addParameter(p1);
    helper::RootFunctionHelper<F>::addParameter(p2);
    helper::RootFunctionHelper<F>::addParameter(p3);
    helper::RootFunctionHelper<F>::addParameter(p4);
    return fun;
  }
  
  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2,
	     const funct::Parameter & p3,
	     const funct::Parameter & p4, 
	     const funct::Parameter & p5) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    helper::RootFunctionHelper<F>::addParameter(p0);
    helper::RootFunctionHelper<F>::addParameter(p1);
    helper::RootFunctionHelper<F>::addParameter(p2);
    helper::RootFunctionHelper<F>::addParameter(p3);
    helper::RootFunctionHelper<F>::addParameter(p4);
    helper::RootFunctionHelper<F>::addParameter(p5);
    return fun;
  }

  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2,
	     const funct::Parameter & p3,
	     const funct::Parameter & p4, 
	     const funct::Parameter & p5, 
	     const funct::Parameter & p6) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    helper::RootFunctionHelper<F>::addParameter(p0);
    helper::RootFunctionHelper<F>::addParameter(p1);
    helper::RootFunctionHelper<F>::addParameter(p2);
    helper::RootFunctionHelper<F>::addParameter(p3);
    helper::RootFunctionHelper<F>::addParameter(p4);
    helper::RootFunctionHelper<F>::addParameter(p5);
    helper::RootFunctionHelper<F>::addParameter(p6);
    return fun;
  }
  
  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2,
	     const funct::Parameter & p3,
	     const funct::Parameter & p4, 
	     const funct::Parameter & p5, 
	     const funct::Parameter & p6, 
	     const funct::Parameter & p7) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    helper::RootFunctionHelper<F>::addParameter(p0);
    helper::RootFunctionHelper<F>::addParameter(p1);
    helper::RootFunctionHelper<F>::addParameter(p2);
    helper::RootFunctionHelper<F>::addParameter(p3);
    helper::RootFunctionHelper<F>::addParameter(p4);
    helper::RootFunctionHelper<F>::addParameter(p5);
    helper::RootFunctionHelper<F>::addParameter(p6);
    helper::RootFunctionHelper<F>::addParameter(p7);
    return fun;
  }
  
  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2,
	     const funct::Parameter & p3,
	     const funct::Parameter & p4, 
	     const funct::Parameter & p5, 
	     const funct::Parameter & p6, 
	     const funct::Parameter & p7, 
	     const funct::Parameter & p8) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    helper::RootFunctionHelper<F>::addParameter(p0);
    helper::RootFunctionHelper<F>::addParameter(p1);
    helper::RootFunctionHelper<F>::addParameter(p2);
    helper::RootFunctionHelper<F>::addParameter(p3);
    helper::RootFunctionHelper<F>::addParameter(p4);
    helper::RootFunctionHelper<F>::addParameter(p5);
    helper::RootFunctionHelper<F>::addParameter(p6);
    helper::RootFunctionHelper<F>::addParameter(p7);
    helper::RootFunctionHelper<F>::addParameter(p8);
    return fun;
  }
  
  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2,
	     const funct::Parameter & p3,
	     const funct::Parameter & p4, 
	     const funct::Parameter & p5, 
	     const funct::Parameter & p6, 
	     const funct::Parameter & p7, 
	     const funct::Parameter & p8, 
	     const funct::Parameter & p9) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    helper::RootFunctionHelper<F>::addParameter(p0);
    helper::RootFunctionHelper<F>::addParameter(p1);
    helper::RootFunctionHelper<F>::addParameter(p2);
    helper::RootFunctionHelper<F>::addParameter(p3);
    helper::RootFunctionHelper<F>::addParameter(p4);
    helper::RootFunctionHelper<F>::addParameter(p5);
    helper::RootFunctionHelper<F>::addParameter(p6);
    helper::RootFunctionHelper<F>::addParameter(p7);
    helper::RootFunctionHelper<F>::addParameter(p8);
    helper::RootFunctionHelper<F>::addParameter(p9);
    return fun;
  }
  
  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
    function(F& f, 
	     const std::vector<funct::Parameter> & pars) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    std::vector<funct::Parameter>::const_iterator i, 
      b = pars.begin(), e = pars.end();
    for(i = b; i != e; ++i)
      helper::RootFunctionHelper<F>::addParameter(*i);
    return fun;
  }
  
  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
    function(F& f, 
	     const std::vector<boost::shared_ptr<double> > & pars) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    std::vector<boost::shared_ptr<double> >::const_iterator i, 
      b = pars.begin(), e = pars.end();
    for(i = b; i != e; ++i)
      helper::RootFunctionHelper<F>::addParameter(*i);
    return fun;
  }

}

#endif
