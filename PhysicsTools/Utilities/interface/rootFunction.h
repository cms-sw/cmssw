#ifndef PhysicsTools_Utilities_rootFunction_h
#define PhysicsTools_Utilities_rootFunction_h
#include "PhysicsTools/Utilities/interface/RootFunctionHelper.h"

namespace root {
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
  function(F& f) {
    return helper::RootFunctionHelper<F, args>::fun(f);
  }

  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
  function(F& f, 
	   const funct::Parameter & p0) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    return fun;
  }

  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    return fun;
  }

  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    return fun;
  }

  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2,
	     const funct::Parameter & p3) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    return fun;
  }

  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2,
	     const funct::Parameter & p3,
	     const funct::Parameter & p4) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2,
	     const funct::Parameter & p3,
	     const funct::Parameter & p4, 
	     const funct::Parameter & p5) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    return fun;
  }

  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2,
	     const funct::Parameter & p3,
	     const funct::Parameter & p4, 
	     const funct::Parameter & p5, 
	     const funct::Parameter & p6) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
    function(F& f, 
	     const funct::Parameter & p0,
	     const funct::Parameter & p1,
	     const funct::Parameter & p2,
	     const funct::Parameter & p3,
	     const funct::Parameter & p4, 
	     const funct::Parameter & p5, 
	     const funct::Parameter & p6, 
	     const funct::Parameter & p7) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    helper::RootFunctionHelper<F, args>::addParameter(p7);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
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
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    helper::RootFunctionHelper<F, args>::addParameter(p7);
    helper::RootFunctionHelper<F, args>::addParameter(p8);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
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
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    helper::RootFunctionHelper<F, args>::addParameter(p7);
    helper::RootFunctionHelper<F, args>::addParameter(p8);
    helper::RootFunctionHelper<F, args>::addParameter(p9);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
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
	     const funct::Parameter & p9, 
	     const funct::Parameter & p10) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    helper::RootFunctionHelper<F, args>::addParameter(p7);
    helper::RootFunctionHelper<F, args>::addParameter(p8);
    helper::RootFunctionHelper<F, args>::addParameter(p9);
    helper::RootFunctionHelper<F, args>::addParameter(p10);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
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
	     const funct::Parameter & p9, 
	     const funct::Parameter & p10, 
	     const funct::Parameter & p11) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    helper::RootFunctionHelper<F, args>::addParameter(p7);
    helper::RootFunctionHelper<F, args>::addParameter(p8);
    helper::RootFunctionHelper<F, args>::addParameter(p9);
    helper::RootFunctionHelper<F, args>::addParameter(p10);
    helper::RootFunctionHelper<F, args>::addParameter(p11);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
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
	     const funct::Parameter & p9, 
	     const funct::Parameter & p10, 
	     const funct::Parameter & p11, 
	     const funct::Parameter & p12) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    helper::RootFunctionHelper<F, args>::addParameter(p7);
    helper::RootFunctionHelper<F, args>::addParameter(p8);
    helper::RootFunctionHelper<F, args>::addParameter(p9);
    helper::RootFunctionHelper<F, args>::addParameter(p10);
    helper::RootFunctionHelper<F, args>::addParameter(p11);
    helper::RootFunctionHelper<F, args>::addParameter(p12);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
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
	     const funct::Parameter & p9, 
	     const funct::Parameter & p10, 
	     const funct::Parameter & p11, 
	     const funct::Parameter & p12, 
	     const funct::Parameter & p13) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    helper::RootFunctionHelper<F, args>::addParameter(p7);
    helper::RootFunctionHelper<F, args>::addParameter(p8);
    helper::RootFunctionHelper<F, args>::addParameter(p9);
    helper::RootFunctionHelper<F, args>::addParameter(p10);
    helper::RootFunctionHelper<F, args>::addParameter(p11);
    helper::RootFunctionHelper<F, args>::addParameter(p12);
    helper::RootFunctionHelper<F, args>::addParameter(p13);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
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
	     const funct::Parameter & p9, 
	     const funct::Parameter & p10, 
	     const funct::Parameter & p11, 
	     const funct::Parameter & p12, 
	     const funct::Parameter & p13, 
	     const funct::Parameter & p14) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    helper::RootFunctionHelper<F, args>::addParameter(p7);
    helper::RootFunctionHelper<F, args>::addParameter(p8);
    helper::RootFunctionHelper<F, args>::addParameter(p9);
    helper::RootFunctionHelper<F, args>::addParameter(p10);
    helper::RootFunctionHelper<F, args>::addParameter(p11);
    helper::RootFunctionHelper<F, args>::addParameter(p12);
    helper::RootFunctionHelper<F, args>::addParameter(p13);
    helper::RootFunctionHelper<F, args>::addParameter(p14);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
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
	     const funct::Parameter & p9, 
	     const funct::Parameter & p10, 
	     const funct::Parameter & p11, 
	     const funct::Parameter & p12, 
	     const funct::Parameter & p13, 
	     const funct::Parameter & p14, 
	     const funct::Parameter & p15) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    helper::RootFunctionHelper<F, args>::addParameter(p7);
    helper::RootFunctionHelper<F, args>::addParameter(p8);
    helper::RootFunctionHelper<F, args>::addParameter(p9);
    helper::RootFunctionHelper<F, args>::addParameter(p10);
    helper::RootFunctionHelper<F, args>::addParameter(p11);
    helper::RootFunctionHelper<F, args>::addParameter(p12);
    helper::RootFunctionHelper<F, args>::addParameter(p13);
    helper::RootFunctionHelper<F, args>::addParameter(p14);
    helper::RootFunctionHelper<F, args>::addParameter(p15);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
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
	     const funct::Parameter & p9, 
	     const funct::Parameter & p10, 
	     const funct::Parameter & p11, 
	     const funct::Parameter & p12, 
	     const funct::Parameter & p13, 
	     const funct::Parameter & p14, 
	     const funct::Parameter & p15, 
	     const funct::Parameter & p16) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    helper::RootFunctionHelper<F, args>::addParameter(p7);
    helper::RootFunctionHelper<F, args>::addParameter(p8);
    helper::RootFunctionHelper<F, args>::addParameter(p9);
    helper::RootFunctionHelper<F, args>::addParameter(p10);
    helper::RootFunctionHelper<F, args>::addParameter(p11);
    helper::RootFunctionHelper<F, args>::addParameter(p12);
    helper::RootFunctionHelper<F, args>::addParameter(p13);
    helper::RootFunctionHelper<F, args>::addParameter(p14);
    helper::RootFunctionHelper<F, args>::addParameter(p15);
    helper::RootFunctionHelper<F, args>::addParameter(p16);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
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
	     const funct::Parameter & p9, 
	     const funct::Parameter & p10, 
	     const funct::Parameter & p11, 
	     const funct::Parameter & p12, 
	     const funct::Parameter & p13, 
	     const funct::Parameter & p14, 
	     const funct::Parameter & p15, 
	     const funct::Parameter & p16, 
	     const funct::Parameter & p17) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    helper::RootFunctionHelper<F, args>::addParameter(p7);
    helper::RootFunctionHelper<F, args>::addParameter(p8);
    helper::RootFunctionHelper<F, args>::addParameter(p9);
    helper::RootFunctionHelper<F, args>::addParameter(p10);
    helper::RootFunctionHelper<F, args>::addParameter(p11);
    helper::RootFunctionHelper<F, args>::addParameter(p12);
    helper::RootFunctionHelper<F, args>::addParameter(p13);
    helper::RootFunctionHelper<F, args>::addParameter(p14);
    helper::RootFunctionHelper<F, args>::addParameter(p15);
    helper::RootFunctionHelper<F, args>::addParameter(p16);
    helper::RootFunctionHelper<F, args>::addParameter(p17);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
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
	     const funct::Parameter & p9, 
	     const funct::Parameter & p10, 
	     const funct::Parameter & p11, 
	     const funct::Parameter & p12, 
	     const funct::Parameter & p13, 
	     const funct::Parameter & p14, 
	     const funct::Parameter & p15, 
	     const funct::Parameter & p16, 
	     const funct::Parameter & p17, 
	     const funct::Parameter & p18) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    helper::RootFunctionHelper<F, args>::addParameter(p7);
    helper::RootFunctionHelper<F, args>::addParameter(p8);
    helper::RootFunctionHelper<F, args>::addParameter(p9);
    helper::RootFunctionHelper<F, args>::addParameter(p10);
    helper::RootFunctionHelper<F, args>::addParameter(p11);
    helper::RootFunctionHelper<F, args>::addParameter(p12);
    helper::RootFunctionHelper<F, args>::addParameter(p13);
    helper::RootFunctionHelper<F, args>::addParameter(p14);
    helper::RootFunctionHelper<F, args>::addParameter(p15);
    helper::RootFunctionHelper<F, args>::addParameter(p16);
    helper::RootFunctionHelper<F, args>::addParameter(p17);
    helper::RootFunctionHelper<F, args>::addParameter(p18);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
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
	     const funct::Parameter & p9, 
	     const funct::Parameter & p10, 
	     const funct::Parameter & p11, 
	     const funct::Parameter & p12, 
	     const funct::Parameter & p13, 
	     const funct::Parameter & p14, 
	     const funct::Parameter & p15, 
	     const funct::Parameter & p16, 
	     const funct::Parameter & p17, 
	     const funct::Parameter & p18, 
	     const funct::Parameter & p19) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    helper::RootFunctionHelper<F, args>::addParameter(p0);
    helper::RootFunctionHelper<F, args>::addParameter(p1);
    helper::RootFunctionHelper<F, args>::addParameter(p2);
    helper::RootFunctionHelper<F, args>::addParameter(p3);
    helper::RootFunctionHelper<F, args>::addParameter(p4);
    helper::RootFunctionHelper<F, args>::addParameter(p5);
    helper::RootFunctionHelper<F, args>::addParameter(p6);
    helper::RootFunctionHelper<F, args>::addParameter(p7);
    helper::RootFunctionHelper<F, args>::addParameter(p8);
    helper::RootFunctionHelper<F, args>::addParameter(p9);
    helper::RootFunctionHelper<F, args>::addParameter(p10);
    helper::RootFunctionHelper<F, args>::addParameter(p11);
    helper::RootFunctionHelper<F, args>::addParameter(p12);
    helper::RootFunctionHelper<F, args>::addParameter(p13);
    helper::RootFunctionHelper<F, args>::addParameter(p14);
    helper::RootFunctionHelper<F, args>::addParameter(p15);
    helper::RootFunctionHelper<F, args>::addParameter(p16);
    helper::RootFunctionHelper<F, args>::addParameter(p17);
    helper::RootFunctionHelper<F, args>::addParameter(p18);
    helper::RootFunctionHelper<F, args>::addParameter(p19);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
    function(F& f, const std::vector<funct::Parameter> & pars) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    std::vector<funct::Parameter>::const_iterator i, 
      b = pars.begin(), e = pars.end();
    for(i = b; i != e; ++i)
      helper::RootFunctionHelper<F, args>::addParameter(*i);
    return fun;
  }
  
  template<unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function 
    function(F& f, const std::vector<boost::shared_ptr<double> > & pars) {
    typename helper::RootFunctionHelper<F, args>::root_function 
      fun = helper::RootFunctionHelper<F, args>::fun(f);
    std::vector<boost::shared_ptr<double> >::const_iterator i, 
      b = pars.begin(), e = pars.end();
    for(i = b; i != e; ++i)
      helper::RootFunctionHelper<F, args>::addParameter(*i);
    return fun;
  }

}

#endif
