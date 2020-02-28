#ifndef PhysicsTools_Utilities_rootFunction_h
#define PhysicsTools_Utilities_rootFunction_h
#include "PhysicsTools/Utilities/interface/RootFunctionHelper.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"

namespace root {
  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f) {
    return helper::RootFunctionHelper<F, args, Tag>::fun(f);
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f) {
    return function_t<args, helper::null_t>(f);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f, const funct::Parameter& p0) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f, const funct::Parameter& p0) {
    return function_t<args, helper::null_t>(f, p0);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1) {
    return function<args, helper::null_t>(f, p0, p1);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2) {
    return function<args, helper::null_t>(f, p0, p1, p2);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3, p4);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3, p4, p5);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3, p4, p5, p6);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6,
                                                                              const funct::Parameter& p7) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p7);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6,
                                                                       const funct::Parameter& p7) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3, p4, p5, p6, p7);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6,
                                                                              const funct::Parameter& p7,
                                                                              const funct::Parameter& p8) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p7);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p8);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6,
                                                                       const funct::Parameter& p7,
                                                                       const funct::Parameter& p8) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6,
                                                                              const funct::Parameter& p7,
                                                                              const funct::Parameter& p8,
                                                                              const funct::Parameter& p9) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p7);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p8);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p9);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6,
                                                                       const funct::Parameter& p7,
                                                                       const funct::Parameter& p8,
                                                                       const funct::Parameter& p9) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6,
                                                                              const funct::Parameter& p7,
                                                                              const funct::Parameter& p8,
                                                                              const funct::Parameter& p9,
                                                                              const funct::Parameter& p10) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p7);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p8);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p9);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p10);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6,
                                                                       const funct::Parameter& p7,
                                                                       const funct::Parameter& p8,
                                                                       const funct::Parameter& p9,
                                                                       const funct::Parameter& p10) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6,
                                                                              const funct::Parameter& p7,
                                                                              const funct::Parameter& p8,
                                                                              const funct::Parameter& p9,
                                                                              const funct::Parameter& p10,
                                                                              const funct::Parameter& p11) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p7);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p8);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p9);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p10);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p11);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6,
                                                                       const funct::Parameter& p7,
                                                                       const funct::Parameter& p8,
                                                                       const funct::Parameter& p9,
                                                                       const funct::Parameter& p10,
                                                                       const funct::Parameter& p11) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6,
                                                                              const funct::Parameter& p7,
                                                                              const funct::Parameter& p8,
                                                                              const funct::Parameter& p9,
                                                                              const funct::Parameter& p10,
                                                                              const funct::Parameter& p11,
                                                                              const funct::Parameter& p12) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p7);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p8);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p9);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p10);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p11);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p12);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6,
                                                                       const funct::Parameter& p7,
                                                                       const funct::Parameter& p8,
                                                                       const funct::Parameter& p9,
                                                                       const funct::Parameter& p10,
                                                                       const funct::Parameter& p11,
                                                                       const funct::Parameter& p12) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6,
                                                                              const funct::Parameter& p7,
                                                                              const funct::Parameter& p8,
                                                                              const funct::Parameter& p9,
                                                                              const funct::Parameter& p10,
                                                                              const funct::Parameter& p11,
                                                                              const funct::Parameter& p12,
                                                                              const funct::Parameter& p13) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p7);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p8);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p9);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p10);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p11);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p12);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p13);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6,
                                                                       const funct::Parameter& p7,
                                                                       const funct::Parameter& p8,
                                                                       const funct::Parameter& p9,
                                                                       const funct::Parameter& p10,
                                                                       const funct::Parameter& p11,
                                                                       const funct::Parameter& p12,
                                                                       const funct::Parameter& p13) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6,
                                                                              const funct::Parameter& p7,
                                                                              const funct::Parameter& p8,
                                                                              const funct::Parameter& p9,
                                                                              const funct::Parameter& p10,
                                                                              const funct::Parameter& p11,
                                                                              const funct::Parameter& p12,
                                                                              const funct::Parameter& p13,
                                                                              const funct::Parameter& p14) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p7);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p8);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p9);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p10);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p11);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p12);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p13);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p14);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6,
                                                                       const funct::Parameter& p7,
                                                                       const funct::Parameter& p8,
                                                                       const funct::Parameter& p9,
                                                                       const funct::Parameter& p10,
                                                                       const funct::Parameter& p11,
                                                                       const funct::Parameter& p12,
                                                                       const funct::Parameter& p13,
                                                                       const funct::Parameter& p14) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6,
                                                                              const funct::Parameter& p7,
                                                                              const funct::Parameter& p8,
                                                                              const funct::Parameter& p9,
                                                                              const funct::Parameter& p10,
                                                                              const funct::Parameter& p11,
                                                                              const funct::Parameter& p12,
                                                                              const funct::Parameter& p13,
                                                                              const funct::Parameter& p14,
                                                                              const funct::Parameter& p15) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p7);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p8);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p9);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p10);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p11);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p12);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p13);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p14);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p15);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6,
                                                                       const funct::Parameter& p7,
                                                                       const funct::Parameter& p8,
                                                                       const funct::Parameter& p9,
                                                                       const funct::Parameter& p10,
                                                                       const funct::Parameter& p11,
                                                                       const funct::Parameter& p12,
                                                                       const funct::Parameter& p13,
                                                                       const funct::Parameter& p14,
                                                                       const funct::Parameter& p15) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6,
                                                                              const funct::Parameter& p7,
                                                                              const funct::Parameter& p8,
                                                                              const funct::Parameter& p9,
                                                                              const funct::Parameter& p10,
                                                                              const funct::Parameter& p11,
                                                                              const funct::Parameter& p12,
                                                                              const funct::Parameter& p13,
                                                                              const funct::Parameter& p14,
                                                                              const funct::Parameter& p15,
                                                                              const funct::Parameter& p16) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p7);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p8);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p9);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p10);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p11);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p12);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p13);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p14);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p15);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p16);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6,
                                                                       const funct::Parameter& p7,
                                                                       const funct::Parameter& p8,
                                                                       const funct::Parameter& p9,
                                                                       const funct::Parameter& p10,
                                                                       const funct::Parameter& p11,
                                                                       const funct::Parameter& p12,
                                                                       const funct::Parameter& p13,
                                                                       const funct::Parameter& p14,
                                                                       const funct::Parameter& p15,
                                                                       const funct::Parameter& p16) {
    return function<args, helper::null_t>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6,
                                                                              const funct::Parameter& p7,
                                                                              const funct::Parameter& p8,
                                                                              const funct::Parameter& p9,
                                                                              const funct::Parameter& p10,
                                                                              const funct::Parameter& p11,
                                                                              const funct::Parameter& p12,
                                                                              const funct::Parameter& p13,
                                                                              const funct::Parameter& p14,
                                                                              const funct::Parameter& p15,
                                                                              const funct::Parameter& p16,
                                                                              const funct::Parameter& p17) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p7);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p8);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p9);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p10);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p11);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p12);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p13);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p14);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p15);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p16);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p17);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6,
                                                                       const funct::Parameter& p7,
                                                                       const funct::Parameter& p8,
                                                                       const funct::Parameter& p9,
                                                                       const funct::Parameter& p10,
                                                                       const funct::Parameter& p11,
                                                                       const funct::Parameter& p12,
                                                                       const funct::Parameter& p13,
                                                                       const funct::Parameter& p14,
                                                                       const funct::Parameter& p15,
                                                                       const funct::Parameter& p16,
                                                                       const funct::Parameter& p17) {
    return function<args, helper::null_t>(
        f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6,
                                                                              const funct::Parameter& p7,
                                                                              const funct::Parameter& p8,
                                                                              const funct::Parameter& p9,
                                                                              const funct::Parameter& p10,
                                                                              const funct::Parameter& p11,
                                                                              const funct::Parameter& p12,
                                                                              const funct::Parameter& p13,
                                                                              const funct::Parameter& p14,
                                                                              const funct::Parameter& p15,
                                                                              const funct::Parameter& p16,
                                                                              const funct::Parameter& p17,
                                                                              const funct::Parameter& p18) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p7);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p8);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p9);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p10);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p11);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p12);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p13);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p14);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p15);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p16);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p17);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p18);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6,
                                                                       const funct::Parameter& p7,
                                                                       const funct::Parameter& p8,
                                                                       const funct::Parameter& p9,
                                                                       const funct::Parameter& p10,
                                                                       const funct::Parameter& p11,
                                                                       const funct::Parameter& p12,
                                                                       const funct::Parameter& p13,
                                                                       const funct::Parameter& p14,
                                                                       const funct::Parameter& p15,
                                                                       const funct::Parameter& p16,
                                                                       const funct::Parameter& p17,
                                                                       const funct::Parameter& p18) {
    return function<args, helper::null_t>(
        f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(F& f,
                                                                              const funct::Parameter& p0,
                                                                              const funct::Parameter& p1,
                                                                              const funct::Parameter& p2,
                                                                              const funct::Parameter& p3,
                                                                              const funct::Parameter& p4,
                                                                              const funct::Parameter& p5,
                                                                              const funct::Parameter& p6,
                                                                              const funct::Parameter& p7,
                                                                              const funct::Parameter& p8,
                                                                              const funct::Parameter& p9,
                                                                              const funct::Parameter& p10,
                                                                              const funct::Parameter& p11,
                                                                              const funct::Parameter& p12,
                                                                              const funct::Parameter& p13,
                                                                              const funct::Parameter& p14,
                                                                              const funct::Parameter& p15,
                                                                              const funct::Parameter& p16,
                                                                              const funct::Parameter& p17,
                                                                              const funct::Parameter& p18,
                                                                              const funct::Parameter& p19) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p0);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p1);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p2);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p3);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p4);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p5);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p6);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p7);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p8);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p9);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p10);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p11);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p12);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p13);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p14);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p15);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p16);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p17);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p18);
    helper::RootFunctionHelper<F, args, Tag>::addParameter(p19);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const funct::Parameter& p0,
                                                                       const funct::Parameter& p1,
                                                                       const funct::Parameter& p2,
                                                                       const funct::Parameter& p3,
                                                                       const funct::Parameter& p4,
                                                                       const funct::Parameter& p5,
                                                                       const funct::Parameter& p6,
                                                                       const funct::Parameter& p7,
                                                                       const funct::Parameter& p8,
                                                                       const funct::Parameter& p9,
                                                                       const funct::Parameter& p10,
                                                                       const funct::Parameter& p11,
                                                                       const funct::Parameter& p12,
                                                                       const funct::Parameter& p13,
                                                                       const funct::Parameter& p14,
                                                                       const funct::Parameter& p15,
                                                                       const funct::Parameter& p16,
                                                                       const funct::Parameter& p17,
                                                                       const funct::Parameter& p18,
                                                                       const funct::Parameter& p19) {
    return function<args, helper::null_t>(
        f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(
      F& f, const std::vector<funct::Parameter>& pars) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    std::vector<funct::Parameter>::const_iterator i, b = pars.begin(), e = pars.end();
    for (i = b; i != e; ++i)
      helper::RootFunctionHelper<F, args, Tag>::addParameter(*i);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(F& f,
                                                                       const std::vector<funct::Parameter>& pars) {
    return function_t<args, helper::null_t>(f, pars);
  }

  template <unsigned int args, typename Tag, typename F>
  typename helper::RootFunctionHelper<F, args, Tag>::root_function function_t(
      F& f, const std::vector<std::shared_ptr<double> >& pars) {
    typename helper::RootFunctionHelper<F, args, Tag>::root_function fun =
        helper::RootFunctionHelper<F, args, Tag>::fun(f);
    std::vector<std::shared_ptr<double> >::const_iterator i, b = pars.begin(), e = pars.end();
    for (i = b; i != e; ++i)
      helper::RootFunctionHelper<F, args, Tag>::addParameter(*i);
    return fun;
  }

  template <unsigned int args, typename F>
  typename helper::RootFunctionHelper<F, args>::root_function function(
      F& f, const std::vector<std::shared_ptr<double> >& pars) {
    return function_t<args, helper::null_t>(f, pars);
  }

}  // namespace root

#endif
