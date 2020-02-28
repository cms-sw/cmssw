#ifndef PhysicsTools_Utilities_rootTf1_h
#define PhysicsTools_Utilities_rootTf1_h
#include "PhysicsTools/Utilities/interface/rootFunction.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "TF1.h"

namespace root {

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name, F& f, double min, double max) {
    TF1 fun(name, root::function_t<1, Tag>(f), min, max, 0);
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name, F& f, double min, double max) {
    return tf1_t<helper::null_t>(name, f, min, max);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name, F& f, double min, double max, const funct::Parameter& p0) {
    TF1 fun(name, root::function_t<1, Tag>(f, p0), min, max, 1);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name, F& f, double min, double max, const funct::Parameter& p0) {
    return tf1_t<helper::null_t>(name, f, min, max, p0);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name, F& f, double min, double max, const funct::Parameter& p0, const funct::Parameter& p1) {
    TF1 fun(name, root::function_t<1, Tag>(f, p0, p1), min, max, 2);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name, F& f, double min, double max, const funct::Parameter& p0, const funct::Parameter& p1) {
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2) {
    TF1 fun(name, root::function_t<1, Tag>(f, p0, p1, p2), min, max, 3);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
          const funct::Parameter& p0,
          const funct::Parameter& p1,
          const funct::Parameter& p2) {
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1, p2);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3) {
    TF1 fun(name, root::function_t<1, Tag>(f, p0, p1, p2, p3), min, max, 4);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
          const funct::Parameter& p0,
          const funct::Parameter& p1,
          const funct::Parameter& p2,
          const funct::Parameter& p3) {
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1, p2, p3);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4) {
    TF1 fun(name, root::function_t<1, Tag>(f, p0, p1, p2, p3, p4), min, max, 5);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
          const funct::Parameter& p0,
          const funct::Parameter& p1,
          const funct::Parameter& p2,
          const funct::Parameter& p3,
          const funct::Parameter& p4) {
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1, p2, p3, p4);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5) {
    TF1 fun(name, root::function_t<1, Tag>(f, p0, p1, p2, p3, p4, p5), min, max, 6);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
          const funct::Parameter& p0,
          const funct::Parameter& p1,
          const funct::Parameter& p2,
          const funct::Parameter& p3,
          const funct::Parameter& p4,
          const funct::Parameter& p5) {
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1, p2, p3, p4, p5);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6) {
    TF1 fun(name, root::function_t<1, Tag>(f, p0, p1, p2, p3, p4, p5, p6), min, max, 7);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
          const funct::Parameter& p0,
          const funct::Parameter& p1,
          const funct::Parameter& p2,
          const funct::Parameter& p3,
          const funct::Parameter& p4,
          const funct::Parameter& p5,
          const funct::Parameter& p6) {
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1, p2, p3, p4, p5, p6);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7) {
    TF1 fun(name, root::function_t<1, Tag>(f, p0, p1, p2, p3, p4, p5, p6, p7), min, max, 8);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    fun.SetParameter(7, *p7.ptr());
    fun.SetParName(7, p7.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
          const funct::Parameter& p0,
          const funct::Parameter& p1,
          const funct::Parameter& p2,
          const funct::Parameter& p3,
          const funct::Parameter& p4,
          const funct::Parameter& p5,
          const funct::Parameter& p6,
          const funct::Parameter& p7) {
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1, p2, p3, p4, p5, p6, p7);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
            const funct::Parameter& p0,
            const funct::Parameter& p1,
            const funct::Parameter& p2,
            const funct::Parameter& p3,
            const funct::Parameter& p4,
            const funct::Parameter& p5,
            const funct::Parameter& p6,
            const funct::Parameter& p7,
            const funct::Parameter& p8) {
    TF1 fun(name, root::function_t<1, Tag>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8), min, max, 9);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    fun.SetParameter(7, *p7.ptr());
    fun.SetParName(7, p7.name().c_str());
    fun.SetParameter(8, *p8.ptr());
    fun.SetParName(8, p8.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
          const funct::Parameter& p0,
          const funct::Parameter& p1,
          const funct::Parameter& p2,
          const funct::Parameter& p3,
          const funct::Parameter& p4,
          const funct::Parameter& p5,
          const funct::Parameter& p6,
          const funct::Parameter& p7,
          const funct::Parameter& p8) {
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
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
    TF1 fun(name, root::function_t<1, Tag>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9), min, max, 10);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    fun.SetParameter(7, *p7.ptr());
    fun.SetParName(7, p7.name().c_str());
    fun.SetParameter(8, *p8.ptr());
    fun.SetParName(8, p8.name().c_str());
    fun.SetParameter(9, *p9.ptr());
    fun.SetParName(9, p9.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
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
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
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
    TF1 fun(name, root::function_t<1, Tag>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10), min, max, 11);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    fun.SetParameter(7, *p7.ptr());
    fun.SetParName(7, p7.name().c_str());
    fun.SetParameter(8, *p8.ptr());
    fun.SetParName(8, p8.name().c_str());
    fun.SetParameter(9, *p9.ptr());
    fun.SetParName(9, p9.name().c_str());
    fun.SetParameter(10, *p10.ptr());
    fun.SetParName(10, p10.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
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
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
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
    TF1 fun(name, root::function_t<1, Tag>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11), min, max, 12);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    fun.SetParameter(7, *p7.ptr());
    fun.SetParName(7, p7.name().c_str());
    fun.SetParameter(8, *p8.ptr());
    fun.SetParName(8, p8.name().c_str());
    fun.SetParameter(9, *p9.ptr());
    fun.SetParName(9, p9.name().c_str());
    fun.SetParameter(10, *p10.ptr());
    fun.SetParName(10, p10.name().c_str());
    fun.SetParameter(11, *p11.ptr());
    fun.SetParName(11, p11.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
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
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
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
    TF1 fun(name, root::function_t<1, Tag>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12), min, max, 13);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    fun.SetParameter(7, *p7.ptr());
    fun.SetParName(7, p7.name().c_str());
    fun.SetParameter(8, *p8.ptr());
    fun.SetParName(8, p8.name().c_str());
    fun.SetParameter(9, *p9.ptr());
    fun.SetParName(9, p9.name().c_str());
    fun.SetParameter(10, *p10.ptr());
    fun.SetParName(10, p10.name().c_str());
    fun.SetParameter(11, *p11.ptr());
    fun.SetParName(11, p11.name().c_str());
    fun.SetParameter(12, *p12.ptr());
    fun.SetParName(12, p12.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
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
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
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
    TF1 fun(
        name, root::function_t<1, Tag>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13), min, max, 14);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    fun.SetParameter(7, *p7.ptr());
    fun.SetParName(7, p7.name().c_str());
    fun.SetParameter(8, *p8.ptr());
    fun.SetParName(8, p8.name().c_str());
    fun.SetParameter(9, *p9.ptr());
    fun.SetParName(9, p9.name().c_str());
    fun.SetParameter(10, *p10.ptr());
    fun.SetParName(10, p10.name().c_str());
    fun.SetParameter(11, *p11.ptr());
    fun.SetParName(11, p11.name().c_str());
    fun.SetParameter(12, *p12.ptr());
    fun.SetParName(12, p12.name().c_str());
    fun.SetParameter(13, *p13.ptr());
    fun.SetParName(13, p13.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
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
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
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
    TF1 fun(name,
            root::function_t<1, Tag>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14),
            min,
            max,
            15);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    fun.SetParameter(7, *p7.ptr());
    fun.SetParName(7, p7.name().c_str());
    fun.SetParameter(8, *p8.ptr());
    fun.SetParName(8, p8.name().c_str());
    fun.SetParameter(9, *p9.ptr());
    fun.SetParName(9, p9.name().c_str());
    fun.SetParameter(10, *p10.ptr());
    fun.SetParName(10, p10.name().c_str());
    fun.SetParameter(11, *p11.ptr());
    fun.SetParName(11, p11.name().c_str());
    fun.SetParameter(12, *p12.ptr());
    fun.SetParName(12, p12.name().c_str());
    fun.SetParameter(13, *p13.ptr());
    fun.SetParName(13, p13.name().c_str());
    fun.SetParameter(14, *p14.ptr());
    fun.SetParName(14, p14.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
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
    return tf1_t<helper::null_t>(name, f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
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
    TF1 fun(name,
            root::function_t<1, Tag>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15),
            min,
            max,
            16);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    fun.SetParameter(7, *p7.ptr());
    fun.SetParName(7, p7.name().c_str());
    fun.SetParameter(8, *p8.ptr());
    fun.SetParName(8, p8.name().c_str());
    fun.SetParameter(9, *p9.ptr());
    fun.SetParName(9, p9.name().c_str());
    fun.SetParameter(10, *p10.ptr());
    fun.SetParName(10, p10.name().c_str());
    fun.SetParameter(11, *p11.ptr());
    fun.SetParName(11, p11.name().c_str());
    fun.SetParameter(12, *p12.ptr());
    fun.SetParName(12, p12.name().c_str());
    fun.SetParameter(13, *p13.ptr());
    fun.SetParName(13, p13.name().c_str());
    fun.SetParameter(14, *p14.ptr());
    fun.SetParName(14, p14.name().c_str());
    fun.SetParameter(15, *p15.ptr());
    fun.SetParName(15, p15.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
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
    return tf1_t<helper::null_t>(
        name, f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
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
    TF1 fun(name,
            root::function_t<1, Tag>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16),
            min,
            max,
            17);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    fun.SetParameter(7, *p7.ptr());
    fun.SetParName(7, p7.name().c_str());
    fun.SetParameter(8, *p8.ptr());
    fun.SetParName(8, p8.name().c_str());
    fun.SetParameter(9, *p9.ptr());
    fun.SetParName(9, p9.name().c_str());
    fun.SetParameter(10, *p10.ptr());
    fun.SetParName(10, p10.name().c_str());
    fun.SetParameter(11, *p11.ptr());
    fun.SetParName(11, p11.name().c_str());
    fun.SetParameter(12, *p12.ptr());
    fun.SetParName(12, p12.name().c_str());
    fun.SetParameter(13, *p13.ptr());
    fun.SetParName(13, p13.name().c_str());
    fun.SetParameter(14, *p14.ptr());
    fun.SetParName(14, p14.name().c_str());
    fun.SetParameter(15, *p15.ptr());
    fun.SetParName(15, p15.name().c_str());
    fun.SetParameter(16, *p16.ptr());
    fun.SetParName(16, p16.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
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
    return tf1_t<helper::null_t>(
        name, f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
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
    TF1 fun(name,
            root::function_t<1, Tag>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17),
            min,
            max,
            18);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    fun.SetParameter(7, *p7.ptr());
    fun.SetParName(7, p7.name().c_str());
    fun.SetParameter(8, *p8.ptr());
    fun.SetParName(8, p8.name().c_str());
    fun.SetParameter(9, *p9.ptr());
    fun.SetParName(9, p9.name().c_str());
    fun.SetParameter(10, *p10.ptr());
    fun.SetParName(10, p10.name().c_str());
    fun.SetParameter(11, *p11.ptr());
    fun.SetParName(11, p11.name().c_str());
    fun.SetParameter(12, *p12.ptr());
    fun.SetParName(12, p12.name().c_str());
    fun.SetParameter(13, *p13.ptr());
    fun.SetParName(13, p13.name().c_str());
    fun.SetParameter(14, *p14.ptr());
    fun.SetParName(14, p14.name().c_str());
    fun.SetParameter(15, *p15.ptr());
    fun.SetParName(15, p15.name().c_str());
    fun.SetParameter(16, *p16.ptr());
    fun.SetParName(16, p16.name().c_str());
    fun.SetParameter(17, *p17.ptr());
    fun.SetParName(17, p17.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
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
    return tf1_t<helper::null_t>(
        name, f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
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
    TF1 fun(name,
            root::function_t<1, Tag>(
                f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18),
            min,
            max,
            19);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    fun.SetParameter(7, *p7.ptr());
    fun.SetParName(7, p7.name().c_str());
    fun.SetParameter(8, *p8.ptr());
    fun.SetParName(8, p8.name().c_str());
    fun.SetParameter(9, *p9.ptr());
    fun.SetParName(9, p9.name().c_str());
    fun.SetParameter(10, *p10.ptr());
    fun.SetParName(10, p10.name().c_str());
    fun.SetParameter(11, *p11.ptr());
    fun.SetParName(11, p11.name().c_str());
    fun.SetParameter(12, *p12.ptr());
    fun.SetParName(12, p12.name().c_str());
    fun.SetParameter(13, *p13.ptr());
    fun.SetParName(13, p13.name().c_str());
    fun.SetParameter(14, *p14.ptr());
    fun.SetParName(14, p14.name().c_str());
    fun.SetParameter(15, *p15.ptr());
    fun.SetParName(15, p15.name().c_str());
    fun.SetParameter(16, *p16.ptr());
    fun.SetParName(16, p16.name().c_str());
    fun.SetParameter(17, *p17.ptr());
    fun.SetParName(17, p17.name().c_str());
    fun.SetParameter(18, *p18.ptr());
    fun.SetParName(18, p18.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
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
    return tf1_t<helper::null_t>(
        name, f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name,
            F& f,
            double min,
            double max,
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
    TF1 fun(name,
            root::function_t<1, Tag>(
                f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19),
            min,
            max,
            20);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    fun.SetParameter(3, *p3.ptr());
    fun.SetParName(3, p3.name().c_str());
    fun.SetParameter(4, *p4.ptr());
    fun.SetParName(4, p4.name().c_str());
    fun.SetParameter(5, *p5.ptr());
    fun.SetParName(5, p5.name().c_str());
    fun.SetParameter(6, *p6.ptr());
    fun.SetParName(6, p6.name().c_str());
    fun.SetParameter(7, *p7.ptr());
    fun.SetParName(7, p7.name().c_str());
    fun.SetParameter(8, *p8.ptr());
    fun.SetParName(8, p8.name().c_str());
    fun.SetParameter(9, *p9.ptr());
    fun.SetParName(9, p9.name().c_str());
    fun.SetParameter(10, *p10.ptr());
    fun.SetParName(10, p10.name().c_str());
    fun.SetParameter(11, *p11.ptr());
    fun.SetParName(11, p11.name().c_str());
    fun.SetParameter(12, *p12.ptr());
    fun.SetParName(12, p12.name().c_str());
    fun.SetParameter(13, *p13.ptr());
    fun.SetParName(13, p13.name().c_str());
    fun.SetParameter(14, *p14.ptr());
    fun.SetParName(14, p14.name().c_str());
    fun.SetParameter(15, *p15.ptr());
    fun.SetParName(15, p15.name().c_str());
    fun.SetParameter(16, *p16.ptr());
    fun.SetParName(16, p16.name().c_str());
    fun.SetParameter(17, *p17.ptr());
    fun.SetParName(17, p17.name().c_str());
    fun.SetParameter(18, *p18.ptr());
    fun.SetParName(18, p18.name().c_str());
    fun.SetParameter(19, *p19.ptr());
    fun.SetParName(19, p19.name().c_str());
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name,
          F& f,
          double min,
          double max,
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
    return tf1_t<helper::null_t>(
        name, f, min, max, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name, F& f, double min, double max, const std::vector<funct::Parameter>& p) {
    TF1 fun(name, root::function_t<1, Tag>(f, p), min, max, p.size());
    for (size_t i = 0; i < p.size(); ++i) {
      fun.SetParameter(i, *p[i].ptr());
      fun.SetParName(i, p[i].name().c_str());
    }
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name, F& f, double min, double max, const std::vector<funct::Parameter>& p) {
    return tf1_t<helper::null_t>(name, f, min, max, p);
  }

  template <typename Tag, typename F>
  TF1 tf1_t(const char* name, F& f, double min, double max, const std::vector<std::shared_ptr<double> >& p) {
    TF1 fun(name, root::function_t<1, Tag>(f, p), min, max, p.size());
    for (size_t i = 0; i < p.size(); ++i)
      fun.SetParameter(i, *p[i]);
    return fun;
  }

  template <typename F>
  TF1 tf1(const char* name, F& f, double min, double max, const std::vector<std::shared_ptr<double> >& p) {
    return tf1_t<helper::null_t>(name, f, min, max, p);
  }

}  // namespace root

#endif
