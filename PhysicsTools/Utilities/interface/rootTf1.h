#ifndef PhysicsTools_Utilities_rootTf1_h
#define PhysicsTools_Utilities_rootTf1_h
#include "PhysicsTools/Utilities/interface/rootFunction.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include "TF1.h"

namespace root {
  
  template<typename F>
  TF1 tf1(const char * name, F& f,
	  double min, double max) {
    TF1 fun(name, root::function<1>(f), min, max, 0);
    return fun;
  }

  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const funct::Parameter & p0) {
    TF1 fun(name, root::function<1>(f, p0), min, max, 1);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    return fun;
  }

  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const funct::Parameter & p0,
	  const funct::Parameter & p1) {
    TF1 fun(name, root::function<1>(f, p0, p1), min, max, 2);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    return fun;
  }

  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const funct::Parameter & p0,
	  const funct::Parameter & p1,
	  const funct::Parameter & p2) {
    TF1 fun(name, root::function<1>(f, p0, p1, p2), min, max, 3);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParName(0, p0.name().c_str());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParName(1, p1.name().c_str());
    fun.SetParameter(2, *p2.ptr());
    fun.SetParName(2, p2.name().c_str());
    return fun;
  }

  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const funct::Parameter & p0,
	  const funct::Parameter & p1,
	  const funct::Parameter & p2,
	  const funct::Parameter & p3) {
    TF1 fun(name, root::function<1>(f, p0, p1, p2, p3), min, max, 4);
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

  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const funct::Parameter & p0,
	  const funct::Parameter & p1,
	  const funct::Parameter & p2,
	  const funct::Parameter & p3,
	  const funct::Parameter & p4) {
    TF1 fun(name, root::function<1>(f, p0, p1, p2, p3, p4), min, max, 5);
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
  
  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const funct::Parameter & p0,
	  const funct::Parameter & p1,
	  const funct::Parameter & p2,
	  const funct::Parameter & p3,
	  const funct::Parameter & p4, 
	  const funct::Parameter & p5) {
    TF1 fun(name, root::function<1>(f, p0, p1, p2, p3, p4, p5), min, max, 6);
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
  
  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const funct::Parameter & p0,
	  const funct::Parameter & p1,
	  const funct::Parameter & p2,
	  const funct::Parameter & p3,
	  const funct::Parameter & p4, 
	  const funct::Parameter & p5, 
	  const funct::Parameter & p6) {
    TF1 fun(name, root::function<1>(f, p0, p1, p2, p3, p4, p5, p6), min, max, 7);
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
  
  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const funct::Parameter & p0,
	  const funct::Parameter & p1,
	  const funct::Parameter & p2,
	  const funct::Parameter & p3,
	  const funct::Parameter & p4, 
	  const funct::Parameter & p5, 
	  const funct::Parameter & p6,
	  const funct::Parameter & p7) {
    TF1 fun(name, root::function<1>(f, p0, p1, p2, p3, p4, p5, p6, p7), min, max, 8);
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
  
  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const funct::Parameter & p0,
	  const funct::Parameter & p1,
	  const funct::Parameter & p2,
	  const funct::Parameter & p3,
	  const funct::Parameter & p4, 
	  const funct::Parameter & p5, 
	  const funct::Parameter & p6,
	  const funct::Parameter & p7, 
	  const funct::Parameter & p8) {
    TF1 fun(name, root::function<1>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8), min, max, 9);
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
  
  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
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
    TF1 fun(name, root::function<1>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9), min, max, 10);
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
  
  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
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
    TF1 fun(name, root::function<1>(f, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10), min, max, 11);
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
  
  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const std::vector<funct::Parameter> & p) {
    TF1 fun(name, root::function<1>(f, p), min, max, p.size());
    for(size_t i = 0; i < p.size(); ++i) {
      fun.SetParameter(i, *p[i].ptr());
      fun.SetParName(i, p[i].name().c_str());
    }
    return fun;
  }  
  
  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const std::vector<boost::shared_ptr<double> > & p) {
    TF1 fun(name, root::function<1>(f, p), min, max, p.size());
    for(size_t i = 0; i < p.size(); ++i)
      fun.SetParameter(i, *p[i]);
    return fun;
  }  
 
}

#endif
