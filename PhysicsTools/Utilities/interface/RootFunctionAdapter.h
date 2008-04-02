#ifndef PhysicTools_Utilities_RootFunctionAdapter_h
#define PhysicTools_Utilities_RootFunctionAdapter_h
#include <vector>
#include <boost/shared_ptr.hpp>
#include "TF1.h"
#include "TH1.h"
#include "TCanvas.h"
#include "PhysicsTools/Utilities/interface/Parameter.h"

namespace root {
  namespace helper {
  
    template<typename F, int arguments = F::arguments>
    struct RootVarsAdapter {
      static double value(F&, const double *);
    };
    
    template<typename F>
    struct RootVarsAdapter<F, 1> {
      static double value(F& f, const double * var) {
        return f(var[0]);
      }
    };
    
    template<typename F>
    struct RootVarsAdapter<F, 2> {
      static double value(F& f, const double * var) {
        return f(var[0], var[1]);
      }
    };
    
    template<typename F>
    struct RootFunctionAdapter {
      RootFunctionAdapter() : f_(0) { }
      RootFunctionAdapter(F & f) : f_(&f) { }
      void addParameter(const boost::shared_ptr<double> & par) {
	pars_.push_back(par);
      }
      void setParameters(const double * pars) {
	for(size_t i = 0; i < pars_.size(); ++i) {
	  *pars_[i] = pars[i];
	}
      }
      double operator()(const double * var) const {
        return RootVarsAdapter<F>::value(*f_, var);
      }
      size_t numberOfParameters() const {
	return pars_.size();
      }
    private:
      F * f_;
      std::vector<boost::shared_ptr<double> > pars_;
    };
    
    template<typename F>
    struct RootFunctionHelper {
      typedef double (*root_function)(const double *, const double *);
      static root_function fun(F& f) { 
	adapter_ = RootFunctionAdapter<F>(f); 
	return &fun_; 
      }
      static void addParameter(const boost::shared_ptr<double> & par) {
	adapter_.addParameter(par);
      }
    private:
      static double fun_(const double * x, const double * par) {
        adapter_.setParameters(par);
	return adapter_(x);
      }
      static RootFunctionAdapter<F> adapter_;
    };

    template<typename F>
    RootFunctionAdapter<F> RootFunctionHelper<F>::adapter_;
  }

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
	   const std::vector<boost::shared_ptr<double> > & pars) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    std::vector<boost::shared_ptr<double> >::const_iterator i, 
      b = pars.begin(), e = pars.end();
    for(i = b; i != e; ++i)
      helper::RootFunctionHelper<F>::addParameter(*i);
    return fun;
  }

  template<typename F>
  TF1 tf1(const char * name, F& f,
	  double min, double max) {
    TF1 fun(name, root::function(f), min, max, 0);
    return fun;
  }

  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const funct::Parameter & p0) {
    TF1 fun(name, root::function(f, p0), min, max, 1);
    fun.SetParameter(0, *p0.ptr());
    return fun;
  }

  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const funct::Parameter & p0,
	  const funct::Parameter & p1) {
    TF1 fun(name, root::function(f, p0, p1), min, max, 2);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParameter(1, *p1.ptr());
    return fun;
  }

  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const funct::Parameter & p0,
	  const funct::Parameter & p1,
	  const funct::Parameter & p2) {
    TF1 fun(name, root::function(f, p0, p1, p2), min, max, 3);
    fun.SetParameter(0, *p0.ptr());
    fun.SetParameter(1, *p1.ptr());
    fun.SetParameter(2, *p2.ptr());
    return fun;
  }

  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  const std::vector<boost::shared_ptr<double> > & p) {
    TF1 fun(name, root::function(f, p), min, max, p.size());
    for(size_t i = 0; i < p.size(); ++i)
      fun.SetParameter(i, *p[i]);
    return fun;
  }  

  inline void plotTF1(const char * name, TF1 & fun, TH1 & histo, 
		      double min, double max,
		      Color_t lineColor = kRed, Width_t lineWidth = 1,
		      Style_t lineStyle = kDashed) {
    fun.SetLineColor(lineColor);
    fun.SetLineWidth(lineWidth);
    fun.SetLineStyle(lineStyle);
    TCanvas *canvas = new TCanvas("canvas");
    histo.Draw("e");
    fun.Draw("same");	
    std::string plotName = name;
    canvas->SaveAs(plotName.c_str());
    canvas->SetLogy();
    std::string logPlotName = "log_" + plotName;
    canvas->SaveAs(logPlotName.c_str());
  }

  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed) {
    TF1 fun = root::tf1("fun", f, min, max);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle);
  }

  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed) {
    TF1 fun = root::tf1("fun", f, min, max, p0);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle);
  }

  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0, 
	    const funct::Parameter & p1,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed) {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle);
  }

  template<typename F>
  void plot(const char * name, TH1 & histo, F& f, double min, double max,
	    const funct::Parameter & p0, 
	    const funct::Parameter & p1,
	    const funct::Parameter & p2,
	    Color_t lineColor = kRed, Width_t lineWidth = 1,
	    Style_t lineStyle = kDashed) {
    TF1 fun = root::tf1("fun", f, min, max, p0, p1, p2);
    plotTF1(name, fun, histo, min, max, lineColor, lineWidth, lineStyle);
  }

   namespace helper {

    template<typename F>
    struct RootGradientHelper {
      typedef typename RootFunctionHelper<F>::root_function root_function;
      static double gradient(F& f, unsigned int index,
			     double min, double max, unsigned int pars) { 
	fun_ = root::function(f);
	index_ = index;
	rootFun_ = TF1("RootDummyInternalFunction", fun_, min, max, pars);
	TF1 grad("RootDummyInternalGradient", grad_, min, max, pars);
	return rootFun_.Integral(min, max);
      }
    private:
      static root_function fun_;
      static unsigned int index_;
      static TF1 rootFun_;
      static double grad_(const double * x, const double * p) { 
	static double grad[1024];
	rootFun_.GradientPar(x, grad);
	return grad[index_]; 
      }
    };

    template <typename F>
    typename RootGradientHelper<F>::root_function 
      RootGradientHelper<F>::fun_ = 0;

    template <typename F>
    unsigned int RootGradientHelper<F>::index_ = 0;

    template <typename F>
    TF1 RootGradientHelper<F>::rootFun_;
  }

  template<typename F>
  double gradient(F& f, unsigned int index, double min, double max, 
		  unsigned int pars) {
    return helper::RootGradientHelper<F>::gradient(f, index, min, max, pars);
  }

}

#endif
