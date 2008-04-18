#ifndef PhysicTools_Utilities_RootFunctionAdapter_h
#define PhysicTools_Utilities_RootFunctionAdapter_h
#include <vector>
#include <boost/shared_ptr.hpp>
#include "TF1.h"

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
      void addParameter(boost::shared_ptr<double> par) {
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
      static void addParameter(boost::shared_ptr<double> par) {
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
	   boost::shared_ptr<double> p0) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    helper::RootFunctionHelper<F>::addParameter(p0);
    return fun;
  }

  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
    function(F& f, 
	     boost::shared_ptr<double> p0,
	     boost::shared_ptr<double> p1) {
    typename helper::RootFunctionHelper<F>::root_function 
      fun = helper::RootFunctionHelper<F>::fun(f);
    helper::RootFunctionHelper<F>::addParameter(p0);
    helper::RootFunctionHelper<F>::addParameter(p1);
    return fun;
  }

  template<typename F>
  typename helper::RootFunctionHelper<F>::root_function 
  function(F& f, 
	   boost::shared_ptr<double> p0,
	   boost::shared_ptr<double> p1,
	   boost::shared_ptr<double> p2) {
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
	  boost::shared_ptr<double> p0) {
    TF1 fun(name, root::function(f, p0), min, max, 1);
    fun.SetParameter(0, *p0);
    return fun;
  }

  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  boost::shared_ptr<double> p0,
	  boost::shared_ptr<double> p1) {
    TF1 fun(name, root::function(f, p0, p1), min, max, 2);
    fun.SetParameter(0, *p0);
    fun.SetParameter(1, *p1);
    return fun;
  }

  template<typename F>
  TF1 tf1(const char * name, F& f, double min, double max,
	  boost::shared_ptr<double> p0,
	  boost::shared_ptr<double> p1,
	  boost::shared_ptr<double> p2) {
    TF1 fun(name, root::function(f, p0, p1, p2), min, max, 3);
    fun.SetParameter(0, *p0);
    fun.SetParameter(1, *p1);
    fun.SetParameter(2, *p2);
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
