#ifndef PhysicsTools_Utilities_RootFunctionHelper_h
#define PhysicsTools_Utilities_RootFunctionHelper_h
#include "PhysicsTools/Utilities/interface/RootFunctionAdapter.h"

namespace root {
  namespace helper {
    
    template<typename F, unsigned int args>
    struct RootFunctionHelper {
      typedef double (*root_function)(const double *, const double *);
      static root_function fun(F& f) { 
	adapter_ = RootFunctionAdapter<F, args>(f); 
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
      static RootFunctionAdapter<F, args> adapter_;
    };

    template<typename F, unsigned int args>
    RootFunctionAdapter<F, args> RootFunctionHelper<F, args>::adapter_;
  }
 }

#endif
