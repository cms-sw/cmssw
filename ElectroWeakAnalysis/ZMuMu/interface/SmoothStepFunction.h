#ifndef PhysicsTools_Utilities_SmoothStepFunction_h
#define PhysicsTools_Utilities_SmoothStepFunction_h
#include "PhysicsTools/Utilities/interface/Parameter.h"
#include <boost/shared_ptr.hpp>
#include <cmath>

namespace function {
  
  struct SmoothStepFunction {
    static const unsigned int arguments = 1;
    SmoothStepFunction(const Parameter & t) : 
      trend(t.ptr()) { }
    SmoothStepFunction(boost::shared_ptr<double> t): 
      trend(t) { }
    SmoothStepFunction(double t): 
      trend(new double(t)){}
    double operator()(double x) const {
      double z = (x - 40)*(*trend);
      if(fabs(z)<0) return 0;
      return 2/(1+exp(-z)) -1;
    }
    boost::shared_ptr<double> trend;
  };

}

#endif
