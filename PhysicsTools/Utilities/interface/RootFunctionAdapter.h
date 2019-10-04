#ifndef PhysicTools_Utilities_RootFunctionAdapter_h
#define PhysicTools_Utilities_RootFunctionAdapter_h
#include <vector>
#include <memory>

#include "PhysicsTools/Utilities/interface/RootVarsAdapter.h"

namespace root {
  namespace helper {

    template <typename F, unsigned int args>
    struct RootFunctionAdapter {
      RootFunctionAdapter() : f_(0) {}
      RootFunctionAdapter(F& f) : f_(&f) {}
      void addParameter(const std::shared_ptr<double>& par) { pars_.push_back(par); }
      void setParameters(const double* pars) {
        for (size_t i = 0; i < pars_.size(); ++i) {
          *pars_[i] = pars[i];
        }
      }
      double operator()(const double* var) const { return RootVarsAdapter<F, args>::value(*f_, var); }
      size_t numberOfParameters() const { return pars_.size(); }

    private:
      F* f_;
      std::vector<std::shared_ptr<double> > pars_;
    };

  }  // namespace helper

}  // namespace root

#endif
