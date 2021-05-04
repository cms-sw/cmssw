#ifndef PhysicsTools_Utilities_RootFunctionHelper_h
#define PhysicsTools_Utilities_RootFunctionHelper_h
/* Warning: this class users a static cache, so multiple
 * instances of the same type would have the same cacke.
 * This should be fixed to handle more general cases
 *
 */
#include "PhysicsTools/Utilities/interface/RootFunctionAdapter.h"

namespace root {
  namespace helper {
    struct null_t;

    template <typename F, unsigned int args, typename Tag = null_t>
    struct RootFunctionHelper {
      typedef double (*root_function)(const double *, const double *);
      static root_function fun(F &f) {
        adapter_ = RootFunctionAdapter<F, args>(f);
        return &fun_;
      }
      static void addParameter(const std::shared_ptr<double> &par) { adapter_.addParameter(par); }

    private:
      static double fun_(const double *x, const double *par) {
        adapter_.setParameters(par);
        return adapter_(x);
      }
      static RootFunctionAdapter<F, args> adapter_;
    };

    template <typename F, unsigned int args, typename Tag>
    RootFunctionAdapter<F, args> RootFunctionHelper<F, args, Tag>::adapter_;
  }  // namespace helper
}  // namespace root

#endif
