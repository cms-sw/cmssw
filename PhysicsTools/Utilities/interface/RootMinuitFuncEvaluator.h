#ifndef PhysicsTools_Utilities_RootMinuitFuncEvaluator_h
#define PhysicsTools_Utilities_RootMinuitFuncEvaluator_h

namespace fit {
  template <typename Function>
  struct RootMinuitFuncEvaluator {
    static double evaluate(const Function& f) { return f(); }
  };
}  // namespace fit

#endif
