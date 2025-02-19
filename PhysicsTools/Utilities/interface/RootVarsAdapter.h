#ifndef PhysicsTools_Utilities_RootVarsAdapter_h
#define PhysicsTools_Utilities_RootVarsAdapter_h

namespace root {
  namespace helper {
  
    template<typename F, unsigned int args>
    struct RootVarsAdapter {
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
  }
}

#endif
