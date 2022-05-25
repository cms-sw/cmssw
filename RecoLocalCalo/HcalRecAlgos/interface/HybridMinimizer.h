#ifndef ROOT_Minuit2_HybridMinimizer
#define ROOT_Minuit2_HybridMinimizer

#include "Minuit2/Minuit2Minimizer.h"

/** 
   Class HybridMinimizer exposes "SetMinimizerType" method
   of Minuit2Minimizer
*/

namespace PSFitter {

  class HybridMinimizer : public ROOT::Minuit2::Minuit2Minimizer {
  public:
    using ROOT::Minuit2::Minuit2Minimizer::Minuit2Minimizer;

    inline ~HybridMinimizer() override {}

    inline void SetMinimizerType(ROOT::Minuit2::EMinimizerType type) {
      ROOT::Minuit2::Minuit2Minimizer::SetMinimizerType(type);
    }
  };

}  // namespace PSFitter

#endif
