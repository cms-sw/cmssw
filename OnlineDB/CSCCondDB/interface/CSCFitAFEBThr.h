#ifndef CSCFitAFEBThr_h
#define CSCFitAFEBThr_h

/** \class CSCFitAFEBThr
 *
 * Concrete algorithmic class used to identify threshold and noise in 
 * AFEB channel threshold scan in the endcap muon CSCs. 
 * Based on CSCFitSCAPulse as an example
 */

#include <Minuit2/VariableMetricMinimizer.h>

class CSCThrTurnOnFcn;

class CSCFitAFEBThr {
public:
  typedef ROOT::Minuit2::ModularFunctionMinimizer ModularFunctionMinimizer;
  CSCFitAFEBThr();
  virtual ~CSCFitAFEBThr();

  /// Find the threshold and noise from the threshold turn-on curve.
  /// The returned bool is success/fail status.
  virtual bool ThresholdNoise(const std::vector<float>& inputx,
                              const std::vector<float>& inputy,
                              const int& npulses,
                              std::vector<int>& dacoccup,
                              std::vector<float>& mypar,
                              std::vector<float>& ermypar,
                              float& ercorr,
                              float& chisq,
                              int& ndf,
                              int& niter,
                              float& edm) const;

private:
  ModularFunctionMinimizer* theFitter;
  CSCThrTurnOnFcn* theOBJfun;
};

#endif
