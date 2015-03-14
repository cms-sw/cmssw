#ifndef PedestalSub_h
#define PedestalSub_h 1

#include <typeinfo>

#include <TH1.h>
#include "TF1.h"
#include "TF2.h"
#include "TMath.h"

class PedestalSub
{
 public:
  enum Method { DoNothing=0, AvgWithThresh=1, AvgWithoutThresh=2, AvgWithThreshNoPedSub=3, Percentile=4 };

  PedestalSub();
  ~PedestalSub();
  
  void Init(Method method, int runCond, float threshold, float quantile);
  
  // This is the CMSSW Implementation of the apply function
  //void apply(const CaloSamples & cs, const std::vector<int> & capidvec, const HcalCalibrations & calibs, std::vector<double> & correctedOutput) const;
  // This is the edited implementation for our standalone test code
  
  void Calculate(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal, std::vector<double> & corrCharge) const;
  double GetCorrection(const std::vector<double> & inputCharge, const std::vector<double> & inputPedestal) const;

  Method fMethod;
  float fThreshold;
  float fQuantile;
  float fNoiseCorr;
  float fCondition;
  float fNoisePara;

 private:
  
};

#endif 
