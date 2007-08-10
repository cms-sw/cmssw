#ifndef ESRecHitSimAlgoTB_H
#define ESRecHitSimAlgoTB_H

#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

using namespace std;

class ESRecHitSimAlgoTB {

 public:

  ESRecHitSimAlgoTB(int gain, double MIPADC, double MIPkeV);
  ~ESRecHitSimAlgoTB(){}
  double EvalAmplitude(const ESDataFrame& digi, double & tdc, int & pedestal, double & CM0, double & CM1, double & CM2) const;
  EcalRecHit reconstruct(const ESDataFrame& digi, double & tdc, int & pedestal, double & CM0, double & CM1, double & CM2) const;

 private:

  int gain_;
  double MIPADC_;
  double MIPkeV_;

};

#endif
