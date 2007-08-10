#ifndef ESRecHitSimAlgoCT_H
#define ESRecHitSimAlgoCT_H

#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

using namespace std;

class ESRecHitSimAlgoCT {

 public:

  ESRecHitSimAlgoCT(int gain, double MIPADC, double MIPkeV);
  ~ESRecHitSimAlgoCT(){}
  double EvalAmplitude(const ESDataFrame& digi, int & tdc, int & pedestal, double & CM0, double & CM1, double & CM2) const;
  EcalRecHit reconstruct(const ESDataFrame& digi, int & tdc, int & pedestal, double & CM0, double & CM1, double & CM2) const;

 private:

  int gain_;
  double MIPADC_;
  double MIPkeV_;

};

#endif
