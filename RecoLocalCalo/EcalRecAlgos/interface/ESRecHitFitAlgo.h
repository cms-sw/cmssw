#ifndef RecoLocalCalo_EcalRecAlgos_ESRecHitFitAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_ESRecHitFitAlgo_HH

#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "TF1.h"

class ESRecHitFitAlgo {

 public:

  ESRecHitFitAlgo(int pedestal, double MIPADC, double MIPkeV);
  ~ESRecHitFitAlgo();
  double* EvalAmplitude(const ESDataFrame& digi) const;
  EcalRecHit reconstruct(const ESDataFrame& digi) const;

 private:

  TF1 *fit_;
  int    ped_;
  double MIPADC_;
  double MIPkeV_;

};

#endif
