#ifndef RecoLocalCalo_EcalRecAlgos_ESRecHitSimAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_ESRecHitSimAlgo_HH

#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

// ESRecHitSimAlgo author : Chia-Ming, Kuo

class ESRecHitSimAlgo {

 public:

  ESRecHitSimAlgo(int gain, int pedestal, double MIPADC, double MIPkeV);
  ~ESRecHitSimAlgo(){}
  double EvalAmplitude(const ESDataFrame& digi) const;
  EcalRecHit reconstruct(const ESDataFrame& digi) const;

 private:

  int gain_;
  double ped_;
  float pw[3];
  double MIPADC_;
  double MIPkeV_;

};

#endif
