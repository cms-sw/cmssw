#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitSimAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ESRecHitSimAlgo::ESRecHitSimAlgo(int gain, int pedestal, double MIPADC, double MIPkeV) :
  gain_(gain), ped_(pedestal), MIPADC_(MIPADC), MIPkeV_(MIPkeV) 
{

  // pulse height parametrization
  // 0 : old gain in ORCA
  // 1 : low gain for data taking
  // 2 : high gain for calibration
  if (gain_ == 0) {
    pw[0] = -1.12521;
    pw[1] = 0.877968;
    pw[2] = 0.247238;
  }
  else if (gain_ == 1) {
    pw[0] = -0.0739126;
    pw[1] = 0.7815982;
    pw[2] = 0.3691372;
  }
  else if (gain_ == 2) {
    pw[0] = -0.01632344;
    pw[1] = 0.7515172;
    pw[2] = 0.4028312;
  }

  LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : Gain "<<gain_<<" Weights : "<<pw[0]<<" "<<pw[1]<<" "<<pw[2];
}

double ESRecHitSimAlgo::EvalAmplitude(const ESDataFrame& digi) const {
  
  float energy = 0;
  
  for (int i=0; i<digi.size(); i++) {
    energy += pw[i]*(digi.sample(i).adc()-ped_);
    LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : Digi "<<i<<" ADC counts "<<digi.sample(i).adc()<<" Ped "<<ped_;
  }
  if (gain_>0) energy *= MIPkeV_/MIPADC_;

  // convert to GeV
  energy /= 1000000.;

  return energy;
}

EcalRecHit ESRecHitSimAlgo::reconstruct(const ESDataFrame& digi) const {

  float energy = 0;
  float time = 0;

  energy = EvalAmplitude(digi);

  LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : reconstructed energy "<<energy;

  return EcalRecHit(digi.id(), energy, time); 
}

