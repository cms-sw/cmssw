#include "EventFilter/ESRawToDigi/interface/ESRecHitSimAlgoCT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

ESRecHitSimAlgoCT::ESRecHitSimAlgoCT(int gain, double MIPADC, double MIPkeV) :
  gain_(gain), MIPADC_(MIPADC), MIPkeV_(MIPkeV)
{
  
}

double ESRecHitSimAlgoCT::EvalAmplitude(const ESDataFrame& digi, int & tdc, int & pedestal, double & CM0, double & CM1, double &CM2) const {
  
  double energy = 0;

  double fts = ((double)tdc-800.)*24.951/194.;

  double w[3];
  if (gain_==1) {
    w[0] = -0.35944  + 0.03199 * fts;
    w[1] =  0.58562  + 0.03724 * fts;
    w[2] =  0.77204  - 0.06913 * fts;
  } else {
    w[0] = -0.26888  + 0.01807 * fts;
    w[1] =  0.54452  + 0.03204 * fts;
    w[2] =  0.72597  - 0.05062 * fts;
  }

  energy = w[0]*(digi.sample(0).adc()-pedestal-CM0) + w[1]*(digi.sample(1).adc()-pedestal-CM1) + w[2]*(digi.sample(2).adc()-pedestal-CM2);
  
  //for (int ii=0; ii<digi.size(); ++ii) {
  //energy += pw[ii]*(digi.sample(ii).adc()-ped_);
  //energy += (digi.sample(ii).adc()-pedestal);
  //}
  if (gain_>0) energy *= MIPkeV_/MIPADC_;

  // convert to GeV
  energy /= 1000000.;
  //cout<<"E : "<<energy<<endl;
  return energy;

}

EcalRecHit ESRecHitSimAlgoCT::reconstruct(const ESDataFrame& digi, int & tdc, int & pedestal, double & CM0, double & CM1, double &CM2) const {

  double energy = 0;
  double time = 0;

  energy = EvalAmplitude(digi, tdc, pedestal, CM0, CM1, CM2);

  //DetId detId = digi.id();

  LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : reconstructed energy "<<energy;

  return EcalRecHit(digi.id(), energy, time); 
}

