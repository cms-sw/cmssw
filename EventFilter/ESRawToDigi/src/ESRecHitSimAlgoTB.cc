#include "EventFilter/ESRawToDigi/interface/ESRecHitSimAlgoTB.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include "math.h"

ESRecHitSimAlgoTB::ESRecHitSimAlgoTB(int gain, double MIPADC, double MIPkeV) :
  gain_(gain), MIPADC_(MIPADC), MIPkeV_(MIPkeV)
{

  if (gain_ == 1) {
    MPV_[0][0][0] = 9.24/1.14;
    MPV_[0][0][1] = 8.84/1.14;
    MPV_[0][0][2] = 9.57/1.14;
    MPV_[0][0][3] = 0;
    MPV_[0][1][0] = 9.20/1.14;
    MPV_[0][1][1] = 9.03/1.14;
    MPV_[0][1][2] = 9.25/1.14;
    MPV_[0][1][3] = 9.17/1.14;
    MPV_[0][2][0] = 9.18/1.14;
    MPV_[0][2][1] = 9.01/1.14;
    MPV_[0][2][2] = 9.09/1.14;
    MPV_[0][2][3] = 8.89/1.14;
    MPV_[0][3][0] = 9.20/1.14;
    MPV_[0][3][1] = 9.03/1.14;
    MPV_[0][3][2] = 9.26/1.14;
    MPV_[0][3][3] = 0;
    MPV_[1][0][0] = 9.20/1.14;
    MPV_[1][0][1] = 9.26/1.14;
    MPV_[1][0][2] = 9.12/1.14;
    MPV_[1][0][3] = 9.34/1.14;
    MPV_[1][1][0] = 9.20/1.14;
    MPV_[1][1][1] = 9.34/1.14;
    MPV_[1][1][2] = 9.39/1.14;
    MPV_[1][1][3] = 9.21/1.14;
    MPV_[1][2][0] = 8.89/1.14;
    MPV_[1][2][1] = 9.26/1.14;
    MPV_[1][2][2] = 9.46/1.14;
    MPV_[1][2][3] = 9.30/1.14;
    MPV_[1][3][0] = 9.20/1.14;
    MPV_[1][3][1] = 9.20/1.14;
    MPV_[1][3][2] = 9.20/1.14;
    MPV_[1][3][3] = 9.20/1.14;
  } else if (gain_==2) {
    MPV_[0][0][0] = 53.00;
    MPV_[0][0][1] = 55.32;
    MPV_[0][0][2] = 55.00;
    MPV_[0][0][3] = 0;
    MPV_[0][1][0] = 53.00;
    MPV_[0][1][1] = 54.87;
    MPV_[0][1][2] = 54.58;
    MPV_[0][1][3] = 54.92;
    MPV_[0][2][0] = 52.00;
    MPV_[0][2][1] = 52.93;
    MPV_[0][2][2] = 52.16;
    MPV_[0][2][3] = 53.56;
    MPV_[0][3][0] = 53.00;
    MPV_[0][3][1] = 53.00;
    MPV_[0][3][2] = 53.00;
    MPV_[0][3][3] = 0;
    MPV_[1][0][0] = 53.00;
    MPV_[1][0][1] = 55.41;
    MPV_[1][0][2] = 56.94;
    MPV_[1][0][3] = 54.38;
    MPV_[1][1][0] = 53.00;
    MPV_[1][1][1] = 54.92;
    MPV_[1][1][2] = 54.33;
    MPV_[1][1][3] = 55.46;
    MPV_[1][2][0] = 53.00;
    MPV_[1][2][1] = 52.81;
    MPV_[1][2][2] = 53.91;
    MPV_[1][2][3] = 53.35;
    MPV_[1][3][0] = 53.00;
    MPV_[1][3][1] = 53.00;
    MPV_[1][3][2] = 53.00;
    MPV_[1][3][3] = 53.00;
  }
}

double ESRecHitSimAlgoTB::EvalAmplitude(const ESDataFrame& digi, double & tdc, int & pedestal, double & CM0, double & CM1, double &CM2) const {
  
  double energy = 0;

  double fts = 0;
  fts = tdc-39.62;

  double w[3];
  if (gain_==1) {

    if (fts<=6) {
      w[0] = 0;
      w[1] = 6.929e-01 + 4.63e-03*fts + 8.3e-04*pow(fts,2);
      w[2] = 5.854e-01 + 8.95e-03*fts + 1.5e-03*pow(fts,2);
    }
    else{
      w[0] = 6.571e-01 - 2.49e-02*fts + 7.5e-04*pow(fts,2);
      w[1] = 8.010e-01 - 1.21e-02*fts + 3.7e-04*pow(fts,2);
      w[2] = 8.615e-01 - 3.20e-02*fts + 1.02e-03*pow(fts,2);
    }

  } else {     

    if (fts<=5) {
      w[0] = 0;
      w[1] = 7.250e-01 - 1.15e-03*fts + 8.6e-04*pow(fts,2);
      w[2] = 4.525e-01 - 1.05e-03*fts + 1.06e-03*pow(fts,2);
    }
    else {
      w[0] = 4.273e-01 - 1.51e-02*fts + 5.3e-04*pow(fts,2);
      w[1] = 7.694e-01 - 6.10e-03*fts + 2.3e-04*pow(fts,2);
      w[2] = 5.428e-01 - 1.32e-02*fts + 5.0e-04*pow(fts,2);
    }
    
  }

  energy = w[0]*(digi.sample(0).adc()-pedestal-CM0) + w[1]*(digi.sample(1).adc()-pedestal-CM1) + w[2]*(digi.sample(2).adc()-pedestal-CM2);
  
  ESDetId id = digi.id();
  
  int pl = id.plane();
  int ix = id.six();
  int iy = id.siy();

  //for (int ii=0; ii<digi.size(); ++ii) {
  //energy += pw[ii]*(digi.sample(ii).adc()-ped_);
  //energy += (digi.sample(ii).adc()-pedestal);
  //}

  // convert to GeV
  //if (gain_>0) energy *= MIPkeV_/MIPADC_;
  //energy /= 1000000.;
  // convert to MIP
  if (MPV_[pl-1][ix-1][iy-1] != 0) 
    energy /= MPV_[pl-1][ix-1][iy-1];
  else 
    energy = 0;
  //cout<<"E : "<<energy<<endl;
  return energy;

}

EcalRecHit ESRecHitSimAlgoTB::reconstruct(const ESDataFrame& digi, double & tdc, int & pedestal, double & CM0, double & CM1, double &CM2) const {

  double energy = 0;
  double time = 0;

  energy = EvalAmplitude(digi, tdc, pedestal, CM0, CM1, CM2);

  //DetId detId = digi.id();

  LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : reconstructed energy "<<energy;

  return EcalRecHit(digi.id(), energy, time); 
}

