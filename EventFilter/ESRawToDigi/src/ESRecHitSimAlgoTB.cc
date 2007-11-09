#include "EventFilter/ESRawToDigi/interface/ESRecHitSimAlgoTB.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

ESRecHitSimAlgoTB::ESRecHitSimAlgoTB(int gain, double MIPADC, double MIPkeV) :
  gain_(gain), MIPADC_(MIPADC), MIPkeV_(MIPkeV)
{

  if (gain_ == 1) {
    MPV_[0][0][0] = 8.80;
    MPV_[0][0][1] = 8.76;
    MPV_[0][0][2] = 9.17;
    MPV_[0][0][3] = 0;
    MPV_[0][1][0] = 8.40;
    MPV_[0][1][1] = 8.89;
    MPV_[0][1][2] = 8.98;
    MPV_[0][1][3] = 9.22;
    MPV_[0][2][0] = 8.81;
    MPV_[0][2][1] = 8.86;
    MPV_[0][2][2] = 9.08;
    MPV_[0][2][3] = 8.79;
    MPV_[0][3][0] = 9.01;
    MPV_[0][3][1] = 8.83;
    MPV_[0][3][2] = 8.84;
    MPV_[0][3][3] = 0;
    MPV_[1][0][0] = 7.88;
    MPV_[1][0][1] = 8.68;
    MPV_[1][0][2] = 8.86;
    MPV_[1][0][3] = 8.89;
    MPV_[1][1][0] = 8.74;
    MPV_[1][1][1] = 8.71;
    MPV_[1][1][2] = 8.79;
    MPV_[1][1][3] = 8.87;
    MPV_[1][2][0] = 8.73;
    MPV_[1][2][1] = 8.98;
    MPV_[1][2][2] = 8.84;
    MPV_[1][2][3] = 8.72;
    MPV_[1][3][0] = 8.86;
    MPV_[1][3][1] = 8.86;
    MPV_[1][3][2] = 8.86;
    MPV_[1][3][3] = 8.86;
  } else if (gain_==2) {
    MPV_[0][0][0] = 50.84;
    MPV_[0][0][1] = 54.67;
    MPV_[0][0][2] = 55.70;
    MPV_[0][0][3] = 0;
    MPV_[0][1][0] = 52.52;
    MPV_[0][1][1] = 55.20;
    MPV_[0][1][2] = 54.65;
    MPV_[0][1][3] = 55.22;
    MPV_[0][2][0] = 51.18;
    MPV_[0][2][1] = 53.54;
    MPV_[0][2][2] = 53.12;
    MPV_[0][2][3] = 53.49;
    MPV_[0][3][0] = 51.23;
    MPV_[0][3][1] = 51.33;
    MPV_[0][3][2] = 50.22;
    MPV_[0][3][3] = 0;
    MPV_[1][0][0] = 52.23;
    MPV_[1][0][1] = 53.50;
    MPV_[1][0][2] = 54.44;
    MPV_[1][0][3] = 53.15;
    MPV_[1][1][0] = 50.58;
    MPV_[1][1][1] = 53.76;
    MPV_[1][1][2] = 53.54;
    MPV_[1][1][3] = 54.12;
    MPV_[1][2][0] = 50.24;
    MPV_[1][2][1] = 51.49;
    MPV_[1][2][2] = 53.60;
    MPV_[1][2][3] = 52.35;
    MPV_[1][3][0] = 50.17;
    MPV_[1][3][1] = 47.90;
    MPV_[1][3][2] = 48.35;
    MPV_[1][3][3] = 49.30;
  }
}

double ESRecHitSimAlgoTB::EvalAmplitude(const ESDataFrame& digi, double & tdc, int & pedestal, double & CM0, double & CM1, double &CM2) const {
  
  double energy = 0;

  double fts = 0;
  if (gain_==1)
    fts = tdc-38.78;
  else
    fts = tdc-38.82;

  double w[3];
  if (gain_==1) {
    
    if ( fts<=6 ) {
      w[0] = 0;
      w[1] = 7.017e-01 + 3.66e-03*fts + 8.6e-04*pow(fts,2);
      w[2] = 5.575e-01 + 6.91e-03*fts + 1.45e-03*pow(fts,2);
    } else {      
      w[0] = 6.925e-01 - 2.87e-02*fts + 8.6e-04*pow(fts,2);
      w[1] = 8.105e-01 - 1.24e-02*fts + 3.7e-04*pow(fts,2);
      w[2] = 8.322e-01 - 3.33e-02*fts + 9.9e-04*pow(fts,2);
    }

  } else {     
    
    if (fts<=5) {
      w[0] = 0;
      w[1] = 6.985e-01 + 1.17e-03*fts + 7.8e-04*pow(fts,2);
      w[2] = 4.945e-01 + 2.14e-03*fts + 1.1e-03*pow(fts,2);
    } else {   
      w[0] = 4.421e-01 - 1.69e-02*fts + 6.1e-04*pow(fts,2);
      w[1] = 7.467e-01 - 5.4e-03*fts + 2.2e-04*pow(fts,2);
      w[2] = 5.916e-01 - 1.19e-02*fts + 4.9e-04*pow(fts,2);
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

