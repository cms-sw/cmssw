#include "EventFilter/ESRawToDigi/interface/ESRecHitSimAlgoTB.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include "math.h"

ESRecHitSimAlgoTB::ESRecHitSimAlgoTB(int gain, double MIPADC, double MIPkeV) :
  gain_(gain), MIPADC_(MIPADC), MIPkeV_(MIPkeV)
{

  if (gain_ == 1) {
    MPV_[0][0][0] = 8.26;
    MPV_[0][0][1] = 8.22;
    MPV_[0][0][2] = 9.61;
    MPV_[0][0][3] = 0;
    MPV_[0][1][0] = 7.88;
    MPV_[0][1][1] = 8.34;
    MPV_[0][1][2] = 8.43;
    MPV_[0][1][3] = 8.65;
    MPV_[0][2][0] = 8.27;
    MPV_[0][2][1] = 8.32;
    MPV_[0][2][2] = 8.51;
    MPV_[0][2][3] = 8.41;
    MPV_[0][3][0] = 9.01;
    MPV_[0][3][1] = 8.30;
    MPV_[0][3][2] = 8.30;
    MPV_[0][3][3] = 0;
    MPV_[1][0][0] = 7.71;
    MPV_[1][0][1] = 8.48;
    MPV_[1][0][2] = 8.68;
    MPV_[1][0][3] = 8.68;
    MPV_[1][1][0] = 8.56;
    MPV_[1][1][1] = 8.52;
    MPV_[1][1][2] = 8.61;
    MPV_[1][1][3] = 8.68;
    MPV_[1][2][0] = 8.47;
    MPV_[1][2][1] = 8.78;
    MPV_[1][2][2] = 8.65;
    MPV_[1][2][3] = 8.53;
    MPV_[1][3][0] = 8.86;
    MPV_[1][3][1] = 8.86;
    MPV_[1][3][2] = 8.86;
    MPV_[1][3][3] = 8.86;
  } else if (gain_==2) {
    MPV_[0][0][0] = 51.26;
    MPV_[0][0][1] = 55.75;
    MPV_[0][0][2] = 55.96;
    MPV_[0][0][3] = 0;
    MPV_[0][1][0] = 52.15;
    MPV_[0][1][1] = 55.20;
    MPV_[0][1][2] = 54.69;
    MPV_[0][1][3] = 55.79;
    MPV_[0][2][0] = 52.41;
    MPV_[0][2][1] = 53.90;
    MPV_[0][2][2] = 53.52;
    MPV_[0][2][3] = 53.47;
    MPV_[0][3][0] = 51.82;
    MPV_[0][3][1] = 51.66;
    MPV_[0][3][2] = 50.41;
    MPV_[0][3][3] = 0;
    MPV_[1][0][0] = 53.20;
    MPV_[1][0][1] = 53.86;
    MPV_[1][0][2] = 54.24;
    MPV_[1][0][3] = 53.37;
    MPV_[1][1][0] = 51.70;
    MPV_[1][1][1] = 53.79;
    MPV_[1][1][2] = 54.01;
    MPV_[1][1][3] = 53.98;
    MPV_[1][2][0] = 49.32;
    MPV_[1][2][1] = 51.88;
    MPV_[1][2][2] = 52.14;
    MPV_[1][2][3] = 51.37;
    MPV_[1][3][0] = 48.03;
    MPV_[1][3][1] = 48.15;
    MPV_[1][3][2] = 48.87;
    MPV_[1][3][3] = 49.60;
  }
}

double ESRecHitSimAlgoTB::EvalAmplitude(const ESDataFrame& digi, double & tdc, int & pedestal, double & CM0, double & CM1, double &CM2) const {
  
  double energy = 0;

  double fts = 0;
  fts = tdc-38.86;

  double w[3];
  if (gain_==1) {

    if ( fts<=7 ) {
      w[0] = 0;
      w[1] = 6.758e-01 + 3.5e-03*fts + 8.7e-04*pow(fts,2);
      w[2] = 5.410e-01 + 6.5e-03*fts + 1.35e-03*pow(fts,2);
    } else {      
      w[0] = 6.959e-01 - 2.90e-02*fts + 8.3e-04*pow(fts,2);
      w[1] = 8.093e-01 - 1.58e-02*fts + 4.7e-04*pow(fts,2);
      w[2] = 9.110e-01 - 4.71e-02*fts + 1.48e-03*pow(fts,2);
    }

  } else {     
    
    if (fts<=5) {
      w[0] = 0;
      w[1] = 6.982e-01 + 1.17e-03*fts + 7.7e-04*pow(fts,2);
      w[2] = 4.951e-01 + 2.15e-03*fts + 1.25e-03*pow(fts,2);
    } else {   
      w[0] = 4.421e-01 - 1.69e-02*fts + 6.1e-04*pow(fts,2);
      w[1] = 7.467e-01 - 5.5e-03*fts + 2.2e-04*pow(fts,2);
      w[2] = 5.916e-01 - 1.2e-02*fts + 5.0e-04*pow(fts,2);
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

