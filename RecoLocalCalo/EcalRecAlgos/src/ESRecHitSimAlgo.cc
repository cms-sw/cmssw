#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitSimAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <math.h>

ESRecHitSimAlgo::ESRecHitSimAlgo() {

}

double* ESRecHitSimAlgo::EvalAmplitude(const ESDataFrame& digi, const double& ped, const double& w0, const double& w1, const double& w2) const {
  
  double *results = new double[4];
  float energy = 0;
  double adc[3];
  float pw[3];
  pw[0] = w0;
  pw[1] = w1;
  pw[2] = w2;

  for (int i=0; i<digi.size(); i++) {
    energy += pw[i]*(digi.sample(i).adc()-ped);
    LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : Digi "<<i<<" ADC counts "<<digi.sample(i).adc()<<" Ped "<<ped;
    //std::cout<<i<<" "<<digi.sample(i).adc()<<" "<<ped<<" "<<pw[i]<<std::endl;
    adc[i] = digi.sample(i).adc() - ped;
  }
  
  double status = 0;
  if (adc[0] > 20) status = 14;
  if (adc[1] <= 0 || adc[2] <= 0) status = 10;
  if (adc[0] > adc[1] && adc[0] > adc[2]) status = 8;
  if (adc[2] > adc[1] && adc[2] > adc[0]) status = 9;
  double r12 = (adc[1] != 0) ? adc[0]/adc[1] : 99;
  double r23 = (adc[2] != 0) ? adc[1]/adc[2] : 99;
  if (r12 > ratioCuts_->getR12High()) status = 5;
  if (r23 > ratioCuts_->getR23High()) status = 6;
  if (r23 < ratioCuts_->getR23Low()) status = 7;

  double A1 = adc[1];
  double A2 = adc[2];

  // t0 from analytical formula:
  double n = 1.798;
  double w = 0.07291;
  double DeltaT = 25.;
  double aaa = (A2 > 0 && A1 > 0) ? log(A2/A1)/n : 20.; // if A1=0, t0=20
  double bbb = w/n*DeltaT;
  double ccc= exp(aaa+bbb);

  double t0 = (2.-ccc)/(1.-ccc) * DeltaT - 5;

  // A from analytical formula:
  double t1 = 20.;
  double A_1 =  pow(w/n*(t1),n) * exp(n-w*(t1));
  double AA1 = (A_1 != 0.) ? A1 / A_1 : 0.;

  if (adc[1] > 2800 && adc[2] > 2800) status = 11;
  else if (adc[1] > 2800) status = 12;
  else if (adc[2] > 2800) status = 13;

  results[0] = energy; // energy with weight method
  results[1] = t0;     // timing
  results[2] = status; // hit status
  results[3] = AA1;    // energy with analytic method

  return results;
}

EcalRecHit ESRecHitSimAlgo::reconstruct(const ESDataFrame& digi) const {

  ESPedestals::const_iterator it_ped = peds_->find(digi.id());

  ESIntercalibConstantMap::const_iterator it_mip = mips_->getMap().find(digi.id());
  ESAngleCorrectionFactors::const_iterator it_ang = ang_->getMap().find(digi.id());

  ESChannelStatusMap::const_iterator it_status = channelStatus_->getMap().find(digi.id());

  double* results;

  results = EvalAmplitude(digi, it_ped->getMean(), w0_, w1_, w2_);

  double energy   = results[0];
  double t0       = results[1];
  int status      = (int) results[2];
  double otenergy = results[3] * 1000000.; // set out-of-time energy to keV
  delete[] results;

  double mipCalib = (fabs(cos(*it_ang)) != 0.) ? (*it_mip)/fabs(cos(*it_ang)) : 0.;
  energy *= (mipCalib != 0.) ? MIPGeV_/mipCalib : 0.;
  otenergy *= (mipCalib != 0.) ? MIPGeV_/mipCalib : 0.;

  DetId detId = digi.id();

  LogDebug("ESRecHitSimAlgo") << "ESRecHitSimAlgo : reconstructed energy "<<energy;

  EcalRecHit rechit(digi.id(), energy, t0);
  rechit.setOutOfTimeEnergy(otenergy);

  if (it_status->getStatusCode() == 1) {
    rechit.setRecoFlag(EcalRecHit::kESDead);
  } else {
    if (status == 0)
      rechit.setRecoFlag(EcalRecHit::kESGood);
    else if (status == 5)
      rechit.setRecoFlag(EcalRecHit::kESBadRatioFor12);
    else if (status == 6)
      rechit.setRecoFlag(EcalRecHit::kESBadRatioFor23Upper);
    else if (status == 7)
      rechit.setRecoFlag(EcalRecHit::kESBadRatioFor23Lower);
    else if (status == 8)
      rechit.setRecoFlag(EcalRecHit::kESTS1Largest);
    else if (status == 9)
      rechit.setRecoFlag(EcalRecHit::kESTS3Largest);
    else if (status == 10)
      rechit.setRecoFlag(EcalRecHit::kESTS3Negative);
    else if (status == 11)
      rechit.setRecoFlag(EcalRecHit::kESSaturated);
    else if (status == 12)
      rechit.setRecoFlag(EcalRecHit::kESTS2Saturated);
    else if (status == 13)
      rechit.setRecoFlag(EcalRecHit::kESTS3Saturated);
    else if (status == 14)
      rechit.setRecoFlag(EcalRecHit::kESTS13Sigmas);
  }

  return rechit;
}

