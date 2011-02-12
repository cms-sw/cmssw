#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitAnalyticAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <cmath>

ESRecHitAnalyticAlgo::ESRecHitAnalyticAlgo() {
}

ESRecHitAnalyticAlgo::~ESRecHitAnalyticAlgo() {
}

double* ESRecHitAnalyticAlgo::EvalAmplitude(const ESDataFrame& digi, double ped) const {
  
  double *fitresults = new double[3];
  double adc[3];  

  for (int i=0; i<digi.size(); i++) 
    adc[i] = digi.sample(i).adc() - ped;

  double status = 0;
  if (adc[0] > 20) status = 14;
  if (adc[1] < 0 || adc[2] < 0) status = 10;
  if (adc[0] > adc[1] && adc[0] > adc[2]) status = 8;
  if (adc[2] > adc[1] && adc[2] > adc[0]) status = 9;  
  double r12 = (adc[1] != 0) ? adc[0]/adc[1] : 99999;
  double r23 = (adc[2] != 0) ? adc[1]/adc[2] : 99999;
  if (r12 > ratioCuts_->getR12High()) status = 5;
  if (r23 > ratioCuts_->getR23High()) status = 6;
  if (r23 < ratioCuts_->getR23Low()) status = 7;

  if (int(status) == 0) {

    double A1 = adc[1];
    double A2 = adc[2];

    // t0 from analytical formula:
    double n = 1.798;
    double w = 0.07291;
    double DeltaT = 25.;
    double aaa = log(A2/A1)/n;
    double bbb = w/n*DeltaT;
    double ccc= exp(aaa+bbb);

    //double t0 = (2.-ccc)/(ccc-1) * DeltaT + 5;
    double t0 = (2.-ccc)/(1.-ccc) * DeltaT - 5;

    // A from analytical formula:
    double t1 = 20.;
    //double t2 = 45.;
    double A_1 =  pow(w/n*(t1),n) * exp(n-w*(t1));
    //double A_2 =  pow(w/n*(t2),n) * exp(n-w*(t2));
    double AA1 = A1 / A_1;
    //double AA2 = A2 / A_2;

    fitresults[0] = AA1;
    fitresults[1] = t0;

    if (adc[1] > 2800 && adc[2] > 2800) status = 11;
    else if (adc[1] > 2800) status = 12;
    else if (adc[2] > 2800) status = 13;

  } else {
    fitresults[0] = 0;
    fitresults[1] = -999;
  }

  fitresults[2] = status;

  return fitresults;
}

EcalRecHit ESRecHitAnalyticAlgo::reconstruct(const ESDataFrame& digi) const {
  
  ESPedestals::const_iterator it_ped = peds_->find(digi.id());
  
  ESIntercalibConstantMap::const_iterator it_mip = mips_->getMap().find(digi.id());
  ESAngleCorrectionFactors::const_iterator it_ang = ang_->getMap().find(digi.id());

  ESChannelStatusMap::const_iterator it_status = channelStatus_->getMap().find(digi.id());

  double* results;

  results = EvalAmplitude(digi, it_ped->getMean());

  double energy = results[0];
  double t0 = results[1];
  int status = (int) results[2];
  delete[] results;

  double mipCalib = (fabs(cos(*it_ang)) != 0.) ? (*it_mip)/fabs(cos(*it_ang)) : 0.;
  energy *= (mipCalib != 0.) ? MIPGeV_/mipCalib : 0.;

  DetId detId = digi.id();

  LogDebug("ESRecHitAnalyticAlgo") << "ESRecHitAnalyticAlgo : reconstructed energy "<<energy<<" T0 "<<t0;

  EcalRecHit rechit(digi.id(), energy, t0);

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

