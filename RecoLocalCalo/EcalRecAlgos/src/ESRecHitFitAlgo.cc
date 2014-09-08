#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitFitAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TMath.h"
#include "TGraph.h"

#include <iostream>
#include <cmath>

// fit function
Double_t fitf(Double_t *x, Double_t *par) {

  Double_t wc = 0.07291;
  Double_t n  = 1.798; // n-1 (in fact)
  Double_t v1 = pow(wc/n*(x[0]-par[1]), n);
  Double_t v2 = TMath::Exp(n-wc*(x[0]-par[1]));
  Double_t v  = par[0]*v1*v2;

  if (x[0] < par[1]) v = 0;

  return v;
}

ESRecHitFitAlgo::ESRecHitFitAlgo() {

  fit_ = new TF1("fit", fitf, -200, 200, 2);
  fit_->SetParameters(50, 10);

}

ESRecHitFitAlgo::~ESRecHitFitAlgo() {
  delete fit_;
}

double* ESRecHitFitAlgo::EvalAmplitude(const ESDataFrame& digi, double ped) const {
  
  double *fitresults = new double[3];
  double adc[3];  
  double tx[3];
  tx[0] = -5.; 
  tx[1] = 20.;
  tx[2] = 45.;

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
    double para[10];
    TGraph *gr = new TGraph(3, tx, adc);
    fit_->SetParameters(50, 10);
    gr->Fit(fit_, "MQ");
    fit_->GetParameters(para); 
    fitresults[0] = para[0]; 
    fitresults[1] = para[1];  
    delete gr;

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

EcalRecHit ESRecHitFitAlgo::reconstruct(const ESDataFrame& digi) const {
  
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

  double mipCalib = (std::fabs(cos(*it_ang)) != 0.) ? (*it_mip)/fabs(cos(*it_ang)) : 0.;
  energy *= (mipCalib != 0.) ? MIPGeV_/mipCalib : 0.;

  LogDebug("ESRecHitFitAlgo") << "ESRecHitFitAlgo : reconstructed energy "<<energy<<" T0 "<<t0;

  EcalRecHit rechit(digi.id(), energy, t0);

  if (it_status->getStatusCode() == 1) {
      rechit.setFlag(EcalRecHit::kESDead);
  } else {
    if (status == 0) 
      rechit.setFlag(EcalRecHit::kESGood);
    else if (status == 5) 
      rechit.setFlag(EcalRecHit::kESBadRatioFor12);
    else if (status == 6) 
      rechit.setFlag(EcalRecHit::kESBadRatioFor23Upper);
    else if (status == 7) 
      rechit.setFlag(EcalRecHit::kESBadRatioFor23Lower);
    else if (status == 8) 
      rechit.setFlag(EcalRecHit::kESTS1Largest);
    else if (status == 9) 
      rechit.setFlag(EcalRecHit::kESTS3Largest);
    else if (status == 10) 
      rechit.setFlag(EcalRecHit::kESTS3Negative);
    else if (status == 11) 
      rechit.setFlag(EcalRecHit::kESSaturated);
    else if (status == 12) 
      rechit.setFlag(EcalRecHit::kESTS2Saturated);
    else if (status == 13) 
      rechit.setFlag(EcalRecHit::kESTS3Saturated);
    else if (status == 14) 
      rechit.setFlag(EcalRecHit::kESTS13Sigmas);
  }

  return rechit;
}

