#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitFitAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TMath.h"
#include "TGraph.h"

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

ESRecHitFitAlgo::ESRecHitFitAlgo(int pedestal, double MIPADC, double MIPkeV) : 
  ped_(pedestal), MIPADC_(MIPADC), MIPkeV_(MIPkeV) 
{

  fit_ = new TF1("fit", fitf, -200, 200, 2);
  fit_->SetParameters(50, 10);

}

ESRecHitFitAlgo::~ESRecHitFitAlgo() {
  delete fit_;
}

double* ESRecHitFitAlgo::EvalAmplitude(const ESDataFrame& digi) const {
  
  double *fitresults = new double[2];
  double adc[3];  
  double tx[3];
  tx[0] = -5.; 
  tx[1] = 20.;
  tx[2] = 45.;

  for (int i=0; i<digi.size(); i++) 
    adc[i] = digi.sample(i).adc() - ped_;

  int status = 0;
  if (adc[0] > 20) status = 1;
  if (adc[1] < 0 || adc[2] < 0) status = 1;
  if (adc[0] > adc[1] || adc[0] > adc[2]) status = 1;
  if (adc[2] > adc[1]) status = 1;  

  if (status == 0) {
    double para[10];
    TGraph *gr = new TGraph(3, tx, adc);
    fit_->SetParameters(50, 10);
    gr->Fit("fit", "MQ");
    fit_->GetParameters(para); 
    fitresults[0] = para[0]; 
    fitresults[1] = para[1]; 
    delete gr;
  } else {
    fitresults[0] = 0;
    fitresults[1] = -999;
  }

  return fitresults;
}

EcalRecHit ESRecHitFitAlgo::reconstruct(const ESDataFrame& digi) const {

  double* results;

  results = EvalAmplitude(digi);
  double energy = results[0];
  double t0 = results[1];
  delete results;

  energy *= MIPkeV_/MIPADC_;
  energy /= 1000000.;

  DetId detId = digi.id();

  LogDebug("ESRecHitFitAlgo") << "ESRecHitFitAlgo : reconstructed energy "<<energy<<" T0 "<<t0;

  return EcalRecHit(digi.id(), energy, t0);
}

