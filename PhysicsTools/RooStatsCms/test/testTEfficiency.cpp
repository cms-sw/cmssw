#include "TROOT.h"
#include "TStyle.h"
#include "TRandom3.h"
#include "TH1.h"
#include "TGraphAsymmErrors.h" 
#include "TCanvas.h"
#include "TEfficiency.h"
#include <cmath>

const double r = 6;
const double mu = 40;

double sigmoid(double x) {
  return 1.0/(1.0+exp(-(x-mu)/r));
}

void eff() {
  gROOT->SetBatch(kTRUE);
  gROOT->SetStyle("Plain");
  
  TRandom3 rndm;
  rndm.SetSeed(123456);
  
  const int bins = 50;
  const double xMax = 100;
  const double counts = 50;
  TH1D h_pass("h_pass", "pass", bins, 0, xMax);
  TH1D h_total("h_total", "total", bins, 0, xMax);
  for(int i = 0; i < bins; ++i) {
    double x = (double(i) + 0.5) * xMax / bins; 
    double efficiency = sigmoid(x);
    int n0 = rndm.Poisson(counts);
    int n1 = rndm.Binomial(n0, efficiency);
    h_pass.SetBinContent(i+1,n1);
    h_total.SetBinContent(i+1,n0);
  }
  TEfficiency * effic = 0;
  if(TEfficiency::CheckConsistency(h_pass,h_total)) {
    effic = new TEfficiency(h_pass,h_total);
    effic->SetStatisticOption(TEfficiency::kFCP);
  }

  TCanvas c;
  gStyle->SetOptStat(0);
  effic->Draw("AP");
  c.SaveAs("eff.pdf");
}  

int main() {
  eff();
}
