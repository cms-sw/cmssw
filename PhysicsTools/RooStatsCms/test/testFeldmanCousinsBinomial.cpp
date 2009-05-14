#include "TCanvas.h"
#include "TROOT.h"
#include "TRandom3.h"
#include "TH1.h"
#include "TGraphAsymmErrors.h" 
#include "PhysicsTools/RooStatsCms/interface/binomial_noncentral_intervals.h"
#include <cmath>

const double r = 6;
const double mu = 40;

double sigmoid(double x) {
  return 1.0/(1.0+exp(-(x-mu)/r));
}

int main() {
  gROOT->SetBatch(kTRUE);
  gROOT->SetStyle("Plain");

  TRandom3 rndm;
  rndm.SetSeed(123456);

  const int bins = 50;
  const double xMax = 100;
  const double counts = 50;
  double x[bins], eff[bins];
  double exl[bins], exh[bins], eeffl[bins], eeffh[bins];

  feldman_cousins fc;
  //  alpha = 1 - CL
  const double alpha = (1-0.682);
  fc.init(alpha);
  TH1D histo("histo", "efficiency", bins, 0, xMax);
  for(int i = 0; i < bins; ++i) {
    x[i] = (double(i) + 0.5) * xMax / bins; 
    double efficiency = sigmoid(x[i]);
    int n0 = rndm.Poisson(counts);
    int n1 = rndm.Binomial(n0, efficiency);
    eff[i] = double(n1)/double(n0);
    exl[i] = exh[i] = 0;
    fc.calculate(n1, n0);
    eeffl[i] = eff[i] - fc.lower();
    eeffh[i] = fc.upper() - eff[i];
    histo.SetBinContent(i+1,eff[i]); 
  }
  TGraphAsymmErrors graph(bins, x, eff, exl, exh, eeffl, eeffh);
  graph.SetTitle("efficiency");
  graph.SetMarkerColor(kBlue);
  graph.SetMarkerStyle(21);
  graph.SetLineWidth(1);
  graph.SetLineColor(kBlue);
  TCanvas c;
  histo.Draw();
  graph.Draw("P");
  c.SaveAs("testFeldmanCousinsBinomial.eps");
}

