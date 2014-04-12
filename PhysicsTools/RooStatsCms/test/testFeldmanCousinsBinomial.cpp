#include "TROOT.h"
#include "TStyle.h"
#include "TRandom3.h"
#include "TH1.h"
#include "TGraphAsymmErrors.h" 
#include "TCanvas.h"
#include <cmath>

#if (defined (STANDALONE) or defined (__CINT__) )
    #include "FeldmanCousinsBinomialInterval.h"
    #include "ClopperPearsonBinomialInterval.h"
#else
    #include "PhysicsTools/RooStatsCms/interface/FeldmanCousinsBinomialInterval.h"
    #include "PhysicsTools/RooStatsCms/interface/ClopperPearsonBinomialInterval.h"
#endif


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
  double exl[bins], exh[bins], eefflFC[bins], eeffhFC[bins], eefflCP[bins], eeffhCP[bins];

  FeldmanCousinsBinomialInterval fc;
  ClopperPearsonBinomialInterval cp;
  //  alpha = 1 - CL
  const double alpha = (1-0.682);
  fc.init(alpha);
  cp.init(alpha);
  TH1D histo("histo", "Efficiency", bins, 0, xMax);
  for(int i = 0; i < bins; ++i) {
    x[i] = (double(i) + 0.5) * xMax / bins; 
    double efficiency = sigmoid(x[i]);
    int n0 = rndm.Poisson(counts);
    int n1 = rndm.Binomial(n0, efficiency);
    eff[i] = double(n1)/double(n0);
    histo.SetBinContent(i+1,eff[i]); 
    exl[i] = exh[i] = 0;
    fc.calculate(n1, n0);
    eefflFC[i] = eff[i] - fc.lower();
    eeffhFC[i] = fc.upper() - eff[i];
    cp.calculate(n1, n0);
    eefflCP[i] = eff[i] - cp.lower();
    eeffhCP[i] = cp.upper() - eff[i];
  }
  TGraphAsymmErrors graphFC(bins, x, eff, exl, exh, eefflFC, eeffhFC);
  TGraphAsymmErrors graphCP(bins, x, eff, exl, exh, eefflCP, eeffhCP);
  graphFC.SetTitle("efficiency (Feldman-Cousins intervals) ");
  graphCP.SetTitle("efficiency (Clopper-Pearson intervals)");
  graphFC.SetMarkerColor(kBlue);
  graphFC.SetMarkerStyle(21);
  graphFC.SetLineWidth(1);
  graphFC.SetLineColor(kBlue);
  graphCP.SetMarkerColor(kRed);
  graphCP.SetMarkerStyle(21);
  graphCP.SetLineWidth(1);
  graphCP.SetLineColor(kRed);
  TCanvas c;
  gStyle->SetOptStat(0);
  histo.SetTitle("Efficiency with Feldman-Cousins intervals"); 
  histo.Draw();
  graphFC.Draw("P");
  c.SaveAs("testFeldmanCousinsBinomial.eps");
  histo.SetTitle("Efficiency with Clopper-Pearson intervals"); 
  histo.Draw();
  graphCP.Draw("P");
  c.SaveAs("testClopperPearsonBinomial.eps");
}

