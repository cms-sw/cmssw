#include "TROOT.h"
#include "TStyle.h"
#include "TFile.h"
#include "TH1.h"
#include "TGraphAsymmErrors.h" 
#include "TCanvas.h"
#include <cmath>

#if (defined (STANDALONE) or defined (__CINT__) )
    #include "ClopperPearsonBinomialInterval.h"
#else
    #include "PhysicsTools/RooStatsCms/interface/ClopperPearsonBinomialInterval.h"
#endif




int main() {
  gROOT->SetBatch(kTRUE);
  gROOT->SetStyle("Plain");

  TFile * root_file = new TFile("L2Mu9MuTriggerStudy_withiutQualityCuts.root","read");
  TH1D * den = (TH1D*) root_file->Get("MuTriggerAnalyzer/hTrigMuonPtDenS");
  TH1D * num = (TH1D*) root_file->Get("MuTriggerAnalyzer/hTrigMuonPtNumS");
 

  const int bins = den->GetXaxis()->GetNbins();
  const double xMax = den->GetXaxis()->GetXmax();
  double * x = new double[bins];
  double *eff = new double[bins];
  double * exl= new double[bins];
  double *exh = new double[bins];
  double *  eefflCP= new double[bins];
  double * eeffhCP = new double[bins];

   ClopperPearsonBinomialInterval cp;
  //  alpha = 1 - CL
  const double alpha = (1-0.682);
  cp.init(alpha);
  TH1D histo("histo", "Efficiency", bins, 0, xMax);

  for(int i = 0; i < bins; ++i) {
    x[i] = (double(i - 0.5 )) * (xMax ) / (bins ); 
    int n0 = den->GetBinContent(i);
    int n1 = num->GetBinContent(i);
     if ( n0!=0) {
      eff[i] = double(n1)/double(n0); 
      histo.SetBinContent(i,eff[i]); 
      exl[i] = exh[i] = 0;
      cp.calculate(n1, n0);
      eefflCP[i] = eff[i] - cp.lower();
      eeffhCP[i] = cp.upper() - eff[i];
      } else { 
      eff[i]=0;
      histo.SetBinContent(i,eff[i]); 
      exl[i] = exh[i] = 0;
      //cp.calculate(n1, n0);
      eefflCP[i] = 0;
      eeffhCP[i] = 0;

      }
     //histo.SetBinContent(i+1,eff[i]); 
     //exl[i] = exh[i] = 0;
     //cp.calculate(n1, n0);
     //eefflCP[i] = eff[i] - cp.lower();
     //eeffhCP[i] = cp.upper() - eff[i];
  }
  TGraphAsymmErrors graphCP(bins, x, eff, exl, exh, eefflCP, eeffhCP);
  graphCP.SetTitle("HLT_L2Mu9 efficiency (Clopper-Pearson intervals)");
  graphCP.SetMarkerColor(kRed);
  graphCP.SetMarkerStyle(21);
  graphCP.SetLineWidth(1);
  graphCP.SetLineColor(kRed);
  TCanvas c;
  gStyle->SetOptStat(0);
  histo.SetTitle("HLT_L2Mu9 Efficiency with Clopper-Pearson intervals"); 
  histo.Draw();
  histo.SetLineColor(kWhite);
  graphCP.Draw("P");
  c.SaveAs("L2Mu9_efficiency_NoQualityCuts.eps");
}

