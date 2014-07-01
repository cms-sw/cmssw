#include <iostream>
#include <TFile.h>
#include <TDirectory.h>
#include <TCanvas.h>
#include <TH1D.h>
#include <TProfile.h>

void GetResol()
{
  TFile * file = new TFile("redrawed_3.root", "READ");
  TFile * funcFile = new TFile("3_MuScleFit.root", "READ");

  TDirectory * dir = (TDirectory*) file->Get("hResolPtGenVSMu");
  TDirectory * funcDir = (TDirectory*) funcFile->Get("hFunctionResolPt");

  TH1D * histo = (TH1D*)dir->Get("hResolPtGenVSMu_ResoVSEta_resol");
  TProfile * funcProfile = (TProfile*) funcDir->Get("hFunctionResolPt_ResoVSEta_prof");

  TH1D * funcHisto = new TH1D(TString(funcProfile->GetName())+"histo", TString(funcProfile->GetTitle())+"histo", funcProfile->GetNbinsX(), histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());
  for( int i=1; i<=funcHisto->GetNbinsX(); ++i ) {
    funcHisto->SetBinContent(i, funcProfile->GetBinContent(i));
    // std::cout << "error["<<i<<"] = " << funcHisto->GetBinContent(i)*(1 - (0.7311228/0.781686)) << std::endl;
    funcHisto->SetBinError( i, funcHisto->GetBinContent(i)*(1 - (0.73391182/0.73723)) ); // (0.7648316/0.781686)) );
    // std::cout << "value["<<i<<"] = " << funcHisto->GetBinContent(i) << " +- " <<  funcHisto->GetBinError(i) << std::endl;
  }

  funcHisto->Draw("E5");
  // funcHisto->SetLineColor(kRed);
  // funcHisto->SetMarkerColor(kRed);
  funcHisto->SetFillColor(kGray);
  histo->Draw("SAME");

}
