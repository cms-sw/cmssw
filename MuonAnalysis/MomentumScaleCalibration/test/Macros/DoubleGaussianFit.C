#include <TFile.h>
#include <TH1F.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TPaveText.h>
#include <TStyle.h>

#include <sstream>
#include <iostream>
#include <iomanip>

/// Helper class holding a TPaveText for better formatting and predefined options
class PaveText
{
 public:
  PaveText(const double & textX = 0.7, const double & textY = 0.4 )
  {
    paveText_ = new TPaveText(textX, textY, textX+0.2, textY+0.17, "NDC");
  }
  void AddText(const TString & text)
  {
    paveText_->AddText(text);
  }
  void Draw(const TString & option)
  {
    paveText_->SetFillColor(0); // text is black on white
    paveText_->SetTextSize(0.03);
    paveText_->SetBorderSize(0);
    paveText_->SetTextAlign(12);
    paveText_->Draw(option);
  }
  void SetTextColor(const int color)
  {
    paveText_->SetTextColor(color);
  }
 protected:
  TPaveText * paveText_;
};

/**
 * Compute the precision to give to the stream operator so that the passed number
 * will be printed with two significant figures.
 */
int precision( const double & value )
{
  // Counter gives the precision
  int precision = 1;
  int k=1;
  while( int(value*k) == 0 ) {
    k*=10;
    ++precision;
  }
  return precision;
}

/// Helper function to extract and format the text for the fitted parameters
void getParameters( const TF1 * func, TString & fit1, TString & fit2, TString & fit3 )
{
  std::stringstream a;

  double error = func->GetParError(1);
  a << std::setprecision(precision(error)) << std::fixed << func->GetParameter(1);
  fit1 += a.str() + "+-";
  a.str("");
  a << error;
  fit1 += a.str();
  a.str("");

  error = func->GetParError(2);

  a << std::setprecision(precision(error)) << std::fixed << func->GetParameter(2);
  fit2 += a.str() + "+-";
  a.str("");
  a << func->GetParError(2);
  fit2 += a.str();
  a.str("");
  a << std::setprecision(1) << std::fixed << func->GetChisquare();
  fit3 += a.str() + "/";
  a.str("");
  a << std::setprecision(0) << std::fixed << func->GetNDF();
  fit3 += a.str();
}

void DoubleGaussianFit()
{
  TFile * file_0 = new TFile("0_MuScleFit.root", "READ");
  TH1F * histo_0 = (TH1F*)file_0->Get("hRecBestRes_Mass");
  TFile * file_1 = new TFile("3_MuScleFit.root", "READ");
  TH1F * histo_1 = (TH1F*)file_1->Get("hRecBestRes_Mass");

  histo_0->Rebin(2);
  histo_1->Rebin(2);

  // TF1 *f1_0 = new TF1("f1", "gaus(0) + gaus(3)", 1., 5.);
  // f1_0->SetParameters(136., 3.096916, 0.03, 136, 3., 0.03);
  // f1_0->FixParameter(1, 3.096916);

  TF1 *f1_0 = new TF1("f1", "gaus(0)", 1., 5.);
  f1_0->SetParameters(136., 3.096916, 0.03);

  // TF1 *f1_1 = new TF1("f1", "gaus(0) + gaus(3)", 1., 5.);
  // f1_1->SetParameters(136., 3.096916, 0.03, 136, 3., 0.03);
  // f1_1->FixParameter(1, 3.096916);
  // histo_1->SetLineColor(kRed);
  // f1_1->SetLineColor(kRed);

  TF1 *f1_1 = new TF1("f1", "gaus(0)", 1., 5.);
  f1_1->SetParameters(136., 3.096916, 0.03);
  histo_1->SetLineColor(kRed);
  f1_1->SetLineColor(kRed);

  // TF1 *f1 = new TF1("f1", "gaus(0)", 1., 5.);
  // f1->SetParameters(136., 3., 0.03);

  histo_0->Fit(f1_0);
  histo_1->Fit(f1_1);

  TCanvas * canvas = new TCanvas("canvas", "canvas", 1000, 800);
  canvas->Draw();

  histo_0->Draw();
  histo_0->GetXaxis()->SetTitle("Mass (GeV)");
  histo_0->GetYaxis()->SetTitle("a.u.");
  histo_0->SetTitle("");
  histo_1->Draw("SAME");



  TString fit11("mean = ");
  TString fit12("sigma = ");
  TString fit13("Chi2/NDF = ");
  getParameters(f1_0, fit11, fit12, fit13);

  PaveText pt1(0.45, 0.15);
  pt1.AddText("before:");
  pt1.AddText(fit11);
  pt1.AddText(fit12);
  pt1.AddText(fit13);
  pt1.Draw("same");

  TString fit21("mean = ");
  TString fit22("sigma = ");
  TString fit23("Chi2/NDF = ");
  getParameters(f1_1, fit21, fit22, fit23);

  PaveText pt2(0.65, 0.15);
  pt2.SetTextColor(2);
  pt2.AddText("after:");
  pt2.AddText(fit21);
  pt2.AddText(fit22);
  pt2.AddText(fit23);
  pt2.Draw("same");

  gStyle->SetOptStat(0);

}
