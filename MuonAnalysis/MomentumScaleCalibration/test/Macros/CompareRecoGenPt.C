/**
 * This compiled macro compares the rms on the difference between reco and gen
 * pt for muons in two different files.
 * It prints the rms distributions superimposed.
 * It can be used to verify the effect of corrections.
 */

#include "TFile.h"
#include "TDirectory.h"
#include "TString.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TF1.h"
#include "TROOT.h"
#include "TStyle.h"

#include <sstream>
#include <iostream>
#include <iomanip>

// #include "boost/lexical_cast.hpp"

/// Helper function getting the histogram from file.
TProfile * getHistogram( const TString & fileName )
{
  TFile * file = new TFile(fileName, "READ");
  if( file == 0 ) {
    std::cout << "Wrong file: " << fileName << std::endl;
    exit(1);
  }
  TDirectory * dir = (TDirectory*) file->Get("hPtRecoVsPtGen");
  if( dir == 0 ) {
    std::cout << "Wrong directory for file: " << fileName << std::endl;
    exit(1);
  }
  TProfile * profile = (TProfile*) dir->Get("hPtRecoVsPtGenProf");
  if( profile == 0 ) {
    std::cout << "Wrong histogram for file: " << fileName << std::endl;
    exit(1);
  }
  return profile;
}

/// Helper function building the histogram from the TProfile settings.
TH1F * makeHistogram( const TProfile * profile, const TString & name )
{
  return new TH1F(TString(profile->GetName())+name, TString(profile->GetTitle())+" "+name, profile->GetNbinsX(), profile->GetXaxis()->GetXmin(), profile->GetXaxis()->GetXmax());
}

/// Helper function to write the histograms to file.
void saveHistograms( TH1 * histo1, TH1 * histo2 )
{
  histo1->Draw();
  histo2->SetLineColor(kRed);
  histo2->Draw("Same");
  TLegend *leg = new TLegend(0.65,0.85,1,1);
  leg->SetFillColor(0);
  leg->AddEntry(histo1,"before calibration","L");
  leg->AddEntry(histo2,"after calibration","L");
  leg->Draw("same");
}

#include "TPaveText.h"
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

  double error = func->GetParError(0);
  a << std::setprecision(precision(error)) << std::fixed << func->GetParameter(0);
  // fit1 += boost::lexical_cast<string>(1);
  fit1 += a.str() + "+-";
  a.str("");
  a << error;
  fit1 += a.str();
  a.str("");

  error = func->GetParError(1);

  a << std::setprecision(precision(error)) << std::fixed << func->GetParameter(1);
  fit2 += a.str() + "+-";
  a.str("");
  a << func->GetParError(1);
  fit2 += a.str();
  a.str("");
  a << std::setprecision(1) << std::fixed << func->GetChisquare();
  fit3 += a.str() + "/";
  a.str("");
  a << std::setprecision(0) << std::fixed << func->GetNDF();
  fit3 += a.str();
}

void CompareRecoGenPt( const TString & fileNum1 = "0",
                       const TString & fileNum2 = "1" )
{
  TFile * outputFile = new TFile("CompareRecoGenPt.root", "RECREATE");

  TProfile * profile1 = getHistogram( fileNum1+"_MuScleFit.root" );
  profile1->SetXTitle("gen muon Pt (GeV)");
  profile1->SetYTitle("reco muon Pt (GeV)");
  TProfile * profile2 = getHistogram( fileNum2+"_MuScleFit.root" );

  int xBins = profile1->GetNbinsX();
  if( xBins != profile2->GetNbinsX() ) {
    std::cout << "Wrong number of bins" << std::endl;
    exit(1);
  }

  // Loop on all bins and fill a histogram with the mean values.
  // Fill also a histogram with the rms values.

  outputFile->cd();

  TH1F * meanHisto1 = makeHistogram(profile1, "mean");
  TH1F * meanHisto2 = makeHistogram(profile2, "mean");
  TH1F * rmsHisto1 = makeHistogram(profile1, "rms");
  TH1F * rmsHisto2 = makeHistogram(profile2, "rms");
  for( int iBin = 1; iBin <= xBins; ++iBin ) {
//     if( profile1->GetBinError(iBin) != 0 ) {
      meanHisto1->SetBinContent( iBin, profile1->GetBinContent(iBin) );
      meanHisto1->SetBinError( iBin, profile1->GetBinError(iBin) );
//     }
//     if( profile2->GetBinError(iBin) ) {
      meanHisto2->SetBinContent( iBin, profile2->GetBinContent(iBin) );
      meanHisto2->SetBinError( iBin, profile2->GetBinError(iBin) );
//     }
    rmsHisto1->SetBinContent( iBin, profile1->GetBinError(iBin) );
    rmsHisto2->SetBinContent( iBin, profile2->GetBinError(iBin) );
  }

  // Setting all weigths to 1 ("W" option) because of Profile errors for low statistics bins biasing the fit

  // meanHisto1->Fit("pol1", "W", "", 2, 1000);
  profile1->Fit("pol1", "W", "", 0, 1000);
  TF1 * func1 = profile1->GetFunction("pol1");
  // TF1 * func1 = meanHisto1->GetFunction("pol1");
  func1->SetLineWidth(1);
  func1->SetLineColor(kBlack);

  profile2->Fit("pol1", "W", "", 0, 1000);
  // meanHisto2->Fit("pol1", "W", "", 2, 1000);
  TF1 * func2 = profile2->GetFunction("pol1");
  // TF1 * func2 = meanHisto2->GetFunction("pol1");
  func2->SetLineWidth(1);
  func2->SetLineColor(kRed);

  TCanvas * canvas = new TCanvas("before", "before corrections", 1000, 800);
  // canvas->Divide(2);
  canvas->cd();
  // canvas->cd(1);
  // canvas->cd(2);
  // saveHistograms(rmsHisto1, rmsHisto2);
  // saveHistograms(meanHisto1, meanHisto2);
  saveHistograms(profile1, profile2);
  func1->Draw("same");
  func2->Draw("same");

  TString fit11("a = ");
  TString fit12("b = ");
  TString fit13("#chi^2/ndf = ");
  getParameters(func1, fit11, fit12, fit13);
  PaveText pt1(0.45, 0.15);
  pt1.AddText("before:");
  pt1.AddText(fit11);
  pt1.AddText(fit12);
  pt1.AddText(fit13);
  pt1.Draw("same");

  TString fit21("a = ");
  TString fit22("b = ");
  TString fit23("#chi^2/ndf = ");
  getParameters(func2, fit21, fit22, fit23);
  PaveText pt2(0.65, 0.15);
  pt2.SetTextColor(2);
  pt2.AddText("after:");
  pt2.AddText(fit21);
  pt2.AddText(fit22);
  pt2.AddText(fit23);
  pt2.Draw("same");
  gStyle->SetOptStat(0);

  canvas->Write();

  outputFile->Write();
  outputFile->Close();

//   TLegend *leg = new TLegend(0.2,0.4,0.4,0.6);
//   leg->SetFillColor(0);
//   leg->AddEntry(func1,"fit of before","L");
//   leg->AddEntry(func2,"fit of after","L");
//   leg->Draw("same");
}

