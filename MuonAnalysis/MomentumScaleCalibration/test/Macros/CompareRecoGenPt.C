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

#include <iostream>

using namespace std;

/// Helper function getting the histogram from file.
TProfile * getHistogram( const TString & fileName )
{
  TFile * file = new TFile(fileName, "READ");
  if( file == 0 ) {
    cout << "Wrong file: " << fileName << endl;
    exit(1);
  }
  TDirectory * dir = (TDirectory*) file->Get("hPtRecoVsPtGen");
  if( dir == 0 ) {
    cout << "Wrong directory for file: " << fileName << endl;
    exit(1);
  }
  TProfile * profile = (TProfile*) dir->Get("hPtRecoVsPtGenProf");
  if( profile == 0 ) {
    cout << "Wrong histogram for file: " << fileName << endl;
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
void saveHistograms( TH1F * histo1, TH1F * histo2 )
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

void CompareRecoGenPt( const TString & fileName1 = "0_MuScleFit.root",
                       const TString & fileName2 = "1_MuScleFit.root" )
{
  TProfile * profile1 = getHistogram( fileName1 );
  TProfile * profile2 = getHistogram( fileName2 );

  int xBins = profile1->GetNbinsX();
  if( xBins != profile2->GetNbinsX() ) {
    cout << "Wrong number of bins" << endl;
    exit(1);
  }

  // Loop on all bins and fill a histogram with the mean values.
  // Fill also a histogram with the rms values.

  TH1F * meanHisto1 = makeHistogram(profile1, "mean");
  TH1F * meanHisto2 = makeHistogram(profile2, "mean");
  TH1F * rmsHisto1 = makeHistogram(profile1, "rms");
  TH1F * rmsHisto2 = makeHistogram(profile2, "rms");
  for( int iBin = 1; iBin <= xBins; ++iBin ) {
    meanHisto1->SetBinContent( iBin, profile1->GetBinContent(iBin) );
    meanHisto2->SetBinContent( iBin, profile2->GetBinContent(iBin) );
    rmsHisto1->SetBinContent( iBin, profile1->GetBinError(iBin) );
    rmsHisto2->SetBinContent( iBin, profile2->GetBinError(iBin) );
  }

  TCanvas * canvas = new TCanvas("before", "before corrections", 1000, 800);
  canvas->Divide(2);
  canvas->cd(1);
  saveHistograms(meanHisto1, meanHisto2);
  canvas->cd(2);
  saveHistograms(rmsHisto1, rmsHisto2);



}
