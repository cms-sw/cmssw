#include "ScaleFraction.C"
#include "TFile.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TLegend.h"
#include "TString.h"

/**
 * This macro can be used to scale the probability distributions to have the same normalization of the
 * corresponding mass distribution in each mass window. <br>
 * The present form assumes Upsilons.
 */

bool first = true;

void mergeAll(const TString & inputFileName, const int color1, const int color2, TLegend * legend, const TString & when, const TString & options = "")
{
  TFile * inputFile = new TFile(inputFileName);
  TH1F * histo1 = (TH1F*)inputFile->Get("hRecBestRes_Mass");
  TDirectory * profileDir = (TDirectory*)inputFile->Get("Mass_fine_P");
  TProfile * histo2 = (TProfile*)profileDir->Get("Mass_fine_PProf");

  histo1->Rebin(8);
  histo2->Rebin(2);

  ScaleFraction scaleFraction;
  pair<TH1*, TH1*> newHistosUpsilon1S = scaleFraction.scale(histo1, histo2, 9, 9.8, "1");

  if( first ) {
    newHistosUpsilon1S.first->Draw(options);
    first = false;
  }
  else {
    newHistosUpsilon1S.first->Draw(options+"same");
  }
  newHistosUpsilon1S.first->SetLineColor(color1);
  newHistosUpsilon1S.second->Scale(newHistosUpsilon1S.first->Integral("width")/newHistosUpsilon1S.second->Integral("width"));
  newHistosUpsilon1S.second->Draw("same");
  newHistosUpsilon1S.second->SetLineColor(color2);

  legend->AddEntry(newHistosUpsilon1S.first, "mass "+when+" correction");
  legend->AddEntry(newHistosUpsilon1S.second, "mass prob "+when+" correction");

  pair<TH1*, TH1*> newHistosUpsilon2S = scaleFraction.scale(histo1, histo2, 9.8, 10.2, "2");

  newHistosUpsilon2S.first->Draw(options+"same");
  newHistosUpsilon2S.first->SetLineColor(color1);
  newHistosUpsilon2S.second->Scale(newHistosUpsilon2S.first->Integral("width")/newHistosUpsilon2S.second->Integral("width"));
  newHistosUpsilon2S.second->Draw("same");
  newHistosUpsilon2S.second->SetLineColor(color2);

  pair<TH1*, TH1*> newHistosUpsilon3S = scaleFraction.scale(histo1, histo2, 10.2, 10.8, "3");

  newHistosUpsilon3S.first->Draw(options+"same");
  newHistosUpsilon3S.first->SetLineColor(color1);
  newHistosUpsilon3S.second->Scale(newHistosUpsilon3S.first->Integral("width")/newHistosUpsilon3S.second->Integral("width"));
  newHistosUpsilon3S.second->Draw("same");
  newHistosUpsilon3S.second->SetLineColor(color2);


//   newHistosUpsilon1S.first->SetLineWidth(2);
//   newHistosUpsilon1S.second->SetLineWidth(2);
//   newHistosUpsilon2S.first->SetLineWidth(2);
//   newHistosUpsilon2S.second->SetLineWidth(2);
//   newHistosUpsilon3S.first->SetLineWidth(2);
//   newHistosUpsilon3S.second->SetLineWidth(2);

  newHistosUpsilon1S.first->GetXaxis()->SetTitle("Mass (GeV)");
  newHistosUpsilon1S.first->GetYaxis()->SetTitle("arbitrary units");
}

void MergeScaled()
{
  TLegend * legend = new TLegend(0.7,0.71,0.98,1.);
  legend->SetTextSize(0.02);
  legend->SetFillColor(0); // Have a white background

  mergeAll("0_MuScleFit.root", 1, 2, legend, "before");
  mergeAll("2_MuScleFit.root", 3, 4, legend, "after");

  legend->Draw("SAME");
}
