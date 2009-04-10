#include "TH1F.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TStyle.h"

#include <iostream>

using namespace std;

/// Helper class containing the histograms
class histos
{
public:
  histos(TFile * inputFile)
  {
    mass     = dynamic_cast<TH1F*> (inputFile->Get("hRecBestRes_Mass")); 
    massProb = dynamic_cast<TProfile*> (inputFile->Get("Mass_P"));
    likePt   = dynamic_cast<TProfile*> (inputFile->Get("hLikeVSMu_LikelihoodVSPt_prof"));
    likePhi  = dynamic_cast<TProfile*> (inputFile->Get("hLikeVSMu_LikelihoodVSPhi_prof"));
    likeEta  = dynamic_cast<TProfile*> (inputFile->Get("hLikeVSMu_LikelihoodVSEta_prof"));
  }
  TH1F * mass;
  TProfile * massProb;
  TProfile * likePt;
  TProfile * likePhi;
  TProfile * likeEta;
};

/// Helper function to draw mass and mass probability histograms
void drawMasses(const double ResMass, const double ResHalfWidth, histos & h)
{
  h.mass->SetAxisRange(ResMass - ResHalfWidth, ResMass + ResHalfWidth);
  h.mass->SetLineColor(kRed);
  // h->mass->SetMarkerColor(kBlack);

  // To get the correct integral for rescaling determine the borders
  TAxis * xAxis = h.mass->GetXaxis();
  double xMin = xAxis->GetXmin();
  // The bins have all the same width, therefore we can use this
  double binWidth = xAxis->GetBinWidth(1);
  int xMinBin = int(((ResMass - ResHalfWidth) - xMin)/binWidth) + 1;
  int xMaxBin = int(((ResMass + ResHalfWidth) - xMin)/binWidth) + 1;
  cout << "xMinBin = " << xMinBin << endl;
  cout << "xMaxBin = " << xMaxBin << endl;
  cout << "binWidth = " << binWidth << endl;

  // Set a consistent binning
  // int massXbins = h.mass->GetNbinsX();
  int massXbins = xMaxBin - xMinBin;
  int massProbXbins = h.massProb->GetNbinsX();
//   if( massXbins > massProbXbins ) {
//     if ( massXbins % massProbXbins != 0 ) {
//       cout << "Warning: number of bins are not multiples: massXbins = " << massXbins << ", massProbXbins = " << massProbXbins << endl;
//       cout << "massXbins % massProbXbins = " << massXbins % massProbXbins << endl;
//     }
//   }
//   else if( massProbXbins % massXbins != 0 ) {
//     cout << "Warning: number of bins are not multiples: massXbins = " << massXbins << ", massProbXbins = " << massProbXbins << endl;
//     cout << "massProbXbins % massXbins = " << massProbXbins % massXbins << endl;
//   }
  if( massProbXbins > massXbins ) {
    cout << "massProbXbins("<<massProbXbins<<") > " << "massXbins("<<massXbins<<")" << endl;
    h.massProb->Rebin(massProbXbins/massXbins);
  }
  else if( massXbins > massProbXbins ) {
    cout << "massXbins("<<massXbins<<") > " << "massProbXbins("<<massProbXbins<<")" << endl;
    h.mass->Rebin(massXbins/massProbXbins);
  }

//   double normFactorPlus = (h.mass->Integral(xMinBin, xMaxBin, "width"))/(h.massProb->Integral("width"));
//   cout << "normFactorPlus = " << normFactorPlus << endl;
//   cout << "scaled plus massProb integral = " << h.massProb->Integral("width")*normFactorPlus << endl;;


  h.mass->SetAxisRange(ResMass - ResHalfWidth, ResMass + ResHalfWidth);
  h.massProb->SetAxisRange(ResMass - ResHalfWidth, ResMass + ResHalfWidth);
  double normFactor = (h.mass->Integral("width"))/(h.massProb->Integral("width"));
//   cout << "ResMass - ResHalfWidth = " << ResMass - ResHalfWidth << endl;
//   cout << "ResMass + ResHalfWidth = " << ResMass + ResHalfWidth << endl;
//   cout << "normFactor = " << normFactor << endl;
//   cout << "mass integral = " << h.mass->Integral("width") << endl;
//   cout << "massProb integral = " << h.massProb->Integral("width") << endl;
//   cout << "scaled massProb integral = " << h.massProb->Integral("width")*normFactor << endl;;
  h.massProb->SetLineColor(kBlue);
  h.massProb->SetNormFactor(normFactor);
  h.mass->SetMarkerColor(kRed);
  cout << (h.massProb->GetMaximum())*normFactor << endl;
  cout << h.mass->GetMaximum() << endl;
  // ATTENTION: put so that the maximum is correct
  // if( (h.massProb->GetMaximum())*normFactor > h.mass->GetMaximum() ) h.massProb->DrawCopy();
  cout << "(h.massProb->GetMaximum())*normFactor = " << (h.massProb->GetMaximum())*normFactor << endl;
  cout << "h.mass->GetMaximum() = " << h.mass->GetMaximum() << endl;
  if( (h.massProb->GetMaximum())*normFactor > h.mass->GetMaximum() ) {
    h.massProb->DrawCopy("PE");
    h.mass->DrawCopy("SAMEHISTO");
  }
  else {
    h.mass->DrawCopy("PE");
    h.massProb->DrawCopy("SAMEHISTO");
  }
}

/// Helper function to draw likelihood histograms
void drawLike(TProfile * likeHisto1, TProfile * likeHisto2)
{
  likeHisto1->SetMinimum(-8);
  likeHisto1->SetMaximum(-4);
  likeHisto1->Draw("");
  likeHisto2->SetLineColor(kRed);
  likeHisto2->Draw("SAME");
}

/// Function drawing the histograms
void Plot_mass(const TString & fileNameBefore = "0", const TString & fileNameAfter = "1") {

  gStyle->SetOptStat ("111111");
  gStyle->SetOptFit (1);
  
  double ResHalfWidth[6] = {20., 0.5, 0.5, 0.5, 0.2, 0.2};
  double ResMass[6] = {90.986, 10.3552, 10.0233, 9.4603, 3.68609, 3.0969};

  TFile * inputFile1 = new TFile(fileNameBefore+"_MuScleFit.root", "READ");
  TFile * inputFile2 = new TFile(fileNameAfter+"_MuScleFit.root", "READ");
  histos histos1(inputFile1);
  histos histos2(inputFile2);

  TFile * outputFile = new TFile("plotMassOutput.root", "RECREATE");

  TCanvas * Allres = new TCanvas ("Allres", "All resonances", 600, 600);
  Allres->Divide (2,3);
  for (int ires=0; ires<6; ires++) {
    Allres->cd(ires+1);
    drawMasses(ResMass[ires], ResHalfWidth[ires], histos1);
  }
  Allres->Write();

  TCanvas * Allres2 = new TCanvas ("Allres2", "All resonances", 600, 600);
  Allres2->Divide (2,3);
  for (int ires=0; ires<6; ires++) {
    Allres2->cd(ires+1);
    drawMasses(ResMass[ires], ResHalfWidth[ires], histos2);
  }
  Allres2->Write();

  TCanvas * LR = new TCanvas ("LR", "Likelihood and resolution before and after corrections", 600, 600 );
  LR->Divide (2,3);

  LR->cd(1);
  drawLike(histos1.likePt, histos2.likePt);
  LR->cd(3);
  drawLike(histos1.likeEta, histos2.likeEta);
  LR->cd(5);
  drawLike(histos1.likePhi, histos2.likePhi);

  LR->Write();

  TCanvas * G = new TCanvas ("G", "Mass before and after corrections", 600, 600 );
  G->Divide (3,3);

  TH1F * MuZ = histos1.mass;
  TH1F * McZ = histos2.mass;

  G->cd(1);
  MuZ->SetAxisRange (2.8, 3.4);
  MuZ->SetLineColor (kBlue);
  McZ->SetLineColor (kRed);
  MuZ->DrawCopy ("HISTO");
  McZ->DrawCopy ("SAME");

  G->cd(2);
  MuZ->SetAxisRange (9., 10.);
  McZ->SetAxisRange (9., 10.);
  MuZ->DrawCopy ("HISTO");
  McZ->DrawCopy ("SAME"); 

  G->cd(3);
  MuZ->SetAxisRange (70., 110.);
  McZ->SetAxisRange (70., 110.);
  MuZ->DrawCopy ("HISTO");
  McZ->DrawCopy ("SAME"); 

  G->cd(4);
  MuZ->SetAxisRange (2.8, 3.4);
  MuZ->Fit ("gaus", "", "", 3., 3.2);
  MuZ->DrawCopy();
  G->cd(7);
  McZ->SetAxisRange (2.8, 3.4);
  McZ->Fit ("gaus", "", "", 3., 3.2);
  McZ->DrawCopy();
  G->cd(5);
  MuZ->SetAxisRange (9., 10.);
  MuZ->Fit ("gaus", "", "", 9., 10.);
  MuZ->DrawCopy();
  G->cd(8);
  McZ->SetAxisRange (9., 10.);
  McZ->Fit ("gaus", "", "", 9., 10.);
  McZ->DrawCopy();
  G->cd(6);
  MuZ->SetAxisRange (70., 110.);
  MuZ->Fit ("gaus", "", "", 70., 110.);
  MuZ->DrawCopy();
  G->cd(9);
  McZ->SetAxisRange (70., 110.);
  McZ->Fit ("gaus", "", "", 70., 110.);
  McZ->DrawCopy();

  G->Write();
 
  TCanvas * Spectrum = new TCanvas ("Spectrum", "Mass before and after corrections", 600, 600 );
  Spectrum->cd(1);
  MuZ->SetAxisRange(1.,15.);
  McZ->SetAxisRange(1.,15.);
  McZ->SetLineColor(kRed);
  McZ->DrawCopy("HISTO");
  Spectrum->cd(2);
  MuZ->SetLineColor(kBlack);
  MuZ->DrawCopy("SAMEHISTO");

  Spectrum->Write();

  outputFile->Close();
}
