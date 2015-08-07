#include "TH1F.h"
#include "TProfile.h"
#include "TProfile.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TStyle.h"
#include "TROOT.h"

#include <iostream>


/// Helper class containing the histograms
class histos
{
public:
  histos(TFile * inputFile)
  {
    mass         = dynamic_cast<TH1F*>(inputFile->Get("hRecBestResAllEvents_Mass"));
    TDirectory * MassPdir = dynamic_cast<TDirectory*>(inputFile->Get("Mass_P"));
    massProb     = dynamic_cast<TProfile*>(MassPdir->Get("Mass_PProf"));
    // TDirectory * MassPdir = dynamic_cast<TDirectory*>(inputFile->Get("Mass_Probability"));
    // massProb     = dynamic_cast<TProfile*>(MassPdir->Get("Mass_Probability"));
    TDirectory * massFinePdir = dynamic_cast<TDirectory*>(inputFile->Get("Mass_fine_P"));
    massFineProb = dynamic_cast<TProfile*>(massFinePdir->Get("Mass_fine_PProf"));
    // TDirectory * massFinePdir = dynamic_cast<TDirectory*>(inputFile->Get("Mass_fine_Probability"));
    // massFineProb = dynamic_cast<TProfile*>(massFinePdir->Get("Mass_fine_Probability"));
    likePt       = dynamic_cast<TProfile*>(inputFile->Get("hLikeVSMu_LikelihoodVSPt_prof"));
    likePhi      = dynamic_cast<TProfile*>(inputFile->Get("hLikeVSMu_LikelihoodVSPhi_prof"));
    likeEta      = dynamic_cast<TProfile*>(inputFile->Get("hLikeVSMu_LikelihoodVSEta_prof"));
  }
  TH1F * mass;
  TProfile * massProb;
  TProfile * massFineProb;
  TProfile * likePt;
  TProfile * likePhi;
  TProfile * likeEta;
};

/// Helper function to compute the bins corresponding to an interval in the x axis
int getXbins(const TH1 * h, const double & xMin, const double & xMax)
{
  // To get the correct integral for rescaling determine the borders
  const TAxis * xAxis = h->GetXaxis();
  double xAxisMin = xAxis->GetXmin();
  // The bins have all the same width, therefore we can use this
  double binWidth = xAxis->GetBinWidth(1);
  int xMinBin = int((xMin - xAxisMin)/binWidth) + 1;
  int xMaxBin = int((xMax - xAxisMin)/binWidth) + 1;
  // std::cout << "xMinBin = " << xMinBin << std::endl;
  // std::cout << "xMaxBin = " << xMaxBin << std::endl;
  // std::cout << "binWidth = " << binWidth << std::endl;
  return( xMaxBin - xMinBin );
}

/// Helper function to draw mass and mass probability histograms
void drawMasses(const double ResMass, const double ResHalfWidth, histos & h, const int ires, const int rebin = 1)
{
  TH1F * mass = (TH1F*)h.mass->Clone();
  TProfile * massProb = 0;
  mass->Rebin(rebin);
  // Use massProb for the Z and fineMass for the other resonances
  if( ires == 0 ) massProb = (TProfile*)h.massProb->Clone();
  else massProb = (TProfile*)h.massFineProb->Clone();
  mass->SetAxisRange(ResMass - ResHalfWidth, ResMass + ResHalfWidth);
  mass->SetLineColor(kRed);

  // Set a consistent binning
  int massXbins = getXbins( mass, (ResMass - ResHalfWidth), (ResMass + ResHalfWidth) );
  int massProbXbins = getXbins( massProb, (ResMass - ResHalfWidth), (ResMass + ResHalfWidth) );

  if( massProbXbins > massXbins && massXbins != 0 ) {
    std::cout << "massProbXbins("<<massProbXbins<<") > " << "massXbins("<<massXbins<<")" << std::endl;
    std::cout << "massProb = " << massProb << std::endl;
    std::cout << "mass = " << mass << std::endl;
    massProb->Rebin(massProbXbins/massXbins);
  }
  else if( massXbins > massProbXbins && massProbXbins != 0 ) {
    std::cout << "massXbins("<<massXbins<<") > " << "massProbXbins("<<massProbXbins<<")" << std::endl;
    std::cout << "massProb = " << massProb << std::endl;
    std::cout << "mass = " << mass << std::endl;
    mass->Rebin(massXbins/massProbXbins);
  }

  mass->SetAxisRange(ResMass - ResHalfWidth, ResMass + ResHalfWidth);
  massProb->SetAxisRange(ResMass - ResHalfWidth, ResMass + ResHalfWidth);

  double massProbIntegral = massProb->Integral("width");
  // double massProbIntegral = massProb->Integral();
  double normFactor = 1.;
  if( massProbIntegral != 0 ) normFactor = (mass->Integral("width"))/massProbIntegral;
  // if( massProbIntegral != 0 ) normFactor = (mass->Integral())/massProbIntegral;

  massProb->SetLineColor(kBlue);
  massProb->Scale(normFactor);
  if( ires == 3 ) {
    std::cout << "massProbIntegralWidth = " << massProb->Integral("width") << std::endl;
    std::cout << "massIntegralWidth = " << mass->Integral("width") << std::endl;
    std::cout << "massProbIntegral = " << massProb->Integral() << std::endl;
    std::cout << "massIntegral = " << mass->Integral() << std::endl;
  }

  mass->SetMarkerColor(kRed);
  // ATTENTION: put so that the maximum is correct
  if( (massProb->GetMaximum()) > mass->GetMaximum() ) {
    massProb->Draw("PE");
    mass->Draw("PESAMEHISTO");
  }
  else {
    mass->Draw("PE");
    massProb->Draw("PESAMEHISTO");
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
void Plot_mass(const TString & fileNameBefore = "0", const TString & fileNameAfter = "1", const int rebin = 1) {

  gROOT->SetBatch(true);
  gStyle->SetOptStat ("111111");
  gStyle->SetOptFit (1);

  double ResHalfWidth[6] = {20., 0.5, 0.5, 0.5, 0.2, 0.2};
  // double ResHalfWidth[6] = {20., 0.25, 0.25, 0.25, 0.2, 0.2};
  // double ResHalfWidth[6] = {20., 0.35, 0.35, 0.35, 0.2, 0.2};
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
    drawMasses(ResMass[ires], ResHalfWidth[ires], histos1, ires, rebin);
  }
  Allres->Write();
  TCanvas * AllresTogether = new TCanvas ("AllresTogether", "All resonances together", 600, 600);
  AllresTogether->Divide(2,2); 
  // The J/Psi + Psi2S
  AllresTogether->cd(1);
  drawMasses(ResMass[5], 2, histos1, 3, rebin);
 // Draw also the Upsilons
  AllresTogether->cd(2);
  drawMasses(ResMass[2], 2, histos1, 3, rebin);
  // All low Pt resonances
  AllresTogether->cd(3);
  drawMasses(6., 6, histos1, 3, rebin);
  // All resonances
  AllresTogether->cd(4);
  drawMasses(50., 50, histos1, 0, rebin);
  AllresTogether->Write();

  TCanvas * Allres2 = new TCanvas ("Allres2", "All resonances", 600, 600);
  Allres2->Divide (2,3);
  for (int ires=0; ires<6; ires++) {
    Allres2->cd(ires+1);
    drawMasses(ResMass[ires], ResHalfWidth[ires], histos2, ires, rebin);
  }
  Allres2->Write();
  TCanvas * AllresTogether2 = new TCanvas ("AllresTogether2", "All resonances together", 600, 600);
  AllresTogether2->Divide(2,2);
  // The J/Psi + Psi2S
  AllresTogether2->cd(1);
  drawMasses(ResMass[5], 2, histos2, 3, rebin);
  // Draw also the Upsilons
  AllresTogether2->cd(2);
  drawMasses(ResMass[2], 2, histos2, 3, rebin);
  // All low Pt resonances
  AllresTogether2->cd(3);
  drawMasses(6., 6, histos2, 3, rebin);
  // All resonances
  AllresTogether2->cd(4);
  drawMasses(50., 50, histos2, 0, rebin);
  AllresTogether2->Write();

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

