#include "TFile.h"
#include "TCanvas.h"
#include "TH2D.h"
#include "TProfile2D.h"
#include <iostream>
#include <map>
#include <sstream>
#include <cmath>

/**
 * This macro compares the pdf used in the likelihood fit with the mass distributions. <br>
 * It can be used to check that the fit converged well in different bins of pt, eta, phi...
 */

void projectHisto(const TH2 * histo, std::map<unsigned int, TH1D*> & profileMap)
{
  unsigned int binsX = histo->GetNbinsX();
  unsigned int binsY = histo->GetNbinsY();
  std::cout << "binsX = " << binsX << " binsY = " << binsY << std::endl;
  for( unsigned int i=1; i<binsX; ++i ) {
    std::stringstream ss;
    ss << i;
    TH1D * projected = histo->ProjectionY(TString(histo->GetName())+"_"+ss.str(), i, i);
    profileMap.insert(std::make_pair(i, projected));
    // if( i == 20 ) {
    //   projected->Draw();
    // }
  }
}

void CompareProbs()
{
  TFile * inputFile = new TFile("2_MuScleFit.root", "READ");

  // Take the pdf histogram
  TProfile2D * profile = (TProfile2D*)inputFile->FindObjectAny("hMassProbVsMu_fine_MassVSEta");
  TH2D * probHisto = profile->ProjectionXY();
  probHisto->SetName(TString(profile->GetName())+"_TH2D");
 
  std::map<unsigned int, TH1D*> projectProb;
  projectHisto(probHisto, projectProb);

  // Take the mass histogram
  TH2F * massHisto = (TH2F*)inputFile->FindObjectAny("hRecBestResVSMu_MassVSEta");
  std::map<unsigned int, TH1D*> projectMass;
  projectHisto(massHisto, projectMass);

  // Draw the histograms
  if( projectProb.size() != projectMass.size() ) {
    std::cout << "Error: bins for pdf = " << projectProb.size() << " != bins for mass = " << projectMass.size() << std::endl;
    return;
  }

  std::map<unsigned int, TH1D*>::const_iterator pIt = projectProb.begin();
  unsigned int totPads = 0;
  for( ; pIt != projectProb.end(); ++pIt ) {
    if( pIt->second->GetEntries() != 0 ) ++totPads;
  }
  TCanvas * newCanvas = new TCanvas("newCanvas", "newCanvas", 1000, 800);
  std::cout << "totPads = " << totPads << ", totPads%2 = " << totPads%2 << std::endl;
  unsigned int psq(sqrt(totPads));
  if( psq*psq == totPads ) newCanvas->Divide(psq, psq);
  else if( (psq+1)*psq >= totPads ) newCanvas->Divide(psq+1, psq);
  else newCanvas->Divide(psq+1, psq+1);
  newCanvas->Draw();
  pIt = projectProb.begin();
  unsigned int pad = 1;
  for( ; pIt != projectProb.end(); ++pIt ) {
    if( pIt->second->GetEntries() == 0 ) continue;
    // if( pIt->first != 20 ) continue;
    newCanvas->cd(pad);

    // double xMin = 3.0969-0.2;
    // double xMax = 3.0969+0.2;
    double xMin = 2.85;
    double xMax = 3.3;

    TH1D * mP = pIt->second;
    mP->Scale(1/mP->Integral(mP->FindBin(xMin), mP->FindBin(xMax)));
    mP->SetLineColor(2);
    mP->SetMarkerColor(2);
    mP->SetMarkerStyle(2);
    mP->SetMarkerSize(0.2);

    TH1D * mM = projectMass[pIt->first];
    mM->Scale(1/mM->Integral(mM->FindBin(xMin), mM->FindBin(xMax)));
  
    mP->Draw();
    std::cout << "pad = " << pad << std::endl;
    mP->GetXaxis()->SetRangeUser(xMin, xMax);
    mM->Draw("same");
    ++pad;
  }
}
