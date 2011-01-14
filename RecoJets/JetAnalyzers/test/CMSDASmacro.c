#include "TFile.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TF1.h"
#include <iostream>

using std::cout;

void CMSDASmacro() {
  TFile* rootfile = new TFile("histos.root");
  rootfile->cd("dijetAna");
  gDirectory->ls();

  //Get pointers to some histograms we will want
  TH1D* h_CorDijetMass=(TH1D*)gROOT->FindObject("hCorDijetMass");
  if(h_CorDijetMass==0) { std::cout << "fdsea" << std::endl; }

  TCanvas *cDijetMass=new TCanvas();
  cDijetMass->SetLogy();

  h_CorDijetMass->Draw();

  TF1 *massfit= new TF1("Dijet Mass Spectrum", "[0]*pow(1-x/7000.0,[1])/pow(x/7000,[2])",200,1000);
  h_CorDijetMass->Fit(massfit, "R");
  
  return;
}
