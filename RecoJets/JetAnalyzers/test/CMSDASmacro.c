#include "TFile.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TF1.h"
#include "TStyle.h"
#include "TLine.h"
#include <iostream>

using std::cout;

void CMSDASmacro() {
  TFile* rootfile = new TFile("histos.root");
  rootfile->cd("dijetAna");
  gDirectory->ls();

  //------------------Dijet mass spectrum---------------------------

  gStyle->SetOptStat(0);
  gStyle->SetOptFit(1111);
  //Get pointers to some histograms we will want
  TH1D* h_CorDijetMass=(TH1D*)gROOT->FindObject("hCorDijetXsec");
  //Error bars, please.
  h_CorDijetMass->Sumw2();

  //Make a home for the pretty plots
  TCanvas *cDijetMass=new TCanvas();
  // Set the current TPad to "Logy()". Not an English word.
  gPad->SetLogy();
  h_CorDijetMass->Draw();

  // Declare the function to fit, over some range, and fit it.
  //TF1 *massfit= new TF1("Dijet Mass Spectrum", "[0]*pow(1-x/7000.0,[1])/pow(x/7000,[2])",528,2000);
  TF1 *massfit= new TF1("Dijet Mass Spectrum", "[0]*pow(1-x/7000.0,[1])/pow(x/7000,[2]+[3]*log(x/7000.))",489,2132);
  massfit->SetParameter(1,5.077);
  massfit->SetParameter(2, 6.994);
  massfit->SetParameter(3,0.2658);

  h_CorDijetMass->Fit(massfit, "R");
  h_CorDijetMass->SetMinimum(1E-3);

  //Make a histogram for the values of the fit, with the binning of h_CorDijetMass
  TH1D* h_DataMinusFit = new TH1D(*h_CorDijetMass);
  h_DataMinusFit->SetTitle("(Data- Fit)/Fit;dijet mass (GeV)");

  //Fill the histogram of the data minus the fit's values
  for (int bin=1; bin<=h_CorDijetMass->GetNbinsX(); bin++){
    double data_val = h_CorDijetMass->GetBinContent(bin);
    double fit_val = massfit->Eval(h_CorDijetMass->GetBinCenter(bin));
    double err_val  = h_CorDijetMass->GetBinError(bin);
    // Skip bins with no data value
    if (data_val != 0.0) {
      h_DataMinusFit->SetBinContent(bin, (data_val - fit_val)/fit_val );
      h_DataMinusFit->SetBinError(bin, err_val /fit_val );
    }
  }

  //Move to the lower TPad and display the result
  TCanvas *cDijetMassResiduals=new TCanvas();
  h_DataMinusFit->SetMinimum(-1.);
  h_DataMinusFit->SetMaximum( 4);
  h_DataMinusFit->Draw();
  TLine * line = new TLine(489,0.,2132,0);
  line->SetLineStyle(2);
  line->Draw("same");

  TCanvas *cDijetMassPulls=new TCanvas();
  TH1D* h_DijetMassPulls = new TH1D(*h_CorDijetMass);
  h_DijetMassPulls->SetTitle("(Data- Fit)/Error;dijet mass (GeV)");

  //Fill the histogram of the data minus the fit's values
  for (int bin=1; bin<=h_CorDijetMass->GetNbinsX(); bin++){
    double data_val = h_CorDijetMass->GetBinContent(bin);
    double fit_val = massfit->Eval(h_CorDijetMass->GetBinCenter(bin));
    double err_val  = h_CorDijetMass->GetBinError(bin);
    // Skip bins with no data value
    if (data_val != 0.0) {
      h_DijetMassPulls->SetBinContent(bin, (data_val - fit_val)/err_val );
      h_DijetMassPulls->SetBinError(bin, err_val /err_val );
    }
  }
  h_DijetMassPulls->Draw();
  h_DijetMassPulls->GetYaxis()->SetRangeUser(-2.5,2.5);
  line->Draw("same");

  //------------------Dijet Centrality Ratio---------------------------

  
  //Get pointers to some histograms we will want
  TH1D* h_InnerDijetMass=(TH1D*)gROOT->FindObject("hInnerDijetMass");
  TH1D* h_OuterDijetMass=(TH1D*)gROOT->FindObject("hOuterDijetMass");
  //Error bars, please.
  h_InnerDijetMass->Sumw2();
  h_OuterDijetMass->Sumw2();
  

  //Make a home for the pretty plots
  TCanvas *cInnerOuterCounts=new TCanvas("cInnerOuterCounts","cInnerOuterCounts");
  // Set the current TPad to "Logy()". Not an English word.
  gPad->SetLogy();
  h_InnerDijetMass->SetLineColor(2);
  h_InnerDijetMass->SetMarkerStyle(25);
  h_InnerDijetMass->SetMarkerSize(0.7);
  h_InnerDijetMass->SetMarkerColor(2);
  h_OuterDijetMass->SetMarkerStyle(20);
  h_OuterDijetMass->SetMarkerSize(0.3);
  h_InnerDijetMass->Draw();
  h_OuterDijetMass->Draw("same");
  
  
  TCanvas *cDijetDeltaEtaRatio=new TCanvas("cDijetDeltaEtaRatio","cDijetDeltaEtaRatio");
  // Make the dijet delta eta ratio in a histogram
  TH1D* h_DijetDeltaEtaRatio = (TH1D*)h_InnerDijetMass->Clone();
  h_DijetDeltaEtaRatio->Divide(h_OuterDijetMass);
  h_DijetDeltaEtaRatio->SetMarkerStyle(20);
  h_DijetDeltaEtaRatio->SetTitle("Dijet |#Delta#eta| Ratio; dijet mass (GeV)");
  h_DijetDeltaEtaRatio->GetYaxis()->SetRangeUser(0,2.5);
  h_DijetDeltaEtaRatio->Draw();

  TF1* flatline = new TF1("flatline", "[0]",489,2132);
  h_DijetDeltaEtaRatio->Fit(flatline, "R");
  flatline->Draw("same");
  
  return;
}
