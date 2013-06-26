

#include <TH1F.h>
#include <TGraphAsymmErrors.h>
#include <TMath.h>
#include <TMatrixD.h>
#include <TVectorD.h>
#include <TArrayD.h>
#include <TString.h>
#include <TCanvas.h>
#include <TPostScript.h>
#include <TPaveText.h>
#include <iostream>

void drawEfficiency(TString histogramTitle, TGraphAsymmErrors* graph, TString xAxisTitle, TH1F* hDummy,TCanvas* canvas, TPaveText* Text, TString loglinearscale="LinearScale", double Minimum=0.)
{ 

  hDummy->SetStats(false);
  hDummy->SetTitle(histogramTitle);
  hDummy->SetMinimum(Minimum);
  hDummy->SetMaximum(1.2);
  hDummy->GetXaxis()->SetTitle(xAxisTitle);
  hDummy->GetYaxis()->CenterTitle();
  hDummy->GetYaxis()->SetTitle("Efficiency");
  hDummy->Reset();
  if ( loglinearscale.CompareTo("LogScale")==0) {
    if (Minimum<0.00001) Minimum = 0.001;
    hDummy->SetMinimum(Minimum);
    hDummy->SetMaximum(3.0);
    canvas->SetLogy();
  }
  hDummy->Draw("p");
  graph->SetMarkerStyle(20);
  graph->SetMarkerSize(1);
  graph->SetMarkerColor(1);
  if ( Text ) Text->Draw();
  canvas->Update();
  graph->Draw("p");  
  canvas->Update();
  canvas->Print(TString(canvas->GetTitle()).Append(".gif"),"gif");
  
}
