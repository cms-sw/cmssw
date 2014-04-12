
#include <TH1F.h>
#include <TGraphAsymmErrors.h>
#include <TMath.h>
#include <TMatrixD.h>
#include <TVectorD.h>
#include <TArrayD.h>
#include <TString.h>
#include <TCanvas.h>
#include <TPostScript.h>
#include <TLegend.h>
#include <TPaveText.h>

void drawEfficiencies(TString histogramTitle, TGraphAsymmErrors* graph1,  TGraphAsymmErrors* graph2, TGraphAsymmErrors* graph3, TGraphAsymmErrors* graph4, TString xAxisTitle, TH1F* hDummy,TCanvas* canvas, TPaveText* Text, TString PForCalo, TString loglinearscale="LinearScale", double Minimum=0.)
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
  graph1->SetMarkerStyle(20);
  graph1->SetMarkerSize(1);
  if ( Text ) Text->Draw();
  canvas->Update();
  graph1->Draw("p");
  canvas->Update();
  graph2->SetMarkerStyle(20);
  graph2->SetMarkerSize(1);  
  graph2->SetMarkerColor(2);  
  graph2->Draw("p");
  canvas->Update();
  graph3->SetMarkerStyle(20);
  graph3->SetMarkerSize(1);  
  graph3->SetMarkerColor(4);  
  graph3->Draw("p");
  canvas->Update();
  graph4->SetMarkerStyle(20);
  graph4->SetMarkerSize(1);  
  graph4->SetMarkerColor(3);  
  graph4->Draw("p");

  //  graph3->SetMarkerColor();

  TLegend* legend;
  legend = new TLegend(0.1, 0.84, 0.5, 0.94);
  if ( PForCalo.CompareTo("Calo")==0) {
    legend->AddEntry(graph1, "Jet Matching Efficiency", "p");
    legend->AddEntry(graph2, "Jet + Leading Track Efficiency","p");
    legend->AddEntry(graph3, "Track Isolation Efficiency", "p");
    legend->AddEntry(graph4, "Ecal Isolation Efficiency", "p");
  }
  if (PForCalo.CompareTo("PFTaus")==0) {
    legend->AddEntry(graph1, "PFTau Matching Efficiency", "p");
    legend->AddEntry(graph2, "PFTau + LeadingChargedHadron Efficiency","p");
    legend->AddEntry(graph3, "No Charged Hadrons Isolation", "p");
    legend->AddEntry(graph4, "No Gammas Isolation", "p");
  }
  legend->Draw();
  canvas->Update();
  canvas->Print(TString(canvas->GetTitle()).Append(".gif"),"gif");
}
