#include <iostream>
#include <TH1D.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TLegend.h>
#include <TGraphAsymmErrors.h>

void CompareErrorResol()
{
  TFile * inputFile = new TFile("test.root", "READ");
  TCanvas * canvas = (TCanvas*)inputFile->Get("canvas");
  TGraphAsymmErrors * graph = (TGraphAsymmErrors*)canvas->GetPrimitive("Graph_from_sigmaPtVsEta");

  TFile * inputFile2 = new TFile("ComparedResol.root", "READ");
  TCanvas * canvas2 = (TCanvas*)inputFile2->Get("resolPtVSeta");
  TH1D * histo = (TH1D*)canvas2->GetPrimitive("hResolPtGenVSMu_ResoVSEta_resol_after");

  TLegend * legend = new TLegend(0.7,0.71,0.98,1.);
  legend->SetTextSize(0.02);
  legend->SetFillColor(0); // Have a white background
  legend->AddEntry(histo, "resolution from MC comparison");
  legend->AddEntry(graph, "resolution from fitted function");

  graph->GetXaxis()->SetRangeUser(-2.39, 2.39);
  graph->GetYaxis()->SetRangeUser(0, 0.06);
  canvas->Draw();
  histo->Draw("SAME");
  legend->Draw("SAME");
}
