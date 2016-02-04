#include <iostream>
#include <TH1D.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TLegend.h>
#include <TGraphAsymmErrors.h>

void drawHisto(TString type, TFile * outputFile,
	       const double & minX, const double & maxX,
	       const double & minY, const double & maxY)
{
  TFile * inputFile = new TFile("test.root", "READ");
  TCanvas * canvas = (TCanvas*)inputFile->Get("canvas"+type);
  TGraphAsymmErrors * graph = (TGraphAsymmErrors*)canvas->GetPrimitive("Graph_from_sigmaPtVs"+type);

  TFile * inputFile2 = new TFile("ComparedResol.root", "READ");
  TCanvas * canvas2 = (TCanvas*)inputFile2->Get("resolPtVS"+type);
  TH1D * histo = (TH1D*)canvas2->GetPrimitive("hResolPtGenVSMu_ResoVS"+type+"_resol_after");

  TLegend * legend = new TLegend(0.7,0.71,0.98,1.);
  legend->SetTextSize(0.02);
  legend->SetFillColor(0); // Have a white background
  legend->AddEntry(histo, "resolution from MC comparison");
  legend->AddEntry(graph, "resolution from fitted function");

  graph->GetXaxis()->SetRangeUser(minX, maxX);
  graph->GetYaxis()->SetRangeUser(minY, maxY);
  canvas->Draw();
  histo->Draw("SAME");
  legend->Draw("SAME");

  outputFile->cd();
  canvas->Write();
}

void CompareErrorResol()
{
  TFile * outputFile = new TFile("output.root", "RECREATE");
  drawHisto("Pt", outputFile, 0., 20., 0., 0.06);
  drawHisto("Eta", outputFile, -2.39, 2.39, 0., 0.06);
  outputFile->Write();
  outputFile->Close();
}
