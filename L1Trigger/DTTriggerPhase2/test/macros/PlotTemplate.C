// Drawing template (originally used for TRG-17-001)
// Author: O. Davignon (CERN)
#include <TCanvas.h>
#include <TF1.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TPaveText.h>
#include <TMarker.h>
#include <TLine.h>
#include <TAxis.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TH1.h>
#include <TGraph.h>
#include <TMultiGraph.h>
#include <TGraphAsymmErrors.h>

TCanvas* CreateCanvas(TString CanvasName = "myPlot", bool LogY = false, bool Grid = true) {
  TCanvas* c = new TCanvas(CanvasName.Data(), CanvasName.Data(), 800, 800);
  c->SetLeftMargin(0.11);
  if (Grid) {
    c->SetGrid();
  }
  if (LogY) {
    c->SetLogy();
  }
  return c;
}

void DrawPrelimLabel(TCanvas* c) {
  c->cd();

  TLatex tex;
  tex.SetTextSize(0.03);
  tex.DrawLatexNDC(0.11, 0.91, "#scale[1.5]{CMS} Phase-2 Simulation");  //typically for Phase-2
  // tex.DrawLatexNDC(0.11,0.91,"#scale[1.5]{CMS}");//typically for Phase-1
  tex.Draw("same");

  return;
}

void DrawLumiLabel(TCanvas* c, TString toDisplay = "14 TeV, 3000 fb^{-1}, 200 PU") {
  c->cd();

  TLatex tex;
  tex.SetTextSize(0.035);
  tex.SetTextAlign(31);
  //TString toDisplay = toDisplay;//typically for Phase-2
  // TString toDisplay = Lumi + " fb^{-1} (13 TeV)";//typically for Phase-1
  tex.DrawLatexNDC(0.90, 0.91, toDisplay.Data());
  tex.Draw("same");

  return;
}

void SaveCanvas(TCanvas* c, TString PlotName = "myPlotName") {
  c->cd();
  c->SaveAs(PlotName + ".png");
  c->SaveAs(PlotName + ".pdf");
  c->SaveAs(PlotName + ".root");

  return;
}
