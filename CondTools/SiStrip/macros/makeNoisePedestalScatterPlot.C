#include "TROOT.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TTree.h"
#include "TBranch.h"
#include "TTree.h"
#include "TChain.h"
#include "TLegend.h"
#include "TGaxis.h"

#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>  // std::setw

/*--------------------------------------------------------------------*/
void makeNicePlotStyle(TH1* hist)
/*--------------------------------------------------------------------*/
{
  hist->SetStats(kFALSE);
  hist->SetLineWidth(2);
  hist->GetXaxis()->CenterTitle(true);
  hist->GetYaxis()->CenterTitle(true);
  hist->GetXaxis()->SetTitleFont(42);
  hist->GetYaxis()->SetTitleFont(42);
  hist->GetXaxis()->SetTitleSize(0.05);
  hist->GetYaxis()->SetTitleSize(0.05);
  hist->GetXaxis()->SetTitleOffset(0.9);
  hist->GetYaxis()->SetTitleOffset(1.3);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelSize(.05);
  hist->GetXaxis()->SetLabelSize(.05);
}

/*--------------------------------------------------------------------*/
void makeNoisePedestalScatterPlot() {
  /*--------------------------------------------------------------------*/

  TGaxis::SetMaxDigits(4);

  TFile* file = new TFile("idealNoise.root");

  TH2F* h2_NoiseVsPedestal = (TH2F*)file->Get("h2_NoiseVsPedestal");

  TH2F* h2_NoiseVsPedestalTIB = (TH2F*)file->Get("h2_NoiseVsPedestalTIB");
  TH2F* h2_NoiseVsPedestalTOB = (TH2F*)file->Get("h2_NoiseVsPedestalTOB");
  TH2F* h2_NoiseVsPedestalTID = (TH2F*)file->Get("h2_NoiseVsPedestalTID");
  TH2F* h2_NoiseVsPedestalTEC = (TH2F*)file->Get("h2_NoiseVsPedestalTEC");

  std::map<std::string, TH2F*> scatters;

  scatters["TIB"] = h2_NoiseVsPedestalTIB;
  scatters["TOB"] = h2_NoiseVsPedestalTOB;
  scatters["TID"] = h2_NoiseVsPedestalTID;
  scatters["TEC"] = h2_NoiseVsPedestalTEC;

  std::map<std::string, int> colormap;
  std::map<std::string, int> markermap;
  colormap["TIB"] = kRed;
  markermap["TIB"] = kFullCircle;
  colormap["TOB"] = kGreen;
  markermap["TOB"] = kFullTriangleUp;
  colormap["TID"] = kCyan;
  markermap["TID"] = kFullSquare;
  colormap["TEC"] = kBlue;
  markermap["TEC"] = kFullTriangleDown;

  std::vector<std::string> parts = {"TEC", "TID", "TOB", "TIB"};

  TCanvas* canvas = new TCanvas("c1", "c1", 1000, 800);
  canvas->cd();

  auto legend2 = new TLegend(0.75, 0.85, 0.95, 0.97);
  legend2->SetTextSize(0.03);
  canvas->cd();
  canvas->cd()->SetTopMargin(0.03);
  canvas->cd()->SetLeftMargin(0.13);
  canvas->cd()->SetRightMargin(0.05);

  for (const auto& part : parts) {
    makeNicePlotStyle(scatters[part]);
    scatters[part]->SetTitle("");
    scatters[part]->SetStats(false);
    scatters[part]->SetMarkerColor(colormap[part]);
    scatters[part]->SetMarkerStyle(markermap[part]);
    scatters[part]->SetMarkerSize(0.2);

    auto temp = (TH2F*)(scatters[part]->Clone());
    temp->SetMarkerSize(1.3);

    if (part == "TEC")
      scatters[part]->Draw("P");
    else
      scatters[part]->Draw("Psame");

    legend2->AddEntry(temp, part.c_str(), "P");
  }

  legend2->Draw("same");

  canvas->SaveAs("NoiseVsPedestals_subdet.png");

  TCanvas* canvas2 = new TCanvas("c2", "c2", 1000, 800);
  canvas2->cd();
  canvas2->cd();
  canvas2->cd()->SetTopMargin(0.03);
  canvas2->cd()->SetLeftMargin(0.13);
  canvas2->cd()->SetRightMargin(0.13);

  makeNicePlotStyle(h2_NoiseVsPedestal);
  h2_NoiseVsPedestal->SetTitle("");
  h2_NoiseVsPedestal->SetStats(false);
  h2_NoiseVsPedestal->Draw("colz");

  canvas2->SaveAs("NoiseVsPedestals.png");
}
