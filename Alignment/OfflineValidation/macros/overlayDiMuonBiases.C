#include <TFile.h>
#include <TLatex.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TGaxis.h>
#include <string>
#include <map>
#include <TH1.h>
#include <TH2.h>
#include <TKey.h>
#include <iostream>

/*--------------------------------------------------------------------*/
void cmsPrel(const TCanvas* canvas, float correction = 0.) {
  /*--------------------------------------------------------------------*/

  // Create and draw the CMS text
  TLatex* latexCMS = new TLatex(canvas->GetLeftMargin(), 1.01 - canvas->GetTopMargin(), "#bf{CMS} #it{Preliminary}");
  latexCMS->SetNDC(kTRUE);
  latexCMS->SetTextFont(42);
  latexCMS->SetTextSize(0.042);
  latexCMS->Draw();

  // Create and draw the Internal (13 TeV) text
  TLatex* latexInternal = new TLatex(
      1 - canvas->GetRightMargin() - correction, 1.01 - canvas->GetTopMargin(), "pp collisions (2022) 13.6 TeV");
  latexInternal->SetNDC(kTRUE);
  latexInternal->SetTextAlign(31);
  latexInternal->SetTextFont(42);
  latexInternal->SetTextSize(0.042);
  latexInternal->Draw();
}

/*--------------------------------------------------------------------*/
void adjustCanvasMargins(TCanvas* canvas) {
  /*--------------------------------------------------------------------*/
  canvas->SetLeftMargin(0.14);
  canvas->SetRightMargin(0.04);
  canvas->SetTopMargin(0.06);
  canvas->SetBottomMargin(0.12);
}

/*--------------------------------------------------------------------*/
void adjustCanvasMargins2D(TCanvas* canvas) {
  /*--------------------------------------------------------------------*/
  canvas->SetLeftMargin(0.10);
  canvas->SetRightMargin(0.185);
  canvas->SetTopMargin(0.06);
  canvas->SetBottomMargin(0.12);
}

// Function to modify axis title if it contains "phi"
/*--------------------------------------------------------------------*/
void modifyAxisTitle(TH1* hist, const TString& axisName)
/*--------------------------------------------------------------------*/
{
  // Get the current axis title
  TString axisTitle;
  if (axisName == "X") {
    axisTitle = hist->GetXaxis()->GetTitle();
  } else if (axisName == "Y") {
    axisTitle = hist->GetYaxis()->GetTitle();
  } else {
    std::cerr << "Invalid axis name: " << axisName << ". Use 'X' or 'Y'." << std::endl;
    return;
  }

  // Convert to lower case for case-insensitive comparison
  TString axisTitleLower = axisTitle;
  axisTitleLower.ToLower();

  // Check if "phi" is in the axis title
  if (axisTitleLower.Contains("phi")) {
    // Append " [rad]" if "phi" is found
    axisTitle += " [rad]";
    if (axisName == "X") {
      hist->GetXaxis()->SetTitle(axisTitle);
    } else if (axisName == "Y") {
      hist->GetYaxis()->SetTitle(axisTitle);
    }
    std::cout << "Updated " << axisName << "-axis title to: " << axisTitle << std::endl;
  }
}

/*--------------------------------------------------------------------*/
void makeNicePlotStyle(TH1* hist, int color, int markerStyle)
/*--------------------------------------------------------------------*/
{
  hist->SetStats(kFALSE);
  hist->SetLineWidth(2);
  hist->SetLineColor(color);
  hist->SetMarkerColor(color);
  hist->SetMarkerStyle(markerStyle);
  //hist->GetXaxis()->CenterTitle(true);
  //hist->GetYaxis()->CenterTitle(true);
  hist->GetXaxis()->SetTitleFont(42);
  hist->GetYaxis()->SetTitleFont(42);
  hist->GetXaxis()->SetTitleSize(0.05);
  hist->GetYaxis()->SetTitleSize(0.05);
  hist->GetZaxis()->SetTitleSize(0.05);
  hist->GetXaxis()->SetTitleOffset(1.2);
  if (hist->InheritsFrom("TH2")) {
    hist->GetYaxis()->SetTitleOffset(1.0);
  } else {
    hist->GetYaxis()->SetTitleOffset(1.3);
  }
  hist->GetZaxis()->SetTitleOffset(1.3);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelSize(.05);
  hist->GetXaxis()->SetLabelSize(.05);
  hist->GetZaxis()->SetLabelSize(.05);

  // Modify the axis titles if they contain "phi"
  modifyAxisTitle(hist, "X");
  modifyAxisTitle(hist, "Y");

  TGaxis::SetExponentOffset(-0.1, 0.01, "y");  // Y offset
}

/*--------------------------------------------------------------------*/
std::pair<Double_t, Double_t> getExtrema(TObjArray* array)
/*--------------------------------------------------------------------*/
{
  Double_t theMaximum = (static_cast<TH1*>(array->At(0)))->GetMaximum();
  Double_t theMinimum = (static_cast<TH1*>(array->At(0)))->GetMinimum();
  for (Int_t i = 0; i < array->GetSize(); i++) {
    if ((static_cast<TH1*>(array->At(i)))->GetMaximum() > theMaximum) {
      theMaximum = (static_cast<TH1*>(array->At(i)))->GetMaximum();
    }
    if ((static_cast<TH1*>(array->At(i)))->GetMinimum() < theMinimum) {
      theMinimum = (static_cast<TH1*>(array->At(i)))->GetMinimum();
    }
  }
  return std::make_pair(theMinimum, theMaximum);
}

void overlayHistograms(const std::vector<std::string>& fileNames,
                       const std::vector<string>& labels,
                       const std::string& type) {
  gStyle->SetOptTitle(0);

  // Create a new canvas for each histogram
  TCanvas* c = nullptr;

  std::vector<int> colors = {kBlack, kRed, kBlue, kGreen, kAzure, kYellow};
  std::vector<int> markers = {20, 21, 22, 23, 24, 25};

  // Map to store histograms with the same name
  std::map<std::string, std::vector<TH1*>> histMap;
  std::map<std::string, std::vector<TH2*>> hist2DMap;

  // Loop over all the input files
  for (const auto& fileName : fileNames) {
    // Open the input file
    TFile* file = TFile::Open(fileName.c_str());
    if (!file || file->IsZombie()) {
      std::cerr << "Could not open file " << fileName << std::endl;
      continue;
    }

    // Loop over all histograms in the directory
    TIter nexthist(file->GetListOfKeys());
    TKey* key = nullptr;
    while ((key = static_cast<TKey*>(nexthist()))) {
      TObject* obj = key->ReadObj();
      if (obj->InheritsFrom(TH1::Class())) {
        TH1* hist = static_cast<TH1*>(obj);
        std::string histName = hist->GetName();
        std::cout << "pushing back: " << histName << std::endl;
        if (!obj->InheritsFrom(TH2::Class())) {
          histMap[histName].push_back(hist);
        } else {
          TH2* hist = static_cast<TH2*>(obj);
          hist2DMap[histName].push_back(hist);
        }
      }
    }
  }
  // Close the input file
  //file->Close();

  // Loop over the histograms in the map
  for (const auto& histPair : histMap) {
    const std::string& histName = histPair.first;
    const std::vector<TH1*>& histVec = histPair.second;

    if (histName.find("delta_iter_") != std::string::npos)
      continue;

    // Create a new canvas for the histogram
    c = new TCanvas((histName + type).c_str(), histName.c_str(), 800, 800);

    //if(histName.find("Delta") != std::string::npos) {
    //  c->SetLogy();
    //}

    adjustCanvasMargins(c);

    TObjArray* array = new TObjArray(histVec.size());
    for (const auto& histo : histVec) {
      array->Add(histo);
    }
    std::pair<Double_t, Double_t> extrema = getExtrema(array);
    const auto& DELTA = std::abs(extrema.second - extrema.first);
    delete array;

    // Draw the first histogram
    histVec[0]->Draw();
    makeNicePlotStyle(histVec[0], kBlack, 20);

    if ((histName.find("avg") != std::string::npos)) {
      histVec[0]->SetMinimum(extrema.first - DELTA / 3.);
      histVec[0]->SetMaximum(extrema.second + DELTA / 3.);
    } else if (histName.find("RMS") != std::string::npos) {
      histVec[0]->SetMinimum(0.);
      histVec[0]->SetMaximum(extrema.second + DELTA / 3.);
    } else if (histName.find("Correction") != std::string::npos) {
      histVec[0]->SetMinimum(extrema.first - DELTA / 3.);
      histVec[0]->SetMaximum(extrema.second + DELTA / 3.);
    } else {
      histVec[0]->SetMaximum(extrema.second + DELTA / 3.);
    }

    // Loop over the remaining histograms and overlay them
    for (size_t i = 1; i < histVec.size(); ++i) {
      histVec[i]->Draw("SAME");
      makeNicePlotStyle(histVec[i], colors[i], markers[i]);
    }

    // Draw the legend
    TLegend* infoBox = new TLegend(0.44, 0.80, 0.94, 0.93, "");
    infoBox->SetShadowColor(0);  // 0 = transparent
    infoBox->SetBorderSize(0);   // 0 = transparent
    infoBox->SetFillColor(kWhite);
    infoBox->SetTextSize(0.035);

    for (unsigned int i = 0; i < histVec.size(); i++) {
      infoBox->AddEntry(histVec[i], labels[i].c_str(), "PL");
    }
    infoBox->Draw("same");

    TLatex* latex = new TLatex();
    latex->SetTextAlign(22);
    latex->SetTextSize(0.045);

    //latex->DrawLatexNDC(0.75, 0.85, "Z^{0} #rightarrow#mu^{+}#mu^{-}");
    cmsPrel(c);

    // Update the canvas
    c->Update();
    c->SaveAs((histName + "_" + type + ".png").c_str());
    c->SaveAs((histName + "_" + type + ".pdf").c_str());
    c->SaveAs((histName + "_" + type + ".eps").c_str());
    c->SaveAs((histName + "_" + type + ".root").c_str());
  }

  //gStyle->SetPalette(kRainbow);
  //gStyle->SetPalette(kWaterMelon);
  gStyle->SetPalette(kTemperatureMap);
  //gStyle->SetPalette(kViridis);

  // Loop over the 2D histograms in the map
  for (const auto& entry : hist2DMap) {
    const std::string& histName = entry.first;
    const std::vector<TH2*>& histList = entry.second;

    if (histList.empty()) {
      std::cerr << "No histograms found for " << histName << std::endl;
      continue;
    }

    TObjArray* array = new TObjArray(histList.size());
    for (const auto& histo : histList) {
      array->Add(histo);
    }

    std::pair<Double_t, Double_t> extrema = getExtrema(array);
    delete array;

    TCanvas* c = new TCanvas((histName + type).c_str(), histName.c_str(), 900 * histList.size(), 800);
    c->Divide(histList.size(), 1);

    for (size_t i = 0; i < histList.size(); ++i) {
      c->cd(i + 1);
      auto current_pad = static_cast<TCanvas*>(gPad);
      adjustCanvasMargins2D(current_pad);
      makeNicePlotStyle(histList[i], kWhite, 0);
      histList[i]->SetMinimum(extrema.first);
      histList[i]->SetMaximum(extrema.second);
      histList[i]->Draw("COLZ");
      cmsPrel(current_pad, -0.062);
    }

    // Update the canvas
    c->Update();
    c->SaveAs((histName + "_" + type + "_side_by_side.png").c_str());
    c->SaveAs((histName + "_" + type + "_side_by_side.pdf").c_str());
    c->SaveAs((histName + "_" + type + "_side_by_side.eps").c_str());
    c->SaveAs((histName + "_" + type + "_side_by_side.root").c_str());

    for (size_t i = 0; i < histList.size(); ++i) {
      TCanvas* c2 = new TCanvas((histName + type + labels[i]).c_str(), histName.c_str(), 900, 800);
      c2->cd();
      auto current_pad = static_cast<TCanvas*>(gPad);
      adjustCanvasMargins2D(current_pad);
      makeNicePlotStyle(histList[i], kWhite, 0);
      histList[i]->Draw("COLZ");
      cmsPrel(c2, -0.050);
      std::string result = labels[i];
      std::replace(result.begin(), result.end(), ' ', '_');
      c2->SaveAs((histName + "_" + type + "_" + result + ".png").c_str());
      c2->SaveAs((histName + "_" + type + "_" + result + ".pdf").c_str());
      c2->SaveAs((histName + "_" + type + "_" + result + ".eps").c_str());
      c2->SaveAs((histName + "_" + type + "_" + result + ".root").c_str());
    }
  }
}

void overlayDiMuonBiases() {
  std::vector<std::string> fileNames = {"histos_asInDataTaking_DiMuonAnalysisResults_Run2022D-v1__d0__FINE.root",
                                        "histos_Run3ReReco_DiMuonAnalysisResults_Run2022D-v1__d0.root"};
  std::vector<std::string> labels = {"fine", "fast"};  // Add your ROOT file names here

  overlayHistograms(fileNames, labels, "d0");

  //fileNames.clear();
  //fileNames = {"histos_asInDataTaking_DiMuonAnalysisResults_Run2022__dz.root","histos_Run3ReReco_DiMuonAnalysisResults_Run2022__dz.root"};
  //overlayHistograms(fileNames,labels,"dz");
}
-- dummy change --
