// ROOT includes
#include "TCanvas.h"
#include "TClass.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TGaxis.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TKey.h"
#include "TLegend.h"
#include "TObjString.h"
#include "TObject.h"
#include "TProfile.h"
#include "TRatioPlot.h"
#include "TStyle.h"

// standard includes
#include <iostream>

// style
#include "Alignment/OfflineValidation/macros/CMS_lumi.h"

// 2 file case
TFile *sourceFile1, *sourceFile2;

// multi-file case
std::vector<TFile *> sourceFiles;
Int_t def_colors[9] = {kBlack, kBlue, kRed, kMagenta, kGreen, kCyan, kViolet, kOrange, kGreen + 2};

std::pair<Double_t, Double_t> getExtrema(TObjArray *array);
template <typename T>
void MakeNicePlotStyle(T *hist);
void MakeNiceProfile(TProfile *prof);

//void MakeNicePlotStyle(TH1 *hist);
void plot2Histograms(TH1 *h1, TH1 *h2, const TString &label1, const TString &label2);
void plot2Profiles(TProfile *h1, TProfile *h2, const TString &label1, const TString &label2);
void recurseOverKeys(TDirectory *target1, const std::vector<TString> &labels, bool isNorm, const TString &Rlabel);
void recurseOverKeys(TDirectory *target1, const TString &label1, const TString &label2);
void plotHistograms(std::vector<TH1 *> histos,
                    const std::vector<TString> &labels,
                    bool isNormalized = false,
                    const TString &Rlabel = "");
void plotHistogramsInv(std::vector<TH1 *> histos,
                       std::vector<TH1 *> invHistos,
                       const std::vector<TString> &labels,
                       const TString &Rlabel = "");

/************************************************/
void loopAndPlot(TString namesandlabels, const TString &Rlabel = "", bool doNormalize = false)
/************************************************/
{
  std::vector<TString> labels;

  namesandlabels.Remove(TString::kTrailing, ',');
  TObjArray *nameandlabelpairs = namesandlabels.Tokenize(",");
  for (Int_t i = 0; i < nameandlabelpairs->GetEntries(); ++i) {
    TObjArray *aFileLegPair = TString(nameandlabelpairs->At(i)->GetName()).Tokenize("=");
    if (aFileLegPair->GetEntries() == 2) {
      sourceFiles.push_back(TFile::Open(aFileLegPair->At(0)->GetName(), "READ"));
      TObjString *s_label = (TObjString *)aFileLegPair->At(1);
      labels.push_back(s_label->String());
    } else {
      std::cout << "Please give file name and legend entry in the following form:\n"
                << " filename1=legendentry1,filename2=legendentry2\n";
      return;
    }
  }

  recurseOverKeys(sourceFiles[0], labels, doNormalize, Rlabel);

  for (const auto &file : sourceFiles) {
    file->Close();
  }
}

/************************************************/
void recurseOverKeys(TDirectory *target1, const std::vector<TString> &labels, bool isNorm, const TString &Rlabel)
/************************************************/
{
  // Figure out where we are
  TString fullPath = target1->GetPath();

  // Check if the prefix "root://eoscms.cern.ch/" is present and remove it
  TString prefixToRemove = "root://eoscms.cern.ch/";
  if (fullPath.BeginsWith(prefixToRemove)) {
    fullPath.Remove(0, prefixToRemove.Length());
  }

  TString path(strstr(fullPath.Data(), ":"));
  path.Remove(0, 2);

  sourceFiles[0]->cd(path);

  std::cout << path << std::endl;

  TDirectory *current_sourcedir = gDirectory;

  TKey *key;
  TIter nextkey(current_sourcedir->GetListOfKeys());

  while ((key = (TKey *)nextkey())) {
    auto obj = key->ReadObj();

    // Check if this is a 1D histogram or a directory
    if (obj->IsA()->InheritsFrom("TH1")) {
      if (obj->IsA()->InheritsFrom("TH2"))
        continue;
      // **************************
      // Plot & Save this Histogram
      std::vector<TH1 *> histos;

      TH1 *htemp1 = (TH1 *)obj;
      TString histName = htemp1->GetName();

      for (const auto &file : sourceFiles) {
        TH1 *htemp;
        if (path != "") {
          file->GetObject(path + "/" + histName, htemp);
        } else {
          file->GetObject(histName, htemp);
        }
        histos.push_back(htemp);
      }
      // If it is a CosPhi histogram, plot it together with the inverted histogram
      if (histName == "CosPhi" or histName == "CosPhi3D") {
        TString histName2;
        if (histName == "CosPhi") {
          histName2 = "CosPhiInv";
        } else if (histName == "CosPhi3D") {
          histName2 = "CosPhiInv3D";
        }
        std::vector<TH1 *> invHistos;
        for (const auto &file : sourceFiles) {
          TH1 *htemp2;
          if (path != "") {
            file->GetObject(path + "/" + histName2, htemp2);
          } else {
            file->GetObject(histName2, htemp2);
          }
          invHistos.push_back(htemp2);
        }
        plotHistogramsInv(histos, invHistos, labels, Rlabel);
      }
      // do now all the normal plotting
      plotHistograms(histos, labels, isNorm, Rlabel);
    } else if (obj->IsA()->InheritsFrom("TDirectory")) {
      // it's a subdirectory

      std::cout << "Found subdirectory " << obj->GetName() << std::endl;
      //gSystem->MakeDirectory(outputFolder+path+"/"+obj->GetName());

      // obj is now the starting point of another round of merging
      // obj still knows its depth within the target file via
      // GetPath(), so we can still figure out where we are in the recursion

      if ((TString(obj->GetName())).Contains("Residuals"))
        continue;

      recurseOverKeys((TDirectory *)obj, labels, isNorm, Rlabel);

    }  // end of IF a TDriectory
  }
}

/************************************************/
void plotHistograms(std::vector<TH1 *> histos,
                    const std::vector<TString> &labels,
                    bool isNormalized,
                    const TString &Rlabel)
/************************************************/
{
  TGaxis::SetMaxDigits(3);

  int W = 800;
  int H = 800;
  // references for T, B, L, R
  float T = 0.08 * H;
  float B = 0.12 * H;
  float L = 0.12 * W;
  float R = 0.04 * W;

  auto c1 = new TCanvas(Form("c1_%s", histos[0]->GetName()), "A ratio example", W, H);
  c1->SetFillColor(0);
  c1->SetBorderMode(0);
  c1->SetFrameFillStyle(0);
  c1->SetFrameBorderMode(0);
  c1->SetLeftMargin(L / W + 0.05);
  c1->SetRightMargin(R / W);
  c1->SetTopMargin(T / H);
  c1->SetBottomMargin(B / H);
  c1->SetTickx(0);
  c1->SetTicky(0);
  c1->SetGrid();
  c1->cd();

  gStyle->SetOptStat(0);

  c1->SetTicks(0, 1);

  TObjArray *array = new TObjArray(histos.size());
  int index = 0;
  for (const auto &histo : histos) {
    MakeNicePlotStyle<TH1>(histo);

    if (isNormalized) {
      Double_t scale = 1. / histo->Integral();
      histo->Scale(scale);
    }

    histo->SetLineColor(def_colors[index]);
    histo->SetMarkerColor(def_colors[index]);
    histo->SetMarkerStyle(20);
    array->Add(histo);
    index++;
  }

  std::pair<Double_t, Double_t> extrema = getExtrema(array);
  delete array;
  float min = (extrema.first > 0) ? (extrema.first) * 0.7 : (extrema.first) * 1.3;
  histos[0]->GetYaxis()->SetRangeUser(min, extrema.second * 1.3);

  TRatioPlot *rp{nullptr};

  for (unsigned int i = 1; i < histos.size(); i++) {
    if (i == 1) {
      rp = new TRatioPlot(histos[0], histos[i]);
      rp->SetLeftMargin(0.15);
      rp->SetRightMargin(0.05);
      rp->SetSeparationMargin(0.01);
      rp->SetLowBottomMargin(0.35);
      rp->Draw();
    } else {
      rp->GetUpperPad()->cd();
      histos[i]->Draw("same");
    }
  }

  if (!rp) {
    std::cerr << "TRatioPlot could not be initialized, exiting!" << std::endl;
    return;
  }

  rp->GetUpperPad()->cd();

  // Draw the legend
  TLegend *infoBox = new TLegend(0.65, 0.75, 0.95, 0.90, "");
  infoBox->SetShadowColor(0);  // 0 = transparent
  infoBox->SetFillColor(kWhite);
  infoBox->SetTextSize(0.035);

  for (unsigned int i = 0; i < histos.size(); i++) {
    if (i == 0) {
      infoBox->AddEntry(histos[i], labels[i], "L");
    } else {
      infoBox->AddEntry(histos[i], labels[i], "P");
    }
  }
  infoBox->Draw("same");

  MakeNicePlotStyle<TGraph>(rp->GetLowerRefGraph());
  rp->GetLowerRefGraph()->GetYaxis()->SetTitle("ratio");
  rp->GetLowerRefGraph()->SetMinimum(0.);
  rp->GetLowerRefGraph()->SetMaximum(2.);
  rp->GetLowerRefGraph()->SetLineColor(def_colors[0]);
  rp->GetLowerRefGraph()->SetMarkerColor(def_colors[0]);
  //c1->Update();

  for (unsigned int i = 1; i < histos.size(); i++) {
    auto c2 = new TCanvas(Form("c2_%s_%i", histos[i]->GetName(), i), "A ratio example 2", 800, 800);
    c2->cd();
    auto rp2 = new TRatioPlot(histos[0], histos[i]);
    rp2->Draw();
    TGraph *g = rp2->GetLowerRefGraph();
    // if(g)
    MakeNicePlotStyle<TGraph>(g);
    g->SetLineColor(def_colors[i]);
    g->SetMarkerColor(def_colors[i]);

    c1->cd();
    rp->GetLowerPad()->cd();
    if (g)
      g->Draw("same");
    c1->Update();
    delete c2;
  }

  //rp->GetLowerPad()->cd();
  //c1->Update();

  CMS_lumi(c1, 0, 3, Rlabel);
  c1->SaveAs(TString(histos[0]->GetName()) + ".png");
  delete c1;
}

/************************************************/
void plotHistogramsInv(std::vector<TH1 *> histos,
                       std::vector<TH1 *> invHistos,
                       const std::vector<TString> &labels,
                       const TString &Rlabel)
/************************************************/
{
  TGaxis::SetMaxDigits(4);

  int W = 800;
  int H = 800;
  // references for T, B, L, R
  float T = 0.08 * H;
  float B = 0.12 * H;
  float L = 0.12 * W;
  float R = 0.04 * W;

  auto c1 = new TCanvas(Form("c1_%s", histos[0]->GetName()), "A ratio example", W, H);
  c1->SetFillColor(0);
  c1->SetBorderMode(0);
  c1->SetFrameFillStyle(0);
  c1->SetFrameBorderMode(0);
  c1->SetLeftMargin(L / W + 0.05);
  c1->SetRightMargin(R / W);
  c1->SetTopMargin(T / H);
  c1->SetBottomMargin(B / H);
  c1->SetTickx(0);
  c1->SetTicky(0);
  c1->SetGrid();
  c1->cd();

  gStyle->SetOptStat(0);

  c1->SetTicks(0, 1);

  TObjArray *array = new TObjArray(histos.size());
  //~ int index = 0;
  for (unsigned int i = 0; i < histos.size(); i++) {
    const Double_t bin1 = histos[i]->GetBinContent(1);
    const Double_t invBin1 = invHistos[i]->GetBinContent(invHistos[i]->GetNbinsX());
    if (bin1 != invBin1) {
      std::cout << "Something went wrong, inverted histograms are not mirrored" << std::endl;
    }
    MakeNicePlotStyle<TH1>(histos[i]);
    MakeNicePlotStyle<TH1>(invHistos[i]);

    histos[i]->SetLineColor(def_colors[i]);
    histos[i]->SetMarkerColor(def_colors[i]);
    histos[i]->SetMarkerStyle(20);

    invHistos[i]->SetLineColor(def_colors[i]);
    invHistos[i]->SetMarkerStyle(kOpenCross);
    invHistos[i]->SetMarkerSize(1.2);
    invHistos[i]->SetMarkerColor(def_colors[i]);
    invHistos[i]->GetXaxis()->SetTitle(histos[0]->GetXaxis()->GetTitle());
    array->Add(histos[i]);
  }

  std::pair<Double_t, Double_t> extrema = getExtrema(array);
  delete array;
  float min = (extrema.first > 0) ? (extrema.first) * 0.7 : (extrema.first) * 1.3;
  histos[0]->GetYaxis()->SetRangeUser(min, extrema.second * 1.3);

  TRatioPlot *rp{nullptr};

  for (unsigned int i = 0; i < histos.size(); i++) {
    invHistos[i]->SetLineWidth(2);
    if (i == 0) {
      rp = new TRatioPlot(invHistos[0], histos[0]);
      rp->SetLeftMargin(0.15);
      rp->SetRightMargin(0.05);
      rp->SetSeparationMargin(0.01);
      rp->SetLowBottomMargin(0.35);
      rp->Draw("hist");
    } else {
      rp->GetUpperPad()->cd();
      invHistos[i]->Draw("same hist");
      histos[i]->Draw("same p0");
    }
  }

  if (!rp) {
    std::cerr << "TRatioPlot could not be initialized, exiting!" << std::endl;
    return;
  }

  rp->GetUpperPad()->cd();

  // Draw the legend
  TLegend *infoBox = new TLegend(0.4, 0.75, 0.65, 0.90, "");
  infoBox->SetShadowColor(0);  // 0 = transparent
  infoBox->SetFillColor(kWhite);
  infoBox->SetTextSize(0.035);

  for (unsigned int i = 0; i < histos.size(); i++) {
    infoBox->AddEntry(histos[i], labels[i], "P");
  }
  infoBox->Draw("same");

  MakeNicePlotStyle<TGraph>(rp->GetLowerRefGraph());
  rp->GetLowerRefGraph()->GetYaxis()->SetTitle("ratio");
  rp->GetLowerRefGraph()->SetMinimum(0.3);
  rp->GetLowerRefGraph()->SetMaximum(1.7);
  rp->GetLowerRefGraph()->SetLineColor(def_colors[0]);
  rp->GetLowerRefGraph()->SetMarkerColor(def_colors[0]);

  for (unsigned int i = 1; i < histos.size(); i++) {
    TLine *line = new TLine(gPad->GetUxmin(), 1, gPad->GetUxmax(), 1);
    auto c2 = new TCanvas(Form("c2_%s_%i", histos[i]->GetName(), i), "A ratio example 2", 800, 800);
    c2->cd();
    auto rp2 = new TRatioPlot(invHistos[i], histos[i]);
    rp2->Draw();
    TGraph *g = rp2->GetLowerRefGraph();
    // if(g)
    MakeNicePlotStyle<TGraph>(g);
    g->SetLineColor(def_colors[i]);
    g->SetMarkerColor(def_colors[i]);
    //~ g->GetXaxis()->SetTitle(histos[0]->GetXaxis()->GetTitle());

    c1->cd();
    rp->GetLowerPad()->cd();
    line->Draw("same");
    if (g)
      g->Draw("same P");
    c1->Update();
    delete c2;
  }

  CMS_lumi(c1, 0, 3, Rlabel);
  c1->SaveAs(TString(histos[0]->GetName()) + "_mirrored.png");
  c1->SaveAs(TString(histos[0]->GetName()) + "_mirrored.pdf");
  delete c1;
}

/************************************************/
void recurseOverKeys(TDirectory *target1, const TString &label1, const TString &label2)
/************************************************/
{
  // Figure out where we are
  TString path((char *)strstr(target1->GetPath(), ":"));
  path.Remove(0, 2);

  sourceFile1->cd(path);

  std::cout << path << std::endl;

  TDirectory *current_sourcedir = gDirectory;

  TKey *key;
  TIter nextkey(current_sourcedir->GetListOfKeys());

  while ((key = (TKey *)nextkey())) {
    auto obj = key->ReadObj();

    // Check if this is a 1D histogram or a directory
    if (obj->IsA()->InheritsFrom("TH1F")) {
      // **************************
      // Plot & Save this Histogram
      TH1F *htemp1, *htemp2;

      htemp1 = (TH1F *)obj;
      TString histName = htemp1->GetName();

      if (path != "") {
        sourceFile2->GetObject(path + "/" + histName, htemp2);
      } else {
        sourceFile2->GetObject(histName, htemp2);
      }

      //outputFilename=histName;
      //plot2Histograms(htemp1, htemp2, outputFolder+path+"/"+outputFilename+"."+imageType);
      plot2Histograms(htemp1, htemp2, label1, label2);

    } else if (obj->IsA()->InheritsFrom("TProfile")) {
      // **************************
      // Plot & Save this Histogram
      TProfile *htemp1, *htemp2;

      htemp1 = (TProfile *)obj;
      TString histName = htemp1->GetName();

      if (path != "") {
        sourceFile2->GetObject(path + "/" + histName, htemp2);
      } else {
        sourceFile2->GetObject(histName, htemp2);
      }

      plot2Profiles(htemp1, htemp2, label1, label2);
    } else if (obj->IsA()->InheritsFrom("TDirectory")) {
      // it's a subdirectory

      std::cout << "Found subdirectory " << obj->GetName() << std::endl;
      //gSystem->MakeDirectory(outputFolder+path+"/"+obj->GetName());

      // obj is now the starting point of another round of merging
      // obj still knows its depth within the target file via
      // GetPath(), so we can still figure out where we are in the recursion

      if ((TString(obj->GetName())).Contains("DQM") && !(TString(obj->GetName())).Contains("DQMData"))
        continue;

      recurseOverKeys((TDirectory *)obj, label1, label2);

    }  // end of IF a TDriectory
  }
}

/************************************************/
void plot2Profiles(TProfile *h1, TProfile *h2, const TString &label1, const TString &label2)
/************************************************/
{
  auto c1 = new TCanvas(Form("c1_%s", h1->GetName()), "example", 800, 800);
  c1->SetLeftMargin(0.15);
  c1->SetRightMargin(0.03);
  gStyle->SetOptStat(0);

  h1->SetLineColor(kBlue);
  h2->SetLineColor(kRed);

  h1->SetMarkerColor(kBlue);
  h2->SetMarkerColor(kRed);

  h1->SetMarkerStyle(20);
  h2->SetMarkerStyle(21);

  MakeNiceProfile(h1);
  MakeNiceProfile(h2);

  TObjArray *array = new TObjArray(2);
  array->Add(h1);
  array->Add(h2);

  std::pair<Double_t, Double_t> extrema = getExtrema(array);

  delete array;

  float min = (extrema.first > 0) ? (extrema.first) * 0.99 : (extrema.first) * 1.01;

  h1->GetYaxis()->SetRangeUser(min, extrema.second * 1.01);
  h2->GetYaxis()->SetRangeUser(min, extrema.second * 1.01);
  c1->cd();
  h1->Draw();
  h2->Draw("same");

  TLegend *infoBox = new TLegend(0.75, 0.75, 0.97, 0.90, "");
  infoBox->AddEntry(h1, label1, "PL");
  infoBox->AddEntry(h2, label2, "PL");
  infoBox->SetShadowColor(0);  // 0 = transparent
  infoBox->SetFillColor(kWhite);
  infoBox->Draw("same");

  c1->SaveAs(TString(h1->GetName()) + ".png");
  delete c1;
}

/************************************************/
void plot2Histograms(TH1 *h1, TH1 *h2, const TString &label1, const TString &label2) {
  /************************************************/

  TGaxis::SetMaxDigits(3);

  auto c1 = new TCanvas(Form("c1_%s", h1->GetName()), "A ratio example", 800, 800);
  gStyle->SetOptStat(0);

  MakeNicePlotStyle<TH1>(h1);
  MakeNicePlotStyle<TH1>(h2);

  h1->SetLineColor(kBlue);
  h2->SetLineColor(kRed);

  TObjArray *array = new TObjArray(2);
  array->Add(h1);
  array->Add(h2);

  std::pair<Double_t, Double_t> extrema = getExtrema(array);

  delete array;

  float min = (extrema.first > 0) ? (extrema.first) * 0.7 : (extrema.first) * 1.3;

  h1->GetYaxis()->SetRangeUser(min, extrema.second * 1.3);
  h2->GetYaxis()->SetRangeUser(min, extrema.second * 1.3);

  auto rp = new TRatioPlot(h1, h2);
  c1->SetTicks(0, 1);
  rp->Draw();

  //rp->GetUpperPad()->SetTopMargin(0.09);
  //rp->GetUpperPad()->SetLeftMargin(0.15);
  //rp->GetUpperPad()->SetRightMargin(0.03);
  //rp->GetLowerPad()->SetBottomMargin(0.5);

  rp->SetLeftMargin(0.15);
  rp->SetRightMargin(0.03);
  rp->SetSeparationMargin(0.01);
  rp->SetLowBottomMargin(0.35);

  rp->GetUpperPad()->cd();
  // Draw the legend
  TLegend *infoBox = new TLegend(0.75, 0.75, 0.97, 0.90, "");
  infoBox->AddEntry(h1, label1, "L");
  infoBox->AddEntry(h2, label2, "L");
  infoBox->SetShadowColor(0);  // 0 = transparent
  infoBox->SetFillColor(kWhite);
  infoBox->Draw("same");

  MakeNicePlotStyle<TGraph>(rp->GetLowerRefGraph());
  rp->GetLowerRefGraph()->GetYaxis()->SetTitle("ratio");
  rp->GetLowerRefGraph()->SetMinimum(0.);
  rp->GetLowerRefGraph()->SetMaximum(2.);
  c1->Update();

  //rp->GetLowerPad()->cd();
  //c1->Update();

  c1->SaveAs(TString(h1->GetName()) + ".png");
  delete c1;
}

/*--------------------------------------------------------------------*/
template <typename T>
void MakeNicePlotStyle(T *hist)
/*--------------------------------------------------------------------*/
{
  //hist->SetStats(kFALSE);
  hist->SetLineWidth(2);
  hist->GetXaxis()->SetNdivisions(505);
  hist->GetXaxis()->CenterTitle(true);
  hist->GetYaxis()->CenterTitle(true);
  hist->GetXaxis()->SetTitleFont(42);
  hist->GetYaxis()->SetTitleFont(42);
  hist->GetXaxis()->SetTitleSize(0.05);
  hist->GetYaxis()->SetTitleSize(0.05);
  hist->GetXaxis()->SetTitleOffset(0.9);
  hist->GetYaxis()->SetTitleOffset(1.4);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  if (((TObject *)hist)->IsA()->InheritsFrom("TGraph")) {
    hist->GetYaxis()->SetLabelSize(.025);
    //hist->GetYaxis()->SetNdivisions(505);
  } else {
    hist->GetYaxis()->SetLabelSize(.05);
  }
  hist->GetXaxis()->SetLabelSize(.05);
}

/*--------------------------------------------------------------------*/
void MakeNiceProfile(TProfile *prof)
/*--------------------------------------------------------------------*/
{
  prof->SetLineWidth(2);
  prof->GetXaxis()->SetNdivisions(505);
  prof->GetXaxis()->CenterTitle(true);
  prof->GetYaxis()->CenterTitle(true);
  prof->GetXaxis()->SetTitleFont(42);
  prof->GetYaxis()->SetTitleFont(42);
  prof->GetXaxis()->SetTitleSize(0.05);
  prof->GetYaxis()->SetTitleSize(0.05);
  prof->GetXaxis()->SetTitleOffset(0.9);
  prof->GetYaxis()->SetTitleOffset(1.4);
  prof->GetXaxis()->SetLabelFont(42);
  prof->GetYaxis()->SetLabelFont(42);
  prof->GetYaxis()->SetLabelSize(.05);
  prof->GetXaxis()->SetLabelSize(.05);
}

//*****************************************************//
std::pair<Double_t, Double_t> getExtrema(TObjArray *array)
//*****************************************************//
{
  Double_t theMaximum = (static_cast<TH1 *>(array->At(0)))->GetMaximum();
  Double_t theMinimum = (static_cast<TH1 *>(array->At(0)))->GetMinimum();
  for (Int_t i = 0; i < array->GetSize(); i++) {
    if ((static_cast<TH1 *>(array->At(i)))->GetMaximum() > theMaximum) {
      theMaximum = (static_cast<TH1 *>(array->At(i)))->GetMaximum();
    }
    if ((static_cast<TH1 *>(array->At(i)))->GetMinimum() < theMinimum) {
      theMinimum = (static_cast<TH1 *>(array->At(i)))->GetMinimum();
    }
  }
  return std::make_pair(theMinimum, theMaximum);
}
-- dummy change --
