///////////////////////////////////////////////////////////////////////////////
//
//   plotEffiAll(approve, ratio, savePlot);
//      Plots reconstructionefficiency as a function of p, pt, eta, phi
//
//   approve  (bool)   To include "CMS Preliminary" or not (true)
//   ratio    (bool)   If the ratio with the first input to be plotted (true)
//   savePlot (int)    Saving the plot or not: (no:gif:jpg:pdf -1:0:1:2) (2)
//
//   plotCompareAll(cdir1, cdir2, cvers, cfil1, cfil2, ctype1, ctype2,
//                  postfix, ratio, logy, save, norm);
//       Plots comparison plots of calorimetric quatities from native/vecgeom
//       versions
//
//   cdir1   (std::string) Name of the native directory ("10.7.r06.g4")
//   cdir2   (std::string) Name of the vecgeom directory ("10.7.r06.vg")
//   cvers   (std::string) Name of the input version ("10.7.r06 MinBias")
//   cfil1   (std::string) Name of the first input file ("minbias.root")
//   cfil2   (std::string) Name of the second input file ("minbias.root")
//   ctype1  (std::string) Name of the first type ("Native")
//   ctype2  (std::string) Name of the second type ("VecGeom v1.1.16")
//   postfix (std::string) Tag to be given to the canvas name ("MBR")
//   ratio   (bool)        If the ratio to be plotted (false)
//   logy    (bool)        If the scale is log or linear (true)
//   save    (bool)        If the canvas is to be saved (true)
//   norm    (bool)        If the plots to be normalized (true)
//
///////////////////////////////////////////////////////////////////////////////
#include "TCanvas.h"
#include "TDirectory.h"
#include "TF1.h"
#include "TFile.h"
#include "TFitResult.h"
#include "TGraph.h"
#include "TGraphAsymmErrors.h"
#include "TH1D.h"
#include "TH2D.h"
#include "THStack.h"
#include "TLegend.h"
#include "TMath.h"
#include "TProfile.h"
#include "TPaveStats.h"
#include "TPaveText.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

TH1D* getEffi(TFile* file, std::string varname, unsigned int ifl);
TCanvas* plotEffi(int type, bool approve);
TCanvas* plotEffiRatio(int type, bool approve);
void plotEffiAll(bool approve = true, bool ratio = true, int savePlot = 2);
void plotCompare(const char* infile1,
                 const char* text1,
                 const char* infile2,
                 const char* text2,
                 int type1 = -1,
                 int type2 = -1,
                 int type3 = -1,
                 bool ratio = false,
                 bool logy = true,
                 std::string postfix = "",
                 bool save = false,
                 bool normalize = false);
void plotCompareAll(std::string cdir1 = "11.0.r06.g4",
                    std::string cdir2 = "11.0.r06.vg",
                    std::string cvers = "11.0.r06 MinBias",
                    std::string cfil1 = "minbias.root",
                    std::string cfil2 = "minbias.root",
                    std::string ctype1 = "Native",
                    std::string ctype2 = "VecGeom v1.1.20",
                    std::string postfix = "MBR",
                    bool ratio = false,
                    bool logy = true,
                    bool save = true,
                    bool norm = true);

int markStyle[7] = {23, 22, 24, 25, 21, 33, 20};
int colors[7] = {2, 4, 6, 7, 46, 3, 1};
int lineStyle[7] = {1, 2, 3, 4, 1, 2, 3};

/*
const unsigned int nmodels = 4;
std::string filem[nmodels] = {"pikp/FBE7r00vMixStudyHLT.root",
			      "pikp/FBE7p01vMixStudyHLT.root",
			      "pikp/FBE7p02vMixStudyHLT.root",
			      "pikp/FBE7p03vMixStudyHLT.root"};
std::string typem[nmodels] = {"10.7  FTFP_BERT_EMM",
			      "10.7.p01 FTFP_BERT_EMM",
			      "10.7.p02 FTFP_BERT_EMM",
			      "10.7.p03 FTFP_BERT_EMM"};
*/
const unsigned int nmodels = 5;
std::string filem[nmodels] = {"pikp/FBE4p3vMixStudyHLT.root",
                              "pikp/FBE7p02vMixStudyHLT.root",
                              "pikp/FBE110p01vMixStudyHLT.root",
                              "pikp/FBE110r04MixStudyHLT.root",
                              "pikp/FBE110r06vMixStudyHLT.root"};
std::string typem[nmodels] = {"10.4.p03 FTFP_BERT_EMM",
                              "10.7.p01 FTFP_BERT_EMM",
                              "11.0.p01 FTFP_BERT_EMM",
                              "11.0.ref04 FTFP_BERT_EMM",
                              "11.0.ref06 FTFP_BERT_EMM"};

TH1D* getEffi(TFile* file, std::string varname, unsigned int ifl) {
  char name[100];
  sprintf(name, "h_%s_All_0", varname.c_str());
  TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
  sprintf(name, "h_%s_All_1", varname.c_str());
  TH1D* hist2 = (TH1D*)file->FindObjectAny(name);
  if (hist1 && hist2) {
    sprintf(name, "h_%s_Effy_%d", varname.c_str(), ifl);
    int nbins = hist1->GetNbinsX();
    double xl = hist1->GetBinLowEdge(1);
    double xh = hist1->GetBinLowEdge(nbins) + hist1->GetBinWidth(nbins);
    TH1D* hist = new TH1D(name, hist1->GetTitle(), nbins, xl, xh);
    for (int i = 1; i < nbins; ++i) {
      double den = hist1->GetBinContent(i);
      double val = (den > 0) ? (hist2->GetBinContent(i)) / den : 0;
      double err = (den > 0) ? (hist1->GetBinError(i)) * (val / den) : 0;
      hist->SetBinContent(i, val);
      hist->SetBinError(i, err);
    }
    return hist;
  } else {
    return 0;
  }
}

TCanvas* plotEffi(int type, bool approve) {
  std::string varnam[4] = {"pt", "p", "eta", "phi"};
  std::string xvtitl[4] = {"p_{T} (GeV)", "p (GeV)", "#eta", "#phi"};
  bool irng[4] = {true, true, true, false};
  double xlowr[4] = {0.0, 0.0, -2.2, -3.1415926};
  double xtopr[4] = {20.0, 20.0, 2.2, 3.1415926};

  TCanvas* c(0);
  if (type < 0 || type > 3)
    type = 0;
  TObjArray histArr;
  for (unsigned k = 0; k < nmodels; ++k) {
    TFile* file = TFile::Open(filem[k].c_str());
    TH1D* hist = getEffi(file, varnam[type], k);
    if (hist) {
      hist->GetXaxis()->SetTitle(xvtitl[type].c_str());
      hist->GetYaxis()->SetTitle("Efficiency");
      if (irng[type])
        hist->GetXaxis()->SetRangeUser(xlowr[type], xtopr[type]);
      histArr.AddLast(hist);
    }
  }
  if (histArr.GetEntries() > 0) {
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);
    gStyle->SetFillColor(kWhite);
    gStyle->SetOptTitle(kFALSE);
    gStyle->SetPadBorderMode(0);
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetOptStat(0);

    char cname[50];
    sprintf(cname, "c_%sEff", varnam[type].c_str());
    c = new TCanvas(cname, cname, 500, 500);
    gPad->SetTopMargin(0.10);
    gPad->SetBottomMargin(0.10);
    gPad->SetLeftMargin(0.15);
    gPad->SetRightMargin(0.025);

    TLegend* legend = new TLegend(0.30, 0.15, 0.975, 0.30);
    TPaveText* text1 = new TPaveText(0.05, 0.94, 0.35, 0.99, "brNDC");
    legend->SetBorderSize(1);
    legend->SetFillColor(kWhite);
    char texts[200];
    sprintf(texts, "CMS Preliminary");
    text1->AddText(texts);
    THStack* Hs = new THStack("hs2", " ");
    for (int i = 0; i < histArr.GetEntries(); i++) {
      TH1D* h = (TH1D*)histArr[i];
      h->SetLineColor(colors[i]);
      h->SetLineWidth(2);
      h->SetMarkerSize(markStyle[i]);
      Hs->Add(h, "hist sames");
      legend->AddEntry(h, typem[i].c_str(), "l");
    }
    Hs->Draw("nostack");
    c->Update();
    Hs->GetHistogram()->GetXaxis()->SetTitle(xvtitl[type].c_str());
    Hs->GetHistogram()->GetXaxis()->SetLabelSize(0.035);
    Hs->GetHistogram()->GetYaxis()->SetTitleOffset(1.6);
    Hs->GetHistogram()->GetYaxis()->SetTitle("Track Reconstruction Efficiency");
    Hs->GetHistogram()->GetYaxis()->SetRangeUser(0.0, 1.2);
    if (irng[type])
      Hs->GetHistogram()->GetXaxis()->SetRangeUser(xlowr[type], xtopr[type]);
    c->Modified();
    c->Update();
    legend->Draw("");
    if (approve)
      text1->Draw("same");
    c->Modified();
    c->Update();
  }
  return c;
}

TCanvas* plotEffiRatio(int type, bool approve) {
  std::string varnam[4] = {"pt", "p", "eta", "phi"};
  std::string xvtitl[4] = {"p_{T} (GeV)", "p (GeV)", "#eta", "#phi"};
  bool irng[4] = {true, true, true, false};
  double xlowr[4] = {0.0, 0.0, -2.2, -3.1415926};
  double xtopr[4] = {20.0, 20.0, 2.2, 3.1415926};

  TCanvas* c(0);
  if (type < 0 || type > 3)
    type = 0;
  TObjArray histArr;
  TH1D* hist0(0);
  for (unsigned k = 0; k < nmodels; ++k) {
    TFile* file = TFile::Open(filem[k].c_str());
    TH1D* hist = getEffi(file, varnam[type], k);
    if (hist) {
      if (hist0 == nullptr) {
        hist0 = hist;
      } else {
        int nbin = hist->GetNbinsX();
        int npt1(0), npt2(0);
        double sumNum(0), sumDen(0);
        for (int i = 1; i < nbin; ++i) {
          double val1 = hist->GetBinContent(i);
          double err1 = (val1 > 0) ? ((hist->GetBinError(i)) / val1) : 0;
          double val2 = hist0->GetBinContent(i);
          double err2 = (val2 > 0) ? ((hist0->GetBinError(i)) / val2) : 0;
          double ratio = (val2 > 0) ? (val1 / val2) : 0;
          double drat = ratio * sqrt(err1 * err1 + err2 * err2);
          if ((((hist->GetBinLowEdge(i)) >= xlowr[type]) &&
               ((hist->GetBinLowEdge(i) + hist->GetBinWidth(i)) <= xtopr[type])) ||
              (!irng[type])) {
            ++npt1;
            if (val2 > 0) {
              double temp1 = (ratio > 1.0) ? 1.0 / ratio : ratio;
              double temp2 = (ratio > 1.0) ? drat / (ratio * ratio) : drat;
              sumNum += (fabs(1 - temp1) / (temp2 * temp2));
              sumDen += (1.0 / (temp2 * temp2));
              ++npt2;
            }
          }
          hist->SetBinContent(i, ratio);
          hist->SetBinError(i, drat);
        }
        sumNum = (sumDen > 0) ? (sumNum / sumDen) : 0;
        sumDen = (sumDen > 0) ? 1.0 / sqrt(sumDen) : 0;
        std::cout << "Get Ratio of mean for " << npt2 << ":" << npt1 << ":" << nbin << " points: Mean " << sumNum
                  << " +- " << sumDen << " from " << filem[k] << std::endl;
        hist->GetXaxis()->SetTitle(xvtitl[type].c_str());
        hist->GetYaxis()->SetTitle("Efficiency Ratio");
        hist->GetYaxis()->SetRangeUser(0.8, 1.2);
        if (irng[type])
          hist->GetXaxis()->SetRangeUser(xlowr[type], xtopr[type]);
        histArr.AddLast(hist);
      }
    }
  }

  if (histArr.GetEntries() > 0) {
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);
    gStyle->SetFillColor(kWhite);
    gStyle->SetOptTitle(kFALSE);
    gStyle->SetPadBorderMode(0);
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetOptStat(0);

    char cname[50];
    sprintf(cname, "c_%sEffRat", varnam[type].c_str());
    c = new TCanvas(cname, cname, 500, 500);
    gPad->SetTopMargin(0.10);
    gPad->SetBottomMargin(0.10);
    gPad->SetLeftMargin(0.15);
    gPad->SetRightMargin(0.025);

    TLegend* legend = new TLegend(0.30, 0.15, 0.975, 0.30);
    TPaveText* text1 = new TPaveText(0.05, 0.94, 0.35, 0.99, "brNDC");
    legend->SetBorderSize(1);
    legend->SetFillColor(kWhite);
    char texts[200];
    sprintf(texts, "CMS Preliminary");
    text1->AddText(texts);
    THStack* Hs = new THStack("hs2", " ");
    for (int i = 0; i < histArr.GetEntries(); i++) {
      TH1D* h = (TH1D*)histArr[i];
      h->SetLineColor(colors[i + 1]);
      h->SetLineWidth(2);
      h->SetMarkerSize(markStyle[i + 1]);
      Hs->Add(h, "hist sames");
      legend->AddEntry(h, typem[i + 1].c_str(), "l");
    }
    Hs->Draw("nostack");
    c->Update();
    Hs->GetHistogram()->GetXaxis()->SetTitle(xvtitl[type].c_str());
    Hs->GetHistogram()->GetXaxis()->SetLabelSize(0.035);
    Hs->GetHistogram()->GetYaxis()->SetTitleOffset(1.6);
    Hs->GetHistogram()->GetYaxis()->SetTitle("Track Reconstruction Efficiency Ratio");
    Hs->GetHistogram()->GetYaxis()->SetRangeUser(0.8, 1.2);
    if (irng[type])
      Hs->GetHistogram()->GetXaxis()->SetRangeUser(xlowr[type], xtopr[type]);
    c->Modified();
    c->Update();
    legend->Draw("");
    if (approve)
      text1->Draw("same");
    c->Modified();
    c->Update();
  }
  return c;
}

void plotEffiAll(bool approve, bool ratio, int savePlot) {
  for (int var = 0; var <= 4; ++var) {
    TCanvas* c = (ratio) ? plotEffiRatio(var, approve) : plotEffi(var, approve);
    if (c != 0 && savePlot >= 0 && savePlot < 3) {
      std::string ext[3] = {"eps", "gif", "pdf"};
      char name[200];
      sprintf(name, "%s.%s", c->GetName(), ext[savePlot].c_str());
      c->Print(name);
    }
  }
}

void plotCompare(const char* infile1,
                 const char* text1,
                 const char* infile2,
                 const char* text2,
                 int type1,
                 int type2,
                 int type3,
                 bool ratio,
                 bool logy,
                 std::string postfix,
                 bool save,
                 bool normalize) {
  int ndets[4] = {1, 9, 9, 15};
  int types[4] = {7, 8, 2, 3};
  std::string htype0[7] = {"PtInc", "EneInc", "EtaInc", "PhiInc", "HitLow", "HitHigh", "HitMu"};
  int rebin0[7] = {1, 1, 1, 1, 10, 10, 10};
  std::string htype1[8] = {"Hit", "Time", "Edep", "Etot", "TimeAll", "EdepEM", "EdepHad", "EtotG"};
  int rebin1[8] = {10, 10, 100, 100, 10, 50, 50, 50};
  std::string htype2[2] = {"EdepCal", "EdepCalT"};
  int rebin2[2] = {50, 50};
  std::string htype3[3] = {"HitTk", "TimeTk", "EdepTk"};
  int rebin3[3] = {10, 10, 50};
  const double ymin(10.0);

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptFit(0);
  if (ratio || normalize)
    gStyle->SetOptStat(0);
  else
    gStyle->SetOptStat(1110);

  int itmin1 = (type1 >= 0) ? type1 : 0;
  int itmax1 = (type1 >= 0) ? type1 : 3;
  TFile* file1 = new TFile(infile1);
  TFile* file2 = new TFile(infile2);
  std::cout << "File1: " << infile1 << ":" << file1 << " File2: " << infile2 << ":" << file2 << std::endl;
  if (file1 != 0 && file2 != 0) {
    for (int it1 = itmin1; it1 <= itmax1; ++it1) {
      int itmin2 = (type2 >= 0) ? type2 : 0;
      int itmax2 = (type2 >= 0) ? type2 : ((type1 == 1) ? 3 : types[it1] - 1);
      int itmin3 = (type3 >= 0) ? type3 : 0;
      int itmax3 = (type3 >= 0) ? type3 : ndets[it1] - 1;
      for (int it2 = itmin2; it2 <= itmax2; ++it2) {
        int rebin(1);
        if (it1 == 0)
          rebin = rebin0[it2];
        else if (it1 == 1)
          rebin = rebin1[it2];
        else if (it1 == 2)
          rebin = rebin2[it2];
        else if (it1 == 3)
          rebin = rebin3[it2];
        for (int it3 = itmin3; it3 <= itmax3; ++it3) {
          if (type1 == 1 && (it3 == 1 || it3 == 2))
            continue;
          char name[20], namec[22];
          if (it1 == 0)
            sprintf(name, "%s", htype0[it2].c_str());
          else if (it1 == 1)
            sprintf(name, "%s%d", htype1[it2].c_str(), it3);
          else if (it1 == 2)
            sprintf(name, "%s%d", htype2[it2].c_str(), it3);
          else
            sprintf(name, "%s%d", htype3[it2].c_str(), it3);
          TH1D* hist[2];
          hist[0] = (TH1D*)file1->FindObjectAny(name);
          hist[1] = (TH1D*)file2->FindObjectAny(name);
          std::cout << name << " Hist " << hist[0] << ":" << hist[1] << std::endl;
          if (hist[0] != 0 && hist[1] != 0) {
            sprintf(namec, "c_%s%s", name, postfix.c_str());
            TCanvas* pad = new TCanvas(namec, namec, 500, 500);
            pad->SetRightMargin(0.10);
            pad->SetTopMargin(0.10);
            if (logy)
              pad->SetLogy();
            double xedge = (ratio) ? 0.88 : 0.64;
            TLegend* legend = new TLegend(0.12, 0.79, xedge, 0.89);
            legend->SetFillColor(kWhite);
            if (ratio) {
              int nbin = hist[0]->GetNbinsX();
              int nbinX = nbin / rebin;
              sprintf(namec, "%sR", name);
              TH1D* hist0 = (TH1D*)gROOT->FindObjectAny(namec);
              if (hist0)
                hist0->Delete();
              double xlow = hist[0]->GetBinLowEdge(1);
              double xhigh = hist[0]->GetBinLowEdge(nbin) + hist[0]->GetBinWidth(nbin);
              hist0 = new TH1D(namec, hist[0]->GetTitle(), nbinX, xlow, xhigh);
              int npt1(0), npt2(0);
              double sumNum(0), sumDen(0);
              xhigh = hist0->GetBinLowEdge(1) + hist0->GetBinWidth(1);
              double fact = (hist[1]->GetEntries()) / (hist[0]->GetEntries());
              bool empty(false);
              for (int i = 1; i < nbinX; ++i) {
                double val1(0), err1(0), val2(0), err2(0);
                for (int j = 0; j < rebin; ++j) {
                  int i1 = (i - 1) * rebin + j + 1;
                  val1 += hist[0]->GetBinContent(i1);
                  val2 += hist[1]->GetBinContent(i1);
                  err1 += ((hist[0]->GetBinError(i1)) * (hist[0]->GetBinError(i1)));
                  err2 += ((hist[1]->GetBinError(i1)) * (hist[1]->GetBinError(i1)));
                }
                err1 = (val1 > 0) ? (sqrt(err1) / val1) : 0;
                err2 = (val2 > 0) ? (sqrt(err2) / val2) : 0;
                double ratio = (((val1 > 0) && (val2 > 0)) ? fact * (val1 / val2) : 1);
                double drat = (((val1 > 0) && (val2 > 0)) ? (ratio * sqrt(err1 * err1 + err2 * err2)) : 0);
                ++npt1;
                if ((val1 > ymin) && (val2 > ymin)) {
                  double temp1 = (ratio > 1.0) ? 1.0 / ratio : ratio;
                  double temp2 = (ratio > 1.0) ? drat / (ratio * ratio) : drat;
                  sumNum += (fabs(1 - temp1) / (temp2 * temp2));
                  sumDen += (1.0 / (temp2 * temp2));
                  ++npt2;
                }
                hist0->SetBinContent(i, ratio);
                hist0->SetBinError(i, drat);
                if (i <= 10)
                  xhigh = (hist0->GetBinLowEdge(i) + hist0->GetBinWidth(i));
                else if (val1 > 0 && val2 > 0 && (!empty))
                  xhigh = (hist0->GetBinLowEdge(i) + hist0->GetBinWidth(i));
                else if ((val1 <= 0 || val2 <= 0) && i > 10)
                  empty = true;
              }
              sumNum = (sumDen > 0) ? (sumNum / sumDen) : 0;
              sumDen = (sumDen > 0) ? 1.0 / sqrt(sumDen) : 0;
              std::cout << "Get Ratio of mean for " << hist[0]->GetTitle() << " with " << npt2 << ":" << npt1 << ":"
                        << nbinX << " points: Mean " << sumNum << " +- " << sumDen << std::endl;
              if (it1 == 0)
                sprintf(namec, "Ratio for %s", htype0[it2].c_str());
              else if (it1 == 1)
                sprintf(namec, "Ratio for %s", htype1[it2].c_str());
              else if (it1 == 2)
                sprintf(namec, "Ratio for %s", htype2[it2].c_str());
              else
                sprintf(namec, "Ratio for %s", htype3[it2].c_str());
              hist0->SetMarkerStyle(markStyle[0]);
              hist0->SetMarkerColor(colors[0]);
              hist0->SetLineStyle(lineStyle[0]);
              hist0->SetLineColor(colors[0]);
              hist0->GetXaxis()->SetTitle(hist[0]->GetTitle());
              hist0->GetXaxis()->SetRangeUser(xlow, xhigh);
              hist0->GetYaxis()->SetTitle(namec);
              hist0->GetYaxis()->SetRangeUser(0.0, 2.0);
              hist0->GetYaxis()->SetTitleOffset(1.3);
              sprintf(namec, "%s vs %s", text1, text2);
              legend->AddEntry(hist0, namec, "lp");
              hist0->Draw();
              pad->Update();
              TLine* line = new TLine(xlow, 1.0, xhigh, 1.0);
              line->SetLineWidth(2);
              line->SetLineStyle(2);
              line->Draw("same");
              TPaveText* text0 = new TPaveText(0.12, 0.12, 0.65, 0.17, "brNDC");
              char texts[200];
              sprintf(texts, "Mean Deviation = %5.3f #pm %5.3f", sumNum, sumDen);
              text0->SetFillColor(kWhite);
              text0->AddText(texts);
              text0->Draw("same");
            } else {
              double ymax(0.90), dy(0.08);
              for (int ih = 0; ih < 2; ++ih) {
                hist[ih]->GetXaxis()->SetTitle(hist[ih]->GetTitle());
                hist[ih]->SetMarkerStyle(markStyle[ih]);
                hist[ih]->SetMarkerColor(colors[ih]);
                hist[ih]->SetLineStyle(lineStyle[ih]);
                hist[ih]->SetLineColor(colors[ih]);
                hist[ih]->SetLineWidth(2);
                hist[ih]->GetYaxis()->SetTitleOffset(1.20);
                if (rebin > 1)
                  hist[ih]->Rebin(rebin);
                if (ih == 0) {
                  legend->AddEntry(hist[ih], text1, "lp");
                  if (normalize)
                    hist[ih]->DrawNormalized("hist");
                  else
                    hist[ih]->Draw();
                } else {
                  legend->AddEntry(hist[ih], text2, "lp");
                  if (normalize)
                    hist[ih]->DrawNormalized("sames hist");
                  else
                    hist[ih]->Draw("sames");
                }
                pad->Update();
                TPaveStats* st1 = (TPaveStats*)hist[ih]->GetListOfFunctions()->FindObject("stats");
                if (st1 != NULL) {
                  st1->SetLineColor(colors[ih]);
                  st1->SetTextColor(colors[ih]);
                  st1->SetY1NDC(ymax - dy);
                  st1->SetY2NDC(ymax);
                  st1->SetX1NDC(0.65);
                  st1->SetX2NDC(0.90);
                  ymax -= dy;
                }
                pad->Update();
              }
            }
            legend->Draw("same");
            pad->Update();
            if (save) {
              sprintf(name, "%s.pdf", pad->GetName());
              pad->Print(name);
            }
          }
        }
      }
    }
  }
}

void plotCompareAll(std::string cdir1,
                    std::string cdir2,
                    std::string cvers,
                    std::string cfil1,
                    std::string cfil2,
                    std::string ctype1,
                    std::string ctype2,
                    std::string postfix,
                    bool ratio,
                    bool logy,
                    bool save,
                    bool norm) {
  char infile1[200], infile2[200], text1[200], text2[200];
  sprintf(infile1, "%s/%s", cdir1.c_str(), cfil1.c_str());
  sprintf(infile2, "%s/%s", cdir2.c_str(), cfil2.c_str());
  sprintf(text1, "%s (%s)", cvers.c_str(), ctype1.c_str());
  sprintf(text2, "%s (%s)", cvers.c_str(), ctype2.c_str());
  plotCompare(infile1, text1, infile2, text2, 1, -1, 0, ratio, logy, postfix, save, norm);
  plotCompare(infile1, text1, infile2, text2, 1, -1, 3, ratio, logy, postfix, save, norm);
  plotCompare(infile1, text1, infile2, text2, 1, -1, 4, ratio, logy, postfix, save, norm);
  plotCompare(infile1, text1, infile2, text2, 1, -1, 5, ratio, logy, postfix, save, norm);
  plotCompare(infile1, text1, infile2, text2, 1, -1, 6, ratio, logy, postfix, save, norm);
  plotCompare(infile1, text1, infile2, text2, 1, -1, 7, ratio, logy, postfix, save, norm);
  plotCompare(infile1, text1, infile2, text2, 1, -1, 8, ratio, logy, postfix, save, norm);
}
