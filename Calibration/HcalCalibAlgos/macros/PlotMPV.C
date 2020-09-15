//////////////////////////////////////////////////////////////////////////////
// Usage:
// .L PlotMPV.C+g
//             For carrying out exponential fit to the MPV's as a function
//             of accumulated luminosity and plotting them on canvas
//     plotMPV(infile, eta, phi, depth, first, drawleg, iyear, lumis, ener,
//             save, debug);
//
//             For combining the fit results for 2 iphi values (63, 65) by
//             weighted mean, carrying out the fit and plotting them on canvas
//     plot2MPV(infile, eta, depth, first, drawleg, iyear, lumis, ener, save,
//              debug)
//
//             Get truncated mean for a range of eta, depth, ...
//     getTruncateMean(infile, frac, type, nvx, save);
//     getTruncatedMeanX(infile, type, nvx, year, phi, save);
//
//             Draw slopes as a function of luminosity
//     drawSlope(infile, type, phi, depth, drawleg, iyear, lumis, ener,
//               save, debug)
//
//             Draw the 2D plot of light loss as a function of eta, depth
//      plotLightLoss(infile, lumis, iyear, ener, save, debug)
//
//  where
//   infile (const char*) Nme of the input file
//   eta    (int)         ieta value of the tower
//   phi    (int)         iphi value of the tower (0 if iphi's of RBX combined
//                        for 2017 and of all iphi's for 2018 and beyond)
//   depth  (int)         depth value of the tower (-1 if all depths combined)
//   first  (uint32_t)    the starting lumi point to be used (0)
//   drawleg (bool)       legend to be drawn or not (true)
//   iyear  (int)         Year of data taking, if > 1000 to be shown (17)
//   lumis  (double)      Total integrated luminosity (49.0)
//   ener   (int)         C.M. energy (13)
//   save   (bool)        if the plot to be saved as a pdf file (false)
//   debug  (bool)        For controlling debug printouts (false)
//   frac   (double)      Fraction of events to be used
//   type   (string)      Specifies the range (e.g. Data2018D7)
//   nvx    (int)         Vertex bin to be considered (e.g. 2)
//   year   (int)         Cuts made for 2017 or 2018
//
///////////////////////////////////////////////////////////////////////////////

#include <TArrow.h>
#include <TASImage.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TGraphAsymmErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <TProfile.h>
#include <TROOT.h>
#include <TStyle.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

std::vector<std::string> splitString(const std::string& fLine) {
  std::vector<std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size(); i++) {
    if (fLine[i] == ' ' || i == fLine.size()) {
      if (!empty) {
        std::string item(fLine, start, i - start);
        result.push_back(item);
        empty = true;
      }
      start = i + 1;
    } else {
      if (empty)
        empty = false;
    }
  }
  return result;
}

void meanMax(const char* infile, bool debug) {
  std::map<std::string, std::pair<double, double> > means;
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer[1024];
    unsigned int all(0), good(0);
    while (fInput.getline(buffer, 1024)) {
      ++all;
      if (buffer[0] == '-')
        continue;  //ignore comment
      std::vector<std::string> items = splitString(std::string(buffer));
      if (items.size() < 5) {
        if (debug)
          std::cout << "Ignore " << items.size() << " in line: " << buffer << std::endl;
      } else {
        ++good;
        double mean = std::atof(items[4].c_str());
        std::map<std::string, std::pair<double, double> >::iterator itr = means.find(items[2]);
        if (itr == means.end()) {
          means[items[2]] = std::pair<double, double>(mean, mean);
        } else {
          double mmin = std::min(mean, itr->second.first);
          double mmax = std::max(mean, itr->second.second);
          means[items[2]] = std::pair<double, double>(mmin, mmax);
        }
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good "
              << " records from " << infile << std::endl;
    for (std::map<std::string, std::pair<double, double> >::iterator itr = means.begin(); itr != means.end(); ++itr)
      std::cout << itr->first << " Mean " << itr->second.first << ":" << itr->second.second << std::endl;
  }
}

void readMPVs(const char* infile,
              const int eta,
              const int phi,
              const int dep,
              const unsigned int maxperiod,
              std::map<int, std::pair<double, double> >& mpvs,
              bool debug) {
  mpvs.clear();
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer[1024];
    unsigned int all(0), good(0), select(0);
    while (fInput.getline(buffer, 1024)) {
      ++all;
      if (buffer[0] == '#')
        continue;  //ignore comment
      std::vector<std::string> items = splitString(std::string(buffer));
      if (items.size() < 7) {
        if (debug)
          std::cout << "Ignore  line: " << buffer << std::endl;
      } else {
        ++good;
        int period = std::atoi(items[0].c_str());
        int ieta = std::atoi(items[1].c_str());
        int iphi = std::atoi(items[2].c_str());
        int depth = std::atoi(items[3].c_str());
        double mpv = std::atof(items[5].c_str());
        double dmpv = std::atof(items[6].c_str());
        if ((ieta == eta) && (iphi == phi) && (depth == dep) && (period > 0) && (period <= (int)(maxperiod))) {
          ++select;
          mpvs[period] = std::pair<double, double>(mpv, dmpv);
        }
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good and " << select << " selected records from "
              << infile << std::endl;
  }
  if (debug) {
    unsigned int k(0);
    for (std::map<int, std::pair<double, double> >::const_iterator itr = mpvs.begin(); itr != mpvs.end(); ++itr, ++k) {
      std::cout << "[" << k << "] Period " << itr->first << " MPV " << (itr->second).first << " +- "
                << (itr->second).second << std::endl;
    }
  }
}

struct rType {
  TCanvas* pad;
  int bin;
  double mean, error, chisq;
  rType() {
    pad = nullptr;
    bin = 0;
    mean = error = chisq = 0;
  }
};

rType drawMPV(const char* name,
              int eta,
              int phi,
              int depth,
              const unsigned int nmax,
              const std::map<int, std::pair<double, double> >& mpvs,
              unsigned int first = 0,
              bool drawleg = true,
              int iyear = 17,
              double lumis = 0,
              int ener = 13,
              int normalize = 0,
              bool save = false,
              bool debug = false) {
  rType results;
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);
  results.pad = new TCanvas(name, name);
  results.pad->SetRightMargin(0.10);
  results.pad->SetTopMargin(0.10);
  TLegend* legend = new TLegend(0.25, 0.84, 0.59, 0.89);
  legend->SetFillColor(kWhite);
  int iy = iyear % 100;
  std::vector<float> lumi, dlumi;
  double xmin(0), xmax(0);
  if (iy == 17) {
    xmax = 50;
    float lum[13] = {
        1.361, 4.540, 8.315, 12.319, 16.381, 20.828, 25.192, 29.001, 32.206, 35.634, 39.739, 43.782, 47.393};
    for (unsigned int k = 0; k < nmax; ++k) {
      lumi.push_back(lum[k]);
      dlumi.push_back(0);
    }
  } else {
    xmax = 70;
    float lum[20] = {1.531,  4.401,  7.656,  11.217, 14.133, 17.382, 21.332, 25.347, 28.982, 32.102,
                     35.191, 38.326, 41.484, 44.645, 47.740, 50.790, 53.867, 56.918, 59.939, 64.453};
    for (unsigned int k = 0; k < nmax; ++k) {
      lumi.push_back(lum[k]);
      dlumi.push_back(0);
    }
  }
  std::cout << iyear << ":" << iy << " N " << nmax << " X " << xmin << ":" << xmax << std::endl;
  const int p1[6] = {2, 5, 6, 11, 15, 19};
  if (mpvs.size() > 0) {
    float lums[nmax], dlum[nmax], mpv[nmax], dmpv[nmax];
    unsigned int np = mpvs.size();
    if (np > nmax)
      np = nmax;
    float ymin(0), ymax(0);
    unsigned int k(0), k1(0);
    double startvalues[2];
    startvalues[1] = 0;
    for (std::map<int, std::pair<double, double> >::const_iterator itr = mpvs.begin(); itr != mpvs.end(); ++itr, ++k1) {
      if (k1 == 0)
        startvalues[0] = (itr->second).first;
      if (k1 >= first && k < np) {
        mpv[k] = (itr->second).first;
        dmpv[k] = (itr->second).second;
        if (mpvs.size() == 6) {
          lums[k] = lumi[p1[k1]];
        } else {
          lums[k] = lumi[k1];
        }
        dlum[k] = 0;
        if (k == 0) {
          ymin = mpv[k] - dmpv[k];
          ymax = mpv[k] + dmpv[k];
        } else {
          if (mpv[k] - dmpv[k] < ymin)
            ymin = mpv[k] - dmpv[k];
          if (mpv[k] + dmpv[k] > ymax)
            ymax = mpv[k] + dmpv[k];
        }
        if (debug)
          std::cout << "[" << k << "] = " << mpv[k] << " +- " << dmpv[k] << "\n";
        ++k;
      }
    }
    np = k;
    if (debug)
      std::cout << "YRange (Initlal) : " << ymin << ":" << ymax << ":" << startvalues[0];
    if (normalize % 10 > 0) {
      for (unsigned int k = 0; k < np; ++k) {
        mpv[k] /= startvalues[0];
        dmpv[k] /= startvalues[0];
        if (debug)
          std::cout << "[" << k << "] = " << mpv[k] << " +- " << dmpv[k] << "\n";
      }
      ymin /= startvalues[0];
      ymax /= startvalues[0];
      startvalues[0] = (mpvs.size() == 6) ? 5.0 : 1.0;
      if (debug)
        std::cout << "YRange (Initlal) : " << ymin << ":" << ymax << ":" << startvalues[0];
    }
    int imin = (int)(0.01 * ymin) - 2;
    int imax = (int)(0.01 * ymax) + 3;
    if (normalize % 10 > 0) {
      ymax = 1.2;
      ymin = 0.6;
    } else {
      ymin = 100 * imin;
      ymax = 100 * imax;
    }
    if (debug)
      std::cout << " (Final) : " << ymin << ":" << ymax << std::endl;

    TGraphErrors* g = new TGraphErrors(np, lums, mpv, dlum, dmpv);
    TF1* f = new TF1("f", "[0]*TMath::Exp(-x*[1])");
    f->SetParameters(startvalues);
    f->SetLineColor(kRed);

    double value1(0), error1(0), factor(0);
    for (int j = 0; j < 2; ++j) {
      TFitResultPtr Fit = g->Fit(f, "QS");
      value1 = Fit->Value(1);
      error1 = Fit->FitResult::Error(1);
      factor = sqrt(f->GetChisquare() / np);
      if (factor > 1.2 && ((normalize / 10) % 10 > 0)) {
        for (unsigned int k = 0; k < np; ++k)
          dmpv[k] *= factor;
        delete g;
        g = new TGraphErrors(np, lums, mpv, dlum, dmpv);
      } else {
        break;
      }
    }
    results.mean = value1;
    results.error = error1;
    results.chisq = factor * factor;
    g->SetMarkerStyle(8);
    g->SetMarkerColor(kBlue);
    std::cout << "ieta " << eta << " iphi " << phi << " depth " << depth << " mu " << value1 << " +- " << error1
              << " Chisq " << f->GetChisquare() << "/" << np << std::endl;
    g->GetXaxis()->SetRangeUser(xmin, xmax);
    g->GetYaxis()->SetRangeUser(ymin, ymax);
    g->SetMarkerSize(1.5);
    if (normalize % 10 > 0) {
      g->GetYaxis()->SetTitle("MPV_{Charge} (L) /MPV_{Charge} (0)");
    } else {
      g->GetYaxis()->SetTitle("MPV_{Charge} (fC)");
    }
    g->GetXaxis()->SetTitle("Integrated Luminosity (fb^{-1})");
    g->GetYaxis()->SetTitleSize(0.04);
    g->GetXaxis()->SetTitleSize(0.04);
    g->GetXaxis()->SetNdivisions(20);
    g->GetXaxis()->SetLabelSize(0.04);
    g->GetYaxis()->SetLabelSize(0.04);
    g->GetXaxis()->SetTitleOffset(1.0);
    g->GetYaxis()->SetTitleOffset(1.2);
    g->SetTitle("");
    g->Draw("AP");

    char namel[100];
    if (phi > 0) {
      if (depth > 0)
        sprintf(namel, "i#eta %d, i#phi %d, depth %d", eta, phi, depth);
      else
        sprintf(namel, "i#eta %d, i#phi %d (all depths)", eta, phi);
    } else {
      if (depth > 0)
        sprintf(namel, "i#eta %d, depth %d", eta, depth);
      else
        sprintf(namel, "i#eta %d (all depths)", eta);
    }
    legend->AddEntry(g, namel, "lp");
    if (drawleg) {
      legend->Draw("same");
      results.pad->Update();

      sprintf(namel, "Slope = %6.4f #pm %6.4f", value1, error1);
      TPaveText* txt0 = new TPaveText(0.60, 0.84, 0.89, 0.89, "blNDC");
      txt0->SetFillColor(0);
      txt0->AddText(namel);
      txt0->Draw("same");
    }
    if (iyear > 1000) {
      char txt[100];
      TPaveText* txt1 = new TPaveText(0.60, 0.91, 0.90, 0.96, "blNDC");
      txt1->SetFillColor(0);
      sprintf(txt, "%d, %d TeV %5.1f fb^{-1}", iyear, ener, lumis);
      txt1->AddText(txt);
      txt1->Draw("same");
      TPaveText* txt2 = new TPaveText(0.10, 0.91, 0.18, 0.96, "blNDC");
      txt2->SetFillColor(0);
      sprintf(txt, "CMS");
      txt2->AddText(txt);
      txt2->Draw("same");
    }
    results.pad->Modified();
    results.pad->Update();
    if (save) {
      sprintf(namel, "%s_comb.pdf", results.pad->GetName());
      results.pad->Print(namel);
    }
  }
  return results;
}

rType plotMPV(const char* infile,
              int eta,
              int phi,
              int depth,
              unsigned int first = 0,
              bool drawleg = true,
              int iyear = 17,
              double lumis = 0.0,
              int ener = 13,
              int normalize = 0,
              bool save = false,
              bool debug = false) {
  char name[100];
  int iy = iyear % 100;
  const unsigned int nmax = (iy == 17) ? 13 : 20;
  double lumi = ((lumis > 0) ? lumis : ((iy == 17) ? 49.0 : 68.5));
  sprintf(name, "mpvE%dF%dD%d", eta, phi, depth);

  std::map<int, std::pair<double, double> > mpvs;
  readMPVs(infile, eta, phi, depth, nmax, mpvs, debug);
  rType results = drawMPV(name, eta, phi, depth, nmax, mpvs, first, drawleg, iyear, lumi, ener, normalize, save, debug);
  return results;
}

rType plot2MPV(const char* infile,
               int eta,
               int depth,
               unsigned int first = 0,
               bool drawleg = true,
               int iyear = 17,
               double lumis = 0.0,
               int ener = 13,
               int normalize = 0,
               bool save = false,
               bool debug = false) {
  char name[100];
  sprintf(name, "mpvE%dD%d", eta, depth);

  int iy = iyear % 100;
  const unsigned int nmax = (iy == 17) ? 13 : 20;
  double lumi = ((lumis > 0) ? lumis : ((iy == 17) ? 49.0 : 68.5));
  std::map<int, std::pair<double, double> > mpvs1, mpvs2, mpvs;
  readMPVs(infile, eta, 63, depth, nmax, mpvs1, debug);
  readMPVs(infile, eta, 65, depth, nmax, mpvs2, debug);
  if (mpvs1.size() == mpvs2.size()) {
    unsigned int k(0);
    for (std::map<int, std::pair<double, double> >::const_iterator itr = mpvs1.begin(); itr != mpvs1.end();
         ++itr, ++k) {
      int period = (itr->first);
      double mpv1 = (itr->second).first;
      double empv1 = (itr->second).second;
      double wt1 = 1.0 / (empv1 * empv1);
      double mpv2 = mpvs2[period].first;
      double empv2 = mpvs2[period].second;
      double wt2 = 1.0 / (empv2 * empv2);
      double mpv = (mpv1 * wt1 + mpv2 * wt2) / (wt1 + wt2);
      double empv = 1.0 / sqrt(wt1 + wt2);
      if (debug)
        std::cout << "Period " << period << " MPV1 " << mpv1 << " +- " << empv1 << " MPV2 " << mpv2 << " +- " << empv2
                  << " --> MPV = " << mpv << " +- " << empv << std::endl;
      mpvs[period] = std::pair<double, double>(mpv, empv);
    }
  }
  rType results = drawMPV(name, eta, 0, depth, nmax, mpvs, first, drawleg, iyear, lumi, ener, normalize, save, debug);
  return results;
}

void plotMPVs(const char* infile,
              int phi,
              int nvx = 3,
              int normalize = 0,
              unsigned int first = 0,
              int iyear = 17,
              double lumis = 0.0,
              int ener = 13,
              bool save = false,
              bool debug = false) {
  int ieta[20] = {-26, -25, -24, -23, -22, -21, -20, -19, -18, -17, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
  int ndep[20] = {6, 6, 6, 6, 6, 6, 6, 6, 5, 3, 3, 5, 6, 6, 6, 6, 6, 6, 6, 6};
  int idep[20] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1};

  char name[100];
  int iy = iyear % 100;
  sprintf(name, "Data%d%d_trunc.txt", phi, iy);
  std::ofstream log(name, std::ios_base::app | std::ios_base::out);
  int k0 = (iy == 17) ? 10 : 0;
  const unsigned int nmax = (iy == 17) ? 13 : 20;
  double lumi = ((lumis > 0) ? lumis : ((iy == 17) ? 49.0 : 68.5));

  for (int k = k0; k < 20; ++k) {
    int eta = ieta[k];
    for (int depth = idep[k]; depth <= ndep[k]; ++depth) {
      sprintf(name, "mpvE%dF%dD%d", eta, phi, depth);
      std::map<int, std::pair<double, double> > mpvs;
      readMPVs(infile, eta, phi, depth, nmax, mpvs, debug);
      rType rt = drawMPV(name, eta, phi, depth, nmax, mpvs, first, true, iyear, lumi, ener, normalize, save, debug);
      if (rt.pad != nullptr) {
        char line[100];
        sprintf(line, "%3d %2d %d %d   %8.4f  %8.4f  %8.4f", eta, phi, depth, nvx, rt.mean, rt.error, rt.chisq);
        std::cout << line << std::endl;
        log << eta << '\t' << phi << '\t' << depth << '\t' << nvx << '\t' << rt.mean << '\t' << rt.error << '\t'
            << rt.chisq << std::endl;
      }
    }
  }
}

rType plotHist(
    const char* infile, double frac, int eta = 25, int depth = 1, int phi = 0, int nvx = 0, bool debug = true) {
  TFile* file = new TFile(infile);
  rType rt;
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(111110);
  if (file != nullptr) {
    char name[100];
    int fi = (phi == 0) ? 1 : phi;
    sprintf(name, "ChrgE%dF%dD%dV%dP0", eta, fi, depth - 1, nvx);
    TH1D* hist = (TH1D*)file->FindObjectAny(name);
    if (hist != nullptr) {
      sprintf(name, "c_E%dF%dD%dV%d", eta, fi, depth - 1, nvx);
      rt.pad = new TCanvas(name, name, 500, 500);
      rt.pad->SetRightMargin(0.10);
      rt.pad->SetTopMargin(0.10);
      hist->GetXaxis()->SetTitleSize(0.04);
      hist->GetXaxis()->SetTitle("Charge (fC)");
      hist->GetYaxis()->SetTitle("Tracks");
      hist->GetYaxis()->SetLabelOffset(0.005);
      hist->GetYaxis()->SetTitleSize(0.04);
      hist->GetYaxis()->SetLabelSize(0.035);
      hist->GetYaxis()->SetTitleOffset(1.10);
      hist->SetMarkerStyle(20);
      hist->SetMarkerColor(2);
      hist->SetLineColor(2);
      hist->Draw();
      rt.pad->Update();
      TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
      if (st1 != nullptr) {
        st1->SetY1NDC(0.75);
        st1->SetY2NDC(0.90);
        st1->SetX1NDC(0.65);
        st1->SetX2NDC(0.90);
      }
      TPaveText* txt1 = new TPaveText(0.60, 0.91, 0.90, 0.96, "blNDC");
      txt1->SetFillColor(0);
      sprintf(name, "#eta = %d, #phi = %d, depth = %d nvx = %d", eta, phi, depth, nvx);
      txt1->AddText(name);
      txt1->Draw("same");
      rt.pad->Modified();
      rt.pad->Update();
      int nbin = hist->GetNbinsX();
      int bins(0);
      double total(0), mom1(0), mom2(0);
      for (int k = 1; k < nbin; ++k) {
        double xx = hist->GetBinLowEdge(k) + 0.5 * hist->GetBinWidth(k);
        double en = hist->GetBinContent(k);
        total += en;
        mom1 += xx * en;
        mom2 += (xx * xx * en);
      }
      mom1 /= total;
      mom2 /= total;
      double err1 = sqrt((mom2 - mom1 * mom1) / total);
      double total0(0), mom10(0), mom20(0);
      for (int k = 1; k < nbin; ++k) {
        double xx = hist->GetBinLowEdge(k) + 0.5 * hist->GetBinWidth(k);
        double en = hist->GetBinContent(k);
        if (total0 < frac * total) {
          bins = k;
          total0 += en;
          mom10 += xx * en;
          mom20 += (xx * xx * en);
        }
      }
      mom10 /= total0;
      mom20 /= total0;
      rt.bin = bins;
      rt.mean = mom10;
      rt.error = sqrt((mom20 - mom10 * mom10) / total0);
      if (debug)
        std::cout << "Eta " << eta << " phi " << phi << " depth " << depth << " nvx " << nvx << " Mean = " << mom1
                  << " +- " << err1 << " " << frac * 100 << "% Truncated Mean = " << rt.mean << " +- " << rt.error
                  << "\n";
      sprintf(name, "%4.2f truncated mean = %8.2f +- %6.2f", frac, rt.mean, rt.error);
      TPaveText* txt2 = new TPaveText(0.10, 0.91, 0.55, 0.96, "blNDC");
      txt2->SetFillColor(0);
      txt2->AddText(name);
      txt2->Draw("same");
      TArrow* arrow = new TArrow(rt.bin, 0.032, rt.bin, 6.282, 0.05, "<");
      arrow->SetFillColor(kBlack);
      arrow->SetFillStyle(1001);
      arrow->SetLineColor(kBlack);
      arrow->SetLineStyle(2);
      arrow->SetLineWidth(2);
      arrow->Draw();
      rt.pad->Modified();
      rt.pad->Update();
    }
  }
  return rt;
}

void getTruncateMean(const char* infile, double frac, std::string type, int nvx = 2, bool save = false) {
  int ieta[20] = {-26, -25, -24, -23, -22, -21, -20, -19, -18, -17, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
  int ndep[20] = {6, 6, 6, 6, 6, 6, 6, 6, 5, 3, 3, 5, 6, 6, 6, 6, 6, 6, 6, 6};
  int idep[20] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1};

  for (int k = 0; k < 20; ++k) {
    for (int depth = idep[k]; depth <= ndep[k]; ++depth) {
      rType rt = plotHist(infile, frac, ieta[k], depth, 0, nvx, false);
      if (rt.pad != nullptr) {
        char line[100];
        sprintf(
            line, "%s %2d   0 %d %d   %8.4f  %8.4f  %d", type.c_str(), ieta[k], depth, nvx, rt.mean, rt.error, rt.bin);
        std::cout << line << std::endl;
        if (save) {
          char name[100];
          sprintf(name, "%s_comb.pdf", rt.pad->GetName());
          rt.pad->Print(name);
        }
      }
    }
  }
}

void getTruncatedMeanX(
    const char* infile, std::string type, int nvx = 2, int year = 18, int phi = 0, bool save = false) {
  int ieta[20] = {-26, -25, -24, -23, -22, -21, -20, -19, -18, -17, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
  int ndep[20] = {6, 6, 6, 6, 6, 6, 6, 6, 5, 3, 3, 5, 6, 6, 6, 6, 6, 6, 6, 6};
  int idep[20] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1};
  double frac18[120] = {0.40, 0.55, 0.50, 0.55, 0.60, 0.60, 0.40, 0.55, 0.60, 0.60, 0.60, 0.60, 0.40, 0.55, 0.60,
                        0.60, 0.60, 0.60, 0.45, 0.55, 0.60, 0.60, 0.60, 0.60, 0.55, 0.60, 0.60, 0.60, 0.60, 0.60,
                        0.55, 0.60, 0.60, 0.60, 0.60, 0.65, 0.60, 0.60, 0.60, 0.60, 0.60, 0.65, 0.60, 0.60, 0.60,
                        0.60, 0.60, 0.65, 0.60, 0.60, 0.60, 0.60, 0.60, 0.65, 0.60, 0.60, 0.60, 0.60, 0.60, 0.65,
                        0.60, 0.60, 0.60, 0.60, 0.60, 0.65, 0.60, 0.60, 0.60, 0.60, 0.65, 0.65, 0.60, 0.60, 0.60,
                        0.60, 0.65, 0.65, 0.60, 0.60, 0.60, 0.60, 0.60, 0.65, 0.55, 0.60, 0.60, 0.60, 0.60, 0.65,
                        0.55, 0.60, 0.60, 0.60, 0.60, 0.60, 0.45, 0.55, 0.60, 0.60, 0.60, 0.60, 0.40, 0.55, 0.60,
                        0.60, 0.60, 0.60, 0.40, 0.55, 0.60, 0.60, 0.60, 0.60, 0.40, 0.55, 0.50, 0.55, 0.60, 0.60};
  double frac17[120] = {0.70, 0.65, 0.55, 0.60, 0.60, 0.65, 0.67, 0.55, 0.55, 0.57, 0.55, 0.62, 0.65, 0.57, 0.60,
                        0.55, 0.70, 0.65, 0.65, 0.60, 0.55, 0.65, 0.75, 0.60, 0.60, 0.60, 0.65, 0.62, 0.75, 0.65,
                        0.70, 0.67, 0.60, 0.65, 0.55, 0.65, 0.60, 0.55, 0.75, 0.65, 0.75, 0.70, 0.60, 0.55, 0.75,
                        0.67, 0.65, 0.60, 0.60, 0.45, 0.50, 0.52, 0.70, 0.65, 0.60, 0.60, 0.60, 0.60, 0.60, 0.65,
                        0.60, 0.60, 0.60, 0.60, 0.60, 0.65, 0.60, 0.45, 0.50, 0.52, 0.70, 0.65, 0.60, 0.55, 0.75,
                        0.67, 0.65, 0.60, 0.60, 0.55, 0.75, 0.65, 0.75, 0.70, 0.70, 0.67, 0.60, 0.65, 0.55, 0.65,
                        0.60, 0.60, 0.65, 0.62, 0.75, 0.65, 0.65, 0.60, 0.55, 0.65, 0.75, 0.60, 0.65, 0.57, 0.60,
                        0.55, 0.70, 0.65, 0.67, 0.55, 0.55, 0.57, 0.55, 0.62, 0.70, 0.65, 0.55, 0.60, 0.60, 0.65};
  double frac63[120] = {0.63, 0.60, 0.57, 0.56, 0.60, 0.61, 0.60, 0.60, 0.60, 0.60, 0.60, 0.61, 0.58, 0.60, 0.60,
                        0.60, 0.62, 0.63, 0.57, 0.60, 0.60, 0.60, 0.65, 0.61, 0.57, 0.60, 0.60, 0.60, 0.62, 0.64,
                        0.60, 0.60, 0.60, 0.60, 0.63, 0.65, 0.65, 0.60, 0.63, 0.60, 0.64, 0.65, 0.60, 0.60, 0.63,
                        0.62, 0.62, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60,
                        0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.63,
                        0.62, 0.62, 0.65, 0.65, 0.60, 0.63, 0.60, 0.64, 0.65, 0.60, 0.60, 0.60, 0.60, 0.63, 0.65,
                        0.57, 0.60, 0.60, 0.60, 0.62, 0.64, 0.57, 0.60, 0.60, 0.60, 0.65, 0.61, 0.58, 0.60, 0.60,
                        0.60, 0.62, 0.63, 0.60, 0.60, 0.60, 0.60, 0.60, 0.61, 0.63, 0.60, 0.57, 0.56, 0.60, 0.61};

  char name[100];
  sprintf(name, "Data%d_trunc.txt", year);
  std::ofstream log(name, std::ios_base::app | std::ios_base::out);
  int k0 = (year == 17) ? 10 : 0;
  for (int k = k0; k < 20; ++k) {
    for (int depth = idep[k]; depth <= ndep[k]; ++depth) {
      int indx = k * 6 + depth - 1;
      double frac = ((year == 17) ? ((phi == 0) ? frac17[indx] : frac63[indx]) : frac18[indx]);
      rType rt = plotHist(infile, frac, ieta[k], depth, phi, nvx, false);
      if (rt.pad != nullptr) {
        char line[100];
        sprintf(line,
                "%s %2d   %d %d %d   %8.4f  %8.4f  %d",
                type.c_str(),
                ieta[k],
                phi,
                depth,
                nvx,
                rt.mean,
                rt.error,
                rt.bin);
        std::cout << line << std::endl;
        log << type << '\t' << ieta[k] << '\t' << phi << '\t' << depth << '\t' << nvx << '\t' << rt.mean << '\t'
            << rt.error << std::endl;
        if (save) {
          sprintf(name, "%s_comb.pdf", rt.pad->GetName());
          rt.pad->Print(name);
        }
      }
    }
  }
}

std::map<int, std::pair<double, double> > readSlopes(
    const char* infile, const int typ, const int phi, const int dep, bool debug) {
  std::map<int, std::pair<double, double> > slopes;
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer[1024];
    unsigned int all(0), good(0), select(0);
    while (fInput.getline(buffer, 1024)) {
      ++all;
      if (buffer[0] == '#')
        continue;  //ignore comment
      std::vector<std::string> items = splitString(std::string(buffer));
      if (items.size() < 6) {
        if (debug)
          std::cout << "Ignore  line: " << buffer << std::endl;
      } else {
        ++good;
        int type = std::atoi(items[0].c_str());
        int ieta = std::atoi(items[1].c_str());
        int iphi = std::atoi(items[2].c_str());
        int depth = std::atoi(items[3].c_str());
        double val = std::atof(items[4].c_str());
        double err = std::atof(items[5].c_str());
        if ((type == typ) && (iphi == phi) && (depth == dep)) {
          ++select;
          slopes[ieta] = std::pair<double, double>(val, err);
        }
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good and " << select << " selected records from "
              << infile << std::endl;
  }
  if (debug) {
    unsigned int k(0);
    for (std::map<int, std::pair<double, double> >::const_iterator itr = slopes.begin(); itr != slopes.end();
         ++itr, ++k) {
      std::cout << "[" << k << "] ieta " << itr->first << " Slope " << (itr->second).first << " +- "
                << (itr->second).second << std::endl;
    }
  }
  return slopes;
}

void drawSlope(const char* infile,
               int type,
               int phi,
               int depth,
               bool drawleg = true,
               int iyear = 2018,
               double lumis = 0,
               int ener = 13,
               bool save = false,
               bool debug = false) {
  std::map<int, std::pair<double, double> > slopes = readSlopes(infile, type, phi, depth, debug);
  if (slopes.size() > 0) {
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);
    gStyle->SetFillColor(kWhite);
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    char name[100], cname[100];
    std::string method[2] = {"Fit", "TM"};
    std::string methods[2] = {"Landau Fit", "Truncated Mean"};
    int iy = iyear % 100;
    sprintf(cname, "%s%d", method[type].c_str(), iy);
    TCanvas* pad = new TCanvas(cname, cname);
    pad->SetLeftMargin(0.12);
    pad->SetRightMargin(0.10);
    pad->SetTopMargin(0.10);
    TLegend* legend = new TLegend(0.2206304, 0.8259023, 0.760745, 0.8768577, NULL, "brNDC");
    //  TLegend *legend = new TLegend(0.25, 0.14, 0.79, 0.19);
    legend->SetFillColor(kWhite);
    legend->SetBorderSize(1);
    unsigned int nmax = slopes.size();
    int np(0);
    double xv[nmax], dxv[nmax], yv[nmax], dyv[nmax];
    for (std::map<int, std::pair<double, double> >::const_iterator itr = slopes.begin(); itr != slopes.end(); ++itr) {
      xv[np] = itr->first;
      dxv[np] = 0;
      yv[np] = (itr->second).first;
      dyv[np] = (itr->second).second;
      ++np;
    }
    TGraphErrors* g = new TGraphErrors(np, xv, yv, dxv, dyv);
    g->SetMarkerStyle(8);
    g->SetMarkerColor(kBlue);
    g->GetXaxis()->SetRangeUser(-30.0, 30.0);
    g->GetYaxis()->SetRangeUser(-0.0005, 0.0075);
    g->SetMarkerSize(1.5);
    g->GetYaxis()->SetTitle("Slope");
    g->GetXaxis()->SetTitle("i#eta");
    g->GetYaxis()->SetTitleSize(0.04);
    g->GetXaxis()->SetTitleSize(0.04);
    g->GetXaxis()->SetLabelSize(0.04);
    g->GetYaxis()->SetLabelSize(0.04);
    g->GetXaxis()->SetTitleOffset(1.0);
    g->GetYaxis()->SetTitleOffset(1.5);
    g->SetTitle("");
    g->Draw("AP");

    if (phi > 0) {
      sprintf(name, "%s Method (i#phi %d, depth %d)", methods[type].c_str(), phi, depth);
    } else {
      if (iy == 17)
        sprintf(name, "%s Method (HEP17 depth %d)", methods[type].c_str(), depth);
      else
        sprintf(name, "%s Method (All #phi's, depth %d)", methods[type].c_str(), depth);
    }
    legend->AddEntry(g, name, "lp");
    if (drawleg) {
      legend->Draw("same");
      pad->Update();
    }
    if (iyear > 1000) {
      char txt[100];
      TPaveText* txt1 = new TPaveText(0.60, 0.91, 0.90, 0.96, "blNDC");
      txt1->SetFillColor(0);
      sprintf(txt, "%d, %d TeV %5.1f fb^{-1}", iyear, ener, lumis);
      txt1->AddText(txt);
      txt1->Draw("same");
      TPaveText* txt2 = new TPaveText(0.10, 0.91, 0.18, 0.96, "blNDC");
      txt2->SetFillColor(0);
      sprintf(txt, "CMS");
      txt2->AddText(txt);
      txt2->Draw("same");
    }
    pad->Modified();
    pad->Update();
    if (save) {
      sprintf(name, "%s_depth%d.pdf", pad->GetName(), depth);
      pad->Print(name);
    }
  }
}

TCanvas* plotLightLoss(std::string infile = "mu_HE_insitu_2018.txt",
                       double lumis = 66.5,
                       int iyear = 2018,
                       int ener = 13,
                       bool save = false,
                       bool debug = true) {
  std::map<std::pair<int, int>, double> losses;
  std::map<std::pair<int, int>, double>::const_iterator itr;
  std::ifstream fInput(infile.c_str());
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer[1024];
    unsigned int all(0), good(0);
    while (fInput.getline(buffer, 1024)) {
      ++all;
      if (buffer[0] == '#')
        continue;  //ignore comment
      std::vector<std::string> items = splitString(std::string(buffer));
      if (items.size() < 5) {
        if (debug)
          std::cout << "Ignore  line: " << buffer << std::endl;
      } else {
        ++good;
        int ieta = std::atoi(items[2].c_str());
        int depth = std::atoi(items[1].c_str());
        double val = std::atof(items[3].c_str());
        double loss = exp(-val * lumis);
        losses[std::pair<int, int>(ieta, depth)] = loss;
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good records "
              << "from " << infile << std::endl;
    if (debug) {
      unsigned int k(0);
      for (itr = losses.begin(); itr != losses.end(); ++itr, ++k)
        std::cout << "[" << k << "] eta|depth " << (itr->first).first << "|" << (itr->first).second << " Loss "
                  << (itr->second) << std::endl;
    }
  }
  TCanvas* pad;
  if (losses.size() > 0) {
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);
    gStyle->SetFillColor(kWhite);
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    gStyle->SetPalette(1);
    char text[100], cname[100];
    int iy = iyear % 100;
    sprintf(cname, "light%d", iy);
    pad = new TCanvas(cname, cname, 600, 700);
    pad->SetLeftMargin(0.10);
    pad->SetRightMargin(0.14);
    pad->SetTopMargin(0.10);
    const int neta = 15;
    TH2D* h = new TH2D("cname", "cname", 8, 0, 8, neta, 0, neta);
    TGraph2D* dt = new TGraph2D();
    dt->SetNpx(8);
    dt->SetNpy(15);
    h->GetXaxis()->SetTitle("Detector depth number");
    h->GetYaxis()->SetTitle("Tower");
    h->GetZaxis()->SetRangeUser(0.75, 1.25);
    std::map<std::pair<int, int>, double>::const_iterator itr;

    unsigned int nmax = losses.size();
    int np(0);
    int ieta, depth;
    for (itr = losses.begin(); itr != losses.end(); ++itr) {
      ieta = (itr->first).first;
      depth = (itr->first).second;
      h->Fill((depth + 0.5), (30.5 - ieta), (itr->second));
    }
    h->Draw("colz");

    gPad->Update();
    TAxis* a = h->GetYaxis();
    a->SetNdivisions(20);
    std::string val[neta] = {"30", "29", "28", "27", "26", "25", "24", "23", "22", "21", "20", "19", "18", "17", "16"};
    for (int i = 0; i < neta; i++)
      a->ChangeLabel((i + 1), -1, -1, -1, -1, -1, val[i]);
    a->CenterLabels(kTRUE);
    gPad->Modified();
    gPad->Update();

    TPaveText* txt1 = new TPaveText(0.60, 0.91, 0.90, 0.97, "blNDC");
    txt1->SetFillColor(0);
    sprintf(text, "%d, %d TeV %5.1f fb^{-1}", iyear, ener, lumis);
    txt1->AddText(text);
    txt1->Draw("same");
    TPaveText* txt2 = new TPaveText(0.11, 0.91, 0.18, 0.97, "blNDC");
    txt2->SetFillColor(0);
    sprintf(text, "CMS");
    txt2->AddText(text);
    txt2->Draw("same");
    pad->Modified();
    pad->Update();
    if (save) {
      sprintf(cname, "%s.pdf", pad->GetName());
      pad->Print(cname);
    }
  }
  return pad;
}
