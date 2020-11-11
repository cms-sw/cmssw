//////////////////////////////////////////////////////////////////////////////
// Usage:
// .L CalibFitPlots.C+g
//             For standard set of histograms from CalibMonitor
//  FitHistStandard(infile, outfile, prefix, mode, type, append, saveAll,
//                  debug);
//      Defaults: mode=11111, type=0, append=true, saveAll=false, debug=false
//
//             For extended set of histograms from CalibMonitor
//  FitHistExtended(infile, outfile, prefix, numb, type, append, fiteta, iname,
//                  debug);
//      Defaults: numb=50, type=3, append=true, fiteta=true, iname=3,
//                debug=false
//
//             For RBX dependence in sets of histograms from CalibMonitor
//  FitHistRBX(infile, outfile, prefix, append, iname);
//      Defaults: append=true, iname=2
//
//             For plotting stored histograms from FitHist's
//  PlotHist(infile, prefix, text, modePlot, kopt, lumi, ener, dataMC,
//           drawStatBox, save);
//      Defaults: modePlot=4, kopt=100, lumi=0, ener=13, dataMC=false,
//                drawStatBox=true, save=false
//
//             For plotting several histograms in the same plot
//             (fits to different data sets for example)
//  PlotHists(infile, prefix, text, drawStatBox, save)
//      Defaults: drawStatBox=true; save=false;
//      Note prefix is common part for all histograms
//
//             For plotting on the same canvas plots with different
//             prefixes residing in the same file with approrprate text
//   PlotTwoHists(infile, prefix1, text1, prefix2, text2, text0, type, iname,
//                lumi, ener, drawStatBox, save);
//      Defaults: type=0; iname=2; lumi=0; ener=13; drawStatBox=true;
//                save=false;
//      Note prefixN, textN have the same meaning as prefix and text for set N
//           text0 is the text for general title added within ()
//           type=0 plots response distributions and MPV of response vs ieta
//               =1 plots MPV of response vs RBX #
//
//             For plotting stored histograms from CalibTree
//   PlotFiveHists(infile, text0, prefix0, type, iname, drawStatBox, normalize,
//                 save, prefix1, text1, prefix2, text2, prefix3, text3,
//                 prefix4, text4, prefix5, text5);
//      Defaults: type=0; iname=0; drawStatBox=true; normalize=false;
//                save=false; prefixN=""; textN=""; (for N > 0)
//      Note prefixN, textN have the same meaning as prefix and text for set N
//           text0 is the text for general title added within ()
//           prefix0 is the tag attached to the canvas name
//           type has the same meaning as in PlotTwoHists
//
//  PlotHistCorrResults(infile, text, prefixF, save);
//      Defaults: save=false
//
//             For plotting correction factors
//  PlotHistCorrFactor(infile, text, prefixF, scale, nmin, dataMC,
//                    drawStatBox, save);
//      Defaults: dataMC=true, drwaStatBox=false, nmin=100, save=false
//
//             For plotting (fractional) asymmetry in the correction factors
//
//  PlotHistCorrAsymmetry(infile, text, prefixF, save);
//      Defaults: prefixF="", save=false
//
//             For plotting correction factors from upto 5 different runs
//             on the same canvas
//
//  PlotHistCorrFactors(infile1, text1, infile2, text2, infile3, text3,
//                      infile4, text4, infile5, text5, prefixF, ratio,
//                      drawStatBox, nmin, dataMC, year, save)
//      Defaults: ratio=false, drawStatBox=true, nmin=100, dataMC=false,
//                year=2018, save=false
//
//             For plotting correction factors including systematics
//  PlotHistCorrSys(infilec, conds, text, save)
//      Defaults: save=false
//
//             For plotting uncertainties in correction factors with decreasing
//             integrated luminpsoties starting from *lumi*
//  PlotHistCorrLumis(infilec, conds, lumi, save)
//      Defaults: save=false
//
//             For plotting correlation of correction factors
//  PlotHistCorrRel(infile1, infile2, text1, text2, save)
//      Defaults: save=false
//
//             For plotting four histograms
//  PlotFourHists(infile, prefix0, type, drawStatBox, normalize, save, prefix1,
//                text1, prefix2, text2, prefix3, text3, prefix4, text4)
//      Defaults: type=0, drawStatBox=0, normalize=false, save=false,
//                prefixN="", textN=""
//
//            For plotting PU corrected histograms (o/p of CalibPlotCombine)
//  PlotPUCorrHists(infile, prefix drawStatBox, approve, save)
//      Defaults: infile = "corrfac.root", prefix = "", drawStatBox = 0,
//                approve = true, save = false
//
//             For plotting histograms obtained from fits to PU correction
//             (o/p of CalibFitPU) for a given ieta using 2D/profile/Graphs
//  PlotHistCorr(infile, prefix, text, eta, mode, drawStatBox, save)
//      Defaults eta = 0 (all ieta values), mode = 1 (profile histograms),
//               drawStatBox = true, save = false
//
//             For plotting histograms created by CalibPlotProperties
//  PlotPropertyHist(infile, prefix, text, etaMax, lumi, ener, dataMC,
//		     drawStatBox, save)
//      Defaults etaMax = 25 (draws for eta = 1 .. etaMax), lumi = 0,
//               ener = 13.0, dataMC = false,  drawStatBox = true, save = false
//
//  where:
//  infile   (std::string)  = Name of the input ROOT file
//  outfile  (std::string)  = Name of the output ROOT file
//  prefix   (std::string)  = Prefix for the histogram names
//  mode     (int)          = Flag to check which set of histograms to be
//                            done. It has the format lthdo where each of
//                            l, t,h,d,o can have a value 0 or 1 to select
//                            or deselect. l,t,h,d,o for momentum range
//                            60-100, 30-40, all, 20-30, 40-60 Gev (11111)
//  type     (int)          = defines eta binning type (see CalibMonitor)
//  append   (bool)         = Open the output in Update/Recreate mode (True)
//  fiteta   (bool)         = fit the eta dependence with pol0
//  iname    (int)          = choose the momentum bin (3: 40-60 GeV)
//  saveAll  (bool)         = Flag to save intermediate plots (False)
//  numb     (int)          = Number of eta bins (42 for -21:21)
//  text     (std::string)  = Extra text to be put in the text title
//  modePlot (int)          = Flag to plot E/p distribution (0);
//                            <E/p> as a function of ieta (1);
//                            <E/p> as a function of distance from L1 (2);
//                            <E/p> as a function of number of vertex (3);
//                            E/p for barrel, endcap and transition (4)
//  kopt     (int)          = Option in format "hdo" where each of d, o can
//                            have a value of 0 or 1 to select or deselect.
//                            o>0 to carry out pol0 fit, o>1 to restrict
//                            fit region between -20 & 20; d=1 to show grid;
//                            h=0,1 to show plots with 2- or 1-Gaussian fit
//  lumi     (double)       = Integrated luminosity of the dataset used which
//                            needs to be drawn on the top of the canvas
//                            along with CM energy (if lumi > 0)
//  ener     (double)       = CM energy of the dataset used
//  save     (bool)         = if true it saves the canvas as a pdf file
//  normalize(bool)         = if the histograms to be scaled to get
//                            normalization to 1
//  prefixF  (string)       = string to be included to the pad name
//  infileN  (char*)        = name of the correction file and the corresponding
//  textN    (string)         texts (if blank ignored)
//  nmin     (int)          = minimum # of #ieta points needed to show the
//                            fitted line
//  scale    (double)       = constant scale factor applied to the factors
//  ratio    (bool)         = set to show the ratio plot (false)
//  drawStatBox (bool)      = set to show the statistical box (true)
//  year     (int)          = Year of data taking (applicable to Data)
//  infilc   (string)       = prefix of the file names of correction factors
//                            (assumes file name would be the prefix followed
//                            by _condX.txt where X=0 for the default version
//                            and 1..conds for the variations)
//  conds    (int)          = number of variations in estimating systematic
//                            checks
//////////////////////////////////////////////////////////////////////////////

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH1D.h>
#include <TMath.h>
#include <TProfile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include "CalibCorr.C"

const double fitrangeFactor = 1.5;

struct cfactors {
  int ieta, depth;
  double corrf, dcorr;
  cfactors(int ie = 0, int dp = 0, double cor = 1, double dc = 0) : ieta(ie), depth(dp), corrf(cor), dcorr(dc){};
};

struct results {
  double mean, errmean, width, errwidth;
  results(double v1 = 0, double er1 = 0, double v2 = 0, double er2 = 0)
      : mean(v1), errmean(er1), width(v2), errwidth(er2){};
};

std::pair<double, double> GetMean(TH1D* hist, double xmin, double xmax, double& rms) {
  double mean(0), err(0), wt(0);
  rms = 0;
  for (int i = 1; i <= hist->GetNbinsX(); ++i) {
    double xlow = hist->GetBinLowEdge(i);
    double xhigh = xlow + hist->GetBinWidth(i);
    if ((xlow >= xmin) && (xhigh <= xmax)) {
      double cont = hist->GetBinContent(i);
      double valu = 0.5 * (xlow + xhigh);
      wt += cont;
      mean += (valu * cont);
      rms += (valu * valu * cont);
    }
  }
  if (wt > 0) {
    mean /= wt;
    rms /= wt;
    err = std::sqrt((rms - mean * mean) / wt);
  }
  return std::pair<double, double>(mean, err);
}

std::pair<double, double> GetWidth(TH1D* hist, double xmin, double xmax) {
  double mean(0), mom2(0), rms(0), err(0), wt(0);
  for (int i = 1; i <= hist->GetNbinsX(); ++i) {
    double xlow = hist->GetBinLowEdge(i);
    double xhigh = xlow + hist->GetBinWidth(i);
    if ((xlow >= xmin) && (xhigh <= xmax)) {
      double cont = hist->GetBinContent(i);
      double valu = 0.5 * (xlow + xhigh);
      wt += cont;
      mean += (valu * cont);
      mom2 += (valu * valu * cont);
    }
  }
  if (wt > 0) {
    mean /= wt;
    mom2 /= wt;
    rms = std::sqrt(mom2 - mean * mean);
    err = rms / std::sqrt(2 * wt);
  }
  return std::pair<double, double>(rms, err);
}

Double_t langaufun(Double_t* x, Double_t* par) {
  //Fit parameters:
  //par[0]=Most Probable (MP, location) parameter of Landau density
  //par[1]=Total area (integral -inf to inf, normalization constant)
  //par[2]=Width (sigma) of convoluted Gaussian function
  //
  //In the Landau distribution (represented by the CERNLIB approximation),
  //the maximum is located at x=-0.22278298 with the location parameter=0.
  //This shift is corrected within this function, so that the actual
  //maximum is identical to the MP parameter.

  // Numeric constants
  Double_t invsq2pi = 0.3989422804014;  // (2 pi)^(-1/2)
  Double_t mpshift = -0.22278298;       // Landau maximum location

  // Control constants
  Double_t np = 100.0;  // number of convolution steps
  Double_t sc = 5.0;    // convolution extends to +-sc Gaussian sigmas

  // Variables
  Double_t xx;
  Double_t mpc;
  Double_t fland;
  Double_t sum = 0.0;
  Double_t xlow, xupp;
  Double_t step;
  Double_t i;
  Double_t par0 = 0.2;

  // MP shift correction
  mpc = par[0] - mpshift * par0 * par[0];

  // Range of convolution integral
  xlow = x[0] - sc * par[2];
  xupp = x[0] + sc * par[2];

  step = (xupp - xlow) / np;

  // Convolution integral of Landau and Gaussian by sum
  for (i = 1.0; i <= np / 2; i++) {
    xx = xlow + (i - .5) * step;
    fland = TMath::Landau(xx, mpc, par0 * par[0], kTRUE);  // / par[0];
    sum += fland * TMath::Gaus(x[0], xx, par[2]);

    xx = xupp - (i - .5) * step;
    fland = TMath::Landau(xx, mpc, par0 * par[0], kTRUE);  // / par[0];
    sum += fland * TMath::Gaus(x[0], xx, par[2]);
  }

  return (par[1] * step * sum * invsq2pi / par[2]);
}

Double_t doubleGauss(Double_t* x, Double_t* par) {
  double x1 = x[0] - par[1];
  double sig1 = par[2];
  double x2 = x[0] - par[4];
  double sig2 = par[5];
  double yval =
      (par[0] * std::exp(-0.5 * (x1 / sig1) * (x1 / sig1)) + par[3] * std::exp(-0.5 * (x2 / sig2) * (x2 / sig2)));
  return yval;
}

TFitResultPtr functionFit(TH1D* hist, double* fitrange, double* startvalues, double* parlimitslo, double* parlimitshi) {
  char FunName[100];
  sprintf(FunName, "Fitfcn_%s", hist->GetName());
  TF1* ffitold = (TF1*)gROOT->GetListOfFunctions()->FindObject(FunName);
  if (ffitold)
    delete ffitold;

  int npar(6);
  TF1* ffit = new TF1(FunName, doubleGauss, fitrange[0], fitrange[1], npar);
  ffit->SetParameters(startvalues);
  ffit->SetLineColor(kBlue);
  ffit->SetParNames("Area1", "Mean1", "Width1", "Area2", "Mean2", "Width2");
  for (int i = 0; i < npar; i++)
    ffit->SetParLimits(i, parlimitslo[i], parlimitshi[i]);
  TFitResultPtr Fit = hist->Fit(FunName, "QRWLS");
  return Fit;
}

std::pair<double, double> fitLanGau(TH1D* hist, bool debug) {
  double rms;
  std::pair<double, double> mrms = GetMean(hist, 0.005, 0.25, rms);
  double mean = mrms.first;
  double LowEdge = 0.005, HighEdge = mean + 3 * rms;
  TFitResultPtr Fit1 = hist->Fit("gaus", "+0wwqRS", "", LowEdge, HighEdge);
  if (debug)
    std::cout << hist->GetName() << " 0 " << Fit1->Value(0) << " 1 " << Fit1->Value(1) << " 2 " << Fit1->Value(2)
              << std::endl;
  double startvalues[3];
  startvalues[0] = Fit1->Value(1);
  startvalues[1] = Fit1->Value(0);
  startvalues[2] = Fit1->Value(2);

  char FunName[100];
  sprintf(FunName, "Fitfcn_%s", hist->GetName());
  TF1* ffitold = (TF1*)gROOT->GetListOfFunctions()->FindObject(FunName);
  if (ffitold)
    delete ffitold;

  TF1* ffit = new TF1(FunName, langaufun, LowEdge, HighEdge, 3);
  ffit->SetParameters(startvalues);
  ffit->SetParNames("MP", "Area", "GSigma");
  TFitResultPtr Fit2 = hist->Fit(FunName, "QRWLS");
  double value = Fit2->Value(0);
  double error = Fit2->FitResult::Error(0);
  if (debug)
    std::cout << hist->GetName() << " 0 " << Fit2->Value(0) << " 1 " << Fit2->Value(1) << " 2 " << Fit2->Value(2)
              << std::endl;
  return std::pair<double, double>(value, error);
}

results fitTwoGauss(TH1D* hist, bool debug) {
  double rms;
  std::pair<double, double> mrms = GetMean(hist, 0.2, 2.0, rms);
  double mean = mrms.first;
  double LowEdge = mean - fitrangeFactor * rms;
  double HighEdge = mean + fitrangeFactor * rms;
  if (LowEdge < 0.15)
    LowEdge = 0.15;
  std::string option = (hist->GetEntries() > 100) ? "QRS" : "QRWLS";
  TF1* g1 = new TF1("g1", "gaus", LowEdge, HighEdge);
  g1->SetLineColor(kGreen);
  TFitResultPtr Fit = hist->Fit(g1, option.c_str(), "");

  if (debug) {
    for (int k = 0; k < 3; ++k)
      std::cout << "Initial Parameter[" << k << "] = " << Fit->Value(k) << " +- " << Fit->FitResult::Error(k)
                << std::endl;
  }
  double startvalues[6], fitrange[2], lowValue[6], highValue[6];
  startvalues[0] = Fit->Value(0);
  lowValue[0] = 0.5 * startvalues[0];
  highValue[0] = 2. * startvalues[0];
  startvalues[1] = Fit->Value(1);
  lowValue[1] = 0.5 * startvalues[1];
  highValue[1] = 2. * startvalues[1];
  startvalues[2] = Fit->Value(2);
  lowValue[2] = 0.5 * startvalues[2];
  highValue[2] = 2. * startvalues[2];
  startvalues[3] = 0.1 * Fit->Value(0);
  lowValue[3] = 0.0;
  highValue[3] = 10. * startvalues[3];
  startvalues[4] = Fit->Value(1);
  lowValue[4] = 0.5 * startvalues[4];
  highValue[4] = 2. * startvalues[4];
  startvalues[5] = 2.0 * Fit->Value(2);
  lowValue[5] = 0.5 * startvalues[5];
  highValue[5] = 100. * startvalues[5];
  //fitrange[0] = mean - 2.0*rms; fitrange[1] = mean + 2.0*rms;
  fitrange[0] = Fit->Value(1) - fitrangeFactor * Fit->Value(2);
  fitrange[1] = Fit->Value(1) + fitrangeFactor * Fit->Value(2);
  TFitResultPtr Fitfun = functionFit(hist, fitrange, startvalues, lowValue, highValue);
  double wt1 = (Fitfun->Value(0)) * (Fitfun->Value(2));
  double value1 = Fitfun->Value(1);
  double error1 = Fitfun->FitResult::Error(1);
  double wval1 = Fitfun->Value(2);
  double werr1 = Fitfun->FitResult::Error(2);
  double wt2 = (Fitfun->Value(3)) * (Fitfun->Value(5));
  double value2 = Fitfun->Value(4);
  double error2 = Fitfun->FitResult::Error(4);
  double wval2 = Fitfun->Value(5);
  double werr2 = Fitfun->FitResult::Error(5);
  double value = (wt1 * value1 + wt2 * value2) / (wt1 + wt2);
  double wval = (wt1 * wval1 + wt2 * wval2) / (wt1 + wt2);
  double error = (sqrt((wt1 * error1) * (wt1 * error1) + (wt2 * error2) * (wt2 * error2)) / (wt1 + wt2));
  double werror = (sqrt((wt1 * werr1) * (wt1 * werr1) + (wt2 * werr2) * (wt2 * werr2)) / (wt1 + wt2));
  std::cout << hist->GetName() << " Fit " << value << "+-" << error << " width " << wval << " +- " << werror
            << " First  " << value1 << "+-" << error1 << ":" << wval1 << "+-" << werr1 << ":" << wt1 << " Second "
            << value2 << "+-" << error2 << ":" << wval2 << "+-" << werr2 << ":" << wt2 << std::endl;
  if (debug) {
    for (int k = 0; k < 6; ++k)
      std::cout << hist->GetName() << ":Parameter[" << k << "] = " << Fitfun->Value(k) << " +- "
                << Fitfun->FitResult::Error(k) << std::endl;
  }
  return results(value, error, wval, werror);
}

results fitOneGauss(TH1D* hist, bool fitTwice, bool debug) {
  double rms;
  std::pair<double, double> mrms = GetMean(hist, 0.2, 2.0, rms);
  double mean = mrms.first;
  double LowEdge = ((mean - fitrangeFactor * rms) < 0.5) ? 0.5 : (mean - fitrangeFactor * rms);
  double HighEdge = (mean + fitrangeFactor * rms);
  if (debug)
    std::cout << hist->GetName() << " Mean " << mean << " RMS " << rms << " Range " << LowEdge << ":" << HighEdge
              << "\n";
  std::string option = (hist->GetEntries() > 100) ? "QRS" : "QRWLS";
  TF1* g1 = new TF1("g1", "gaus", LowEdge, HighEdge);
  g1->SetLineColor(kGreen);
  TFitResultPtr Fit1 = hist->Fit(g1, option.c_str(), "");
  double value = Fit1->Value(1);
  double error = Fit1->FitResult::Error(1);
  double width = Fit1->Value(2);
  double werror = Fit1->FitResult::Error(2);
  if (fitTwice) {
    LowEdge = Fit1->Value(1) - fitrangeFactor * Fit1->Value(2);
    HighEdge = Fit1->Value(1) + fitrangeFactor * Fit1->Value(2);
    if (LowEdge < 0.5)
      LowEdge = 0.5;
    if (HighEdge > 5.0)
      HighEdge = 5.0;
    if (debug)
      std::cout << " Range for second Fit " << LowEdge << ":" << HighEdge << std::endl;
    TFitResultPtr Fit = hist->Fit("gaus", option.c_str(), "", LowEdge, HighEdge);
    value = Fit->Value(1);
    error = Fit->FitResult::Error(1);
    width = Fit->Value(2);
    werror = Fit->FitResult::Error(2);
  }

  std::pair<double, double> meaner = GetMean(hist, 0.2, 2.0, rms);
  std::pair<double, double> rmserr = GetWidth(hist, 0.2, 2.0);
  if (debug) {
    std::cout << "Fit " << value << ":" << error << ":" << hist->GetMeanError() << " Mean " << meaner.first << ":"
              << meaner.second << " Width " << rmserr.first << ":" << rmserr.second;
  }
  double minvalue(0.30);
  if (value < minvalue || value > 2.0 || error > 0.5) {
    value = meaner.first;
    error = meaner.second;
    width = rmserr.first;
    werror = rmserr.second;
  }
  if (debug) {
    std::cout << " Final " << value << "+-" << error << ":" << width << "+-" << werror << std::endl;
  }
  return results(value, error, width, werror);
}

void readCorrFactors(
    char* infile, double scale, std::map<int, cfactors>& cfacs, int& etamin, int& etamax, int& maxdepth) {
  cfacs.clear();
  std::ifstream fInput(infile);
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
      if (items.size() != 5) {
        std::cout << "Ignore  line: " << buffer << std::endl;
      } else {
        ++good;
        int ieta = std::atoi(items[1].c_str());
        int depth = std::atoi(items[2].c_str());
        float corrf = std::atof(items[3].c_str());
        float dcorr = std::atof(items[4].c_str());
        cfactors cfac(ieta, depth, scale * corrf, scale * dcorr);
        int detId = std::atoi(items[0].c_str());
        cfacs[detId] = cfactors(ieta, depth, corrf, dcorr);
        if (ieta > etamax)
          etamax = ieta;
        if (ieta < etamin)
          etamin = ieta;
        if (depth > maxdepth)
          maxdepth = depth;
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good records"
              << " from " << infile << std::endl;
  }
  /*
  unsigned k(0);
  std::cout << "Eta Range " << etamin << ":" << etamax << " Max Depth "
	    << maxdepth << std::endl;
  for (std::map<int,cfactors>::const_iterator itr = cfacs.begin();
       itr != cfacs.end(); ++itr, ++k)  
    std::cout << "[" << k << "] " << std::hex << itr->first << std::dec << ": "
	      << (itr->second).ieta << " "  << (itr->second).depth << " " 
	      << (itr->second).corrf << " " << (itr->second).dcorr << std::endl;
  */
}

void FitHistStandard(std::string infile,
                     std::string outfile,
                     std::string prefix,
                     int mode = 11111,
                     int type = 0,
                     bool append = true,
                     bool saveAll = false,
                     bool debug = false) {
  int iname[6] = {0, 1, 2, 3, 4, 5};
  int checkmode[6] = {1000, 10, 10000, 1, 100000, 100};
  double xbin0[9] = {-21.0, -16.0, -12.0, -6.0, 0.0, 6.0, 12.0, 16.0, 21.0};
  double xbins[11] = {-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0};
  double vbins[6] = {0.0, 7.0, 10.0, 13.0, 16.0, 50.0};
  double dlbins[9] = {0.0, 0.10, 0.20, 0.50, 1.0, 2.0, 2.5, 3.0, 10.0};
  std::string sname[4] = {"ratio", "etaR", "dl1R", "nvxR"};
  std::string lname[4] = {"Z", "E", "L", "V"};
  std::string wname[4] = {"W", "F", "M", "X"};
  std::string xname[4] = {"i#eta", "i#eta", "d_{L1}", "# Vertex"};
  int numb[4] = {10, 8, 8, 5};

  if (type == 0) {
    numb[0] = 8;
    for (int i = 0; i < 9; ++i)
      xbins[i] = xbin0[i];
  }
  TFile* file = new TFile(infile.c_str());
  std::vector<TH1D*> hists;
  char name[100], namw[100];
  if (file != nullptr) {
    for (int m1 = 0; m1 < 4; ++m1) {
      for (int m2 = 0; m2 < 6; ++m2) {
        sprintf(name, "%s%s%d0", prefix.c_str(), sname[m1].c_str(), iname[m2]);
        TH1D* hist0 = (TH1D*)file->FindObjectAny(name);
        bool ok = ((hist0 != nullptr) && (hist0->GetEntries() > 25));
        if ((mode / checkmode[m2]) % 10 > 0 && ok) {
          TH1D *histo(0), *histw(0);
          sprintf(name, "%s%s%d", prefix.c_str(), lname[m1].c_str(), iname[m2]);
          sprintf(namw, "%s%s%d", prefix.c_str(), wname[m1].c_str(), iname[m2]);
          if (m1 <= 1) {
            histo = new TH1D(name, hist0->GetTitle(), numb[m1], xbins);
            histw = new TH1D(namw, hist0->GetTitle(), numb[m1], xbins);
          } else if (m1 == 2) {
            histo = new TH1D(name, hist0->GetTitle(), numb[m1], dlbins);
            histw = new TH1D(namw, hist0->GetTitle(), numb[m1], dlbins);
          } else {
            histo = new TH1D(name, hist0->GetTitle(), numb[m1], vbins);
            histw = new TH1D(namw, hist0->GetTitle(), numb[m1], vbins);
          }
          int jmin(numb[m1]), jmax(0);
          for (int j = 0; j <= numb[m1]; ++j) {
            sprintf(name, "%s%s%d%d", prefix.c_str(), sname[m1].c_str(), iname[m2], j);
            TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
            TH1D* hist = (TH1D*)hist1->Clone();
            double value(0), error(0), width(0), werror(0);
            if (hist->GetEntries() > 0) {
              value = hist->GetMean();
              error = hist->GetRMS();
              std::pair<double, double> rmserr = GetWidth(hist, 0.2, 2.0);
              width = rmserr.first;
              werror = rmserr.second;
            }
            if (hist->GetEntries() > 4) {
              bool flag = (j == 0) ? true : false;
              results meaner = fitOneGauss(hist, flag, debug);
              value = meaner.mean;
              error = meaner.errmean;
              width = meaner.width;
              werror = meaner.errwidth;
              if (j != 0) {
                if (j < jmin)
                  jmin = j;
                if (j > jmax)
                  jmax = j;
              }
            }
            if (j == 0) {
              hists.push_back(hist);
            } else {
              double wbyv = width / value;
              double wverr = wbyv * std::sqrt((werror * werror) / (width * width) + (error * error) / (value * value));
              if (saveAll)
                hists.push_back(hist);
              histo->SetBinContent(j, value);
              histo->SetBinError(j, error);
              histw->SetBinContent(j, wbyv);
              histw->SetBinError(j, wverr);
            }
          }
          if (histo->GetEntries() > 2) {
            double LowEdge = histo->GetBinLowEdge(jmin);
            double HighEdge = histo->GetBinLowEdge(jmax) + histo->GetBinWidth(jmax);
            TFitResultPtr Fit = histo->Fit("pol0", "+QRWLS", "", LowEdge, HighEdge);
            if (debug) {
              std::cout << "Fit to Pol0: " << Fit->Value(0) << " +- " << Fit->FitResult::Error(0) << std::endl;
            }
            histo->GetXaxis()->SetTitle(xname[m1].c_str());
            histo->GetYaxis()->SetTitle("MPV(E_{HCAL}/(p-E_{ECAL}))");
            histo->GetYaxis()->SetRangeUser(0.4, 1.6);
            histw->GetXaxis()->SetTitle(xname[m1].c_str());
            histw->GetYaxis()->SetTitle("Width/MPV of (E_{HCAL}/(p-E_{ECAL}))");
            histw->GetYaxis()->SetRangeUser(0.0, 0.5);
          }
          hists.push_back(histo);
          hists.push_back(histw);
        }
      }
    }
    TFile* theFile(0);
    if (append) {
      theFile = new TFile(outfile.c_str(), "UPDATE");
    } else {
      theFile = new TFile(outfile.c_str(), "RECREATE");
    }
    theFile->cd();
    for (unsigned int i = 0; i < hists.size(); ++i) {
      TH1D* hnew = (TH1D*)hists[i]->Clone();
      hnew->Write();
    }
    theFile->Close();
    file->Close();
  }
}

void FitHistExtended(const char* infile,
                     const char* outfile,
                     std::string prefix,
                     int numb = 50,
                     int type = 3,
                     bool append = true,
                     bool fiteta = true,
                     int iname = 3,
                     bool debug = false) {
  std::string sname("ratio"), lname("Z"), wname("W"), ename("etaB");
  double xbins[99];
  double xbin[23] = {-23.0, -21.0, -19.0, -17.0, -15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, 0.0,
                     3.0,   5.0,   7.0,   9.0,   11.0,  13.0,  15.0,  17.0, 19.0, 21.0, 23.0};
  if (type == 2) {
    numb = 22;
    for (int k = 0; k <= numb; ++k)
      xbins[k] = xbin[k];
  } else {
    int neta = numb / 2;
    for (int k = 0; k < neta; ++k) {
      xbins[k] = (k - neta) - 0.5;
      xbins[numb - k] = (neta - k) + 0.5;
    }
    xbins[neta] = 0;
  }
  if (debug) {
    for (int k = 0; k <= numb; ++k)
      std::cout << " " << xbins[k];
    std::cout << std::endl;
  }
  TFile* file = new TFile(infile);
  std::vector<TH1D*> hists;
  char name[200];
  if (debug) {
    std::cout << infile << " " << file << std::endl;
  }
  if (file != nullptr) {
    sprintf(name, "%s%s%d0", prefix.c_str(), sname.c_str(), iname);
    TH1D* hist0 = (TH1D*)file->FindObjectAny(name);
    bool ok = (hist0 != nullptr);
    if (debug) {
      std::cout << name << " Pointer " << hist0 << " " << ok << std::endl;
    }
    if (ok) {
      TH1D *histo(0), *histw(0);
      if (numb > 0) {
        sprintf(name, "%s%s%d", prefix.c_str(), lname.c_str(), iname);
        histo = new TH1D(name, hist0->GetTitle(), numb, xbins);
        sprintf(name, "%s%s%d", prefix.c_str(), wname.c_str(), iname);
        histw = new TH1D(name, hist0->GetTitle(), numb, xbins);
        if (debug)
          std::cout << name << " " << histo->GetNbinsX() << std::endl;
      }
      if (hist0->GetEntries() > 10) {
        double rms;
        results meaner0 = fitOneGauss(hist0, true, debug);
        std::pair<double, double> meaner1 = GetMean(hist0, 0.2, 2.0, rms);
        std::pair<double, double> meaner2 = GetWidth(hist0, 0.2, 2.0);
        if (debug) {
          std::cout << "Fit " << meaner0.mean << ":" << meaner0.errmean << " Mean1 " << hist0->GetMean() << ":"
                    << hist0->GetMeanError() << " Mean2 " << meaner1.first << ":" << meaner1.second << " Width "
                    << meaner2.first << ":" << meaner2.second << std::endl;
        }
      }
      int nv1(100), nv2(0);
      int jmin(numb), jmax(0);
      for (int j = 0; j <= numb; ++j) {
        sprintf(name, "%s%s%d%d", prefix.c_str(), sname.c_str(), iname, j);
        TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
        if (debug) {
          std::cout << "Get Histogram for " << name << " at " << hist1 << std::endl;
        }
        double value(0), error(0), total(0), width(0), werror(0);
        if (hist1 == nullptr) {
          value = 1.0;
        } else {
          TH1D* hist = (TH1D*)hist1->Clone();
          if (hist->GetEntries() > 0) {
            value = hist->GetMean();
            error = hist->GetRMS();
            for (int i = 1; i <= hist->GetNbinsX(); ++i)
              total += hist->GetBinContent(i);
            std::pair<double, double> rmserr = GetWidth(hist, 0.2, 2.0);
            width = rmserr.first;
            werror = rmserr.second;
          }
          if (total > 4) {
            if (nv1 > j)
              nv1 = j;
            if (nv2 < j)
              nv2 = j;
            if (j == 0) {
              sprintf(name, "%sOne", hist1->GetName());
              TH1D* hist2 = (TH1D*)hist1->Clone(name);
              fitOneGauss(hist2, true, debug);
              hists.push_back(hist2);
              results meaner = fitTwoGauss(hist, debug);
              value = meaner.mean;
              error = meaner.errmean;
              width = meaner.width;
              werror = meaner.errwidth;
            } else {
              results meaner = fitOneGauss(hist, true, debug);
              value = meaner.mean;
              error = meaner.errmean;
              width = meaner.width;
              werror = meaner.errwidth;
            }
            if (j != 0) {
              if (j < jmin)
                jmin = j;
              if (j > jmax)
                jmax = j;
            }
          }
          hists.push_back(hist);
        }
        if (debug) {
          std::cout << "Hist " << j << " Value " << value << " +- " << error << std::endl;
        }
        if (j != 0) {
          double wbyv = width / value;
          double wverr = wbyv * std::sqrt((werror * werror) / (width * width) + (error * error) / (value * value));
          histo->SetBinContent(j, value);
          histo->SetBinError(j, error);
          histw->SetBinContent(j, wbyv);
          histw->SetBinError(j, wverr);
        }
      }
      if (histo != nullptr) {
        if (histo->GetEntries() > 2 && fiteta) {
          if (debug) {
            std::cout << "Jmin/max " << jmin << ":" << jmax << ":" << histo->GetNbinsX() << std::endl;
          }
          double LowEdge = histo->GetBinLowEdge(jmin);
          double HighEdge = histo->GetBinLowEdge(jmax) + histo->GetBinWidth(jmax);
          TFitResultPtr Fit = histo->Fit("pol0", "+QRWLS", "", LowEdge, HighEdge);
          if (debug) {
            std::cout << "Fit to Pol0: " << Fit->Value(0) << " +- " << Fit->FitResult::Error(0) << " in range " << nv1
                      << ":" << xbins[nv1] << ":" << nv2 << ":" << xbins[nv2] << std::endl;
          }
          histo->GetXaxis()->SetTitle("i#eta");
          histo->GetYaxis()->SetTitle("MPV(E_{HCAL}/(p-E_{ECAL}))");
          histo->GetYaxis()->SetRangeUser(0.4, 1.6);
          histw->GetXaxis()->SetTitle("i#eta");
          histw->GetYaxis()->SetTitle("MPV/Width(E_{HCAL}/(p-E_{ECAL}))");
          histw->GetYaxis()->SetRangeUser(0.0, 0.5);
        }
        hists.push_back(histo);
        hists.push_back(histw);
      } else {
        hists.push_back(hist0);
      }

      //Barrel,Endcap
      for (int j = 1; j <= 4; ++j) {
        sprintf(name, "%s%s%d%d", prefix.c_str(), ename.c_str(), iname, j);
        TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
        if (debug) {
          std::cout << "Get Histogram for " << name << " at " << hist1 << std::endl;
        }
        if (hist1 != nullptr) {
          TH1D* hist = (TH1D*)hist1->Clone();
          double value(0), error(0), total(0), width(0), werror(0);
          if (hist->GetEntries() > 0) {
            value = hist->GetMean();
            error = hist->GetRMS();
            for (int i = 1; i <= hist->GetNbinsX(); ++i)
              total += hist->GetBinContent(i);
          }
          if (total > 4) {
            sprintf(name, "%sOne", hist1->GetName());
            TH1D* hist2 = (TH1D*)hist1->Clone(name);
            results meanerr = fitOneGauss(hist2, true, debug);
            value = meanerr.mean;
            error = meanerr.errmean;
            width = meanerr.width;
            werror = meanerr.errwidth;
            double wbyv = width / value;
            double wverr = wbyv * std::sqrt((werror * werror) / (width * width) + (error * error) / (value * value));
            std::cout << hist2->GetName() << " MPV " << value << " +- " << error << " Width " << width << " +- "
                      << werror << " W/M " << wbyv << " +- " << wverr << std::endl;
            hists.push_back(hist2);
            if (hist1->GetBinLowEdge(1) < 0.1) {
              sprintf(name, "%sTwo", hist1->GetName());
              TH1D* hist3 = (TH1D*)hist1->Clone(name);
              fitLanGau(hist3, debug);
              hists.push_back(hist3);
            }
            results meaner0 = fitTwoGauss(hist, debug);
            value = meaner0.mean;
            error = meaner0.errmean;
            double rms;
            std::pair<double, double> meaner = GetMean(hist, 0.2, 2.0, rms);
            if (debug) {
              std::cout << "Fit " << value << ":" << error << ":" << hist->GetMeanError() << " Mean " << meaner.first
                        << ":" << meaner.second << std::endl;
            }
          }
          hists.push_back(hist);
        }
      }
    }
    TFile* theFile(0);
    if (append) {
      if (debug) {
        std::cout << "Open file " << outfile << " in append mode" << std::endl;
      }
      theFile = new TFile(outfile, "UPDATE");
    } else {
      if (debug) {
        std::cout << "Open file " << outfile << " in recreate mode" << std::endl;
      }
      theFile = new TFile(outfile, "RECREATE");
    }

    theFile->cd();
    for (unsigned int i = 0; i < hists.size(); ++i) {
      TH1D* hnew = (TH1D*)hists[i]->Clone();
      if (debug) {
        std::cout << "Write Histogram " << hnew->GetTitle() << std::endl;
      }
      hnew->Write();
    }
    theFile->Close();
    file->Close();
  }
}

void FitHistRBX(const char* infile, const char* outfile, std::string prefix, bool append = true, int iname = 3) {
  std::string sname("RBX"), lname("R");
  int numb(18);
  bool debug(false);
  char name[200];

  TFile* file = new TFile(infile);
  std::vector<TH1D*> hists;
  sprintf(name, "%s%s%d", prefix.c_str(), lname.c_str(), iname);
  TH1D* histo = new TH1D(name, "", numb, 0, numb);
  if (debug) {
    std::cout << infile << " " << file << std::endl;
  }
  if (file != nullptr) {
    for (int j = 0; j < numb; ++j) {
      sprintf(name, "%s%s%d%d", prefix.c_str(), sname.c_str(), iname, j + 1);
      TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
      if (debug) {
        std::cout << "Get Histogram for " << name << " at " << hist1 << std::endl;
      }
      TH1D* hist = (TH1D*)hist1->Clone();
      double value(0), error(0), total(0);
      if (hist->GetEntries() > 0) {
        value = hist->GetMean();
        error = hist->GetRMS();
        for (int i = 1; i <= hist->GetNbinsX(); ++i)
          total += hist->GetBinContent(i);
      }
      if (total > 4) {
        results meaner = fitOneGauss(hist, false, debug);
        value = meaner.mean;
        error = meaner.errmean;
      }
      hists.push_back(hist);
      histo->SetBinContent(j + 1, value);
      histo->SetBinError(j + 1, error);
    }
    histo->GetXaxis()->SetTitle("RBX #");
    histo->GetYaxis()->SetTitle("MPV(E_{HCAL}/(p-E_{ECAL}))");
    histo->GetYaxis()->SetRangeUser(0.75, 1.20);
    hists.push_back(histo);

    TFile* theFile(0);
    if (append) {
      if (debug) {
        std::cout << "Open file " << outfile << " in append mode" << std::endl;
      }
      theFile = new TFile(outfile, "UPDATE");
    } else {
      if (debug) {
        std::cout << "Open file " << outfile << " in recreate mode" << std::endl;
      }
      theFile = new TFile(outfile, "RECREATE");
    }

    theFile->cd();
    for (unsigned int i = 0; i < hists.size(); ++i) {
      TH1D* hnew = (TH1D*)hists[i]->Clone();
      if (debug) {
        std::cout << "Write Histogram " << hnew->GetTitle() << std::endl;
      }
      hnew->Write();
    }
    theFile->Close();
    file->Close();
  }
}

void PlotHist(const char* infile,
              std::string prefix,
              std::string text,
              int mode = 4,
              int kopt = 100,
              double lumi = 0,
              double ener = 13.0,
              bool dataMC = false,
              bool drawStatBox = true,
              bool save = false) {
  std::string name0[6] = {"ratio00", "ratio10", "ratio20", "ratio30", "ratio40", "ratio50"};
  std::string name1[5] = {"Z0", "Z1", "Z2", "Z3", "Z4"};
  std::string name2[5] = {"L0", "L1", "L2", "L3", "L4"};
  std::string name3[5] = {"V0", "V1", "V2", "V3", "V4"};
  std::string name4[12] = {"etaB31",
                           "etaB32",
                           "etaB33",
                           "etaB34",
                           "etaB11",
                           "etaB12",
                           "etaB13",
                           "etaB14",
                           "etaB01",
                           "etaB02",
                           "etaB03",
                           "etaB04"};
  std::string name5[5] = {"W0", "W1", "W2", "W3", "W4"};
  std::string title[6] = {"Tracks with p = 10:20 GeV",
                          "Tracks with p = 20:30 GeV",
                          "Tracks with p = 30:40 GeV",
                          "Tracks with p = 40:60 GeV",
                          "Tracks with p = 60:100 GeV",
                          "Tracks with p = 20:100 GeV"};
  std::string title1[12] = {"Tracks with p = 40:60 GeV (Barrel)",
                            "Tracks with p = 40:60 GeV (Transition)",
                            "Tracks with p = 40:60 GeV (Endcap)",
                            "Tracks with p = 40:60 GeV",
                            "Tracks with p = 20:30 GeV (Barrel)",
                            "Tracks with p = 20:30 GeV (Transition)",
                            "Tracks with p = 20:30 GeV (Endcap)",
                            "Tracks with p = 20:30 GeV",
                            "Tracks with p = 10:20 GeV (Barrel)",
                            "Tracks with p = 10:20 GeV (Transition)",
                            "Tracks with p = 10:20 GeV (Endcap)",
                            "Tracks with p = 10:20 GeV"};
  std::string xtitl[5] = {"E_{HCAL}/(p-E_{ECAL})", "i#eta", "d_{L1}", "# Vertex", "E_{HCAL}/(p-E_{ECAL})"};
  std::string ytitl[5] = {
      "Tracks", "MPV(E_{HCAL}/(p-E_{ECAL}))", "MPV(E_{HCAL}/(p-E_{ECAL}))", "MPV(E_{HCAL}/(p-E_{ECAL}))", "Tracks"};

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if (mode < 0 || mode > 5)
    mode = 0;
  if (drawStatBox) {
    int iopt(1110);
    if (mode != 0)
      iopt = 10;
    gStyle->SetOptStat(iopt);
    gStyle->SetOptFit(1);
  } else {
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
  }
  TFile* file = new TFile(infile);
  TLine* line(0);
  char name[100], namep[100];
  int kmax = (mode == 4) ? 12 : (((mode < 1) && (mode > 5)) ? 6 : 5);
  for (int k = 0; k < kmax; ++k) {
    if (mode == 1) {
      sprintf(name, "%s%s", prefix.c_str(), name1[k].c_str());
    } else if (mode == 2) {
      sprintf(name, "%s%s", prefix.c_str(), name2[k].c_str());
    } else if (mode == 3) {
      sprintf(name, "%s%s", prefix.c_str(), name3[k].c_str());
    } else if (mode == 4) {
      if ((kopt / 100) % 10 == 0) {
        sprintf(name, "%s%s", prefix.c_str(), name4[k].c_str());
      } else if ((kopt / 100) % 10 == 2) {
        sprintf(name, "%s%sTwo", prefix.c_str(), name4[k].c_str());
      } else {
        sprintf(name, "%s%sOne", prefix.c_str(), name4[k].c_str());
      }
    } else if (mode == 5) {
      sprintf(name, "%s%s", prefix.c_str(), name5[k].c_str());
    } else {
      if ((kopt / 100) % 10 == 0) {
        sprintf(name, "%s%s", prefix.c_str(), name0[k].c_str());
      } else if ((kopt / 100) % 10 == 2) {
        sprintf(name, "%s%sTwo", prefix.c_str(), name0[k].c_str());
      } else {
        sprintf(name, "%s%sOne", prefix.c_str(), name0[k].c_str());
      }
    }
    TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
    if (hist1 != nullptr) {
      TH1D* hist = (TH1D*)(hist1->Clone());
      double p0(1);
      double ymin(0.90);
      sprintf(namep, "c_%s", name);
      TCanvas* pad = new TCanvas(namep, namep, 700, 500);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      if ((kopt / 10) % 10 > 0)
        gPad->SetGrid();
      hist->GetXaxis()->SetTitleSize(0.04);
      hist->GetXaxis()->SetTitle(xtitl[mode].c_str());
      hist->GetYaxis()->SetTitle(ytitl[mode].c_str());
      hist->GetYaxis()->SetLabelOffset(0.005);
      hist->GetYaxis()->SetTitleSize(0.04);
      hist->GetYaxis()->SetLabelSize(0.035);
      hist->GetYaxis()->SetTitleOffset(1.10);
      if (mode == 0 || mode == 4) {
        if ((kopt / 100) % 10 == 2)
          hist->GetXaxis()->SetRangeUser(0.0, 0.30);
        else
          hist->GetXaxis()->SetRangeUser(0.25, 2.25);
      } else {
        if (mode == 5)
          hist->GetYaxis()->SetRangeUser(0.1, 0.50);
        else
          hist->GetYaxis()->SetRangeUser(0.8, 1.20);
        if (kopt % 10 > 0) {
          int nbin = hist->GetNbinsX();
          double LowEdge = (kopt % 10 == 1) ? hist->GetBinLowEdge(1) : -20;
          double HighEdge = (kopt % 10 == 1) ? hist->GetBinLowEdge(nbin) + hist->GetBinWidth(nbin) : 20;
          TFitResultPtr Fit = hist->Fit("pol0", "+QRWLS", "", LowEdge, HighEdge);
          p0 = Fit->Value(0);
        }
      }
      hist->SetMarkerStyle(20);
      hist->SetMarkerColor(2);
      hist->SetLineColor(2);
      hist->Draw();
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
      if (st1 != nullptr) {
        ymin = (mode == 0 || mode == 4) ? 0.70 : 0.80;
        st1->SetY1NDC(ymin);
        st1->SetY2NDC(0.90);
        st1->SetX1NDC(0.65);
        st1->SetX2NDC(0.90);
      }
      if (mode != 0 && mode != 4 && kopt % 10 > 0) {
        double xmin = hist->GetBinLowEdge(1);
        int nbin = hist->GetNbinsX();
        double xmax = hist->GetBinLowEdge(nbin) + hist->GetBinWidth(nbin);
        double mean(0), rms(0), total(0);
        int kount(0);
        for (int k = 2; k < nbin; ++k) {
          double x = hist->GetBinContent(k);
          double w = hist->GetBinError(k);
          mean += (x * w);
          rms += (x * x * w);
          total += w;
          ++kount;
        }
        mean /= total;
        rms /= total;
        double error = sqrt(rms - mean * mean) / total;
        line = new TLine(xmin, p0, xmax, p0);  //etamin,1.0,etamax,1.0);
        std::cout << xmin << ":" << xmax << ":" << p0 << " Mean " << nbin << ":" << kount << ":" << total << ":" << mean
                  << ":" << rms << ":" << error << std::endl;
        line->SetLineWidth(2);
        line->SetLineStyle(2);
        line->Draw("same");
      }
      pad->Update();
      double ymx(0.96), xmi(0.25), xmx(0.90);
      char txt[100];
      if (lumi > 0.1) {
        ymx = ymin - 0.005;
        xmi = 0.45;
        TPaveText* txt0 = new TPaveText(0.65, 0.91, 0.90, 0.96, "blNDC");
        txt0->SetFillColor(0);
        sprintf(txt, "%4.1f TeV %5.1f fb^{-1}", ener, lumi);
        txt0->AddText(txt);
        txt0->Draw("same");
      }
      double ymi = ymx - 0.05;
      TPaveText* txt1 = new TPaveText(xmi, ymi, xmx, ymx, "blNDC");
      txt1->SetFillColor(0);
      if (text == "") {
        if (mode == 4)
          sprintf(txt, "%s", title1[k].c_str());
        else
          sprintf(txt, "%s", title[k].c_str());
      } else {
        if (mode == 4)
          sprintf(txt, "%s (%s)", title1[k].c_str(), text.c_str());
        else
          sprintf(txt, "%s (%s)", title[k].c_str(), text.c_str());
      }
      txt1->AddText(txt);
      txt1->Draw("same");
      double xmax = (dataMC) ? 0.33 : 0.44;
      ymi = (lumi > 0.1) ? 0.91 : 0.84;
      ymx = ymi + 0.05;
      TPaveText* txt2 = new TPaveText(0.11, ymi, xmax, ymx, "blNDC");
      txt2->SetFillColor(0);
      if (dataMC)
        sprintf(txt, "CMS Preliminary");
      else
        sprintf(txt, "CMS Simulation Preliminary");
      txt2->AddText(txt);
      txt2->Draw("same");
      pad->Modified();
      pad->Update();
      if (save) {
        sprintf(name, "%s.pdf", pad->GetName());
        pad->Print(name);
      }
    }
  }
}

void PlotHists(std::string infile, std::string prefix, std::string text, bool drawStatBox = true, bool save = false) {
  int colors[6] = {1, 6, 4, 7, 2, 9};
  std::string types[6] = {"B", "C", "D", "E", "F", "G"};
  std::string names[3] = {"ratio20", "Z2", "W2"};
  std::string xtitl[3] = {"E_{HCAL}/(p-E_{ECAL})", "i#eta", "i#eta"};
  std::string ytitl[3] = {"Tracks", "MPV(E_{HCAL}/(p-E_{ECAL}))", "MPV/Width(E_{HCAL}/(p-E_{ECAL}))"};

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if (drawStatBox)
    gStyle->SetOptFit(10);
  else
    gStyle->SetOptFit(0);

  char name[100], namep[100];
  TFile* file = new TFile(infile.c_str());
  for (int i = 0; i < 3; ++i) {
    std::vector<TH1D*> hists;
    std::vector<int> kks;
    double ymax(0.77);
    if (drawStatBox) {
      if (i == 0)
        gStyle->SetOptStat(1100);
      else
        gStyle->SetOptStat(10);
    } else {
      gStyle->SetOptStat(0);
      ymax = 0.89;
    }
    for (int k = 0; k < 6; ++k) {
      sprintf(name, "%s%s%s", prefix.c_str(), types[k].c_str(), names[i].c_str());
      TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
      if (hist1 != nullptr) {
        hists.push_back((TH1D*)(hist1->Clone()));
        kks.push_back(k);
      }
    }
    if (hists.size() > 0) {
      sprintf(namep, "c_%s%s", prefix.c_str(), names[i].c_str());
      TCanvas* pad = new TCanvas(namep, namep, 700, 500);
      TLegend* legend = new TLegend(0.44, 0.89 - 0.055 * hists.size(), 0.69, ymax);
      legend->SetFillColor(kWhite);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      double ymax(0.90);
      double dy = (i == 0) ? 0.13 : 0.08;
      for (unsigned int jk = 0; jk < hists.size(); ++jk) {
        int k = kks[jk];
        hists[jk]->GetXaxis()->SetTitle(xtitl[i].c_str());
        hists[jk]->GetYaxis()->SetTitle(ytitl[i].c_str());
        hists[jk]->GetYaxis()->SetLabelOffset(0.005);
        hists[jk]->GetYaxis()->SetLabelSize(0.035);
        hists[jk]->GetYaxis()->SetTitleOffset(1.15);
        if (i == 0) {
          hists[jk]->GetXaxis()->SetRangeUser(0.0, 2.5);
        } else if (i == 1) {
          hists[jk]->GetYaxis()->SetRangeUser(0.5, 2.0);
        } else {
          hists[jk]->GetYaxis()->SetRangeUser(0.0, 0.5);
        }
        hists[jk]->SetMarkerStyle(20);
        hists[jk]->SetMarkerColor(colors[k]);
        hists[jk]->SetLineColor(colors[k]);
        if (jk == 0)
          hists[jk]->Draw();
        else
          hists[jk]->Draw("sames");
        pad->Update();
        TPaveStats* st1 = (TPaveStats*)hists[jk]->GetListOfFunctions()->FindObject("stats");
        if (st1 != nullptr) {
          double ymin = ymax - dy;
          st1->SetLineColor(colors[k]);
          st1->SetTextColor(colors[k]);
          st1->SetY1NDC(ymin);
          st1->SetY2NDC(ymax);
          st1->SetX1NDC(0.70);
          st1->SetX2NDC(0.90);
          ymax = ymin;
        }
        sprintf(name, "%s%s", text.c_str(), types[k].c_str());
        legend->AddEntry(hists[jk], name, "lp");
      }
      legend->Draw("same");
      pad->Update();
      TPaveText* txt1 = new TPaveText(0.34, 0.825, 0.69, 0.895, "blNDC");
      txt1->SetFillColor(0);
      char txt[100];
      sprintf(txt, "Tracks with p = 40:60 GeV");
      txt1->AddText(txt);
      txt1->Draw("same");
      TPaveText* txt2 = new TPaveText(0.11, 0.825, 0.33, 0.895, "blNDC");
      txt2->SetFillColor(0);
      sprintf(txt, "CMS Preliminary");
      txt2->AddText(txt);
      txt2->Draw("same");
      if (!drawStatBox && i == 1) {
        double xmin = hists[0]->GetBinLowEdge(1);
        int nbin = hists[0]->GetNbinsX();
        double xmax = hists[0]->GetBinLowEdge(nbin) + hists[0]->GetBinWidth(nbin);
        TLine line = TLine(xmin, 1.0, xmax, 1.0);  //etamin,1.0,etamax,1.0);
        line.SetLineWidth(4);
        line.Draw("same");
        pad->Update();
      }
      pad->Modified();
      pad->Update();
      if (save) {
        sprintf(name, "%s.pdf", pad->GetName());
        pad->Print(name);
      }
    }
  }
}

void PlotTwoHists(std::string infile,
                  std::string prefix1,
                  std::string text1,
                  std::string prefix2,
                  std::string text2,
                  std::string text0,
                  int type = 0,
                  int iname = 3,
                  double lumi = 0,
                  double ener = 13.0,
                  int drawStatBox = 0,
                  bool save = false) {
  int colors[2] = {2, 4};
  int numb[2] = {5, 1};
  std::string names0[5] = {"ratio00", "ratio00One", "etaB04One", "Z0", "W0"};
  std::string names1[5] = {"ratio10", "ratio10One", "etaB14One", "Z1", "W1"};
  std::string names2[5] = {"ratio30", "ratio30One", "etaB34One", "Z3", "W3"};
  std::string xtitl1[5] = {"E_{HCAL}/(p-E_{ECAL})", "E_{HCAL}/(p-E_{ECAL})", "E_{HCAL}/(p-E_{ECAL})", "i#eta", "i#eta"};
  std::string ytitl1[5] = {
      "Tracks", "Tracks", "Tracks", "MPV(E_{HCAL}/(p-E_{ECAL}))", "MPV/Width(E_{HCAL}/(p-E_{ECAL}))"};
  std::string names3[1] = {"R"};
  std::string xtitl2[1] = {"RBX #"};
  std::string ytitl2[1] = {"MPV(E_{HCAL}/(p-E_{ECAL}))"};

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if ((drawStatBox / 10) % 10 > 0)
    gStyle->SetOptFit(10);
  else
    gStyle->SetOptFit(0);

  if (type != 1)
    type = 0;
  char name[100], namep[100];
  TFile* file = new TFile(infile.c_str());
  for (int i = 0; i < numb[type]; ++i) {
    std::vector<TH1D*> hists;
    std::vector<int> kks;
    double ymax(0.77);
    if (drawStatBox % 10 > 0) {
      if (i != 2)
        gStyle->SetOptStat(1100);
      else
        gStyle->SetOptStat(10);
    } else {
      gStyle->SetOptStat(0);
      ymax = 0.82;
    }
    for (int k = 0; k < 2; ++k) {
      std::string prefix = (k == 0) ? prefix1 : prefix2;
      if (type == 0) {
        if (iname == 0)
          sprintf(name, "%s%s", prefix.c_str(), names0[i].c_str());
        else if (iname == 1)
          sprintf(name, "%s%s", prefix.c_str(), names1[i].c_str());
        else
          sprintf(name, "%s%s", prefix.c_str(), names2[i].c_str());
      } else {
        sprintf(name, "%s%s%d", prefix.c_str(), names3[i].c_str(), iname);
      }
      TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
      if (hist1 != nullptr) {
        hists.push_back((TH1D*)(hist1->Clone()));
        kks.push_back(k);
      }
    }
    if (hists.size() == 2) {
      if (type == 0) {
        if (iname == 0)
          sprintf(namep, "c_%s%s%s", prefix1.c_str(), prefix2.c_str(), names0[i].c_str());
        else if (iname == 1)
          sprintf(namep, "c_%s%s%s", prefix1.c_str(), prefix2.c_str(), names1[i].c_str());
        else
          sprintf(namep, "c_%s%s%s", prefix1.c_str(), prefix2.c_str(), names2[i].c_str());
      } else {
        sprintf(namep, "c_%s%s%s%d", prefix1.c_str(), prefix2.c_str(), names3[i].c_str(), iname);
      }
      double ymax(0.90);
      double dy = (i == 0) ? 0.13 : 0.08;
      double ymx0 = (drawStatBox == 0) ? (ymax - .01) : (ymax - dy * hists.size() - .01);
      TCanvas* pad = new TCanvas(namep, namep, 700, 500);
      TLegend* legend = new TLegend(0.64, ymx0 - 0.05 * hists.size(), 0.89, ymx0);
      legend->SetFillColor(kWhite);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      for (unsigned int jk = 0; jk < hists.size(); ++jk) {
        int k = kks[jk];
        hists[jk]->GetXaxis()->SetTitleSize(0.040);
        if (type == 0) {
          hists[jk]->GetXaxis()->SetTitle(xtitl1[i].c_str());
          hists[jk]->GetYaxis()->SetTitle(ytitl1[i].c_str());
        } else {
          hists[jk]->GetXaxis()->SetTitle(xtitl2[i].c_str());
          hists[jk]->GetYaxis()->SetTitle(ytitl2[i].c_str());
        }
        hists[jk]->GetYaxis()->SetLabelOffset(0.005);
        hists[jk]->GetYaxis()->SetLabelSize(0.035);
        hists[jk]->GetYaxis()->SetTitleSize(0.040);
        hists[jk]->GetYaxis()->SetTitleOffset(1.15);
        if ((type == 0) && (i != 3) && (i != 4))
          hists[jk]->GetXaxis()->SetRangeUser(0.0, 5.0);
        if (type == 0) {
          if (i == 3)
            hists[jk]->GetYaxis()->SetRangeUser(0.8, 1.2);
          else if (i == 4)
            hists[jk]->GetYaxis()->SetRangeUser(0.0, 0.5);
        }
        if (type != 0)
          hists[jk]->GetYaxis()->SetRangeUser(0.75, 1.2);
        hists[jk]->SetMarkerStyle(20);
        hists[jk]->SetMarkerColor(colors[k]);
        hists[jk]->SetLineColor(colors[k]);
        if (jk == 0)
          hists[jk]->Draw();
        else
          hists[jk]->Draw("sames");
        pad->Update();
        TPaveStats* st1 = (TPaveStats*)hists[jk]->GetListOfFunctions()->FindObject("stats");
        if (st1 != nullptr) {
          double ymin = ymax - dy;
          st1->SetLineColor(colors[k]);
          st1->SetTextColor(colors[k]);
          st1->SetY1NDC(ymin);
          st1->SetY2NDC(ymax);
          st1->SetX1NDC(0.70);
          st1->SetX2NDC(0.90);
          ymax = ymin;
        }
        if (k == 0)
          sprintf(name, "%s", text1.c_str());
        else
          sprintf(name, "%s", text2.c_str());
        legend->AddEntry(hists[jk], name, "lp");
      }
      legend->Draw("same");
      pad->Update();
      char txt[100];
      double xmi(0.10), xmx(0.895), ymx(0.95);
      if (lumi > 0.01) {
        xmx = 0.70;
        xmi = 0.30;
        TPaveText* txt0 = new TPaveText(0.705, 0.905, 0.90, 0.95, "blNDC");
        txt0->SetFillColor(0);
        sprintf(txt, "%4.1f TeV %5.1f fb^{-1}", ener, lumi);
        txt0->AddText(txt);
        txt0->Draw("same");
      }
      double ymi = ymx - 0.045;
      TPaveText* txt1 = new TPaveText(xmi, ymi, xmx, ymx, "blNDC");
      txt1->SetFillColor(0);
      if (iname == 0)
        sprintf(txt, "p = 20:30 GeV %s", text0.c_str());
      else
        sprintf(txt, "p = 40:60 GeV %s", text0.c_str());
      txt1->AddText(txt);
      txt1->Draw("same");
      ymi = (lumi > 0.1) ? 0.905 : 0.85;
      ymx = ymi + 0.045;
      TPaveText* txt2 = new TPaveText(0.105, ymi, 0.295, ymx, "blNDC");
      txt2->SetFillColor(0);
      sprintf(txt, "CMS Preliminary");
      txt2->AddText(txt);
      txt2->Draw("same");
      pad->Modified();
      if ((drawStatBox == 0) && (i == 3)) {
        double xmin = hists[0]->GetBinLowEdge(1);
        int nbin = hists[0]->GetNbinsX();
        double xmax = hists[0]->GetBinLowEdge(nbin) + hists[0]->GetBinWidth(nbin);
        TLine* line = new TLine(xmin, 1.0, xmax, 1.0);  //etamin,1.0,etamax,1.0);
        line->SetLineWidth(2);
        line->Draw("same");
        pad->Update();
      }
      pad->Update();
      if (save) {
        sprintf(name, "%s.pdf", pad->GetName());
        pad->Print(name);
      }
    }
  }
}

void PlotFiveHists(std::string infile,
                   std::string text0,
                   std::string prefix0,
                   int type = 0,
                   int iname = 3,
                   int drawStatBox = 0,
                   bool normalize = false,
                   bool save = false,
                   std::string prefix1 = "",
                   std::string text1 = "",
                   std::string prefix2 = "",
                   std::string text2 = "",
                   std::string prefix3 = "",
                   std::string text3 = "",
                   std::string prefix4 = "",
                   std::string text4 = "",
                   std::string prefix5 = "",
                   std::string text5 = "") {
  int colors[5] = {2, 4, 6, 1, 7};
  int numb[3] = {5, 1, 4};
  std::string names0[5] = {"ratio00", "ratio00One", "etaB04", "Z0", "W0"};
  std::string names1[5] = {"ratio10", "ratio10One", "etaB14", "Z1", "W1"};
  std::string names2[5] = {"ratio30", "ratio30One", "etaB34", "Z3", "W3"};
  std::string xtitl1[5] = {"E_{HCAL}/(p-E_{ECAL})", "E_{HCAL}/(p-E_{ECAL})", "E_{HCAL}/(p-E_{ECAL})", "i#eta", "i#eta"};
  std::string ytitl1[5] = {
      "Tracks", "Tracks", "Tracks", "MPV(E_{HCAL}/(p-E_{ECAL}))", "MPV/Width(E_{HCAL}/(p-E_{ECAL}))"};
  std::string names3[1] = {"R"};
  std::string xtitl2[1] = {"RBX #"};
  std::string ytitl2[1] = {"MPV(E_{HCAL}/(p-E_{ECAL}))"};
  std::string names4[4] = {"pp21", "pp22", "pp23", "pp24"};
  std::string xtitl3[4] = {"p (GeV)", "p (GeV)", "p (GeV)", "p (GeV)"};
  std::string ytitl3[4] = {"Tracks", "Tracks", "Tracks", "Tracks"};
  std::string title3[4] = {"Barrel", "Transition", "Endcap", "Combined"};

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if ((drawStatBox / 10) % 10 > 0)
    gStyle->SetOptFit(10);
  else
    gStyle->SetOptFit(0);

  if (type != 1 && type != 2)
    type = 0;
  char name[100], namep[100];
  TFile* file = new TFile(infile.c_str());
  for (int i = 0; i < numb[type]; ++i) {
    std::vector<TH1D*> hists;
    std::vector<int> kks;
    std::vector<std::string> texts;
    double ymax(0.77);
    if (drawStatBox % 10 > 0) {
      if (type == 2)
        gStyle->SetOptStat(1110);
      else if (i != 3)
        gStyle->SetOptStat(1100);
      else
        gStyle->SetOptStat(10);
    } else {
      gStyle->SetOptStat(0);
      ymax = 0.82;
    }
    for (int k = 0; k < 5; ++k) {
      std::string prefix, text;
      if (k == 0) {
        prefix = prefix1;
        text = text1;
      } else if (k == 1) {
        prefix = prefix2;
        text = text2;
      } else if (k == 2) {
        prefix = prefix3;
        text = text3;
      } else if (k == 3) {
        prefix = prefix4;
        text = text4;
      } else {
        prefix = prefix5;
        text = text5;
      }
      if (prefix != "") {
        if (type == 0) {
          if (iname == 0)
            sprintf(name, "%s%s", prefix.c_str(), names0[i].c_str());
          else if (iname == 1)
            sprintf(name, "%s%s", prefix.c_str(), names1[i].c_str());
          else
            sprintf(name, "%s%s", prefix.c_str(), names2[i].c_str());
        } else if (type == 1) {
          sprintf(name, "%s%s%d", prefix.c_str(), names3[i].c_str(), iname);
        } else {
          sprintf(name, "%s%s", prefix.c_str(), names4[i].c_str());
        }
        TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
        if (hist1 != nullptr) {
          hists.push_back((TH1D*)(hist1->Clone()));
          kks.push_back(k);
          texts.push_back(text);
        }
      }
    }
    if (hists.size() > 0) {
      if (type == 0) {
        if (iname == 0)
          sprintf(namep, "c_%s%s", prefix0.c_str(), names0[i].c_str());
        else if (iname == 1)
          sprintf(namep, "c_%s%s", prefix0.c_str(), names1[i].c_str());
        else
          sprintf(namep, "c_%s%s", prefix0.c_str(), names2[i].c_str());
      } else if (type == 1) {
        sprintf(namep, "c_%s%s%d", prefix0.c_str(), names3[i].c_str(), iname);
      } else {
        sprintf(namep, "c_%s%s", prefix0.c_str(), names4[i].c_str());
      }
      double ymax(0.90);
      double dy = (i == 0 && type == 0) ? 0.13 : 0.08;
      double ymx0 = (drawStatBox == 0) ? (ymax - .01) : (ymax - dy * hists.size() - .01);
      TCanvas* pad = new TCanvas(namep, namep, 700, 500);
      TLegend* legend = new TLegend(0.64, ymx0 - 0.05 * hists.size(), 0.89, ymx0);
      legend->SetFillColor(kWhite);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      for (unsigned int jk = 0; jk < hists.size(); ++jk) {
        int k = kks[jk];
        hists[jk]->GetXaxis()->SetTitleSize(0.040);
        if (type == 0) {
          hists[jk]->GetXaxis()->SetTitle(xtitl1[i].c_str());
          hists[jk]->GetYaxis()->SetTitle(ytitl1[i].c_str());
        } else if (type == 1) {
          hists[jk]->GetXaxis()->SetTitle(xtitl2[i].c_str());
          hists[jk]->GetYaxis()->SetTitle(ytitl2[i].c_str());
        } else {
          hists[jk]->GetXaxis()->SetTitle(xtitl3[i].c_str());
          hists[jk]->GetYaxis()->SetTitle(ytitl3[i].c_str());
        }
        hists[jk]->GetYaxis()->SetLabelOffset(0.005);
        hists[jk]->GetYaxis()->SetLabelSize(0.035);
        hists[jk]->GetYaxis()->SetTitleSize(0.040);
        hists[jk]->GetYaxis()->SetTitleOffset(1.15);
        if ((type == 0) && (i != 3) && (i != 4))
          hists[jk]->GetXaxis()->SetRangeUser(0.0, 2.5);
        if (type == 0) {
          if (i == 3)
            hists[jk]->GetYaxis()->SetRangeUser(0.8, 1.2);
          else if (i == 4)
            hists[jk]->GetYaxis()->SetRangeUser(0.0, 0.5);
        }
        if (type == 1)
          hists[jk]->GetYaxis()->SetRangeUser(0.75, 1.2);
        hists[jk]->SetMarkerStyle(20);
        hists[jk]->SetMarkerColor(colors[k]);
        hists[jk]->SetLineColor(colors[k]);
        if (normalize && ((type == 2) || ((type == 0) && (i != 3)))) {
          if (jk == 0)
            hists[jk]->DrawNormalized();
          else
            hists[jk]->DrawNormalized("sames");
        } else {
          if (jk == 0)
            hists[jk]->Draw();
          else
            hists[jk]->Draw("sames");
        }
        pad->Update();
        TPaveStats* st1 = (TPaveStats*)hists[jk]->GetListOfFunctions()->FindObject("stats");
        if (st1 != nullptr) {
          double ymin = ymax - dy;
          st1->SetLineColor(colors[k]);
          st1->SetTextColor(colors[k]);
          st1->SetY1NDC(ymin);
          st1->SetY2NDC(ymax);
          st1->SetX1NDC(0.70);
          st1->SetX2NDC(0.90);
          ymax = ymin;
        }
        sprintf(name, "%s", texts[jk].c_str());
        legend->AddEntry(hists[jk], name, "lp");
      }
      legend->Draw("same");
      pad->Update();
      TPaveText* txt1 = new TPaveText(0.10, 0.905, 0.80, 0.95, "blNDC");
      txt1->SetFillColor(0);
      char txt[100];
      if (type == 2) {
        sprintf(txt, "p = 40:60 GeV (%s)", title3[i].c_str());
      } else if (((type == 0) && (iname == 0))) {
        sprintf(txt, "p = 20:30 GeV %s", text0.c_str());
      } else {
        sprintf(txt, "p = 40:60 GeV %s", text0.c_str());
      }
      txt1->AddText(txt);
      txt1->Draw("same");
      TPaveText* txt2 = new TPaveText(0.11, 0.825, 0.33, 0.895, "blNDC");
      txt2->SetFillColor(0);
      sprintf(txt, "CMS Preliminary");
      txt2->AddText(txt);
      txt2->Draw("same");
      pad->Modified();
      if ((drawStatBox == 0) && (i == 3)) {
        double xmin = hists[0]->GetBinLowEdge(1);
        int nbin = hists[0]->GetNbinsX();
        double xmax = hists[0]->GetBinLowEdge(nbin) + hists[0]->GetBinWidth(nbin);
        TLine* line = new TLine(xmin, 1.0, xmax, 1.0);  //etamin,1.0,etamax,1.0);
        line->SetLineWidth(2);
        line->Draw("same");
        pad->Update();
      }
      pad->Update();
      if (save) {
        sprintf(name, "%s.pdf", pad->GetName());
        pad->Print(name);
      }
    }
  }
}

void PlotHistCorrResults(std::string infile, std::string text, std::string prefixF, bool save = false) {
  std::string name[5] = {"Eta1Bf", "Eta2Bf", "Eta1Af", "Eta2Af", "Cvg0"};
  std::string title[5] = {"Mean at the start of itertions",
                          "Median at the start of itertions",
                          "Mean at the end of itertions",
                          "Median at the end of itertions",
                          ""};
  int type[5] = {0, 0, 0, 0, 1};

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(10);
  gStyle->SetOptFit(10);
  TFile* file = new TFile(infile.c_str());
  char namep[100];
  for (int k = 0; k < 5; ++k) {
    TH1D* hist1 = (TH1D*)file->FindObjectAny(name[k].c_str());
    if (hist1 != nullptr) {
      TH1D* hist = (TH1D*)(hist1->Clone());
      sprintf(namep, "c_%s%s", prefixF.c_str(), name[k].c_str());
      TCanvas* pad = new TCanvas(namep, namep, 700, 500);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      hist->GetYaxis()->SetLabelOffset(0.005);
      hist->GetYaxis()->SetTitleOffset(1.20);
      double xmin = hist->GetBinLowEdge(1);
      int nbin = hist->GetNbinsX();
      double xmax = hist->GetBinLowEdge(nbin) + hist->GetBinWidth(nbin);
      std::cout << hist->GetTitle() << " Bins " << nbin << ":" << xmin << ":" << xmax << std::endl;
      double xlow(0.12), ylow(0.82);
      char txt[100], option[2];
      if (type[k] == 0) {
        sprintf(namep, "f_%s", name[k].c_str());
        TF1* func = new TF1(namep, "pol0", xmin, xmax);
        hist->Fit(func, "+QWL", "");
        if (text == "")
          sprintf(txt, "%s", title[k].c_str());
        else
          sprintf(txt, "%s (balancing the %s)", title[k].c_str(), text.c_str());
        sprintf(option, "%s", "");
      } else {
        hist->GetXaxis()->SetTitle("Iterations");
        hist->GetYaxis()->SetTitle("Convergence");
        hist->SetMarkerStyle(20);
        hist->SetMarkerColor(2);
        hist->SetMarkerSize(0.8);
        xlow = 0.50;
        ylow = 0.86;
        sprintf(txt, "(%s)", text.c_str());
        sprintf(option, "%s", "p");
      }
      hist->Draw(option);
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
      if (st1 != nullptr) {
        st1->SetY1NDC(ylow);
        st1->SetY2NDC(0.90);
        st1->SetX1NDC(0.70);
        st1->SetX2NDC(0.90);
      }
      TPaveText* txt1 = new TPaveText(xlow, 0.82, 0.68, 0.88, "blNDC");
      txt1->SetFillColor(0);
      txt1->AddText(txt);
      txt1->Draw("same");
      pad->Modified();
      pad->Update();
      if (save) {
        sprintf(namep, "%s.pdf", pad->GetName());
        pad->Print(namep);
      }
    }
  }
}

void PlotHistCorrFactor(char* infile,
                        std::string text,
                        std::string prefixF = "",
                        double scale = 1.0,
                        int nmin = 100,
                        bool dataMC = false,
                        bool drawStatBox = true,
                        bool save = false) {
  std::map<int, cfactors> cfacs;
  int etamin(100), etamax(-100), maxdepth(0);
  readCorrFactors(infile, scale, cfacs, etamin, etamax, maxdepth);

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if (drawStatBox) {
    gStyle->SetOptStat(10);
    gStyle->SetOptFit(10);
  } else {
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
  }
  int colors[6] = {1, 6, 4, 7, 2, 9};
  int mtype[6] = {20, 21, 22, 23, 24, 33};
  int nbin = etamax - etamin + 1;
  std::vector<TH1D*> hists;
  std::vector<int> entries;
  char name[100];
  double dy(0);
  int fits(0);
  for (int j = 0; j < maxdepth; ++j) {
    sprintf(name, "hd%d", j + 1);
    TH1D* h = new TH1D(name, name, nbin, etamin, etamax);
    int nent(0);
    for (std::map<int, cfactors>::const_iterator itr = cfacs.begin(); itr != cfacs.end(); ++itr) {
      if ((itr->second).depth == j + 1) {
        int ieta = (itr->second).ieta;
        int bin = ieta - etamin + 1;
        float val = (itr->second).corrf;
        float dvl = (itr->second).dcorr;
        h->SetBinContent(bin, val);
        h->SetBinError(bin, dvl);
        nent++;
      }
    }
    if (nent > nmin) {
      fits++;
      dy += 0.025;
      sprintf(name, "hdf%d", j + 1);
      TF1* func = new TF1(name, "pol0", etamin, etamax);
      h->Fit(func, "+QWLR", "");
    }
    h->SetLineColor(colors[j]);
    h->SetMarkerColor(colors[j]);
    h->SetMarkerStyle(mtype[j]);
    h->GetXaxis()->SetTitle("i#eta");
    h->GetYaxis()->SetTitle("Correction Factor");
    h->GetYaxis()->SetLabelOffset(0.005);
    h->GetYaxis()->SetTitleOffset(1.20);
    h->GetYaxis()->SetRangeUser(0.0, 2.0);
    hists.push_back(h);
    entries.push_back(nent);
    dy += 0.025;
  }
  sprintf(name, "c_%sCorrFactor", prefixF.c_str());
  TCanvas* pad = new TCanvas(name, name, 700, 500);
  pad->SetRightMargin(0.10);
  pad->SetTopMargin(0.10);
  double yh = 0.90;
  // double yl = yh - 0.025 * hists.size() - dy - 0.01;
  double yl = 0.15;
  TLegend* legend = new TLegend(0.35, yl, 0.65, yl + 0.04 * hists.size());
  legend->SetFillColor(kWhite);
  for (unsigned int k = 0; k < hists.size(); ++k) {
    if (k == 0)
      hists[k]->Draw("");
    else
      hists[k]->Draw("sames");
    pad->Update();
    TPaveStats* st1 = (TPaveStats*)hists[k]->GetListOfFunctions()->FindObject("stats");
    if (st1 != nullptr) {
      dy = (entries[k] > nmin) ? 0.05 : 0.025;
      st1->SetLineColor(colors[k]);
      st1->SetTextColor(colors[k]);
      st1->SetY1NDC(yh - dy);
      st1->SetY2NDC(yh);
      st1->SetX1NDC(0.70);
      st1->SetX2NDC(0.90);
      yh -= dy;
    }
    sprintf(name, "Depth %d (%s)", k + 1, text.c_str());
    legend->AddEntry(hists[k], name, "lp");
  }
  legend->Draw("same");
  pad->Update();
  if (fits < 1) {
    double xmin = hists[0]->GetBinLowEdge(1);
    int nbin = hists[0]->GetNbinsX();
    double xmax = hists[0]->GetBinLowEdge(nbin) + hists[0]->GetBinWidth(nbin);
    TLine* line = new TLine(xmin, 1.0, xmax, 1.0);
    line->SetLineColor(9);
    line->SetLineWidth(2);
    line->SetLineStyle(2);
    line->Draw("same");
    pad->Modified();
    pad->Update();
  }
  char txt1[30];
  double xmax = (dataMC) ? 0.33 : 0.44;
  TPaveText* txt2 = new TPaveText(0.11, 0.85, xmax, 0.89, "blNDC");
  txt2->SetFillColor(0);
  if (dataMC)
    sprintf(txt1, "CMS Preliminary");
  else
    sprintf(txt1, "CMS Simulation Preliminary");
  txt2->AddText(txt1);
  txt2->Draw("same");
  pad->Modified();
  pad->Update();
  if (save) {
    sprintf(name, "%s.pdf", pad->GetName());
    pad->Print(name);
  }
}

void PlotHistCorrAsymmetry(char* infile, std::string text, std::string prefixF = "", bool save = false) {
  std::map<int, cfactors> cfacs;
  int etamin(100), etamax(-100), maxdepth(0);
  double scale(1.0);
  readCorrFactors(infile, scale, cfacs, etamin, etamax, maxdepth);

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(10);
  int colors[6] = {1, 6, 4, 7, 2, 9};
  int mtype[6] = {20, 21, 22, 23, 24, 33};
  int nbin = etamax + 1;
  std::vector<TH1D*> hists;
  std::vector<int> entries;
  char name[100];
  double dy(0);
  for (int j = 0; j < maxdepth; ++j) {
    sprintf(name, "hd%d", j + 1);
    TH1D* h = new TH1D(name, name, nbin, 0, etamax);
    int nent(0);
    for (std::map<int, cfactors>::const_iterator itr = cfacs.begin(); itr != cfacs.end(); ++itr) {
      if ((itr->second).depth == j + 1) {
        int ieta = (itr->second).ieta;
        float vl1 = (itr->second).corrf;
        float dv1 = (itr->second).dcorr;
        if (ieta > 0) {
          for (std::map<int, cfactors>::const_iterator ktr = cfacs.begin(); ktr != cfacs.end(); ++ktr) {
            if (((ktr->second).depth == j + 1) && ((ktr->second).ieta == -ieta)) {
              float vl2 = (ktr->second).corrf;
              float dv2 = (ktr->second).dcorr;
              float val = 2.0 * (vl1 - vl2) / (vl1 + vl2);
              float dvl = (4.0 * sqrt(vl1 * vl1 * dv2 * dv2 + vl2 * vl2 * dv1 * dv1) / ((vl1 + vl2) * (vl1 + vl2)));
              int bin = ieta;
              h->SetBinContent(bin, val);
              h->SetBinError(bin, dvl);
              nent++;
            }
          }
        }
      }
    }
    h->SetLineColor(colors[j]);
    h->SetMarkerColor(colors[j]);
    h->SetMarkerStyle(mtype[j]);
    h->GetXaxis()->SetTitle("i#eta");
    h->GetYaxis()->SetTitle("Asymmetry in Correction Factor");
    h->GetYaxis()->SetLabelOffset(0.005);
    h->GetYaxis()->SetTitleOffset(1.20);
    h->GetYaxis()->SetRangeUser(-0.25, 0.25);
    hists.push_back(h);
    entries.push_back(nent);
    dy += 0.025;
  }
  sprintf(name, "c_%sCorrAsymmetry", prefixF.c_str());
  TCanvas* pad = new TCanvas(name, name, 700, 500);
  pad->SetRightMargin(0.10);
  pad->SetTopMargin(0.10);
  double yh = 0.90;
  double yl = yh - 0.035 * hists.size() - dy - 0.01;
  TLegend* legend = new TLegend(0.60, yl, 0.90, yl + 0.035 * hists.size());
  legend->SetFillColor(kWhite);
  for (unsigned int k = 0; k < hists.size(); ++k) {
    if (k == 0)
      hists[k]->Draw("");
    else
      hists[k]->Draw("sames");
    pad->Update();
    sprintf(name, "Depth %d (%s)", k + 1, text.c_str());
    legend->AddEntry(hists[k], name, "lp");
  }
  legend->Draw("same");
  pad->Update();

  TLine* line = new TLine(0.0, 0.0, etamax, 0.0);
  line->SetLineColor(9);
  line->SetLineWidth(2);
  line->SetLineStyle(2);
  line->Draw("same");
  pad->Update();

  if (save) {
    sprintf(name, "%s.pdf", pad->GetName());
    pad->Print(name);
  }
}

void PlotHistCorrFactors(char* infile1,
                         std::string text1,
                         char* infile2,
                         std::string text2,
                         char* infile3,
                         std::string text3,
                         char* infile4,
                         std::string text4,
                         char* infile5,
                         std::string text5,
                         std::string prefixF,
                         bool ratio = false,
                         bool drawStatBox = true,
                         int nmin = 100,
                         bool dataMC = false,
                         int year = 2018,
                         bool save = false) {
  std::map<int, cfactors> cfacs[5];
  std::vector<std::string> texts;
  int nfile(0), etamin(100), etamax(-100), maxdepth(0);
  const char* blank("");
  if (infile1 != blank) {
    readCorrFactors(infile1, 1.0, cfacs[nfile], etamin, etamax, maxdepth);
    if (cfacs[nfile].size() > 0) {
      texts.push_back(text1);
      ++nfile;
    }
  }
  if (infile2 != blank) {
    readCorrFactors(infile2, 1.0, cfacs[nfile], etamin, etamax, maxdepth);
    if (cfacs[nfile].size() > 0) {
      texts.push_back(text2);
      ++nfile;
    }
  }
  if (infile3 != blank) {
    readCorrFactors(infile3, 1.0, cfacs[nfile], etamin, etamax, maxdepth);
    if (cfacs[nfile].size() > 0) {
      texts.push_back(text3);
      ++nfile;
    }
  }
  if (infile4 != blank) {
    readCorrFactors(infile4, 1.0, cfacs[nfile], etamin, etamax, maxdepth);
    if (cfacs[nfile].size() > 0) {
      texts.push_back(text4);
      ++nfile;
    }
  }
  if (infile5 != blank) {
    readCorrFactors(infile5, 1.0, cfacs[nfile], etamin, etamax, maxdepth);
    if (cfacs[nfile].size() > 0) {
      texts.push_back(text5);
      ++nfile;
    }
  }

  if (nfile > 1) {
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);
    gStyle->SetFillColor(kWhite);
    gStyle->SetOptTitle(0);
    if ((!ratio) && drawStatBox) {
      gStyle->SetOptStat(10);
      gStyle->SetOptFit(10);
    } else {
      gStyle->SetOptStat(0);
      gStyle->SetOptFit(0);
    }
    int colors[6] = {1, 6, 4, 7, 2, 9};
    int mtype[6] = {20, 24, 22, 23, 21, 33};
    int nbin = etamax - etamin + 1;
    std::vector<TH1D*> hists;
    std::vector<int> entries, htype, depths;
    std::vector<double> fitr;
    char name[100];
    double dy(0);
    int fits(0);
    if (ratio) {
      for (int ih = 1; ih < nfile; ++ih) {
        for (int j = 0; j < maxdepth; ++j) {
          sprintf(name, "h%dd%d", ih, j + 1);
          TH1D* h = new TH1D(name, name, nbin, etamin, etamax);
          double sumNum(0), sumDen(0);
          std::map<int, cfactors>::const_iterator ktr = cfacs[ih].begin();
          for (std::map<int, cfactors>::const_iterator itr = cfacs[0].begin(); itr != cfacs[0].end(); ++itr, ++ktr) {
            int dep = (itr->second).depth;
            if (dep == j + 1) {
              int ieta = (itr->second).ieta;
              int bin = ieta - etamin + 1;
              float val = (itr->second).corrf / (ktr->second).corrf;
              float dvl =
                  val *
                  sqrt((((itr->second).dcorr * (itr->second).dcorr) / ((itr->second).corrf * (itr->second).corrf)) +
                       (((ktr->second).dcorr * (ktr->second).dcorr) / ((ktr->second).corrf * (ktr->second).corrf)));
              h->SetBinContent(bin, val);
              h->SetBinError(bin, dvl);
              sumNum += (val / (dvl * dvl));
              sumDen += (1.0 / (dvl * dvl));
            }
          }
          double fit = (sumDen > 0) ? (sumNum / sumDen) : 1.0;
          std::cout << "Fit to Pol0: " << fit << std::endl;
          h->SetLineColor(colors[ih]);
          h->SetMarkerColor(colors[ih]);
          h->SetMarkerStyle(mtype[j]);
          h->SetMarkerSize(0.9);
          h->GetXaxis()->SetTitle("i#eta");
          sprintf(name, "CF_{%s}/CF_{Set}", texts[0].c_str());
          h->GetYaxis()->SetTitle(name);
          h->GetYaxis()->SetLabelOffset(0.005);
          h->GetYaxis()->SetTitleSize(0.036);
          h->GetYaxis()->SetTitleOffset(1.20);
          h->GetYaxis()->SetRangeUser(0.80, 1.20);
          hists.push_back(h);
          fitr.push_back(fit);
          htype.push_back(ih);
          depths.push_back(j + 1);
        }
      }
    } else {
      for (int k1 = 0; k1 < nfile; ++k1) {
        for (int j = 0; j < maxdepth; ++j) {
          sprintf(name, "h%dd%d", k1, j + 1);
          TH1D* h = new TH1D(name, name, nbin, etamin, etamax);
          int nent(0);
          for (std::map<int, cfactors>::const_iterator itr = cfacs[k1].begin(); itr != cfacs[k1].end(); ++itr) {
            int dep = (itr->second).depth;
            if (dep == j + 1) {
              int ieta = (itr->second).ieta;
              int bin = ieta - etamin + 1;
              float val = (itr->second).corrf;
              float dvl = (itr->second).dcorr;
              h->SetBinContent(bin, val);
              h->SetBinError(bin, dvl);
              nent++;
            }
          }
          if (nent > nmin) {
            fits++;
            if (drawStatBox)
              dy += 0.025;
            sprintf(name, "h%ddf%d", k1, j + 1);
            TF1* func = new TF1(name, "pol0", etamin, etamax);
            h->Fit(func, "+QWLR", "");
          }
          h->SetLineColor(colors[k1]);
          h->SetMarkerColor(colors[k1]);
          h->SetMarkerStyle(mtype[j]);
          h->SetMarkerSize(0.9);
          h->GetXaxis()->SetTitle("i#eta");
          h->GetYaxis()->SetTitle("Correction Factor");
          h->GetYaxis()->SetLabelOffset(0.005);
          h->GetYaxis()->SetTitleOffset(1.20);
          h->GetYaxis()->SetRangeUser(0.5, 1.5);
          hists.push_back(h);
          entries.push_back(nent);
          if (drawStatBox)
            dy += 0.025;
          htype.push_back(k1);
          depths.push_back(j + 1);
        }
      }
    }
    if (ratio)
      sprintf(name, "c_Corr%sRatio", prefixF.c_str());
    else
      sprintf(name, "c_Corr%s", prefixF.c_str());
    TCanvas* pad = new TCanvas(name, name, 700, 500);
    pad->SetRightMargin(0.10);
    pad->SetTopMargin(0.10);
    double yh = 0.90;
    double yl = yh - 0.035 * hists.size() - dy - 0.01;
    TLegend* legend = new TLegend(0.55, yl, 0.90, yl + 0.035 * hists.size());
    legend->SetFillColor(kWhite);
    for (unsigned int k = 0; k < hists.size(); ++k) {
      if (k == 0)
        hists[k]->Draw("");
      else
        hists[k]->Draw("sames");
      pad->Update();
      int k1 = htype[k];
      if (!ratio) {
        TPaveStats* st1 = (TPaveStats*)hists[k]->GetListOfFunctions()->FindObject("stats");
        if (st1 != nullptr) {
          dy = (entries[k] > nmin) ? 0.05 : 0.025;
          st1->SetLineColor(colors[k1]);
          st1->SetTextColor(colors[k1]);
          st1->SetY1NDC(yh - dy);
          st1->SetY2NDC(yh);
          st1->SetX1NDC(0.70);
          st1->SetX2NDC(0.90);
          yh -= dy;
        }
        sprintf(name, "Depth %d (%s)", depths[k], texts[k1].c_str());
      } else {
        sprintf(name, "Depth %d (%s Mean = %5.3f)", depths[k], texts[k1].c_str(), fitr[k]);
      }
      legend->AddEntry(hists[k], name, "lp");
    }
    legend->Draw("same");
    TPaveText* txt0 = new TPaveText(0.12, 0.84, 0.49, 0.89, "blNDC");
    txt0->SetFillColor(0);
    char txt[40];
    if (dataMC)
      sprintf(txt, "CMS Preliminary (%d)", year);
    else
      sprintf(txt, "CMS Simulation Preliminary (%d)", year);
    txt0->AddText(txt);
    txt0->Draw("same");
    pad->Update();
    if (fits < 1) {
      double xmin = hists[0]->GetBinLowEdge(1);
      int nbin = hists[0]->GetNbinsX();
      double xmax = hists[0]->GetBinLowEdge(nbin) + hists[0]->GetBinWidth(nbin);
      TLine* line = new TLine(xmin, 1.0, xmax, 1.0);
      line->SetLineColor(9);
      line->SetLineWidth(2);
      line->SetLineStyle(2);
      line->Draw("same");
      pad->Update();
    }
    if (save) {
      sprintf(name, "%s.pdf", pad->GetName());
      pad->Print(name);
    }
  }
}

void PlotHistCorrSys(std::string infilec, int conds, std::string text, bool save = false) {
  char fname[100];
  sprintf(fname, "%s_cond0.txt", infilec.c_str());
  int etamin(100), etamax(-100), maxdepth(0);
  std::map<int, cfactors> cfacs;
  readCorrFactors(fname, 1.0, cfacs, etamin, etamax, maxdepth);
  // There are good records from the master file
  if (cfacs.size() > 0) {
    // Now read the other files
    std::map<int, cfactors> errfacs;
    for (int i = 0; i < conds; ++i) {
      sprintf(fname, "%s_cond%d.txt", infilec.c_str(), i + 1);
      std::map<int, cfactors> cfacx;
      int etamin1(100), etamax1(-100), maxdepth1(0);
      readCorrFactors(fname, 1.0, cfacx, etamin1, etamax1, maxdepth1);
      for (std::map<int, cfactors>::const_iterator itr1 = cfacx.begin(); itr1 != cfacx.end(); ++itr1) {
        std::map<int, cfactors>::iterator itr2 = errfacs.find(itr1->first);
        float corrf = (itr1->second).corrf;
        if (itr2 == errfacs.end()) {
          errfacs[itr1->first] = cfactors(1, 0, corrf, corrf * corrf);
        } else {
          int nent = (itr2->second).ieta + 1;
          float c1 = (itr2->second).corrf + corrf;
          float c2 = (itr2->second).dcorr + (corrf * corrf);
          errfacs[itr1->first] = cfactors(nent, 0, c1, c2);
        }
      }
    }
    // find the RMS from the distributions
    for (std::map<int, cfactors>::iterator itr = errfacs.begin(); itr != errfacs.end(); ++itr) {
      int nent = (itr->second).ieta;
      float mean = (itr->second).corrf / nent;
      float rms2 = (itr->second).dcorr / nent - (mean * mean);
      float rms = rms2 > 0 ? sqrt(rms2) : 0;
      errfacs[itr->first] = cfactors(nent, 0, mean, rms);
    }
    // Now combine the errors and plot
    int k(0);
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);
    gStyle->SetFillColor(kWhite);
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(10);
    gStyle->SetOptFit(10);
    int colors[6] = {1, 6, 4, 7, 2, 9};
    int mtype[6] = {20, 21, 22, 23, 24, 33};
    std::vector<TH1D*> hists;
    char name[100];
    int nbin = etamax - etamin + 1;
    for (int j = 0; j < maxdepth; ++j) {
      sprintf(name, "hd%d", j + 1);
      TH1D* h = new TH1D(name, name, nbin, etamin, etamax);
      h->SetLineColor(colors[j]);
      h->SetMarkerColor(colors[j]);
      h->SetMarkerStyle(mtype[j]);
      h->GetXaxis()->SetTitle("i#eta");
      h->GetYaxis()->SetTitle("Correction Factor");
      h->GetYaxis()->SetLabelOffset(0.005);
      h->GetYaxis()->SetTitleOffset(1.20);
      h->GetYaxis()->SetRangeUser(0.0, 2.0);
      hists.push_back(h);
    }
    for (std::map<int, cfactors>::iterator itr = cfacs.begin(); itr != cfacs.end(); ++itr, ++k) {
      std::map<int, cfactors>::iterator itr2 = errfacs.find(itr->first);
      float mean2 = (itr2 == errfacs.end()) ? 0 : (itr2->second).corrf;
      float ersys = (itr2 == errfacs.end()) ? 0 : (itr2->second).dcorr;
      float erstt = (itr->second).dcorr;
      float ertot = sqrt(erstt * erstt + ersys * ersys);
      float mean = (itr->second).corrf;
      int ieta = (itr->second).ieta;
      int depth = (itr->second).depth;
      std::cout << "[" << k << "] " << ieta << " " << depth << " " << mean << ":" << mean2 << " " << erstt << ":"
                << ersys << ":" << ertot << std::endl;
      int bin = ieta - etamin + 1;
      hists[depth - 1]->SetBinContent(bin, mean);
      hists[depth - 1]->SetBinError(bin, ertot);
    }
    TCanvas* pad = new TCanvas("CorrFactorSys", "CorrFactorSys", 700, 500);
    pad->SetRightMargin(0.10);
    pad->SetTopMargin(0.10);
    double yh = 0.90;
    double yl = yh - 0.050 * hists.size() - 0.01;
    TLegend* legend = new TLegend(0.60, yl, 0.90, yl + 0.025 * hists.size());
    legend->SetFillColor(kWhite);
    for (unsigned int k = 0; k < hists.size(); ++k) {
      if (k == 0)
        hists[k]->Draw("");
      else
        hists[k]->Draw("sames");
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hists[k]->GetListOfFunctions()->FindObject("stats");
      if (st1 != nullptr) {
        st1->SetLineColor(colors[k]);
        st1->SetTextColor(colors[k]);
        st1->SetY1NDC(yh - 0.025);
        st1->SetY2NDC(yh);
        st1->SetX1NDC(0.70);
        st1->SetX2NDC(0.90);
        yh -= 0.025;
      }
      sprintf(name, "Depth %d (%s)", k + 1, text.c_str());
      legend->AddEntry(hists[k], name, "lp");
    }
    legend->Draw("same");
    pad->Update();
    if (save) {
      sprintf(name, "%s.pdf", pad->GetName());
      pad->Print(name);
    }
  }
}

void PlotHistCorrLumis(std::string infilec, int conds, double lumi, bool save = false) {
  char fname[100];
  sprintf(fname, "%s_0.txt", infilec.c_str());
  std::map<int, cfactors> cfacs;
  int etamin(100), etamax(-100), maxdepth(0);
  readCorrFactors(fname, 1.0, cfacs, etamin, etamax, maxdepth);
  int nbin = etamax - etamin + 1;
  std::cout << "Max Depth " << maxdepth << " and " << nbin << " eta bins for " << etamin << ":" << etamax << std::endl;

  // There are good records from the master file
  int colors[8] = {4, 2, 6, 7, 1, 9, 3, 5};
  int mtype[8] = {20, 21, 22, 23, 24, 25, 26, 27};
  if (cfacs.size() > 0) {
    // Now read the other files
    std::vector<TH1D*> hists;
    char name[100];
    for (int i = 0; i < conds; ++i) {
      int ih = (int)(hists.size());
      for (int j = 0; j < maxdepth; ++j) {
        sprintf(name, "hd%d%d", j + 1, i);
        TH1D* h = new TH1D(name, name, nbin, etamin, etamax);
        h->SetLineColor(colors[j]);
        h->SetMarkerColor(colors[j]);
        h->SetMarkerStyle(mtype[i]);
        h->SetMarkerSize(0.9);
        h->GetXaxis()->SetTitle("i#eta");
        h->GetYaxis()->SetTitle("Fractional Error");
        h->GetYaxis()->SetLabelOffset(0.005);
        h->GetYaxis()->SetTitleOffset(1.20);
        h->GetYaxis()->SetRangeUser(0.0, 0.10);
        hists.push_back(h);
      }
      sprintf(fname, "%s_%d.txt", infilec.c_str(), i);
      int etamin1(100), etamax1(-100), maxdepth1(0);
      readCorrFactors(fname, 1.0, cfacs, etamin1, etamax1, maxdepth1);
      for (std::map<int, cfactors>::const_iterator itr = cfacs.begin(); itr != cfacs.end(); ++itr) {
        double value = (itr->second).dcorr / (itr->second).corrf;
        int bin = (itr->second).ieta - etamin + 1;
        hists[ih + (itr->second).depth - 1]->SetBinContent(bin, value);
        hists[ih + (itr->second).depth - 1]->SetBinError(bin, 0.0001);
      }
    }

    // Now plot the histograms
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);
    gStyle->SetFillColor(kWhite);
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    TCanvas* pad = new TCanvas("CorrFactorErr", "CorrFactorErr", 700, 500);
    pad->SetRightMargin(0.10);
    pad->SetTopMargin(0.10);
    double yh(0.89);
    TLegend* legend = new TLegend(0.60, yh - 0.04 * conds, 0.89, yh);
    legend->SetFillColor(kWhite);
    double lumic(lumi);
    for (unsigned int k = 0; k < hists.size(); ++k) {
      if (k == 0)
        hists[k]->Draw("");
      else
        hists[k]->Draw("sames");
      pad->Update();
      if (k % maxdepth == 0) {
        sprintf(name, "L = %5.2f fb^{-1}", lumic);
        legend->AddEntry(hists[k], name, "lp");
        lumic *= 0.5;
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

void PlotHistCorrRel(char* infile1, char* infile2, std::string text1, std::string text2, bool save = false) {
  std::map<int, cfactors> cfacs1, cfacs2;
  int etamin(100), etamax(-100), maxdepth(0);
  readCorrFactors(infile1, 1.0, cfacs1, etamin, etamax, maxdepth);
  readCorrFactors(infile2, 1.0, cfacs2, etamin, etamax, maxdepth);
  std::map<int, std::pair<cfactors, cfactors> > cfacs;
  for (std::map<int, cfactors>::iterator itr = cfacs1.begin(); itr != cfacs1.end(); ++itr) {
    std::map<int, cfactors>::iterator ktr = cfacs2.find(itr->first);
    if (ktr == cfacs2.end()) {
      cfactors fac2(((itr->second).ieta), ((itr->second).depth), 0, -1);
      cfacs[itr->first] = std::pair<cfactors, cfactors>((itr->second), fac2);
    } else {
      cfactors fac2(ktr->second);
      cfacs[itr->first] = std::pair<cfactors, cfactors>((itr->second), fac2);
    }
  }
  for (std::map<int, cfactors>::iterator itr = cfacs2.begin(); itr != cfacs2.end(); ++itr) {
    std::map<int, cfactors>::const_iterator ktr = cfacs1.find(itr->first);
    if (ktr == cfacs1.end()) {
      cfactors fac1(((itr->second).ieta), ((itr->second).depth), 0, -1);
      cfacs[itr->first] = std::pair<cfactors, cfactors>(fac1, (itr->second));
    }
  }

  // There are good records in bothe the files
  if ((cfacs1.size() > 0) && (cfacs2.size() > 0)) {
    int k(0);
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);
    gStyle->SetFillColor(kWhite);
    gStyle->SetOptTitle(0);
    gStyle->SetOptStat(10);
    gStyle->SetOptFit(10);
    int colors[6] = {1, 6, 4, 7, 2, 9};
    int mtype[6] = {20, 21, 22, 23, 24, 33};
    std::vector<TH1D*> hists;
    char name[100];
    int nbin = etamax - etamin + 1;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < maxdepth; ++j) {
        int j1 = (i == 0) ? j : maxdepth + j;
        sprintf(name, "hd%d%d", i, j + 1);
        TH1D* h = new TH1D(name, name, nbin, etamin, etamax);
        h->SetLineColor(colors[j1]);
        h->SetMarkerColor(colors[j1]);
        h->SetMarkerStyle(mtype[i]);
        h->GetXaxis()->SetTitle("i#eta");
        h->GetYaxis()->SetTitle("Correction Factor");
        h->GetYaxis()->SetLabelOffset(0.005);
        h->GetYaxis()->SetTitleOffset(1.20);
        h->GetYaxis()->SetRangeUser(0.0, 2.0);
        hists.push_back(h);
      }
    }
    for (std::map<int, std::pair<cfactors, cfactors> >::iterator it = cfacs.begin(); it != cfacs.end(); ++it, ++k) {
      float mean1 = (it->second).first.corrf;
      float error1 = (it->second).first.dcorr;
      float mean2 = (it->second).second.corrf;
      float error2 = (it->second).second.dcorr;
      int ieta = (it->second).first.ieta;
      int depth = (it->second).first.depth;
      /*
      std::cout << "[" << k << "] " << ieta << " " << depth << " " 
		<< mean1 << ":" << mean2 << " " << error1 << ":" << error2
		<< std::endl;
      */
      int bin = ieta - etamin + 1;
      if (error1 >= 0) {
        hists[depth - 1]->SetBinContent(bin, mean1);
        hists[depth - 1]->SetBinError(bin, error1);
      }
      if (error2 >= 0) {
        hists[maxdepth + depth - 1]->SetBinContent(bin, mean2);
        hists[maxdepth + depth - 1]->SetBinError(bin, error2);
      }
    }
    TCanvas* pad = new TCanvas("CorrFactors", "CorrFactors", 700, 500);
    pad->SetRightMargin(0.10);
    pad->SetTopMargin(0.10);
    double yh = 0.90;
    double yl = yh - 0.050 * hists.size() - 0.01;
    TLegend* legend = new TLegend(0.60, yl, 0.90, yl + 0.025 * hists.size());
    legend->SetFillColor(kWhite);
    for (unsigned int k = 0; k < hists.size(); ++k) {
      if (k == 0)
        hists[k]->Draw("");
      else
        hists[k]->Draw("sames");
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hists[k]->GetListOfFunctions()->FindObject("stats");
      if (st1 != nullptr) {
        st1->SetLineColor(colors[k]);
        st1->SetTextColor(colors[k]);
        st1->SetY1NDC(yh - 0.025);
        st1->SetY2NDC(yh);
        st1->SetX1NDC(0.70);
        st1->SetX2NDC(0.90);
        yh -= 0.025;
      }
      if (k < (unsigned int)(maxdepth)) {
        sprintf(name, "Depth %d (%s)", k + 1, text1.c_str());
      } else {
        sprintf(name, "Depth %d (%s)", k - maxdepth + 1, text2.c_str());
      }
      legend->AddEntry(hists[k], name, "lp");
    }
    legend->Draw("same");
    pad->Update();
    if (save) {
      sprintf(name, "%s.pdf", pad->GetName());
      pad->Print(name);
    }
  }
}

void PlotFourHists(std::string infile,
                   std::string prefix0,
                   int type = 0,
                   int drawStatBox = 0,
                   bool normalize = false,
                   bool save = false,
                   std::string prefix1 = "",
                   std::string text1 = "",
                   std::string prefix2 = "",
                   std::string text2 = "",
                   std::string prefix3 = "",
                   std::string text3 = "",
                   std::string prefix4 = "",
                   std::string text4 = "") {
  int colors[4] = {2, 4, 6, 1};
  std::string names[5] = {"eta03", "eta13", "eta23", "eta33", "eta43"};
  std::string xtitle[5] = {"i#eta", "i#eta", "i#eta", "i#eta", "i#eta"};
  std::string ytitle[5] = {"Tracks", "Tracks", "Tracks", "Tracks", "Tracks"};
  std::string title[5] = {"All Tracks (p = 40:60 GeV)",
                          "Good Quality Tracks (p = 40:60 GeV)",
                          "Selected Tracks (p = 40:60 GeV)",
                          "Isolated Tracks (p = 40:60 GeV)",
                          "Isolated MIP Tracks (p = 40:60 GeV)"};

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptFit(0);
  if (drawStatBox == 0)
    gStyle->SetOptStat(0);
  else
    gStyle->SetOptStat(1110);

  if (type < 0 || type > 4)
    type = 0;
  char name[100], namep[100];
  TFile* file = new TFile(infile.c_str());

  std::vector<TH1D*> hists;
  std::vector<int> kks;
  std::vector<std::string> texts;
  for (int k = 0; k < 4; ++k) {
    std::string prefix, text;
    if (k == 0) {
      prefix = prefix1;
      text = text1;
    } else if (k == 1) {
      prefix = prefix2;
      text = text2;
    } else if (k == 2) {
      prefix = prefix3;
      text = text3;
    } else {
      prefix = prefix4;
      text = text4;
    }
    if (prefix != "") {
      sprintf(name, "%s%s", prefix.c_str(), names[type].c_str());
      TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
      if (hist1 != nullptr) {
        hists.push_back((TH1D*)(hist1->Clone()));
        kks.push_back(k);
        texts.push_back(text);
      }
    }
  }
  if (hists.size() > 0) {
    sprintf(namep, "c_%s%s", prefix0.c_str(), names[type].c_str());
    double ymax(0.90), dy(0.13);
    double ymx0 = (drawStatBox == 0) ? (ymax - .01) : (ymax - dy * hists.size() - .01);
    TCanvas* pad = new TCanvas(namep, namep, 700, 500);
    TLegend* legend = new TLegend(0.64, ymx0 - 0.05 * hists.size(), 0.89, ymx0);
    legend->SetFillColor(kWhite);
    pad->SetRightMargin(0.10);
    pad->SetTopMargin(0.10);
    for (unsigned int jk = 0; jk < hists.size(); ++jk) {
      int k = kks[jk];
      hists[jk]->GetXaxis()->SetTitleSize(0.040);
      hists[jk]->GetXaxis()->SetTitle(xtitle[type].c_str());
      hists[jk]->GetYaxis()->SetTitle(ytitle[type].c_str());
      hists[jk]->GetYaxis()->SetLabelOffset(0.005);
      hists[jk]->GetYaxis()->SetLabelSize(0.035);
      hists[jk]->GetYaxis()->SetTitleSize(0.040);
      hists[jk]->GetYaxis()->SetTitleOffset(1.15);
      hists[jk]->SetMarkerStyle(20);
      hists[jk]->SetMarkerColor(colors[k]);
      hists[jk]->SetLineColor(colors[k]);
      if (normalize) {
        if (jk == 0)
          hists[jk]->DrawNormalized();
        else
          hists[jk]->DrawNormalized("sames");
      } else {
        if (jk == 0)
          hists[jk]->Draw();
        else
          hists[jk]->Draw("sames");
      }
      pad->Update();
      TPaveStats* st1 = (TPaveStats*)hists[jk]->GetListOfFunctions()->FindObject("stats");
      if (st1 != nullptr) {
        double ymin = ymax - dy;
        st1->SetLineColor(colors[k]);
        st1->SetTextColor(colors[k]);
        st1->SetY1NDC(ymin);
        st1->SetY2NDC(ymax);
        st1->SetX1NDC(0.70);
        st1->SetX2NDC(0.90);
        ymax = ymin;
      }
      sprintf(name, "%s", texts[jk].c_str());
      legend->AddEntry(hists[jk], name, "lp");
    }
    legend->Draw("same");
    pad->Update();
    TPaveText* txt1 = new TPaveText(0.10, 0.905, 0.80, 0.95, "blNDC");
    txt1->SetFillColor(0);
    char txt[100];
    sprintf(txt, "%s", title[type].c_str());
    txt1->AddText(txt);
    txt1->Draw("same");
    /*
    TPaveText *txt2 = new TPaveText(0.11,0.825,0.33,0.895,"blNDC");
    txt2->SetFillColor(0);
    sprintf (txt, "CMS Preliminary");
    txt2->AddText(txt);
    txt2->Draw("same");
    */
    pad->Modified();
    pad->Update();
    if (save) {
      sprintf(name, "%s.pdf", pad->GetName());
      pad->Print(name);
    }
  }
}

void PlotPUCorrHists(std::string infile = "corrfac.root",
                     std::string prefix = "",
                     int drawStatBox = 0,
                     bool approve = true,
                     bool save = false) {
  std::string name1[4] = {"W0", "W1", "W2", "P"};
  std::string name2[4] = {"All", "Barrel", "Endcap", ""};
  std::string name3[2] = {"", "p = 40:60 GeV"};
  std::string name4[2] = {"Loose Isolation", "Tight Isolation"};
  std::string xtitle[4] = {"Correction Factor", "Correction Factor", "Correction Factor", "i#eta"};
  std::string ytitle[4] = {"Tracks", "Tracks", "Tracks", "Correction Factor"};

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptFit(0);
  if (drawStatBox == 0)
    gStyle->SetOptStat(0);
  else
    gStyle->SetOptStat(1110);

  char name[100], namep[100], title[100];
  TFile* file = new TFile(infile.c_str());

  if (file != nullptr) {
    for (int i1 = 0; i1 < 4; ++i1) {
      for (int i2 = 0; i2 < 2; ++i2) {
        for (int i3 = 0; i3 < 2; ++i3) {
          sprintf(name, "%s%d%d", name1[i1].c_str(), i2, i3);
          if (i2 == 0)
            sprintf(title, "%s Tracks Selected with %s", name2[i1].c_str(), name4[i3].c_str());
          else
            sprintf(title, "%s Tracks Selected with %s (%s)", name2[i1].c_str(), name4[i3].c_str(), name3[i2].c_str());
          TH1D* hist1(nullptr);
          TProfile* hist2(nullptr);
          if (i1 != 3) {
            TH1D* hist = (TH1D*)file->FindObjectAny(name);
            if (hist != nullptr) {
              hist1 = (TH1D*)(hist->Clone());
              hist1->GetXaxis()->SetTitleSize(0.040);
              hist1->GetXaxis()->SetTitle(xtitle[i1].c_str());
              hist1->GetYaxis()->SetTitle(ytitle[i1].c_str());
              hist1->GetYaxis()->SetLabelOffset(0.005);
              hist1->GetYaxis()->SetLabelSize(0.035);
              hist1->GetYaxis()->SetTitleSize(0.040);
              hist1->GetYaxis()->SetTitleOffset(1.15);
            }
          } else {
            TProfile* hist = (TProfile*)file->FindObjectAny(name);
            if (hist != nullptr) {
              hist2 = (TProfile*)(hist->Clone());
              hist2->GetXaxis()->SetTitleSize(0.040);
              hist2->GetXaxis()->SetTitle(xtitle[i1].c_str());
              hist2->GetYaxis()->SetTitle(ytitle[i1].c_str());
              hist2->GetYaxis()->SetLabelOffset(0.005);
              hist2->GetYaxis()->SetLabelSize(0.035);
              hist2->GetYaxis()->SetTitleSize(0.040);
              hist2->GetYaxis()->SetTitleOffset(1.15);
              //	      hist2->GetYaxis()->SetRangeUser(0.0, 1.5);
              hist2->SetMarkerStyle(20);
            }
          }
          if ((hist1 != nullptr) || (hist2 != nullptr)) {
            sprintf(namep, "c_%s%s", name, prefix.c_str());
            TCanvas* pad = new TCanvas(namep, namep, 700, 500);
            pad->SetRightMargin(0.10);
            pad->SetTopMargin(0.10);
            if (hist1 != nullptr) {
              pad->SetLogy();
              hist1->Draw();
              pad->Update();
              TPaveStats* st1 = (TPaveStats*)hist1->GetListOfFunctions()->FindObject("stats");
              if (st1 != nullptr) {
                st1->SetY1NDC(0.77);
                st1->SetY2NDC(0.90);
                st1->SetX1NDC(0.70);
                st1->SetX2NDC(0.90);
              }
            } else {
              hist2->Draw();
              pad->Update();
            }
            TPaveText* txt1 = new TPaveText(0.10, 0.905, 0.80, 0.95, "blNDC");
            txt1->SetFillColor(0);
            char txt[100];
            sprintf(txt, "%s", title);
            txt1->AddText(txt);
            txt1->Draw("same");
            if (approve) {
              double xoff = (i1 == 3) ? 0.11 : 0.22;
              TPaveText* txt2 = new TPaveText(xoff, 0.825, xoff + 0.22, 0.895, "blNDC");
              txt2->SetFillColor(0);
              sprintf(txt, "CMS Preliminary");
              txt2->AddText(txt);
              txt2->Draw("same");
            }
            pad->Modified();
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

void PlotHistCorr(const char* infile,
                  std::string prefix,
                  std::string text0,
                  int eta = 0,
                  int mode = 1,
                  bool drawStatBox = true,
                  bool save = false) {
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if (drawStatBox)
    gStyle->SetOptStat(1100);
  else
    gStyle->SetOptStat(0);

  std::string tags[3] = {"UnNoPU", "UnPU", "Cor"};
  std::string text[3] = {"Uncorrected no PU", "Uncorrected PU", "Corrected PU"};
  int colors[3] = {1, 4, 2};
  int styles[3] = {1, 3, 2};
  TFile* file = new TFile(infile);
  if (mode < 0 || mode > 2)
    mode = 1;
  int etamin = (eta == 0) ? -27 : eta;
  int etamax = (eta == 0) ? 27 : eta;
  for (int ieta = etamin; ieta <= etamax; ++ieta) {
    char name[20];
    double yh(0.90), dy(0.09);
    double yh1 = drawStatBox ? (yh - 3 * dy - 0.01) : (yh - 0.01);
    TLegend* legend = new TLegend(0.55, yh1 - 0.15, 0.89, yh1);
    legend->SetFillColor(kWhite);
    sprintf(name, "c_%sEovp%d", prefix.c_str(), ieta);
    TCanvas* pad = new TCanvas(name, name, 700, 500);
    pad->SetRightMargin(0.10);
    pad->SetTopMargin(0.10);
    TH1D* hist[3];
    double ymax(0);
    for (int k = 0; k < 3; ++k) {
      if (k < 2)
        sprintf(name, "EovP_ieta%d%s", ieta, tags[k].c_str());
      else
        sprintf(name, "EovP_ieta%dCor%dPU", ieta, mode);
      TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
      if (hist1 != nullptr) {
        hist[k] = (TH1D*)(hist1->Clone());
        ymax = std::max(ymax, (hist1->GetMaximum()));
      }
    }
    int imax = 10 * (2 + int(0.1 * ymax));
    for (int k = 0; k < 3; ++k) {
      hist[k]->GetYaxis()->SetLabelOffset(0.005);
      hist[k]->GetYaxis()->SetTitleOffset(1.20);
      hist[k]->GetXaxis()->SetTitle("E/p");
      hist[k]->GetYaxis()->SetTitle("Tracks");
      hist[k]->SetLineColor(colors[k]);
      hist[k]->SetLineStyle(styles[k]);
      hist[k]->GetYaxis()->SetRangeUser(0.0, imax);
      if (k == 0)
        hist[k]->Draw();
      else
        hist[k]->Draw("sames");
      legend->AddEntry(hist[k], text[k].c_str(), "lp");
      pad->Update();
      if (drawStatBox) {
        TPaveStats* st1 = (TPaveStats*)hist[k]->GetListOfFunctions()->FindObject("stats");
        if (st1 != nullptr) {
          st1->SetLineColor(colors[k]);
          st1->SetTextColor(colors[k]);
          st1->SetY1NDC(yh - dy);
          st1->SetY2NDC(yh);
          st1->SetX1NDC(0.70);
          st1->SetX2NDC(0.90);
          yh -= dy;
        }
      }
    }
    pad->Update();
    legend->Draw("same");
    pad->Update();
    TPaveText* txt1 = new TPaveText(0.10, 0.905, 0.80, 0.95, "blNDC");
    txt1->SetFillColor(0);
    char title[100];
    sprintf(title, "%s for i#eta = %d", text0.c_str(), ieta);
    txt1->AddText(title);
    txt1->Draw("same");
    pad->Modified();
    pad->Update();
    if (save) {
      sprintf(name, "%s.pdf", pad->GetName());
      pad->Print(name);
    }
  }
}

void PlotPropertyHist(const char* infile,
                      std::string prefix,
                      std::string text,
                      int etaMax = 25,
                      double lumi = 0,
                      double ener = 13.0,
                      bool dataMC = false,
                      bool drawStatBox = true,
                      bool save = false) {
  std::string name0[3] = {"energyE2", "energyH2", "energyP2"};
  std::string title0[3] = {"Energy in ECAL", "Energy in HCAL", "Track Momentum"};
  std::string xtitl0[3] = {"Energy (GeV)", "Energy (GeV)", "p (GeV)"};
  std::string name1[5] = {"eta02", "eta12", "eta22", "eta32", "eta42"};
  std::string name10[5] = {"eta0", "eta1", "eta2", "eta3", "eta4"};
  std::string xtitl1 = "i#eta";
  std::string name2[5] = {"p0", "p1", "p2", "p3", "p4"};
  std::string xtitl2 = "p (GeV)";
  std::string title1[5] = {"Tracks with p=40:60 GeV",
                           "Good Quality Tracks with p=40:60 GeV",
                           "Selected Tracks with p=40:60 GeV",
                           "Isolated Tracks with p=40:60 GeV",
                           "Isolated Tracks with p=40:60 GeV and MIPS in ECAL"};
  std::string title2[5] = {
      "All Tracks", "Good Quality Tracks", "Selected Tracks", "Isolated Tracks", "Isolated Tracks with MIPS in ECAL"};
  std::string ytitle = "Tracks";

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if (drawStatBox)
    gStyle->SetOptStat(1110);
  else
    gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);

  TFile* file = new TFile(infile);
  char name[100], namep[100];
  for (int k = 1; k <= etaMax; ++k) {
    for (int j = 0; j < 3; ++j) {
      sprintf(name, "%s%s%d", prefix.c_str(), name0[j].c_str(), k);
      TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
      if (hist1 != nullptr) {
        TH1D* hist = (TH1D*)(hist1->Clone());
        double ymin(0.90);
        sprintf(namep, "c_%s", name);
        TCanvas* pad = new TCanvas(namep, namep, 700, 500);
        pad->SetLogy();
        pad->SetRightMargin(0.10);
        pad->SetTopMargin(0.10);
        hist->GetXaxis()->SetTitleSize(0.04);
        hist->GetXaxis()->SetTitle(xtitl0[j].c_str());
        hist->GetYaxis()->SetTitle(ytitle.c_str());
        hist->GetYaxis()->SetLabelOffset(0.005);
        hist->GetYaxis()->SetTitleSize(0.04);
        hist->GetYaxis()->SetLabelSize(0.035);
        hist->GetYaxis()->SetTitleOffset(1.10);
        hist->SetMarkerStyle(20);
        hist->Draw();
        pad->Update();
        TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
        if (st1 != nullptr) {
          st1->SetY1NDC(0.78);
          st1->SetY2NDC(0.90);
          st1->SetX1NDC(0.65);
          st1->SetX2NDC(0.90);
        }
        pad->Update();
        double ymx(0.96), xmi(0.35), xmx(0.95);
        char txt[100];
        if (lumi > 0.1) {
          ymx = ymin - 0.005;
          xmi = 0.45;
          TPaveText* txt0 = new TPaveText(0.65, 0.91, 0.90, 0.96, "blNDC");
          txt0->SetFillColor(0);
          sprintf(txt, "%4.1f TeV %5.1f fb^{-1}", ener, lumi);
          txt0->AddText(txt);
          txt0->Draw("same");
        }
        double ymi = ymx - 0.05;
        TPaveText* txt1 = new TPaveText(xmi, ymi, xmx, ymx, "blNDC");
        txt1->SetFillColor(0);
        if (text == "") {
          sprintf(txt, "%s", title0[j].c_str());
        } else {
          sprintf(txt, "%s (%s)", title0[j].c_str(), text.c_str());
        }
        txt1->AddText(txt);
        txt1->Draw("same");
        double xmax = (dataMC) ? 0.24 : 0.35;
        ymi = 0.91;
        ymx = ymi + 0.05;
        TPaveText* txt2 = new TPaveText(0.02, ymi, xmax, ymx, "blNDC");
        txt2->SetFillColor(0);
        if (dataMC)
          sprintf(txt, "CMS Preliminary");
        else
          sprintf(txt, "CMS Simulation Preliminary");
        txt2->AddText(txt);
        txt2->Draw("same");
        pad->Modified();
        pad->Update();
        if (save) {
          sprintf(name, "%s.pdf", pad->GetName());
          pad->Print(name);
        }
      }
    }
  }

  for (int k = 0; k < 3; ++k) {
    for (int j = 0; j < 5; ++j) {
      if (k == 0)
        sprintf(name, "%s%s", prefix.c_str(), name1[j].c_str());
      else if (k == 1)
        sprintf(name, "%s%s", prefix.c_str(), name10[j].c_str());
      else
        sprintf(name, "%s%s", prefix.c_str(), name2[j].c_str());
      TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
      if (hist1 != nullptr) {
        TH1D* hist = (TH1D*)(hist1->Clone());
        double ymin(0.90);
        sprintf(namep, "c_%s", name);
        TCanvas* pad = new TCanvas(namep, namep, 700, 500);
        pad->SetLogy();
        pad->SetRightMargin(0.10);
        pad->SetTopMargin(0.10);
        hist->GetXaxis()->SetTitleSize(0.04);
        if (k <= 1)
          hist->GetXaxis()->SetTitle(xtitl1.c_str());
        else
          hist->GetXaxis()->SetTitle(xtitl2.c_str());
        hist->GetYaxis()->SetTitle(ytitle.c_str());
        hist->GetYaxis()->SetLabelOffset(0.005);
        hist->GetYaxis()->SetTitleSize(0.04);
        hist->GetYaxis()->SetLabelSize(0.035);
        hist->GetYaxis()->SetTitleOffset(1.10);
        hist->SetMarkerStyle(20);
        hist->Draw();
        pad->Update();
        TPaveStats* st1 = (TPaveStats*)hist->GetListOfFunctions()->FindObject("stats");
        if (st1 != nullptr) {
          st1->SetY1NDC(0.78);
          st1->SetY2NDC(0.90);
          st1->SetX1NDC(0.65);
          st1->SetX2NDC(0.90);
        }
        pad->Update();
        double ymx(0.96), xmi(0.35), xmx(0.95);
        char txt[100];
        if (lumi > 0.1) {
          ymx = ymin - 0.005;
          xmi = 0.45;
          TPaveText* txt0 = new TPaveText(0.65, 0.91, 0.90, 0.96, "blNDC");
          txt0->SetFillColor(0);
          sprintf(txt, "%4.1f TeV %5.1f fb^{-1}", ener, lumi);
          txt0->AddText(txt);
          txt0->Draw("same");
        }
        double ymi = ymx - 0.05;
        TPaveText* txt1 = new TPaveText(xmi, ymi, xmx, ymx, "blNDC");
        txt1->SetFillColor(0);
        if (text == "") {
          if (k == 0)
            sprintf(txt, "%s", title1[j].c_str());
          else
            sprintf(txt, "%s", title2[j].c_str());
        } else {
          if (k == 0)
            sprintf(txt, "%s (%s)", title1[j].c_str(), text.c_str());
          else
            sprintf(txt, "%s (%s)", title2[j].c_str(), text.c_str());
        }
        txt1->AddText(txt);
        txt1->Draw("same");
        double xmax = (dataMC) ? 0.24 : 0.35;
        ymi = 0.91;
        ymx = ymi + 0.05;
        TPaveText* txt2 = new TPaveText(0.02, ymi, xmax, ymx, "blNDC");
        txt2->SetFillColor(0);
        if (dataMC)
          sprintf(txt, "CMS Preliminary");
        else
          sprintf(txt, "CMS Simulation Preliminary");
        txt2->AddText(txt);
        txt2->Draw("same");
        pad->Modified();
        pad->Update();
        if (save) {
          sprintf(name, "%s.pdf", pad->GetName());
          pad->Print(name);
        }
      }
    }
  }
}
