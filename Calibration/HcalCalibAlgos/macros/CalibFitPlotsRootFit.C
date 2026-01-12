//////////////////////////////////////////////////////////////////////////////
// Usage:
// .L CalibFitPlotsRooFit.C+g
//
//             For extended set of histograms from CalibMonitor using RooFit
//  FitHistExtended_RootFit(infile, outfile, prefix, numb, type, append,
//			    fiteta, iname, debug);
//      Defaults: numb=54, type=13, append=true, fiteta=true, iname=3,
//                debug=false
//
//             For plotting stird histograms from FitHistExtended_RootFit
//  PlotHist_RooFit(infile, prefix, text, modePlot, kopt, lumi, ener,
//		    isRealData, drawStatBox, save, debug);
//      Defaults: modePlot=4, kopt=100, lumi=0, ener="13.6", isRealData=false,
//                drawStatBox=true, save=0, debug=false
//
//
//  where:
//  infile   (std::string)  = Name of the input ROOT file
//  outfile  (std::string)  = Name of the output ROOT file
//  prefix   (std::string)  = Prefix for the histogram names
//  mode     (int)          = Flag to check which set of histograms to be
//                            done. It has the format lthdo where each of
//                            l,t,h,d,o can have a value 0 or 1 to select
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
//  ieta     (int)          = specific ieta histogram to be plotted; if 0
//                            histograms for all ieta's from -numb/2 to numb/2
//                            will be plotted
//  lumi     (double)       = Integrated luminosity of the dataset used which
//                            needs to be drawn on the top of the canvas
//                            along with CM energy (if lumi > 0)
//  ener     (std::string)  = CM energy of the dataset used
//  isRealData  (bool)      = Flag to show if Real/simulated data
//  drawStatBox (bool)      = set to show the statistical box
//  save     (int)          = if > 0 it saves the canvas as a pdf file; or
//                            if < 0 it saves the canvas as a C file
//////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include <TCanvas.h>
#include <TChain.h>
#include <TProfile.h>
#include <TF1.h>
#include <TFile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TH1D.h>
#include <TLegend.h>
#include <TLine.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TGraphAsymmErrors.h>
#include <TMath.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <TROOT.h>
#include <TStyle.h>

#include "CalibCorr.C"
#include "RooRealVar.h"
#include "RooDataHist.h"
#include "RooFitResult.h"
#include "RooPlot.h"
#include "RooFit.h"
#include "RooWorkspace.h"
#include "RooAbsPdf.h"
#include "RooGaussian.h"
#include "RooLandau.h"
#include "RooFFTConvPdf.h"
#include "RooAddPdf.h"
#include "RooPolyVar.h"
#include "RooPolynomial.h"
#include "RooRealProxy.h"
#include "RooAbsReal.h"
#include "RooRealConstant.h"
#include "RooTrace.h"

#include "CalibGBRMath.h"

using namespace RooFit;

const double fitrangeFactor = 1.5;
const double fitrangeFactor1 = 1.2;

struct cfactors {
  int ieta, depth;
  double corrf, dcorr;
  cfactors(int ie = 0, int dp = 0, double cor = 1, double dc = 0) : ieta(ie), depth(dp), corrf(cor), dcorr(dc) {};
};

struct results {
  double mean, errmean, width, errwidth;
  results(double v1 = 0, double er1 = 0, double v2 = 0, double er2 = 0)
      : mean(v1), errmean(er1), width(v2), errwidth(er2) {};
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

class RooDoubleCBFast : public RooAbsPdf {
public:
  RooDoubleCBFast();
  RooDoubleCBFast(const char* name,
                  const char* title,
                  RooAbsReal& _x,
                  RooAbsReal& _mean,
                  RooAbsReal& _width,
                  RooAbsReal& _alpha1,
                  RooAbsReal& _n1,
                  RooAbsReal& _alpha2,
                  RooAbsReal& _n2);

  RooDoubleCBFast(const RooDoubleCBFast& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooDoubleCBFast(*this, newname); }
  inline virtual ~RooDoubleCBFast() {}
  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName = 0) const;
  Double_t analyticalIntegral(Int_t code, const char* rangeName = 0) const;

protected:
  RooRealProxy x;
  RooRealProxy mean;
  RooRealProxy width;
  RooRealProxy alpha1;
  RooRealProxy n1;
  RooRealProxy alpha2;
  RooRealProxy n2;

  Double_t evaluate() const;

private:
  ClassDef(RooDoubleCBFast, 1)
};

RooDoubleCBFast::RooDoubleCBFast() { TRACE_CREATE };

RooDoubleCBFast::RooDoubleCBFast(const char* name,
                                 const char* title,
                                 RooAbsReal& _x,
                                 RooAbsReal& _mean,
                                 RooAbsReal& _width,
                                 RooAbsReal& _alpha1,
                                 RooAbsReal& _n1,
                                 RooAbsReal& _alpha2,
                                 RooAbsReal& _n2)
    : RooAbsPdf(name, title),
      x("x", "x", this, _x),
      mean("mean", "mean", this, _mean),
      width("width", "width", this, _width),
      alpha1("alpha1", "alpha1", this, _alpha1),
      n1("n1", "n1", this, _n1),
      alpha2("alpha2", "alpha2", this, _alpha2),
      n2("n2", "n2", this, _n2) {}

RooDoubleCBFast::RooDoubleCBFast(const RooDoubleCBFast& other, const char* name)
    : RooAbsPdf(other, name),
      x("x", this, other.x),
      mean("mean", this, other.mean),
      width("width", this, other.width),
      alpha1("alpha1", this, other.alpha1),
      n1("n1", this, other.n1),
      alpha2("alpha2", this, other.alpha2),
      n2("n2", this, other.n2) {}

double RooDoubleCBFast::evaluate() const {
  double t = (x - mean) * vdt::fast_inv(width);
  double val = -99.;
  if (t > -alpha1 && t < alpha2) {
    val = vdt::fast_exp(-0.5 * t * t);
  } else if (t <= -alpha1) {
    double alpha1invn1 = alpha1 * vdt::fast_inv(n1);
    val = vdt::fast_exp(-0.5 * alpha1 * alpha1) * gbrmath::fast_pow(1. - alpha1invn1 * (alpha1 + t), -n1);
  } else if (t >= alpha2) {
    double alpha2invn2 = alpha2 * vdt::fast_inv(n2);
    val = vdt::fast_exp(-0.5 * alpha2 * alpha2) * gbrmath::fast_pow(1. - alpha2invn2 * (alpha2 - t), -n2);
  }
  if (!std::isnormal(val)) {
    printf("bad val: x = %5f, t = %5f, mean = %5f, sigma = %5f, alpha1 = %5f, n1 = %5f, alpha2 = %5f, n2 = %5f\n",
           double(x),
           t,
           double(mean),
           double(width),
           double(alpha1),
           double(n1),
           double(alpha2),
           double(n2));
    printf("val = %5f\n", val);
  }

  return val;
}

Int_t RooDoubleCBFast::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* range) const {
  if (matchArgs(allVars, analVars, x))
    return 1;
  return 0;
}

Double_t RooDoubleCBFast::analyticalIntegral(Int_t code, const char* rangeName) const {
  assert(code == 1);

  double central = 0;
  double left = 0;
  double right = 0;

  double xmin = x.min(rangeName);
  double xmax = x.max(rangeName);

  static const double rootPiBy2 = sqrt(atan2(0.0, -1.0) / 2.0);
  static const double invRoot2 = 1.0 / sqrt(2);

  double invwidth = vdt::fast_inv(width);

  double tmin = (xmin - mean) * invwidth;
  double tmax = (xmax - mean) * invwidth;

  bool isfullrange = (tmin < -1000. && tmax > 1000.);

  //compute gaussian contribution
  double central_low = std::max(xmin, mean - alpha1 * width);
  double central_high = std::min(xmax, mean + alpha2 * width);

  double tcentral_low = (central_low - mean) * invwidth;
  double tcentral_high = (central_high - mean) * invwidth;
  if (central_low < central_high) {  // is the gaussian part in range?
    central = rootPiBy2 * width * (TMath::Erf(tcentral_high * invRoot2) - TMath::Erf(tcentral_low * invRoot2));
  }
  //compute left tail;
  if (isfullrange && (n1 - 1.0) > 1.e-5) {
    left = width * vdt::fast_exp(-0.5 * alpha1 * alpha1) * n1 * vdt::fast_inv(alpha1 * (n1 - 1.));
  } else {
    double left_low = xmin;
    double left_high = std::min(xmax, mean - alpha1 * width);
    double thigh = (left_high - mean) * invwidth;

    if (left_low < left_high) {  //is the left tail in range?
      double n1invalpha1 = n1 * vdt::fast_inv(fabs(alpha1));
      if (fabs(n1 - 1.0) > 1.e-5) {
        double invn1m1 = vdt::fast_inv(n1 - 1.);
        double leftpow = gbrmath::fast_pow(n1invalpha1, -n1 * invn1m1);
        double left0 = width * vdt::fast_exp(-0.5 * alpha1 * alpha1) * invn1m1;
        double left1, left2;

        if (xmax > (mean - alpha1 * width))
          left1 = n1invalpha1;
        else
          left1 = gbrmath::fast_pow(leftpow * (n1invalpha1 - alpha1 - thigh), 1. - n1);

        if (tmin < -1000.)
          left2 = 0.;
        else
          left2 = gbrmath::fast_pow(leftpow * (n1invalpha1 - alpha1 - tmin), 1. - n1);

        left = left0 * (left1 - left2);

      } else {
        double A1 = gbrmath::fast_pow(n1invalpha1, n1) * vdt::fast_exp(-0.5 * alpha1 * alpha1);
        double B1 = n1invalpha1 - fabs(alpha1);
        left = A1 * width *
               (vdt::fast_log(B1 - (left_low - mean) * invwidth) - vdt::fast_log(B1 - (left_high - mean) * invwidth));
      }
    }
  }

  //compute right tail;
  if (isfullrange && (n2 - 1.0) > 1.e-5) {
    right = width * vdt::fast_exp(-0.5 * alpha2 * alpha2) * n2 * vdt::fast_inv(alpha2 * (n2 - 1.));
  } else {
    double right_low = std::max(xmin, mean + alpha2 * width);
    double right_high = xmax;
    double tlow = (right_low - mean) * invwidth;

    if (right_low < right_high) {  //is the right tail in range?
      double n2invalpha2 = n2 * vdt::fast_inv(fabs(alpha2));
      if (fabs(n2 - 1.0) > 1.e-5) {
        double invn2m2 = vdt::fast_inv(n2 - 1.);
        double rightpow = gbrmath::fast_pow(n2invalpha2, -n2 * invn2m2);
        double right0 = width * vdt::fast_exp(-0.5 * alpha2 * alpha2) * invn2m2;
        double right1, right2;

        if (xmin < (mean + alpha2 * width))
          right1 = n2invalpha2;
        else
          right1 = gbrmath::fast_pow(rightpow * (n2invalpha2 - alpha2 + tlow), 1. - n2);

        if (tmax > 1000.)
          right2 = 0.;
        else
          right2 = gbrmath::fast_pow(rightpow * (n2invalpha2 - alpha2 + tmax), 1. - n2);

        right = right0 * (right1 - right2);

      } else {
        double A2 = gbrmath::fast_pow(n2invalpha2, n2) * vdt::fast_exp(-0.5 * alpha2 * alpha2);
        double B2 = n2invalpha2 - fabs(alpha2);
        right =
            A2 * width *
            (vdt::fast_log(B2 + (right_high - mean) * invwidth) - vdt::fast_log(B2 + (right_low - mean) * invwidth));
      }
    }
  }

  double sum = left + central + right;

  if (!std::isnormal(sum)) {
    printf("bad int: mean = %5f, sigma = %5f, alpha1 = %5f, n1 = %5f, alpha2 = %5f, n2 = %5f\n",
           double(mean),
           double(width),
           double(alpha1),
           double(n1),
           double(alpha2),
           double(n2));
    printf("left = %5f, central = %5f, right = %5f, integral = %5f\n", left, central, right, sum);
  }

  return sum;
}

results fitDoubleSidedCrystalball_RooFit(TH1D* hist, bool debug, RooWorkspace* ws) {
  const double fitrangeFactor = 2.0;
  const double fitrangeFactor1 = 1.5;

  double rms0;
  auto meanPair = GetMean(hist, 0.2, 2.0, rms0);
  double mean0 = meanPair.first;

  double LowEdge = std::max(0.5, mean0 - fitrangeFactor * rms0);
  double diff = mean0 - LowEdge;
  double HighEdge = mean0 + std::min(fitrangeFactor1 * rms0, diff);
  HighEdge = std::min(HighEdge, hist->GetXaxis()->GetXmax());

  if (debug) {
    std::cout << hist->GetName() << " initial Mean=" << mean0 << " RMS=" << rms0 << " Range=[" << LowEdge << ","
              << HighEdge << "]\n";
  }

  RooRealVar x("x", "x", hist->GetXaxis()->GetXmin(), hist->GetXaxis()->GetXmax());
  RooDataHist data("data", "dataset from TH1", x, Import(*hist));

  RooRealVar alpha1("alpha1", "alpha low tail", 1.0, 0.1, 5.0);
  RooRealVar n1("n1", "power low tail", 2.0, 0.1, 5.0);
  RooRealVar alpha2("alpha2", "alpha high tail", 1.5, 0.1, 5.0);
  RooRealVar n2("n2", "power high tail", 3.0, 0.1, 5.0);
  RooRealVar mean("mean", "peak position", mean0, hist->GetXaxis()->GetXmin(), hist->GetXaxis()->GetXmax());
  RooRealVar sigma(
      "sigma", "Gaussian sigma", rms0, 0.01, (hist->GetXaxis()->GetXmax() - hist->GetXaxis()->GetXmin()) / 2.);
  RooRealVar norm("norm", "signal yield", hist->Integral(), 0.0, hist->GetEntries() * 2);

  RooDoubleCBFast pdf("dblCB", "double-sided Crystal Ball (fast)", x, mean, sigma, alpha1, n1, alpha2, n2);

  if (!debug) {
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    RooMsgService::instance().setSilentMode(true);
  }

  RooFitResult* fitRes = pdf.fitTo(data, Save(true), PrintLevel(debug ? 1 : -1), Range(LowEdge, HighEdge));

  if (ws) {
    std::string dataName = std::string(hist->GetName()) + "_data";
    std::string pdfName = std::string(hist->GetName()) + "_pdf";
    std::string fitResName = std::string(hist->GetName()) + "_fitRes";
    data.SetName(dataName.c_str());
    pdf.SetName(pdfName.c_str());

    RooRealVar lowRangeVar(Form("%s_low", hist->GetName()), "fit lower edge", LowEdge);
    RooRealVar highRangeVar(Form("%s_high", hist->GetName()), "fit upper edge", HighEdge);
    ws->import(lowRangeVar);
    ws->import(highRangeVar);
    fitRes->SetName(fitResName.c_str());
    ws->import(*fitRes);
    ws->import(data);
    ws->import(pdf);
  }

  double mean_val = mean.getValV();
  double mean_err = mean.getError();
  double sigma_val = sigma.getValV();
  double sigma_err = sigma.getError();

  if (mean_val < hist->GetXaxis()->GetXmin() || mean_val > hist->GetXaxis()->GetXmax() ||
      mean_err > fabs(mean_val) * 0.5) {
    double rms_fallback;
    auto mm = GetMean(hist, 0.2, 2.0, rms_fallback);
    auto ww = GetWidth(hist, 0.2, 2.0);
    mean_val = mm.first;
    mean_err = mm.second;
    sigma_val = ww.first;
    sigma_err = ww.second;
  }

  return results(mean_val, mean_err, sigma_val, sigma_err);
}

// Landau-Gauss convolution fit
std::pair<double, double> fitLanGau_RooFit(TH1D* hist, bool debug, RooWorkspace* ws) {
  double rms0;
  auto mm = GetMean(hist, 0.005, 2.5, rms0);
  double mean0 = mm.first;

  RooRealVar x("x", "x", hist->GetXaxis()->GetXmin(), hist->GetXaxis()->GetXmax());
  RooDataHist data("data", "dataset from TH1", x, Import(*hist));

  RooRealVar lanMPV("lanMPV", "Landau MPV", mean0, 0.0, hist->GetXaxis()->GetXmax());
  RooRealVar lanSigma("lanSigma", "Landau width", rms0, 0.001, hist->GetXaxis()->GetXmax());
  RooLandau landau("landau", "Landau component", x, lanMPV, lanSigma);

  RooRealVar gaussSigma("gaussSigma", "Gaussian sigma", rms0, 0.001, hist->GetXaxis()->GetXmax());
  RooGaussian gauss("gauss", "Gaussian component", x, lanMPV, gaussSigma);

  RooFFTConvPdf convPdf("langau", "Landau ↔ Gaussian Convolution", x, landau, gauss);

  double LowEdge = 0.005;
  double HighEdge = mean0 + 3 * rms0;

  if (!debug) {
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    RooMsgService::instance().setSilentMode(true);
  }

  RooFitResult* fitRes = convPdf.fitTo(data, Save(true), PrintLevel(debug ? 1 : -1), Range(LowEdge, HighEdge));

  if (ws) {
    data.SetName((std::string(hist->GetName()) + "_data").c_str());
    convPdf.SetName((std::string(hist->GetName()) + "_pdf").c_str());
    ws->import(data);
    ws->import(convPdf);
    RooRealVar lowRangeVar(Form("%s_low", hist->GetName()), "fit lower edge", LowEdge);
    RooRealVar highRangeVar(Form("%s_high", hist->GetName()), "fit upper edge", HighEdge);
    ws->import(lowRangeVar);
    ws->import(highRangeVar);
    std::string fitResName = std::string(hist->GetName()) + "_fitRes";
    fitRes->SetName(fitResName.c_str());
    ws->import(*fitRes);
  }

  return {lanMPV.getValV(), lanMPV.getError()};
}

// 2) Two-Gaussian mixture fit
results fitTwoGauss_RooFit(TH1D* hist, bool debug, RooWorkspace* ws) {
  double rms;
  std::pair<double, double> mrms = GetMean(hist, 0.2, 2.0, rms);
  double mean = mrms.first;
  double LowEdge = mean - fitrangeFactor * rms;
  double HighEdge = mean + fitrangeFactor1 * rms;
  if (LowEdge < 0.15)
    LowEdge = 0.15;

  RooRealVar x("x", "x", hist->GetXaxis()->GetXmin(), hist->GetXaxis()->GetXmax());
  RooDataHist data("data", "dataset from TH1", x, Import(*hist));

  RooRealVar mean1("mean1", "mean1", mean, hist->GetXaxis()->GetXmin(), hist->GetXaxis()->GetXmax());
  RooRealVar sigma1("sigma1", "sigma1", rms, 0.001, hist->GetXaxis()->GetXmax());
  RooGaussian gauss1("gauss1", "gauss1", x, mean1, sigma1);
  RooRealVar norm1("norm1", "norm1", hist->GetEntries() * 0.8, 0, hist->GetEntries());

  RooRealVar mean2("mean2", "mean2", mean, hist->GetXaxis()->GetXmin(), hist->GetXaxis()->GetXmax());
  RooRealVar sigma2("sigma2", "sigma2", 2 * rms, 0.001, hist->GetXaxis()->GetXmax());
  RooGaussian gauss2("gauss2", "gauss2", x, mean2, sigma2);
  RooRealVar norm2("norm2", "norm2", hist->GetEntries() * 0.2, 0, hist->GetEntries());

  RooAddPdf model("model", "g1+g2", RooArgList(gauss1, gauss2), RooArgList(norm1, norm2));

  if (!debug) {
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    RooMsgService::instance().setSilentMode(true);
  }

  RooFitResult* fitRes = model.fitTo(data, Save(true), PrintLevel(debug ? 1 : -1), Range(LowEdge, HighEdge));

  if (ws) {
    data.SetName((std::string(hist->GetName()) + "_data").c_str());
    model.SetName((std::string(hist->GetName()) + "_pdf").c_str());
    ws->import(data);
    ws->import(model);
    RooRealVar lowRangeVar(Form("%s_low", hist->GetName()), "fit lower edge", LowEdge);
    RooRealVar highRangeVar(Form("%s_high", hist->GetName()), "fit upper edge", HighEdge);
    ws->import(lowRangeVar);
    ws->import(highRangeVar);
    std::string fitResName = std::string(hist->GetName()) + "_fitRes";
    fitRes->SetName(fitResName.c_str());
    ws->import(*fitRes);
  }
  double w1 = norm1.getValV();
  double w2 = norm2.getValV();
  double v1 = mean1.getValV(), v2 = mean2.getValV();
  double s1v = sigma1.getValV(), s2v = sigma2.getValV();
  double total = w1 + w2;
  double val = (w1 * v1 + w2 * v2) / total;
  double width = (w1 * s1v + w2 * s2v) / total;
  double err = std::sqrt(pow(norm1.getError() * v1 / total, 2) + pow(norm2.getError() * v2 / total, 2));
  double werr = std::sqrt(pow(norm1.getError() * s1v / total, 2) + pow(norm2.getError() * s2v / total, 2));

  return results(val, err, width, werr);
}

// Single-Gaussian fit
results fitOneGauss_RooFit(TH1D* hist, /*bool fitTwice,*/ bool debug, RooWorkspace* ws) {
  const double fitrangeFactor = 2.0;
  const double fitrangeFactor1 = 1.5;

  double rms0;
  auto meanPair = GetMean(hist, 0.2, 2.0, rms0);
  double mean0 = meanPair.first;

  double LowEdge = std::max(0.5, mean0 - fitrangeFactor * rms0);
  double diff = mean0 - LowEdge;
  double HighEdge = mean0 + std::min(fitrangeFactor1 * rms0, diff);
  HighEdge = std::min(HighEdge, hist->GetXaxis()->GetXmax());

  RooRealVar x("x", "x", hist->GetXaxis()->GetXmin(), hist->GetXaxis()->GetXmax());
  RooDataHist data("data", "dataset from TH1", x, Import(*hist));

  RooRealVar meanVar("mean", "mean", mean0, hist->GetXaxis()->GetXmin(), hist->GetXaxis()->GetXmax());
  RooRealVar sigmaVar("sigma", "sigma", rms0, 0.001, hist->GetXaxis()->GetXmax());
  RooRealVar normVar("norm", "norm", hist->Integral(), 0, hist->GetEntries() * 2);
  RooGaussian gauss("gauss", "gauss", x, meanVar, sigmaVar);

  if (!debug) {
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    RooMsgService::instance().setSilentMode(true);
  }

  RooFitResult* fitRes = gauss.fitTo(data, Save(true), PrintLevel(debug ? 1 : -1), Range(LowEdge, HighEdge));

  if (ws) {
    data.SetName((std::string(hist->GetName()) + "_data").c_str());
    gauss.SetName((std::string(hist->GetName()) + "_pdf").c_str());
    ws->import(data);
    ws->import(gauss);
    RooRealVar lowRangeVar(Form("%s_low", hist->GetName()), "fit lower edge", LowEdge);
    RooRealVar highRangeVar(Form("%s_high", hist->GetName()), "fit upper edge", HighEdge);
    ws->import(lowRangeVar);
    ws->import(highRangeVar);
    std::string fitResName = std::string(hist->GetName()) + "_fitRes";
    fitRes->SetName(fitResName.c_str());
    ws->import(*fitRes);
  }

  double meanVal = meanVar.getValV();
  double meanErr = meanVar.getError();
  double sigmaVal = sigmaVar.getValV();
  double sigmaErr = sigmaVar.getError();

  return results(meanVal, meanErr, sigmaVal, sigmaErr);
}

// 4) Constant (pol0) fit via RooFit
void fitConstPol0_RooFit(TH1D* histo, double LowEdge, double HighEdge, bool debug, RooWorkspace* ws) {
  RooRealVar x("x", "x", LowEdge, HighEdge);
  RooDataHist dataHist("dataHist", "dataHist from TH1", RooArgList(x), Import(*histo));

  RooRealVar c0("c0", "Constant term", 1.0, -10.0, 10.0);
  RooPolynomial poly0("poly0", "Constant Polynomial", x, RooArgList(c0), 0);
  if (!debug) {
    RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
    RooMsgService::instance().setSilentMode(true);
  }
  RooFitResult* fitRes = poly0.fitTo(dataHist, Save(true), Range(LowEdge, HighEdge), Verbose(debug));

  for (int i = 0; i < dataHist.numEntries(); ++i) {
    dataHist.get(i);
    const RooArgSet* row = dataHist.get();
    double binCenter = ((RooRealVar*)row->find("x"))->getVal();
    double binContent = dataHist.weight();
  }

  if (debug) {
    std::cout << "Fit to Pol0: " << c0.getValV() << " +- " << c0.getError() << " in range " << LowEdge << ":"
              << HighEdge << std::endl;
  }

  // Axis titles & ranges
  histo->GetXaxis()->SetTitle("i#eta");
  histo->GetYaxis()->SetTitle("MPV(E_{HCAL}/(p-E_{ECAL}))");
  histo->GetYaxis()->SetRangeUser(0.4, 1.6);

  // Optionally import to workspace
  if (ws) {
    dataHist.SetName((std::string(histo->GetName()) + "_data").c_str());
    poly0.SetName((std::string(histo->GetName()) + "_pdf").c_str());
    ws->import(dataHist);
    ws->import(poly0);
    RooRealVar lowRangeVar(Form("%s_low", histo->GetName()), "fit lower edge", LowEdge);
    RooRealVar highRangeVar(Form("%s_high", histo->GetName()), "fit upper edge", HighEdge);
    ws->import(lowRangeVar);
    ws->import(highRangeVar);
    std::string fitResName = std::string(histo->GetName()) + "_fitRes";
    fitRes->SetName(fitResName.c_str());
    ws->import(*fitRes);
  }
}

void readCorrFactors(char* infile,
                     double scale,
                     std::map<int, cfactors>& cfacs,
                     int& etamin,
                     int& etamax,
                     int& maxdepth,
                     int iformat = 0,
                     bool debug = false) {
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
        int ieta = (iformat == 1) ? std::atoi(items[0].c_str()) : std::atoi(items[1].c_str());
        int depth = (iformat == 1) ? std::atoi(items[1].c_str()) : std::atoi(items[2].c_str());
        float corrf = std::atof(items[3].c_str());
        float dcorr = (iformat == 1) ? (0.02 * corrf) : std::atof(items[4].c_str());
        cfactors cfac(ieta, depth, scale * corrf, scale * dcorr);
        int detId = (iformat == 1) ? repackId(items[2], ieta, depth) : repackId(ieta, depth);
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
  if (debug) {
    unsigned k(0);
    std::cout << "Eta Range " << etamin << ":" << etamax << " Max Depth " << maxdepth << std::endl;
    for (std::map<int, cfactors>::const_iterator itr = cfacs.begin(); itr != cfacs.end(); ++itr, ++k)
      std::cout << "[" << k << "] " << std::hex << itr->first << std::dec << ": " << (itr->second).ieta << " "
                << (itr->second).depth << " " << (itr->second).corrf << " " << (itr->second).dcorr << std::endl;
  }
}

void FitHistExtended_RootFit(const char* infile,
                             const char* outfile,
                             std::string prefix,
                             int numb = 54,
                             int type = 13,
                             bool append = true,
                             bool fiteta = true,
                             int iname = 3,
                             bool debug = false) {
  std::string sname("ratio"), lname("Z"), wname("W"), ename("etaB");
  double xbins[99];
  double xbin[23] = {-23.0, -21.0, -19.0, -17.0, -15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, 0.0,
                     3.0,   5.0,   7.0,   9.0,   11.0,  13.0,  15.0,  17.0, 19.0, 21.0, 23.0};
  if ((type % 10) == 2) {
    numb = 22;
    for (int k = 0; k <= numb; ++k)
      xbins[k] = xbin[k];
  } else if ((type % 10) == 1) {
    numb = 1;
    xbins[0] = -25;
    xbins[1] = 25;
  } else {
    int neta = numb / 2;
    for (int k = 0; k <= (numb + 1); ++k) {
      xbins[k] = (k - neta) - 0.5;
    }
  }
  if (debug) {
    for (int k = 0; k <= (numb + 1); ++k)
      std::cout << " " << xbins[k];
    std::cout << std::endl;
  }
  TFile* file = new TFile(infile);
  std::vector<TH1D*> hists;
  std::vector<RooWorkspace*> Workspace;
  char name[200];
  if (debug) {
    std::cout << infile << " " << file << std::endl;
  }

  if (file != nullptr) {
    sprintf(name, "%s%s%d0", prefix.c_str(), sname.c_str(), iname);
    TH1D* hist0 = (TH1D*)file->FindObjectAny(name);
    std::string wsName = Form("%s_ws", hist0->GetName());
    RooWorkspace* ws0 = new RooWorkspace(wsName.c_str(), hist0->GetTitle());
    bool ok = (hist0 != nullptr);
    if (debug) {
      std::cout << name << " Pointer " << hist0 << " " << ok << std::endl;
    }
    if (ok) {
      TH1D *histo(0), *histw(0);
      RooWorkspace* wso(0);
      if (numb > 0) {
        sprintf(name, "%s%s%d", prefix.c_str(), lname.c_str(), iname);
        histo = new TH1D(name, hist0->GetTitle(), numb, xbins);
        sprintf(name, "%s%s%d", prefix.c_str(), wname.c_str(), iname);
        histw = new TH1D(name, hist0->GetTitle(), numb, xbins);
        std::string wsName = Form("%s_ws", histo->GetName());
        wso = new RooWorkspace(wsName.c_str(), histo->GetTitle());
        if (debug)
          std::cout << name << " " << histo->GetNbinsX() << std::endl;
      }
      if (hist0->GetEntries() > 10) {
        double rms;
        results meaner0 = (((type / 10) % 10) == 0) ? fitOneGauss_RooFit(hist0, debug, ws0)
                                                    : fitDoubleSidedCrystalball_RooFit(hist0, debug, ws0);

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
          std::string wsName = Form("%s_ws", hist->GetName());
          RooWorkspace* ws = new RooWorkspace(wsName.c_str(), hist->GetTitle());
          if (debug)
            std::cout << "Histogram " << name << ":" << (hist->GetName()) << " with " << (hist->GetEntries())
                      << " entries" << std::endl;
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
              std::string wsName = Form("%s_ws", hist2->GetName());
              RooWorkspace* ws2 = new RooWorkspace(wsName.c_str(), hist2->GetTitle());
              fitOneGauss_RooFit(hist2, debug, ws2);
              hists.push_back(hist2);
              Workspace.push_back(ws2);
              results meaner = (((type / 10) % 10) == 0) ? fitOneGauss_RooFit(hist, debug, ws)
                                                         : fitDoubleSidedCrystalball_RooFit(hist, debug, ws);
              value = meaner.mean;
              error = meaner.errmean;
              width = meaner.width;
              werror = meaner.errwidth;
            } else {
              results meaner = (((type / 10) % 10) == 0) ? fitOneGauss_RooFit(hist, debug, ws)
                                                         : fitDoubleSidedCrystalball_RooFit(hist, debug, ws);
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
          Workspace.push_back(ws);
        }
        if (debug) {
          std::cout << "Hist************** " << j << " Value " << value << " +- " << error << std::endl;
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

          fitConstPol0_RooFit(histo, LowEdge, HighEdge, debug, wso);
          histw->GetXaxis()->SetTitle("i#eta");
          histw->GetYaxis()->SetTitle("MPV/Width(E_{HCAL}/(p-E_{ECAL}))");
          histw->GetYaxis()->SetRangeUser(0.0, 0.5);
        }
        hists.push_back(histo);
        hists.push_back(histw);
        Workspace.push_back(wso);
      } else {
        hists.push_back(hist0);
        Workspace.push_back(ws0);
      }

      // Barrel,Endcap
      for (int j = 1; j <= 4; ++j) {
        sprintf(name, "%s%s%d%d", prefix.c_str(), ename.c_str(), iname, j);
        TH1D* hist1 = (TH1D*)file->FindObjectAny(name);
        if (debug) {
          std::cout << "Get Histogram for " << name << " at " << hist1 << std::endl;
        }
        if (hist1 != nullptr) {
          TH1D* hist = (TH1D*)hist1->Clone();
          std::string wsName = Form("%s_ws", hist->GetName());
          RooWorkspace* ws = new RooWorkspace(wsName.c_str(), hist->GetTitle());
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
            std::string wsName = Form("%s_ws", hist2->GetName());
            RooWorkspace* ws2 = new RooWorkspace(wsName.c_str(), hist2->GetTitle());
            results meanerr = (((type / 10) % 10) == 0) ? fitOneGauss_RooFit(hist2, debug, ws2)
                                                        : fitDoubleSidedCrystalball_RooFit(hist2, debug, ws2);
            value = meanerr.mean;
            error = meanerr.errmean;
            width = meanerr.width;
            werror = meanerr.errwidth;
            double wbyv = width / value;
            double wverr = wbyv * std::sqrt((werror * werror) / (width * width) + (error * error) / (value * value));
            std::cout << hist2->GetName() << " MPV " << value << " +- " << error << " Width " << width << " +- "
                      << werror << " W/M " << wbyv << " +- " << wverr << std::endl;
            hists.push_back(hist2);
            Workspace.push_back(ws2);

            if (hist1->GetBinLowEdge(1) < 0.1) {
              sprintf(name, "%sTwo", hist1->GetName());
              TH1D* hist3 = (TH1D*)hist1->Clone(name);
              std::string wsName = Form("%s_ws", hist3->GetName());
              RooWorkspace* ws3 = new RooWorkspace(wsName.c_str(), hist3->GetTitle());
              fitLanGau_RooFit(hist3, debug, ws3);
              hists.push_back(hist3);
              Workspace.push_back(ws3);
            }
            results meaner0 = (((type / 10) % 10) == 0) ? fitOneGauss_RooFit(hist, debug, ws)
                                                        : fitDoubleSidedCrystalball_RooFit(hist, debug, ws);
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
          Workspace.push_back(ws);
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
    for (unsigned int i = 0; i < Workspace.size(); ++i) {
      RooWorkspace* ws = (RooWorkspace*)Workspace[i];
      if (debug) {
        std::cout << "Write Workspace  " << ws->GetName() << std::endl;
      }
      ws->Write();
      //delete ws;
    }
    theFile->Close();
    file->Close();
  }
}

void PlotHist_RooFit(const char* infile,
                     std::string prefix,
                     std::string text,
                     int mode = 4,
                     int kopt = 100,
                     double lumi = 0,
                     std::string ener = "13.6",
                     bool isRealData = false,
                     bool drawStatBox = true,
                     int save = 0,
                     bool debug = false) {
  // Define histogram and title arrays (unchanged)
  std::string name0[6] = {"ratio00", "ratio10", "ratio20", "ratio30", "ratio40", "ratio50"};
  std::string name1[5] = {"Z0", "Z1", "Z2", "Z3", "Z4"};
  std::string name2[5] = {"L0", "L1", "L2", "L3", "L4"};
  std::string name3[5] = {"V0", "V1", "V2", "V3", "V4"};
  std::string name4[20] = {"etaB41", "etaB42", "etaB43", "etaB44", "etaB31", "etaB32", "etaB33",
                           "etaB34", "etaB21", "etaB22", "etaB23", "etaB24", "etaB11", "etaB12",
                           "etaB13", "etaB14", "etaB01", "etaB02", "etaB03", "etaB04"};
  std::string name5[5] = {"W0", "W1", "W2", "W3", "W4"};
  std::string title[6] = {"Tracks with p = 10:20 GeV",
                          "Tracks with p = 20:30 GeV",
                          "Tracks with p = 30:40 GeV",
                          "Tracks with p = 40:60 GeV",
                          "Tracks with p = 60:100 GeV",
                          "Tracks with p = 20:100 GeV"};
  std::string title1[20] = {"Tracks with p = 60:100 GeV (Barrel)", "Tracks with p = 60:100 GeV (Transition)",
                            "Tracks with p = 60:100 GeV (Endcap)", "Tracks with p = 60:100 GeV",
                            "Tracks with p = 40:60 GeV (Barrel)",  "Tracks with p = 40:60 GeV (Transition)",
                            "Tracks with p = 40:60 GeV (Endcap)",  "Tracks with p = 40:60 GeV",
                            "Tracks with p = 30:40 GeV (Barrel)",  "Tracks with p = 30:40 GeV (Transition)",
                            "Tracks with p = 30:40 GeV (Endcap)",  "Tracks with p = 30:40 GeV",
                            "Tracks with p = 20:30 GeV (Barrel)",  "Tracks with p = 20:30 GeV (Transition)",
                            "Tracks with p = 20:30 GeV (Endcap)",  "Tracks with p = 20:30 GeV",
                            "Tracks with p = 10:20 GeV (Barrel)",  "Tracks with p = 10:20 GeV (Transition)",
                            "Tracks with p = 10:20 GeV (Endcap)",  "Tracks with p = 10:20 GeV"};
  std::string xtitl[5] = {"E_{HCAL}/(p-E_{ECAL})", "i#eta", "d_{L1}", "# Vertex", "E_{HCAL}/(p-E_{ECAL})"};
  std::string ytitl[5] = {
      "Tracks", "MPV(E_{HCAL}/(p-E_{ECAL}))", "MPV(E_{HCAL}/(p-E_{ECAL}))", "MPV(E_{HCAL}/(p-E_{ECAL}))", "Tracks"};

  // Style settings
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  if (mode < 0 || mode > 5)
    mode = 0;
  if (drawStatBox) {
    int iopt = (mode != 0) ? 10 : 1110;
    gStyle->SetOptStat(iopt);
    gStyle->SetOptFit(1);
  } else {
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
  }

  TFile* file = TFile::Open(infile);

  char name[100], namep[100];
  int kmax = (mode == 4) ? 20 : (((mode < 1) || (mode > 5)) ? 6 : 5);

  for (int k = 0; k < kmax; ++k) {
    // Construct histogram and PDF names
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
    RooWorkspace* w = nullptr;
    file->GetObject((std::string(name) + "_ws").c_str(), w);
    if (w == nullptr)
      continue;

    // Get RooDataHist and RooAbsPdf from workspace
    std::string dataName = std::string(name) + "_data";
    std::string pdfName = std::string(name) + "_pdf";
    std::string rngLowName = std::string(name) + "_low";
    std::string rngHighName = std::string(name) + "_high";

    RooDataHist* dataHist = (RooDataHist*)w->data(dataName.c_str());
    RooAbsPdf* pdf = (RooAbsPdf*)w->pdf(pdfName.c_str());

    RooRealVar* lowEdgeVar = w->var(rngLowName.c_str());
    RooRealVar* highEdgeVar = w->var(rngHighName.c_str());

    if (!dataHist || !pdf || !lowEdgeVar || !highEdgeVar) {
      if (debug)
        std::cout << "Warning: Could not find data (" << dataName << "), PDF (" << pdfName << "), low edge ("
                  << rngLowName << "), or high edge (" << rngHighName << ") for " << name << std::endl;
      continue;
    }
    double lowEdge = lowEdgeVar->getVal();
    double highEdge = highEdgeVar->getVal();

    if (dataHist && pdf) {
      // Get the observable
      RooRealVar* x = w->var("x");
      if (!x) {
        if (debug)
          std::cout << "Error: Variable 'x' not found in workspace" << std::endl;
        continue;
      }
      // Create a canvas
      sprintf(namep, "c_%s", name);
      TCanvas* pad = new TCanvas(namep, namep, 700, 500);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      if ((kopt / 10) % 10 > 0)
        gPad->SetGrid();

      // Create a RooPlot
      RooPlot* frame = x->frame();
      frame->SetTitle("");
      frame->GetXaxis()->SetTitle(xtitl[mode].c_str());
      frame->GetXaxis()->SetTitleSize(0.04);
      frame->GetYaxis()->SetTitle(ytitl[mode].c_str());
      frame->GetYaxis()->SetLabelOffset(0.005);
      frame->GetYaxis()->SetTitleSize(0.04);
      frame->GetYaxis()->SetLabelSize(0.035);
      frame->GetYaxis()->SetTitleOffset(1.10);

      // Set X-axis range based on mode
      if (mode == 0 || mode == 4) {
        if ((kopt / 100) % 10 == 2) {
          x->setRange(0.0, 0.30);
          frame->GetXaxis()->SetRangeUser(0.0, 0.3);
        } else {
          x->setRange(0.25, 2.25);
          frame->GetXaxis()->SetRangeUser(0.25, 2.25);
        }
      } else if (mode == 5) {
        frame->SetMinimum(0.1);
        frame->SetMaximum(0.50);
      } else if (isRealData) {
        frame->SetMinimum(0.5);
        frame->SetMaximum(1.50);
      } else {
        frame->SetMinimum(0.8);
        frame->SetMaximum(1.20);
      }

      dataHist->plotOn(frame,
                       RooFit::MarkerStyle(20),
                       RooFit::MarkerColor(2),
                       RooFit::LineColor(2),
                       RooFit::DataError(RooAbsData::None));
      pdf->plotOn(frame, RooFit::LineColor(4), RooFit::LineWidth(2));

      // Draw the frame
      frame->Draw();

      // Adjust stats box if enabled
      if (drawStatBox) {
        double ymin = (mode == 0 || mode == 4) ? 0.70 : 0.80;
        TPaveStats* st1 = (TPaveStats*)pad->GetPrimitive("stats");
        if (st1) {
          st1->SetY1NDC(ymin);
          st1->SetY2NDC(0.90);
          st1->SetX1NDC(0.65);
          st1->SetX2NDC(0.90);
        }
      }

      // Add line for modes other than 0 and 4 if kopt % 10 > 0
      TLine* line = nullptr;
      if (mode != 0 && mode != 4 && kopt % 10 > 0) {
        TH1* hist = dataHist->createHistogram("hist", *x);
        double p0 = 0.0;
        if (kopt % 10 > 0) {
          int nbin = hist->GetNbinsX();
          double LowEdge = (kopt % 10 == 1) ? hist->GetBinLowEdge(1) : -20;
          double HighEdge = (kopt % 10 == 1) ? hist->GetBinLowEdge(nbin) + hist->GetBinWidth(nbin) : 20;
          TFitResultPtr Fit = hist->Fit("pol0", "+QRWLS", "", LowEdge, HighEdge);
          p0 = Fit->Value(0);
        }
        double xmin = x->getMin();
        double xmax = x->getMax();
        line = new TLine(xmin, p0, xmax, p0);
        line->SetLineWidth(2);
        line->SetLineStyle(2);
        line->Draw("same");
        delete hist;  // Clean up
      }

      // Add text labels (unchanged)
      double ymx = 0.96, xmi = 0.25, xmx = 0.90;
      char txt[100];
      if (lumi > 0.1) {
        ymx = (mode == 0 || mode == 4) ? 0.70 - 0.005 : 0.80 - 0.005;
        xmi = 0.45;
        TPaveText* txt0 = new TPaveText(0.65, 0.91, 0.90, 0.96, "blNDC");
        txt0->SetFillColor(0);
        sprintf(txt, "%s TeV %5.1f fb^{-1}", ener.c_str(), lumi);
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
      double xmax = (isRealData) ? 0.33 : 0.44;
      ymi = (lumi > 0.1) ? 0.91 : 0.84;
      ymx = ymi + 0.05;
      TPaveText* txt2 = new TPaveText(0.11, ymi, xmax, ymx, "blNDC");
      txt2->SetFillColor(0);
      if (isRealData)
        sprintf(txt, "CMS Preliminary");
      else
        sprintf(txt, "CMS Simulation Preliminary");
      txt2->AddText(txt);
      txt2->Draw("same");

      // Update and save canvas
      pad->Modified();
      pad->Update();
      if (save > 0) {
        sprintf(name, "%s.pdf", pad->GetName());
        pad->Print(name);
      } else if (save < 0) {
        sprintf(name, "%s.C", pad->GetName());
        pad->Print(name);
      }
      delete frame;
    } else {
      if (debug)
        std::cout << "Warning: Could not find data (" << dataName << ") or PDF (" << pdfName << ") for " << name
                  << std::endl;
    }
  }
  file->Close();
}
