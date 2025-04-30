// ROOT includes
#include "RooAddPdf.h"
#include "RooCBShape.h"
#include "RooDataHist.h"
#include "RooExponential.h"
#include "RooGaussian.h"
#include "RooPlot.h"
#include "RooRealVar.h"
#include "RooVoigtian.h"
#include "TCanvas.h"
#include "TClass.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TGaxis.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TKey.h"
#include "TLegend.h"
#include "TObjString.h"
#include "TObject.h"
#include "TProfile.h"
#include "TRatioPlot.h"
#include "TStyle.h"

// standard includes
#include <iomanip>
#include <iostream>
#include <map>
#include <fmt/core.h>

// CMSSW classes / style
#include "Alignment/OfflineValidation/interface/FitWithRooFit.h"
#include "Alignment/OfflineValidation/macros/CMS_lumi.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1D.h"

bool debugMode_{false};
Int_t def_markers[9] = {kFullSquare,
                        kFullCircle,
                        kFullTriangleDown,
                        kOpenSquare,
                        kDot,
                        kOpenCircle,
                        kFullTriangleDown,
                        kFullTriangleUp,
                        kOpenTriangleDown};

Int_t def_colors[9] = {kBlack, kRed, kBlue, kMagenta, kGreen, kCyan, kViolet, kOrange, kGreen + 2};

namespace diMuonMassBias {

  struct fitOutputs {
  public:
    fitOutputs(const Measurement1D& bias, const Measurement1D& width) : m_bias(bias), m_width(width) {}

    // getters
    const Measurement1D getBias() { return m_bias; }
    const Measurement1D getWidth() { return m_width; }
    bool isInvalid() const {
      return (m_bias.value() == 0.f && m_bias.error() == 0.f && m_width.value() == 0.f && m_width.error() == 0.f);
    }

  private:
    Measurement1D m_bias;
    Measurement1D m_width;
  };

  static constexpr int minimumHits = 10;

  using histoMap = std::map<std::string, TH1F*>;
  using histo2DMap = std::map<std::string, TH2F*>;
  using histo3DMap = std::map<std::string, TH3F*>;

  using extrema = std::pair<Double_t, Double_t>;
  using extremaMap = std::map<std::string, diMuonMassBias::extrema>;

}  // namespace diMuonMassBias

/*--------------------------------------------------------------------*/
template <typename T>
std::map<std::string, std::vector<T>> transformMaps(const std::vector<std::map<std::string, T>>& vecOfMaps)
/*--------------------------------------------------------------------*/
{
  std::map<std::string, std::vector<T>> result;

  // Lambda to insert each key-value pair into the result map
  auto insert_into_result = [&result](const std::map<std::string, T>& map) {
    for (const auto& pair : map) {
      result[pair.first].push_back(pair.second);
    }
  };

  // Apply the lambda to each map in the vector
  std::for_each(vecOfMaps.begin(), vecOfMaps.end(), insert_into_result);

  return result;
}

/*--------------------------------------------------------------------*/
template <typename T>
diMuonMassBias::extremaMap getExtrema(const T& inputColl, const float sigma)
/*--------------------------------------------------------------------*/
{
  diMuonMassBias::extremaMap result;  // output

  // first transform the map vector of maps with a map of vectors
  const auto& mapOfVecs = transformMaps(inputColl);

  for (const auto& [key, vec] : mapOfVecs) {
    TObjArray* array = new TObjArray(vec.size());

    for (const auto& histo : vec) {
      array->Add(histo);
    }

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
    delete array;

    const auto& delta = theMaximum - theMinimum;
    result.insert({key, std::make_pair(theMinimum - (sigma * delta), theMaximum + (sigma * delta))});
  }
  return result;
}

/*--------------------------------------------------------------------*/
std::pair<int, int> getClosestFactors(int input)
/*--------------------------------------------------------------------*/
{
  if ((input % 2 != 0) && input > 1) {
    input += 1;
  }

  int testNum = (int)sqrt(input);
  while (input % testNum != 0) {
    testNum--;
  }
  return std::make_pair(testNum, input / testNum);
}

/*--------------------------------------------------------------------*/
template <typename T>
void MakeNicePlotStyle(T* hist)
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
  if (((TObject*)hist)->IsA()->InheritsFrom("TH2")) {
    hist->GetZaxis()->CenterTitle(true);
    hist->GetZaxis()->SetTitleSize(0.04);
    hist->GetZaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelSize(.05);
    hist->GetXaxis()->SetTitleOffset(1.0);
    hist->GetYaxis()->SetTitleOffset(1.0);
    hist->GetZaxis()->SetTitleOffset(1.15);
  } else {
    hist->GetXaxis()->SetTitleOffset(0.9);
    hist->GetYaxis()->SetTitleOffset(1.7);
  }
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  if (((TObject*)hist)->IsA()->InheritsFrom("TGraph")) {
    hist->GetYaxis()->SetLabelSize(.025);
    //hist->GetYaxis()->SetNdivisions(505);
  } else {
    hist->GetYaxis()->SetLabelSize(.05);
  }
  hist->GetXaxis()->SetLabelSize(.05);
}

//-----------------------------------------------------------------------------------
diMuonMassBias::fitOutputs fitBWTimesCB(TH1* hist)
//-----------------------------------------------------------------------------------
{
  if (hist->GetEntries() < diMuonMassBias::minimumHits) {
    std::cout << " Input histogram:" << hist->GetName() << " has not enough entries (" << hist->GetEntries()
              << ") for a meaningful Voigtian fit!\n"
              << "Skipping!";

    return diMuonMassBias::fitOutputs(Measurement1D(0., 0.), Measurement1D(0., 0.));
  }

  TCanvas* c1 = new TCanvas();
  if (debugMode_) {
    c1->Clear();
    c1->SetLeftMargin(0.15);
    c1->SetRightMargin(0.10);
  }

  // silence messages
  RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

  Double_t xMean = 91.1876;
  Double_t xMin = hist->GetXaxis()->GetXmin();
  Double_t xMax = hist->GetXaxis()->GetXmax();

  if (debugMode_) {
    std::cout << " fitting range: (" << xMin << "-" << xMax << ")" << std::endl;
  }

  double sigma(2.);
  double sigmaMin(0.1);
  double sigmaMax(10.);

  double sigma2(0.1);
  double sigma2Min(0.);
  double sigma2Max(10.);

  std::unique_ptr<FitWithRooFit> fitter = std::make_unique<FitWithRooFit>();

  bool useChi2(false);

  fitter->useChi2_ = useChi2;
  fitter->initMean(xMean, xMin, xMax);
  fitter->initSigma(sigma, sigmaMin, sigmaMax);
  fitter->initSigma2(sigma2, sigma2Min, sigma2Max);
  fitter->initAlpha(1.5, 0.05, 10.);
  fitter->initN(1, 0.01, 100.);
  fitter->initFGCB(0.4, 0., 1.);
  fitter->initGamma(2.4952, 0., 10.);
  fitter->gamma()->setConstant(kTRUE);
  fitter->initMean2(0., -20., 20.);
  fitter->mean2()->setConstant(kTRUE);
  fitter->initSigma(1.2, 0., 5.);
  fitter->initAlpha(1.5, 0.05, 10.);
  fitter->initN(1, 0.01, 100.);
  fitter->initExpCoeffA0(-1., -10., 10.);
  fitter->initExpCoeffA1(0., -10., 10.);
  fitter->initExpCoeffA2(0., -2., 2.);
  fitter->initFsig(0.9, 0., 1.);
  fitter->initA0(0., -10., 10.);
  fitter->initA1(0., -10., 10.);
  fitter->initA2(0., -10., 10.);
  fitter->initA3(0., -10., 10.);
  fitter->initA4(0., -10., 10.);
  fitter->initA5(0., -10., 10.);
  fitter->initA6(0., -10., 10.);

  fitter->fit(hist, "breitWignerTimesCB", "exponential", xMin, xMax, false);

  TString histName = hist->GetName();

  if (debugMode_) {
    c1->Print("fit_debug" + histName + ".pdf");
  }
  delete c1;

  Measurement1D resultM(fitter->mean()->getValV(), fitter->mean()->getError());
  Measurement1D resultW(fitter->sigma()->getValV(), fitter->sigma()->getError());

  return diMuonMassBias::fitOutputs(resultM, resultW);
}

//-----------------------------------------------------------------------------------
void fitAndFillHisto(std::pair<std::string, TH2F*> toHarvest,
                     diMuonMassBias::histoMap& meanHistos_,
                     diMuonMassBias::histoMap& widthHistos_)
//-----------------------------------------------------------------------------------
{
  const auto& key = toHarvest.first;
  const auto& ME = toHarvest.second;

  if (debugMode_)
    std::cout << "dealing with key: " << key << std::endl;

  if (ME == nullptr) {
    std::cout << "could not find MonitorElement for key: " << key << std::endl;
    return;
  }

  for (int bin = 1; bin <= ME->GetNbinsY(); bin++) {
    const auto& yaxis = ME->GetYaxis();
    const auto& low_edge = yaxis->GetBinLowEdge(bin);
    const auto& high_edge = yaxis->GetBinUpEdge(bin);

    if (debugMode_)
      std::cout << "dealing with bin: " << bin << " range: (" << low_edge << "," << high_edge << ")";
    TH1D* Proj = ME->ProjectionX(Form("%s_proj_%i", key.c_str(), bin), bin, bin);
    Proj->SetTitle(Form("%s #in (%.2f,%.2f), bin: %i", Proj->GetTitle(), low_edge, high_edge, bin));

    diMuonMassBias::fitOutputs results = fitBWTimesCB(Proj);

    if (results.isInvalid()) {
      std::cout << "the current bin has invalid data" << std::endl;
      continue;
    }

    // fill the mean profiles
    const Measurement1D& bias = results.getBias();
    meanHistos_[key]->SetBinContent(bin, bias.value());
    meanHistos_[key]->SetBinError(bin, bias.error());

    // fill the width profiles
    const Measurement1D& width = results.getWidth();
    widthHistos_[key]->SetBinContent(bin, width.value());
    widthHistos_[key]->SetBinError(bin, width.error());

    if (debugMode_) {
      std::cout << "key: " << key << " bin: " << bin << " bias: " << bias.value()
                << " (in histo: " << meanHistos_[key]->GetBinContent(bin) << ") width: " << width.value()
                << " (in histo: " << widthHistos_[key]->GetBinContent(bin) << ")" << std::endl;
    }
  }
}

//-----------------------------------------------------------------------------------
void bookMaps(const diMuonMassBias::histo3DMap& harvestTargets_,
              diMuonMassBias::histo2DMap& meanMaps_,
              diMuonMassBias::histo2DMap& widthMaps_,
              const unsigned int counter)
//-----------------------------------------------------------------------------------
{
  for (const auto& [key, ME] : harvestTargets_) {
    if (ME == nullptr) {
      std::cout << "could not find MonitorElement for key: " << key << std::endl;
      continue;
    }

    const auto& title = ME->GetTitle();
    const auto& xtitle = ME->GetYaxis()->GetTitle();
    const auto& ytitle = ME->GetZaxis()->GetTitle();

    const auto& nxbins = ME->GetNbinsY();
    const auto& xmin = ME->GetYaxis()->GetXmin();
    const auto& xmax = ME->GetYaxis()->GetXmax();

    const auto& nybins = ME->GetNbinsZ();
    const auto& ymin = ME->GetZaxis()->GetXmin();
    const auto& ymax = ME->GetZaxis()->GetXmax();

    if (debugMode_) {
      std::cout << "Booking " << key << std::endl;
    }

    TH2F* meanToBook =
        new TH2F(fmt::format("Mean_{}_{}", counter, key).c_str(),
                 fmt::format("{};{};{};#LT M_{{#mu^{{-}}#mu^{{+}}}} #GT [GeV]", title, xtitle, ytitle).c_str(),
                 nxbins,
                 xmin,
                 xmax,
                 nybins,
                 ymin,
                 ymax);

    if (debugMode_) {
      std::cout << "after creating mean" << key << std::endl;
    }

    meanMaps_.insert({key, meanToBook});

    if (debugMode_) {
      std::cout << "after inserting mean" << key << std::endl;
    }

    TH2F* sigmaToBook =
        new TH2F(fmt::format("Sigma_{}_{}", counter, key).c_str(),
                 fmt::format("{};{};#sigma of  M_{{#mu^{{-}}#mu^{{+}}}} [GeV]", title, xtitle, ytitle).c_str(),
                 nxbins,
                 xmin,
                 xmax,
                 nybins,
                 ymin,
                 ymax);

    if (debugMode_) {
      std::cout << "after creating sigma" << key << std::endl;
    }

    widthMaps_.insert({key, sigmaToBook});

    if (debugMode_) {
      std::cout << "after inserting sigma" << key << std::endl;
    }
  }
}

//-----------------------------------------------------------------------------------
void fitAndFillMap(std::pair<std::string, TH3F*> toHarvest,
                   diMuonMassBias::histo2DMap& meanMaps_,
                   diMuonMassBias::histo2DMap& widthMaps_)
//-----------------------------------------------------------------------------------
{
  const auto& key = toHarvest.first;
  const auto& ME = toHarvest.second;

  if (debugMode_)
    std::cout << "dealing with key: " << key << std::endl;

  if (ME == nullptr) {
    std::cout << "could not find MonitorElement for key: " << key << std::endl;
    return;
  }

  for (int binY = 1; binY <= ME->GetNbinsY(); binY++) {
    const auto& yaxis = ME->GetYaxis();
    const auto& y_low_edge = yaxis->GetBinLowEdge(binY);
    const auto& y_high_edge = yaxis->GetBinUpEdge(binY);

    for (int binZ = 1; binZ <= ME->GetNbinsZ(); binZ++) {
      const auto& zaxis = ME->GetZaxis();
      const auto& z_low_edge = zaxis->GetBinLowEdge(binZ);
      const auto& z_high_edge = zaxis->GetBinUpEdge(binZ);

      if (debugMode_) {
        std::cout << "dealing with y bin: " << binY << " range: (" << y_low_edge << "," << y_high_edge << ")";
        std::cout << " dealing with z bin: " << binZ << " range: (" << z_low_edge << "," << z_high_edge << ")"
                  << std::endl;
      }

      const auto& ProjYZ = ME->ProjectionX(Form("%s_proj_%i_proj_%i", key.c_str(), binY, binZ), binY, binY, binZ, binZ);
      ProjYZ->SetTitle(Form("%s #in (%.2f,%.2f) - (%.2f,%.2f), bin: %i-%i ",
                            ProjYZ->GetTitle(),
                            y_low_edge,
                            y_high_edge,
                            z_low_edge,
                            z_high_edge,
                            binY,
                            binZ));

      diMuonMassBias::fitOutputs results = fitBWTimesCB(ProjYZ);

      if (results.isInvalid()) {
        std::cout << "the current bin has invalid data" << std::endl;
        continue;
      }

      // fill the mean profiles
      const Measurement1D& bias = results.getBias();
      meanMaps_[key]->SetBinContent(binY, binZ, bias.value());
      meanMaps_[key]->SetBinError(binY, binZ, bias.error());

      // fill the width profiles
      const Measurement1D& width = results.getWidth();
      widthMaps_[key]->SetBinContent(binY, binZ, width.value());
      widthMaps_[key]->SetBinError(binY, binZ, width.error());

      if (debugMode_) {
        std::cout << "key: " << key << " bin: (" << binY << "," << binZ << ")  bias: " << bias.value()
                  << " (in histo: " << meanMaps_[key]->GetBinContent(binY, binZ) << ") width: " << width.value()
                  << " (in histo: " << widthMaps_[key]->GetBinContent(binY, binZ) << ")" << std::endl;
      }
    }
  }
}

//-----------------------------------------------------------------------------------
void bookHistos(const diMuonMassBias::histo2DMap& harvestTargets_,
                diMuonMassBias::histoMap& meanHistos_,
                diMuonMassBias::histoMap& widthHistos_,
                const unsigned int counter)
//-----------------------------------------------------------------------------------
{
  for (const auto& [key, ME] : harvestTargets_) {
    if (ME == nullptr) {
      std::cout << "could not find MonitorElement for key: " << key << std::endl;
      continue;
    }

    const auto& title = ME->GetTitle();
    const auto& xtitle = ME->GetYaxis()->GetTitle();
    const auto& ytitle = ME->GetXaxis()->GetTitle();

    const auto& nxbins = ME->GetNbinsY();
    const auto& xmin = ME->GetYaxis()->GetXmin();
    const auto& xmax = ME->GetYaxis()->GetXmax();

    if (debugMode_) {
      std::cout << "Booking " << key << std::endl;
    }

    TH1F* meanToBook = new TH1F(fmt::format("Mean_{}_{}", counter, key).c_str(),
                                fmt::format("{};{};#LT M_{{#mu^{{-}}#mu^{{+}}}} #GT [GeV]", title, xtitle).c_str(),
                                nxbins,
                                xmin,
                                xmax);

    if (debugMode_) {
      std::cout << "after creating mean" << key << std::endl;
    }

    meanHistos_.insert({key, meanToBook});

    if (debugMode_) {
      std::cout << "after inserting mean" << key << std::endl;
    }

    TH1F* sigmaToBook = new TH1F(fmt::format("Sigma_{}_{}", counter, key).c_str(),
                                 fmt::format("{};{};#sigma of  M_{{#mu^{{-}}#mu^{{+}}}} [GeV]", title, xtitle).c_str(),
                                 nxbins,
                                 xmin,
                                 xmax);

    if (debugMode_) {
      std::cout << "after creating sigma" << key << std::endl;
    }

    widthHistos_.insert({key, sigmaToBook});

    if (debugMode_) {
      std::cout << "after inserting sigma" << key << std::endl;
    }
  }
}

//-----------------------------------------------------------------------------------
void getMEsToHarvest(diMuonMassBias::histo2DMap& harvestTargets_,
                     diMuonMassBias::histo3DMap& harvestTargets3D_,
                     TFile* file)
//-----------------------------------------------------------------------------------
{
  std::string inFolder = "DiMuonMassValidation";

  // list of 2D histograms to flatten
  std::vector<std::string> MEtoHarvest_ = {"th2d_mass_CosThetaCS",
                                           "th2d_mass_DeltaEta",
                                           "th2d_mass_EtaMinus",
                                           "th2d_mass_EtaPlus",
                                           "th2d_mass_PhiCS",
                                           "th2d_mass_PhiMinus",
                                           "th2d_mass_PhiPlus",
                                           "TkTkMassVsPhiMinusInEtaBins/th2d_mass_PhiMinus_barrel-barrel",
                                           "TkTkMassVsPhiMinusInEtaBins/th2d_mass_PhiMinus_barrel-forward",
                                           "TkTkMassVsPhiMinusInEtaBins/th2d_mass_PhiMinus_barrel-backward",
                                           "TkTkMassVsPhiMinusInEtaBins/th2d_mass_PhiMinus_forward-forward",
                                           "TkTkMassVsPhiMinusInEtaBins/th2d_mass_PhiMinus_backward-backward",
                                           "TkTkMassVsPhiMinusInEtaBins/th2d_mass_PhiMinus_forward-backward",
                                           "TkTkMassVsPhiPlusInEtaBins/th2d_mass_PhiPlus_barrel-barrel",
                                           "TkTkMassVsPhiPlusInEtaBins/th2d_mass_PhiPlus_barrel-forward",
                                           "TkTkMassVsPhiPlusInEtaBins/th2d_mass_PhiPlus_barrel-backward",
                                           "TkTkMassVsPhiPlusInEtaBins/th2d_mass_PhiPlus_forward-forward",
                                           "TkTkMassVsPhiPlusInEtaBins/th2d_mass_PhiPlus_backward-backward",
                                           "TkTkMassVsPhiPlusInEtaBins/th2d_mass_PhiPlus_forward-backward"};

  //loop on the list of histograms to harvest
  for (const auto& hname : MEtoHarvest_) {
    std::cout << "trying to get: " << hname << std::endl;
    TH2F* toHarvest = (TH2F*)file->Get((inFolder + "/" + hname).c_str());

    if (toHarvest == nullptr) {
      std::cout << "could not find input MonitorElement: " << inFolder + "/" + hname << std::endl;
      continue;
    }
    harvestTargets_.insert({hname, toHarvest});
  }

  // list of 3D histograms to flatten
  std::vector<std::string> MEtoHarvest3D_ = {"th3d_mass_vs_eta_phi_plus", "th3d_mass_vs_eta_phi_minus"};

  for (const auto& hname : MEtoHarvest3D_) {
    std::cout << "trying to get: " << hname << std::endl;
    TH3F* toHarvest = (TH3F*)file->Get((inFolder + "/" + hname).c_str());

    if (toHarvest == nullptr) {
      std::cout << "could not find input MonitorElement: " << inFolder + "/" + hname << std::endl;
      continue;
    }
    harvestTargets3D_.insert({hname, toHarvest});
  }
}

/************************************************/
void producePlots(const std::vector<diMuonMassBias::histoMap>& inputMap,
                  const std::vector<std::string>& MEtoHarvest,
                  const std::vector<TString>& labels,
                  const TString& Rlabel,
                  const bool useAutoLimits,
                  const bool isWidth)
/************************************************/
{
  int W = 800;
  int H = 800;
  // references for T, B, L, R
  float T = 0.08 * H;
  float B = 0.12 * H;
  float L = 0.12 * W;
  float R = 0.04 * W;

  // Draw the legend
  TLegend* infoBox = new TLegend(0.65, 0.75, 0.95, 0.90, "");
  infoBox->SetShadowColor(0);  // 0 = transparent
  infoBox->SetFillColor(kWhite);
  infoBox->SetTextSize(0.035);

  // get the extrema
  diMuonMassBias::extremaMap extrema = getExtrema(inputMap, 3.f);

  for (const auto& var : MEtoHarvest) {
    TCanvas* c = new TCanvas(
        ((isWidth ? "width_" : "mean_") + var).c_str(), ((isWidth ? "width_" : "mean_") + var).c_str(), W, H);
    c->SetFillColor(0);
    c->SetBorderMode(0);
    c->SetFrameFillStyle(0);
    c->SetFrameBorderMode(0);
    c->SetLeftMargin(L / W + 0.05);
    c->SetRightMargin(R / W);
    c->SetTopMargin(T / H);
    c->SetBottomMargin(B / H);
    c->SetTickx(0);
    c->SetTicky(0);
    c->SetGrid();

    unsigned int count{0};

    for (const auto& histoMap : inputMap) {
      if (debugMode_) {
        std::cout << var << "  n.bins: " << histoMap.at(var)->GetNbinsX()
                  << " entries: " << histoMap.at(var)->GetEntries() << "title: " << histoMap.at(var)->GetTitle()
                  << " x-axis title: " << histoMap.at(var)->GetXaxis()->GetTitle() << std::endl;
      }

      if (debugMode_) {
        for (int bin = 1; bin <= histoMap.at(var)->GetNbinsX(); bin++) {
          std::cout << var << " bin " << bin << " : " << histoMap.at(var)->GetBinContent(bin) << " +/-"
                    << histoMap.at(var)->GetBinError(bin) << std::endl;
        }
      }

      //histoMap.at(var)->SaveAs((var+".root").c_str());

      histoMap.at(var)->SetLineColor(def_colors[count]);
      histoMap.at(var)->SetMarkerColor(def_colors[count]);
      histoMap.at(var)->SetMarkerStyle(def_markers[count]);
      histoMap.at(var)->SetMarkerSize(1.5);

      // set the limits
      if (!useAutoLimits) {
        if (isWidth) {
          // for width resolution between 0.5 and 2.8
          histoMap.at(var)->GetYaxis()->SetRangeUser(0.5, 2.85);
        } else {
          // for mass between 90.5 and 91.5
          histoMap.at(var)->GetYaxis()->SetRangeUser(90.5, 91.5);
        }
      } else {
        histoMap.at(var)->GetYaxis()->SetRangeUser(extrema.at(var).first, extrema.at(var).second);
      }

      MakeNicePlotStyle<TH1>(histoMap.at(var));

      c->cd();
      if (count == 0) {
        histoMap.at(var)->Draw("E1");
      } else {
        histoMap.at(var)->Draw("E1same");
      }

      // fill the legend only if that's the first element in the vector of variables
      if (var == MEtoHarvest[0]) {
        infoBox->AddEntry(histoMap.at(var), labels[count], "LP");
      }
      infoBox->Draw("same");
      count++;
    }

    CMS_lumi(c, 0, 3, Rlabel);

    // Find the position of the first '/'
    size_t pos = var.find('/');
    std::string outputName{var};

    // Check if '/' is found
    if (pos != std::string::npos) {
      // Erase the substring before the '/' (including the '/')
      outputName.erase(0, pos + 1);
    }

    c->SaveAs(((isWidth ? "width_" : "mean_") + outputName + ".png").c_str());
    c->SaveAs(((isWidth ? "width_" : "mean_") + outputName + ".pdf").c_str());
  }
}

/************************************************/
void produceMaps(const std::vector<diMuonMassBias::histo2DMap>& inputMap,
                 const std::vector<std::string>& MEtoHarvest,
                 const std::vector<TString>& labels,
                 const TString& Rlabel,
                 const bool useAutoLimits,
                 const bool isWidth)
/************************************************/
{
  const auto& sides = getClosestFactors(labels.size());

  if (debugMode_) {
    std::cout << "SIDES:" << sides.second << " :" << sides.first << std::endl;
  }

  int W = 800 * sides.second;
  int H = 600 * sides.first;
  // references for T, B, L, R
  float T = 0.08 * H;
  float B = 0.12 * H;
  float L = 0.12 * W;
  float R = 0.04 * W;

  // Draw the legend
  TLegend* infoBox = new TLegend(0.65, 0.75, 0.95, 0.90, "");
  infoBox->SetShadowColor(0);  // 0 = transparent
  infoBox->SetFillColor(kWhite);
  infoBox->SetTextSize(0.035);

  // get the extrema
  diMuonMassBias::extremaMap extrema = getExtrema(inputMap, 0.f);

  for (const auto& var : MEtoHarvest) {
    TCanvas* c = new TCanvas(
        ((isWidth ? "width_" : "mean_") + var).c_str(), ((isWidth ? "width_" : "mean_") + var).c_str(), W, H);

    c->SetFillColor(0);
    c->SetBorderMode(0);
    c->SetFrameFillStyle(0);
    c->SetFrameBorderMode(0);
    c->SetLeftMargin(L / W);
    c->SetRightMargin(R / W);
    c->SetTopMargin(T / H);
    c->SetBottomMargin(B / H);
    c->SetTickx(0);
    c->SetTicky(0);

    c->Divide(sides.second, sides.first);

    unsigned int count{0};
    for (const auto& histoMap : inputMap) {
      if (debugMode_) {
        std::cout << var << "  n.bins: " << histoMap.at(var)->GetNbinsX()
                  << " entries: " << histoMap.at(var)->GetEntries() << "title: " << histoMap.at(var)->GetTitle()
                  << " x-axis title: " << histoMap.at(var)->GetXaxis()->GetTitle() << std::endl;
      }

      if (debugMode_) {
        for (int bin = 1; bin <= histoMap.at(var)->GetNbinsX(); bin++) {
          std::cout << var << " bin " << bin << " : " << histoMap.at(var)->GetBinContent(bin) << " +/-"
                    << histoMap.at(var)->GetBinError(bin) << std::endl;
        }
      }

      // set the limits
      if (!useAutoLimits) {
        if (isWidth) {
          // for width resolution between 0.5 and 2.8
          histoMap.at(var)->GetZaxis()->SetRangeUser(0.5, 2.85);
        } else {
          // for mass between 90.5 and 91.5
          histoMap.at(var)->GetZaxis()->SetRangeUser(90., 92.);
        }
      } else {
        histoMap.at(var)->GetZaxis()->SetRangeUser(extrema.at(var).first, extrema.at(var).second);
      }

      MakeNicePlotStyle<TH2>(histoMap.at(var));

      c->cd(count + 1);
      c->cd(count + 1)->SetRightMargin(0.15);
      histoMap.at(var)->Draw("colz");

      // fill the legend only if that's the first element in the vector of variables
      if (var == MEtoHarvest[0]) {
        infoBox->AddEntry(histoMap.at(var), labels[count], "LP");
      }
      //infoBox->Draw("same");

      CMS_lumi(dynamic_cast<TPad*>(gPad), 0, 3, labels[count]);
      count++;
    }

    // Find the position of the first '/'
    size_t pos = var.find('/');
    std::string outputName{var};

    // Check if '/' is found
    if (pos != std::string::npos) {
      // Erase the substring before the '/' (including the '/')
      outputName.erase(0, pos + 1);
    }

    c->SaveAs(((isWidth ? "width_" : "mean_") + outputName + ".png").c_str());
    c->SaveAs(((isWidth ? "width_" : "mean_") + outputName + ".pdf").c_str());
  }
}

/************************************************/
void DiMuonMassProfiles(TString namesandlabels, const TString& Rlabel = "", const bool useAutoLimits = false)
/************************************************/
{
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  std::vector<TString> labels;
  std::vector<TFile*> sourceFiles;

  namesandlabels.Remove(TString::kTrailing, ',');
  TObjArray* nameandlabelpairs = namesandlabels.Tokenize(",");
  for (Int_t i = 0; i < nameandlabelpairs->GetEntries(); ++i) {
    TObjArray* aFileLegPair = TString(nameandlabelpairs->At(i)->GetName()).Tokenize("=");
    if (aFileLegPair->GetEntries() == 2) {
      sourceFiles.push_back(TFile::Open(aFileLegPair->At(0)->GetName(), "READ"));
      TObjString* s_label = (TObjString*)aFileLegPair->At(1);
      labels.push_back(s_label->String());
    } else {
      std::cout << "Please give file name and legend entry in the following form:\n"
                << " filename1=legendentry1,filename2=legendentry2\n";
      return;
    }
  }

  // for the bias plots
  std::vector<diMuonMassBias::histoMap> v_meanHistos;
  std::vector<diMuonMassBias::histoMap> v_widthHistos;

  // for the maps
  std::vector<diMuonMassBias::histo2DMap> v_meanMaps;
  std::vector<diMuonMassBias::histo2DMap> v_widthMaps;

  unsigned int countFiles{0};
  for (const auto& file : sourceFiles) {
    diMuonMassBias::histo2DMap harvestTargets;
    diMuonMassBias::histo3DMap harvestTargets3D;

    getMEsToHarvest(harvestTargets, harvestTargets3D, file);

    diMuonMassBias::histoMap meanHistos;
    diMuonMassBias::histoMap widthHistos;

    bookHistos(harvestTargets, meanHistos, widthHistos, countFiles);

    diMuonMassBias::histo2DMap meanMaps;
    diMuonMassBias::histo2DMap widthMaps;

    bookMaps(harvestTargets3D, meanMaps, widthMaps, countFiles);

    for (const auto& element : harvestTargets) {
      fitAndFillHisto(element, meanHistos, widthHistos);
    }

    v_meanHistos.push_back(meanHistos);
    v_widthHistos.push_back(widthHistos);

    for (const auto& element : harvestTargets3D) {
      fitAndFillMap(element, meanMaps, widthMaps);
    }

    v_meanMaps.push_back(meanMaps);
    v_widthMaps.push_back(widthMaps);

    countFiles++;
  }

  // now do the plotting
  std::vector<std::string> MEtoHarvest = {"th2d_mass_CosThetaCS",
                                          "th2d_mass_DeltaEta",
                                          "th2d_mass_EtaMinus",
                                          "th2d_mass_EtaPlus",
                                          "th2d_mass_PhiCS",
                                          "th2d_mass_PhiMinus",
                                          "th2d_mass_PhiPlus",
                                          "TkTkMassVsPhiMinusInEtaBins/th2d_mass_PhiMinus_barrel-barrel",
                                          "TkTkMassVsPhiMinusInEtaBins/th2d_mass_PhiMinus_barrel-forward",
                                          "TkTkMassVsPhiMinusInEtaBins/th2d_mass_PhiMinus_barrel-backward",
                                          "TkTkMassVsPhiMinusInEtaBins/th2d_mass_PhiMinus_forward-forward",
                                          "TkTkMassVsPhiMinusInEtaBins/th2d_mass_PhiMinus_backward-backward",
                                          "TkTkMassVsPhiPlusInEtaBins/th2d_mass_PhiPlus_barrel-barrel",
                                          "TkTkMassVsPhiPlusInEtaBins/th2d_mass_PhiPlus_barrel-forward",
                                          "TkTkMassVsPhiPlusInEtaBins/th2d_mass_PhiPlus_barrel-backward",
                                          "TkTkMassVsPhiPlusInEtaBins/th2d_mass_PhiPlus_forward-forward",
                                          "TkTkMassVsPhiPlusInEtaBins/th2d_mass_PhiPlus_backward-backward",
                                          "TkTkMassVsPhiPlusInEtaBins/th2d_mass_PhiPlus_forward-backward"};

  producePlots(v_meanHistos, MEtoHarvest, labels, Rlabel, useAutoLimits, false);
  producePlots(v_widthHistos, MEtoHarvest, labels, Rlabel, useAutoLimits, true);

  std::vector<std::string> MEtoHarvest3D = {"th3d_mass_vs_eta_phi_plus", "th3d_mass_vs_eta_phi_minus"};

  produceMaps(v_meanMaps, MEtoHarvest3D, labels, Rlabel, useAutoLimits, false);
  produceMaps(v_widthMaps, MEtoHarvest3D, labels, Rlabel, useAutoLimits, true);

  // finally close the file
  for (const auto& file : sourceFiles) {
    file->Close();
  }
}
-- dummy change --
