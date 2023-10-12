#include "TArrow.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TCut.h"
#include "TDatime.h"
#include "TError.h"
#include "TF1.h"
#include "TFile.h"
#include "TGaxis.h"
#include "TGraphErrors.h"
#include "TH1.h"
#include "TH2.h"
#include "THStack.h"
#include "TLegend.h"
#include "TLegendEntry.h"
#include "TList.h"
#include "TMath.h"
#include "TMinuit.h"
#include "TNtuple.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TPaveStats.h"
#include "TPaveText.h"
#include "TROOT.h"
#include "TSpectrum.h"
#include "TStopwatch.h"
#include "TString.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTimeStamp.h"
#include "TTree.h"
#include "TVectorD.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
//#include "Alignment/OfflineValidation/macros/TkAlStyle.cc"
#include "Alignment/OfflineValidation/macros/CMS_lumi.h"
#define PLOTTING_MACRO  // to remove message logger
#include "Alignment/OfflineValidation/interface/PVValidationHelpers.h"

/* 
   This is an auxilliary class to store the list of files
   to be used to plot
*/

class PVValidationVariables {
public:
  PVValidationVariables(TString fileName, TString baseDir, TString legName = "", int color = 1, int style = 1);
  int getLineColor() { return lineColor; }
  int getMarkerStyle() { return markerStyle; }
  int getLineStyle() { return lineStyle; }
  TString getName() { return legendName; }
  TFile *getFile() { return file; }
  TString getFileName() { return fname; }

private:
  TFile *file;
  int lineColor;
  int lineStyle;
  int markerStyle;
  TString legendName;
  TString fname;
};

PVValidationVariables::PVValidationVariables(
    TString fileName, TString baseDir, TString legName, int lColor, int lStyle) {
  fname = fileName;
  lineColor = lColor;

  int ndigits = 1 + std::floor(std::log10(lStyle));
  if (ndigits == 4) {
    lineStyle = lStyle % 100;
    markerStyle = lStyle / 100;
  } else {
    lineStyle = 1;
    markerStyle = lStyle;
  }

  if (legName == "") {
    std::string s_fileName = fileName.Data();
    int start = 0;
    if (s_fileName.find('/'))
      start = s_fileName.find_last_of('/') + 1;
    int stop = s_fileName.find_last_of('.');
    legendName = s_fileName.substr(start, stop - start);
  } else {
    legendName = legName;
  }

  // check if the base dir exists
  file = TFile::Open(fileName.Data(), "READ");

  if (!file) {
    std::cout << "ERROR! file " << fileName.Data() << " does not exist!" << std::endl;
    assert(false);
  }

  if (file->Get(baseDir.Data())) {
    std::cout << "found base directory: " << baseDir.Data() << std::endl;
  } else {
    std::cout << "no directory named: " << baseDir.Data() << std::endl;
    assert(false);
  }
}

/*
  This is an auxilliary enum and typedef used to handle the fit parameter types
*/

namespace params {

  typedef std::pair<Double_t, Double_t> measurement;

  enum estimator { MEAN = 1, WIDTH = 2, MEDIAN = 3, MAD = 4, UNKWN = -1 };
}  // namespace params

/*
  This is an auxilliary struct used to handle the plot limits
*/

struct Limits {
  // initializers list

  Limits()
      : _m_dxyPhiMax(80.),
        _m_dzPhiMax(80.),
        _m_dxyEtaMax(80.),
        _m_dzEtaMax(80.),
        _m_dxyPtMax(80.),
        _m_dzPtMax(80.),
        _m_dxyPhiNormMax(0.5),
        _m_dzPhiNormMax(0.5),
        _m_dxyEtaNormMax(0.5),
        _m_dzEtaNormMax(0.5),
        _m_dxyPtNormMax(0.5),
        _m_dzPtNormMax(0.5),
        _w_dxyPhiMax(120.),
        _w_dzPhiMax(180.),
        _w_dxyEtaMax(120.),
        _w_dzEtaMax(1000.),
        _w_dxyPtMax(120.),
        _w_dzPtMax(180.),
        _w_dxyPhiNormMax(2.0),
        _w_dzPhiNormMax(2.0),
        _w_dxyEtaNormMax(2.0),
        _w_dzEtaNormMax(2.0),
        _w_dxyPtNormMax(2.0),
        _w_dzPtNormMax(2.0) {}

  // getter methods

  std::pair<float, float> get_dxyPhiMax() const {
    std::pair<float, float> res(_m_dxyPhiMax, _w_dxyPhiMax);
    return res;
  }

  std::pair<float, float> get_dzPhiMax() const {
    std::pair<float, float> res(_m_dzPhiMax, _w_dzPhiMax);
    return res;
  }

  std::pair<float, float> get_dxyEtaMax() const {
    std::pair<float, float> res(_m_dxyEtaMax, _w_dxyEtaMax);
    return res;
  }

  std::pair<float, float> get_dzEtaMax() const {
    std::pair<float, float> res(_m_dzEtaMax, _w_dzEtaMax);
    return res;
  }

  std::pair<float, float> get_dxyPtMax() const { return std::make_pair(_m_dxyPtMax, _w_dxyPtMax); }

  std::pair<float, float> get_dzPtMax() const { return std::make_pair(_m_dzPtMax, _w_dzPtMax); }

  std::pair<float, float> get_dxyPhiNormMax() const {
    std::pair<float, float> res(_m_dxyPhiNormMax, _w_dxyPhiNormMax);
    return res;
  }

  std::pair<float, float> get_dzPhiNormMax() const {
    std::pair<float, float> res(_m_dzPhiNormMax, _w_dzPhiNormMax);
    return res;
  }

  std::pair<float, float> get_dxyEtaNormMax() const {
    std::pair<float, float> res(_m_dxyEtaNormMax, _w_dxyEtaNormMax);
    return res;
  }

  std::pair<float, float> get_dzEtaNormMax() const {
    std::pair<float, float> res(_m_dzEtaNormMax, _w_dzEtaNormMax);
    return res;
  }

  std::pair<float, float> get_dxyPtNormMax() const { return std::make_pair(_m_dxyPtNormMax, _w_dxyPtNormMax); }

  std::pair<float, float> get_dzPtNormMax() const { return std::make_pair(_m_dzPtNormMax, _w_dzPtNormMax); }

  // initializes to different values, if needed

  void init(float m_dxyPhiMax,
            float m_dzPhiMax,
            float m_dxyEtaMax,
            float m_dzEtaMax,
            float m_dxyPtMax,
            float m_dzPtMax,
            float m_dxyPhiNormMax,
            float m_dzPhiNormMax,
            float m_dxyEtaNormMax,
            float m_dzEtaNormMax,
            float m_dxyPtNormMax,
            float m_dzPtNormMax,
            float w_dxyPhiMax,
            float w_dzPhiMax,
            float w_dxyEtaMax,
            float w_dzEtaMax,
            float w_dxyPtMax,
            float w_dzPtMax,
            float w_dxyPhiNormMax,
            float w_dzPhiNormMax,
            float w_dxyEtaNormMax,
            float w_dzEtaNormMax,
            float w_dxyPtNormMax,
            float w_dzPtNormMax) {
    _m_dxyPhiMax = m_dxyPhiMax;
    _m_dzPhiMax = m_dzPhiMax;
    _m_dxyEtaMax = m_dxyEtaMax;
    _m_dzEtaMax = m_dzEtaMax;
    _m_dxyPtMax = m_dxyPtMax;
    _m_dzPtMax = m_dzPtMax;
    _m_dxyPhiNormMax = m_dxyPhiNormMax;
    _m_dzPhiNormMax = m_dzPhiNormMax;
    _m_dxyEtaNormMax = m_dxyEtaNormMax;
    _m_dzEtaNormMax = m_dzEtaNormMax;
    _m_dxyPtNormMax = m_dxyPtNormMax;
    _m_dzPtNormMax = m_dzPtNormMax;
    _w_dxyPhiMax = w_dxyPhiMax;
    _w_dzPhiMax = w_dzPhiMax;
    _w_dxyEtaMax = w_dxyEtaMax;
    _w_dzEtaMax = w_dzEtaMax;
    _w_dxyPtMax = w_dxyPtMax;
    _w_dzPtMax = w_dzPtMax;
    _w_dxyPhiNormMax = w_dxyPhiNormMax;
    _w_dzPhiNormMax = w_dzPhiNormMax;
    _w_dxyEtaNormMax = w_dxyEtaNormMax;
    _w_dzEtaNormMax = w_dzEtaNormMax;
    _w_dxyPtNormMax = w_dxyPtNormMax;
    _w_dzPtNormMax = w_dzPtNormMax;
  }

  void printAll() {
    std::cout << "======================================================" << std::endl;
    std::cout << "  The y-axis ranges on the plots will be the following:      " << std::endl;

    std::cout << "  mean of dxy vs Phi:         " << _m_dxyPhiMax << std::endl;
    std::cout << "  mean of dz  vs Phi:         " << _m_dzPhiMax << std::endl;
    std::cout << "  mean of dxy vs Eta:         " << _m_dxyEtaMax << std::endl;
    std::cout << "  mean of dz  vs Eta:         " << _m_dzEtaMax << std::endl;
    std::cout << "  mean of dxy vs Pt :         " << _m_dxyPtMax << std::endl;
    std::cout << "  mean of dz  vs Pt :         " << _m_dzPtMax << std::endl;

    std::cout << "  mean of dxy vs Phi (norm):  " << _m_dxyPhiNormMax << std::endl;
    std::cout << "  mean of dz  vs Phi (norm):  " << _m_dzPhiNormMax << std::endl;
    std::cout << "  mean of dxy vs Eta (norm):  " << _m_dxyEtaNormMax << std::endl;
    std::cout << "  mean of dz  vs Eta (norm):  " << _m_dzEtaNormMax << std::endl;
    std::cout << "  mean of dxy vs Pt  (norm):  " << _m_dxyPtNormMax << std::endl;
    std::cout << "  mean of dz  vs Pt  (norm):  " << _m_dzPtNormMax << std::endl;

    std::cout << "  width of dxy vs Phi:        " << _w_dxyPhiMax << std::endl;
    std::cout << "  width of dz  vs Phi:        " << _w_dzPhiMax << std::endl;
    std::cout << "  width of dxy vs Eta:        " << _w_dxyEtaMax << std::endl;
    std::cout << "  width of dz  vs Eta:        " << _w_dzEtaMax << std::endl;
    std::cout << "  width of dxy vs Pt :        " << _w_dxyPtMax << std::endl;
    std::cout << "  width of dz  vs Pt :        " << _w_dzPtMax << std::endl;

    std::cout << "  width of dxy vs Phi (norm): " << _w_dxyPhiNormMax << std::endl;
    std::cout << "  width of dz  vs Phi (norm): " << _w_dzPhiNormMax << std::endl;
    std::cout << "  width of dxy vs Eta (norm): " << _w_dxyEtaNormMax << std::endl;
    std::cout << "  width of dz  vs Eta (norm): " << _w_dzEtaNormMax << std::endl;
    std::cout << "  width of dxy vs Pt  (norm): " << _w_dxyPtNormMax << std::endl;
    std::cout << "  width of dz  vs Pt  (norm): " << _w_dzPtNormMax << std::endl;

    std::cout << "======================================================" << std::endl;
  }

private:
  float _m_dxyPhiMax;
  float _m_dzPhiMax;
  float _m_dxyEtaMax;
  float _m_dzEtaMax;
  float _m_dxyPtMax;
  float _m_dzPtMax;
  float _m_dxyPhiNormMax;
  float _m_dzPhiNormMax;
  float _m_dxyEtaNormMax;
  float _m_dzEtaNormMax;
  float _m_dxyPtNormMax;
  float _m_dzPtNormMax;

  float _w_dxyPhiMax;
  float _w_dzPhiMax;
  float _w_dxyEtaMax;
  float _w_dzEtaMax;
  float _w_dxyPtMax;
  float _w_dzPtMax;
  float _w_dxyPhiNormMax;
  float _w_dzPhiNormMax;
  float _w_dxyEtaNormMax;
  float _w_dzEtaNormMax;
  float _w_dxyPtNormMax;
  float _w_dzPtNormMax;
};

Limits *thePlotLimits = new Limits();
std::vector<PVValidationVariables *> sourceList;

#define ARRAY_SIZE(array) (sizeof((array)) / sizeof((array[0])))

void arrangeCanvas(TCanvas *canv,
                   TH1F *meanplots[100],
                   TH1F *widthplots[100],
                   Int_t nFiles,
                   TString LegLabels[10],
                   TString theDate = "bogus",
                   bool onlyBias = false,
                   bool setAutoLimits = true);
void arrangeCanvas2D(TCanvas *canv,
                     TH2F *meanmaps[100],
                     TH2F *widthmaps[100],
                     Int_t nFiles,
                     TString LegLabels[10],
                     TString theDate = "bogus");
void arrangeFitCanvas(
    TCanvas *canv, TH1F *meanplots[100], Int_t nFiles, TString LegLabels[10], TString theDate = "bogus");

void arrangeBiasCanvas(TCanvas *canv,
                       TH1F *dxyPhiMeanTrend[100],
                       TH1F *dzPhiMeanTrend[100],
                       TH1F *dxyEtaMeanTrend[100],
                       TH1F *dzEtaMeanTrend[100],
                       Int_t nFiles,
                       TString LegLabels[10],
                       TString theDate = "bogus",
                       bool setAutoLimits = true);

params::measurement getMedian(TH1F *histo);
params::measurement getMAD(TH1F *histo);

std::pair<params::measurement, params::measurement> fitResiduals(TH1 *hist, bool singleTime = false);

Double_t DoubleSidedCB(double *x, double *par);
std::pair<params::measurement, params::measurement> fitResidualsCB(TH1 *hist);

Double_t tp0Fit(Double_t *x, Double_t *par5);
std::pair<params::measurement, params::measurement> fitStudentTResiduals(TH1 *hist);

void FillTrendPlot(TH1F *trendPlot, TH1F *residualsPlot[100], params::estimator firPar_, TString var_, Int_t nbins);

// global (here for the moment)
Int_t nBins_ = 48;
void FillMap(TH2F *trendMap,
             std::vector<std::vector<TH1F *> > residualsMapPlot,
             params::estimator fitPar_,
             const int nBinsX = nBins_,
             const int nBinsY = nBins_);

std::pair<TH2F *, TH2F *> trimTheMap(TH2 *hist);

void MakeNiceTrendPlotStyle(TH1 *hist, Int_t color, Int_t style);
void MakeNicePlotStyle(TH1 *hist);
void MakeNiceMapStyle(TH2 *hist);
void MakeNiceTF1Style(TF1 *f1, Int_t color);

void FitPVResiduals(TString namesandlabels,
                    bool stdres = true,
                    bool do2DMaps = false,
                    TString theDate = "bogus",
                    bool setAutoLimits = true,
                    TString CMSlabel = "",
                    TString Rlabel = "");
TH1F *DrawZero(TH1F *hist, Int_t nbins, Double_t lowedge, Double_t highedge, Int_t iter);
TH1F *DrawConstant(TH1F *hist, Int_t nbins, Double_t lowedge, Double_t highedge, Int_t iter, Double_t theConst);
void makeNewXAxis(TH1F *h);
void makeNewPairOfAxes(TH2F *h);

// ancillary fitting functions
Double_t fULine(Double_t *x, Double_t *par);
Double_t fDLine(Double_t *x, Double_t *par);
void FitULine(TH1 *hist);
void FitDLine(TH1 *hist);

params::measurement getTheRangeUser(TH1F *thePlot, Limits *thePlotLimits, bool tag = false);

void setStyle(TString customCMSLabel = "", TString customRightLabel = "");

// global variables

std::ofstream outfile("FittedDeltaZ.txt");

// use the maximum of the three supported phases
Int_t nLadders_ = 20;
Int_t nModZ_ = 9;

const Int_t nPtBins_ = 48;
Float_t _boundMin = -0.5;
Float_t _boundSx = (nBins_ / 4.) - 0.5;
Float_t _boundDx = 3 * (nBins_ / 4.) - 0.5;
Float_t _boundMax = nBins_ - 0.5;
Float_t etaRange = 2.5;
Float_t minPt_ = 1.;
Float_t maxPt_ = 20.;
bool isDebugMode = false;

// pT binning as in paragraph 3.2 of CMS-PAS-TRK-10-005 (https://cds.cern.ch/record/1279383/files/TRK-10-005-pas.pdf)
// this is the default

std::array<float, nPtBins_ + 1> mypT_bins = {{0.5,  0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,  1.7,
                                              1.8,  1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,  3.0,
                                              3.1,  3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.25, 4.5,
                                              4.75, 5.0, 5.5, 6.,  7.,  8.,  9.,  11., 14., 20.}};

// inline function
int check(const double a[], int n) {
  //while (--n > 0 && a[n] == a[0])    // exact match
  while (--n > 0 && (a[n] - a[0]) < 0.01)  // merged input files, protection agains numerical precision
    ;
  return n != 0;
}

// fill the list of files
//*************************************************************
void loadFileList(const char *inputFile, TString baseDir, TString legendName, int lineColor, int lineStyle)
//*************************************************************
{
  gErrorIgnoreLevel = kFatal;
  sourceList.push_back(new PVValidationVariables(inputFile, baseDir, legendName, lineColor, lineStyle));
}

//*************************************************************
void FitPVResiduals(TString namesandlabels,
                    bool stdres,
                    bool do2DMaps,
                    TString theDate,
                    bool setAutoLimits,
                    TString CMSlabel,
                    TString Rlabel)
//*************************************************************
{
  // only for fatal errors (useful in debugging)
  gErrorIgnoreLevel = kFatal;

  TH1::AddDirectory(kFALSE);
  bool fromLoader = false;

  TTimeStamp start_time;
  TStopwatch timer;
  timer.Start();

  if (!setAutoLimits) {
    std::cout << "FitPVResiduals::FitPVResiduals(): Overriding autolimits!" << std::endl;
    thePlotLimits->printAll();
  } else {
    std::cout << "FitPVResiduals::FitPVResiduals(): plot axis range will be automatically adjusted" << std::endl;
  }

  //TkAlStyle::set(INTERNAL);	// set publication status

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

  Int_t markers[9];
  Int_t colors[9];

  setStyle(CMSlabel, Rlabel);

  // check if the loader is empty
  if (!sourceList.empty()) {
    fromLoader = true;
  }

  // if enters here, whatever is passed from command line is neglected
  if (fromLoader) {
    std::cout << "FitPVResiduals::FitPVResiduals(): file list specified from loader" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "!!    arguments passed from CLI will be neglected   !!" << std::endl;
    std::cout << "======================================================" << std::endl;
    for (std::vector<PVValidationVariables *>::iterator it = sourceList.begin(); it != sourceList.end(); ++it) {
      std::cout << "name:  " << std::setw(20) << (*it)->getName() << " |file:  " << std::setw(15) << (*it)->getFile()
                << " |color: " << std::setw(5) << (*it)->getLineColor() << " |style: " << std::setw(5)
                << (*it)->getLineStyle() << " |marker:" << std::setw(5) << (*it)->getMarkerStyle() << std::endl;
    }
    std::cout << "======================================================" << std::endl;
  }

  Int_t theFileCount = 0;
  TList *FileList = new TList();
  TList *LabelList = new TList();

  if (!fromLoader) {
    namesandlabels.Remove(TString::kTrailing, ',');
    TObjArray *nameandlabelpairs = namesandlabels.Tokenize(",");
    for (Int_t i = 0; i < nameandlabelpairs->GetEntries(); ++i) {
      TObjArray *aFileLegPair = TString(nameandlabelpairs->At(i)->GetName()).Tokenize("=");

      if (aFileLegPair->GetEntries() == 2) {
        FileList->Add(TFile::Open(aFileLegPair->At(0)->GetName(), "READ"));  // 2
        LabelList->Add(aFileLegPair->At(1));
      } else {
        std::cout << "Please give file name and legend entry in the following form:\n"
                  << " filename1=legendentry1,filename2=legendentry2\n";
        exit(EXIT_FAILURE);
      }
    }
    theFileCount = FileList->GetSize();
  } else {
    for (std::vector<PVValidationVariables *>::iterator it = sourceList.begin(); it != sourceList.end(); ++it) {
      //FileList->Add((*it)->getFile()); // was extremely slow
      FileList->Add(TFile::Open((*it)->getFileName(), "READ"));
    }
    theFileCount = sourceList.size();
  }

  if (theFileCount == 0) {
    std::cout << "FitPVResiduals::FitPVResiduals(): empty input file list has been passed." << std::endl;
    exit(EXIT_FAILURE);
  }

  //  const Int_t nFiles_ = FileList->GetSize();
  const Int_t nFiles_ = theFileCount;
  TString LegLabels[10];
  TFile *fins[nFiles_];

  // already used in the global variables
  //const Int_t nBins_ =24;

  for (Int_t j = 0; j < nFiles_; j++) {
    // Retrieve files
    fins[j] = (TFile *)FileList->At(j);

    // Retrieve labels
    if (!fromLoader) {
      TObjString *legend = (TObjString *)LabelList->At(j);
      LegLabels[j] = legend->String();
      markers[j] = def_markers[j];
      colors[j] = def_colors[j];
    } else {
      LegLabels[j] = sourceList[j]->getName();
      markers[j] = sourceList[j]->getMarkerStyle();
      colors[j] = sourceList[j]->getLineColor();
    }
    LegLabels[j].ReplaceAll("_", " ");
    std::cout << "FitPVResiduals::FitPVResiduals(): label[" << j << "] " << LegLabels[j] << std::endl;
  }

  //
  // initialize all the histograms to be taken from file
  //

  // integrated residuals
  TH1F *dxyRefit[nFiles_];
  TH1F *dzRefit[nFiles_];

  TH1F *dxySigRefit[nFiles_];
  TH1F *dzSigRefit[nFiles_];

  // dca absolute residuals
  TH1F *dxyPhiResiduals[nFiles_][nBins_];
  TH1F *dxyEtaResiduals[nFiles_][nBins_];

  TH1F *dzPhiResiduals[nFiles_][nBins_];
  TH1F *dzEtaResiduals[nFiles_][nBins_];

  // dca transvers residuals
  TH1F *dxPhiResiduals[nFiles_][nBins_];
  TH1F *dxEtaResiduals[nFiles_][nBins_];

  TH1F *dyPhiResiduals[nFiles_][nBins_];
  TH1F *dyEtaResiduals[nFiles_][nBins_];

  // dca normalized residuals
  TH1F *dxyNormPhiResiduals[nFiles_][nBins_];
  TH1F *dxyNormEtaResiduals[nFiles_][nBins_];

  TH1F *dzNormPhiResiduals[nFiles_][nBins_];
  TH1F *dzNormEtaResiduals[nFiles_][nBins_];

  // double-differential residuals
  TH1F *dxyMapResiduals[nFiles_][nBins_][nBins_];
  TH1F *dzMapResiduals[nFiles_][nBins_][nBins_];

  TH1F *dxyNormMapResiduals[nFiles_][nBins_][nBins_];
  TH1F *dzNormMapResiduals[nFiles_][nBins_][nBins_];

  // double-differential residuals L1
  TH1F *dxyL1MapResiduals[nFiles_][nLadders_][nModZ_];
  TH1F *dzL1MapResiduals[nFiles_][nLadders_][nModZ_];

  TH1F *dxyL1NormMapResiduals[nFiles_][nLadders_][nModZ_];
  TH1F *dzL1NormMapResiduals[nFiles_][nLadders_][nModZ_];

  // dca residuals vs pT
  TH1F *dzNormPtResiduals[nFiles_][nPtBins_];
  TH1F *dxyNormPtResiduals[nFiles_][nPtBins_];

  TH1F *dzPtResiduals[nFiles_][nPtBins_];
  TH1F *dxyPtResiduals[nFiles_][nPtBins_];

  // dca residuals vs module number/ladder
  TH1F *dzNormLadderResiduals[nFiles_][nLadders_];
  TH1F *dxyNormLadderResiduals[nFiles_][nLadders_];

  TH1F *dzLadderResiduals[nFiles_][nLadders_];
  TH1F *dxyLadderResiduals[nFiles_][nLadders_];

  TH1F *dzNormModZResiduals[nFiles_][nModZ_];
  TH1F *dxyNormModZResiduals[nFiles_][nModZ_];

  TH1F *dzModZResiduals[nFiles_][nModZ_];
  TH1F *dxyModZResiduals[nFiles_][nModZ_];

  // for sanity checks
  TH1F *theEtaHistos[nFiles_];
  TH1F *thebinsHistos[nFiles_];
  TH1F *theLaddersHistos[nFiles_];
  TH1F *theModZHistos[nFiles_];
  TH1F *thePtInfoHistos[nFiles_];

  double theEtaMax_[nFiles_];
  double theNBINS[nFiles_];
  double theLadders[nFiles_];
  double theModZ[nFiles_];

  double thePtMax[nFiles_];
  double thePtMin[nFiles_];
  double thePTBINS[nFiles_];

  TTimeStamp initialization_done;

  if (isDebugMode) {
    timer.Stop();
    std::cout << "check point 1: " << timer.CpuTime() << " " << timer.RealTime() << std::endl;
    timer.Continue();
  }

  for (Int_t i = 0; i < nFiles_; i++) {
    fins[i]->cd("PVValidation/EventFeatures/");

    if (gDirectory->GetListOfKeys()->Contains("etaMax")) {
      gDirectory->GetObject("etaMax", theEtaHistos[i]);
      theEtaMax_[i] = theEtaHistos[i]->GetBinContent(1) / theEtaHistos[i]->GetEntries();
      std::cout << "File n. " << i << " has theEtaMax[" << i << "] = " << theEtaMax_[i] << std::endl;
    } else {
      theEtaMax_[i] = 2.5;
      std::cout << "File n. " << i << " getting the default pseudo-rapidity range: " << theEtaMax_[i] << std::endl;
    }

    if (gDirectory->GetListOfKeys()->Contains("nbins")) {
      gDirectory->GetObject("nbins", thebinsHistos[i]);
      theNBINS[i] = thebinsHistos[i]->GetBinContent(1) / thebinsHistos[i]->GetEntries();
      std::cout << "File n. " << i << " has theNBINS[" << i << "] = " << theNBINS[i] << std::endl;
    } else {
      theNBINS[i] = 48.;
      std::cout << "File n. " << i << " getting the default n. of bins: " << theNBINS[i] << std::endl;
    }

    if (gDirectory->GetListOfKeys()->Contains("nladders")) {
      gDirectory->GetObject("nladders", theLaddersHistos[i]);
      theLadders[i] = theLaddersHistos[i]->GetBinContent(1) / theLaddersHistos[i]->GetEntries();
      std::cout << "File n. " << i << " has theNLadders[" << i << "] = " << theLadders[i] << std::endl;
    } else {
      theLadders[i] = -1.;
      std::cout << "File n. " << i << " getting the default n. ladders: " << theLadders[i] << std::endl;
    }

    if (gDirectory->GetListOfKeys()->Contains("nModZ")) {
      gDirectory->GetObject("nModZ", theModZHistos[i]);
      theModZ[i] = theModZHistos[i]->GetBinContent(1) / theModZHistos[i]->GetEntries();
      std::cout << "File n. " << i << " has theNModZ[" << i << "] = " << theModZ[i] << std::endl;
    } else {
      theModZ[i] = -1.;
      std::cout << "File n. " << i << " getting the default n. modules along Z: " << theModZ[i] << std::endl;
    }

    if (gDirectory->GetListOfKeys()->Contains("pTinfo")) {
      gDirectory->GetObject("pTinfo", thePtInfoHistos[i]);
      thePTBINS[i] = thePtInfoHistos[i]->GetBinContent(1) * 3. / thePtInfoHistos[i]->GetEntries();
      ;
      thePtMin[i] = thePtInfoHistos[i]->GetBinContent(2) * 3. / thePtInfoHistos[i]->GetEntries();
      thePtMax[i] = thePtInfoHistos[i]->GetBinContent(3) * 3. / thePtInfoHistos[i]->GetEntries();
      std::cout << "File n. " << i << " has thePTBINS[" << i << "] = " << thePTBINS[i] << " pT min:  " << thePtMin[i]
                << " pT max: " << thePtMax[i] << std::endl;
    } else {
      // if there is no histogram then set the extremes to 0, but the n. of bins should be still the default
      // this protects running against the old file format.
      thePTBINS[i] = nPtBins_;
      thePtMin[i] = 0.;
      thePtMax[i] = 0.;
      std::cout << "File n. " << i << " getting the default pT binning: ";
      for (const auto &bin : mypT_bins) {
        std::cout << bin << " ";
      }
      std::cout << std::endl;
    }

    // get the non-differential residuals plots

    fins[i]->cd("PVValidation/ProbeTrackFeatures");

    gDirectory->GetObject("h_probedxyRefitV", dxyRefit[i]);
    gDirectory->GetObject("h_probedzRefitV", dzRefit[i]);

    gDirectory->GetObject("h_probeRefitVSigXY", dxySigRefit[i]);
    gDirectory->GetObject("h_probeRefitVSigZ", dzSigRefit[i]);

    for (Int_t j = 0; j < theNBINS[i]; j++) {
      if (stdres) {
        // DCA absolute residuals

        fins[i]->cd("PVValidation/Abs_Transv_Phi_Residuals/");
        gDirectory->GetObject(Form("histo_dxy_phi_plot%i", j), dxyPhiResiduals[i][j]);
        gDirectory->GetObject(Form("histo_dx_phi_plot%i", j), dxPhiResiduals[i][j]);
        gDirectory->GetObject(Form("histo_dy_phi_plot%i", j), dyPhiResiduals[i][j]);

        fins[i]->cd("PVValidation/Abs_Transv_Eta_Residuals/");
        gDirectory->GetObject(Form("histo_dxy_eta_plot%i", j), dxyEtaResiduals[i][j]);
        gDirectory->GetObject(Form("histo_dx_eta_plot%i", j), dxEtaResiduals[i][j]);
        gDirectory->GetObject(Form("histo_dy_eta_plot%i", j), dyEtaResiduals[i][j]);

        dzPhiResiduals[i][j] = (TH1F *)fins[i]->Get(Form("PVValidation/Abs_Long_Phi_Residuals/histo_dz_phi_plot%i", j));
        dzEtaResiduals[i][j] = (TH1F *)fins[i]->Get(Form("PVValidation/Abs_Long_Eta_Residuals/histo_dz_eta_plot%i", j));

        // DCA normalized residuals
        dxyNormPhiResiduals[i][j] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Norm_Transv_Phi_Residuals/histo_norm_dxy_phi_plot%i", j));
        dxyNormEtaResiduals[i][j] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Norm_Transv_Eta_Residuals/histo_norm_dxy_eta_plot%i", j));
        dzNormPhiResiduals[i][j] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Norm_Long_Phi_Residuals/histo_norm_dz_phi_plot%i", j));
        dzNormEtaResiduals[i][j] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Norm_Long_Eta_Residuals/histo_norm_dz_eta_plot%i", j));

        // double differential residuals

        if (do2DMaps) {
          for (Int_t k = 0; k < theNBINS[i]; k++) {
            // absolute residuals
            fins[i]->cd("PVValidation/Abs_DoubleDiffResiduals/");
            gDirectory->GetObject(Form("histo_dxy_eta_plot%i_phi_plot%i", j, k), dxyMapResiduals[i][j][k]);
            gDirectory->GetObject(Form("histo_dz_eta_plot%i_phi_plot%i", j, k), dzMapResiduals[i][j][k]);

            // normalized residuals
            fins[i]->cd("PVValidation/Norm_DoubleDiffResiduals/");
            gDirectory->GetObject(Form("histo_norm_dxy_eta_plot%i_phi_plot%i", j, k), dxyNormMapResiduals[i][j][k]);
            gDirectory->GetObject(Form("histo_norm_dz_eta_plot%i_phi_plot%i", j, k), dzNormMapResiduals[i][j][k]);
          }
        }
      } else {
        // DCA absolute residuals
        dxyPhiResiduals[i][j] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Abs_Transv_Phi_Residuals/histo_IP2D_phi_plot%i", j));
        dxyEtaResiduals[i][j] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Abs_Transv_Eta_Residuals/histo_IP2D_eta_plot%i", j));
        dzPhiResiduals[i][j] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Abs_Long_Phi_Residuals/histo_resz_phi_plot%i", j));
        dzEtaResiduals[i][j] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Abs_Long_Eta_Residuals/histo_resz_eta_plot%i", j));

        // DCA normalized residuals
        dxyNormPhiResiduals[i][j] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Norm_Transv_Phi_Residuals/histo_norm_IP2D_phi_plot%i", j));
        dxyNormEtaResiduals[i][j] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Norm_Transv_Eta_Residuals/histo_norm_IP2D_eta_plot%i", j));
        dzNormPhiResiduals[i][j] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Norm_Long_Phi_Residuals/histo_norm_resz_phi_plot%i", j));
        dzNormEtaResiduals[i][j] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Norm_Long_Eta_Residuals/histo_norm_resz_eta_plot%i", j));

        // double differential residuals
        if (do2DMaps) {
          for (Int_t k = 0; k < theNBINS[i]; k++) {
            // absolute residuals
            fins[i]->cd("PVValidation/Abs_DoubleDiffResiduals");
            gDirectory->GetObject(Form("PVValidation/Abs_DoubleDiffResiduals/histo_dxy_eta_plot%i_phi_plot%i", j, k),
                                  dxyMapResiduals[i][j][k]);
            gDirectory->GetObject(Form("PVValidation/Abs_DoubleDiffResiduals/histo_dz_eta_plot%i_phi_plot%i", j, k),
                                  dzMapResiduals[i][j][k]);

            // normalized residuals
            fins[i]->cd("PVValidation/Norm_DoubleDiffResiduals");
            gDirectory->GetObject(
                Form("PVValidation/Norm_DoubleDiffResiduals/histo_norm_dxy_eta_plot%i_phi_plot%i", j, k),
                dxyNormMapResiduals[i][j][k]);
            gDirectory->GetObject(
                Form("PVValidation/Norm_DoubleDiffResiduals/histo_norm_dz_eta_plot%i_phi_plot%i", j, k),
                dzNormMapResiduals[i][j][k]);
          }
        }  // if do2DMaps
      }
    }

    // residuals vs pT

    for (Int_t l = 0; l < thePTBINS[i] - 1; l++) {
      dxyPtResiduals[i][l] = (TH1F *)fins[i]->Get(Form("PVValidation/Abs_Transv_pT_Residuals/histo_dxy_pT_plot%i", l));
      dzPtResiduals[i][l] = (TH1F *)fins[i]->Get(Form("PVValidation/Abs_Long_pT_Residuals/histo_dz_pT_plot%i", l));

      dxyNormPtResiduals[i][l] =
          (TH1F *)fins[i]->Get(Form("PVValidation/Norm_Transv_pT_Residuals/histo_norm_dxy_pT_plot%i", l));
      dzNormPtResiduals[i][l] =
          (TH1F *)fins[i]->Get(Form("PVValidation/Norm_Long_pT_Residuals/histo_norm_dz_pT_plot%i", l));
    }

    // residuals vs module number / ladder

    if (theLadders[i] > 0 && theModZ[i] > 0) {
      for (Int_t iLadder = 0; iLadder < theLadders[i]; iLadder++) {
        dzNormLadderResiduals[i][iLadder] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Norm_Long_ladder_Residuals/histo_norm_dz_ladder_plot%i", iLadder));
        dxyNormLadderResiduals[i][iLadder] = (TH1F *)fins[i]->Get(
            Form("PVValidation/Norm_Transv_ladder_Residuals/histo_norm_dxy_ladder_plot%i", iLadder));

        dzLadderResiduals[i][iLadder] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Abs_Long_ladder_Residuals/histo_dz_ladder_plot%i", iLadder));
        dxyLadderResiduals[i][iLadder] = (TH1F *)fins[i]->Get(
            Form("PVValidation/Abs_Transv_ladderNoOverlap_Residuals/histo_dxy_ladder_plot%i", iLadder));

        if (do2DMaps) {
          for (Int_t iMod = 0; iMod < theModZ[i]; iMod++) {
            dxyL1MapResiduals[i][iLadder][iMod] =
                (TH1F *)fins[i]->Get(Form("PVValidation/Abs_L1Residuals/histo_dxy_ladder%i_module%i", iLadder, iMod));
            dzL1MapResiduals[i][iLadder][iMod] =
                (TH1F *)fins[i]->Get(Form("PVValidation/Abs_L1Residuals/histo_dz_ladder%i_module%i", iLadder, iMod));

            dxyL1NormMapResiduals[i][iLadder][iMod] = (TH1F *)fins[i]->Get(
                Form("PVValidation/Norm_L1Residuals/histo_norm_dxy_ladder%i_module%i", iLadder, iMod));
            dzL1NormMapResiduals[i][iLadder][iMod] = (TH1F *)fins[i]->Get(
                Form("PVValidation/Norm_L1Residuals/histo_norm_dz_ladder%i_module%i", iLadder, iMod));
          }
        }
      }
    }

    if (theModZ[i] > 0) {
      for (Int_t iMod = 0; iMod < theModZ[i]; iMod++) {
        dzNormModZResiduals[i][iMod] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Norm_Long_modZ_Residuals/histo_norm_dz_modZ_plot%i", iMod));
        dxyNormModZResiduals[i][iMod] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Norm_Transv_modZ_Residuals/histo_norm_dxy_modZ_plot%i", iMod));

        dzModZResiduals[i][iMod] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Abs_Long_modZ_Residuals/histo_dz_modZ_plot%i", iMod));
        dxyModZResiduals[i][iMod] =
            (TH1F *)fins[i]->Get(Form("PVValidation/Abs_Transv_modZ_Residuals/histo_dxy_modZ_plot%i", iMod));
      }
    }

    // close the files after retrieving them
    fins[i]->Close();
  }

  TTimeStamp caching_done;

  if (isDebugMode) {
    timer.Stop();
    std::cout << "check point 2: " << timer.CpuTime() << " " << timer.RealTime() << std::endl;
    timer.Continue();
  }

  // checks if all pseudo-rapidity ranges coincide
  // if not, exits
  if (check(theEtaMax_, nFiles_)) {
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResiduals::FitPVResiduals(): the eta range is different" << std::endl;
    std::cout << "exiting..." << std::endl;
    exit(EXIT_FAILURE);
  } else {
    etaRange = theEtaMax_[0];
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResiduals::FitPVResiduals(): the eta range is [" << -etaRange << " ; " << etaRange << "]"
              << std::endl;
    std::cout << "======================================================" << std::endl;
  }

  // checks if all nbins ranges coincide
  // if not, exits
  if (check(theNBINS, nFiles_)) {
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResiduals::FitPVResiduals(): the number of bins is different" << std::endl;
    std::cout << "exiting..." << std::endl;
    exit(EXIT_FAILURE);
  } else {
    nBins_ = theNBINS[0];

    // adjust also the limit for the fit
    _boundSx = (nBins_ / 4.) - 0.5;
    _boundDx = 3 * (nBins_ / 4.) - 0.5;
    _boundMax = nBins_ - 0.5;

    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResiduals::FitPVResiduals(): the number of bins is: " << nBins_ << std::endl;
    std::cout << "======================================================" << std::endl;
  }

  // checks if the geometries are consistent to produce the ladder plots
  if (check(theLadders, nFiles_)) {
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResiduals::FitPVResiduals(): the number of ladders is different" << std::endl;
    std::cout << "won't do the ladder analysis..." << std::endl;
    std::cout << "======================================================" << std::endl;
    nLadders_ = -1;
  } else {
    nLadders_ = theLadders[0];
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResiduals::FitPVResiduals(): the number of ladders is: " << nLadders_ << std::endl;
    std::cout << "======================================================" << std::endl;
  }

  // checks if the geometries are consistent to produce the moduleZ plots
  if (check(theModZ, nFiles_)) {
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResiduals::FitPVResiduals(): the number of modules in Z is different" << std::endl;
    std::cout << "won't do the ladder analysis..." << std::endl;
    std::cout << "======================================================" << std::endl;
    nModZ_ = -1;
  } else {
    nModZ_ = theModZ[0];
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResiduals::FitPVResiduals(): the number of modules in Z is: " << nModZ_ << std::endl;
    std::cout << "======================================================" << std::endl;
  }

  // checks if pT boundaries are consistent to produce the pT-binned plots
  if (check(thePtMax, nFiles_) || check(thePtMin, nFiles_)) {
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResiduals::FitPVResiduals(): the pT binning is different" << std::endl;
    std::cout << "won't do the pT analysis..." << std::endl;
    std::cout << "======================================================" << std::endl;
    minPt_ = -1.;
  } else {
    if (thePtMin[0] != 0.) {
      minPt_ = thePtMin[0];
      maxPt_ = thePtMax[0];
      mypT_bins = PVValHelper::makeLogBins<float, nPtBins_>(thePtMin[0], thePtMax[0]);
      std::cout << "======================================================" << std::endl;
      std::cout << "FitPVResiduals::FitPVResiduals(): log bins [" << thePtMin[0] << "," << thePtMax[0] << "]"
                << std::endl;
      std::cout << "======================================================" << std::endl;
    } else {
      std::cout << "======================================================" << std::endl;
      std::cout << "FitPVResiduals::FitPVResiduals(): using default bins ";
      for (const auto &bin : mypT_bins) {
        std::cout << bin << " ";
      }
      std::cout << std::endl;
      std::cout << "======================================================" << std::endl;
    }
  }

  // check now that all the files have events
  bool areAllFilesFull = true;
  for (Int_t i = 0; i < nFiles_; i++) {
    if (dxyRefit[i]->GetEntries() == 0.) {
      areAllFilesFull = false;
      break;
    }
  }

  if (!areAllFilesFull) {
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResiduals::FitPVResiduals(): not all the files have events" << std::endl;
    std::cout << "exiting (...to prevent a segmentation fault)" << std::endl;
    exit(EXIT_FAILURE);
  }

  Double_t highedge = nBins_ - 0.5;
  Double_t lowedge = -0.5;

  // DCA absolute

  TH1F *dxyPhiMeanTrend[nFiles_];
  TH1F *dxyPhiWidthTrend[nFiles_];

  TH1F *dxPhiMeanTrend[nFiles_];
  TH1F *dxPhiWidthTrend[nFiles_];

  TH1F *dyPhiMeanTrend[nFiles_];
  TH1F *dyPhiWidthTrend[nFiles_];

  TH1F *dzPhiMeanTrend[nFiles_];
  TH1F *dzPhiWidthTrend[nFiles_];

  TH1F *dxyEtaMeanTrend[nFiles_];
  TH1F *dxyEtaWidthTrend[nFiles_];

  TH1F *dxEtaMeanTrend[nFiles_];
  TH1F *dxEtaWidthTrend[nFiles_];

  TH1F *dyEtaMeanTrend[nFiles_];
  TH1F *dyEtaWidthTrend[nFiles_];

  TH1F *dzEtaMeanTrend[nFiles_];
  TH1F *dzEtaWidthTrend[nFiles_];

  TH1F *dxyPtMeanTrend[nFiles_];
  TH1F *dxyPtWidthTrend[nFiles_];

  TH1F *dzPtMeanTrend[nFiles_];
  TH1F *dzPtWidthTrend[nFiles_];

  // vs ladder and module number

  TH1F *dxyLadderMeanTrend[nFiles_];
  TH1F *dxyLadderWidthTrend[nFiles_];
  TH1F *dzLadderMeanTrend[nFiles_];
  TH1F *dzLadderWidthTrend[nFiles_];

  TH1F *dxyModZMeanTrend[nFiles_];
  TH1F *dxyModZWidthTrend[nFiles_];
  TH1F *dzModZMeanTrend[nFiles_];
  TH1F *dzModZWidthTrend[nFiles_];

  // DCA normalized

  TH1F *dxyNormPhiMeanTrend[nFiles_];
  TH1F *dxyNormPhiWidthTrend[nFiles_];
  TH1F *dzNormPhiMeanTrend[nFiles_];
  TH1F *dzNormPhiWidthTrend[nFiles_];

  TH1F *dxyNormEtaMeanTrend[nFiles_];
  TH1F *dxyNormEtaWidthTrend[nFiles_];
  TH1F *dzNormEtaMeanTrend[nFiles_];
  TH1F *dzNormEtaWidthTrend[nFiles_];

  TH1F *dxyNormPtMeanTrend[nFiles_];
  TH1F *dxyNormPtWidthTrend[nFiles_];
  TH1F *dzNormPtMeanTrend[nFiles_];
  TH1F *dzNormPtWidthTrend[nFiles_];

  TH1F *dxyNormLadderMeanTrend[nFiles_];
  TH1F *dxyNormLadderWidthTrend[nFiles_];
  TH1F *dzNormLadderMeanTrend[nFiles_];
  TH1F *dzNormLadderWidthTrend[nFiles_];

  TH1F *dxyNormModZMeanTrend[nFiles_];
  TH1F *dxyNormModZWidthTrend[nFiles_];
  TH1F *dzNormModZMeanTrend[nFiles_];
  TH1F *dzNormModZWidthTrend[nFiles_];

  // 2D maps

  // bias
  TH2F *dxyMeanMap[nFiles_];
  TH2F *dzMeanMap[nFiles_];
  TH2F *dxyNormMeanMap[nFiles_];
  TH2F *dzNormMeanMap[nFiles_];

  // width
  TH2F *dxyWidthMap[nFiles_];
  TH2F *dzWidthMap[nFiles_];
  TH2F *dxyNormWidthMap[nFiles_];
  TH2F *dzNormWidthMap[nFiles_];

  // trimmed maps

  // bias
  TH2F *t_dxyMeanMap[nFiles_];
  TH2F *t_dzMeanMap[nFiles_];
  TH2F *t_dxyNormMeanMap[nFiles_];
  TH2F *t_dzNormMeanMap[nFiles_];

  // width
  TH2F *t_dxyWidthMap[nFiles_];
  TH2F *t_dzWidthMap[nFiles_];
  TH2F *t_dxyNormWidthMap[nFiles_];
  TH2F *t_dzNormWidthMap[nFiles_];

  // 2D L1 maps

  // bias
  TH2F *dxyMeanL1Map[nFiles_];
  TH2F *dzMeanL1Map[nFiles_];
  TH2F *dxyNormMeanL1Map[nFiles_];
  TH2F *dzNormMeanL1Map[nFiles_];

  // width
  TH2F *dxyWidthL1Map[nFiles_];
  TH2F *dzWidthL1Map[nFiles_];
  TH2F *dxyNormWidthL1Map[nFiles_];
  TH2F *dzNormWidthL1Map[nFiles_];

  // trimmed maps

  // bias
  TH2F *t_dxyMeanL1Map[nFiles_];
  TH2F *t_dzMeanL1Map[nFiles_];
  TH2F *t_dxyNormMeanL1Map[nFiles_];
  TH2F *t_dzNormMeanL1Map[nFiles_];

  // width
  TH2F *t_dxyWidthL1Map[nFiles_];
  TH2F *t_dzWidthL1Map[nFiles_];
  TH2F *t_dxyNormWidthL1Map[nFiles_];
  TH2F *t_dzNormWidthL1Map[nFiles_];

  for (Int_t i = 0; i < nFiles_; i++) {
    // DCA trend plots

    dxyPhiMeanTrend[i] = new TH1F(Form("means_dxy_phi_%i", i),
                                  "#LT d_{xy} #GT vs #phi sector;track #phi [rad];#LT d_{xy} #GT [#mum]",
                                  nBins_,
                                  lowedge,
                                  highedge);
    dxyPhiWidthTrend[i] = new TH1F(Form("widths_dxy_phi_%i", i),
                                   "#sigma(d_{xy}) vs #phi sector;track #phi [rad];#sigma(d_{xy}) [#mum]",
                                   nBins_,
                                   lowedge,
                                   highedge);

    dxPhiMeanTrend[i] = new TH1F(Form("means_dx_phi_%i", i),
                                 "#LT d_{x} #GT vs #phi sector;track #phi [rad];#LT d_{x} #GT [#mum]",
                                 nBins_,
                                 lowedge,
                                 highedge);
    dxPhiWidthTrend[i] = new TH1F(Form("widths_dx_phi_%i", i),
                                  "#sigma(d_{x}) vs #phi sector;track #phi [rad];#sigma(d_{x}) [#mum]",
                                  nBins_,
                                  lowedge,
                                  highedge);

    dyPhiMeanTrend[i] = new TH1F(Form("means_dy_phi_%i", i),
                                 "#LT d_{y} #GT vs #phi sector;track #phi [rad];#LT d_{y} #GT [#mum]",
                                 nBins_,
                                 lowedge,
                                 highedge);
    dyPhiWidthTrend[i] = new TH1F(Form("widths_dy_phi_%i", i),
                                  "#sigma(d_{y}) vs #phi sector;track #phi [rad];#sigma(d_{y}) [#mum]",
                                  nBins_,
                                  lowedge,
                                  highedge);

    dzPhiMeanTrend[i] = new TH1F(Form("means_dz_phi_%i", i),
                                 "#LT d_{z} #GT vs #phi sector;track #phi [rad];#LT d_{z} #GT [#mum]",
                                 nBins_,
                                 lowedge,
                                 highedge);
    dzPhiWidthTrend[i] = new TH1F(Form("widths_dz_phi_%i", i),
                                  "#sigma(d_{z}) vs #phi sector;track #phi [rad];#sigma(d_{z}) [#mum]",
                                  nBins_,
                                  lowedge,
                                  highedge);

    dxyEtaMeanTrend[i] = new TH1F(Form("means_dxy_eta_%i", i),
                                  "#LT d_{xy} #GT vs #eta sector;track #eta;#LT d_{xy} #GT [#mum]",
                                  nBins_,
                                  lowedge,
                                  highedge);
    dxyEtaWidthTrend[i] = new TH1F(Form("widths_dxy_eta_%i", i),
                                   "#sigma(d_{xy}) vs #eta sector;track #eta;#sigma(d_{xy}) [#mum]",
                                   nBins_,
                                   lowedge,
                                   highedge);

    dxEtaMeanTrend[i] = new TH1F(Form("means_dx_eta_%i", i),
                                 "#LT d_{x} #GT vs #eta sector;track #eta;#LT d_{x} #GT [#mum]",
                                 nBins_,
                                 lowedge,
                                 highedge);
    dxEtaWidthTrend[i] = new TH1F(Form("widths_dx_eta_%i", i),
                                  "#sigma(d_{x}) vs #eta sector;track #eta;#sigma(d_{x}) [#mum]",
                                  nBins_,
                                  lowedge,
                                  highedge);

    dyEtaMeanTrend[i] = new TH1F(Form("means_dy_eta_%i", i),
                                 "#LT d_{y} #GT vs #eta sector;track #eta;#LT d_{y} #GT [#mum]",
                                 nBins_,
                                 lowedge,
                                 highedge);
    dyEtaWidthTrend[i] = new TH1F(Form("widths_dy_eta_%i", i),
                                  "#sigma(d_{y}) vs #eta sector;track #eta;#sigma(d_{y}) [#mum]",
                                  nBins_,
                                  lowedge,
                                  highedge);

    dzEtaMeanTrend[i] = new TH1F(Form("means_dz_eta_%i", i),
                                 "#LT d_{z} #GT vs #eta sector;track #eta;#LT d_{z} #GT [#mum]",
                                 nBins_,
                                 lowedge,
                                 highedge);
    dzEtaWidthTrend[i] = new TH1F(Form("widths_dz_eta_%i", i),
                                  "#sigma(d_{xy}) vs #eta sector;track #eta;#sigma(d_{z}) [#mum]",
                                  nBins_,
                                  lowedge,
                                  highedge);

    if (minPt_ > 0.) {
      dxyPtMeanTrend[i] = new TH1F(Form("means_dxy_pT_%i", i),
                                   "#LT d_{xy} #GT vs p_{T} sector;track p_{T} [GeV];#LT d_{xy} #GT [#mum]",
                                   mypT_bins.size() - 1,
                                   mypT_bins.data());
      dxyPtWidthTrend[i] = new TH1F(Form("widths_dxy_pT_%i", i),
                                    "#sigma(d_{xy}) vs p_{T} sector;track p_{T} [GeV];#sigma(d_{xy}) [#mum]",
                                    mypT_bins.size() - 1,
                                    mypT_bins.data());
      dzPtMeanTrend[i] = new TH1F(Form("means_dz_pT_%i", i),
                                  "#LT d_{z} #GT vs p_{T} sector;track p_{T} [GeV];#LT d_{z} #GT [#mum]",
                                  mypT_bins.size() - 1,
                                  mypT_bins.data());
      dzPtWidthTrend[i] = new TH1F(Form("widths_dz_pT_%i", i),
                                   "#sigma(d_{z}) vs p_{T} sector;track p_{T} [GeV];#sigma(d_{z}) [#mum]",
                                   mypT_bins.size() - 1,
                                   mypT_bins.data());
    }

    if (nModZ_ > 0) {
      dxyModZMeanTrend[i] = new TH1F(Form("means_dxy_modZ_%i", i),
                                     "#LT d_{xy} #GT vs Layer 1 module number;module number;#LT d_{xy} #GT [#mum]",
                                     theModZ[i],
                                     0.,
                                     theModZ[i]);
      dxyModZWidthTrend[i] = new TH1F(Form("widths_dxy_modZ_%i", i),
                                      "#sigma(d_{xy}) vs Layer 1 module number;module number;#sigma(d_{xy}) [#mum]",
                                      theModZ[i],
                                      0.,
                                      theModZ[i]);
      dzModZMeanTrend[i] = new TH1F(Form("means_dz_modZ_%i", i),
                                    "#LT d_{z} #GT vs Layer 1 module number;module number;#LT d_{z} #GT [#mum]",
                                    theModZ[i],
                                    0.,
                                    theModZ[i]);
      dzModZWidthTrend[i] = new TH1F(Form("widths_dz_modZ_%i", i),
                                     "#sigma(d_{z}) vs Layer 1 module number;module number;#sigma(d_{z}) [#mum]",
                                     theModZ[i],
                                     0.,
                                     theModZ[i]);
    }

    if (nLadders_ > 0) {
      dxyLadderMeanTrend[i] = new TH1F(Form("means_dxy_ladder_%i", i),
                                       "#LT d_{xy} #GT vs Layer 1 ladder;ladder number;#LT d_{xy} #GT [#mum]",
                                       theLadders[i],
                                       0.,
                                       theLadders[i]);
      dxyLadderWidthTrend[i] = new TH1F(Form("widths_dxy_ladder_%i", i),
                                        "#sigma(d_{xy}) vs Layer 1 ladder;ladder number;#sigma(d_{xy}) [#mum]",
                                        theLadders[i],
                                        0.,
                                        theLadders[i]);
      dzLadderMeanTrend[i] = new TH1F(Form("means_dz_ladder_%i", i),
                                      "#LT d_{z} #GT vs Layer 1 ladder;ladder number;#LT d_{z} #GT [#mum]",
                                      theLadders[i],
                                      0.,
                                      theLadders[i]);
      dzLadderWidthTrend[i] = new TH1F(Form("widths_dz_ladder_%i", i),
                                       "#sigma(d_{z}) vs Layer 1 ladder;ladder number;#sigma(d_{z}) [#mum]",
                                       theLadders[i],
                                       0.,
                                       theLadders[i]);
    }

    // DCA normalized trend plots

    dxyNormPhiMeanTrend[i] =
        new TH1F(Form("means_dxyNorm_phi_%i", i),
                 "#LT d_{xy}/#sigma_{d_{xy}} #GT vs #phi sector;track #phi [rad];#LT d_{xy}/#sigma_{d_{xy}} #GT",
                 nBins_,
                 lowedge,
                 highedge);
    dxyNormPhiWidthTrend[i] =
        new TH1F(Form("widths_dxyNorm_phi_%i", i),
                 "#sigma(d_{xy}/#sigma_{d_{xy}}) vs #phi sector;track #phi [rad];#sigma(d_{xy}/#sigma_{d_{xy}})",
                 nBins_,
                 lowedge,
                 highedge);
    dzNormPhiMeanTrend[i] =
        new TH1F(Form("means_dzNorm_phi_%i", i),
                 "#LT d_{z}/#sigma_{d_{z}} #GT vs #phi sector;track #phi [rad];#LT d_{z}/#sigma_{d_{z}} #GT",
                 nBins_,
                 lowedge,
                 highedge);
    dzNormPhiWidthTrend[i] =
        new TH1F(Form("widths_dzNorm_phi_%i", i),
                 "#sigma(d_{z}/#sigma_{d_{z}}) vs #phi sector;track #phi [rad];#sigma(d_{z}/#sigma_{d_{z}})",
                 nBins_,
                 lowedge,
                 highedge);

    dxyNormEtaMeanTrend[i] =
        new TH1F(Form("means_dxyNorm_eta_%i", i),
                 "#LT d_{xy}/#sigma_{d_{xy}} #GT vs #eta sector;track #eta;#LT d_{xy}/#sigma_{d_{xy}} #GT",
                 nBins_,
                 lowedge,
                 highedge);
    dxyNormEtaWidthTrend[i] =
        new TH1F(Form("widths_dxyNorm_eta_%i", i),
                 "#sigma(d_{xy}/#sigma_{d_{xy}}) vs #eta sector;track #eta;#sigma(d_{xy}/#sigma_{d_{xy}})",
                 nBins_,
                 lowedge,
                 highedge);
    dzNormEtaMeanTrend[i] =
        new TH1F(Form("means_dzNorm_eta_%i", i),
                 "#LT d_{z}/#sigma_{d_{z}} #GT vs #eta sector;track #eta;#LT d_{z}/#sigma_{d_{z}} #GT",
                 nBins_,
                 lowedge,
                 highedge);
    dzNormEtaWidthTrend[i] =
        new TH1F(Form("widths_dzNorm_eta_%i", i),
                 "#sigma(d_{z}/#sigma_{d_{z}}) vs #eta sector;track #eta;#sigma(d_{z}/#sigma_{d_{z}})",
                 nBins_,
                 lowedge,
                 highedge);

    if (minPt_ > 0.) {
      dxyNormPtMeanTrend[i] =
          new TH1F(Form("means_dxyNorm_pT_%i", i),
                   "#LT d_{xy}/#sigma_{d_{xy}} #GT vs p_{T} sector;track p_{T} [GeV];#LT d_{xy}/#sigma_{d_{xy}} #GT",
                   mypT_bins.size() - 1,
                   mypT_bins.data());
      dxyNormPtWidthTrend[i] =
          new TH1F(Form("widths_dxyNorm_pT_%i", i),
                   "#sigma(d_{xy}/#sigma_{d_{xy}}) vs p_{T} sector;track p_{T} [GeV];#sigma(d_{xy}/#sigma_{d_{xy}})",
                   mypT_bins.size() - 1,
                   mypT_bins.data());
      dzNormPtMeanTrend[i] =
          new TH1F(Form("means_dzNorm_pT_%i", i),
                   "#LT d_{z}/#sigma_{d_{z}} #GT vs p_{T} sector;track p_{T} [GeV];#LT d_{z}/#sigma_{d_{z}} #GT",
                   mypT_bins.size() - 1,
                   mypT_bins.data());
      dzNormPtWidthTrend[i] =
          new TH1F(Form("widths_dzNorm_pT_%i", i),
                   "#sigma(d_{z}/#sigma_{d_{z}}) vs p_{T} sector;track p_{T} [GeV];#sigma(d_{z}/#sigma_{d_{z}})",
                   mypT_bins.size() - 1,
                   mypT_bins.data());
    }

    if (nLadders_ > 0) {
      dxyNormLadderMeanTrend[i] =
          new TH1F(Form("means_dxyNorm_ladder_%i", i),
                   "#LT d_{xy}/#sigma_{d_{xy}} #GT vs Layer 1 ladder;ladder number;#LT d_{xy}/#sigma_{d_{xy}} #GT",
                   theLadders[i],
                   0.,
                   theLadders[i]);
      dxyNormLadderWidthTrend[i] =
          new TH1F(Form("widths_dxyNorm_ladder_%i", i),
                   "#sigma(d_{xy}/#sigma_{d_{xy}}) vs Layer 1 ladder;ladder number;#sigma(d_{xy}/#sigma_{d_{xy}})",
                   theLadders[i],
                   0.,
                   theLadders[i]);
      dzNormLadderMeanTrend[i] =
          new TH1F(Form("means_dzNorm_ladder_%i", i),
                   "#LT d_{z}/#sigma_{d_{z}} #GT vs Layer 1 ladder;ladder number;#LT d_{z}/#sigma_{d_{z}} #GT",
                   theLadders[i],
                   0.,
                   theLadders[i]);
      dzNormLadderWidthTrend[i] =
          new TH1F(Form("widths_dzNorm_ladder_%i", i),
                   "#sigma(d_{z}/#sigma_{d_{z}}) vs Layer 1 ladder;ladder number;#sigma(d_{z}/#sigma_{d_{z}})",
                   theLadders[i],
                   0.,
                   theLadders[i]);
    }

    if (nModZ_ > 0) {
      dxyNormModZMeanTrend[i] = new TH1F(
          Form("means_dxyNorm_modZ_%i", i),
          "#LT d_{xy}/#sigma_{d_{xy}} #GT vs Layer 1 module number;module number;#LT d_{xy}/#sigma_{d_{xy}} #GT",
          theModZ[i],
          0.,
          theModZ[i]);
      dxyNormModZWidthTrend[i] = new TH1F(
          Form("widths_dxyNorm_modZ_%i", i),
          "#sigma(d_{xy}/#sigma_{d_{xy}}) vs Layer 1 module number;module number;#sigma(d_{xy}/#sigma_{d_{xy}})",
          theModZ[i],
          0.,
          theModZ[i]);
      dzNormModZMeanTrend[i] =
          new TH1F(Form("means_dzNorm_modZ_%i", i),
                   "#LT d_{z}/#sigma_{d_{z}} #GT vs Layer 1 module number;module number;#LT d_{z}/#sigma_{d_{z}} #GT",
                   theModZ[i],
                   0.,
                   theModZ[i]);
      dzNormModZWidthTrend[i] =
          new TH1F(Form("widths_dzNorm_modZ_%i", i),
                   "#sigma(d_{z}/#sigma_{d_{z}}) vs Layer 1 module number;module number;#sigma(d_{z}/#sigma_{d_{z}})",
                   theModZ[i],
                   0.,
                   theModZ[i]);
    }

    // 2D maps
    dxyMeanMap[i] = new TH2F(Form("means_dxy_map_%i", i),
                             "#LT d_{xy} #GT map;track #eta;track #phi [rad];#LT d_{xy} #GT [#mum]",
                             nBins_,
                             lowedge,
                             highedge,
                             nBins_,
                             lowedge,
                             highedge);
    dzMeanMap[i] = new TH2F(Form("means_dz_map_%i", i),
                            "#LT d_{z} #GT map;track #eta;track #phi [rad];#LT d_{z} #GT [#mum]",
                            nBins_,
                            lowedge,
                            highedge,
                            nBins_,
                            lowedge,
                            highedge);
    dxyNormMeanMap[i] =
        new TH2F(Form("norm_means_dxy_map_%i", i),
                 "#LT d_{xy}/#sigma_{d_{xy}} #GT map;track #eta;track #phi [rad];#LT d_{xy}/#sigma_{d_{xy}} #GT",
                 nBins_,
                 lowedge,
                 highedge,
                 nBins_,
                 lowedge,
                 highedge);
    dzNormMeanMap[i] =
        new TH2F(Form("norm_means_dz_map_%i", i),
                 "#LT d_{z}/#sigma_{d_{z}} #GT map;track #eta;track #phi[rad];#LT d_{xy}/#sigma_{d_{z}} #GT",
                 nBins_,
                 lowedge,
                 highedge,
                 nBins_,
                 lowedge,
                 highedge);

    dxyWidthMap[i] = new TH2F(Form("widths_dxy_map_%i", i),
                              "#sigma_{d_{xy}} map;track #eta;track #phi [rad];#sigma(d_{xy}) [#mum]",
                              nBins_,
                              lowedge,
                              highedge,
                              nBins_,
                              lowedge,
                              highedge);
    dzWidthMap[i] = new TH2F(Form("widths_dz_map_%i", i),
                             "#sigma_{d_{z}} map;track #eta;track #phi [rad];#sigma(d_{z}) [#mum]",
                             nBins_,
                             lowedge,
                             highedge,
                             nBins_,
                             lowedge,
                             highedge);
    dxyNormWidthMap[i] =
        new TH2F(Form("norm_widths_dxy_map_%i", i),
                 "width(d_{xy}/#sigma_{d_{xy}}) map;track #eta;track #phi[rad];#sigma(d_{xy}/#sigma_{d_{xy}})",
                 nBins_,
                 lowedge,
                 highedge,
                 nBins_,
                 lowedge,
                 highedge);
    dzNormWidthMap[i] =
        new TH2F(Form("norm_widths_dz_map_%i", i),
                 "width(d_{z}/#sigma_{d_{z}}) map;track #eta;track #phi [rad];#sigma(d_{z}/#sigma_{d_{z}})",
                 nBins_,
                 lowedge,
                 highedge,
                 nBins_,
                 lowedge,
                 highedge);

    // 2D maps L1
    dxyMeanL1Map[i] = new TH2F(Form("means_dxy_L1Map_%i", i),
                               "#LT d_{xy} #GT map;module number;ladder number;#LT d_{xy} #GT [#mum]",
                               nModZ_,
                               -0.5,
                               nModZ_ - 0.5,
                               nLadders_,
                               -0.5,
                               nLadders_ - 0.5);
    dzMeanL1Map[i] = new TH2F(Form("means_dz_L1Map_%i", i),
                              "#LT d_{z} #GT map;module number;ladder number;#LT d_{z} #GT [#mum]",
                              nModZ_,
                              -0.5,
                              nModZ_ - 0.5,
                              nLadders_,
                              -0.5,
                              nLadders_ - 0.5);
    dxyNormMeanL1Map[i] =
        new TH2F(Form("norm_means_dxy_L1Map_%i", i),
                 "#LT d_{xy}/#sigma_{d_{xy}} #GT map;module number;ladder number;#LT d_{xy}/#sigma_{d_{xy}} #GT",
                 nModZ_,
                 -0.5,
                 nModZ_ - 0.5,
                 nLadders_,
                 -0.5,
                 nLadders_ - 0.5);
    dzNormMeanL1Map[i] =
        new TH2F(Form("norm_means_dz_L1Map_%i", i),
                 "#LT d_{z}/#sigma_{d_{z}} #GT map;module number;ladder number;#LT d_{xy}/#sigma_{d_{z}} #GT",
                 nModZ_,
                 -0.5,
                 nModZ_ - 0.5,
                 nLadders_,
                 -0.5,
                 nLadders_ - 0.5);

    dxyWidthL1Map[i] = new TH2F(Form("widths_dxy_L1Map_%i", i),
                                "#sigma_{d_{xy}} map;module number;ladder number;#sigma(d_{xy}) [#mum]",
                                nModZ_,
                                -0.5,
                                nModZ_ - 0.5,
                                nLadders_,
                                -0.5,
                                nLadders_ - 0.5);
    dzWidthL1Map[i] = new TH2F(Form("widths_dz_L1Map_%i", i),
                               "#sigma_{d_{z}} map;module number;ladder number;#sigma(d_{z}) [#mum]",
                               nModZ_,
                               -0.5,
                               nModZ_ - 0.5,
                               nLadders_,
                               -0.5,
                               nLadders_ - 0.5);
    dxyNormWidthL1Map[i] =
        new TH2F(Form("norm_widths_dxy_L1Map_%i", i),
                 "width(d_{xy}/#sigma_{d_{xy}}) map;module number;ladder number;#sigma(d_{xy}/#sigma_{d_{xy}})",
                 nModZ_,
                 -0.5,
                 nModZ_ - 0.5,
                 nLadders_,
                 -0.5,
                 nLadders_ - 0.5);
    dzNormWidthL1Map[i] =
        new TH2F(Form("norm_widths_dz_L1Map_%i", i),
                 "width(d_{z}/#sigma_{d_{z}}) map;module number;ladder number;#sigma(d_{z}/#sigma_{d_{z}})",
                 nModZ_,
                 -0.5,
                 nModZ_ - 0.5,
                 nLadders_,
                 -0.5,
                 nLadders_ - 0.5);

    // DCA absolute

    if (isDebugMode) {
      timer.Stop();
      std::cout << "check point 3-" << i << " " << timer.CpuTime() << " " << timer.RealTime() << std::endl;
      timer.Continue();
    }

    FillTrendPlot(dxyPhiMeanTrend[i], dxyPhiResiduals[i], params::MEAN, "phi", nBins_);
    FillTrendPlot(dxyPhiWidthTrend[i], dxyPhiResiduals[i], params::WIDTH, "phi", nBins_);

    FillTrendPlot(dxPhiMeanTrend[i], dxPhiResiduals[i], params::MEAN, "phi", nBins_);
    FillTrendPlot(dxPhiWidthTrend[i], dxPhiResiduals[i], params::WIDTH, "phi", nBins_);

    FillTrendPlot(dyPhiMeanTrend[i], dyPhiResiduals[i], params::MEAN, "phi", nBins_);
    FillTrendPlot(dyPhiWidthTrend[i], dyPhiResiduals[i], params::WIDTH, "phi", nBins_);

    FillTrendPlot(dzPhiMeanTrend[i], dzPhiResiduals[i], params::MEAN, "phi", nBins_);
    FillTrendPlot(dzPhiWidthTrend[i], dzPhiResiduals[i], params::WIDTH, "phi", nBins_);

    FillTrendPlot(dxyEtaMeanTrend[i], dxyEtaResiduals[i], params::MEAN, "eta", nBins_);
    FillTrendPlot(dxyEtaWidthTrend[i], dxyEtaResiduals[i], params::WIDTH, "eta", nBins_);

    FillTrendPlot(dxEtaMeanTrend[i], dxEtaResiduals[i], params::MEAN, "eta", nBins_);
    FillTrendPlot(dxEtaWidthTrend[i], dxEtaResiduals[i], params::WIDTH, "eta", nBins_);

    FillTrendPlot(dyEtaMeanTrend[i], dyEtaResiduals[i], params::MEAN, "eta", nBins_);
    FillTrendPlot(dyEtaWidthTrend[i], dyEtaResiduals[i], params::WIDTH, "eta", nBins_);

    FillTrendPlot(dzEtaMeanTrend[i], dzEtaResiduals[i], params::MEAN, "eta", nBins_);
    FillTrendPlot(dzEtaWidthTrend[i], dzEtaResiduals[i], params::WIDTH, "eta", nBins_);

    if (minPt_ > 0.) {
      FillTrendPlot(dxyPtMeanTrend[i], dxyPtResiduals[i], params::MEAN, "pT", nPtBins_);
      FillTrendPlot(dxyPtWidthTrend[i], dxyPtResiduals[i], params::WIDTH, "pT", nPtBins_);
      FillTrendPlot(dzPtMeanTrend[i], dzPtResiduals[i], params::MEAN, "pT", nPtBins_);
      FillTrendPlot(dzPtWidthTrend[i], dzPtResiduals[i], params::WIDTH, "pT", nPtBins_);
    }

    if (nLadders_ > 0) {
      FillTrendPlot(dxyLadderMeanTrend[i], dxyLadderResiduals[i], params::MEAN, "else", nLadders_);
      FillTrendPlot(dxyLadderWidthTrend[i], dxyLadderResiduals[i], params::WIDTH, "else", nLadders_);
      FillTrendPlot(dzLadderMeanTrend[i], dzLadderResiduals[i], params::MEAN, "else", nLadders_);
      FillTrendPlot(dzLadderWidthTrend[i], dzLadderResiduals[i], params::WIDTH, "else", nLadders_);
    }

    if (nModZ_ > 0) {
      FillTrendPlot(dxyModZMeanTrend[i], dxyModZResiduals[i], params::MEAN, "else", nModZ_);
      FillTrendPlot(dxyModZWidthTrend[i], dxyModZResiduals[i], params::WIDTH, "else", nModZ_);
      FillTrendPlot(dzModZMeanTrend[i], dzModZResiduals[i], params::MEAN, "else", nModZ_);
      FillTrendPlot(dzModZWidthTrend[i], dzModZResiduals[i], params::WIDTH, "else", nModZ_);
    }

    MakeNiceTrendPlotStyle(dxyPhiMeanTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dxyPhiWidthTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dxPhiMeanTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dxPhiWidthTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dyPhiMeanTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dyPhiWidthTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dzPhiMeanTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dzPhiWidthTrend[i], colors[i], markers[i]);

    MakeNiceTrendPlotStyle(dxyEtaMeanTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dxyEtaWidthTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dxEtaMeanTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dxEtaWidthTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dyEtaMeanTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dyEtaWidthTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dzEtaMeanTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dzEtaWidthTrend[i], colors[i], markers[i]);

    if (minPt_ > 0.) {
      MakeNiceTrendPlotStyle(dxyPtMeanTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dxyPtWidthTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dzPtMeanTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dzPtWidthTrend[i], colors[i], markers[i]);
    }

    if (nLadders_ > 0) {
      MakeNiceTrendPlotStyle(dxyLadderMeanTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dxyLadderWidthTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dzLadderMeanTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dzLadderWidthTrend[i], colors[i], markers[i]);
    }

    if (nModZ_ > 0) {
      MakeNiceTrendPlotStyle(dxyModZMeanTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dxyModZWidthTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dzModZMeanTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dzModZWidthTrend[i], colors[i], markers[i]);
    }

    // DCA normalized

    FillTrendPlot(dxyNormPhiMeanTrend[i], dxyNormPhiResiduals[i], params::MEAN, "phi", nBins_);
    FillTrendPlot(dxyNormPhiWidthTrend[i], dxyNormPhiResiduals[i], params::WIDTH, "phi", nBins_);
    FillTrendPlot(dzNormPhiMeanTrend[i], dzNormPhiResiduals[i], params::MEAN, "phi", nBins_);
    FillTrendPlot(dzNormPhiWidthTrend[i], dzNormPhiResiduals[i], params::WIDTH, "phi", nBins_);

    FillTrendPlot(dxyNormEtaMeanTrend[i], dxyNormEtaResiduals[i], params::MEAN, "eta", nBins_);
    FillTrendPlot(dxyNormEtaWidthTrend[i], dxyNormEtaResiduals[i], params::WIDTH, "eta", nBins_);
    FillTrendPlot(dzNormEtaMeanTrend[i], dzNormEtaResiduals[i], params::MEAN, "eta", nBins_);
    FillTrendPlot(dzNormEtaWidthTrend[i], dzNormEtaResiduals[i], params::WIDTH, "eta", nBins_);

    if (minPt_ > 0.) {
      FillTrendPlot(dxyNormPtMeanTrend[i], dxyNormPtResiduals[i], params::MEAN, "pT", nPtBins_);
      FillTrendPlot(dxyNormPtWidthTrend[i], dxyNormPtResiduals[i], params::WIDTH, "pT", nPtBins_);
      FillTrendPlot(dzNormPtMeanTrend[i], dzNormPtResiduals[i], params::MEAN, "pT", nPtBins_);
      FillTrendPlot(dzNormPtWidthTrend[i], dzNormPtResiduals[i], params::WIDTH, "pT", nPtBins_);
    }

    if (nLadders_ > 0) {
      FillTrendPlot(dxyNormLadderMeanTrend[i], dxyNormLadderResiduals[i], params::MEAN, "else", nLadders_);
      FillTrendPlot(dxyNormLadderWidthTrend[i], dxyNormLadderResiduals[i], params::WIDTH, "else", nLadders_);
      FillTrendPlot(dzNormLadderMeanTrend[i], dzNormLadderResiduals[i], params::MEAN, "else", nLadders_);
      FillTrendPlot(dzNormLadderWidthTrend[i], dzNormLadderResiduals[i], params::WIDTH, "else", nLadders_);
    }

    if (nModZ_ > 0) {
      FillTrendPlot(dxyNormModZMeanTrend[i], dxyNormModZResiduals[i], params::MEAN, "else", nModZ_);
      FillTrendPlot(dxyNormModZWidthTrend[i], dxyNormModZResiduals[i], params::WIDTH, "else", nModZ_);
      FillTrendPlot(dzNormModZMeanTrend[i], dzNormModZResiduals[i], params::MEAN, "else", nModZ_);
      FillTrendPlot(dzNormModZWidthTrend[i], dzNormModZResiduals[i], params::WIDTH, "else", nModZ_);
    }

    MakeNiceTrendPlotStyle(dxyNormPhiMeanTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dxyNormPhiWidthTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dzNormPhiMeanTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dzNormPhiWidthTrend[i], colors[i], markers[i]);

    MakeNiceTrendPlotStyle(dxyNormEtaMeanTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dxyNormEtaWidthTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dzNormEtaMeanTrend[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dzNormEtaWidthTrend[i], colors[i], markers[i]);

    if (minPt_ > 0.) {
      MakeNiceTrendPlotStyle(dxyNormPtMeanTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dxyNormPtWidthTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dzNormPtMeanTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dzNormPtWidthTrend[i], colors[i], markers[i]);
    }

    if (nLadders_ > 0) {
      MakeNiceTrendPlotStyle(dxyNormLadderMeanTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dxyNormLadderWidthTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dzNormLadderMeanTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dzNormLadderWidthTrend[i], colors[i], markers[i]);
    }

    if (nModZ_ > 0) {
      MakeNiceTrendPlotStyle(dxyNormModZMeanTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dxyNormModZWidthTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dzNormModZMeanTrend[i], colors[i], markers[i]);
      MakeNiceTrendPlotStyle(dzNormModZWidthTrend[i], colors[i], markers[i]);
    }

    // maps

    //TTimeStamp filling1D_done;

    if (do2DMaps) {
      if (isDebugMode) {
        timer.Stop();
        std::cout << "check point 4-" << i << " " << timer.CpuTime() << " " << timer.RealTime() << std::endl;
        timer.Continue();
      }

      std::vector<std::vector<TH1F *> > v_dxyAbsMap;  //(nBins_, std::vector<TH1F*>(nBins_));
      std::vector<std::vector<TH1F *> > v_dzAbsMap;
      std::vector<std::vector<TH1F *> > v_dxyNormMap;
      std::vector<std::vector<TH1F *> > v_dzNormMap;

      for (Int_t index1 = 0; index1 < nBins_; index1++) {
        std::vector<TH1F *> a_temp_vec_xy;
        std::vector<TH1F *> n_temp_vec_xy;
        std::vector<TH1F *> a_temp_vec_z;
        std::vector<TH1F *> n_temp_vec_z;

        for (Int_t index2 = 0; index2 < nBins_; index2++) {
          if (isDebugMode)
            std::cout << index1 << " " << index2 << " " << (dxyMapResiduals[i][index1][index2])->GetName() << " "
                      << (dxyMapResiduals[i][index1][index2])->GetEntries() << std::endl;

          a_temp_vec_xy.push_back(dxyMapResiduals[i][index1][index2]);
          n_temp_vec_xy.push_back(dxyNormMapResiduals[i][index1][index2]);
          a_temp_vec_z.push_back(dzMapResiduals[i][index1][index2]);
          n_temp_vec_z.push_back(dzNormMapResiduals[i][index1][index2]);
        }

        v_dxyAbsMap.push_back(a_temp_vec_xy);
        v_dzAbsMap.push_back(a_temp_vec_z);
        v_dxyNormMap.push_back(n_temp_vec_xy);
        v_dzNormMap.push_back(n_temp_vec_z);
      }

      FillMap(dxyMeanMap[i], v_dxyAbsMap, params::MEAN);
      FillMap(dxyWidthMap[i], v_dxyAbsMap, params::WIDTH);
      FillMap(dzMeanMap[i], v_dzAbsMap, params::MEAN);
      FillMap(dzWidthMap[i], v_dzAbsMap, params::WIDTH);

      FillMap(dxyNormMeanMap[i], v_dxyNormMap, params::MEAN);
      FillMap(dxyNormWidthMap[i], v_dxyNormMap, params::WIDTH);
      FillMap(dzNormMeanMap[i], v_dzNormMap, params::MEAN);
      FillMap(dzNormWidthMap[i], v_dzNormMap, params::WIDTH);

      if (isDebugMode) {
        timer.Stop();
        std::cout << "check point 5-" << i << " " << timer.CpuTime() << " " << timer.RealTime() << std::endl;
        timer.Continue();
      }

      t_dxyMeanMap[i] = trimTheMap(dxyMeanMap[i]).first;
      t_dxyWidthMap[i] = trimTheMap(dxyWidthMap[i]).first;
      t_dzMeanMap[i] = trimTheMap(dzMeanMap[i]).first;
      t_dzWidthMap[i] = trimTheMap(dzWidthMap[i]).first;

      t_dxyNormMeanMap[i] = trimTheMap(dxyNormMeanMap[i]).first;
      t_dxyNormWidthMap[i] = trimTheMap(dxyNormWidthMap[i]).first;
      t_dzNormMeanMap[i] = trimTheMap(dzNormMeanMap[i]).first;
      t_dzNormWidthMap[i] = trimTheMap(dzNormWidthMap[i]).first;

      MakeNiceMapStyle(t_dxyMeanMap[i]);
      MakeNiceMapStyle(t_dxyWidthMap[i]);
      MakeNiceMapStyle(t_dzMeanMap[i]);
      MakeNiceMapStyle(t_dzWidthMap[i]);

      MakeNiceMapStyle(t_dxyNormMeanMap[i]);
      MakeNiceMapStyle(t_dxyNormWidthMap[i]);
      MakeNiceMapStyle(t_dzNormMeanMap[i]);
      MakeNiceMapStyle(t_dzNormWidthMap[i]);

      // clear the vectors of vectors
      for (Int_t index1 = 0; index1 < nBins_; index1++) {
        v_dxyAbsMap.clear();
        v_dzAbsMap.clear();
        v_dxyNormMap.clear();
        v_dzNormMap.clear();
      }

      for (Int_t index1 = 0; index1 < nLadders_; index1++) {
        std::vector<TH1F *> a_temp_vec_xy;
        std::vector<TH1F *> n_temp_vec_xy;
        std::vector<TH1F *> a_temp_vec_z;
        std::vector<TH1F *> n_temp_vec_z;

        for (Int_t index2 = 0; index2 < nModZ_; index2++) {
          a_temp_vec_xy.push_back(dxyL1MapResiduals[i][index1][index2]);
          a_temp_vec_z.push_back(dzL1MapResiduals[i][index1][index2]);
          n_temp_vec_xy.push_back(dxyL1NormMapResiduals[i][index1][index2]);
          n_temp_vec_z.push_back(dzL1NormMapResiduals[i][index1][index2]);
        }

        v_dxyAbsMap.push_back(a_temp_vec_xy);
        v_dzAbsMap.push_back(a_temp_vec_z);
        v_dxyNormMap.push_back(n_temp_vec_xy);
        v_dzNormMap.push_back(n_temp_vec_z);
      }

      FillMap(dxyMeanL1Map[i], v_dxyAbsMap, params::MEAN, nModZ_, nLadders_);
      FillMap(dxyWidthL1Map[i], v_dxyAbsMap, params::WIDTH, nModZ_, nLadders_);
      FillMap(dzMeanL1Map[i], v_dzAbsMap, params::MEAN, nModZ_, nLadders_);
      FillMap(dzWidthL1Map[i], v_dzAbsMap, params::WIDTH, nModZ_, nLadders_);

      FillMap(dxyNormMeanL1Map[i], v_dxyNormMap, params::MEAN, nModZ_, nLadders_);
      FillMap(dxyNormWidthL1Map[i], v_dxyNormMap, params::WIDTH, nModZ_, nLadders_);
      FillMap(dzNormMeanL1Map[i], v_dzNormMap, params::MEAN, nModZ_, nLadders_);
      FillMap(dzNormWidthL1Map[i], v_dzNormMap, params::WIDTH, nModZ_, nLadders_);

      if (isDebugMode) {
        timer.Stop();
        std::cout << "check point 5-" << i << " " << timer.CpuTime() << " " << timer.RealTime() << std::endl;
        timer.Continue();
      }

      t_dxyMeanL1Map[i] = trimTheMap(dxyMeanL1Map[i]).first;
      t_dxyWidthL1Map[i] = trimTheMap(dxyWidthL1Map[i]).first;
      t_dzMeanL1Map[i] = trimTheMap(dzMeanL1Map[i]).first;
      t_dzWidthL1Map[i] = trimTheMap(dzWidthL1Map[i]).first;

      t_dxyNormMeanL1Map[i] = trimTheMap(dxyNormMeanL1Map[i]).first;
      t_dxyNormWidthL1Map[i] = trimTheMap(dxyNormWidthL1Map[i]).first;
      t_dzNormMeanL1Map[i] = trimTheMap(dzNormMeanL1Map[i]).first;
      t_dzNormWidthL1Map[i] = trimTheMap(dzNormWidthL1Map[i]).first;

      MakeNiceMapStyle(t_dxyMeanL1Map[i]);
      MakeNiceMapStyle(t_dxyWidthL1Map[i]);
      MakeNiceMapStyle(t_dzMeanL1Map[i]);
      MakeNiceMapStyle(t_dzWidthL1Map[i]);

      MakeNiceMapStyle(t_dxyNormMeanL1Map[i]);
      MakeNiceMapStyle(t_dxyNormWidthL1Map[i]);
      MakeNiceMapStyle(t_dzNormMeanL1Map[i]);
      MakeNiceMapStyle(t_dzNormWidthL1Map[i]);

    }  // if do2DMaps

    MakeNiceTrendPlotStyle(dxyRefit[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dzRefit[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dxySigRefit[i], colors[i], markers[i]);
    MakeNiceTrendPlotStyle(dzSigRefit[i], colors[i], markers[i]);
  }

  TTimeStamp filling2D_done;

  TString theStrDate = theDate;
  TString theStrAlignment = LegLabels[0];

  /*
    // in case labels are needed
    std::vector<TString> vLabels(LegLabels, LegLabels+10);
    vLabels.shrink_to_fit();
  */

  for (Int_t j = 1; j < nFiles_; j++) {
    theStrAlignment += ("_vs_" + LegLabels[j]);
  }

  theStrDate.ReplaceAll(" ", "");
  theStrAlignment.ReplaceAll(" ", "_");

  // non-differential
  TCanvas *BareResiduals = new TCanvas("BareResiduals", "BareResiduals", 1200, 1200);
  arrangeBiasCanvas(BareResiduals, dxyRefit, dxySigRefit, dzRefit, dzSigRefit, nFiles_, LegLabels, theDate, true);

  BareResiduals->SaveAs("ResidualsCanvas_" + theStrDate + theStrAlignment + ".pdf");
  BareResiduals->SaveAs("ResidualsCanvas_" + theStrDate + theStrAlignment + ".png");

  // DCA absolute

  TCanvas *dxyPhiTrend = new TCanvas("dxyPhiTrend", "dxyPhiTrend", 1200, 600);
  arrangeCanvas(dxyPhiTrend, dxyPhiMeanTrend, dxyPhiWidthTrend, nFiles_, LegLabels, theDate, false, setAutoLimits);

  dxyPhiTrend->SaveAs("dxyPhiTrend_" + theStrDate + theStrAlignment + ".pdf");
  dxyPhiTrend->SaveAs("dxyPhiTrend_" + theStrDate + theStrAlignment + ".png");

  TCanvas *dzPhiTrend = new TCanvas("dzPhiTrend", "dzPhiTrend", 1200, 600);
  arrangeCanvas(dzPhiTrend, dzPhiMeanTrend, dzPhiWidthTrend, nFiles_, LegLabels, theDate, false, setAutoLimits);

  dzPhiTrend->SaveAs("dzPhiTrend_" + theStrDate + theStrAlignment + ".pdf");
  dzPhiTrend->SaveAs("dzPhiTrend_" + theStrDate + theStrAlignment + ".png");

  TCanvas *dxyEtaTrend = new TCanvas("dxyEtaTrend", "dxyEtaTrend", 1200, 600);
  arrangeCanvas(dxyEtaTrend, dxyEtaMeanTrend, dxyEtaWidthTrend, nFiles_, LegLabels, theDate, false, setAutoLimits);

  dxyEtaTrend->SaveAs("dxyEtaTrend_" + theStrDate + theStrAlignment + ".pdf");
  dxyEtaTrend->SaveAs("dxyEtaTrend_" + theStrDate + theStrAlignment + ".png");

  TCanvas *dzEtaTrend = new TCanvas("dzEtaTrend", "dzEtaTrend", 1200, 600);
  arrangeCanvas(dzEtaTrend, dzEtaMeanTrend, dzEtaWidthTrend, nFiles_, LegLabels, theDate, false, setAutoLimits);

  dzEtaTrend->SaveAs("dzEtaTrend_" + theStrDate + theStrAlignment + ".pdf");
  dzEtaTrend->SaveAs("dzEtaTrend_" + theStrDate + theStrAlignment + ".png");

  if (nLadders_ > 0) {
    TCanvas *dxyLadderTrend = new TCanvas("dxyLadderTrend", "dxyLadderTrend", 600, 600);
    arrangeCanvas(
        dxyLadderTrend, dxyLadderMeanTrend, dxyLadderWidthTrend, nFiles_, LegLabels, theDate, true, setAutoLimits);

    dxyLadderTrend->SaveAs("dxyLadderTrend_" + theStrDate + theStrAlignment + ".pdf");
    dxyLadderTrend->SaveAs("dxyLadderTrend_" + theStrDate + theStrAlignment + ".png");

    delete dxyLadderTrend;
  }

  // fit dz vs phi
  TCanvas *dzPhiTrendFit = new TCanvas("dzPhiTrendFit", "dzPhiTrendFit", 1200, 600);
  arrangeFitCanvas(dzPhiTrendFit, dzPhiMeanTrend, nFiles_, LegLabels, theDate);

  dzPhiTrendFit->SaveAs("dzPhiTrendFit_" + theStrDate + theStrAlignment + ".pdf");
  dzPhiTrendFit->SaveAs("dzPhiTrendFit_" + theStrDate + theStrAlignment + ".png");

  if (minPt_ > 0.) {
    TCanvas *dxyPtTrend = new TCanvas("dxyPtTrend", "dxyPtTrend", 1200, 600);
    arrangeCanvas(dxyPtTrend, dxyPtMeanTrend, dxyPtWidthTrend, nFiles_, LegLabels, theDate, false, setAutoLimits);

    dxyPtTrend->SaveAs("dxyPtTrend_" + theStrDate + theStrAlignment + ".pdf");
    dxyPtTrend->SaveAs("dxyPtTrend_" + theStrDate + theStrAlignment + ".png");

    TCanvas *dzPtTrend = new TCanvas("dzPtTrend", "dzPtTrend", 1200, 600);
    arrangeCanvas(dzPtTrend, dzPtMeanTrend, dzPtWidthTrend, nFiles_, LegLabels, theDate, false, setAutoLimits);

    dzPtTrend->SaveAs("dzPtTrend_" + theStrDate + theStrAlignment + ".pdf");
    dzPtTrend->SaveAs("dzPtTrend_" + theStrDate + theStrAlignment + ".png");

    delete dxyPtTrend;
    delete dzPtTrend;
  }

  // delete all news

  delete BareResiduals;
  delete dxyPhiTrend;
  delete dzPhiTrend;
  delete dxyEtaTrend;
  delete dzEtaTrend;
  delete dzPhiTrendFit;

  // DCA normalized

  TCanvas *dxyNormPhiTrend = new TCanvas("dxyNormPhiTrend", "dxyNormPhiTrend", 1200, 600);
  arrangeCanvas(
      dxyNormPhiTrend, dxyNormPhiMeanTrend, dxyNormPhiWidthTrend, nFiles_, LegLabels, theDate, false, setAutoLimits);

  dxyNormPhiTrend->SaveAs("dxyPhiTrendNorm_" + theStrDate + theStrAlignment + ".pdf");
  dxyNormPhiTrend->SaveAs("dxyPhiTrendNorm_" + theStrDate + theStrAlignment + ".png");

  TCanvas *dzNormPhiTrend = new TCanvas("dzNormPhiTrend", "dzNormPhiTrend", 1200, 600);
  arrangeCanvas(
      dzNormPhiTrend, dzNormPhiMeanTrend, dzNormPhiWidthTrend, nFiles_, LegLabels, theDate, false, setAutoLimits);

  dzNormPhiTrend->SaveAs("dzPhiTrendNorm_" + theStrDate + theStrAlignment + ".pdf");
  dzNormPhiTrend->SaveAs("dzPhiTrendNorm_" + theStrDate + theStrAlignment + ".png");

  TCanvas *dxyNormEtaTrend = new TCanvas("dxyNormEtaTrend", "dxyNormEtaTrend", 1200, 600);
  arrangeCanvas(
      dxyNormEtaTrend, dxyNormEtaMeanTrend, dxyNormEtaWidthTrend, nFiles_, LegLabels, theDate, false, setAutoLimits);

  dxyNormEtaTrend->SaveAs("dxyEtaTrendNorm_" + theStrDate + theStrAlignment + ".pdf");
  dxyNormEtaTrend->SaveAs("dxyEtaTrendNorm_" + theStrDate + theStrAlignment + ".png");

  TCanvas *dzNormEtaTrend = new TCanvas("dzNormEtaTrend", "dzNormEtaTrend", 1200, 600);
  arrangeCanvas(
      dzNormEtaTrend, dzNormEtaMeanTrend, dzNormEtaWidthTrend, nFiles_, LegLabels, theDate, false, setAutoLimits);

  dzNormEtaTrend->SaveAs("dzEtaTrendNorm_" + theStrDate + theStrAlignment + ".pdf");
  dzNormEtaTrend->SaveAs("dzEtaTrendNorm_" + theStrDate + theStrAlignment + ".png");

  if (minPt_ > 0.) {
    TCanvas *dxyNormPtTrend = new TCanvas("dxyNormPtTrend", "dxyNormPtTrend", 1200, 600);
    arrangeCanvas(
        dxyNormPtTrend, dxyNormPtMeanTrend, dxyNormPtWidthTrend, nFiles_, LegLabels, theDate, false, setAutoLimits);

    dxyNormPtTrend->SaveAs("dxyPtTrendNorm_" + theStrDate + theStrAlignment + ".pdf");
    dxyNormPtTrend->SaveAs("dxyPtTrendNorm_" + theStrDate + theStrAlignment + ".png");

    TCanvas *dzNormPtTrend = new TCanvas("dzNormPtTrend", "dzNormPtTrend", 1200, 600);
    arrangeCanvas(
        dzNormPtTrend, dzNormPtMeanTrend, dzNormPtWidthTrend, nFiles_, LegLabels, theDate, false, setAutoLimits);

    dzNormPtTrend->SaveAs("dzPtTrendNorm_" + theStrDate + theStrAlignment + ".pdf");
    dzNormPtTrend->SaveAs("dzPtTrendNorm_" + theStrDate + theStrAlignment + ".png");

    delete dxyNormPtTrend;
    delete dzNormPtTrend;
  }

  // delete all news

  delete dxyNormPhiTrend;
  delete dzNormPhiTrend;
  delete dxyNormEtaTrend;
  delete dzNormEtaTrend;

  // Bias plots

  TCanvas *BiasesCanvas = new TCanvas("BiasCanvas", "BiasCanvas", 1200, 1200);
  arrangeBiasCanvas(BiasesCanvas,
                    dxyPhiMeanTrend,
                    dzPhiMeanTrend,
                    dxyEtaMeanTrend,
                    dzEtaMeanTrend,
                    nFiles_,
                    LegLabels,
                    theDate,
                    setAutoLimits);

  BiasesCanvas->SaveAs("BiasesCanvas_" + theStrDate + theStrAlignment + ".pdf");
  BiasesCanvas->SaveAs("BiasesCanvas_" + theStrDate + theStrAlignment + ".png");

  // Bias plots (x and y)

  TCanvas *BiasesCanvasXY = new TCanvas("BiasCanvasXY", "BiasCanvasXY", 1200, 1200);
  arrangeBiasCanvas(BiasesCanvasXY,
                    dxPhiMeanTrend,
                    dyPhiMeanTrend,
                    dxEtaMeanTrend,
                    dyEtaMeanTrend,
                    nFiles_,
                    LegLabels,
                    theDate,
                    setAutoLimits);

  BiasesCanvasXY->SaveAs("BiasesCanvasXY_" + theStrDate + theStrAlignment + ".pdf");
  BiasesCanvasXY->SaveAs("BiasesCanvasXY_" + theStrDate + theStrAlignment + ".png");

  // Bias plots (ladders and module number)
  if (nLadders_ > 0 && nModZ_ > 0) {
    TCanvas *BiasesCanvasLayer1 = new TCanvas("BiasCanvasLayer1", "BiasCanvasLayer1", 1200, 1200);
    arrangeBiasCanvas(BiasesCanvasLayer1,
                      dxyLadderMeanTrend,
                      dzLadderMeanTrend,
                      dxyModZMeanTrend,
                      dzModZMeanTrend,
                      nFiles_,
                      LegLabels,
                      theDate,
                      setAutoLimits);

    BiasesCanvasLayer1->SaveAs("BiasesCanvasLayer1_" + theStrDate + theStrAlignment + ".pdf");
    BiasesCanvasLayer1->SaveAs("BiasesCanvasLayer1_" + theStrDate + theStrAlignment + ".png");
    delete BiasesCanvasLayer1;
  }

  TCanvas *dxyPhiBiasCanvas = new TCanvas("dxyPhiBiasCanvas", "dxyPhiBiasCanvas", 600, 600);
  TCanvas *dxyEtaBiasCanvas = new TCanvas("dxyEtaBiasCanvas", "dxyEtaBiasCanvas", 600, 600);
  TCanvas *dzPhiBiasCanvas = new TCanvas("dzPhiBiasCanvas", "dzPhiBiasCanvas", 600, 600);
  TCanvas *dzEtaBiasCanvas = new TCanvas("dzEtaBiasCanvas", "dzEtaBiasCanvas", 600, 600);

  arrangeCanvas(dxyPhiBiasCanvas, dxyPhiMeanTrend, dxyPhiWidthTrend, nFiles_, LegLabels, theDate, true, setAutoLimits);
  arrangeCanvas(dzPhiBiasCanvas, dzPhiMeanTrend, dzPhiWidthTrend, nFiles_, LegLabels, theDate, true, setAutoLimits);
  arrangeCanvas(dxyEtaBiasCanvas, dxyEtaMeanTrend, dxyEtaWidthTrend, nFiles_, LegLabels, theDate, true, setAutoLimits);
  arrangeCanvas(dzEtaBiasCanvas, dzEtaMeanTrend, dzEtaWidthTrend, nFiles_, LegLabels, theDate, true, setAutoLimits);

  dxyPhiBiasCanvas->SaveAs("dxyPhiBiasCanvas_" + theStrDate + theStrAlignment + ".pdf");
  dxyEtaBiasCanvas->SaveAs("dxyEtaBiasCanvas_" + theStrDate + theStrAlignment + ".pdf");
  dzPhiBiasCanvas->SaveAs("dzPhiBiasCanvas_" + theStrDate + theStrAlignment + ".pdf");
  dzEtaBiasCanvas->SaveAs("dzEtaBiasCanvas_" + theStrDate + theStrAlignment + ".pdf");

  dxyPhiBiasCanvas->SaveAs("dxyPhiBiasCanvas_" + theStrDate + theStrAlignment + ".png");
  dxyEtaBiasCanvas->SaveAs("dxyEtaBiasCanvas_" + theStrDate + theStrAlignment + ".png");
  dzPhiBiasCanvas->SaveAs("dzPhiBiasCanvas_" + theStrDate + theStrAlignment + ".png");
  dzEtaBiasCanvas->SaveAs("dzEtaBiasCanvas_" + theStrDate + theStrAlignment + ".png");

  // delete all news

  delete BiasesCanvas;
  delete BiasesCanvasXY;
  delete dxyPhiBiasCanvas;
  delete dxyEtaBiasCanvas;
  delete dzPhiBiasCanvas;
  delete dzEtaBiasCanvas;

  // Resolution plots
  TCanvas *ResolutionsCanvas = new TCanvas("ResolutionsCanvas", "ResolutionsCanvas", 1200, 1200);
  arrangeBiasCanvas(ResolutionsCanvas,
                    dxyPhiWidthTrend,
                    dzPhiWidthTrend,
                    dxyEtaWidthTrend,
                    dzEtaWidthTrend,
                    nFiles_,
                    LegLabels,
                    theDate,
                    setAutoLimits);

  ResolutionsCanvas->SaveAs("ResolutionsCanvas_" + theStrDate + theStrAlignment + ".pdf");
  ResolutionsCanvas->SaveAs("ResolutionsCanvas_" + theStrDate + theStrAlignment + ".png");

  TCanvas *ResolutionsCanvasXY = new TCanvas("ResolutionsCanvasXY", "ResolutionsCanvasXY", 1200, 1200);
  arrangeBiasCanvas(ResolutionsCanvasXY,
                    dxPhiWidthTrend,
                    dyPhiWidthTrend,
                    dxEtaWidthTrend,
                    dyEtaWidthTrend,
                    nFiles_,
                    LegLabels,
                    theDate,
                    setAutoLimits);

  ResolutionsCanvasXY->SaveAs("ResolutionsCanvasXY_" + theStrDate + theStrAlignment + ".pdf");
  ResolutionsCanvasXY->SaveAs("ResolutionsCanvasXY_" + theStrDate + theStrAlignment + ".png");

  if (nLadders_ > 0 && nModZ_ > 0) {
    TCanvas *ResolutionsCanvasLayer1 = new TCanvas("ResolutionsCanvasLayer1", "ResolutionsCanvasLayer1", 1200, 1200);
    arrangeBiasCanvas(ResolutionsCanvasLayer1,
                      dxyLadderWidthTrend,
                      dzLadderWidthTrend,
                      dxyModZWidthTrend,
                      dzModZWidthTrend,
                      nFiles_,
                      LegLabels,
                      theDate,
                      setAutoLimits);

    ResolutionsCanvasLayer1->SaveAs("ResolutionsCanvasLayer1_" + theStrDate + theStrAlignment + ".pdf");
    ResolutionsCanvasLayer1->SaveAs("ResolutionsCanvasLayer1_" + theStrDate + theStrAlignment + ".png");
    delete ResolutionsCanvasLayer1;
  }

  // Pull plots
  TCanvas *PullsCanvas = new TCanvas("PullsCanvas", "PullsCanvas", 1200, 1200);
  arrangeBiasCanvas(PullsCanvas,
                    dxyNormPhiWidthTrend,
                    dzNormPhiWidthTrend,
                    dxyNormEtaWidthTrend,
                    dzNormEtaWidthTrend,
                    nFiles_,
                    LegLabels,
                    theDate,
                    setAutoLimits);

  PullsCanvas->SaveAs("PullsCanvas_" + theStrDate + theStrAlignment + ".pdf");
  PullsCanvas->SaveAs("PullsCanvas_" + theStrDate + theStrAlignment + ".png");

  if (nLadders_ > 0 && nModZ_ > 0) {
    TCanvas *PullsCanvasLayer1 = new TCanvas("PullsCanvasLayer1", "PullsCanvasLayer1", 1200, 1200);
    arrangeBiasCanvas(PullsCanvasLayer1,
                      dxyNormLadderWidthTrend,
                      dzNormLadderWidthTrend,
                      dxyNormModZWidthTrend,
                      dzNormModZWidthTrend,
                      nFiles_,
                      LegLabels,
                      theDate,
                      setAutoLimits);

    PullsCanvasLayer1->SaveAs("PullsCanvasLayer1_" + theStrDate + theStrAlignment + ".pdf");
    PullsCanvasLayer1->SaveAs("PullsCanvasLayer1_" + theStrDate + theStrAlignment + ".png");
    delete PullsCanvasLayer1;
  }

  // delete all news
  delete ResolutionsCanvas;
  delete ResolutionsCanvasXY;
  delete PullsCanvas;

  // 2D Maps

  if (do2DMaps) {
    TCanvas *dxyAbsMap = new TCanvas("dxyAbsMap", "dxyAbsMap", 1200, 500 * nFiles_);
    arrangeCanvas2D(dxyAbsMap, t_dxyMeanMap, t_dxyWidthMap, nFiles_, LegLabels, theDate);
    dxyAbsMap->SaveAs("dxyAbsMap_" + theStrDate + theStrAlignment + ".pdf");
    dxyAbsMap->SaveAs("dxyAbsMap_" + theStrDate + theStrAlignment + ".png");

    TCanvas *dzAbsMap = new TCanvas("dzAbsMap", "dzAbsMap", 1200, 500 * nFiles_);
    arrangeCanvas2D(dzAbsMap, t_dzMeanMap, t_dzWidthMap, nFiles_, LegLabels, theDate);
    dzAbsMap->SaveAs("dzAbsMap_" + theStrDate + theStrAlignment + ".pdf");
    dzAbsMap->SaveAs("dzAbsMap_" + theStrDate + theStrAlignment + ".png");

    TCanvas *dxyNormMap = new TCanvas("dxyNormMap", "dxyNormMap", 1200, 500 * nFiles_);
    arrangeCanvas2D(dxyNormMap, t_dxyNormMeanMap, t_dxyNormWidthMap, nFiles_, LegLabels, theDate);
    dxyNormMap->SaveAs("dxyNormMap_" + theStrDate + theStrAlignment + ".pdf");
    dxyNormMap->SaveAs("dxyNormMap_" + theStrDate + theStrAlignment + ".png");

    TCanvas *dzNormMap = new TCanvas("dzNormMap", "dzNormMap", 1200, 500 * nFiles_);
    arrangeCanvas2D(dzNormMap, t_dzNormMeanMap, t_dzNormWidthMap, nFiles_, LegLabels, theDate);
    dzNormMap->SaveAs("dzNormMap_" + theStrDate + theStrAlignment + ".pdf");
    dzNormMap->SaveAs("dzNormMap_" + theStrDate + theStrAlignment + ".png");

    delete dxyAbsMap;
    delete dzAbsMap;
    delete dxyNormMap;
    delete dzNormMap;

    // L1 Map

    TCanvas *dxyAbsL1Map = new TCanvas("dxyAbsL1Map", "dxyAbsL1Map", 1200, 500 * nFiles_);
    arrangeCanvas2D(dxyAbsL1Map, t_dxyMeanL1Map, t_dxyWidthL1Map, nFiles_, LegLabels, theDate);
    dxyAbsL1Map->SaveAs("dxyAbsL1Map_" + theStrDate + theStrAlignment + ".pdf");
    dxyAbsL1Map->SaveAs("dxyAbsL1Map_" + theStrDate + theStrAlignment + ".png");

    TCanvas *dzAbsL1Map = new TCanvas("dzAbsL1Map", "dzAbsL1Map", 1200, 500 * nFiles_);
    arrangeCanvas2D(dzAbsL1Map, t_dzMeanL1Map, t_dzWidthL1Map, nFiles_, LegLabels, theDate);
    dzAbsL1Map->SaveAs("dzAbsL1Map_" + theStrDate + theStrAlignment + ".pdf");
    dzAbsL1Map->SaveAs("dzAbsL1Map_" + theStrDate + theStrAlignment + ".png");

    TCanvas *dxyNormL1Map = new TCanvas("dxyNormL1Map", "dxyNormL1Map", 1200, 500 * nFiles_);
    arrangeCanvas2D(dxyNormL1Map, t_dxyNormMeanL1Map, t_dxyNormWidthL1Map, nFiles_, LegLabels, theDate);
    dxyNormL1Map->SaveAs("dxyNormL1Map_" + theStrDate + theStrAlignment + ".pdf");
    dxyNormL1Map->SaveAs("dxyNormL1Map_" + theStrDate + theStrAlignment + ".png");

    TCanvas *dzNormL1Map = new TCanvas("dzNormL1Map", "dzNormL1Map", 1200, 500 * nFiles_);
    arrangeCanvas2D(dzNormL1Map, t_dzNormMeanL1Map, t_dzNormWidthL1Map, nFiles_, LegLabels, theDate);
    dzNormL1Map->SaveAs("dzNormL1Map_" + theStrDate + theStrAlignment + ".pdf");
    dzNormL1Map->SaveAs("dzNormL1Map_" + theStrDate + theStrAlignment + ".png");

    delete dxyAbsL1Map;
    delete dzAbsL1Map;
    delete dxyNormL1Map;
    delete dzNormL1Map;
  }

  delete thePlotLimits;

  // delete everything in the source list
  for (std::vector<PVValidationVariables *>::iterator it = sourceList.begin(); it != sourceList.end(); ++it) {
    delete (*it);
  }

  TTimeStamp plotting_done;

  std::cout << " ======   TIMING REPORT ====== " << std::endl;
  std::cout << "time tp initialize = " << initialization_done.AsDouble() - start_time.AsDouble() << "s" << std::endl;
  std::cout << "time to cache      = " << caching_done.AsDouble() - initialization_done.AsDouble() << "s" << std::endl;
  //  std::cout<<"time to fill 1D    = "<<filling1D_done.AsDouble()-caching_done.AsDouble()<<"s"<<std::endl;
  std::cout << "time to fit        = " << filling2D_done.AsDouble() - caching_done.AsDouble() << "s" << std::endl;
  std::cout << "time to plot       = " << plotting_done.AsDouble() - filling2D_done.AsDouble() << "s" << std::endl;

  timer.Stop();
  timer.Print();
}

//*************************************************************
void arrangeBiasCanvas(TCanvas *canv,
                       TH1F *dxyPhiMeanTrend[100],
                       TH1F *dzPhiMeanTrend[100],
                       TH1F *dxyEtaMeanTrend[100],
                       TH1F *dzEtaMeanTrend[100],
                       Int_t nFiles,
                       TString LegLabels[10],
                       TString theDate,
                       bool setAutoLimits) {
  //*************************************************************

  TLegend *lego = new TLegend(0.22, 0.80, 0.79, 0.91);
  // might be useful if many objects are compared
  if (nFiles > 3) {
    lego->SetNColumns(2);
  }

  lego->SetFillColor(10);
  if (nFiles > 3) {
    lego->SetTextSize(0.032);
  } else {
    lego->SetTextSize(0.042);
  }
  lego->SetTextFont(42);
  lego->SetFillColor(10);
  lego->SetLineColor(10);
  lego->SetShadowColor(10);

  TPaveText *ptDate = new TPaveText(0.20, 0.95, 0.50, 0.99, "blNDC");
  //ptDate->SetFillColor(kYellow);
  ptDate->SetFillColor(10);
  ptDate->SetBorderSize(1);
  ptDate->SetLineColor(kBlue);
  ptDate->SetLineWidth(1);
  ptDate->SetTextFont(32);
  TText *textDate = ptDate->AddText(theDate);
  textDate->SetTextSize(0.04);
  textDate->SetTextColor(kBlue);

  canv->SetFillColor(10);
  canv->Divide(2, 2);

  canv->cd(1)->SetBottomMargin(0.14);
  canv->cd(1)->SetLeftMargin(0.18);
  canv->cd(1)->SetRightMargin(0.01);
  canv->cd(1)->SetTopMargin(0.06);

  canv->cd(2)->SetBottomMargin(0.14);
  canv->cd(2)->SetLeftMargin(0.18);
  canv->cd(2)->SetRightMargin(0.01);
  canv->cd(2)->SetTopMargin(0.06);

  canv->cd(3)->SetBottomMargin(0.14);
  canv->cd(3)->SetLeftMargin(0.18);
  canv->cd(3)->SetRightMargin(0.01);
  canv->cd(3)->SetTopMargin(0.06);

  canv->cd(4)->SetBottomMargin(0.14);
  canv->cd(4)->SetLeftMargin(0.18);
  canv->cd(4)->SetRightMargin(0.01);
  canv->cd(4)->SetTopMargin(0.06);

  TH1F *dBiasTrend[4][nFiles];

  for (Int_t i = 0; i < nFiles; i++) {
    dBiasTrend[0][i] = dxyPhiMeanTrend[i];
    dBiasTrend[1][i] = dzPhiMeanTrend[i];
    dBiasTrend[2][i] = dxyEtaMeanTrend[i];
    dBiasTrend[3][i] = dzEtaMeanTrend[i];
  }

  Double_t absmin[4] = {999., 999., 999., 999.};
  Double_t absmax[4] = {-999., -999. - 999., -999.};

  for (Int_t k = 0; k < 4; k++) {
    canv->cd(k + 1);

    for (Int_t i = 0; i < nFiles; i++) {
      if (TString(canv->GetName()).Contains("BareResiduals")) {
        dBiasTrend[k][i]->Scale(1. / dBiasTrend[k][i]->GetSumOfWeights());
      }

      if (dBiasTrend[k][i]->GetMaximum() > absmax[k])
        absmax[k] = dBiasTrend[k][i]->GetMaximum();
      if (dBiasTrend[k][i]->GetMinimum() < absmin[k])
        absmin[k] = dBiasTrend[k][i]->GetMinimum();
    }

    Double_t safeDelta = (absmax[k] - absmin[k]) / 8.;

    // if(safeDelta<0.1*absmax[k]) safeDelta*=16;

    Double_t theExtreme = std::max(absmax[k], TMath::Abs(absmin[k]));

    for (Int_t i = 0; i < nFiles; i++) {
      if (i == 0) {
        TString theTitle = dBiasTrend[k][i]->GetName();

        //std::cout << theTitle<< " --->" << safeDelta << std::endl;

        // if the autoLimits are not set
        if (!setAutoLimits) {
          params::measurement range = getTheRangeUser(dBiasTrend[k][i], thePlotLimits);
          dBiasTrend[k][i]->GetYaxis()->SetRangeUser(range.first, range.second);

        } else {
          if (theTitle.Contains("width")) {
            if (theTitle.Contains("Norm"))
              safeDelta = (theTitle.Contains("ladder") == true || theTitle.Contains("modZ") == true) ? 1. : safeDelta;
            else
              safeDelta = (theTitle.Contains("ladder") == true || theTitle.Contains("modZ") == true) ? safeDelta * 10.
                                                                                                     : safeDelta;

            dBiasTrend[k][i]->GetYaxis()->SetRangeUser(0., theExtreme + (safeDelta / 2.));
          } else {
            if (theTitle.Contains("Norm")) {
              dBiasTrend[k][i]->GetYaxis()->SetRangeUser(std::min(-0.48, absmin[k] - (safeDelta / 2.)),
                                                         std::max(0.48, absmax[k] + (safeDelta / 2.)));
            } else if (theTitle.Contains("h_probe")) {
              TGaxis::SetMaxDigits(4);
              dBiasTrend[k][i]->GetYaxis()->SetRangeUser(0., theExtreme + (safeDelta * 2.));
            } else {
              safeDelta = (theTitle.Contains("ladder") == true || theTitle.Contains("modZ") == true) ? safeDelta * 10.
                                                                                                     : safeDelta;

              dBiasTrend[k][i]->GetYaxis()->SetRangeUser(-theExtreme - (safeDelta / 2.), theExtreme + (safeDelta / 2.));
            }
          }
        }

        if (TString(canv->GetName()).Contains("BareResiduals") && (k == 0 || k == 2)) {
          dBiasTrend[k][i]->GetXaxis()->SetRangeUser(-0.11, 0.11);
        }

        dBiasTrend[k][i]->Draw("e1");
        makeNewXAxis(dBiasTrend[k][i]);
        Int_t nbins = dBiasTrend[k][i]->GetNbinsX();
        Double_t lowedge = dBiasTrend[k][i]->GetBinLowEdge(1);
        Double_t highedge = dBiasTrend[k][i]->GetBinLowEdge(nbins + 1);

        /*
	  TH1F* zeros = DrawZero(dBiasTrend[k][i],nbins,lowedge,highedge,1);
	  zeros->Draw("PLsame"); 
	*/

        Double_t theC = -1.;

        if (theTitle.Contains("width")) {
          if (theTitle.Contains("Norm")) {
            theC = 1.;
          } else {
            theC = -1.;
          }
        } else {
          theC = 0.;
        }

        TH1F *theConst = DrawConstant(dBiasTrend[k][i], nbins, lowedge, highedge, 1, theC);
        theConst->Draw("PLsame");

      } else {
        if (TString(canv->GetName()).Contains("BareResiduals") && (k == 0 || k == 2)) {
          dBiasTrend[k][i]->GetXaxis()->SetRangeUser(-0.11, 0.11);
        }

        dBiasTrend[k][i]->Draw("e1sames");
      }

      if (k == 0) {
        lego->AddEntry(dBiasTrend[k][i], LegLabels[i]);
      }
    }

    lego->Draw();

    TPad *current_pad = static_cast<TPad *>(canv->GetPad(k + 1));
    CMS_lumi(current_pad, 6, 33);
    if (theDate != "")
      ptDate->Draw("same");
  }
}

//*************************************************************
void arrangeCanvas(TCanvas *canv,
                   TH1F *meanplots[100],
                   TH1F *widthplots[100],
                   Int_t nFiles,
                   TString LegLabels[10],
                   TString theDate,
                   bool onlyBias,
                   bool setAutoLimits) {
  //*************************************************************

  TPaveText *ali = new TPaveText(0.18, 0.85, 0.50, 0.93, "NDC");
  ali->SetFillColor(10);
  ali->SetTextColor(1);
  ali->SetTextFont(42);
  ali->SetMargin(0.);
  ali->SetLineColor(10);
  ali->SetShadowColor(10);
  TText *alitext = ali->AddText("Alignment: PCL");
  alitext->SetTextSize(0.04);

  TLegend *lego = new TLegend(0.22, 0.80, 0.78, 0.91);
  // in case many objects are compared
  if (nFiles > 3) {
    lego->SetNColumns(2);
  }
  // TLegend *lego = new TLegend(0.18,0.77,0.50,0.86);
  lego->SetFillColor(10);
  if (nFiles > 3) {
    lego->SetTextSize(0.03);
  } else {
    lego->SetTextSize(0.04);
  }
  lego->SetTextFont(42);
  lego->SetFillColor(10);
  lego->SetLineColor(10);
  lego->SetShadowColor(10);

  TPaveText *ptDate = nullptr;

  canv->SetFillColor(10);

  if (!onlyBias) {
    ptDate = new TPaveText(0.20, 0.95, 0.50, 0.99, "blNDC");
  } else {
    ptDate = new TPaveText(0.20, 0.95, 0.50, 0.99, "blNDC");
  }

  //ptDate->SetFillColor(kYellow);
  ptDate->SetFillColor(10);
  ptDate->SetBorderSize(1);
  ptDate->SetLineColor(kBlue);
  ptDate->SetLineWidth(1);
  ptDate->SetTextFont(42);
  TText *textDate = ptDate->AddText(theDate);
  textDate->SetTextSize(0.04);
  textDate->SetTextColor(kBlue);

  if (!onlyBias) {
    canv->Divide(2, 1);

    canv->cd(1)->SetBottomMargin(0.14);
    canv->cd(1)->SetLeftMargin(0.17);
    canv->cd(1)->SetRightMargin(0.02);
    canv->cd(1)->SetTopMargin(0.06);

    canv->cd(2)->SetBottomMargin(0.14);
    canv->cd(2)->SetLeftMargin(0.17);
    canv->cd(2)->SetRightMargin(0.02);
    canv->cd(2)->SetTopMargin(0.06);
    canv->cd(1);

  } else {
    canv->cd()->SetBottomMargin(0.14);
    canv->cd()->SetLeftMargin(0.17);
    canv->cd()->SetRightMargin(0.02);
    canv->cd()->SetTopMargin(0.06);
    canv->cd();
  }

  Double_t absmin(999.);
  Double_t absmax(-999.);

  for (Int_t i = 0; i < nFiles; i++) {
    if (meanplots[i]->GetMaximum() > absmax)
      absmax = meanplots[i]->GetMaximum();
    if (meanplots[i]->GetMinimum() < absmin)
      absmin = meanplots[i]->GetMinimum();
  }

  Double_t safeDelta = (absmax - absmin) / 2.;
  Double_t theExtreme = std::max(absmax, TMath::Abs(absmin));

  for (Int_t i = 0; i < nFiles; i++) {
    if (i == 0) {
      // if the autoLimits are not set
      if (!setAutoLimits) {
        params::measurement range = getTheRangeUser(meanplots[i], thePlotLimits);
        meanplots[i]->GetYaxis()->SetRangeUser(range.first, range.second);

      } else {
        TString theTitle = meanplots[i]->GetName();
        if (theTitle.Contains("Norm")) {
          meanplots[i]->GetYaxis()->SetRangeUser(std::min(-0.48, absmin - safeDelta),
                                                 std::max(0.48, absmax + safeDelta));
        } else {
          if (!onlyBias) {
            meanplots[i]->GetYaxis()->SetRangeUser(absmin - safeDelta, absmax + safeDelta);
          } else {
            meanplots[i]->GetYaxis()->SetRangeUser(-theExtreme - (TMath::Abs(absmin) / 10.),
                                                   theExtreme + (TMath::Abs(absmax / 10.)));
          }
        }
      }

      meanplots[i]->Draw("e1");
      if (TString(meanplots[i]->GetName()).Contains("pT")) {
        //meanplots[i]->Draw("HIST][same");
        gPad->SetLogx();
        gPad->SetGridx();
        gPad->SetGridy();
      } else {
        makeNewXAxis(meanplots[i]);
      }

      if (onlyBias) {
        canv->cd();
        Int_t nbins = meanplots[i]->GetNbinsX();
        Double_t lowedge = meanplots[i]->GetBinLowEdge(1);
        Double_t highedge = meanplots[i]->GetBinLowEdge(nbins + 1);

        TH1F *hzero = DrawZero(meanplots[i], nbins, lowedge, highedge, 2);
        hzero->Draw("PLsame");
      }
    } else
      meanplots[i]->Draw("e1sames");
    //if(TString(meanplots[i]->GetName()).Contains("pT")){
    //  meanplots[i]->Draw("HIST][same");
    // }

    lego->AddEntry(meanplots[i], LegLabels[i]);
  }

  lego->Draw();

  //ali->Draw("same");
  //ptDate->Draw("same");

  TPad *current_pad;
  if (!onlyBias) {
    current_pad = static_cast<TPad *>(canv->GetPad(1));
  } else {
    current_pad = static_cast<TPad *>(canv->GetPad(0));
  }

  CMS_lumi(current_pad, 6, 33);
  if (theDate != "")
    ptDate->Draw("same");

  if (!onlyBias) {
    canv->cd(2);
    Double_t absmax2(-999.);

    for (Int_t i = 0; i < nFiles; i++) {
      if (widthplots[i]->GetMaximum() > absmax2)
        absmax2 = widthplots[i]->GetMaximum();
    }

    Double_t safeDelta2 = absmax2 / 3.;

    for (Int_t i = 0; i < nFiles; i++) {
      widthplots[i]->GetXaxis()->SetLabelOffset(999);
      widthplots[i]->GetXaxis()->SetTickLength(0);

      if (i == 0) {
        if (!setAutoLimits) {
          params::measurement range = getTheRangeUser(widthplots[i], thePlotLimits);
          widthplots[i]->GetYaxis()->SetRangeUser(range.first, range.second);
        } else {
          widthplots[i]->SetMinimum(0.5);
          widthplots[i]->SetMaximum(absmax2 + safeDelta2);
        }

        widthplots[i]->Draw("e1");
        if (TString(widthplots[i]->GetName()).Contains("pT")) {
          //widthplots[i]->Draw("HIST][same");
          gPad->SetGridx();
          gPad->SetGridy();
        }
        makeNewXAxis(widthplots[i]);
      } else {
        widthplots[i]->Draw("e1sames");
        if (TString(widthplots[i]->GetName()).Contains("pT")) {
          //widthplots[i]->Draw("HIST][same");
        }
      }
    }

    lego->Draw();

    TPad *current_pad2 = static_cast<TPad *>(canv->GetPad(2));
    CMS_lumi(current_pad2, 6, 33);
    if (theDate != "")
      ptDate->Draw("same");
  }
}

//*************************************************************
void arrangeCanvas2D(
    TCanvas *canv, TH2F *meanmaps[100], TH2F *widthmaps[100], Int_t nFiles, TString LegLabels[10], TString theDate)
//*************************************************************
{
  TLegend *lego = new TLegend(0.18, 0.75, 0.58, 0.92);
  lego->SetFillColor(10);
  lego->SetTextSize(0.05);
  lego->SetTextFont(42);
  lego->SetFillColor(10);
  lego->SetLineColor(10);
  lego->SetShadowColor(10);

  TPaveText *pt[nFiles];
  TPaveText *pt2[nFiles];
  TPaveText *pt3[nFiles];

  for (Int_t i = 0; i < nFiles; i++) {
    pt[i] = new TPaveText(0.13, 0.95, 0.191, 0.975, "NDC");
    //pt[i] = new TPaveText(gPad->GetUxmin(),gPad->GetUymax()+0.3,gPad->GetUxmin()+0.6,gPad->GetUymax()+0.3,"NDC");
    //std::cout<<"gPad->GetUymax():"<<gPad->GetUymax()<<std::endl;
    //pt[i] = new TPaveText(gPad->GetLeftMargin(),0.95,gPad->GetLeftMargin()+0.3,0.98,"NDC");
    pt[i]->SetFillColor(10);
    pt[i]->SetTextColor(1);
    pt[i]->SetTextFont(61);
    pt[i]->SetTextAlign(22);
    TText *text1 = pt[i]->AddText("CMS");  // preliminary 2015 p-p data, #sqrt{s}=8 TeV "+LegLabels[i]);
    text1->SetTextSize(0.05);
    //delete text1;

    //float extraOverCmsTextSize  = 0.76;

    pt2[i] = new TPaveText(0.21, 0.95, 0.25, 0.975, "NDC");
    pt2[i]->SetFillColor(10);
    pt2[i]->SetTextColor(1);
    //pt[i]->SetTextSize(0.05);
    pt2[i]->SetTextFont(52);
    pt2[i]->SetTextAlign(12);
    // TText *text2 = pt2->AddText("run: "+theDate);
    TText *text2 = pt2[i]->AddText("INTERNAL");
    text2->SetTextSize(0.06 * extraOverCmsTextSize);

    pt3[i] = new TPaveText(0.55, 0.955, 0.95, 0.98, "NDC");
    pt3[i]->SetFillColor(10);
    pt3[i]->SetTextColor(kBlue);
    pt3[i]->SetTextFont(61);
    pt3[i]->SetTextAlign(22);
    // TText *text2 = pt2->AddText("run: "+theDate);
    TText *text3 = pt3[i]->AddText(LegLabels[i]);
    text3->SetTextSize(0.05);
  }

  canv->SetFillColor(10);
  canv->Divide(2, nFiles);

  Double_t absmin(999.);
  Double_t absmax(-999.);
  Double_t maxwidth(-999.);

  for (Int_t i = 0; i < nFiles; i++) {
    if (widthmaps[i]->GetMaximum() > maxwidth)
      maxwidth = widthmaps[i]->GetMaximum();
    if (meanmaps[i]->GetMaximum() > absmax)
      absmax = meanmaps[i]->GetMaximum();
    if (meanmaps[i]->GetMinimum() < absmin)
      absmin = meanmaps[i]->GetMinimum();
  }

  /*
  const Int_t nLevels = 255;
  Double_t levels[nLevels];

  for(int i = 0; i < nLevels; i++) {
    levels[i] = absmin + (absmax - absmin) / (nLevels - 1) * (i);
  }
  */

  //Double_t theExtreme = std::min(std::abs(absmin),std::abs(absmax));

  for (Int_t i = 0; i < nFiles; i++) {
    canv->cd(2 * i + 1)->SetBottomMargin(0.13);
    canv->cd(2 * i + 1)->SetLeftMargin(0.12);
    canv->cd(2 * i + 1)->SetRightMargin(0.19);
    canv->cd(2 * i + 1)->SetTopMargin(0.08);

    //meanmaps[i]->SetContour((sizeof(levels)/sizeof(Double_t)), levels);
    meanmaps[i]->GetZaxis()->SetRangeUser(absmin, absmax);
    //meanmaps[i]->GetZaxis()->SetRangeUser(-theExtreme,theExtreme);
    meanmaps[i]->Draw("colz1");

    //TH2F* cloned = (TH2F*)meanmaps[i]->DrawClone("colz");// draw "axes", "contents", "statistics box
    //makeNewPairOfAxes(cloned);
    //meanmaps[i]->GetZaxis()->SetRangeUser(absmin, absmax); // ... set the range ...
    //meanmaps[i]->Draw("colzsame"); // draw the "color palette"

    makeNewPairOfAxes(meanmaps[i]);

    pt[i]->Draw("same");
    pt2[i]->Draw("same");
    pt3[i]->Draw("same");

    canv->cd(2 * (i + 1))->SetBottomMargin(0.13);
    canv->cd(2 * (i + 1))->SetLeftMargin(0.12);
    canv->cd(2 * (i + 1))->SetRightMargin(0.19);
    canv->cd(2 * (i + 1))->SetTopMargin(0.08);

    widthmaps[i]->Draw("colz1");
    makeNewPairOfAxes(widthmaps[i]);

    widthmaps[i]->GetZaxis()->SetRangeUser(0., maxwidth);

    pt[i]->Draw("same");
    pt2[i]->Draw("same");
    pt3[i]->Draw("same");
  }
}

//*************************************************************
void arrangeFitCanvas(TCanvas *canv, TH1F *meanplots[100], Int_t nFiles, TString LegLabels[10], TString theDate)
//*************************************************************
{
  canv->SetBottomMargin(0.14);
  canv->SetLeftMargin(0.1);
  canv->SetRightMargin(0.02);
  canv->SetTopMargin(0.08);

  TLegend *lego = new TLegend(0.12, 0.80, 0.82, 0.89);
  lego->SetFillColor(10);
  lego->SetTextSize(0.035);
  lego->SetTextFont(42);
  lego->SetFillColor(10);
  lego->SetLineColor(10);
  if (nFiles > 3) {
    lego->SetNColumns(2);
  }
  lego->SetShadowColor(10);

  TPaveText *ptDate = new TPaveText(0.12, 0.95, 0.50, 0.99, "blNDC");
  //ptDate->SetFillColor(kYellow);
  ptDate->SetFillColor(10);
  ptDate->SetBorderSize(1);
  ptDate->SetLineColor(kBlue);
  ptDate->SetLineWidth(1);
  ptDate->SetTextFont(32);
  TText *textDate = ptDate->AddText(theDate);
  textDate->SetTextSize(0.04);
  textDate->SetTextColor(kBlue);

  TF1 *fleft[nFiles];
  TF1 *fright[nFiles];
  TF1 *fall[nFiles];

  TF1 *FitDzUp[nFiles];
  TF1 *FitDzDown[nFiles];

  for (Int_t j = 0; j < nFiles; j++) {
    Double_t deltaZ(0);
    Double_t sigmadeltaZ(-1);

    TCanvas *theNewCanvas2 = new TCanvas("NewCanvas2", "Fitting Canvas 2", 800, 600);
    theNewCanvas2->Divide(2, 1);

    TH1F *hnewUp = (TH1F *)meanplots[j]->Clone("hnewUp_dz_phi");
    TH1F *hnewDown = (TH1F *)meanplots[j]->Clone("hnewDown_dz_phi");

    fleft[j] = new TF1(Form("fleft_%i", j), fULine, _boundMin, _boundSx, 1);
    fright[j] = new TF1(Form("fright_%i", j), fULine, _boundDx, _boundMax, 1);
    fall[j] = new TF1(Form("fall_%i", j), fDLine, _boundSx, _boundDx, 1);

    FitULine(hnewUp);
    FitDzUp[j] = (TF1 *)hnewUp->GetListOfFunctions()->FindObject("lineUp");
    if (FitDzUp[j]) {
      fleft[j]->SetParameters(FitDzUp[j]->GetParameters());
      fleft[j]->SetParErrors(FitDzUp[j]->GetParErrors());
      hnewUp->GetListOfFunctions()->Add(fleft[j]);
      fright[j]->SetParameters(FitDzUp[j]->GetParameters());
      fright[j]->SetParErrors(FitDzUp[j]->GetParErrors());
      hnewUp->GetListOfFunctions()->Add(fright[j]);
      FitDzUp[j]->Delete();

      theNewCanvas2->cd(1);
      MakeNiceTF1Style(fright[j], meanplots[j]->GetLineColor());
      MakeNiceTF1Style(fleft[j], meanplots[j]->GetLineColor());
      fright[j]->Draw("same");
      fleft[j]->Draw("same");
    }

    FitDLine(hnewDown);
    FitDzDown[j] = (TF1 *)hnewDown->GetListOfFunctions()->FindObject("lineDown");

    if (FitDzDown[j]) {
      fall[j]->SetParameters(FitDzDown[j]->GetParameters());
      fall[j]->SetParErrors(FitDzDown[j]->GetParErrors());
      hnewDown->GetListOfFunctions()->Add(fall[j]);
      FitDzDown[j]->Delete();
      theNewCanvas2->cd(2);
      MakeNiceTF1Style(fall[j], meanplots[j]->GetLineColor());
      fall[j]->Draw("same");
      canv->cd();
      hnewUp->GetYaxis()->SetTitleOffset(0.7);
      if (j == 0) {
        hnewUp->Draw();
        makeNewXAxis(hnewUp);
      } else {
        hnewUp->Draw("same");
        makeNewXAxis(hnewUp);
      }
      fright[j]->Draw("same");
      fleft[j]->Draw("same");
      fall[j]->Draw("same");
    }

    if (j == nFiles - 1) {
      theNewCanvas2->Close();
    }

    deltaZ = (fright[j]->GetParameter(0) - fall[j]->GetParameter(0)) / 2;
    sigmadeltaZ = 0.5 * TMath::Sqrt(fright[j]->GetParError(0) * fright[j]->GetParError(0) +
                                    fall[j]->GetParError(0) * fall[j]->GetParError(0));
    TString MYOUT = Form(" : #Delta z = %.f #pm %.f #mum", deltaZ, sigmadeltaZ);

    lego->AddEntry(meanplots[j], LegLabels[j] + MYOUT);

    if (j == nFiles - 1) {
      outfile << deltaZ << "|" << sigmadeltaZ << std::endl;
    }

    delete theNewCanvas2;
  }

  //TkAlStyle::drawStandardTitle(Coll0T15);
  lego->Draw("same");
  CMS_lumi(canv, 6, 33);
  if (theDate != "")
    ptDate->Draw("same");
  //pt->Draw("same");
}

//*************************************************************
std::pair<params::measurement, params::measurement> fitStudentTResiduals(TH1 *hist)
//*************************************************************
{
  hist->SetMarkerStyle(21);
  hist->SetMarkerSize(0.8);
  hist->SetStats(true);

  double dx = hist->GetBinWidth(1);
  double nmax = hist->GetBinContent(hist->GetMaximumBin());
  double xmax = hist->GetBinCenter(hist->GetMaximumBin());
  double nn = 7 * nmax;

  int nb = hist->GetNbinsX();
  double n1 = hist->GetBinContent(1);
  double n9 = hist->GetBinContent(nb);
  double bg = 0.5 * (n1 + n9);

  double x1 = hist->GetBinCenter(1);
  double x9 = hist->GetBinCenter(nb);

  // create a TF1 with the range from x1 to x9 and 5 parameters

  TF1 *tp0Fcn = new TF1("tmp", tp0Fit, x1, x9, 5);

  tp0Fcn->SetParName(0, "mean");
  tp0Fcn->SetParName(1, "sigma");
  tp0Fcn->SetParName(2, "nu");
  tp0Fcn->SetParName(3, "area");
  tp0Fcn->SetParName(4, "BG");

  tp0Fcn->SetNpx(500);
  tp0Fcn->SetLineWidth(2);
  //tp0Fcn->SetLineColor(kMagenta);
  //tp0Fcn->SetLineColor(kGreen);
  tp0Fcn->SetLineColor(kRed);

  // set start values for some parameters:

  tp0Fcn->SetParameter(0, xmax);    // peak position
  tp0Fcn->SetParameter(1, 4 * dx);  // width
  tp0Fcn->SetParameter(2, 2.2);     // nu
  tp0Fcn->SetParameter(3, nn);      // N
  tp0Fcn->SetParameter(4, bg);

  hist->Fit("tmp", "R", "ep");
  // h->Fit("tmp","V+","ep");

  hist->Draw("histepsame");  // data again on top

  float res_mean = tp0Fcn->GetParameter(0);
  float res_width = tp0Fcn->GetParameter(1);

  float res_mean_err = tp0Fcn->GetParError(0);
  float res_width_err = tp0Fcn->GetParError(1);

  params::measurement resultM;
  params::measurement resultW;

  resultM = std::make_pair(res_mean, res_mean_err);
  resultW = std::make_pair(res_width, res_width_err);

  std::pair<params::measurement, params::measurement> result;

  result = std::make_pair(resultM, resultW);
  return result;
}

//*************************************************************
Double_t tp0Fit(Double_t *x, Double_t *par5)
//*************************************************************
{
  static int nn = 0;
  nn++;
  static double dx = 0.1;
  static double b1 = 0;
  if (nn == 1)
    b1 = x[0];
  if (nn == 2)
    dx = x[0] - b1;
  //
  //--  Mean and width:
  //
  double xm = par5[0];
  double t = (x[0] - xm) / par5[1];
  double tt = t * t;
  //
  //--  exponent:
  //
  double rn = par5[2];
  double xn = 0.5 * (rn + 1.0);
  //
  //--  Normalization needs Gamma function:
  //
  double pk = 0.0;

  if (rn > 0.0) {
    double pi = 3.14159265358979323846;
    double aa = dx / par5[1] / sqrt(rn * pi) * TMath::Gamma(xn) / TMath::Gamma(0.5 * rn);

    pk = par5[3] * aa * exp(-xn * log(1.0 + tt / rn));
  }

  return pk + par5[4];
}

//*************************************************************
params::measurement getMedian(TH1F *histo)
//*************************************************************
{
  Double_t median = 999;
  int nbins = histo->GetNbinsX();

  //extract median from histogram
  double *x = new double[nbins];
  double *y = new double[nbins];
  for (int j = 0; j < nbins; j++) {
    x[j] = histo->GetBinCenter(j + 1);
    y[j] = histo->GetBinContent(j + 1);
  }
  median = TMath::Median(nbins, x, y);

  delete[] x;
  x = nullptr;
  delete[] y;
  y = nullptr;

  params::measurement result;
  result = std::make_pair(median, median / TMath::Sqrt(histo->GetEntries()));

  return result;
}

//*************************************************************
params::measurement getMAD(TH1F *histo)
//*************************************************************
{
  int nbins = histo->GetNbinsX();
  Double_t median = getMedian(histo).first;
  Double_t x_lastBin = histo->GetBinLowEdge(nbins + 1);
  const char *HistoName = histo->GetName();
  TString Finalname = Form("resMed%s", HistoName);
  TH1F *newHisto = new TH1F(Finalname, Finalname, nbins, 0., x_lastBin);
  Double_t *residuals = new Double_t[nbins];
  Double_t *weights = new Double_t[nbins];

  for (int j = 0; j < nbins; j++) {
    residuals[j] = TMath::Abs(median - histo->GetBinCenter(j + 1));
    weights[j] = histo->GetBinContent(j + 1);
    newHisto->Fill(residuals[j], weights[j]);
  }

  Double_t theMAD = (getMedian(newHisto).first) * 1.4826;
  newHisto->Delete("");

  params::measurement result;
  result = std::make_pair(theMAD, theMAD / histo->GetEntries());

  return result;
}

//*************************************************************
std::pair<params::measurement, params::measurement> fitResiduals(TH1 *hist, bool singleTime)
//*************************************************************
{
  assert(hist != nullptr);

  if (hist->GetEntries() < 10) {
    // std::cout<<"hist name: "<<hist->GetName() << std::endl;
    return std::make_pair(std::make_pair(0., 0.), std::make_pair(0., 0.));
  }

  float maxHist = hist->GetXaxis()->GetXmax();
  float minHist = hist->GetXaxis()->GetXmin();
  float mean = hist->GetMean();
  float sigma = hist->GetRMS();

  if (TMath::IsNaN(mean) || TMath::IsNaN(sigma)) {
    mean = 0;
    //sigma= - hist->GetXaxis()->GetBinLowEdge(1) + hist->GetXaxis()->GetBinLowEdge(hist->GetNbinsX()+1);
    sigma = -minHist + maxHist;
    std::cout << "FitPVResiduals::fitResiduals(): histogram" << hist->GetName() << " mean or sigma are NaN!!"
              << std::endl;
  }

  TF1 func("tmp", "gaus", mean - 2. * sigma, mean + 2. * sigma);
  if (0 == hist->Fit(&func, "QNR")) {  // N: do not blow up file by storing fit!
    mean = func.GetParameter(1);
    sigma = func.GetParameter(2);

    if (!singleTime) {
      // second fit: three sigma of first fit around mean of first fit
      func.SetRange(std::max(mean - 2 * sigma, minHist), std::min(mean + 2 * sigma, maxHist));
      // I: integral gives more correct results if binning is too wide
      // L: Likelihood can treat empty bins correctly (if hist not weighted...)
      if (0 == hist->Fit(&func, "Q0LR")) {
        if (hist->GetFunction(func.GetName())) {  // Take care that it is later on drawn:
          hist->GetFunction(func.GetName())->ResetBit(TF1::kNotDraw);
        }
      }
    }
  }

  /*
  float res_mean  = func.GetParameter(1);
  float res_width = func.GetParameter(2);
  
  float res_mean_err  = func.GetParError(1);
  float res_width_err = func.GetParError(2);

  params::measurement resultM;
  params::measurement resultW;

  resultM = std::make_pair(res_mean,res_mean_err);
  resultW = std::make_pair(res_width,res_width_err);

  std::pair<params::measurement, params::measurement  > result;
  
  result = std::make_pair(resultM,resultW);
  */
  return std::make_pair(std::make_pair(func.GetParameter(1), func.GetParError(1)),
                        std::make_pair(func.GetParameter(2), func.GetParError(2)));
}

//*************************************************************
Double_t DoubleSidedCB(double *x, double *par) {
  //*************************************************************

  double m = x[0];
  double m0 = par[0];
  double sigma = par[1];
  double alphaL = par[2];
  double alphaR = par[3];
  double nL = par[4];
  double nR = par[5];
  double N = par[6];

  Double_t arg = m - m0;

  if (arg < 0.0) {
    Double_t t = (m - m0) / sigma;               //t < 0
    Double_t absAlpha = fabs((Double_t)alphaL);  //slightly redundant since alpha > 0 anyway, but never mind
    if (t >= -absAlpha) {                        //-absAlpha <= t < 0
      return N * exp(-0.5 * t * t);
    } else {
      Double_t a = TMath::Power(nL / absAlpha, nL) * exp(-0.5 * absAlpha * absAlpha);
      Double_t b = nL / absAlpha - absAlpha;
      return N * (a / TMath::Power(b - t, nL));  //b - t
    }
  } else {
    Double_t t = (m - m0) / sigma;  //t > 0
    Double_t absAlpha = fabs((Double_t)alphaR);
    if (t <= absAlpha) {  //0 <= t <= absAlpha
      return N * exp(-0.5 * t * t);
    } else {
      Double_t a = TMath::Power(nR / absAlpha, nR) * exp(-0.5 * absAlpha * absAlpha);
      Double_t b = nR / absAlpha - absAlpha;
      return N * (a / TMath::Power(b + t, nR));  //b + t
    }
  }
}

//*************************************************************
std::pair<params::measurement, params::measurement> fitResidualsCB(TH1 *hist)
//*************************************************************
{
  //hist->Rebin(2);

  float mean = hist->GetMean();
  float sigma = hist->GetRMS();
  //int   nbinsX   = hist->GetNbinsX();
  float nentries = hist->GetEntries();
  float meanerr = sigma / TMath::Sqrt(nentries);
  float sigmaerr = TMath::Sqrt(sigma * sigma * TMath::Sqrt(2 / nentries));

  float lowBound = hist->GetXaxis()->GetBinLowEdge(1);
  float highBound = hist->GetXaxis()->GetBinLowEdge(hist->GetNbinsX() + 1);

  if (TMath::IsNaN(mean) || TMath::IsNaN(sigma)) {
    mean = 0;
    sigma = -lowBound + highBound;
  }

  TF1 func("tmp", "gaus", mean - 1. * sigma, mean + 1. * sigma);
  if (0 == hist->Fit(&func, "QNR")) {  // N: do not blow up file by storing fit!
    mean = func.GetParameter(1);
    sigma = func.GetParameter(2);
  }

  // first round
  TF1 *doubleCB = new TF1("myDoubleCB", DoubleSidedCB, lowBound, highBound, 7);
  doubleCB->SetParameters(mean, sigma, 1.5, 1.5, 2.5, 2.5, 100);
  doubleCB->SetParLimits(0, mean - meanerr, mean + meanerr);
  doubleCB->SetParLimits(1, 0., sigma + 2 * sigmaerr);
  doubleCB->SetParLimits(2, 0., 30.);
  doubleCB->SetParLimits(3, 0., 30.);
  doubleCB->SetParLimits(4, 0., 50.);
  doubleCB->SetParLimits(5, 0., 50.);
  doubleCB->SetParLimits(6, 0., 100 * nentries);

  doubleCB->SetParNames("#mu", "#sigma", "#alpha_{L}", "#alpha_{R}", "n_{L}", "n_{R}", "N");
  doubleCB->SetLineColor(kRed);
  doubleCB->SetNpx(1000);
  // doubleCB->SetRange(0.8*lowBound,0.8*highBound);

  hist->Fit(doubleCB, "QM");

  // second round

  float p0 = doubleCB->GetParameter(0);
  float p1 = doubleCB->GetParameter(1);
  float p2 = doubleCB->GetParameter(2);
  float p3 = doubleCB->GetParameter(3);
  float p4 = doubleCB->GetParameter(4);
  float p5 = doubleCB->GetParameter(5);
  float p6 = doubleCB->GetParameter(6);

  float p0err = doubleCB->GetParError(0);
  float p1err = doubleCB->GetParError(1);
  float p2err = doubleCB->GetParError(2);
  float p3err = doubleCB->GetParError(3);
  float p4err = doubleCB->GetParError(4);
  float p5err = doubleCB->GetParError(5);
  float p6err = doubleCB->GetParError(6);

  if ((doubleCB->GetChisquare() / doubleCB->GetNDF()) > 5) {
    std::cout << "------------------------" << std::endl;
    std::cout << "chi2 1st:" << doubleCB->GetChisquare() << std::endl;

    //std::cout<<"p0: "<<p0<<"+/-"<<p0err<<std::endl;
    //std::cout<<"p1: "<<p1<<"+/-"<<p1err<<std::endl;
    //std::cout<<"p2: "<<p2<<"+/-"<<p2err<<std::endl;
    //std::cout<<"p3: "<<p3<<"+/-"<<p3err<<std::endl;
    //std::cout<<"p4: "<<p4<<"+/-"<<p4err<<std::endl;
    //std::cout<<"p5: "<<p5<<"+/-"<<p5err<<std::endl;
    //std::cout<<"p6: "<<p6<<"+/-"<<p6err<<std::endl;

    doubleCB->SetParameters(p0, p1, 3, 3, 6, 6, p6);
    doubleCB->SetParLimits(0, p0 - 2 * p0err, p0 + 2 * p0err);
    doubleCB->SetParLimits(1, p1 - 2 * p1err, p0 + 2 * p1err);
    doubleCB->SetParLimits(2, p2 - 2 * p2err, p0 + 2 * p2err);
    doubleCB->SetParLimits(3, p3 - 2 * p3err, p0 + 2 * p3err);
    doubleCB->SetParLimits(4, p4 - 2 * p4err, p0 + 2 * p4err);
    doubleCB->SetParLimits(5, p5 - 2 * p5err, p0 + 2 * p5err);
    doubleCB->SetParLimits(6, p6 - 2 * p6err, p0 + 2 * p6err);

    hist->Fit(doubleCB, "MQ");

    //gMinuit->Command("SCAn 1");
    //TGraph *gr = (TGraph*)gMinuit->GetPlot();
    //gr->SetMarkerStyle(21);
    //gr->Draw("alp");

    std::cout << "chi2 2nd:" << doubleCB->GetChisquare() << std::endl;
  }

  float res_mean = doubleCB->GetParameter(0);
  float res_width = doubleCB->GetParameter(1);

  float res_mean_err = doubleCB->GetParError(0);
  float res_width_err = doubleCB->GetParError(1);

  params::measurement resultM;
  params::measurement resultW;

  resultM = std::make_pair(res_mean, res_mean_err);
  resultW = std::make_pair(res_width, res_width_err);

  std::pair<params::measurement, params::measurement> result;

  result = std::make_pair(resultM, resultW);
  return result;
}

//*************************************************************
void FillTrendPlot(TH1F *trendPlot, TH1F *residualsPlot[100], params::estimator fitPar_, TString var_, Int_t myBins_)
//*************************************************************
{
  //std::cout<<"trendPlot name: "<<trendPlot->GetName()<<std::endl;

  // float phiInterval = (360.)/myBins_;
  float phiInterval = (2 * TMath::Pi() / myBins_);
  float etaInterval = 5. / myBins_;

  for (int i = 0; i < myBins_; ++i) {
    //int binn = i+1;

    char phipositionString[129];
    // float phiposition = (-180+i*phiInterval)+(phiInterval/2);
    float phiposition = (-TMath::Pi() + i * phiInterval) + (phiInterval / 2);
    sprintf(phipositionString, "%.1f", phiposition);

    char etapositionString[129];
    float etaposition = (-etaRange + i * etaInterval) + (etaInterval / 2);
    sprintf(etapositionString, "%.1f", etaposition);

    std::pair<params::measurement, params::measurement> myFit =
        std::make_pair(std::make_pair(0., 0.), std::make_pair(0., 0.));

    if (((TString)trendPlot->GetName()).Contains("Norm")) {
      myFit = fitResiduals(residualsPlot[i]);
    } else {
      // myFit = fitStudentTResiduals(residualsPlot[i]);
      myFit = fitResiduals(residualsPlot[i]);

      /*
	if(TString(residualsPlot[i]->GetName()).Contains("dx") ||
	TString(residualsPlot[i]->GetName()).Contains("dy")) {
	std::cout<<residualsPlot[i]->GetName() << " " << myFit.first.first << "+/- " << myFit.first.second  << std::endl; 
	}
      */
    }

    switch (fitPar_) {
      case params::MEAN: {
        float mean_ = myFit.first.first;
        float meanErr_ = myFit.first.second;
        trendPlot->SetBinContent(i + 1, mean_);
        trendPlot->SetBinError(i + 1, meanErr_);
        break;
      }
      case params::WIDTH: {
        float width_ = myFit.second.first;
        float widthErr_ = myFit.second.second;
        trendPlot->SetBinContent(i + 1, width_);
        trendPlot->SetBinError(i + 1, widthErr_);
        break;
      }
      case params::MEDIAN: {
        float median_ = getMedian(residualsPlot[i]).first;
        float medianErr_ = getMedian(residualsPlot[i]).second;
        trendPlot->SetBinContent(i + 1, median_);
        trendPlot->SetBinError(i + 1, medianErr_);
        break;
      }
      case params::MAD: {
        float mad_ = getMAD(residualsPlot[i]).first;
        float madErr_ = getMAD(residualsPlot[i]).second;
        trendPlot->SetBinContent(i + 1, mad_);
        trendPlot->SetBinError(i + 1, madErr_);
        break;
      }
      default:
        std::cout << "PrimaryVertexValidation::FillTrendPlot() " << fitPar_ << " unknown estimator!" << std::endl;
        break;
    }
  }

  //trendPlot->GetXaxis()->LabelsOption("h");

  if (fitPar_ == params::MEAN || fitPar_ == params::MEDIAN) {
    TString res;
    if (TString(residualsPlot[0]->GetName()).Contains("dxy"))
      res = "dxy";
    else if (TString(residualsPlot[0]->GetName()).Contains("dx"))
      res = "dx";
    else if (TString(residualsPlot[0]->GetName()).Contains("dy"))
      res = "dy";
    else if (TString(residualsPlot[0]->GetName()).Contains("dz"))
      res = "dz";
    else if (TString(residualsPlot[0]->GetName()).Contains("IP2D"))
      res = "IP2D";
    else if (TString(residualsPlot[0]->GetName()).Contains("resz"))
      res = "resz";

    TCanvas *fitOutput = new TCanvas(Form("fitOutput_%s_%s_%s", res.Data(), var_.Data(), trendPlot->GetName()),
                                     Form("fitOutput_%s_%s", res.Data(), var_.Data()),
                                     1200,
                                     1200);
    fitOutput->Divide(5, 5);

    TCanvas *fitPulls = new TCanvas(Form("fitPulls_%s_%s_%s", res.Data(), var_.Data(), trendPlot->GetName()),
                                    Form("fitPulls_%s_%s", res.Data(), var_.Data()),
                                    1200,
                                    1200);
    fitPulls->Divide(5, 5);

    TH1F *residualsPull[myBins_];

    for (Int_t i = 0; i < myBins_; i++) {
      TF1 *tmp1 = (TF1 *)residualsPlot[i]->GetListOfFunctions()->FindObject("tmp");
      if (tmp1 && residualsPlot[i]->GetEntries() > 0. && residualsPlot[i]->GetMinimum() > 0.) {
        fitOutput->cd(i + 1)->SetLogy();
      }
      fitOutput->cd(i + 1)->SetBottomMargin(0.16);
      //fitOutput->cd(i+1)->SetTopMargin(0.05);
      //residualsPlot[i]->Sumw2();
      MakeNicePlotStyle(residualsPlot[i]);
      residualsPlot[i]->SetMarkerStyle(20);
      residualsPlot[i]->SetMarkerSize(1.);
      residualsPlot[i]->SetStats(false);
      //residualsPlot[i]->GetXaxis()->SetRangeUser(-3*(tmp1->GetParameter(1)),3*(tmp1->GetParameter(1)));
      residualsPlot[i]->Draw("e1");
      residualsPlot[i]->GetYaxis()->UnZoom();

      //std::cout<<"*********************"<<std::endl;
      //std::cout<<"fitOutput->cd("<<i+1<<")"<<std::endl;
      //std::cout<<"residualsPlot["<<i<<"]->GetTitle() = "<<residualsPlot[i]->GetTitle()<<std::endl;

      // -- for chi2 ----
      TPaveText *pt = new TPaveText(0.13, 0.78, 0.33, 0.88, "NDC");
      pt->SetFillColor(10);
      pt->SetTextColor(1);
      pt->SetTextSize(0.07);
      pt->SetTextFont(42);
      pt->SetTextAlign(22);

      //TF1 *tmp1 = (TF1*)residualsPlot[i]->GetListOfFunctions()->FindObject("tmp");
      TString MYOUT;
      if (tmp1) {
        MYOUT = Form("#chi^{2}/ndf=%.1f", tmp1->GetChisquare() / tmp1->GetNDF());
      } else {
        MYOUT = "!! no plot !!";
      }

      TText *text1 = pt->AddText(MYOUT);
      text1->SetTextFont(72);
      text1->SetTextColor(kBlue);
      pt->Draw("same");

      // -- for bins --

      TPaveText *title = new TPaveText(0.1, 0.93, 0.8, 0.95, "NDC");
      title->SetFillColor(10);
      title->SetTextColor(1);
      title->SetTextSize(0.07);
      title->SetTextFont(42);
      title->SetTextAlign(22);

      //TText *text2 = title->AddText(residualsPlot[i]->GetTitle());
      //text2->SetTextFont(72);
      //text2->SetTextColor(kBlue);

      title->Draw("same");

      fitPulls->cd(i + 1);
      fitPulls->cd(i + 1)->SetBottomMargin(0.15);
      fitPulls->cd(i + 1)->SetLeftMargin(0.15);
      fitPulls->cd(i + 1)->SetRightMargin(0.05);

      residualsPull[i] = (TH1F *)residualsPlot[i]->Clone(Form("pull_%s", residualsPlot[i]->GetName()));
      for (Int_t nbin = 1; nbin <= residualsPull[i]->GetNbinsX(); nbin++) {
        if (residualsPlot[i]->GetBinContent(nbin) != 0 && tmp1) {
          residualsPull[i]->SetBinContent(
              nbin,
              (residualsPlot[i]->GetBinContent(nbin) - tmp1->Eval(residualsPlot[i]->GetBinCenter(nbin))) /
                  residualsPlot[i]->GetBinContent(nbin));
          residualsPull[i]->SetBinError(nbin, 0.1);
        }
      }

      TF1 *toDel = (TF1 *)residualsPull[i]->FindObject("tmp");
      if (toDel)
        residualsPull[i]->GetListOfFunctions()->Remove(toDel);
      residualsPull[i]->SetMarkerStyle(20);
      residualsPull[i]->SetMarkerSize(1.);
      residualsPull[i]->SetStats(false);

      residualsPull[i]->GetYaxis()->SetTitle("(res-fit)/res");
      // residualsPull[i]->SetOptTitle(1);
      residualsPull[i]->GetXaxis()->SetLabelFont(42);
      residualsPull[i]->GetYaxis()->SetLabelFont(42);
      residualsPull[i]->GetYaxis()->SetLabelSize(.07);
      residualsPull[i]->GetXaxis()->SetLabelSize(.07);
      residualsPull[i]->GetYaxis()->SetTitleSize(.07);
      residualsPull[i]->GetXaxis()->SetTitleSize(.07);
      residualsPull[i]->GetXaxis()->SetTitleOffset(0.9);
      residualsPull[i]->GetYaxis()->SetTitleOffset(1.2);
      residualsPull[i]->GetXaxis()->SetTitleFont(42);
      residualsPull[i]->GetYaxis()->SetTitleFont(42);

      residualsPull[i]->Draw("e1");
      residualsPull[i]->GetYaxis()->UnZoom();
    }

    TString tpName = trendPlot->GetName();

    TString FitNameToSame = Form("fitOutput_%s", (tpName.ReplaceAll("means_", "").Data()));
    //fitOutput->SaveAs(FitNameToSame+".pdf");
    //fitOutput->SaveAs(FitNameToSame+".png");
    TString PullNameToSave = Form("fitPulls_%s", (tpName.ReplaceAll("means_", "").Data()));
    //fitPulls->SaveAs(PullNameToSave+".pdf");
    //fitPulls->SaveAs(PullNameToSave+".png");

    if (isDebugMode) {
      fitOutput->SaveAs(Form("fitOutput_%s_%s_%s.pdf", res.Data(), var_.Data(), trendPlot->GetName()));
      fitOutput->SaveAs(Form("fitOutput_%s.pdf", (((TString)trendPlot->GetName()).ReplaceAll("means_", "")).Data()));
      fitPulls->SaveAs(Form("fitPulls_%s.pdf", (((TString)trendPlot->GetName()).ReplaceAll("means_", "")).Data()));
      fitOutput->SaveAs(Form("fitOutput_%s.png", (((TString)trendPlot->GetName()).ReplaceAll("means_", "")).Data()));
    }

    delete fitOutput;
    delete fitPulls;
  }
}

//*************************************************************
void FillMap_old(TH2F *trendMap, TH1F *residualsMapPlot[48][48], params::estimator fitPar_)
//*************************************************************
{
  float phiInterval = (360.) / nBins_;
  float etaInterval = (etaRange * 2.0) / nBins_;

  switch (fitPar_) {
    case params::MEAN: {
      for (int i = 0; i < nBins_; ++i) {
        char phipositionString[129];
        float phiposition = (-180 + i * phiInterval) + (phiInterval / 2);
        sprintf(phipositionString, "%.f", phiposition);
        trendMap->GetYaxis()->SetBinLabel(i + 1, phipositionString);

        for (int j = 0; j < nBins_; ++j) {
          char etapositionString[129];
          float etaposition = (-etaRange + j * etaInterval) + (etaInterval / 2);
          sprintf(etapositionString, "%.1f", etaposition);

          if (i == 0) {
            trendMap->GetXaxis()->SetBinLabel(j + 1, etapositionString);
          }

          std::pair<params::measurement, params::measurement> myFit =
              std::make_pair(std::make_pair(0., 0.), std::make_pair(0., 0.));

          myFit = fitResiduals(residualsMapPlot[i][j], true);

          float mean_ = myFit.first.first;
          float meanErr_ = myFit.first.second;

          trendMap->SetBinContent(j + 1, i + 1, mean_);
          trendMap->SetBinError(j + 1, i + 1, meanErr_);
        }
      }

      break;
    }

    case params::WIDTH: {
      for (int i = 0; i < nBins_; ++i) {
        char phipositionString[129];
        float phiposition = (-180 + i * phiInterval) + (phiInterval / 2);
        sprintf(phipositionString, "%.f", phiposition);
        trendMap->GetYaxis()->SetBinLabel(i + 1, phipositionString);

        for (int j = 0; j < nBins_; ++j) {
          char etapositionString[129];
          float etaposition = (-etaRange + j * etaInterval) + (etaInterval / 2);
          sprintf(etapositionString, "%.1f", etaposition);

          if (i == 0) {
            trendMap->GetXaxis()->SetBinLabel(j + 1, etapositionString);
          }

          std::pair<params::measurement, params::measurement> myFit =
              std::make_pair(std::make_pair(0., 0.), std::make_pair(0., 0.));
          myFit = fitResiduals(residualsMapPlot[i][j], true);

          float width_ = myFit.second.first;
          float widthErr_ = myFit.second.second;
          trendMap->SetBinContent(j + 1, i + 1, width_);
          trendMap->SetBinError(j + 1, i + 1, widthErr_);
        }
      }
      break;
    }
    case params::MEDIAN: {
      for (int i = 0; i < nBins_; ++i) {
        char phipositionString[129];
        float phiposition = (-180 + i * phiInterval) + (phiInterval / 2);
        sprintf(phipositionString, "%.f", phiposition);
        trendMap->GetYaxis()->SetBinLabel(i + 1, phipositionString);

        for (int j = 0; j < nBins_; ++j) {
          char etapositionString[129];
          float etaposition = (-etaRange + j * etaInterval) + (etaInterval / 2);
          sprintf(etapositionString, "%.1f", etaposition);

          if (i == 0) {
            trendMap->GetXaxis()->SetBinLabel(j + 1, etapositionString);
          }

          float median_ = getMedian(residualsMapPlot[i][j]).first;
          float medianErr_ = getMedian(residualsMapPlot[i][j]).second;
          trendMap->SetBinContent(j + 1, i + 1, median_);
          trendMap->SetBinError(j + 1, i + 1, medianErr_);
        }
      }
      break;
    }
    case params::MAD: {
      for (int i = 0; i < nBins_; ++i) {
        char phipositionString[129];
        float phiposition = (-180 + i * phiInterval) + (phiInterval / 2);
        sprintf(phipositionString, "%.f", phiposition);
        trendMap->GetYaxis()->SetBinLabel(i + 1, phipositionString);

        for (int j = 0; j < nBins_; ++j) {
          char etapositionString[129];
          float etaposition = (-etaRange + j * etaInterval) + (etaInterval / 2);
          sprintf(etapositionString, "%.1f", etaposition);

          if (i == 0) {
            trendMap->GetXaxis()->SetBinLabel(j + 1, etapositionString);
          }

          float mad_ = getMAD(residualsMapPlot[i][j]).first;
          float madErr_ = getMAD(residualsMapPlot[i][j]).second;
          trendMap->SetBinContent(j + 1, i + 1, mad_);
          trendMap->SetBinError(j + 1, i + 1, madErr_);
        }
      }
      break;
    }
    default:
      std::cout << "FitPVResiduals::FillMap() " << fitPar_ << " unknown estimator!" << std::endl;
      break;
  }
}

//*************************************************************
void FillMap(TH2F *trendMap,
             std::vector<std::vector<TH1F *> > residualsMapPlot,
             params::estimator fitPar_,
             const int nBinsX,
             const int nBinsY)
//*************************************************************
{
  float phiInterval = (360.) / nBinsY;
  float etaInterval = 5. / nBinsX;

  for (int i = 0; i < nBinsY; ++i) {
    char phipositionString[129];
    float phiposition = (-180 + i * phiInterval) + (phiInterval / 2);
    sprintf(phipositionString, "%.f", phiposition);

    trendMap->GetYaxis()->SetBinLabel(i + 1, phipositionString);

    for (int j = 0; j < nBinsX; ++j) {
      //std::cout<<"(i,j)="<<i<<","<<j<<std::endl;

      char etapositionString[129];
      float etaposition = (-etaRange + j * etaInterval) + (etaInterval / 2);
      sprintf(etapositionString, "%.1f", etaposition);

      if (i == 0) {
        trendMap->GetXaxis()->SetBinLabel(j + 1, etapositionString);
      }

      std::pair<params::measurement, params::measurement> myFit =
          std::make_pair(std::make_pair(0., 0.), std::make_pair(0., 0.));

      myFit = fitResiduals(residualsMapPlot[i][j], true);

      // check if plot is normalized
      bool isNormalized = false;
      if (((TString)trendMap->GetName()).Contains("Norm"))
        isNormalized = true;

      switch (fitPar_) {
        case params::MEAN: {
          Double_t mean_ = myFit.first.first;

          // do not allow crazy values
          if (!isNormalized)
            mean_ = (mean_ > 0.) ? std::min(mean_, 100.) : std::max(mean_, -100.);
          else
            mean_ = (mean_ > 0.) ? std::min(mean_, 2.) : std::max(mean_, -2.);

          float meanErr_ = myFit.first.second;
          //std::cout<<"bin i: "<<i<<" bin j: "<<j<<" mean: "<<mean_<<"+/-"<<meanErr_<<endl;
          trendMap->SetBinContent(j + 1, i + 1, mean_);
          trendMap->SetBinError(j + 1, i + 1, meanErr_);
          break;
        }
        case params::WIDTH: {
          Double_t width_ = myFit.second.first;

          // do not allow crazy values
          if (!isNormalized)
            width_ = std::min(width_, 1500.);
          else
            width_ = std::min(width_, 3.);

          float widthErr_ = myFit.second.second;
          trendMap->SetBinContent(j + 1, i + 1, width_);
          trendMap->SetBinError(j + 1, i + 1, widthErr_);
          break;
          //std::cout<<"bin i: "<<i<<" bin j: "<<j<<" width: "<<width_<<"+/-"<<widthErr_<<endl;
        }
        case params::MEDIAN: {
          float median_ = getMedian(residualsMapPlot[i][j]).first;
          float medianErr_ = getMedian(residualsMapPlot[i][j]).second;
          trendMap->SetBinContent(j + 1, i + 1, median_);
          trendMap->SetBinError(j + 1, i + 1, medianErr_);
          break;
        }
        case params::MAD: {
          float mad_ = getMAD(residualsMapPlot[i][j]).first;
          float madErr_ = getMAD(residualsMapPlot[i][j]).second;
          trendMap->SetBinContent(j + 1, i + 1, mad_);
          trendMap->SetBinError(j + 1, i + 1, madErr_);
          break;
        }
        default:
          std::cout << "FitPVResiduals::FillMap() " << fitPar_ << " unknown estimator!" << std::endl;
          break;
      }  // closes the switch statement
    }    // closes loop on eta bins
  }      // cloeses loop on phi bins
}

/*--------------------------------------------------------------------*/
void MakeNiceTrendPlotStyle(TH1 *hist, Int_t color, Int_t style)
/*--------------------------------------------------------------------*/
{
  hist->SetStats(kFALSE);
  hist->SetLineWidth(2);
  hist->GetXaxis()->CenterTitle(true);
  hist->GetYaxis()->CenterTitle(true);
  hist->GetXaxis()->SetTitleFont(42);
  hist->GetYaxis()->SetTitleFont(42);
  hist->GetXaxis()->SetTitleSize(0.065);
  hist->GetYaxis()->SetTitleSize(0.065);
  hist->GetXaxis()->SetTitleOffset(1.0);
  hist->GetYaxis()->SetTitleOffset(1.2);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelSize(.05);
  hist->GetXaxis()->SetLabelSize(.07);
  //hist->GetXaxis()->SetNdivisions(505);
  if (color != 8) {
    hist->SetMarkerSize(1.0);
  } else {
    hist->SetLineWidth(3);
    hist->SetMarkerSize(0.0);
  }
  hist->SetMarkerStyle(style);
  hist->SetLineColor(color);
  hist->SetMarkerColor(color);
}

/*--------------------------------------------------------------------*/
void MakeNicePlotStyle(TH1 *hist)
/*--------------------------------------------------------------------*/
{
  hist->SetStats(kFALSE);
  hist->SetLineWidth(2);
  hist->GetXaxis()->SetNdivisions(505);
  hist->GetXaxis()->CenterTitle(true);
  hist->GetYaxis()->CenterTitle(true);
  hist->GetXaxis()->SetTitleFont(42);
  hist->GetYaxis()->SetTitleFont(42);
  hist->GetXaxis()->SetTitleSize(0.07);
  hist->GetYaxis()->SetTitleSize(0.07);
  hist->GetXaxis()->SetTitleOffset(0.9);
  hist->GetYaxis()->SetTitleOffset(1.3);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelSize(.07);
  hist->GetXaxis()->SetLabelSize(.07);
}

/*--------------------------------------------------------------------*/
void MakeNiceMapStyle(TH2 *hist)
/*--------------------------------------------------------------------*/
{
  hist->SetStats(kFALSE);
  hist->GetXaxis()->CenterTitle(true);
  hist->GetYaxis()->CenterTitle(true);
  hist->GetZaxis()->CenterTitle(true);
  hist->GetXaxis()->SetTitleFont(42);
  hist->GetYaxis()->SetTitleFont(42);
  hist->GetXaxis()->LabelsOption("v");
  hist->GetZaxis()->SetTitleFont(42);
  hist->GetXaxis()->SetTitleSize(0.06);
  hist->GetYaxis()->SetTitleSize(0.06);
  hist->GetZaxis()->SetTitleSize(0.06);
  hist->GetXaxis()->SetTitleOffset(1.1);
  hist->GetZaxis()->SetTitleOffset(1.1);
  hist->GetYaxis()->SetTitleOffset(1.0);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  hist->GetZaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelSize(.05);
  hist->GetXaxis()->SetLabelSize(.05);
  hist->GetZaxis()->SetLabelSize(.05);
}

/*--------------------------------------------------------------------*/
std::pair<TH2F *, TH2F *> trimTheMap(TH2 *hist) {
  /*--------------------------------------------------------------------*/

  Int_t nXCells = hist->GetNbinsX();
  Int_t nYCells = hist->GetNbinsY();
  Int_t nCells = nXCells * nYCells;

  Double_t min = 9999.;
  Double_t max = -9999.;

  for (Int_t nX = 1; nX <= nXCells; nX++) {
    for (Int_t nY = 1; nY <= nYCells; nY++) {
      Double_t binContent = hist->GetBinContent(nX, nY);
      if (binContent > max)
        max = binContent;
      if (binContent < min)
        min = binContent;
    }
  }

  TH1F *histContentByCell =
      new TH1F(Form("histContentByCell_%s", hist->GetName()), "histContentByCell", nCells, min, max);

  for (Int_t nX = 1; nX <= nXCells; nX++) {
    for (Int_t nY = 1; nY <= nYCells; nY++) {
      histContentByCell->Fill(hist->GetBinContent(nX, nY));
    }
  }

  Double_t theMeanOfCells = histContentByCell->GetMean();
  Double_t theRMSOfCells = histContentByCell->GetRMS();
  params::measurement theMAD = getMAD(histContentByCell);

  if (isDebugMode) {
    std::cout << std::setw(24) << std::left << hist->GetName() << "| mean: " << std::setw(10) << std::setprecision(4)
              << theMeanOfCells << "| min: " << std::setw(10) << std::setprecision(4) << min
              << "| max: " << std::setw(10) << std::setprecision(4) << max << "| rms: " << std::setw(10)
              << std::setprecision(4) << theRMSOfCells << "| mad: " << std::setw(10) << std::setprecision(4)
              << theMAD.first << std::endl;
  }

  TCanvas *cCheck = new TCanvas(Form("cCheck_%s", hist->GetName()), Form("cCheck_%s", hist->GetName()), 1200, 1000);

  cCheck->Divide(2, 2);
  for (Int_t i = 1; i <= 4; i++) {
    cCheck->cd(i)->SetBottomMargin(0.13);
    cCheck->cd(i)->SetLeftMargin(0.12);
    if (i % 2 == 1)
      cCheck->cd(i)->SetRightMargin(0.19);
    else
      cCheck->cd(i)->SetRightMargin(0.07);
    cCheck->cd(i)->SetTopMargin(0.08);
  }

  cCheck->cd(1);
  hist->SetStats(kFALSE);
  hist->Draw("colz");
  //makeNewPairOfAxes(hist);

  cCheck->cd(2)->SetLogy();
  MakeNicePlotStyle(histContentByCell);
  histContentByCell->SetStats(kTRUE);
  histContentByCell->GetYaxis()->SetTitleOffset(0.9);
  histContentByCell->Draw();

  //Double_t theNewMin = theMeanOfCells-theRMSOfCells;
  //Double_t theNewMax = theMeanOfCells+theRMSOfCells;

  Double_t theNewMin = theMeanOfCells - theMAD.first * 3;
  Double_t theNewMax = theMeanOfCells + theMAD.first * 3;

  TArrow *l0 =
      new TArrow(theMeanOfCells, cCheck->GetUymin(), theMeanOfCells, histContentByCell->GetMaximum(), 0.3, "|>");
  l0->SetAngle(60);
  l0->SetLineColor(kRed);
  l0->SetLineWidth(4);
  l0->Draw("same");

  TArrow *l1 = new TArrow(theNewMin, cCheck->GetUymin(), theNewMin, histContentByCell->GetMaximum(), 0.3, "|>");
  l1->SetAngle(60);
  l1->SetLineColor(kBlue);
  l1->SetLineWidth(4);
  l1->Draw("same");

  TArrow *l2 = new TArrow(theNewMax, cCheck->GetUymin(), theNewMax, histContentByCell->GetMaximum(), 0.3, "|>");
  l2->SetAngle(60);
  l2->SetLineColor(kBlue);
  l2->SetLineWidth(4);
  l2->Draw("same");

  TH2F *histoTrimmed = new TH2F(Form("%s_trimmed", hist->GetName()),
                                Form("Trimmed %s;%s;%s;%s",
                                     hist->GetTitle(),
                                     hist->GetXaxis()->GetTitle(),
                                     hist->GetYaxis()->GetTitle(),
                                     hist->GetZaxis()->GetTitle()),
                                hist->GetNbinsX(),
                                hist->GetXaxis()->GetXmin(),
                                hist->GetXaxis()->GetXmax(),
                                hist->GetNbinsY(),
                                hist->GetYaxis()->GetXmin(),
                                hist->GetYaxis()->GetXmax());

  TH2F *histoMissed = new TH2F(Form("%s_Missed", hist->GetName()),
                               Form("Missed %s", hist->GetTitle()),
                               hist->GetNbinsX(),
                               hist->GetXaxis()->GetXmin(),
                               hist->GetXaxis()->GetXmax(),
                               hist->GetNbinsY(),
                               hist->GetYaxis()->GetXmin(),
                               hist->GetYaxis()->GetXmax());

  for (Int_t nX = 1; nX <= nXCells; nX++) {
    for (Int_t nY = 1; nY <= nYCells; nY++) {
      Double_t binContent = hist->GetBinContent(nX, nY);
      Double_t binError = hist->GetBinError(nX, nY);

      if (binContent == 0. && binError == 0.) {
        histoMissed->SetBinContent(nX, nY, 1);
      } else if (binContent <= theNewMin) {
        histoTrimmed->SetBinContent(nX, nY, theNewMin);
      } else if (binContent >= theNewMax) {
        histoTrimmed->SetBinContent(nX, nY, theNewMax);
      } else {
        histoTrimmed->SetBinContent(nX, nY, binContent);
      }
    }
  }

  cCheck->cd(3);
  histoTrimmed->SetStats(kFALSE);
  histoTrimmed->Draw("COLZ1");
  histoMissed->SetFillColor(kRed);
  gStyle->SetPaintTextFormat("0.1f");
  //histoMissed->SetMarkerSize(1.8);
  histoMissed->SetFillColor(kMagenta);
  histoMissed->SetMarkerColor(kMagenta);
  histoMissed->Draw("boxsame");
  //makeNewPairOfAxes(histoTrimmed);

  cCheck->cd(4);
  histoMissed->SetStats(kFALSE);
  histoMissed->Draw("box");
  makeNewPairOfAxes(histoMissed);

  if (isDebugMode) {
    cCheck->SaveAs(Form("cCheck_%s.png", hist->GetName()));
    cCheck->SaveAs(Form("cCheck_%s.pdf", hist->GetName()));

    std::cout << "histo:" << std::setw(25) << hist->GetName() << " old min: " << std::setw(10) << hist->GetMinimum()
              << " old max: " << std::setw(10) << hist->GetMaximum();
    std::cout << " | new min: " << std::setw(15) << hist->GetMinimum() << " new max: " << std::setw(10)
              << hist->GetMaximum() << std::endl;
  }

  delete histContentByCell;
  // hist->GetZaxis()->SetRangeUser(1.0001*theNewMin,0.999*theNewMax);
  // hist->SetMinimum(1.001*theNewMin);
  // hist->SetMaximum(0.999*theNewMax);
  delete cCheck;

  return std::make_pair(histoTrimmed, histoMissed);
}

/*--------------------------------------------------------------------*/
void setStyle(TString customCMSLabel, TString customRightLabel) {
  /*--------------------------------------------------------------------*/

  writeExtraText = true;  // if extra text
  writeExraLumi = false;  // if write sqrt(s) info
  if (customRightLabel != "") {
    lumi_13TeV = customRightLabel;
    lumi_13p6TeV = customRightLabel;
    lumi_0p9TeV = customRightLabel;
  } else {
    lumi_13TeV = "pp collisions";
    lumi_13p6TeV = "pp collisions";
    lumi_0p9TeV = "pp collisions";
  }
  if (customCMSLabel != "") {
    extraText = customCMSLabel;
  } else {
    extraText = "Internal";
  }

  TH1::StatOverflows(kTRUE);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat("e");
  //gStyle->SetPadTopMargin(0.05);
  //gStyle->SetPadBottomMargin(0.15);
  //gStyle->SetPadLeftMargin(0.17);
  //gStyle->SetPadRightMargin(0.02);
  gStyle->SetPadBorderMode(0);
  gStyle->SetTitleFillColor(10);
  gStyle->SetTitleFont(42);
  gStyle->SetTitleColor(1);
  gStyle->SetTitleTextColor(1);
  gStyle->SetTitleFontSize(0.06);
  gStyle->SetTitleBorderSize(0);
  gStyle->SetStatColor(kWhite);
  gStyle->SetStatFont(42);
  gStyle->SetStatFontSize(0.05);  ///---> gStyle->SetStatFontSize(0.025);
  gStyle->SetStatTextColor(1);
  gStyle->SetStatFormat("6.4g");
  gStyle->SetStatBorderSize(1);
  gStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  gStyle->SetPadTickY(1);
  gStyle->SetPadBorderMode(0);
  gStyle->SetOptFit(1);
  gStyle->SetNdivisions(510);

  // this is the standard palette
  const Int_t NRGBs = 5;
  const Int_t NCont = 255;

  Double_t stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
  Double_t red[NRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
  Double_t green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
  Double_t blue[NRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};
  TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
  gStyle->SetNumberContours(NCont);

  /*
  // try an alternative palette
  const Int_t NRGBs = 6;
  const Int_t NCont = 999;

  Double_t stops[NRGBs] = { 0.00, 0.1, 0.34, 0.61, 0.84, 1.00 };
  Double_t red[NRGBs]   = { 0.99, 0.0, 0.00, 0.87, 1.00, 0.51 };
  Double_t green[NRGBs] = { 0.00, 0.0, 0.81, 1.00, 0.20, 0.00 };
  Double_t blue[NRGBs]  = { 0.99, 0.0, 1.00, 0.12, 0.00, 0.00 };

  TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
  gStyle->SetNumberContours(NCont);

  */

  /*
  const Int_t NRGBs = 9;
  const Int_t NCont = 255;
 
  Double_t stops[NRGBs] = { 0.0000, 0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750, 1.0000};

  // dark body radiator
  // Double_t red[NRGBs]   = { 0./255., 45./255., 99./255., 156./255., 212./255., 230./255., 237./255., 234./255., 242./255.};
  // Double_t green[NRGBs] = { 0./255.,  0./255.,  0./255.,  45./255., 101./255., 168./255., 238./255., 238./255., 243./255.};
  // Double_t blue[NRGBs]  = { 0./255.,  1./255.,  1./255.,   3./255.,   9./255.,   8./255.,  11./255.,  95./255., 230./255.};
  
  // printable on grey
  //Double_t red[9]   = { 0./255.,   0./255.,   0./255.,  70./255., 148./255., 231./255., 235./255., 237./255., 244./255.};
  //Double_t green[9] = { 0./255.,   0./255.,   0./255.,   0./255.,   0./255.,  69./255.,  67./255., 216./255., 244./255.};
  //Double_t blue[9]  = { 0./255., 102./255., 228./255., 231./255., 177./255., 124./255., 137./255.,  20./255., 244./255.};

  // thermometer
  //Double_t red[9]   = {  34./255.,  70./255., 129./255., 187./255., 225./255., 226./255., 216./255., 193./255., 179./255.};
  //Double_t green[9] = {  48./255.,  91./255., 147./255., 194./255., 226./255., 229./255., 196./255., 110./255.,  12./255.};
  //Double_t blue[9]  = { 234./255., 212./255., 216./255., 224./255., 206./255., 110./255.,  53./255.,  40./255.,  29./255.};

  // visible spectrum
  Double_t red[9]   = { 18./255.,  72./255.,   5./255.,  23./255.,  29./255., 201./255., 200./255., 98./255., 29./255.};
  Double_t green[9] = {  0./255.,   0./255.,  43./255., 167./255., 211./255., 117./255.,   0./255.,  0./255.,  0./255.};
  Double_t blue[9]  = { 51./255., 203./255., 177./255.,  26./255.,  10./255.,   9./255.,   8./255.,  3./255.,  0./255.};

  TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
  gStyle->SetNumberContours(NCont);
  */
}

/*--------------------------------------------------------------------*/
TH1F *DrawZero(TH1F *hist, Int_t nbins, Double_t lowedge, Double_t highedge, Int_t iter)
/*--------------------------------------------------------------------*/
{
  TH1F *hzero = new TH1F(
      Form("hzero_%s_%i", hist->GetName(), iter), Form("hzero_%s_%i", hist->GetName(), iter), nbins, lowedge, highedge);
  for (Int_t i = 0; i < hzero->GetNbinsX(); i++) {
    hzero->SetBinContent(i, 0.);
    hzero->SetBinError(i, 0.);
  }
  hzero->SetLineWidth(2);
  hzero->SetLineStyle(9);
  hzero->SetLineColor(kMagenta);

  return hzero;
}

/*--------------------------------------------------------------------*/
TH1F *DrawConstant(TH1F *hist, Int_t nbins, Double_t lowedge, Double_t highedge, Int_t iter, Double_t theConst)
/*--------------------------------------------------------------------*/
{
  TH1F *hzero = new TH1F(Form("hconst_%s_%i", hist->GetName(), iter),
                         Form("hconst_%s_%i", hist->GetName(), iter),
                         nbins,
                         lowedge,
                         highedge);
  for (Int_t i = 0; i <= hzero->GetNbinsX(); i++) {
    hzero->SetBinContent(i, theConst);
    hzero->SetBinError(i, 0.);
  }
  hzero->SetLineWidth(2);
  hzero->SetLineStyle(9);
  hzero->SetLineColor(kMagenta);

  return hzero;
}

/*--------------------------------------------------------------------*/
void makeNewXAxis(TH1F *h)
/*--------------------------------------------------------------------*/
{
  TString myTitle = h->GetName();
  float axmin = -999;
  float axmax = 999.;
  int ndiv = 510;
  if (myTitle.Contains("eta")) {
    axmin = -etaRange;
    axmax = etaRange;
    ndiv = 505;
  } else if (myTitle.Contains("phi")) {
    axmin = -TMath::Pi();
    axmax = TMath::Pi();
    ndiv = 510;
  } else if (myTitle.Contains("pT")) {
    axmin = minPt_;
    axmax = maxPt_;
    ndiv = 510;
  } else if (myTitle.Contains("ladder")) {
    axmin = 0.5;
    axmax = nLadders_ + 0.5;
  } else if (myTitle.Contains("modZ")) {
    axmin = 0.5;
    axmax = nModZ_ + 0.5;
  } else if (myTitle.Contains("h_probe")) {
    ndiv = 505;
    axmin = h->GetXaxis()->GetBinCenter(h->GetXaxis()->GetFirst());
    axmax = h->GetXaxis()->GetBinCenter(h->GetXaxis()->GetLast());
  } else {
    std::cout << "unrecognized variable for histogram title: " << myTitle << std::endl;
  }

  // Remove the current axis
  h->GetXaxis()->SetLabelOffset(999);
  h->GetXaxis()->SetTickLength(0);

  // Redraw the new axis
  gPad->Update();

  TGaxis *newaxis =
      new TGaxis(gPad->GetUxmin(), gPad->GetUymin(), gPad->GetUxmax(), gPad->GetUymin(), axmin, axmax, ndiv, "SDH");

  TGaxis *newaxisup =
      new TGaxis(gPad->GetUxmin(), gPad->GetUymax(), gPad->GetUxmax(), gPad->GetUymax(), axmin, axmax, ndiv, "-SDH");

  newaxis->SetLabelOffset(0.02);
  newaxis->SetLabelFont(42);
  newaxis->SetLabelSize(0.05);

  newaxisup->SetLabelOffset(-0.02);
  newaxisup->SetLabelFont(42);
  newaxisup->SetLabelSize(0);

  newaxis->Draw();
  newaxisup->Draw();
}

/*--------------------------------------------------------------------*/
void makeNewPairOfAxes(TH2F *h)
/*--------------------------------------------------------------------*/
{
  TString myTitle = h->GetName();
  // fake defaults
  float axmin = -999;
  float axmax = 999.;
  float aymin = -999;
  float aymax = 999.;
  int ndivx = h->GetXaxis()->GetNdivisions();
  int ndivy = h->GetYaxis()->GetNdivisions();

  if (!myTitle.Contains("L1Map")) {
    ndivx = 505;
    ndivy = 510;
    axmin = -etaRange;
    axmax = etaRange;
    aymin = -TMath::Pi();
    aymax = TMath::Pi();
  } else {
    // this is a L1 map
    axmin = 0.5;
    axmax = nModZ_ + 0.5;
    aymin = 0.5;
    aymax = nLadders_ + 0.5;
  }

  // Remove the current axis
  h->GetXaxis()->SetLabelOffset(999);
  h->GetXaxis()->SetTickLength(0);

  h->GetYaxis()->SetLabelOffset(999);
  h->GetYaxis()->SetTickLength(0);

  // Redraw the new axis
  gPad->Update();

  TGaxis *newXaxis =
      new TGaxis(gPad->GetUxmin(), gPad->GetUymin(), gPad->GetUxmax(), gPad->GetUymin(), axmin, axmax, ndivx, "SDH");

  TGaxis *newXaxisup =
      new TGaxis(gPad->GetUxmin(), gPad->GetUymax(), gPad->GetUxmax(), gPad->GetUymax(), axmin, axmax, ndivx, "-SDH");

  TGaxis *newYaxisR =
      new TGaxis(gPad->GetUxmin(), gPad->GetUymin(), gPad->GetUxmin(), gPad->GetUymax(), aymin, aymax, ndivy, "SDH");

  TGaxis *newYaxisL =
      new TGaxis(gPad->GetUxmax(), gPad->GetUymin(), gPad->GetUxmax(), gPad->GetUymax(), aymin, aymax, ndivy, "-SDH");

  newXaxis->SetLabelOffset(0.02);
  newXaxis->SetLabelFont(42);
  newXaxis->SetLabelSize(0.055);

  newXaxisup->SetLabelOffset(-0.02);
  newXaxisup->SetLabelFont(42);
  newXaxisup->SetLabelSize(0);

  newXaxis->Draw();
  newXaxisup->Draw();

  newYaxisR->SetLabelOffset(0.02);
  newYaxisR->SetLabelFont(42);
  newYaxisR->SetLabelSize(0.055);

  newYaxisL->SetLabelOffset(-0.02);
  newYaxisL->SetLabelFont(42);
  newYaxisL->SetLabelSize(0);

  newYaxisR->Draw();
  newYaxisL->Draw();
}

/*--------------------------------------------------------------------*/
Double_t fDLine(Double_t *x, Double_t *par)
/*--------------------------------------------------------------------*/
{
  if (x[0] < _boundSx && x[0] > _boundDx) {
    TF1::RejectPoint();
    return 0;
  }
  return par[0];
}

/*--------------------------------------------------------------------*/
Double_t fULine(Double_t *x, Double_t *par)
/*--------------------------------------------------------------------*/
{
  if (x[0] >= _boundSx && x[0] <= _boundDx) {
    TF1::RejectPoint();
    return 0;
  }
  return par[0];
}

/*--------------------------------------------------------------------*/
void FitULine(TH1 *hist)
/*--------------------------------------------------------------------*/
{
  // define fitting function
  TF1 func1("lineUp", fULine, _boundMin, _boundMax, 1);
  //TF1 func1("lineUp","pol0",-0.5,11.5);

  if (0 == hist->Fit(&func1, "QR")) {
    if (hist->GetFunction(func1.GetName())) {  // Take care that it is later on drawn:
      hist->GetFunction(func1.GetName())->ResetBit(TF1::kNotDraw);
    }
    //std::cout<<"FitPVResiduals() fit Up done!"<<std::endl;
  }
}

/*--------------------------------------------------------------------*/
void FitDLine(TH1 *hist)
/*--------------------------------------------------------------------*/
{
  // define fitting function
  // TF1 func1("lineDown",fDLine,-0.5,11.5,1);

  TF1 func2("lineDown", "pol0", _boundSx, _boundDx);
  func2.SetRange(_boundSx, _boundDx);

  if (0 == hist->Fit(&func2, "QR")) {
    if (hist->GetFunction(func2.GetName())) {  // Take care that it is later on drawn:
      hist->GetFunction(func2.GetName())->ResetBit(TF1::kNotDraw);
    }
    // std::cout<<"FitPVResiduals() fit Down done!"<<std::endl;
  }
}

/*--------------------------------------------------------------------*/
void MakeNiceTF1Style(TF1 *f1, Int_t color)
/*--------------------------------------------------------------------*/
{
  f1->SetLineColor(color);
  f1->SetLineWidth(3);
  f1->SetLineStyle(2);
}

/*--------------------------------------------------------------------*/
params::measurement getTheRangeUser(TH1F *thePlot, Limits *lims, bool tag)
/*--------------------------------------------------------------------*/
{
  TString theTitle = thePlot->GetName();
  theTitle.ToLower();

  /*
    Double_t m_dxyPhiMax     = 40;
    Double_t m_dzPhiMax      = 40;
    Double_t m_dxyEtaMax     = 40;
    Double_t m_dzEtaMax      = 40;
    Double_t m_dxyPtMax      = 40;
    Double_t m_dzPtMax       = 40;
    
    Double_t m_dxyPhiNormMax = 0.5;
    Double_t m_dzPhiNormMax  = 0.5;
    Double_t m_dxyEtaNormMax = 0.5;
    Double_t m_dzEtaNormMax  = 0.5;
    Double_t m_dxyPtNormMax  = 0.5;
    Double_t m_dzPtNormMax   = 0.5;
    
    Double_t w_dxyPhiMax     = 150;
    Double_t w_dzPhiMax      = 150;
    Double_t w_dxyEtaMax     = 150;
    Double_t w_dzEtaMax      = 1000;
    Double_t w_dxyPtMax      = 150;
    Double_t w_dzPtMax       = 150;
    
    Double_t w_dxyPhiNormMax = 1.8;
    Double_t w_dzPhiNormMax  = 1.8;
    Double_t w_dxyEtaNormMax = 1.8;
    Double_t w_dzEtaNormMax  = 1.8;   
    Double_t w_dxyPtNormMax  = 1.8;
    Double_t w_dzPtNormMax   = 1.8;
  */

  params::measurement result;

  if (theTitle.Contains("norm")) {
    if (theTitle.Contains("means")) {
      if (theTitle.Contains("dxy") || theTitle.Contains("dx") || theTitle.Contains("dy")) {
        if (theTitle.Contains("phi") || theTitle.Contains("ladder")) {
          result = std::make_pair(-lims->get_dxyPhiNormMax().first, lims->get_dxyPhiNormMax().first);
        } else if (theTitle.Contains("eta") || theTitle.Contains("mod")) {
          result = std::make_pair(-lims->get_dxyEtaNormMax().first, lims->get_dxyEtaNormMax().first);
        } else if (theTitle.Contains("pt")) {
          result = std::make_pair(-lims->get_dxyPtNormMax().first, lims->get_dxyPtNormMax().first);
        } else {
          result = std::make_pair(-0.8, 0.8);
        }
      } else if (theTitle.Contains("dz")) {
        if (theTitle.Contains("phi") || theTitle.Contains("ladder")) {
          result = std::make_pair(-lims->get_dzPhiNormMax().first, lims->get_dzPhiNormMax().first);
        } else if (theTitle.Contains("eta") || theTitle.Contains("mod")) {
          result = std::make_pair(-lims->get_dzEtaNormMax().first, lims->get_dzEtaNormMax().first);
        } else if (theTitle.Contains("pt")) {
          result = std::make_pair(-lims->get_dzPtNormMax().first, lims->get_dzPtNormMax().first);
        } else {
          result = std::make_pair(-0.8, 0.8);
        }
      }
    } else if (theTitle.Contains("widths")) {
      if (theTitle.Contains("dxy") || theTitle.Contains("dx") || theTitle.Contains("dy")) {
        if (theTitle.Contains("phi") || theTitle.Contains("ladder")) {
          result = std::make_pair(0., lims->get_dxyPhiNormMax().second);
        } else if (theTitle.Contains("eta") || theTitle.Contains("mod")) {
          result = std::make_pair(0., lims->get_dxyEtaNormMax().second);
        } else if (theTitle.Contains("pt")) {
          result = std::make_pair(0., lims->get_dxyPtNormMax().second);
        } else {
          result = std::make_pair(0., 2.);
        }
      } else if (theTitle.Contains("dz")) {
        if (theTitle.Contains("phi") || theTitle.Contains("ladder")) {
          result = std::make_pair(0., lims->get_dzPhiNormMax().second);
        } else if (theTitle.Contains("eta") || theTitle.Contains("mod")) {
          result = std::make_pair(0., lims->get_dzEtaNormMax().second);
        } else if (theTitle.Contains("pt")) {
          result = std::make_pair(0., lims->get_dzPtNormMax().second);
        } else {
          result = std::make_pair(0., 2.);
        }
      }
    }
  } else {
    if (theTitle.Contains("means")) {
      if (theTitle.Contains("dxy") || theTitle.Contains("dx") || theTitle.Contains("dy")) {
        if (theTitle.Contains("phi") || theTitle.Contains("ladder")) {
          result = std::make_pair(-lims->get_dxyPhiMax().first, lims->get_dxyPhiMax().first);
        } else if (theTitle.Contains("eta") || theTitle.Contains("mod")) {
          result = std::make_pair(-lims->get_dxyEtaMax().first, lims->get_dxyEtaMax().first);
        } else if (theTitle.Contains("pt")) {
          result = std::make_pair(-lims->get_dxyPtMax().first, lims->get_dxyPtMax().first);
        } else {
          result = std::make_pair(-40., 40.);
        }
      } else if (theTitle.Contains("dz")) {
        if (theTitle.Contains("phi") || theTitle.Contains("ladder")) {
          result = std::make_pair(-lims->get_dzPhiMax().first, lims->get_dzPhiMax().first);
        } else if (theTitle.Contains("eta") || theTitle.Contains("mod")) {
          result = std::make_pair(-lims->get_dzEtaMax().first, lims->get_dzEtaMax().first);
        } else if (theTitle.Contains("pt")) {
          result = std::make_pair(-lims->get_dzPtMax().first, lims->get_dzPtMax().first);
        } else {
          result = std::make_pair(-80., 80.);
        }
      }
    } else if (theTitle.Contains("widths")) {
      if (theTitle.Contains("dxy") || theTitle.Contains("dx") || theTitle.Contains("dy")) {
        if (theTitle.Contains("phi") || theTitle.Contains("ladder")) {
          result = std::make_pair(0., lims->get_dxyPhiMax().second);
        } else if (theTitle.Contains("eta") || theTitle.Contains("mod")) {
          result = std::make_pair(0., lims->get_dxyEtaMax().second);
        } else if (theTitle.Contains("pt")) {
          result = std::make_pair(0., lims->get_dxyPtMax().second);
        } else {
          result = std::make_pair(0., 150.);
        }
      } else if (theTitle.Contains("dz")) {
        if (theTitle.Contains("phi") || theTitle.Contains("ladder")) {
          result = std::make_pair(0., lims->get_dzPhiMax().second);
        } else if (theTitle.Contains("eta") || theTitle.Contains("mod")) {
          result = std::make_pair(0., lims->get_dzEtaMax().second);
        } else if (theTitle.Contains("pt")) {
          result = std::make_pair(0., lims->get_dzPtMax().second);
        } else {
          result = std::make_pair(0., 300.);
        }
      }
    }
  }

  if (tag)
    std::cout << theTitle << " " << result.first << " " << result.second << std::endl;
  return result;
}
