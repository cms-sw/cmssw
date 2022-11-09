#include "TFile.h"
#include "TMath.h"
#include "TF1.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TKey.h"
#include "TDirectory.h"
#include "TCanvas.h"
#include "TObject.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TClass.h"
#include "TLegend.h"
#include "TObjString.h"
#include <string>
#include <iomanip>
#include "TPaveText.h"
#include <fstream>  // std::ofstream
#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <regex>
#include <unordered_map>
#define PLOTTING_MACRO  // to remove message logger
#include "Alignment/OfflineValidation/interface/PVValidationHelpers.h"
#include "Alignment/OfflineValidation/macros/CMS_lumi.h"

/* 
   Store here the globals binning settings
*/

float SUMPTMIN = 1.;
float SUMPTMAX = 1e3;
int TRACKBINS = 120;
int VTXBINS = 60;

/* 
   This is an auxilliary class to store the list of files
   to be used to plot
*/

class PVResolutionVariables {
public:
  PVResolutionVariables(TString fileName, TString baseDir, TString legName = "", int color = 1, int style = 1);
  int getLineColor() { return lineColor; }
  int getLineStyle() { return lineStyle; }
  TString getName() { return legendName; }
  TFile* getFile() { return file; }
  TString getFileName() { return fname; }

private:
  TFile* file;
  int lineColor;
  int lineStyle;
  TString legendName;
  TString fname;
};

PVResolutionVariables::PVResolutionVariables(
    TString fileName, TString baseDir, TString legName, int lColor, int lStyle) {
  fname = fileName;
  lineColor = lColor;
  lineStyle = lStyle % 100;
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
    std::cout << fileName.Data() << ", found base directory: " << baseDir.Data() << std::endl;
  } else {
    std::cout << fileName.Data() << ", no directory named: " << baseDir.Data() << std::endl;
    assert(false);
  }
}

namespace PVResolution {
  std::vector<PVResolutionVariables*> sourceList;

  // fill the list of files
  //*************************************************************
  void loadFileList(const char* inputFile, TString baseDir, TString legendName, int lineColor, int lineStyle)
  //*************************************************************
  {
    gErrorIgnoreLevel = kFatal;
    sourceList.push_back(new PVResolutionVariables(inputFile, baseDir, legendName, lineColor, lineStyle));
  }

  //*************************************************************
  void clearFileList()
  //*************************************************************
  {
    sourceList.clear();
  }
}  // namespace PVResolution

namespace statmode {
  using fitParams = std::pair<std::pair<double, double>, std::pair<double, double> >;
}

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

// auxilliary functions
void setPVResolStyle();
void fillTrendPlotByIndex(TH1F* trendPlot,
                          std::unordered_map<std::string, TH1F*>& h,
                          std::regex toMatch,
                          PVValHelper::estimator fitPar_);
statmode::fitParams fitResolutions(TH1* hist, bool singleTime = false);
void makeNiceTrendPlotStyle(TH1* hist, Int_t color, Int_t style);
void adjustMaximum(TH1F* histos[], int size);

// inline function
namespace pvresol {
  int check(const double a[], int n) {
    //while (--n > 0 && a[n] == a[0])    // exact match
    while (--n > 0 && (a[n] - a[0]) < 0.01)  // merged input files, protection agains numerical precision
      ;
    return n != 0;
  }
}  // namespace pvresol

// MAIN
//*************************************************************
void FitPVResolution(TString namesandlabels, TString theDate = "", bool isStrict = false) {
  //*************************************************************

  bool fromLoader = false;
  setPVResolStyle();

  // check if the loader is empty
  if (!PVResolution::sourceList.empty()) {
    fromLoader = true;
  }

  // if enters here, whatever is passed from command line is neglected
  if (fromLoader) {
    std::cout << "FitPVResiduals::FitPVResiduals(): file list specified from loader" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "!!    arguments passed from CLI will be neglected   !!" << std::endl;
    std::cout << "======================================================" << std::endl;
    for (std::vector<PVResolutionVariables*>::iterator it = PVResolution::sourceList.begin();
         it != PVResolution::sourceList.end();
         ++it) {
      std::cout << "name:  " << std::setw(20) << (*it)->getName() << " |file:  " << std::setw(15) << (*it)->getFile()
                << " |color: " << std::setw(5) << (*it)->getLineColor() << " |style: " << std::setw(5)
                << (*it)->getLineStyle() << std::endl;
    }
    std::cout << "======================================================" << std::endl;
  }

  Int_t theFileCount = 0;
  TList* FileList = new TList();
  TList* LabelList = new TList();

  if (!fromLoader) {
    namesandlabels.Remove(TString::kTrailing, ',');
    TObjArray* nameandlabelpairs = namesandlabels.Tokenize(",");
    for (Int_t i = 0; i < nameandlabelpairs->GetEntries(); ++i) {
      TObjArray* aFileLegPair = TString(nameandlabelpairs->At(i)->GetName()).Tokenize("=");

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
    for (std::vector<PVResolutionVariables*>::iterator it = PVResolution::sourceList.begin();
         it != PVResolution::sourceList.end();
         ++it) {
      //FileList->Add((*it)->getFile()); // was extremely slow
      FileList->Add(TFile::Open((*it)->getFileName(), "READ"));
    }
    theFileCount = PVResolution::sourceList.size();
  }

  const Int_t nFiles_ = theFileCount;
  TString LegLabels[10];
  TFile* fins[nFiles_];
  Int_t markers[9];
  Int_t colors[9];

  for (Int_t j = 0; j < nFiles_; j++) {
    // Retrieve files
    fins[j] = (TFile*)FileList->At(j);

    if (!fromLoader) {
      TObjString* legend = (TObjString*)LabelList->At(j);
      LegLabels[j] = legend->String();
      markers[j] = def_markers[j];
      colors[j] = def_colors[j];
    } else {
      LegLabels[j] = PVResolution::sourceList[j]->getName();
      markers[j] = PVResolution::sourceList[j]->getLineStyle();
      colors[j] = PVResolution::sourceList[j]->getLineColor();
    }

    LegLabels[j].ReplaceAll("_", " ");
    std::cout << "FitPVResolution::FitPVResolution(): label[" << j << "] " << LegLabels[j] << std::endl;
  }

  // get the binnings
  TH1F* theBinHistos[nFiles_];
  double theSumPtMin_[nFiles_];
  double theSumPtMax_[nFiles_];
  double theTrackBINS_[nFiles_];
  double theVtxBINS_[nFiles_];

  for (Int_t i = 0; i < nFiles_; i++) {
    fins[i]->cd("PrimaryVertexResolution/BinningFeatures/");
    if (gDirectory->GetListOfKeys()->Contains("h_profileBinnings")) {
      gDirectory->GetObject("h_profileBinnings", theBinHistos[i]);
      theSumPtMin_[i] =
          theBinHistos[i]->GetBinContent(1) / (theBinHistos[i]->GetEntries() / theBinHistos[i]->GetNbinsX());
      std::cout << "File n. " << i << " has theSumPtMin[" << i << "] = " << theSumPtMin_[i] << std::endl;
      theSumPtMax_[i] =
          theBinHistos[i]->GetBinContent(2) / (theBinHistos[i]->GetEntries() / theBinHistos[i]->GetNbinsX());
      std::cout << "File n. " << i << " has theSumPtMax[" << i << "] = " << theSumPtMax_[i] << std::endl;
      theTrackBINS_[i] =
          theBinHistos[i]->GetBinContent(3) / (theBinHistos[i]->GetEntries() / theBinHistos[i]->GetNbinsX());
      std::cout << "File n. " << i << " has theTrackBINS[" << i << "] = " << theTrackBINS_[i] << std::endl;
      theVtxBINS_[i] =
          theBinHistos[i]->GetBinContent(4) / (theBinHistos[i]->GetEntries() / theBinHistos[i]->GetNbinsX());
      std::cout << "File n. " << i << " has theVtxBINS[" << i << "] = " << theVtxBINS_[i] << std::endl;
    } else {
      theSumPtMin_[i] = 1.;
      std::cout << "File n. " << i << " getting the default minimum sum pT range: " << theSumPtMin_[i] << std::endl;
      theSumPtMax_[i] = 1e3;
      std::cout << "File n. " << i << " getting the default maxmum sum pT range: " << theSumPtMax_[i] << std::endl;
      theTrackBINS_[i] = 120.;
      std::cout << "File n. " << i << " getting the default number of tracks bins: " << theTrackBINS_[i] << std::endl;
      theTrackBINS_[i] = 60.;
      std::cout << "File n. " << i << " getting the default number of vertices bins: " << theVtxBINS_[i] << std::endl;
    }
  }

  // checks if all minimum sum pT ranges coincide
  // if not, exits
  if (pvresol::check(theSumPtMin_, nFiles_)) {
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResolution::FitPVResolution(): the minimum sum pT is different" << std::endl;
    std::cout << "exiting..." << std::endl;
    exit(EXIT_FAILURE);
  } else {
    SUMPTMIN = theSumPtMin_[0];
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResolution::FitPVResolution(): the minimum sum pT is: " << SUMPTMIN << std::endl;
    std::cout << "======================================================" << std::endl;
  }

  // checks if all maximum sum pT ranges coincide
  // if not, exits
  if (pvresol::check(theSumPtMax_, nFiles_)) {
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResolution::FitPVResolution(): the maximum sum pT is different" << std::endl;
    std::cout << "exiting..." << std::endl;
    exit(EXIT_FAILURE);
  } else {
    SUMPTMAX = theSumPtMax_[0];
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResolution::FitPVResolution(): the maximum sum pT is: " << SUMPTMAX << std::endl;
    std::cout << "======================================================" << std::endl;
  }

  // checks if all number of tracks bins coincide
  // if not, exits
  if (pvresol::check(theTrackBINS_, nFiles_)) {
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResolution::FitPVResolution(): the number of track bins is different" << std::endl;
    if (isStrict) {
      std::cout << "exiting..." << std::endl;
      std::cout << "======================================================" << std::endl;
      exit(EXIT_FAILURE);  //this is stricter
    } else {
      TRACKBINS = *std::max_element(theTrackBINS_, theTrackBINS_ + nFiles_);
      std::cout << "chosen the maximum: " << TRACKBINS << std::endl;
      std::cout << "======================================================" << std::endl;
    }
  } else {
    TRACKBINS = int(theTrackBINS_[0]);
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResolution::FitPVResolution(): the number of track bins is: " << TRACKBINS << std::endl;
    std::cout << "======================================================" << std::endl;
  }

  // checks if all number of vertices bins coincide
  // if not, exits
  if (pvresol::check(theVtxBINS_, nFiles_)) {
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResolution::FitPVResolution(): the number of vertices bins is different" << std::endl;
    if (isStrict) {
      std::cout << "exiting..." << std::endl;
      std::cout << "======================================================" << std::endl;
      exit(EXIT_FAILURE);  //this is stricter
    } else {
      VTXBINS = *std::max_element(theVtxBINS_, theVtxBINS_ + nFiles_);
      std::cout << "chosen the maximum: " << VTXBINS << std::endl;
      std::cout << "======================================================" << std::endl;
    }
  } else {
    VTXBINS = int(theVtxBINS_[0]);
    std::cout << "======================================================" << std::endl;
    std::cout << "FitPVResolution::FitPVResolution(): the number of vertices bins is: " << VTXBINS << std::endl;
    std::cout << "======================================================" << std::endl;
  }

  // max vertices
  const int max_n_vertices = std::min(60, VTXBINS);  // take the minimum to avoid overflow
  std::vector<float> myNVtx_bins_;
  for (float i = 0; i <= max_n_vertices; i++) {
    myNVtx_bins_.push_back(i - 0.5f);
  }

  // max track
  const int max_n_tracks = std::min(120, TRACKBINS);  // take the minimum to avoid overflow
  std::vector<float> myNTrack_bins_;
  for (float i = 0; i <= max_n_tracks; i++) {
    myNTrack_bins_.push_back(i - 0.5f);
  }

  // max sumPt
  const int max_sum_pt = 30;
  std::array<float, max_sum_pt + 1> mypT_bins_ = PVValHelper::makeLogBins<float, max_sum_pt>(SUMPTMIN, SUMPTMAX);

  // define the maps
  std::unordered_map<std::string, TH1F*> hpulls_;
  std::unordered_map<std::string, TH1F*> hdiffs_;

  // summary plots

  TH1F* p_resolX_vsSumPt_[nFiles_];
  TH1F* p_resolY_vsSumPt_[nFiles_];
  TH1F* p_resolZ_vsSumPt_[nFiles_];

  TH1F* p_resolX_vsNtracks_[nFiles_];
  TH1F* p_resolY_vsNtracks_[nFiles_];
  TH1F* p_resolZ_vsNtracks_[nFiles_];

  TH1F* p_resolX_vsNVtx_[nFiles_];
  TH1F* p_resolY_vsNVtx_[nFiles_];
  TH1F* p_resolZ_vsNVtx_[nFiles_];

  TH1F* p_pullX_vsSumPt_[nFiles_];
  TH1F* p_pullY_vsSumPt_[nFiles_];
  TH1F* p_pullZ_vsSumPt_[nFiles_];

  TH1F* p_pullX_vsNtracks_[nFiles_];
  TH1F* p_pullY_vsNtracks_[nFiles_];
  TH1F* p_pullZ_vsNtracks_[nFiles_];

  TH1F* p_pullX_vsNVtx_[nFiles_];
  TH1F* p_pullY_vsNVtx_[nFiles_];
  TH1F* p_pullZ_vsNVtx_[nFiles_];

  // load in the map all the relevant histograms

  for (Int_t j = 0; j < nFiles_; j++) {
    // vs n. tracks

    p_pullX_vsNtracks_[j] = new TH1F(Form("p_pullX_vsNtracks_%i", j),
                                     "x-pull vs n_{tracks};n_{tracks} in vertex; x vertex pull",
                                     myNTrack_bins_.size() - 1,
                                     myNTrack_bins_.data());
    p_pullY_vsNtracks_[j] = new TH1F(Form("p_pullY_vsNtracks_%i", j),
                                     "y-pull vs n_{tracks};n_{tracks} in vertex; y vertex pull",
                                     myNTrack_bins_.size() - 1,
                                     myNTrack_bins_.data());
    p_pullZ_vsNtracks_[j] = new TH1F(Form("p_pullZ_vsNtracks_%i", j),
                                     "z-pull vs n_{tracks};n_{tracks} in vertex; z vertex pull",
                                     myNTrack_bins_.size() - 1,
                                     myNTrack_bins_.data());

    p_resolX_vsNtracks_[j] = new TH1F(Form("p_resolX_vsNtracks_%i", j),
                                      "x-resolution vs n_{tracks};n_{tracks} in vertex; x vertex resolution [#mum]",
                                      myNTrack_bins_.size() - 1,
                                      myNTrack_bins_.data());
    p_resolY_vsNtracks_[j] = new TH1F(Form("p_resolY_vsNtracks_%i", j),
                                      "y-resolution vs n_{tracks};n_{tracks} in vertex; y vertex resolution [#mum]",
                                      myNTrack_bins_.size() - 1,
                                      myNTrack_bins_.data());
    p_resolZ_vsNtracks_[j] = new TH1F(Form("p_resolZ_vsNtracks_%i", j),
                                      "z-resolution vs n_{tracks};n_{tracks} in vertex; z vertex resolution [#mum]",
                                      myNTrack_bins_.size() - 1,
                                      myNTrack_bins_.data());

    for (int i = 0; i < max_n_tracks; i++) {
      hpulls_[Form("pullX_%dTrks_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/xPullNtracks/histo_pullX_Ntracks_plot%i", i));
      hpulls_[Form("pullY_%dTrks_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/yPullNtracks/histo_pullY_Ntracks_plot%i", i));
      hpulls_[Form("pullZ_%dTrks_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/zPullNtracks/histo_pullZ_Ntracks_plot%i", i));
      hdiffs_[Form("diffX_%dTrks_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/xResolNtracks/histo_resolX_Ntracks_plot%i", i));
      hdiffs_[Form("diffY_%dTrks_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/yResolNtracks/histo_resolY_Ntracks_plot%i", i));
      hdiffs_[Form("diffZ_%dTrks_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/zResolNtracks/histo_resolZ_Ntracks_plot%i", i));
    }

    // vs sumPt

    p_pullX_vsSumPt_[j] = new TH1F(Form("p_pullX_vsSumPt_%i", j),
                                   "x-pull vs #Sigma p_{T};#sum p_{T} [GeV]; x vertex pull",
                                   mypT_bins_.size() - 1,
                                   mypT_bins_.data());
    p_pullY_vsSumPt_[j] = new TH1F(Form("p_pullY_vsSumPt_%i", j),
                                   "y-pull vs #Sigma p_{T};#sum p_{T} [GeV]; y vertex pull",
                                   mypT_bins_.size() - 1,
                                   mypT_bins_.data());
    p_pullZ_vsSumPt_[j] = new TH1F(Form("p_pullZ_vsSumPt_%i", j),
                                   "z-pull vs #Sigma p_{T};#sum p_{T} [GeV]; z vertex pull",
                                   mypT_bins_.size() - 1,
                                   mypT_bins_.data());

    p_resolX_vsSumPt_[j] = new TH1F(Form("p_resolX_vsSumPt_%i", j),
                                    "x-resolution vs #Sigma p_{T};#sum p_{T} [GeV]; x vertex resolution [#mum]",
                                    mypT_bins_.size() - 1,
                                    mypT_bins_.data());
    p_resolY_vsSumPt_[j] = new TH1F(Form("p_resolY_vsSumPt_%i", j),
                                    "y-resolution vs #Sigma p_{T};#sum p_{T} [GeV]; y vertex resolution [#mum]",
                                    mypT_bins_.size() - 1,
                                    mypT_bins_.data());
    p_resolZ_vsSumPt_[j] = new TH1F(Form("p_resolZ_vsSumPt_%i", j),
                                    "z-resolution vs #Sigma p_{T};#sum p_{T} [GeV]; z vertex resolution [#mum]",
                                    mypT_bins_.size() - 1,
                                    mypT_bins_.data());

    for (int i = 0; i < max_sum_pt; i++) {
      hpulls_[Form("pullX_%dsumPt_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/xPullSumPt/histo_pullX_sumPt_plot%i", i));
      hpulls_[Form("pullY_%dsumPt_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/yPullSumPt/histo_pullY_sumPt_plot%i", i));
      hpulls_[Form("pullZ_%dsumPt_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/zPullSumPt/histo_pullZ_sumPt_plot%i", i));
      hdiffs_[Form("diffX_%dsumPt_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/xResolSumPt/histo_resolX_sumPt_plot%i", i));
      hdiffs_[Form("diffY_%dsumPt_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/yResolSumPt/histo_resolY_sumPt_plot%i", i));
      hdiffs_[Form("diffZ_%dsumPt_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/zResolSumPt/histo_resolZ_sumPt_plot%i", i));
    }

    // vs n. vertices

    p_pullX_vsNVtx_[j] = new TH1F(Form("p_pullX_vsNVtx_%i", j),
                                  "x-pull vs n_{vertices};n_{vertices} in event; x vertex pull",
                                  myNVtx_bins_.size() - 1,
                                  myNVtx_bins_.data());
    p_pullY_vsNVtx_[j] = new TH1F(Form("p_pullY_vsNVtx_%i", j),
                                  "y-pull vs n_{vertices};n_{vertices} in event; y vertex pull",
                                  myNVtx_bins_.size() - 1,
                                  myNVtx_bins_.data());
    p_pullZ_vsNVtx_[j] = new TH1F(Form("p_pullZ_vsNVtx_%i", j),
                                  "z-pull vs n_{vertices};n_{vertices} in event; z vertex pull",
                                  myNVtx_bins_.size() - 1,
                                  myNVtx_bins_.data());

    p_resolX_vsNVtx_[j] = new TH1F(Form("p_resolX_vsNVtx_%i", j),
                                   "x-resolution vs n_{vertices};n_{vertices} in event; x vertex resolution [#mum]",
                                   myNVtx_bins_.size() - 1,
                                   myNVtx_bins_.data());
    p_resolY_vsNVtx_[j] = new TH1F(Form("p_resolY_vsNVtx_%i", j),
                                   "y-resolution vs n_{vertices};n_{vertices} in event; y vertex resolution [#mum]",
                                   myNVtx_bins_.size() - 1,
                                   myNVtx_bins_.data());
    p_resolZ_vsNVtx_[j] = new TH1F(Form("p_resolZ_vsNVtx_%i", j),
                                   "z-resolution vs n_{vertices};n_{vertices} in event; z vertex resolution [#mum]",
                                   myNVtx_bins_.size() - 1,
                                   myNVtx_bins_.data());

    for (int i = 0; i < max_n_vertices; i++) {
      hpulls_[Form("pullX_%dNVtx_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/xPullNvtx/histo_pullX_Nvtx_plot%i", i));
      hpulls_[Form("pullY_%dNVtx_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/yPullNvtx/histo_pullY_Nvtx_plot%i", i));
      hpulls_[Form("pullZ_%dNVtx_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/zPullNvtx/histo_pullZ_Nvtx_plot%i", i));
      hdiffs_[Form("diffX_%dNVtx_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/xResolNvtx/histo_resolX_Nvtx_plot%i", i));
      hdiffs_[Form("diffY_%dNVtx_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/yResolNvtx/histo_resolY_Nvtx_plot%i", i));
      hdiffs_[Form("diffZ_%dNVtx_file_%i", i, j)] =
          (TH1F*)fins[j]->Get(Form("PrimaryVertexResolution/zResolNvtx/histo_resolZ_Nvtx_plot%i", i));
    }
  }

  // dump the list of keys and check all needed histograms are available
  for (const auto& key : hpulls_) {
    if (key.second == nullptr) {
      std::cout << "!!!WARNING!!! FitPVResolution::FitPVResolution(): missing histograms for key " << key.first
                << ". I am going to exit. Good-bye!" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // compute and store the trend plots
  for (Int_t j = 0; j < nFiles_; j++) {
    // resolutions
    fillTrendPlotByIndex(p_resolX_vsSumPt_[j],
                         hdiffs_,
                         std::regex(("diffX_(.*)sumPt_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);
    fillTrendPlotByIndex(p_resolY_vsSumPt_[j],
                         hdiffs_,
                         std::regex(("diffY_(.*)sumPt_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);
    fillTrendPlotByIndex(p_resolZ_vsSumPt_[j],
                         hdiffs_,
                         std::regex(("diffZ_(.*)sumPt_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);

    fillTrendPlotByIndex(p_resolX_vsNtracks_[j],
                         hdiffs_,
                         std::regex(("diffX_(.*)Trks_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);
    fillTrendPlotByIndex(p_resolY_vsNtracks_[j],
                         hdiffs_,
                         std::regex(("diffY_(.*)Trks_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);
    fillTrendPlotByIndex(p_resolZ_vsNtracks_[j],
                         hdiffs_,
                         std::regex(("diffZ_(.*)Trks_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);

    fillTrendPlotByIndex(p_resolX_vsNVtx_[j],
                         hdiffs_,
                         std::regex(("diffX_(.*)NVtx_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);
    fillTrendPlotByIndex(p_resolY_vsNVtx_[j],
                         hdiffs_,
                         std::regex(("diffY_(.*)NVtx_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);
    fillTrendPlotByIndex(p_resolZ_vsNVtx_[j],
                         hdiffs_,
                         std::regex(("diffZ_(.*)NVtx_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);

    // pulls

    fillTrendPlotByIndex(p_pullX_vsSumPt_[j],
                         hpulls_,
                         std::regex(("pullX_(.*)sumPt_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);
    fillTrendPlotByIndex(p_pullY_vsSumPt_[j],
                         hpulls_,
                         std::regex(("pullY_(.*)sumPt_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);
    fillTrendPlotByIndex(p_pullZ_vsSumPt_[j],
                         hpulls_,
                         std::regex(("pullZ_(.*)sumPt_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);

    fillTrendPlotByIndex(p_pullX_vsNtracks_[j],
                         hpulls_,
                         std::regex(("pullX_(.*)Trks_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);
    fillTrendPlotByIndex(p_pullY_vsNtracks_[j],
                         hpulls_,
                         std::regex(("pullY_(.*)Trks_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);
    fillTrendPlotByIndex(p_pullZ_vsNtracks_[j],
                         hpulls_,
                         std::regex(("pullZ_(.*)Trks_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);

    fillTrendPlotByIndex(p_pullX_vsNVtx_[j],
                         hpulls_,
                         std::regex(("pullX_(.*)NVtx_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);
    fillTrendPlotByIndex(p_pullY_vsNVtx_[j],
                         hpulls_,
                         std::regex(("pullY_(.*)NVtx_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);
    fillTrendPlotByIndex(p_pullZ_vsNVtx_[j],
                         hpulls_,
                         std::regex(("pullZ_(.*)NVtx_file_" + std::to_string(j)).c_str()),
                         PVValHelper::WIDTH);

    // beautify

    makeNiceTrendPlotStyle(p_resolX_vsSumPt_[j], colors[j], markers[j]);
    makeNiceTrendPlotStyle(p_resolY_vsSumPt_[j], colors[j], markers[j]);
    makeNiceTrendPlotStyle(p_resolZ_vsSumPt_[j], colors[j], markers[j]);

    makeNiceTrendPlotStyle(p_resolX_vsNtracks_[j], colors[j], markers[j]);
    makeNiceTrendPlotStyle(p_resolY_vsNtracks_[j], colors[j], markers[j]);
    makeNiceTrendPlotStyle(p_resolZ_vsNtracks_[j], colors[j], markers[j]);

    makeNiceTrendPlotStyle(p_resolX_vsNVtx_[j], colors[j], markers[j]);
    makeNiceTrendPlotStyle(p_resolY_vsNVtx_[j], colors[j], markers[j]);
    makeNiceTrendPlotStyle(p_resolZ_vsNVtx_[j], colors[j], markers[j]);

    // pulls

    makeNiceTrendPlotStyle(p_pullX_vsSumPt_[j], colors[j], markers[j]);
    makeNiceTrendPlotStyle(p_pullY_vsSumPt_[j], colors[j], markers[j]);
    makeNiceTrendPlotStyle(p_pullZ_vsSumPt_[j], colors[j], markers[j]);

    makeNiceTrendPlotStyle(p_pullX_vsNtracks_[j], colors[j], markers[j]);
    makeNiceTrendPlotStyle(p_pullY_vsNtracks_[j], colors[j], markers[j]);
    makeNiceTrendPlotStyle(p_pullZ_vsNtracks_[j], colors[j], markers[j]);

    makeNiceTrendPlotStyle(p_pullX_vsNVtx_[j], colors[j], markers[j]);
    makeNiceTrendPlotStyle(p_pullY_vsNVtx_[j], colors[j], markers[j]);
    makeNiceTrendPlotStyle(p_pullZ_vsNVtx_[j], colors[j], markers[j]);
  }

  // adjust the maxima

  adjustMaximum(p_resolX_vsSumPt_, nFiles_);
  adjustMaximum(p_resolY_vsSumPt_, nFiles_);
  adjustMaximum(p_resolZ_vsSumPt_, nFiles_);

  adjustMaximum(p_resolX_vsNtracks_, nFiles_);
  adjustMaximum(p_resolY_vsNtracks_, nFiles_);
  adjustMaximum(p_resolZ_vsNtracks_, nFiles_);

  adjustMaximum(p_resolX_vsNVtx_, nFiles_);
  adjustMaximum(p_resolY_vsNVtx_, nFiles_);
  adjustMaximum(p_resolZ_vsNVtx_, nFiles_);

  adjustMaximum(p_pullX_vsSumPt_, nFiles_);
  adjustMaximum(p_pullY_vsSumPt_, nFiles_);
  adjustMaximum(p_pullZ_vsSumPt_, nFiles_);

  adjustMaximum(p_pullX_vsNtracks_, nFiles_);
  adjustMaximum(p_pullY_vsNtracks_, nFiles_);
  adjustMaximum(p_pullZ_vsNtracks_, nFiles_);

  adjustMaximum(p_pullX_vsNVtx_, nFiles_);
  adjustMaximum(p_pullY_vsNVtx_, nFiles_);
  adjustMaximum(p_pullZ_vsNVtx_, nFiles_);

  TCanvas* c1 = new TCanvas("VertexResolVsSumPt", "Vertex Resolution vs #sum p_{T} [GeV]", 1500, 700);
  c1->Divide(3, 1);
  TCanvas* c2 = new TCanvas("VertexPullVsSumPt", "Vertex Resolution vs #sum p_{T} [GeV]", 1500, 700);
  c2->Divide(3, 1);
  TCanvas* c3 = new TCanvas("VertexResolVsNTracks", "Vertex Resolution vs n. tracks", 1500, 700);
  c3->Divide(3, 1);
  TCanvas* c4 = new TCanvas("VertexPullVsNTracks", "Vertex Resolution vs n. tracks", 1500, 700);
  c4->Divide(3, 1);
  TCanvas* c5 = new TCanvas("VertexResolVsNVtx", "Vertex Resolution vs n. vertices", 1500, 700);
  c5->Divide(3, 1);
  TCanvas* c6 = new TCanvas("VertexPullVsNVtx", "Vertex Resolution vs n. vertices", 1500, 700);
  c6->Divide(3, 1);

  for (Int_t c = 1; c <= 3; c++) {
    c1->cd(c)->SetBottomMargin(0.14);
    c1->cd(c)->SetLeftMargin(0.15);
    c1->cd(c)->SetRightMargin(0.05);
    c1->cd(c)->SetTopMargin(0.05);

    c2->cd(c)->SetBottomMargin(0.14);
    c2->cd(c)->SetLeftMargin(0.15);
    c2->cd(c)->SetRightMargin(0.05);
    c2->cd(c)->SetTopMargin(0.05);

    c3->cd(c)->SetBottomMargin(0.14);
    c3->cd(c)->SetLeftMargin(0.15);
    c3->cd(c)->SetRightMargin(0.05);
    c3->cd(c)->SetTopMargin(0.05);

    c4->cd(c)->SetBottomMargin(0.14);
    c4->cd(c)->SetLeftMargin(0.15);
    c4->cd(c)->SetRightMargin(0.05);
    c4->cd(c)->SetTopMargin(0.05);

    c5->cd(c)->SetBottomMargin(0.14);
    c5->cd(c)->SetLeftMargin(0.15);
    c5->cd(c)->SetRightMargin(0.05);
    c5->cd(c)->SetTopMargin(0.05);

    c6->cd(c)->SetBottomMargin(0.14);
    c6->cd(c)->SetLeftMargin(0.15);
    c6->cd(c)->SetRightMargin(0.05);
    c6->cd(c)->SetTopMargin(0.05);
  }

  TLegend* lego = new TLegend(0.18, 0.80, 0.79, 0.93);
  // might be useful if many objects are compared
  if (nFiles_ > 4) {
    lego->SetNColumns(2);
  }

  lego->SetFillColor(10);
  if (nFiles_ > 3) {
    lego->SetTextSize(0.032);
  } else {
    lego->SetTextSize(0.042);
  }
  lego->SetTextFont(42);
  lego->SetFillColor(10);
  lego->SetLineColor(10);
  lego->SetShadowColor(10);

  TPaveText* ptDate = new TPaveText(0.17, 0.96, 0.50, 0.99, "blNDC");
  ptDate->SetFillColor(kYellow);
  //ptDate->SetFillColor(10);
  ptDate->SetBorderSize(1);
  ptDate->SetLineColor(kBlue);
  ptDate->SetLineWidth(1);
  ptDate->SetTextFont(32);
  TText* textDate = ptDate->AddText(theDate);
  textDate->SetTextSize(0.04);
  textDate->SetTextColor(kBlue);

  for (Int_t j = 0; j < nFiles_; j++) {
    // first canvas

    //p_resolX_vsSumPt_[j]->GetXaxis()->SetRangeUser(10., 400.);
    //p_resolY_vsSumPt_[j]->GetXaxis()->SetRangeUser(10., 400.);
    //p_resolZ_vsSumPt_[j]->GetXaxis()->SetRangeUser(10., 400.);

    c1->cd(1);
    j == 0 ? p_resolX_vsSumPt_[j]->Draw("E1") : p_resolX_vsSumPt_[j]->Draw("E1same");
    lego->AddEntry(p_resolX_vsSumPt_[j], LegLabels[j]);

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    TPad* current_pad = static_cast<TPad*>(c1->GetPad(1));
    CMS_lumi(current_pad, 6, 33);

    c1->cd(2);
    j == 0 ? p_resolY_vsSumPt_[j]->Draw("E1") : p_resolY_vsSumPt_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c1->GetPad(2));
    CMS_lumi(current_pad, 6, 33);

    c1->cd(3);
    j == 0 ? p_resolZ_vsSumPt_[j]->Draw("E1") : p_resolZ_vsSumPt_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c1->GetPad(3));
    CMS_lumi(current_pad, 6, 33);

    // second canvas

    c2->cd(1);
    j == 0 ? p_pullX_vsSumPt_[j]->Draw("E1") : p_pullX_vsSumPt_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c2->GetPad(1));
    CMS_lumi(current_pad, 6, 33);

    c2->cd(2);
    j == 0 ? p_pullY_vsSumPt_[j]->Draw("E1") : p_pullY_vsSumPt_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c2->GetPad(2));
    CMS_lumi(current_pad, 6, 33);

    c2->cd(3);
    j == 0 ? p_pullZ_vsSumPt_[j]->Draw("E1") : p_pullZ_vsSumPt_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c2->GetPad(3));
    CMS_lumi(current_pad, 6, 33);

    // third canvas

    c3->cd(1);
    j == 0 ? p_resolX_vsNtracks_[j]->Draw("E1") : p_resolX_vsNtracks_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c3->GetPad(1));
    CMS_lumi(current_pad, 6, 33);

    c3->cd(2);
    j == 0 ? p_resolY_vsNtracks_[j]->Draw("E1") : p_resolY_vsNtracks_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c3->GetPad(2));
    CMS_lumi(current_pad, 6, 33);

    c3->cd(3);
    j == 0 ? p_resolZ_vsNtracks_[j]->Draw("E1") : p_resolZ_vsNtracks_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c3->GetPad(3));
    CMS_lumi(current_pad, 6, 33);

    // fourth canvas

    c4->cd(1);
    j == 0 ? p_pullX_vsNtracks_[j]->Draw("E1") : p_pullX_vsNtracks_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c4->GetPad(1));
    CMS_lumi(current_pad, 6, 33);

    c4->cd(2);
    j == 0 ? p_pullY_vsNtracks_[j]->Draw("E1") : p_pullY_vsNtracks_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c4->GetPad(2));
    CMS_lumi(current_pad, 6, 33);

    c4->cd(3);
    j == 0 ? p_pullZ_vsNtracks_[j]->Draw("E1") : p_pullZ_vsNtracks_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c4->GetPad(3));
    CMS_lumi(current_pad, 6, 33);

    // fifth canvas

    c5->cd(1);
    j == 0 ? p_resolX_vsNVtx_[j]->Draw("E1") : p_resolX_vsNVtx_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c5->GetPad(1));
    CMS_lumi(current_pad, 6, 33);

    c5->cd(2);
    j == 0 ? p_resolY_vsNVtx_[j]->Draw("E1") : p_resolY_vsNVtx_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c5->GetPad(2));
    CMS_lumi(current_pad, 6, 33);

    c5->cd(3);
    j == 0 ? p_resolZ_vsNVtx_[j]->Draw("E1") : p_resolZ_vsNVtx_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c5->GetPad(3));
    CMS_lumi(current_pad, 6, 33);

    // sixth canvas

    c6->cd(1);
    j == 0 ? p_pullX_vsNVtx_[j]->Draw("E1") : p_pullX_vsNVtx_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c6->GetPad(1));
    CMS_lumi(current_pad, 6, 33);

    c6->cd(2);
    j == 0 ? p_pullY_vsNVtx_[j]->Draw("E1") : p_pullY_vsNVtx_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c6->GetPad(2));
    CMS_lumi(current_pad, 6, 33);

    c6->cd(3);
    j == 0 ? p_pullZ_vsNVtx_[j]->Draw("E1") : p_pullZ_vsNVtx_[j]->Draw("E1same");

    if (j == nFiles_ - 1)
      lego->Draw("same");
    if (theDate.Length() != 0)
      ptDate->Draw("same");
    current_pad = static_cast<TPad*>(c6->GetPad(3));
    CMS_lumi(current_pad, 6, 33);
  }

  if (theDate.Length() != 0)
    theDate.Prepend("_");

  TString theStrAlignment = LegLabels[0];
  for (Int_t j = 1; j < nFiles_; j++) {
    theStrAlignment += ("_vs_" + LegLabels[j]);
  }
  theStrAlignment.ReplaceAll(" ", "_");

  c1->SaveAs("VertexResolVsSumPt_" + theStrAlignment + theDate + ".png");
  c2->SaveAs("VertexPullVsSumPt_" + theStrAlignment + theDate + ".png");
  c3->SaveAs("VertexResolVsNTracks_" + theStrAlignment + theDate + ".png");
  c4->SaveAs("VertexPullVsNTracks_" + theStrAlignment + theDate + ".png");
  c5->SaveAs("VertexResolVsNVertices_" + theStrAlignment + theDate + ".png");
  c6->SaveAs("VertexPullVsNVertices_" + theStrAlignment + theDate + ".png");

  c1->SaveAs("VertexResolVsSumPt_" + theStrAlignment + theDate + ".pdf");
  c2->SaveAs("VertexPullVsSumPt_" + theStrAlignment + theDate + ".pdf");
  c3->SaveAs("VertexResolVsNTracks_" + theStrAlignment + theDate + ".pdf");
  c4->SaveAs("VertexPullVsNTracks_" + theStrAlignment + theDate + ".pdf");
  c5->SaveAs("VertexResolVsNVertices_" + theStrAlignment + theDate + ".pdf");
  c6->SaveAs("VertexPullVsNVertices_" + theStrAlignment + theDate + ".pdf");

  delete c1;
  delete c2;
  delete c3;
  delete c4;
  delete c5;
  delete c6;

  // delete everything in the source list
  for (std::vector<PVResolutionVariables*>::iterator it = PVResolution::sourceList.begin();
       it != PVResolution::sourceList.end();
       ++it) {
    delete (*it);
  }
}

/*--------------------------------------------------------------------*/
void fillTrendPlotByIndex(TH1F* trendPlot,
                          std::unordered_map<std::string, TH1F*>& h,
                          std::regex toMatch,
                          PVValHelper::estimator fitPar_)
/*--------------------------------------------------------------------*/
{
  for (const auto& iterator : h) {
    statmode::fitParams myFit = fitResolutions(iterator.second, false);

    int bin = -1;
    std::string result;
    try {
      std::smatch match;
      if (std::regex_search(iterator.first, match, toMatch) && match.size() > 1) {
        result = match.str(1);
        bin = std::stoi(result) + 1;
      } else {
        result = std::string("");
        continue;
      }
    } catch (std::regex_error& e) {
      // Syntax error in the regular expression
    }

    switch (fitPar_) {
      case PVValHelper::MEAN: {
        float mean_ = myFit.first.first;
        float meanErr_ = myFit.first.second;
        trendPlot->SetBinContent(bin, mean_);
        trendPlot->SetBinError(bin, meanErr_);
        break;
      }
      case PVValHelper::WIDTH: {
        float width_ = myFit.second.first;
        float widthErr_ = myFit.second.second;
        trendPlot->SetBinContent(bin, width_);
        trendPlot->SetBinError(bin, widthErr_);
        break;
      }
      case PVValHelper::MEDIAN: {
        float median_ = PVValHelper::getMedian(iterator.second).value();
        float medianErr_ = PVValHelper::getMedian(iterator.second).error();
        trendPlot->SetBinContent(bin, median_);
        trendPlot->SetBinError(bin, medianErr_);
        break;
      }
      case PVValHelper::MAD: {
        float mad_ = PVValHelper::getMAD(iterator.second).value();
        float madErr_ = PVValHelper::getMAD(iterator.second).error();
        trendPlot->SetBinContent(bin, mad_);
        trendPlot->SetBinError(bin, madErr_);
        break;
      }
      default:
        std::cout << "fillTrendPlotByIndex() " << fitPar_ << " unknown estimator!" << std::endl;
        break;
    }
  }
}

/*--------------------------------------------------------------------*/
statmode::fitParams fitResolutions(TH1* hist, bool singleTime)
/*--------------------------------------------------------------------*/
{
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
    std::cout << "FitPVResolution::fitResolutions(): histogram" << hist->GetName() << " mean or sigma are NaN!!"
              << std::endl;
  }

  TF1 func("tmp", "gaus", mean - 2. * sigma, mean + 2. * sigma);
  if (0 == hist->Fit(&func, "QNR")) {  // N: do not blow up file by storing fit!
    mean = func.GetParameter(1);
    sigma = func.GetParameter(2);

    if (!singleTime) {
      // second fit: three sigma of first fit around mean of first fit
      func.SetRange(std::max(mean - 3 * sigma, minHist), std::min(mean + 3 * sigma, maxHist));
      // I: integral gives more correct results if binning is too wide
      // L: Likelihood can treat empty bins correctly (if hist not weighted...)
      if (0 == hist->Fit(&func, "Q0LR")) {
        if (hist->GetFunction(func.GetName())) {  // Take care that it is later on drawn:
          hist->GetFunction(func.GetName())->ResetBit(TF1::kNotDraw);
        }
      }
    }
  }

  return std::make_pair(std::make_pair(func.GetParameter(1), func.GetParError(1)),
                        std::make_pair(func.GetParameter(2), func.GetParError(2)));
}

/*--------------------------------------------------------------------*/
void makeNiceTrendPlotStyle(TH1* hist, Int_t color, Int_t style)
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
  hist->GetYaxis()->SetLabelSize(.055);
  hist->GetXaxis()->SetLabelSize(.055);
  hist->GetXaxis()->SetNdivisions(505);
  if (color != 8) {
    hist->SetMarkerSize(1.2);
  } else {
    hist->SetLineWidth(3);
    hist->SetMarkerSize(0.0);
  }
  hist->SetMarkerStyle(style);
  hist->SetLineColor(color);
  hist->SetMarkerColor(color);

  if (TString(hist->GetName()).Contains("pull"))
    hist->GetYaxis()->SetRangeUser(0., 2.);
}

/*--------------------------------------------------------------------*/
void adjustMaximum(TH1F* histos[], int size)
/*--------------------------------------------------------------------*/
{
  float theMax(-999.);
  for (int i = 0; i < size; i++) {
    if (histos[i]->GetMaximum() > theMax)
      theMax = histos[i]->GetMaximum();
  }

  for (int i = 0; i < size; i++) {
    histos[i]->SetMaximum(theMax * 1.25);
  }
}

/*--------------------------------------------------------------------*/
void setPVResolStyle() {
  /*--------------------------------------------------------------------*/

  writeExtraText = true;  // if extra text
  lumi_13p6TeV = "pp collisions";
  lumi_13TeV = "pp collisions";
  lumi_0p9TeV = "pp collisions";
  extraText = "Internal";

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
