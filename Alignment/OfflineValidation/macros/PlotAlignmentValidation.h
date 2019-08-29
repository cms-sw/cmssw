#ifndef PLOTALIGNNMENTVALIDATION_H_
#define PLOTALIGNNMENTVALIDATION_H_

#include "Alignment/OfflineValidation/interface/TkOffTreeVariables.h"

#include "TCanvas.h"
#include "TDirectory.h"
#include "TDirectoryFile.h"
#include "TFile.h"
#include "THStack.h"
#include "TLegend.h"
#include "TString.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTree.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

class TkOfflineVariables {
public:
  TkOfflineVariables(std::string fileName, std::string baseDir, std::string legName = "", int color = 1, int style = 1);
  ~TkOfflineVariables();
  int getLineColor() { return lineColor; }
  int getLineStyle() { return lineStyle; }
  std::string getName() { return legendName; }
  TTree* getTree() { return tree; }
  TFile* getFile() { return file; }
  int getPhase() { return phase; }

private:
  TFile* file;
  TTree* tree;
  int lineColor;
  int lineStyle;
  int phase;
  std::string legendName;
};

TkOfflineVariables::TkOfflineVariables(
    std::string fileName, std::string baseDir, std::string legName, int lColor, int lStyle) {
  lineColor = lColor;
  lineStyle = lStyle % 100;
  if (legName == "") {
    int start = 0;
    if (fileName.find('/'))
      start = fileName.find_last_of('/') + 1;
    int stop = fileName.find_last_of('.');
    legendName = fileName.substr(start, stop - start);
  } else {
    legendName = legName;
  }

  //fill the tree pointer
  file = TFile::Open(fileName.c_str());
  TDirectoryFile* d = 0;
  if (file->Get(baseDir.c_str())) {
    d = (TDirectoryFile*)file->Get(baseDir.c_str());
    if ((*d).Get("TkOffVal")) {
      tree = (TTree*)(*d).Get("TkOffVal");
    } else {
      std::cout << "no tree named TkOffVal" << std::endl;
      assert(false);
    }
    TDirectoryFile* d2 = (TDirectoryFile*)d->Get("Pixel");
    assert(d2);
    phase = (int)((bool)d2->Get("P1PXBBarrel_1"));
  } else {
    std::cout << "no directory named " << baseDir.c_str() << std::endl;
    assert(false);
  }
}

TkOfflineVariables::~TkOfflineVariables() { delete file; }

class PlotAlignmentValidation {
public:
  //PlotAlignmentValidation(TString *tmp);
  PlotAlignmentValidation(bool bigtext = false);
  PlotAlignmentValidation(
      const char* inputFile, std::string fileName = "", int lineColor = 1, int lineStyle = 1, bool bigtext = false);
  ~PlotAlignmentValidation();
  void loadFileList(const char* inputFile, std::string fileName = "", int lineColor = 2, int lineStyle = 1);
  void useFitForDMRplots(bool usefit = false);
  void legendOptions(TString options);
  void plotOutlierModules(const char* outputFileName = "OutlierModules.ps",
                          std::string plotVariable = "chi2PerDofX",
                          float chi2_cut = 10,
                          unsigned int minHits = 50);  //method dumps selected modules into ps file
  void plotSubDetResiduals(
      bool plotNormHisto = false,
      unsigned int subDetId =
          7);  //subDetector number :1.TPB, 2.TBE+, 3.TBE-, 4.TIB, 5.TID+, 6.TID-, 7.TOB, 8.TEC+ or 9.TEC-
  void plotDMR(
      const std::string& plotVar = "medianX",
      Int_t minHits = 50,
      const std::string& options =
          "plain");  // plotVar=mean,meanX,meanY,median,rms etc., comma-separated list can be given; minHits=the minimum hits needed for module to appear in plot; options="plain" for regular DMR, "split" for inwards/outwards split, "layers" for layerwise DMR, "layer=N" for Nth layer, or combination of the previous (e.g. "split layers")
  void plotSurfaceShapes(const std::string& options = "layers", const std::string& variable = "");
  void plotChi2(const char* inputFile);
  // plotSurfaceShapes: options="split","layers"/"layer","subdet"
  void plotHitMaps();
  void setOutputDir(std::string dir);
  void setTreeBaseDir(std::string dir = "TrackerOfflineValidationStandalone");
  void residual_by_moduleID(unsigned int moduleid);
  int numberOfLayers(int phase, int subdetector);
  int maxNumberOfLayers(int subdetector);

  THStack* addHists(
      const TString& selection,
      const TString& residType = "xPrime",
      TLegend** myLegend = 0,
      bool printModuleIds = false,
      bool validforphase0 =
          false);  //add hists fulfilling 'selection' on TTree; residType: xPrime,yPrime,xPrimeNorm,yPrimeNorm,x,y,xNorm; if (printModuleIds): cout DetIds

  float twotailedStudentTTestEqualMean(float t, float v);

  // These are helpers for DMR plotting

  struct DMRPlotInfo {
    std::string variable;
    int nbins;
    double min, max;
    int minHits;
    bool plotPlain, plotSplits, plotLayers;
    int subDetId, nLayers;
    THStack* hstack;
    TLegend* legend;
    TkOfflineVariables* vars;
    float maxY;
    TH1F* h;
    TH1F* h1;
    TH1F* h2;
    bool firsthisto;
  };

private:
  TList* getTreeList();
  std::string treeBaseDir;

  bool useFit_;
  bool showMean_;
  bool showRMS_;
  bool showMeanError_;
  bool showRMSError_;
  bool showModules_;
  bool showUnderOverFlow_;
  bool twolines_;
  bool bigtext_;
  const static TString summaryfilename;
  ofstream summaryfile;
  bool openedsummaryfile = false;
  TFile* rootsummaryfile;

  std::vector<double> vmean, vdeltamean, vrms, vmeanerror, vPValueEqualSplitMeans, vPValueMeanEqualIdeal,
      vPValueRMSEqualIdeal, vAlignmentUncertainty;
  double resampleTestOfEqualMeans(TH1F* h1, TH1F* h2, int numSamples);
  double resampleTestOfEqualRMS(TH1F* h1, TH1F* h2, int numSamples);

  void storeHistogramInRootfile(TH1* hist);
  TF1* fitGauss(TH1* hist, int color);
  //void plotBoxOverview(TCanvas &c1, TList &treeList,std::string plot_Var1a,std::string plot_Var1b, std::string plot_Var2, Int_t filenumber,Int_t minHits);
  //void plot1DDetailsSubDet(TCanvas &c1, TList &treeList, std::string plot_Var1a,std::string plot_Var1b, std::string plot_Var2, Int_t minHits);
  //void plot1DDetailsBarrelLayer(TCanvas &c1, TList &treeList, std::string plot_Var1a,std::string plot_Var1b, Int_t minHits);
  //void plot1DDetailsDiskWheel(TCanvas &c1, TList &treelist, std::string plot_Var1a,std::string plot_Var1b, Int_t minHits);
  void plotSS(const std::string& options = "layers", const std::string& variable = "");
  void setHistStyle(TH1& hist, const char* titleX, const char* titleY, int color);
  void setTitleStyle(TNamed& h,
                     const char* titleX,
                     const char* titleY,
                     int subDetId,
                     bool isSurfaceDeformation = false,
                     TString secondline = "");
  void setNiceStyle();
  void setCanvasStyle(TCanvas& canv);
  void setLegendStyle(TLegend& leg);
  void scaleXaxis(TH1* hist, Int_t scale);
  TObject* findObjectFromCanvas(TCanvas* canv, const char* className, Int_t n = 1);

  TString outputFile;
  std::string outputDir;
  TList* sourcelist;
  std::vector<TkOfflineVariables*> sourceList;
  bool moreThanOneSource;
  std::string fileNames[10];
  int fileCounter;

  std::string getSelectionForDMRPlot(int minHits, int subDetId, int direction = 0, int layer = 0);
  std::string getVariableForDMRPlot(
      const std::string& histoname, const std::string& variable, int nbins, double min, double max);
  void setDMRHistStyleAndLegend(TH1F* h, DMRPlotInfo& plotinfo, int direction = 0, int layer = 0);
  void plotDMRHistogram(DMRPlotInfo& plotinfo, int direction = 0, int layer = 0, std::string subdet = "");
  void modifySSHistAndLegend(THStack* hs, TLegend* legend);
  void openSummaryFile();
  vector<TH1*> findmodule(TFile* f, unsigned int moduleid);
};

#endif  // PLOTALIGNNMENTVALIDATION_H_
