#ifndef PLOTALIGNNMENTVALIDATION_H_
#define PLOTALIGNNMENTVALIDATION_H_

#include <TStyle.h>
#include <TSystem.h>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include "TTree.h"
#include "TString.h"
#include "TDirectory.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TDirectoryFile.h"
#include "TLegend.h"
#include "THStack.h"
#include <exception>

// This line works only if we have a CMSSW environment...
#include "Alignment/OfflineValidation/interface/TkOffTreeVariables.h"

class TkOfflineVariables {
public:
  TkOfflineVariables(std::string fileName, std::string baseDir, std::string legName="", int color=1, int style=1);
  int getLineColor(){ return lineColor; };
  int getLineStyle(){ return lineStyle; };
  std::string getName(){ return legendName; }
  TTree* getTree(){ return tree; };
  TFile* getFile(){ return file; };
private:
  TFile* file;
  TTree* tree;
  int lineColor;
  int lineStyle;
  std::string legendName;
};


TkOfflineVariables::TkOfflineVariables(std::string fileName, std::string baseDir, std::string legName, int lColor, int lStyle)
{
  lineColor = lColor;
  lineStyle = lStyle;
  if (legName=="") {
    int start = 0;
    if (fileName.find('/') ) start =fileName.find_last_of('/')+1;
    int stop = fileName.find_last_of('.');
    legendName = fileName.substr(start,stop-start);
  } else { 
    legendName = legName;
  }

  //fill the tree pointer
  file = TFile::Open( fileName.c_str() );
  TDirectoryFile *d = 0;
  if (file->Get( baseDir.c_str() ) )  {
    d = (TDirectoryFile*)file->Get( baseDir.c_str() );
    if ((*d).Get("TkOffVal")) {
      tree = (TTree*)(*d).Get("TkOffVal");
    } else {
      std::cout<<"no tree named TkOffVal"<<std::endl;
    }
  } else {
    std::cout<<"no directory named "<<baseDir.c_str()<<std::endl;
  }
}

class PlotAlignmentValidation {
public:
  //PlotAlignmentValidation(TString *tmp);
  PlotAlignmentValidation(const char *inputFile,std::string fileName="", int lineColor=1, int lineStyle=1);
  ~PlotAlignmentValidation();
  void loadFileList(const char *inputFile, std::string fileName="", int lineColor=2, int lineStyle=1);
  void useFitForDMRplots(bool usefit = false);
  void plotOutlierModules(const char *outputFileName="OutlierModules.ps",std::string plotVariable = "chi2PerDofX" ,float chi2_cut = 10,unsigned int minHits = 50);//method dumps selected modules into ps file
  void plotSubDetResiduals(bool plotNormHisto=false, unsigned int subDetId=7);//subDetector number :1.TPB, 2.TBE+, 3.TBE-, 4.TIB, 5.TID+, 6.TID-, 7.TOB, 8.TEC+ or 9.TEC-
  void plotDMR(const std::string& plotVar="medianX",Int_t minHits = 50, const std::string& options = "plain"); // plotVar=mean,meanX,meanY,median,rms etc., comma-separated list can be given; minHits=the minimum hits needed for module to appear in plot; options="plain" for regular DMR, "split" for inwards/outwards split, "layers" for layerwise DMR, "layer=N" for Nth layer, or combination of the previous (e.g. "split layers")
  void plotSurfaceShapes(const std::string& options = "layers",const std::string& variable="");
  // plotSurfaceShapes: options="split","layers"/"layer","subdet"
  void plotHitMaps();
  void setOutputDir( std::string dir );
  void setTreeBaseDir( std::string dir = "TrackerOfflineValidationStandalone");
  
  THStack* addHists(const char *selection, const TString &residType = "xPrime", bool printModuleIds = false);//add hists fulfilling 'selection' on TTree; residType: xPrime,yPrime,xPrimeNorm,yPrimeNorm,x,y,xNorm; if (printModuleIds): cout DetIds
  
private : 
  TList getTreeList();
  std::string treeBaseDir;

  bool useFit_;

  std::pair<float,float> fitGauss(TH1 *hist,int color);
  //void plotBoxOverview(TCanvas &c1, TList &treeList,std::string plot_Var1a,std::string plot_Var1b, std::string plot_Var2, Int_t filenumber,Int_t minHits);
  //void plot1DDetailsSubDet(TCanvas &c1, TList &treeList, std::string plot_Var1a,std::string plot_Var1b, std::string plot_Var2, Int_t minHits);
  //void plot1DDetailsBarrelLayer(TCanvas &c1, TList &treeList, std::string plot_Var1a,std::string plot_Var1b, Int_t minHits);
  //void plot1DDetailsDiskWheel(TCanvas &c1, TList &treelist, std::string plot_Var1a,std::string plot_Var1b, Int_t minHits);
  void plotSS(const std::string& options = "layers",const std::string& variable="");
  void setHistStyle( TH1& hist,const char* titleX, const char* titleY, int color);
  void setTitleStyle( TNamed& h,const char* titleX, const char* titleY, int subDetId);
  void setNiceStyle();
  void setCanvasStyle( TCanvas& canv );
  void setLegendStyle( TLegend& leg );

  TString outputFile;
  std::string outputDir;
  TList *sourcelist;
  std::vector<TkOfflineVariables*> sourceList;
  bool moreThanOneSource;
  std::string fileNames[10];
  int fileCounter;	

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

  std::string getSelectionForDMRPlot(int minHits, int subDetId, int direction = 0, int layer = 0);
  std::string getVariableForDMRPlot(const std::string& histoname, const std::string& variable,
				    int nbins, double min, double max);
  void setDMRHistStyleAndLegend(TH1F* h, DMRPlotInfo& plotinfo, int direction = 0, int layer = 0);
  void plotDMRHistogram(DMRPlotInfo& plotinfo, int direction = 0, int layer = 0);

};

#endif // PLOTALIGNNMENTVALIDATION_H_
