#ifndef PhysicsTools_SideBandSubtraction_h
#define PhysicsTools_SideBandSubtraction_h
#include <iostream>
#include <string>
#include <cstring>
#include <cstdio>
#include <sstream>
#include "TRandom.h"
#include "TCanvas.h"
#include <TROOT.h>
#include <TFile.h>
#include <TH1F.h>
#include "TPaveText.h"
#include <TObject.h>
#include <TLegend.h>
#include <TPaveLabel.h>
#include <TStyle.h>
#include <TString.h>
#include "TDatime.h"
#include <TString.h>
#include <TKey.h>
#include <TClass.h>

#include <RooAddPdf.h>
#include <RooRealVar.h>
#include "RooAbsPdf.h"
#include "RooDataSet.h"
#include "RooCmdArg.h"
#include "RooGlobalFunc.h"
#include "RooPlot.h"
#include "RooFitResult.h"
#include "RooNLLVar.h"
#include "RooMinuit.h"
#include "RooExtendPdf.h"
#include "RooGaussian.h"
#include "RooKeysPdf.h"
#include "RooDecay.h"
#include "RooTruthModel.h"
#include "RooGaussModel.h"
#include "RooHistPdf.h"
#include "RooExponential.h"
#include "RooUnblindPrecision.h"
#include "RooUnblindOffset.h"
#include "RooProdPdf.h"
#include "RooCategory.h"
#include "RooThresholdCategory.h"

typedef struct 
{
  float min;
  float max;
  std::string RegionName;
} SbsRegion;

class SideBandSubtract 
{
 private:
  void print_plot(RooRealVar printVar, string outname);
  Double_t getYield(std::vector<SbsRegion> Regions, RooAbsPdf *PDF);
  RooAbsPdf *BackgroundPDF;
  RooAbsPdf *ModelPDF;
  RooDataSet* Data;
  RooRealVar* SeparationVariable;
  bool verbose;
  std::vector<SbsRegion> SignalRegions;
  std::vector<SbsRegion> SideBandRegions;
  std::vector<TH1F> SideBandHistos;
  std::vector<TH1F> RawHistos;
  std::vector<TH1F> SBSHistos;
  const std::vector<TH1F*> BaseHistos;
  RooFitResult *fit_result;
  Double_t SignalSidebandRatio;
 public:
  SideBandSubtract(RooAbsPdf *model_shape, 
		   RooAbsPdf *bkg_shape, 
		   RooDataSet* data, 
		   RooRealVar* sep_var, 
		   const std::vector<TH1F*> base, 
		   bool verb);
  ~SideBandSubtract();
  void addSignalRegion(float min, float max);
  void addSideBandRegion(float min, float max);
  int doGlobalFit();
  int doSubtraction(RooRealVar* variable,Double_t stsratio,Int_t index); //stsratio -> signal to sideband ratio
  void printResults(std::string prefix="");
  void saveResults(std::string outname);
  //the user may want to change the dataset pointer so they can do
  //various subtractions on subsets of the original dataset...
  void setDataSet(RooDataSet* newData);
  //void fitAndPlotSlice(); //will eventually be side-band subtraction in slices
  //user should have access to these things to play with :)
  RooFitResult* getFitResult();
  std::vector<TH1F> getRawHistos();
  std::vector<TH1F> getSBSHistos();
  Double_t getSTSRatio(); //returns signal-to-sideband ratio
  void resetSBSProducts(); //empties histograms 
};

#endif
