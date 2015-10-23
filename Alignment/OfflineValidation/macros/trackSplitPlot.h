#ifndef TRACKSPLITPLOT_H
#define TRACKSPLITPLOT_H

#include <iostream>
#include <sstream>
#include <cmath>
#include "TCanvas.h"
#include "TClass.h"
#include "TColor.h"
#include "TFile.h"
#include "TGaxis.h"
#include "TGraph2DErrors.h"
#include "TGraphErrors.h"
#include "TH1.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TLegendEntry.h"
#include "TList.h"
#include "TMath.h"
#include "TMultiGraph.h"
#include "TObject.h"
#include "TProfile.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TText.h"
#include "TTree.h"

using namespace std;

enum PlotType {ScatterPlot,Profile,Histogram,OrgHistogram,Resolution};
enum Statistic {Minimum, Maximum, Average, RMS};

const Double_t pi = TMath::Pi();
vector<Color_t> colors;
vector<Style_t> styles;
bool styleset = false;
Int_t minrun = -1;
Int_t maxrun = -1;
const Int_t xsize = 10;
const Int_t ysize = 9;
Int_t legendGrid = 100;
Double_t margin = .1;
Double_t increaseby = .1;
Int_t binsScatterPlotx = 1000;
Int_t binsScatterPloty = 1000;
Int_t binsHistogram = 100;
Int_t runNumberBins = 30;
Int_t binsProfileResolution = 30;    //for everything but runNumber and nHits
                                     //(nHits gets a bin for each integer between the minimum and the maximum)

TString xvariables[xsize]      = {"",   "pt",  "eta", "phi", "dz",  "dxy", "theta", "qoverpt", "runNumber", "nHits"};
TString yvariables[ysize]      = {"pt", "pt",  "eta", "phi", "dz",  "dxy", "theta", "qoverpt", ""};
Bool_t relativearray[ysize]    = {true, false, false, false, false, false, false,   false,     false};

TList *stufftodelete = new TList();

/***********************************
Table Of Contents
0. Track Split Plot
1. Misalignment Dependence
2. Make Plots
3. Axis Label
4. Axis Limits
5. Place Legend
6. TDR Style
***********************************/

#include "trackSplitPlot.h"

//===================
//0. Track Split Plot
//===================

TCanvas *trackSplitPlot(Int_t nFiles,TString *files,TString *names,TString xvar,TString yvar,
                        Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false,
                        TString saveas = "");
TCanvas *trackSplitPlot(Int_t nFiles,TString *files,TString *names,TString var,
                        Bool_t relative = false,Bool_t pull = false,TString saveas = "");
TCanvas *trackSplitPlot(TString file,TString xvar,TString yvar,Bool_t profile = false,
                        Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false,
                        TString saveas = "");
TCanvas *trackSplitPlot(TString file,TString var,
                        Bool_t relative = false,Bool_t pull = false,
                        TString saveas = "");
void placeholder(TString saveas = "",Bool_t wide = false);
void saveplot(TCanvas *c1,TString saveas);
void deleteCanvas(TObject *canvas);
void runNumberZoomed(Int_t nFiles,TString *files,TString *names,TString yvar,
                     Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false,
                     Int_t firstRun = -1,Int_t lastRun = -1,TString saveas = "");

//==========================
//1. Misalignment Dependence
//==========================

void misalignmentDependence(TCanvas *c1old,
                            Int_t nFiles,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TF1 *function,Int_t parameter,TString parametername = "",TString functionname = "",
                            Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false,
                            TString saveas = "");
void misalignmentDependence(TCanvas *c1old,
                            Int_t nFiles,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TF1 *function,Int_t nParameters,Int_t *parameters,TString *parameternames,TString functionname = "",
                            Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false,
                            TString saveas = "");
void misalignmentDependence(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TF1 *function,Int_t parameter,TString parametername = "",TString functionname = "",
                            Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false,
                            TString saveas = "");
void misalignmentDependence(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TF1 *function,Int_t nParameters,Int_t *parameters,TString *parameternames,TString functionname = "",
                            Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false,
                            TString saveas = "");
void misalignmentDependence(TCanvas *c1old,
                            Int_t nFiles,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TString function,Int_t parameter,TString parametername = "",TString functionname = "",
                            Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false,
                            TString saveas = "");
void misalignmentDependence(TCanvas *c1old,
                            Int_t nFiles,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TString function,Int_t nParameters,Int_t *parameters,TString *parameternames,TString functionname = "",
                            Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false,
                            TString saveas = "");
void misalignmentDependence(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TString function,Int_t parameter,TString parametername = "",TString functionname = "",
                            Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false,
                            TString saveas = "");
void misalignmentDependence(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                            TString function,Int_t nParameters,Int_t *parameters,TString *parameternames,TString functionname = "",
                            Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false,
                            TString saveas = "");
Bool_t misalignmentDependence(TCanvas *c1old,
                              Int_t nFiles,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                              Bool_t drawfits = true,
                              Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false,
                              TString saveas = "");
Bool_t misalignmentDependence(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString xvar,TString yvar,
                              Bool_t drawfits = true,
                              Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false,
                              TString saveas = "");
Bool_t hasFit(TString misalignment,TString xvar,TString yvar,Bool_t relative = false,Bool_t resolution = false,Bool_t pull = false);


//=============
//2. Make Plots
//=============

void makePlots(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString directory,Bool_t matrix[xsize][ysize]);
void makePlots(Int_t nFiles,TString *files,TString *names,TString directory, Bool_t matrix[xsize][ysize]);
void makePlots(TString file,TString misalignment,Double_t *values,Double_t *phases,TString directory,Bool_t matrix[xsize][ysize]);
void makePlots(TString file,TString directory,Bool_t matrix[xsize][ysize]);
void makePlots(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString directory,TString xvar,TString yvar);
void makePlots(Int_t nFiles,TString *files,TString *names,TString directory,TString xvar,TString yvar);
void makePlots(TString file,TString misalignment,Double_t *values,Double_t *phases,TString directory,
               TString xvar,TString yvar);
void makePlots(TString file,TString directory,TString xvar,TString yvar);
void makePlots(Int_t nFiles,TString *files,TString *names,TString misalignment,Double_t *values,Double_t *phases,TString directory);
void makePlots(Int_t nFiles,TString *files,TString *names,TString directory);
void makePlots(TString file,TString misalignment,Double_t *values,Double_t *phases,TString directory);
void makePlots(TString file,TString directory);

//=============
//3. Axis Label
//=============

TString fancyname(TString variable);
TString units(TString variable,Char_t axis);
TString axislabel(TString variable, Char_t axis, Bool_t relative = false, Bool_t resolution = false, Bool_t pull = false);
void setAxisLabels(TH1 *p, PlotType type,TString xvar,TString yvar,Bool_t relative,Bool_t pull);
void setAxisLabels(TMultiGraph *p, PlotType type,TString xvar,TString yvar,Bool_t relative,Bool_t pull);
TString nPart(Int_t part,TString string,TString delimit = ";",Bool_t removerest = true);

//==============
//4. Axis Limits
//==============

Double_t findStatistic(Statistic what,Int_t nFiles,TString *files,TString var,Char_t axis,Bool_t relative = false,Bool_t pull = false);
Double_t findAverage(Int_t nFiles,TString *files,TString var,Char_t axis,Bool_t relative = false,Bool_t pull = false);
Double_t findMin(Int_t nFiles,TString *files,TString var,Char_t axis,Bool_t relative = false,Bool_t pull = false);
Double_t findMax(Int_t nFiles,TString *files,TString var,Char_t axis,Bool_t relative = false,Bool_t pull = false);
Double_t findRMS(Int_t nFiles,TString *files,TString var,Char_t axis,Bool_t relative = false,Bool_t pull = false);
Double_t findStatistic(Statistic what,TString file,TString var,Char_t axis,Bool_t relative = false,Bool_t pull = false);
Double_t findAverage(TString file,TString var,Char_t axis,Bool_t relative = false,Bool_t pull = false);
Double_t findMin(TString file,TString var,Char_t axis,Bool_t relative = false,Bool_t pull = false);
Double_t findMax(TString file,TString var,Char_t axis,Bool_t relative = false,Bool_t pull = false);
Double_t findRMS(TString file,TString var,Char_t axis,Bool_t relative = false,Bool_t pull = false);
void axislimits(Int_t nFiles,TString *files,TString var,Char_t axis,Bool_t relative,Bool_t pull,Double_t &min,Double_t &max);

//===============
//5. Place Legend
//===============

Double_t placeLegend(TLegend *l, Double_t width, Double_t height, Double_t x1min, Double_t y1min, Double_t x2max, Double_t y2max);
Bool_t fitsHere(TLegend *l,Double_t x1, Double_t y1, Double_t x2, Double_t y2);

//============
//6. TDR Style
//============

void setTDRStyle();
void set_plot_style();
void setupcolors();

#endif
