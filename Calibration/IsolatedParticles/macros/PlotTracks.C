////////////////////////////////////////////////////////////////////////////////
//
// void plotAll(std::string fname, std::string HLT, int var, int ien, int eta,
//              bool varbin, int rebin, bool approve, bool logy, int pos,
//              bool pv, int savePlot)
// Gets a group of plots by calling plotEnergy or plotEnergyPV for a given
// input file
//   fname = name of the input file                             ["hlt.root"]
//   HLT   = type of HLT used (to be given in the figure legend)["All HLTs"]
//   var   = variable name (-1 means all variables)                 [-1]
//   ien   = energy bin (-1 means all energy bins)                  [-1]
//   eta   = eta bin (-1 means all eta bins)                        [-1]
//   varbin= flag for using variable bin width                      [false]
//   rebin = # of bins to be re-binned together                     [5]
//   approve= If meant for approval talks                           [true]
//   logy  = If y-axis scale shuld by linear/logarithmic            [true]
//   pos   = position of the statistics boxes                       [0]
//   pv    = flag deciding call to plotEnergyPV vs plotEnergy       [false]
//   savePlot= Plot to be saved: no(-1), eps(0), gif(1), pdf(2), C(3) [-1]
//
// void plotEnergyAll(std::string fname, std::string hlt, int pv, int data,
//                    bool varbin, int rebin, bool approve, bool logy, int pos,
//                    int var, int ene, int eta, int savePlot)
// Plots energy distributions for a number of variables, eta and energy bins
//
//   fname = name of the i/p root file                              [""]
//   HLT   = type of HLT used (to be given in the figure legend)  ["All HLTs"]
//   models= packed flag to select which files to be plotted        [15]
//   pv    = selection of # of good PV's used                       [0]
//   data  = flag to say if it is Data (1)/MC (2)/or comparison (4) [4]
//   varbin= flag for using variable bin width                      [false]
//   rebin = # of bins to be re-binned together                     [5]
//   approve= If meant for approval talks                           [true]
//   logy  = If y-axis scale shuld by linear/logarithmic            [true]
//   pos   = position of the statistics boxes                       [0]
//   var   = variable name (-1 means all variables)                 [-1]
//   ene   = energy bin (-1 means all energy bins)                  [-1]
//   eta   = eta bin (-1 means all eta bins)                        [-1]
//   savePlot= Plot to be saved: no(-1), eps(0), gif(1), pdf(2), C(3) [-1]
//
// void plotEMeanAll(int data, int models, bool ratio, bool approve,
//                   std::string postfix, int savePlot)
//
// Plots mean energy response as a function of track momentum or the ratio
//
//   data  = flag to say if it is Data (1)/MC (2)/or comparison (4) [4]
//   models= packed flag to select which files to be plotted        [15]
//   ratio = flag to say if raw mean will be shown or ratio         [false]
//           wrt a reference
//   approve= If meant for approval talks                           [true]
//   postfix= String to be added in the name of saved file          [""]
//   savePlot= Plot to be saved: no(-1), eps(0), gif(1), pdf(2), C(3) [-1]
//
// void plotEMean(std::string fname, std::string hlt, int models, int var,
//                int eta, int pv, int data, bool ratio, bool approve,
//                std::string postfix, int savePlot)
//
// Plots mean energy response as a function of track momentum or the ratio
// MC/Data
//
//   fname = name of the i/p root file                              [""]
//   HLT   = type of HLT used (to be given in the figure legend)   ["All HLTs"]
//   models= packed flag to select which files to be plotted        [15]
//   var   = type of energy reponse with values between 0:5         [0]
//           E_{7x7}/p, H_{3x3}/p, (E_{7x7}+H_{3x3})/p,
//           E_{11x11}/p, H_{5x5}/p, (E_{11x11}+H_{5x5})/p
//   eta   = calorimeter cell where track will reach 0:3            [0]
//           ieta (HCAL) values 1:7, 7-13, 13:17, 17:23
//   pv    = selection of # of good PV's used                       [0]
//   data  = flag to say if it is Data (1)/MC (2)/or comparison (4) [1]
//   ratio = flag to say if raw mean will be shown or ratio wrt     [false]
//           a reference
//   approve= If meant for approval talks                           [true]
//   postfix= String to be added in the name of saved file          [""]
//   savePlot= Plot to be saved: no(-1), eps(0), gif(1), pdf(2), C(3) [-1]
//
// void plotEnergy(std::string fname, std::string HLT, int var, int ien,
//                 int eta, bool varbin, int rebin, bool approve, bool logy,
//                 int pos, int coloff)
//
// Plots energy response distribution measured energy/track momentum for tracks
// of given momentum range in a eta window
//
//   fname = name of the i/p root file                            ["hlt.root"]
//   HLT   = type of HLT used (to be given in the figure legend)  ["All HLTs"]
//   var   = type of energy reponse with values between 0:5       [0]
//           E_{7x7}/p, H_{3x3}/p, (E_{7x7}+H_{3x3})/p,
//           E_{11x11}/p, H_{5x5}/p, (E_{11x11}+H_{5x5})/p
//   ien   = Track momentum range with values between 0:9         [0]
//           1:2,2:3,3:4,4:5,5:6,6:7,7:9,9:11,11:15,15:20
//   eta   = calorimeter cell where track will reach 0:3          [0]
//           ieta (HCAL) values 1:7, 7-13, 13:17, 17:23
//   varbin= flag for using variable bin width                    [false]
//   rebin = # of bins to be re-binned together                   [1]
//   approve= If meant for approval talks                         [false]
//   logy  = If y-axis scale shuld by linear/logarithmic          [true]
//   pos   = position of the statistics boxes                     [0]
//   coloff= color offset                                         [0]
//
// TCanvas* plotEnergyPV(std::string fname, std::string HLT, int var, int ien,
//                       int eta, bool varbin, int rebin, bool approve,
//                       bool logy, int pos)
//
// Plots energy response distribution measured energy/track momentum for tracks
// of given momentum range in a eta window for 4 different selections of # of
// primary vertex 1:2,2:3,3:5,5:100
//
//   fname = name of the i/p root file            ["StudyHLT_HLTZeroBias.root"]
//   HLT   = type of HLT used (to be given in the figure legend) ["Zero Bias"]
//   var   = type of energy reponse with values between 0:5      [0]
//           E_{7x7}/p, H_{3x3}/p, (E_{7x7}+H_{3x3})/p,
//           E_{11x11}/p, H_{5x5}/p, (E_{11x11}+H_{5x5})/p
//   ien   = Track momentum range with values between 0:9        [0]
//           1:2,2:3,3:4,4:5,5:6,6:7,7:9,9:11,11:15,15:20
//   eta   = calorimeter cell where track will reach 0:3         [0]
//           ieta (HCAL) values 1:7, 7-13, 13:17, 17:23
//   varbin= flag for using variable bin width                   [false]
//   rebin = # of bins to be re-binned together                  [1]
//   approve= If meant for approval talks                        [false]
//   logy  = If y-axis scale shuld by linear/logarithmic         [true]
//   pos   = position of the statistics boxes                    [0]
//
// void plotTrack(std::string fname, std::string HLT, int var, bool varbin,
//                int rebin, bool approve, bool logy, int pos)
//
// Plots kinematic propeties of the track
//
//   fname = name of the i/p root file                           ["hlt.root"]
//   HLT   = type of HLT used (to be given in the figure legend  ["All HLTs"]
//   var   = kinematic variable 0:3 --> p, pt, eta, phi          [0]
//   varbin= flag for using variable bin width                   [false]
//   rebin = # of bins to be re-binned together                  [1]
//   approve= If meant for approval talks                        [false]
//   logy  = If y-axis scale shuld by linear/logarithmic         [true]
//   pos   = position of the statistics boxes                    [0]
//
// void plotIsolation(std::string fname, std::string HLT, int var, bool varbin,
//                    int rebin, bool approve, bool logy, int pos)
//
// Plots variables used for deciding on track isolation
//
//   fname = name of the i/p root file                           ["hlt.root"]
//   HLT   = type of HLT used (to be given in the figure legend  ["All HLTs"]
//   var   = isolation variable 0:3 --> Charge isolation energy, [0]
//	   Neutral isolation energy, Energy in smaller cone,
//	   Energy in larger cone
//   varbin= flag for using variable bin width                   [false]
//   rebin = # of bins to be re-binned together                  [1]
//   approve= If meant for approval talks                        [false]
//   logy  = If y-axis scale shuld by linear/logarithmic         [true]
//   pos   = position of the statistics boxes                    [0]
//
// void plotHLT(std::string fname, std::string HLT, int run, bool varbin,
//                    int rebin, bool approve, bool logy, int pos)
//
// Plots HLT accept information for a given run or a summary
//
//   fname = name of the i/p root file                           ["hlt.root"]
//   HLT   = type of HLT used (to be given in the figure legend  ["All HLTs"]
//   run   = run number; if <=0 the overall summary              [-1]
//   varbin= flag for using variable bin width                   [false]
//   rebin = # of bins to be re-binned together                  [1]
//   approve= If meant for approval talks                        [false]
//   logy  = If y-axis scale shuld by linear/logarithmic         [true]
//   pos   = position of the statistics boxes                    [0]
//
// void plotEMeanRatioXAll(std::string fname=, std::string hlt, std::string postfix="F", int savePlot=2);
//
//   fname    = name of the i/p root file       ["pikp/FBE4p3vMixStudyHLT.root"]
//   hlt      = name of the tag for the file    ["10.4p03 FTFP_BERT_EMM"]
//   postfix  = string fixed to the saved file  ["F"]
//   savePlot = flag to save the canvas         [2]
//
////////////////////////////////////////////////////////////////////////////////

#include "TCanvas.h"
#include "TDirectory.h"
#include "TF1.h"
#include "TFile.h"
#include "TFitResult.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TH1D.h"
#include "TH2D.h"
#include "THStack.h"
#include "TLegend.h"
#include "TMath.h"
#include "TPolyLine.h"
#include "TProfile.h"
#include "TPaveStats.h"
#include "TPaveText.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

//const int nmodelm=1, nmodelx=2, nmodels=3;
//const int nmodelm=15, nmodelx=16, nmodels=2;
//const int nmodelm=3, nmodelx=4, nmodels=3;
//const int nmodelm=4, nmodelx=5, nmodels=3;
//const int nmodelm=4, nmodelx=5, nmodels=2;
//const int nmodelm=2, nmodelx=3, nmodels=3;
const int nmodelm = 5, nmodelx = 6, nmodels = 3;
const int etaMax = 3;
//int         styles[7]  = {20, 23, 22, 24, 25, 21, 33};
//int         colors[7]  = {1, 2, 4, 6, 7, 46, 38};
int styles[7] = {20, 23, 21, 22, 24, 25, 33};
int colors[7] = {1, 2, 6, 4, 3, 7, 38};
int lstyle[7] = {1, 2, 3, 4, 5, 6, 7};
std::string names[7] = {"All", "Quality", "okEcal", "EcalCharIso", "HcalCharIso", "EcalNeutIso", "HcalNeutIso"};
std::string namefull[7] = {"All tracks",
                           "Good quality tracks",
                           "Tracks reaching ECAL",
                           "Charge isolation in ECAL",
                           "Charge isolation in HCAL",
                           "Isolated in ECAL",
                           "Isolated in HCAL"};
std::string nameEta[4] = {"i#eta 1:6", "i#eta 7:12", "i#eta 13:16", "i#eta 17:22"};
std::string nameEtas[4] = {"|#eta| < 0.52", "0.52 < |#eta| < 1.04", "1.04 < |#eta| < 1.39", "1.39 < |#eta| < 2.01"};
std::string namePV[5] = {"all PV", "PV 1:1", "PV 2:2", "PV 3:5", "PV > 5"};
std::string varname[4] = {"p", "pt", "eta", "phi"};
std::string vartitle[4] = {"p (GeV/c)", "p_{T} (GeV/c)", "#eta", "#phi"};
std::string nameC[2] = {"Ecal", "Hcal"};
std::string nameCF[2] = {"ECAL", "HCAL"};
std::string varnameC[4] = {"maxNearP", "ediff", "ene1", "ene2"};
std::string vartitlC[4] = {
    "Charge isolation energy", "Neutral isolation energy", "Energy in smaller cone", "Energy in larger cone"};
std::string cmsP = "CMS Preliminary";
//std::string cmsP = "CMS";
std::string fileData = "AllDataStudyHLT.root";
const int NPT = 10;
double mom[NPT] = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.0, 10.0, 13.0, 17.5};
double dmom[NPT] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 2.5};
std::string varPs[NPT] = {"1:2", "2:3", "3:4", "4:5", "5:6", "6:7", "7:9", "9:11", "11:15", "15:20"};
std::string varPs1[NPT] = {"1", "2", "3", "4", "5", "6", "7", "9", "11", "15"};
std::string varPPs[NPT] = {
    "1-2 GeV", "2-3 GeV", "3-4 GeV", "4-5 GeV", "5-6 GeV", "6-7 GeV", "7-9 GeV", "9-11 GeV", "11-15 GeV", "15-20 GeV"};
std::string varEta[4] = {"1:6", "7:12", "13:16", "17:23"};
std::string varEta1[4] = {"1", "2", "3", "4"};
std::string varEne[6] = {"E_{7x7}", "H_{3x3}", "(E_{7x7}+H_{3x3})", "E_{11x11}", "H_{5x5}", "(E_{11x11}+H_{5x5})"};
std::string varEne1[6] = {"E7x7", "H3x3", "E7x7H3x3", "E11x11", "H5x5", "E11x11H5x5"};
const int nbins = 100;
double xbins[nbins + 1] = {0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14,
                           0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29,
                           0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44,
                           0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68,
                           0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98,
                           1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70,
                           1.75, 1.80, 1.85, 1.90, 1.95, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50};
int ibins[nbins + 1] = {11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
                        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,
                        45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
                        63,  65,  67,  69,  71,  73,  75,  77,  79,  81,  83,  85,  87,  89,  91,  93,  95,
                        97,  99,  101, 103, 105, 107, 109, 111, 116, 121, 126, 131, 136, 141, 146, 151, 156,
                        161, 166, 171, 176, 181, 186, 191, 196, 201, 206, 211, 221, 231, 241, 251, 261};
/*
std::string files[nmodels]={"ZeroBiasStudyHLT.root","MinimumBiasStudyHLT.root"};
std::string types[nmodels]={"Zero Bias Data","Minimum Bias Data"};
*/
std::string files[nmodels] = {"2017C_ZB.root", "2017G_ZB.root", "2017H_ZB.root"};
std::string types[nmodels] = {"Zero Bias (2017C)", "Zero Bias (2017G)", "Zero Bias (2017H)"};

/*
std::string filem[nmodelm]={"pikp/FBE2p2StudyHLT.root",
			    "pikp/FBE3p3MixStudyHLT.root",
			    "pikp/FBE4bMixStudyHLT.root",
			    "pikp/FBE4r00MixStudyHLT.root",
			    "pikp/FBE4c01MixStudyHLT.root",
			    "pikp/FBE4c02MixStudyHLT.root",
			    "pikp/FBE3r11MixStudyHLT.root",
			    "pikp/FBE3r10MixStudyHLT.root",
			    "pikp/FBE3r8MixStudyHLT.root",
			    "pikp/FBE3r9MixStudyHLT.root",
			    "pikp/QFBE0p2StudyHLT.root",
			    "pikp/QFBE2p2StudyHLT.root",
			    "pikp/QFBE4bMixStudyHLT.root",
			    "pikp/FBAE2p2StudyHLT.root",
			    "pikp/FBAE4bMixStudyHLT.root"};
std::string typem[nmodelm]={"10.2.p02 FTFP_BERT_EMM",
			    "10.3.p03 FTFP_BERT_EMM",
			    "10.4.beta FTFP_BERT_EMM",
			    "10.4 FTFP_BERT_EMM",
			    "10.4.cand01 FTFP_BERT_EMM",
			    "10.4.cand02 FTFP_BERT_EMM",
			    "10.3.ref11 FTFP_BERT_EMM",
			    "10.3.ref10 FTFP_BERT_EMM",
			    "10.3.ref08 FTFP_BERT_EMM",
			    "10.3.ref09 FTFP_BERT_EMM",
			    "10.0.p02 QGSP_FTFP_BERT_EML",
			    "10.2.p02 QGSP_FTFP_BERT_EMM",
			    "10.4.b01 QGSP_FTFP_BERT_EMM",
			    "10.2.p02 FTFP_BERT_ATL_EMM",
			    "10.4.b01 FTFP_BERT_ATL_EMM"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE2p2MixStudyHLT10.root",
			    "pikp/FBE4MixStudyHLT10.root",
			    "pikp/FBE4vMixStudyHLT10.root",
			    "pikp/FBE4vcMixStudyHLT10.root"};
std::string typem[nmodelm]={"10.2.p02 FTFP_BERT_EMM",
			    "10.4 FTFP_BERT_EMM",
			    "10.4 VecGeom FTFP_BERT_EMM",
			    "10.4 VecGeom+CLHEP FTFP_BERT_EMM"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE4cMixStudyHLT10.root",
			    "pikp/FBE4vcMixStudyHLT10.root"};
std::string typem[nmodelm]={"10.4 CLHEP FTFP_BERT_EMM",
			    "10.4 VecGeom+CLHEP FTFP_BERT_EMM"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE3r9MixStudyHLT.root",
			    "pikp/FBE4r00MixStudyHLT.root",
			    "pikp/FBE4r01MixStudyHLT.root"};
std::string typem[nmodelm]={"10.3.ref09 FTFP_BERT_EMM",
			    "10.4.ref00 FTFP_BERT_EMM",
			    "10.4.ref01 FTFP_BERT_EMM"};
*/
/*
std::string filem[nmodelm]={"pikp/QFB3r9MixStudyHLT.root",
			    "pikp/QFB4r00MixStudyHLT.root",
			    "pikp/QFB4r01MixStudyHLT.root"};
std::string typem[nmodelm]={"10.3.ref09 QGSP_FTFP_BERT_EML",
			    "10.4.ref00 QGSP_FTFP_BERT_EML",
			    "10.4.ref01 QGSP_FTFP_BERT_EML"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE2p2StudyHLT.root",
			    "pikp/FBE4r00MixStudyHLT.root",
			    "pikp/FBE4r00vMixStudyHLT.root"};
std::string typem[nmodelm]={"10.2.p02 FTFP_BERT_EMM",
			    "10.4 FTFP_BERT_EMM (Native)",
			    "10.4 FTFP_BERT_EMM (VecGeom)"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE3r6MixStudyHLT.root",
			    "pikp/FBE3r6vMixStudyHLT.root",
			    "pikp/FBE3r6vRMixStudyHLT.root"};
std::string typem[nmodelm]={"10.3.ref06 FTFP_BERT_EMM (Native)",
			    "10.3.ref06 FTFP_BERT_EMM (VecGeom 4)",
			    "10.3.ref06 FTFP_BERT_EMM (VecGeom Corr)"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE4r00vMixStudyHLT.root",
			    "pikp/FBE4r01MixStudyHLT.root",
			    "pikp/FBE4r01vMixStudyHLT.root"};
std::string typem[nmodelm]={"10.4 VecGeom FTFP_BERT_EMM",
			    "10.4.ref01 FTFP_BERT_EMM",
			    "10.4.ref01 VecGeom FTFP_BERT_EMM"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE4MixStudyHLT10d.root",
			    "pikp/FBE4MixStudyHLT10s.root",
			    "pikp/FBE4MixStudyHLT10t.root",
			    "pikp/FBE4MixStudyHLT10st.root"};
std::string typem[nmodelm]={"10.4 FTFP_BERT_EMM (Default)",
			    "10.4 FTFP_BERT_EMM (New StepIntegrator)",
			    "10.4 FTFP_BERT_EMM (Smart Track)",
			    "10.4 FTFP_BERT_EMM (New Stepper + Smart Track)"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE2p2StudyHLT.root",
			    "pikp/FBE4MixStudyHLT1025.root",
			    "pikp/FBE5c01vMixStudyHLT.root",
			    "pikp/FBE5c02vMixStudyHLT.root",
			    "pikp/QFB5c02vMixStudyHLT.root"};
std::string typem[nmodelm]={"10.2.p02 FTFP_BERT_EMM",
			    "10.4 FTFP_BERT_EMM",
			    "10.5.cand01 FTFP_BERT_EMM",
			    "10.5.cand02 FTFP_BERT_EMM",
			    "10.5.cand02 QGSP_FTFP_BERT_EML"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE4p3vMixStudyHLT.root",
			    "pikp/FBE5r00vMixStudyHLT.root",
			    "pikp/FBE5r05vMixStudyHLT.root",
			    "pikp/FBE5r07vMixStudyHLT.root",
			    "pikp/FBE5r08vMixStudyHLT.root"};
std::string typem[nmodelm]={"10.4.p03 FTFP_BERT_EMM",
			    "10.5 FTFP_BERT_EMM",
			    "10.5.ref05 FTFP_BERT_EMM",
			    "10.5.ref07 FTFP_BERT_EMM",
			    "10.5.ref08 FTFP_BERT_EMM"};
*/
/*
std::string filem[nmodelm]={"pikp/QFB4p3vMixStudyHLT.root",
			    "pikp/QFB5r00vMixStudyHLT.root",
			    "pikp/QFB5r05vMixStudyHLT.root",
			    "pikp/QFB5r07vMixStudyHLT.root",
			    "pikp/QFB5r08vMixStudyHLT.root"};
std::string typem[nmodelm]={"10.4.p03 QGSP_FTFP_BERT_EML",
			    "10.5 QGSP_FTFP_BERT_EML",
			    "10.5.ref05 QGSP_FTFP_BERT_EML",
			    "10.5.ref07 QGSP_FTFP_BERT_EML",
			    "10.5.ref08 QGSP_FTFP_BERT_EML"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE2p2StudyHLT.root",
			    "pikp/FBE4p3vMixStudyHLT.root",
			    "pikp/FBE6p00vMixStudyHLT.root",
			    "pikp/FBE6r01vMixStudyHLT.root"};
std::string typem[nmodelm]={"10.2.p02 FTFP_BERT_EMM",
			    "10.4.p03 FTFP_BERT_EMM",
			    "10.6.ref00 FTFP_BERT_EMM + Birk",
			    "10.6.ref01 FTFP_BERT_EMM + Birk"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE4p3vMixStudyHLT.root",
			    "pikp/FBE6p01AvMixStudyHLT.root",
			    "pikp/FBE6p01BvMixStudyHLT.root",
			    "pikp/FBE6p01CvMixStudyHLT.root"};
std::string typem[nmodelm]={"10.4.p03 FTFP_BERT_EMM",
			    "10.6.p01 FTFP_BERT_EMM + Birk C1+C3",
			    "10.6.p01 FTFP_BERT_EMM + Birk C1",
			    "10.6.p01 FTFP_BERT_EMM + Birk C1 + #pi"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE4p3vMixStudyHLT.root",
			    "pikp/FBE6p01CvMixStudyHLT.root"};
std::string typem[nmodelm]={"10.4.p03 FTFP_BERT_EMM",
			    "10.6.p01 FTFP_BERT_EMM + Birk C1 + #pi"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE4p3vMixStudyHLT.root",
			    "pikp/FBE5r08vMixStudyHLT.root"};
std::string typem[nmodelm]={"10.4.p03 FTFP_BERT_EMM",
			    "10.5.ref08 FTFP_BERT_EMM"};
*/
/*
std::string filem[nmodelm]={"pikp/QFBE2p2StudyHLT.root",
			    "pikp/QFB4p3vMixStudyHLT.root",
			    "pikp/QFB6p00vMixStudyHLT.root",
			    "pikp/QFB6r01vMixStudyHLT.root"};
std::string typem[nmodelm]={"10.2.p02 QGSP_FTFP_BERT_EMM",
			    "10.4.p03 QGSP_FTFP_BERT_EML",
			    "10.6.ref00 QGSP_FTFP_BERT_EML + Birk",
			    "10.6.ref01 QGSP_FTFP_BERT_EML + Birk"};
*/
/*
std::string filem[nmodelm]={"pikp/QFB4p3vMixStudyHLT.root",
			    "pikp/QFB6p01AvMixStudyHLT.root",
			    "pikp/QFB6p01BvMixStudyHLT.root",
			    "pikp/QFB6p01CvMixStudyHLT.root"};
std::string typem[nmodelm]={"10.4.p03 QGSP_FTFP_BERT_EML",
			    "10.6.p01 QGSP_FTFP_BERT_EML + Birk C1+C3",
			    "10.6.p01 QGSP_FTFP_BERT_EML + Birk C1",
			    "10.6.p01 QGSP_FTFP_BERT_EML + Birk C1 + #pi"};
*/
std::string filem[nmodelm] = {"pikp/FBE4p3vMixStudyHLT.root",
                              "pikp/FBE7p02vMixStudyHLT.root",
                              "pikp/FBE110p01vMixStudyHLT.root",
                              "pikp/FBE110r05MixStudyHLT.root",
                              "pikp/FBE110r06vMixStudyHLT.root"};
std::string typem[nmodelm] = {"10.4.p03 FTFP_BERT_EMM",
                              "10.7.p02 FTFP_BERT_EMM + Birk C1 + #pi",
                              "11.0.p01 FTFP_BERT_EMM + Birk C1 + #pi",
                              "11.0.ref05 FTFP_BERT_EMM + Birk C1 + #pi",
                              "11.0.ref06 FTFP_BERT_EMM + Birk C1 + #pi"};
/*
std::string filem[nmodelm]={"pikp/QFB4p3vMixStudyHLT.root",
			    "pikp/QFB7p02vMixStudyHLT.root",
			    "pikp/QFB110p01vMixStudyHLT.root",
			    "pikp/QFB110r05MixStudyHLT.root",
			    "pikp/QFB110r06vMixStudyHLT.root"};
std::string typem[nmodelm]={"10.4.p03 QGSP_FTFP_BERT_EML",
			    "10.7.p02 QGSP_FTFP_BERT_EML + Birk C1",
			    "11.0.p01 QGSP_FTFP_BERT_EML + Birk C1",
			    "11.0.ref05 QGSP_FTFP_BERT_EML + Birk C1",
			    "11.0.ref06 QGSP_FTFP_BERT_EML + Birk C1"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE7r00vMixStudyHLT.root",
			    "pikp/FBE7p01MixStudyHLT.root",
			    "pikp/FBE7p02vMixStudyHLT.root",
			    "pikp/FBE7p03vMixStudyHLT.root"};
std::string typem[nmodelm]={"10.7 FTFP_BERT_EMM + Birk C1 + #pi",
			    "10.7.p01 FTFP_BERT_EMM + Birk C1 + #pi",
			    "10.7.p02 FTFP_BERT_EMM + Birk C1 + #pi",
			    "10.7.p03 FTFP_BERT_EMM + Birk C1 + #pi"};
*/
/*
std::string filem[nmodelm]={"pikp/QFB7r00vMixStudyHLT.root",
			    "pikp/QFB7p01MixStudyHLT.root",
			    "pikp/QFB7p02vMixStudyHLT.root",
			    "pikp/QFB7p03vMixStudyHLT.root"};
std::string typem[nmodelm]={"10.7 QGSP_BERT_EML + Birk C1 + #pi",
			    "10.7.p01 QGSP_BERT_EML + Birk C1 + #pi",
			    "10.7.p02 QGSP_BERT_EML + Birk C1 + #pi",
			    "10.7.p03 QGSP_BERT_EML + Birk C1 + #pi"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE4p3vMixStudyHLT.root",
			    "pikp/FBE6p2vMixStudyHLT.root",
			    "pikp/FBE7p1vMixStudyHLT.root"};
std::string typem[nmodelm]={"10.4p03 FTFP_BERT_EMM",
			    "10.6p02 FTFP_BERT_EMM",
			    "10.7p01 FTFP_BERT_EMM"};
*/
/*
std::string filem[nmodelm]={"pikp/FBE4p3vMixStudyHLT.root"};
std::string typem[nmodelm]={"10.4p03 FTFP_BERT_EMM"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE2p2StudyHLT.root",
			    "pikp/FBE3p3MixStudyHLT.root",
			    "pikp/FBE4bMixStudyHLT.root",
			    "pikp/FBE4r00MixStudyHLT.root",
			    "pikp/FBE4c01MixStudyHLT.root",
			    "pikp/FBE4c02MixStudyHLT.root",
			    "pikp/FBE3r11MixStudyHLT.root",
			    "pikp/FBE3r10MixStudyHLT.root",
			    "pikp/FBE3r8MixStudyHLT.root",
			    "pikp/FBE3r9MixStudyHLT.root",
			    "pikp/QFBE0p2StudyHLT.root",
			    "pikp/QFBE2p2StudyHLT.root",
			    "pikp/QFBE4bMixStudyHLT.root",
			    "pikp/FBAE2p2StudyHLT.root",
			    "pikp/FBAE4bMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.2.p02 FTFP_BERT_EMM",
			    "10.3.p03 FTFP_BERT_EMM",
			    "10.4.beta FTFP_BERT_EMM",
			    "10.4 FTFP_BERT_EMM",
			    "10.4.cand01 FTFP_BERT_EMM",
			    "10.4.cand02 FTFP_BERT_EMM",
			    "10.3.ref11 FTFP_BERT_EMM",
			    "10.3.ref10 FTFP_BERT_EMM",
			    "10.3.ref08 FTFP_BERT_EMM",
			    "10.3.ref09 FTFP_BERT_EMM",
			    "10.0.p02 QGSP_FTFP_BERT_EML",
			    "10.2.p02 QGSP_FTFP_BERT_EMM",
			    "10.4.b01 QGSP_FTFP_BERT_EMM",
			    "10.2.p02 FTFP_BERT_ATL_EMM",
			    "10.4.b01 FTFP_BERT_ATL_EMM"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE2p2MixStudyHLT10.root",
			    "pikp/FBE4MixStudyHLT10.root",
			    "pikp/FBE4vMixStudyHLT10.root",
			    "pikp/FBE4vcMixStudyHLT10.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.2.p02 FTFP_BERT_EMM",
			    "10.4 FTFP_BERT_EMM",
			    "10.4 VecGeom FTFP_BERT_EMM",
			    "10.4 VecGeom+CLHEP FTFP_BERT_EMM"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
                            "pikp/FBE4cMixStudyHLT10.root",
			    "pikp/FBE4vcMixStudyHLT10.root"};
std::string typex[nmodelx]={"Data (2016B)",
                            "10.4 CLHEP FTFP_BERT_EMM",
			    "10.4 VecGeom+CLHEP FTFP_BERT_EMM"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE3r9MixStudyHLT.root",
			    "pikp/FBE4r00MixStudyHLT.root",
			    "pikp/FBE4r01MixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.3.ref09 FTFP_BERT_EMM",
			    "10.4.ref00 FTFP_BERT_EMM",
			    "10.4.ref01 FTFP_BERT_EMM"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/QFB3r9MixStudyHLT.root",
			    "pikp/QFB4r00MixStudyHLT.root",
			    "pikp/QFB4r01MixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.3.ref09 QGSP_FTFP_BERT_EML",
			    "10.4.ref00 QGSP_FTFP_BERT_EML",
			    "10.4.ref01 QGSP_FTFP_BERT_EML"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE2p2StudyHLT.root",
			    "pikp/FBE4r00MixStudyHLT.root",
			    "pikp/FBE4r00vMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.2.p02 FTFP_BERT_EMM",
			    "10.4 FTFP_BERT_EMM (Native)",
			    "10.4 FTFP_BERT_EMM (VecGeom)"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE3r6MixStudyHLT.root",
			    "pikp/FBE3r6vMixStudyHLT.root",
			    "pikp/FBE3r6vRMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.3.ref06 FTFP_BERT_EMM (Native)",
			    "10.3.ref06 FTFP_BERT_EMM (VecGeom 4)",
			    "10.3.ref06 FTFP_BERT_EMM (VecGeom Corr)"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE4r00vMixStudyHLT.root",
			    "pikp/FBE4r01MixStudyHLT.root",
			    "pikp/FBE4r01vMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.4 VecGeom FTFP_BERT_EMM",
			    "10.4.ref01 FTFP_BERT_EMM",
			    "10.4.ref01 VecGeom FTFP_BERT_EMM"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE4MixStudyHLT10d.root",
			    "pikp/FBE4MixStudyHLT10s.root",
			    "pikp/FBE4MixStudyHLT10t.root",
			    "pikp/FBE4MixStudyHLT10st.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.4 FTFP_BERT_EMM (Default)",
			    "10.4 FTFP_BERT_EMM (New StepIntegrator)",
			    "10.4 FTFP_BERT_EMM (Smart Track)",
			    "10.4 FTFP_BERT_EMM (New Stepper + Smart Track)"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE2p2StudyHLT.root",
			    "pikp/FBE4MixStudyHLT1025.root",
			    "pikp/FBE5c01vMixStudyHLT.root",
			    "pikp/FBE5c02vMixStudyHLT.root",
			    "pikp/QFB5c02vMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.2.p02 FTFP_BERT_EMM",
			    "10.4 FTFP_BERT_EMM",
			    "10.5.cand01 FTFP_BERT_EMM",
			    "10.5.cand02 FTFP_BERT_EMM",
			    "10.5.cand02 QGSP_FTFP_BERT_EML"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE4p3vMixStudyHLT.root",
			    "pikp/FBE5r00vMixStudyHLT.root",
			    "pikp/FBE5r05vMixStudyHLT.root",
			    "pikp/FBE5r07vMixStudyHLT.root",
			    "pikp/FBE5r08vMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.4.p03 FTFP_BERT_EMM",
			    "10.5 FTFP_BERT_EMM",
			    "10.5.ref05 FTFP_BERT_EMM",
			    "10.5.ref07 FTFP_BERT_EMM",
			    "10.5.ref08 FTFP_BERT_EMM"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/QFB4p3vMixStudyHLT.root",
			    "pikp/QFB5r00vMixStudyHLT.root",
			    "pikp/QFB5r05vMixStudyHLT.root",
			    "pikp/QFB5r07vMixStudyHLT.root",
			    "pikp/QFB5r08vMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.4.p03  QGSP_FTFP_BERT_EML",
			    "10.5  QGSP_FTFP_BERT_EML",
			    "10.5.ref05 QGSP_FTFP_BERT_EML",
			    "10.5.ref07 QGSP_FTFP_BERT_EML",
			    "10.5.ref08 QGSP_FTFP_BERT_EML"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE4p3vMixStudyHLT.root",
			    "pikp/FBE5r10vMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.4.p03 FTFP_BERT_EMM",
			    "10.5.ref10 FTFP_BERT_EMM"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE2p2StudyHLT.root",
			    "pikp/FBE4p3vMixStudyHLT.root",
			    "pikp/FBE6p00vMixStudyHLT.root",
			    "pikp/FBE6r01vMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.2.p02 FTFP_BERT_EMM",
			    "10.4.p03 FTFP_BERT_EMM",
			    "10.6.ref00 FTFP_BERT_EMM + Birk",
			    "10.6.ref01 FTFP_BERT_EMM + Birk"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE4p3vMixStudyHLT.root",
			    "pikp/FBE6p01AvMixStudyHLT.root",
			    "pikp/FBE6p01BvMixStudyHLT.root",
			    "pikp/FBE6p01CvMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.4.p03 FTFP_BERT_EMM",
			    "10.6.p01 FTFP_BERT_EMM + Birk C1+C3",
			    "10.6.p01 FTFP_BERT_EMM + Birk C1",
			    "10.6.p01 FTFP_BERT_EMM + Birk C1 + #pi"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE4p3vMixStudyHLT.root",
			    "pikp/FBE6p01CvMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.4.p03 FTFP_BERT_EMM",
			    "10.6.p01 FTFP_BERT_EMM + Birk C1 + #pi"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/QFB4p3vMixStudyHLT.root",
			    "pikp/QFB5r00vMixStudyHLT.root",
			    "pikp/QFB6p00vMixStudyHLT.root",
			    "pikp/QFB6r01vMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.4.p03 QGSP_FTFP_BERT_EML",
			    "10.5     QGSP_FTFP_BERT_EML",
			    "10.6.ref00 QGSP_FTFP_BERT_EML + Birk",
			    "10.6.ref01 QGSP_FTFP_BERT_EML + Birk"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/QFB4p3vMixStudyHLT.root",
			    "pikp/QFB6p01AvMixStudyHLT.root",
			    "pikp/QFB6p01BvMixStudyHLT.root",
			    "pikp/QFB6p01CvMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.4.p03 QGSP_FTFP_BERT_EML",
			    "10.6.p01 QGSP_FTFP_BERT_EML + Birk C1+C3",
			    "10.6.p01 QGSP_FTFP_BERT_EML + Birk C1",
			    "10.6.p01 QGSP_FTFP_BERT_EML + Birk C1 + #pi"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE4p3vMixStudyHLT.root",
			    "pikp/FBE7p02vMixStudyHLT.root",
			    "pikp/FBE110p01vMixStudyHLT.root",
			    "pikp/FBE110r05MixStudyHLT.root",
			    "pikp/FBE110r06vMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.4.p03 FTFP_BERT_EMM",
			    "10.7.p02 FTFP_BERT_EMM",
			    "11.0.p01 FTFP_BERT_EMM",
			    "11.0.ref05 FTFP_BERT_EMM",
			    "11.0.ref06 FTFP_BERT_EMM"};
*/
std::string filex[nmodelx] = {"AllDataStudyHLT.root",
                              "pikp/QFB4p3vMixStudyHLT.root",
                              "pikp/QFB7p02vMixStudyHLT.root",
                              "pikp/QFB110p01vMixStudyHLT.root",
                              "pikp/QFB110r05MixStudyHLT.root",
                              "pikp/QFB110r06vMixStudyHLT.root"};
std::string typex[nmodelx] = {"Data (2016B)",
                              "10.4.p03 QGSP_FTFP_BERT_EML",
                              "10.7.p02 QGSP_FTFP_BERT_EML",
                              "11.0.p01 QGSP_FTFP_BERT_EML",
                              "11.0.ref05 QGSP_FTFP_BERT_EML",
                              "11.0.ref06 QGSP_FTFP_BERT_EML"};
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/QFB110p01vMixStudyHLT.root",
			    "pikp/QFB110r01vMixStudyHLT.root",
			    "pikp/QFB110r02vMixStudyHLT.root",
			    "pikp/QFB110r03MixStudyHLT.root",
			    "pikp/QFB110r04MixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "11.0.p01 QGSP_BERT_EMM + Birk C1 + #pi",
			    "11.0.ref01 QGSP_BERT_EMM + Birk C1 + #pi",
			    "11.0.ref02 QGSP_BERT_EMM + Birk C1 + #pi",
			    "11.0.ref03 QGSP_BERT_EMM + Birk C1 + #pi",
			    "11.0.ref04 QGSP_BERT_EMM + Birk C1 + #pi"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/QFB7r00vMixStudyHLT.root",
			    "pikp/QFB7p01vMixStudyHLT.root",
			    "pikp/QFB7p02vMixStudyHLT.root",
			    "pikp/QFB7p03vMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.7 QGSP_BERT_EML + Birk C1 + #pi",
			    "10.7.p01 QGSP_BERT_EML + Birk C1 + #pi",
			    "10.7.p02 QGSP_BERT_EML + Birk C1 + #pi",
			    "10.7.p03 QGSP_BERT_EML + Birk C1 + #pi"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE1250p30vMixStudyHLT.root",
			    "pikp/FBE1250p31vMixStudyHLT.root",
			    "pikp/FBE1250p32vMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.7.p02 FTFP_BERT_EMM Default",
			    "10.7.p02 FTFP_BERT_EMM Emax = 0.02",
			    "10.7.p02 FTFP_BERT_EMM Emax = 0.04"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE4p3vMixStudyHLT.root",
			    "pikp/FBE6p2MixStudyHLT.root",
			    "pikp/FBE7p1MixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "Geant4.10.4p03",
			    "Geant4.10.6p02",
			    "Geant4.10.7p01"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/FBE4p3vMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "Geant4.10.4p03"};
*/
/*
std::string filex[nmodelx]={"AllDataStudyHLT.root",
			    "pikp/QFB4vp3vMixStudyHLT.root",
			    "pikp/QFB6vp2vMixStudyHLT.root",
			    "pikp/QFB7vb1vMixStudyHLT.root"};
std::string typex[nmodelx]={"Data (2016B)",
			    "10.4.p03 QGSP_FTFP_BERT_EML",
			    "10.6.p02 QGSP_FTFP_BERT_EML",
			    "10.7.beta QGSP_FTFP_BERT_EML"};
*/
/*
  std::string files[nmodelx]={"StudyHLT_ZeroBias_1PV.root","StudyHLT_PixelTrack_1PV.root","StudyHLT_1PV.root"};
  std::string types[nmodels]={"Zero Bias HLT","Pixel Track HLT","All HLTs"};
  std::string files[nmodels]={"StudyHLT_HLTZeroBias.root","StudyHLT_PixelTrack.root","StudyHLT_HLTJetE.root","StudyHLT_HLTPhysics.root","StudyHLT_All.root"};
  std::string types[nmodels]={"Zero Bias HLT","Pixel Track HLT","JetE HLT","Physics HLT","All HLTs"};
  std::string filem[nmodelm]={"StudyHLT_95p02_QGSP_FTFP_BERT.root", "StudyHLT_96p02_QGSP_FTFP_BERT.root", "StudyHLT_96p02_FTFP_BERT.root"};
  std::string typem[nmodelm]={"Pythia8 (9.5.p02 QGSP_FTFP_BERT)","Pythia8 (9.6.p02 QGSP_FTFP_BERT)","Pythia8 (9.6.p02 FTFP_BERT)"};
  std::string filex[nmodelx]={"AllDataStudyHLT.root","pikp/QFBE0p2StudyHLT.root", "pikp/FBE2p2StudyHLT.root"};
  std::string typex[nmodelx]={"Data (2016B)","10.0.p02 QGSP_FTFP_BERT_EML","10.2.p02 FTFP_BERT_EMM"};
*/

void plotAll(std::string fname = "",
             std::string HLT = "",
             int var = -1,
             int ien = -1,
             int eta = -1,
             bool varbin = false,
             int rebin = 1,
             bool approve = true,
             bool logy = true,
             int pos = 0,
             bool pv = false,
             int savePlot = -1);
void plotEnergyAll(std::string fname = "",
                   std::string hlt = "All HLTs",
                   int models = 15,
                   int pv = 0,
                   int data = 4,
                   bool varbin = false,
                   int rebin = 5,
                   bool approve = true,
                   bool logy = true,
                   int pos = 0,
                   int var = -1,
                   int ene = -1,
                   int eta = -1,
                   int savePlot = -1);
void plotEMeanAll(
    int data = 4, int models = 63, bool ratio = true, bool approve = true, std::string postfix = "F", int savePlot = 2);
void plotEMean(std::string fname = "",
               std::string hlt = "All HLTs",
               int models = 15,
               int var = 0,
               int eta = 0,
               int pv = 0,
               int dataMC = 1,
               bool raio = false,
               bool approve = true,
               std::string postfix = "",
               int savePlot = -1);
TCanvas* plotEMeanDraw(std::vector<std::string> fnames,
                       std::vector<std::string> hlts,
                       int var,
                       int eta,
                       int pv = 0,
                       bool approve = false,
                       std::string dtype = "Data",
                       int coloff = 0);
TCanvas* plotEMeanRatioDraw(std::vector<std::string> fnames,
                            std::vector<std::string> hlts,
                            int var,
                            int eta,
                            int pv = 0,
                            bool approve = false,
                            std::string dtype = "Data",
                            int coloff = 0);
void plotEMeanRatioXAll(std::string fname = "pikp/FBE4p3vMixStudyHLT.root",
                        std::string hlt = "10.4p03 FTFP_BERT_EMM",
                        std::string postfix = "F",
                        int savePlot = 2);
TCanvas* plotEMeanRatioDrawX(std::string fname = "pikp/FBE4p3vMixStudyHLT.root",
                             std::string hlt = "10.4p03 FTFP_BERT_EMM",
                             int var = 0,
                             int pv = 0);
TCanvas* plotEnergies(std::vector<std::string> fnames,
                      std::vector<std::string> hlts,
                      int var = 0,
                      int ien = 0,
                      int eta = 0,
                      int pv = 0,
                      bool varbin = false,
                      int rebin = 1,
                      bool approve = false,
                      std::string dtype = "Data",
                      bool logy = true,
                      int pos = 0,
                      int coloff = 0);
TCanvas* plotEnergy(std::string fname = "hlt.root",
                    std::string HLT = "All HLTs",
                    int var = 0,
                    int ien = 0,
                    int eta = 0,
                    bool varbin = false,
                    int rebin = 1,
                    bool approve = false,
                    bool logy = true,
                    int pos = 0,
                    int coloff = 0);
void plotEMeanPVAll(std::string fname = "StudyHLT_ZeroBias_1PV.root",
                    std::string HLT = "Zero Bias",
                    int var = -1,
                    int eta = -1,
                    bool approve = true);
TCanvas* plotEMeanDrawPV(std::string fname = "StudyHLT_ZeroBias_1PV.root",
                         std::string HLT = "Zero Bias",
                         int var = 0,
                         int eta = 0,
                         bool approve = true);
TCanvas* plotEnergyPV(std::string fnamee = "StudyHLT_HLTZeroBias.root",
                      std::string HLT = "Zero Bias",
                      int var = 0,
                      int ien = 0,
                      int eta = 0,
                      bool varbin = false,
                      int rebin = 1,
                      bool approve = false,
                      bool logy = true,
                      int pos = 0);
TCanvas* plotTrack(std::string fname = "hlt.root",
                   std::string HLT = "All HLTs",
                   int var = 0,
                   bool varbin = false,
                   int rebin = 1,
                   bool approve = false,
                   bool logy = true,
                   int pos = 0);
TCanvas* plotIsolation(std::string fname = "hlt.root",
                       std::string HLT = "All HLTs",
                       int var = 0,
                       bool varbin = false,
                       int rebin = 1,
                       bool approve = false,
                       bool logy = true,
                       int pos = 0);
TCanvas* plotHLT(std::string fname = "hlt.root",
                 std::string HLT = "All HLTs",
                 int run = -1,
                 bool varbin = false,
                 int rebin = 1,
                 bool approve = false,
                 bool logy = true,
                 int pos = 0);
TCanvas* plotHisto(char* cname,
                   std::string HLT,
                   TObjArray& histArr,
                   std::vector<std::string>& labels,
                   std::vector<int>& color,
                   char* name,
                   double ymx0,
                   bool logy,
                   int pos,
                   double yloff,
                   double yhoff,
                   double xmax = -1,
                   bool varbin = false,
                   int rebin = 1,
                   bool approve = false);
void getHistStats(TH1D* h,
                  int& entries,
                  int& integral,
                  double& mean,
                  double& meanE,
                  double& rms,
                  double& rmsE,
                  int& uflow,
                  int& oflow);
TFitResultPtr getHistFitStats(
    TH1F* h, const char* formula, double xlow, double xup, unsigned int& nPar, double* par, double* epar);
void setHistAttr(TH1F* h, int icol, int lwid = 1, int ltype = 1);
double getWeightedMean(int npt, int Start, std::vector<double>& mean, std::vector<double>& emean);
TH1D* rebin(TH1D* histin, int);

void plotAll(std::string fname,
             std::string HLT,
             int var,
             int ien,
             int eta,
             bool varbin,
             int rebin,
             bool approve,
             bool logy,
             int pos,
             bool pv,
             int savePlot) {
  int varmin(0), varmax(5), enemin(0), enemax(9), etamin(0), etamax(etaMax);
  if (var >= 0)
    varmin = varmax = var;
  if (ien >= 0)
    enemin = enemax = ien;
  if (eta >= 0)
    etamin = etamax = eta;
  for (int var = varmin; var <= varmax; ++var) {
    for (int ene = enemin; ene <= enemax; ++ene) {
      for (int eta = etamin; eta <= etamax; ++eta) {
        TCanvas* c(0);
        if (pv) {
          c = plotEnergyPV(fname, HLT, var, ene, eta, varbin, rebin, approve, logy, pos);
        } else {
          c = plotEnergy(fname, HLT, var, ene, eta, varbin, rebin, approve, logy, pos);
        }
        if (c != 0 && savePlot >= 0 && savePlot <= 3) {
          std::string ext[4] = {"eps", "gif", "pdf", "C"};
          char name[200];
          sprintf(name, "%s.%s", c->GetName(), ext[savePlot].c_str());
          c->Print(name);
        }
      }
    }
  }
}

void plotEnergyAll(std::string fname,
                   std::string hlt,
                   int models,
                   int pv,
                   int data,
                   bool varbin,
                   int rebin,
                   bool approve,
                   bool logy,
                   int pos,
                   int var,
                   int ene,
                   int eta,
                   int savePlot) {
  std::vector<std::string> fnames, hlts;
  std::string dtype = (data == 1) ? "Data" : "MC";
  int modeluse(models);
  int coloff = (data == 4) ? 0 : 1;
  if (fname == "") {
    if (data == 1) {
      for (int i = 0; i < nmodels; ++i) {
        if (modeluse % 2 == 1) {
          fnames.push_back(files[i]);
          hlts.push_back(types[i]);
        }
        modeluse /= 2;
      }
    } else if (data == 4) {
      for (int i = 0; i < nmodelx; ++i) {
        if (modeluse % 2 == 1) {
          fnames.push_back(filex[i]);
          hlts.push_back(typex[i]);
        }
        modeluse /= 2;
      }
    } else {
      for (int i = 0; i < nmodelm; ++i) {
        if (modeluse % 2 == 1) {
          fnames.push_back(filem[i]);
          hlts.push_back(typem[i]);
        }
        modeluse /= 2;
      }
    }
  } else {
    fnames.push_back(fname);
    hlts.push_back(hlt);
  }
  int varmin(0), varmax(5), enemin(0), enemax(9), etamin(0), etamax(etaMax);
  if (var >= varmin && var <= varmax)
    varmin = varmax = var;
  if (ene >= enemin && ene <= enemax)
    enemin = enemax = ene;
  if (eta >= etamin && eta <= etamax)
    etamin = etamax = eta;
  for (int var = varmin; var <= varmax; ++var) {
    for (int ene = enemin; ene <= enemax; ++ene) {
      for (int eta = etamin; eta <= etamax; ++eta) {
        TCanvas* c = plotEnergies(fnames, hlts, var, ene, eta, pv, varbin, rebin, approve, dtype, logy, pos, coloff);
        if (c != 0 && savePlot >= 0 && savePlot <= 3) {
          std::string ext[4] = {"eps", "gif", "pdf", "C"};
          char name[200];
          sprintf(name, "%s.%s", c->GetName(), ext[savePlot].c_str());
          c->Print(name);
        }
      }
    }
  }
}

void plotEMeanAll(int data, int models, bool ratio, bool approve, std::string postfix, int savePlot) {
  int varmin(0), varmax(5), pvmin(0), pvmax(0), etamin(0), etamax(etaMax);
  for (int var = varmin; var <= varmax; ++var) {
    for (int eta = etamin; eta <= etamax; ++eta) {
      for (int pv = pvmin; pv <= pvmax; ++pv) {
        plotEMean("", "", models, var, eta, pv, data, ratio, approve, postfix, savePlot);
      }
    }
  }
}

void plotEMean(std::string fname,
               std::string hlt,
               int models,
               int var,
               int eta,
               int pv,
               int data,
               bool ratio,
               bool approve,
               std::string postfix,
               int savePlot) {
  std::vector<std::string> fnames, hlts;
  std::string dtype = (data == 1) ? "Data" : "MC";
  int modeluse(models);
  int coloff = (data == 4 || data == 3) ? 0 : 1;
  if (fname == "") {
    if (data == 1) {
      for (int i = 0; i < nmodels; ++i) {
        if (modeluse % 2 == 1) {
          fnames.push_back(files[i]);
          hlts.push_back(types[i]);
        }
        modeluse /= 2;
      }
    } else if (data == 4) {
      for (int i = 0; i < nmodelx; ++i) {
        if (modeluse % 2 == 1) {
          fnames.push_back(filex[i]);
          hlts.push_back(typex[i]);
        }
        modeluse /= 2;
      }
    } else if (data == 3) {
      for (int i = 0; i < nmodelx; ++i) {
        if (modeluse % 2 == 1) {
          fnames.push_back(filex[i]);
          hlts.push_back(typex[i]);
        }
        modeluse /= 2;
      }
    } else {
      for (int i = 0; i < nmodelm; ++i) {
        if (modeluse % 2 == 1) {
          fnames.push_back(filem[i]);
          hlts.push_back(typem[i]);
        }
        modeluse /= 2;
      }
    }
  } else {
    fnames.push_back(fname);
    hlts.push_back(hlt);
  }
  int varmin(0), varmax(5), etamin(0), etamax(etaMax), pvmin(0), pvmax(4);
  if (var >= 0)
    varmin = varmax = var;
  if (eta >= 0)
    etamin = etamax = eta;
  if (pv >= 0)
    pvmin = pvmax = pv;
  for (int var = varmin; var <= varmax; ++var) {
    for (int eta = etamin; eta <= etamax; ++eta) {
      for (int pv = pvmin; pv <= pvmax; ++pv) {
        TCanvas* c = ((ratio) ? plotEMeanRatioDraw(fnames, hlts, var, eta, pv, approve, dtype, coloff)
                              : plotEMeanDraw(fnames, hlts, var, eta, pv, approve, dtype, coloff));
        if (c != 0 && savePlot >= 0 && savePlot <= 3) {
          std::string ext[4] = {"eps", "gif", "pdf", "C"};
          char name[200];
          sprintf(name, "%s%s.%s", c->GetName(), postfix.c_str(), ext[savePlot].c_str());
          c->Print(name);
        }
      }
    }
  }
}

TCanvas* plotEMeanDraw(std::vector<std::string> fnames,
                       std::vector<std::string> hlts,
                       int var,
                       int eta,
                       int pv,
                       bool approve,
                       std::string dtype,
                       int coloff) {
  bool debug(false);
  std::vector<TGraphAsymmErrors*> graphs;
  double yminx = (fnames.size() < 3) ? 0.85 : 0.75;
  TLegend* legend = new TLegend(0.60, yminx, 0.975, 0.95);
  legend->SetBorderSize(1);
  legend->SetFillColor(kWhite);
  legend->SetMargin(0.2);
  for (unsigned int k = 0; k < fnames.size(); ++k) {
    TFile* file = TFile::Open(fnames[k].c_str());
    double mean[NPT], dmean[NPT];
    for (int i = 0; i < NPT; ++i) {
      char name[100];
      sprintf(name, "h_energy_%d_%d_%d_%d", pv + 3, i, eta, var);
      TH1D* histo = (TH1D*)file->FindObjectAny(name);
      if (histo) {
        mean[i] = histo->GetMean();
        dmean[i] = histo->GetMeanError();
      } else {
        mean[i] = -100.;
        dmean[i] = 0;
      }
    }
    if (debug) {
      std::cout << "Get mean for " << NPT << " points" << std::endl;
      for (int i = 0; i < NPT; ++i)
        std::cout << "[" << i << "]"
                  << " Momentum " << mom[i] << " +- " << dmom[i] << " Mean " << mean[i] << " +- " << dmean[i]
                  << std::endl;
    }
    TGraphAsymmErrors* graph = new TGraphAsymmErrors(NPT, mom, mean, dmom, dmom, dmean, dmean);
    graph->SetMarkerStyle(styles[coloff + k]);
    graph->SetMarkerColor(colors[coloff + k]);
    graph->SetMarkerSize(1.2);
    graph->SetLineColor(colors[coloff + k]);
    graph->SetLineWidth(2);
    graphs.push_back(graph);
    legend->AddEntry(graph, hlts[k].c_str(), "lp");
    if (debug)
      std::cout << "Complete " << hlts[k] << std::endl;
    file->Close();
  }

  char cname[100], name[200];
  sprintf(cname, "c_%s_%d_%d_%s", varEne1[var].c_str(), eta, pv, dtype.c_str());
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(kFALSE);
  gStyle->SetPadBorderMode(0);
  gStyle->SetCanvasBorderMode(0);
  TCanvas* canvas = new TCanvas(cname, cname, 500, 400);
  gStyle->SetOptStat(0);
  gPad->SetTopMargin(0.05);
  gPad->SetLeftMargin(0.15);
  gPad->SetRightMargin(0.025);
  gPad->SetBottomMargin(0.20);
  TH1F* vFrame = canvas->DrawFrame(0.0, 0.01, 50.0, 0.5);
  vFrame->GetYaxis()->SetRangeUser(0.0, 1.6);
  vFrame->GetXaxis()->SetLabelSize(0.06);
  vFrame->GetYaxis()->SetLabelSize(0.05);
  vFrame->GetXaxis()->SetTitleSize(0.06);
  vFrame->GetYaxis()->SetTitleSize(0.06);
  vFrame->GetYaxis()->SetTitleOffset(0.9);
  vFrame->GetXaxis()->SetRangeUser(1.0, 20.0);
  if (approve) {
    sprintf(name, "Mean of %s/p_{Track}", varEne[var].c_str());
  } else {
    sprintf(name, "<%s/p_{Track}>", varEne[var].c_str());
  }
  vFrame->GetYaxis()->SetTitle(name);
  sprintf(name, "p_{Track} (GeV/c)");
  vFrame->GetXaxis()->SetTitle(name);
  for (unsigned int ii = 0; ii < graphs.size(); ++ii)
    graphs[ii]->Draw("P");
  legend->Draw();
  TLine* line = new TLine(1.0, 1.0, 20.0, 1.0);
  line->SetLineStyle(2);
  line->SetLineWidth(2);
  line->SetLineColor(kRed);
  line->Draw();
  TPaveText* text = new TPaveText(0.25, 0.74, 0.55, 0.79, "brNDC");
  if (approve) {
    sprintf(name, "(%s)", nameEtas[eta].c_str());
  } else {
    sprintf(name, "(%s, %s)", nameEta[eta].c_str(), namePV[pv].c_str());
  }
  if (debug)
    std::cout << "Name " << name << " |" << std::endl;
  text->AddText(name);
  text->Draw("same");
  TPaveText* text2 = new TPaveText(0.55, yminx - 0.06, 0.97, yminx - 0.01, "brNDC");
  sprintf(name, cmsP.c_str());
  text2->AddText(name);
  text2->Draw("same");
  return canvas;
}

TCanvas* plotEMeanRatioDraw(std::vector<std::string> fnames,
                            std::vector<std::string> hlts,
                            int var,
                            int eta,
                            int pv,
                            bool approve,
                            std::string dtype,
                            int coloff) {
  bool debug(false);
  std::vector<TGraphAsymmErrors*> graphs;
  double yminx = (fnames.size() < 3) ? 0.88 : 0.80;
  TLegend* legend = new TLegend(0.60, yminx, 0.975, 0.95);
  legend->SetBorderSize(1);
  legend->SetFillColor(kWhite);
  legend->SetMargin(0.2);
  const int NPT2 = 2 * NPT + 2;
  double mean0[NPT], dmean0[NPT], ones[NPT];
  double pmom1[NPT2 + 1], dmean1[NPT2 + 1];
  for (unsigned int k = 0; k < fnames.size(); ++k) {
    TFile* file = TFile::Open(fnames[k].c_str());
    double mean[NPT], dmean[NPT];
    for (int i = 0; i < NPT; ++i) {
      char name[100];
      sprintf(name, "h_energy_%d_%d_%d_%d", pv + 3, i, eta, var);
      TH1D* histo = (TH1D*)file->FindObjectAny(name);
      if (histo) {
        mean[i] = histo->GetMean();
        dmean[i] = histo->GetMeanError();
      } else {
        mean[i] = -100.;
        dmean[i] = 0;
      }
    }
    if (debug) {
      std::cout << "Get mean for " << NPT << " points" << std::endl;
      for (int i = 0; i < NPT; ++i)
        std::cout << "[" << i << "]"
                  << " Momentum " << mom[i] << " +- " << dmom[i] << " Mean " << mean[i] << " +- " << dmean[i]
                  << std::endl;
    }
    if (k == 0) {
      for (int i = 0; i < NPT; ++i) {
        ones[i] = 1.0;
        mean0[i] = mean[i];
        dmean0[i] = dmean[i];
        pmom1[i] = mom[i] - dmom[i];
        pmom1[NPT2 - i - 1] = pmom1[i];
        dmean1[i] = (mean[i] > 0) ? dmean[i] / mean[i] : 0.0;
        dmean1[NPT2 - i - 1] = -dmean1[i];
      }
      pmom1[NPT + 1] = pmom1[NPT] = mom[NPT - 1] + dmom[NPT - 1];
      dmean1[NPT] = dmean1[NPT - 1];
      dmean1[NPT + 1] = -dmean1[NPT];
      pmom1[NPT2] = pmom1[0];
      dmean1[NPT2] = dmean1[0];
      if (debug)
        for (int k = 0; k < NPT2 + 1; ++k)
          std::cout << "[" << k << "] " << pmom1[k] << " " << dmean1[k] << "\n";
    } else {
      double sumNum(0), sumDen(0), sumNum1(0), sumDen1(0);
      for (int i = 0; i < NPT; ++i) {
        if (dmean[i] > 0 && dmean0[i] > 0) {
          double er1 = dmean[i] / mean[i];
          double er2 = dmean0[i] / mean0[i];
          mean[i] = mean[i] / mean0[i];
          dmean[i] = mean[i] * sqrt(er1 * er1 + er2 * er2);
          double temp1 = (mean[i] > 1.0) ? 1.0 / mean[i] : mean[i];
          double temp2 = (mean[i] > 1.0) ? dmean[i] / (mean[i] * mean[i]) : dmean[i];
          if (i > 0) {
            sumNum += (fabs(1 - temp1) / (temp2 * temp2));
            sumDen += (1.0 / (temp2 * temp2));
          }
          sumNum1 += (fabs(1 - temp1) / (temp2 * temp2));
          sumDen1 += (1.0 / (temp2 * temp2));
        } else {
          mean[i] = -100.;
          dmean[i] = 0;
        }
      }
      sumNum = (sumDen > 0) ? (sumNum / sumDen) : 0;
      sumDen = (sumDen > 0) ? 1.0 / sqrt(sumDen) : 0;
      sumNum1 = (sumDen1 > 0) ? (sumNum1 / sumDen1) : 0;
      sumDen1 = (sumDen1 > 0) ? 1.0 / sqrt(sumDen1) : 0;
      std::cout << "Get Ratio of mean for " << NPT << " points: Mean " << sumNum << " +- " << sumDen << " (" << sumNum1
                << " +- " << sumDen1 << ") Input: " << fnames[k] << " var|eta|pv " << var << ":" << eta << ":" << pv
                << std::endl;
      if (debug) {
        std::cout << "Get Ratio of mean for " << NPT << " points: Mean " << sumNum << " +- " << sumDen << std::endl;
        for (int i = 0; i < NPT; ++i)
          std::cout << "[" << i << "]"
                    << " Momentum " << mom[i] << " +- " << dmom[i] << " Mean " << mean[i] << " +- " << dmean[i]
                    << std::endl;
      }
      TGraphAsymmErrors* graph = new TGraphAsymmErrors(NPT, mom, mean, dmom, dmom, dmean, dmean);
      graph->SetMarkerStyle(styles[coloff + k]);
      graph->SetMarkerColor(colors[coloff + k]);
      graph->SetMarkerSize(1.2);
      graph->SetLineColor(colors[coloff + k]);
      graph->SetLineWidth(2);
      graphs.push_back(graph);
      char text[100];
      if (approve) {
        sprintf(text, "%s", hlts[k].c_str());
      } else {
        sprintf(text, "%5.3f #pm %5.3f  %s", sumNum, sumDen, hlts[k].c_str());
      }
      legend->AddEntry(graph, text, "lp");
      if (debug)
        std::cout << "Complete " << hlts[k] << std::endl;
    }
    file->Close();
  }
  TGraphErrors* graph = new TGraphErrors(NPT, mom, ones, dmean1);
  graph->SetFillColor(4);
  graph->SetFillStyle(3005);
  TPolyLine* pline = new TPolyLine(NPT2 + 1, pmom1, dmean1);
  pline->SetFillColor(5);
  pline->SetLineColor(6);
  pline->SetLineWidth(2);

  char cname[100], name[200];
  sprintf(cname, "cR_%s_%d_%d_%s", varEne1[var].c_str(), eta, pv, dtype.c_str());
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(kFALSE);
  gStyle->SetPadBorderMode(0);
  gStyle->SetCanvasBorderMode(0);
  TCanvas* canvas = new TCanvas(cname, cname, 500, 400);
  gStyle->SetOptStat(0);
  gPad->SetTopMargin(0.05);
  gPad->SetLeftMargin(0.15);
  gPad->SetRightMargin(0.025);
  gPad->SetBottomMargin(0.20);
  TH1F* vFrame = canvas->DrawFrame(0.0, 0.01, 50.0, 0.5);
  vFrame->GetYaxis()->SetRangeUser(0.4, 1.5);
  vFrame->GetXaxis()->SetLabelSize(0.06);
  vFrame->GetYaxis()->SetLabelSize(0.05);
  vFrame->GetXaxis()->SetTitleSize(0.06);
  vFrame->GetYaxis()->SetTitleSize(0.045);
  vFrame->GetYaxis()->SetTitleOffset(1.2);
  vFrame->GetXaxis()->SetRangeUser(1.0, 20.0);
  if (approve) {
    sprintf(name, "%s/Data for mean of %s/p_{Track}", dtype.c_str(), varEne[var].c_str());
  } else {
    sprintf(name, "#frac{%s}{%s} for #frac{%s}{p_{Track}}", dtype.c_str(), hlts[0].c_str(), varEne[var].c_str());
  }
  vFrame->GetYaxis()->SetTitle(name);
  sprintf(name, "p_{Track} (GeV/c)");
  vFrame->GetXaxis()->SetTitle(name);
  for (unsigned int ii = 0; ii < graphs.size(); ++ii)
    graphs[ii]->Draw("P");
  graph->Draw("3");
  legend->Draw();
  TLine* line = new TLine(1.0, 1.0, 20.0, 1.0);
  line->SetLineStyle(2);
  line->SetLineWidth(2);
  line->SetLineColor(kRed);
  line->Draw();
  pline->Draw("f L SAME");
  TPaveText* text = new TPaveText(0.55, 0.40, 0.95, 0.45, "brNDC");
  if (approve) {
    sprintf(name, "(%s)", nameEtas[eta].c_str());
  } else {
    sprintf(name, "(%s, %s)", nameEta[eta].c_str(), namePV[pv].c_str());
  }
  if (debug)
    std::cout << "Name " << name << " |" << std::endl;
  text->AddText(name);
  text->Draw("same");
  TPaveText* text2 = new TPaveText(0.55, 0.45, 0.95, 0.50, "brNDC");
  sprintf(name, cmsP.c_str());
  text2->AddText(name);
  text2->Draw("same");
  return canvas;
}

void plotEMeanRatioXAll(std::string fname, std::string hlt, std::string postfix, int savePlot) {
  int varmin(0), varmax(5), pvmin(0), pvmax(0);
  for (int var = varmin; var <= varmax; ++var) {
    for (int pv = pvmin; pv <= pvmax; ++pv) {
      TCanvas* c = plotEMeanRatioDrawX(fname, hlt, var, pv);
      if (savePlot >= 0 && savePlot <= 3) {
        std::string ext[4] = {"eps", "gif", "pdf", "C"};
        char name[200];
        sprintf(name, "%s%s.%s", c->GetName(), postfix.c_str(), ext[savePlot].c_str());
        c->Print(name);
      }
    }
  }
}

TCanvas* plotEMeanRatioDrawX(std::string fname, std::string hlt, int var, int pv) {
  bool debug(false);
  std::vector<TGraphAsymmErrors*> graphs;
  double yminx = 0.80;
  TLegend* legend = new TLegend(0.60, yminx, 0.975, 0.95);
  legend->SetBorderSize(1);
  legend->SetFillColor(kWhite);
  legend->SetMargin(0.2);
  double mean0[NPT], dmean0[NPT], mean[NPT], dmean[NPT];
  char name[200], cname[100];
  for (int eta = 0; eta <= etaMax; ++eta) {
    // First Data
    TFile* file = TFile::Open(fileData.c_str());
    for (int i = 0; i < NPT; ++i) {
      sprintf(name, "h_energy_%d_%d_%d_%d", pv + 3, i, eta, var);
      TH1D* histo = (TH1D*)file->FindObjectAny(name);
      if (histo) {
        mean0[i] = histo->GetMean();
        dmean0[i] = histo->GetMeanError();
      } else {
        mean0[i] = -100.;
        dmean0[i] = 0;
      }
    }
    file->Close();
    // Now MC
    file = TFile::Open(fname.c_str());
    for (int i = 0; i < NPT; ++i) {
      sprintf(name, "h_energy_%d_%d_%d_%d", pv + 3, i, eta, var);
      TH1D* histo = (TH1D*)file->FindObjectAny(name);
      if (histo) {
        mean[i] = histo->GetMean();
        dmean[i] = histo->GetMeanError();
      } else {
        mean[i] = -100.;
        dmean[i] = 0;
      }
    }
    file->Close();
    if (debug) {
      std::cout << "Get mean for " << NPT << " points" << std::endl;
      for (int i = 0; i < NPT; ++i)
        std::cout << "[" << i << "]"
                  << " Momentum " << mom[i] << " +- " << dmom[i] << " Data:Mean " << mean0[i] << " +- " << dmean0[i]
                  << " MC:Mean " << mean[i] << " +- " << dmean[i] << std::endl;
    }
    double sumNum(0), sumDen(0), sumNum1(0), sumDen1(0);
    for (int i = 0; i < NPT; ++i) {
      if (dmean[i] > 0 && dmean0[i] > 0) {
        double er1 = dmean[i] / mean[i];
        double er2 = dmean0[i] / mean0[i];
        mean[i] = mean[i] / mean0[i];
        dmean[i] = mean[i] * sqrt(er1 * er1 + er2 * er2);
        double temp1 = (mean[i] > 1.0) ? 1.0 / mean[i] : mean[i];
        double temp2 = (mean[i] > 1.0) ? dmean[i] / (mean[i] * mean[i]) : dmean[i];
        if (i > 0) {
          sumNum += (fabs(1 - temp1) / (temp2 * temp2));
          sumDen += (1.0 / (temp2 * temp2));
        }
        sumNum1 += (fabs(1 - temp1) / (temp2 * temp2));
        sumDen1 += (1.0 / (temp2 * temp2));
      } else {
        mean[i] = -100.;
        dmean[i] = 0;
      }
    }
    sumNum = (sumDen > 0) ? (sumNum / sumDen) : 0;
    sumDen = (sumDen > 0) ? 1.0 / sqrt(sumDen) : 0;
    sumNum1 = (sumDen1 > 0) ? (sumNum1 / sumDen1) : 0;
    sumDen1 = (sumDen1 > 0) ? 1.0 / sqrt(sumDen1) : 0;
    std::cout << "Get Ratio of mean for " << NPT << " points: Mean " << sumNum << " +- " << sumDen << " (" << sumNum1
              << " +- " << sumDen1 << ") Input: " << fname << " var|eta|pv " << var << ":" << eta << ":" << pv
              << std::endl;
    if (debug) {
      std::cout << "Get Ratio of mean for " << NPT << " points: Mean " << sumNum << " +- " << sumDen << std::endl;
      for (int i = 0; i < NPT; ++i)
        std::cout << "[" << i << "]"
                  << " Momentum " << mom[i] << " +- " << dmom[i] << " Mean " << mean[i] << " +- " << dmean[i]
                  << std::endl;
    }
    TGraphAsymmErrors* graph = new TGraphAsymmErrors(NPT, mom, mean, dmom, dmom, dmean, dmean);
    graph->SetMarkerStyle(styles[eta]);
    graph->SetMarkerColor(colors[eta]);
    graph->SetMarkerSize(1.2);
    graph->SetLineColor(colors[eta]);
    graph->SetLineWidth(2);
    graphs.push_back(graph);
    sprintf(cname, "%s", nameEtas[eta].c_str());
    legend->AddEntry(graph, cname, "lp");
  }

  sprintf(cname, "cR_%s_%d", varEne1[var].c_str(), pv);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(kFALSE);
  gStyle->SetPadBorderMode(0);
  gStyle->SetCanvasBorderMode(0);
  TCanvas* canvas = new TCanvas(cname, cname, 750, 600);
  gStyle->SetOptStat(0);
  gPad->SetTopMargin(0.05);
  gPad->SetLeftMargin(0.15);
  gPad->SetRightMargin(0.025);
  gPad->SetBottomMargin(0.20);
  TH1F* vFrame = canvas->DrawFrame(0.0, 0.01, 50.0, 0.5);
  vFrame->GetYaxis()->SetRangeUser(0.4, 1.5);
  vFrame->GetXaxis()->SetLabelSize(0.06);
  vFrame->GetYaxis()->SetLabelSize(0.05);
  vFrame->GetXaxis()->SetTitleSize(0.06);
  vFrame->GetYaxis()->SetTitleSize(0.045);
  vFrame->GetYaxis()->SetTitleOffset(1.2);
  vFrame->GetXaxis()->SetRangeUser(1.0, 20.0);
  sprintf(name, "MC/Data for mean of %s/p_{Track}", varEne[var].c_str());
  vFrame->GetYaxis()->SetTitle(name);
  sprintf(name, "p_{Track} (GeV/c)");
  vFrame->GetXaxis()->SetTitle(name);
  for (unsigned int ii = 0; ii < graphs.size(); ++ii)
    graphs[ii]->Draw("P");
  legend->Draw();
  TLine* line = new TLine(1.0, 1.0, 20.0, 1.0);
  line->SetLineStyle(2);
  line->SetLineWidth(2);
  line->SetLineColor(kRed);
  line->Draw();
  TPaveText* text = new TPaveText(0.60, 0.40, 0.92, 0.45, "brNDC");
  sprintf(name, "%s", hlt.c_str());
  text->AddText(name);
  text->Draw("same");
  TPaveText* text2 = new TPaveText(0.55, 0.45, 0.95, 0.50, "brNDC");
  sprintf(name, "%s", cmsP.c_str());
  text2->AddText(name);
  text2->Draw("same");
  return canvas;
}

TCanvas* plotEnergies(std::vector<std::string> fnames,
                      std::vector<std::string> hlts,
                      int var,
                      int ien,
                      int eta,
                      int pv,
                      bool varbin,
                      int rebin,
                      bool approve,
                      std::string dtype,
                      bool logy,
                      int pos,
                      int coloff) {
  TLegend* legend = new TLegend(0.55, 0.70, 0.95, 0.85);
  legend->SetBorderSize(1);
  legend->SetFillColor(kWhite);
  legend->SetMargin(0.4);
  TObjArray histArr;
  char name[100];
  std::vector<std::string> labels;
  std::vector<int> color;
  double ymx0(0), ent0(0);
  sprintf(name, "h_energy_%d_%d_%d_%d", pv + 3, ien, eta, var);
  for (unsigned int k = 0; k < fnames.size(); ++k) {
    TFile* file = TFile::Open(fnames[k].c_str());
    TH1D* histo = (TH1D*)file->FindObjectAny(name);
    if (histo) {
      if (k == 0) {
        ent0 = histo->GetEntries();
      } else {
        double scale = (histo->GetEntries() > 0) ? ent0 / histo->GetEntries() : 1;
        histo->Scale(scale);
      }
      histArr.AddLast(histo);
      labels.push_back(hlts[k]);
      color.push_back(colors[coloff + k]);
      int ibin = histo->GetMaximumBin();
      if (histo->GetBinContent(ibin) > ymx0)
        ymx0 = histo->GetBinContent(ibin);
    }
  }
  TCanvas* c(0);
  if (histArr.GetEntries() > 0) {
    sprintf(name, "p=%s, %s", varPPs[ien].c_str(), nameEtas[eta].c_str());
    /*
    if (approve) {
      sprintf (name, "p=%s, %s", varPPs[ien].c_str(), nameEtas[eta].c_str());
    } else {
      sprintf (name, "p=%s, i#eta=%s, %s", varPs[ien].c_str(), varEta[eta].c_str(), namePV[pv].c_str());
    }
    */
    std::string clabel(name);
    char cname[50];
    sprintf(cname, "c_%s_%d_%d_%s", varEne1[var].c_str(), ien, eta, dtype.c_str());
    sprintf(name, "%s/p", varEne[var].c_str());
    c = plotHisto(
        cname, clabel, histArr, labels, color, name, ymx0, logy, pos, 0.10, 0.05, 2.5, varbin, rebin, approve);
  }
  return c;
}

TCanvas* plotEnergy(std::string fname,
                    std::string HLT,
                    int var,
                    int ien,
                    int eta,
                    bool varbin,
                    int rebin,
                    bool approve,
                    bool logy,
                    int pos,
                    int coloff) {
  TFile* file = TFile::Open(fname.c_str());
  char name[100];
  TObjArray histArr;
  std::vector<std::string> labels;
  std::vector<int> color;
  double ymx0(0);
  for (int i = 0; i < 4; ++i) {
    sprintf(name, "h_energy_%d_%d_%d_%d", i, ien, eta, var);
    TH1D* histo = (TH1D*)file->FindObjectAny(name);
    if (histo) {
      histArr.AddLast(histo);
      sprintf(name, "p=%s, #eta=%s %s", varPs[ien].c_str(), varEta[eta].c_str(), namefull[i + 3].c_str());
      labels.push_back(name);
      color.push_back(colors[coloff + i]);
      int ibin = histo->GetMaximumBin();
      if (histo->GetBinContent(ibin) > ymx0)
        ymx0 = histo->GetBinContent(ibin);
    }
  }
  TCanvas* c(0);
  if (histArr.GetEntries() > 0) {
    char cname[50];
    sprintf(cname, "c_%s_%d_%d", varEne1[var].c_str(), ien, eta);
    sprintf(name, "%s/p", varEne[var].c_str());
    c = plotHisto(cname, HLT, histArr, labels, color, name, ymx0, logy, pos, 0.10, 0.05, 2.5, varbin, rebin, approve);
  }
  return c;
}

void plotEMeanPVAll(std::string fname, std::string HLT, int var, int eta, bool approve) {
  int varmin(0), varmax(5), etamin(0), etamax(etaMax);
  if (var >= 0)
    varmin = varmax = var;
  if (eta >= 0)
    etamin = etamax = eta;
  for (int var = varmin; var <= varmax; ++var) {
    for (int eta = etamin; eta <= etamax; ++eta) {
      plotEMeanDrawPV(fname, HLT, var, eta, approve);
    }
  }
}

TCanvas* plotEMeanDrawPV(std::string fname, std::string HLT, int var, int eta, bool approve) {
  bool debug(false);
  std::vector<TGraphAsymmErrors*> graphs;
  TLegend* legend = new TLegend(0.575, 0.80, 0.975, 0.95);
  legend->SetBorderSize(1);
  legend->SetFillColor(kWhite);
  legend->SetMargin(0.4);
  TFile* file = TFile::Open(fname.c_str());
  const int nPVBin = 4;
  int pvBins[nPVBin + 1] = {1, 2, 3, 5, 100};
  for (int k = 0; k < nPVBin; ++k) {
    char name[100];
    double mean[NPT], dmean[NPT];
    for (int i = 0; i < NPT; ++i) {
      sprintf(name, "h_energy_%d_%d_%d_%d", k, i, eta, var);
      TH1D* histo = (TH1D*)file->FindObjectAny(name);
      if (histo) {
        mean[i] = histo->GetMean();
        dmean[i] = histo->GetMeanError();
      } else {
        mean[i] = -100.;
        dmean[i] = 0;
      }
    }
    if (debug) {
      std::cout << "Get mean for " << NPT << " points" << std::endl;
      for (int i = 0; i < NPT; ++i)
        std::cout << "[" << i << "]"
                  << " Momentum " << mom[i] << " +- " << dmom[i] << " Mean " << mean[i] << " +- " << dmean[i]
                  << std::endl;
    }
    TGraphAsymmErrors* graph = new TGraphAsymmErrors(NPT, mom, mean, dmom, dmom, dmean, dmean);
    graph->SetMarkerStyle(styles[k]);
    graph->SetMarkerColor(colors[k]);
    graph->SetMarkerSize(1.2);
    graph->SetLineColor(colors[k]);
    graph->SetLineWidth(2);
    graphs.push_back(graph);
    sprintf(name, "PV=%d:%d", pvBins[k], pvBins[k + 1] - 1);
    legend->AddEntry(graph, name, "lp");
    if (debug)
      std::cout << "Complete " << name << std::endl;
  }
  file->Close();

  char cname[100], name[200];
  sprintf(cname, "c_%s_PV_%d", varEne1[var].c_str(), eta);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(kFALSE);
  gStyle->SetPadBorderMode(0);
  gStyle->SetCanvasBorderMode(0);
  TCanvas* canvas = new TCanvas(cname, cname, 500, 400);
  gStyle->SetOptStat(0);
  gPad->SetTopMargin(0.05);
  gPad->SetLeftMargin(0.15);
  gPad->SetRightMargin(0.025);
  gPad->SetBottomMargin(0.20);
  TH1F* vFrame = canvas->DrawFrame(0.0, 0.01, 50.0, 0.5);
  vFrame->GetYaxis()->SetRangeUser(0.0, 1.5);
  vFrame->GetXaxis()->SetLabelSize(0.06);
  vFrame->GetYaxis()->SetLabelSize(0.05);
  vFrame->GetXaxis()->SetTitleSize(0.06);
  vFrame->GetYaxis()->SetTitleSize(0.06);
  vFrame->GetYaxis()->SetTitleOffset(0.9);
  vFrame->GetXaxis()->SetRangeUser(1.0, 20.0);
  if (approve) {
    sprintf(name, "Mean of %s/p_{Track}", varEne[var].c_str());
  } else {
    sprintf(name, "<%s/p_{Track}>", varEne[var].c_str());
  }
  vFrame->GetYaxis()->SetTitle(name);
  sprintf(name, "p_{Track} (GeV/c)");
  vFrame->GetXaxis()->SetTitle(name);
  for (unsigned int ii = 0; ii < graphs.size(); ++ii)
    graphs[ii]->Draw("P");
  legend->Draw();
  TLine* line = new TLine(1.0, 1.0, 20.0, 1.0);
  line->SetLineStyle(2);
  line->SetLineWidth(2);
  line->SetLineColor(kRed);
  line->Draw();
  TPaveText* text = new TPaveText(0.575, 0.75, 0.97, 0.79, "brNDC");
  sprintf(name, "%s (%s)", HLT.c_str(), nameEtas[eta].c_str());
  if (debug)
    std::cout << "Name " << name << " |" << std::endl;
  text->AddText(name);
  text->Draw("same");
  TPaveText* text2 = new TPaveText(0.575, 0.71, 0.97, 0.75, "brNDC");
  sprintf(name, cmsP.c_str());
  text2->AddText(name);
  text2->Draw("same");
  return canvas;
}

TCanvas* plotEnergyPV(std::string fname,
                      std::string HLT,
                      int var,
                      int ien,
                      int eta,
                      bool varbin,
                      int rebin,
                      bool approve,
                      bool logy,
                      int pos) {
  const int nPVBin = 4;
  int pvBins[nPVBin + 1] = {1, 2, 3, 5, 100};
  TFile* file = TFile::Open(fname.c_str());
  char name[100];
  TObjArray histArr;
  std::vector<std::string> labels;
  std::vector<int> color;
  double ymx0(0);
  for (int i = 4; i < nPVBin + 4; ++i) {
    sprintf(name, "h_energy_%d_%d_%d_%d", i, ien, eta, var);
    TH1D* histo = (TH1D*)file->FindObjectAny(name);
    if (histo) {
      histArr.AddLast(histo);
      sprintf(name,
              "p=%s, #eta=%s, PV=%d:%d (%s)",
              varPs[ien].c_str(),
              varEta[eta].c_str(),
              pvBins[i - 4],
              pvBins[i - 3] - 1,
              namefull[6].c_str());
      labels.push_back(name);
      color.push_back(colors[i - 4]);
      int ibin = histo->GetMaximumBin();
      if (histo->GetBinContent(ibin) > ymx0)
        ymx0 = histo->GetBinContent(ibin);
    }
  }
  TCanvas* c(0);
  if (histArr.GetEntries() > 0) {
    char cname[50];
    sprintf(cname, "c_%s_%d_%d", varEne1[var].c_str(), ien, eta);
    sprintf(name, "%s/p", varEne[var].c_str());
    c = plotHisto(cname, HLT, histArr, labels, color, name, ymx0, logy, pos, 0.10, 0.05, 2.5, varbin, rebin, approve);
  }
  return c;
}

TCanvas* plotTrack(
    std::string fname, std::string HLT, int var, bool varbin, int rebin, bool approve, bool logy, int pos) {
  TFile* file = TFile::Open(fname.c_str());
  char name[100];
  TObjArray histArr;
  std::vector<std::string> labels;
  std::vector<int> color;
  double ymx0(0);
  for (int i = 0; i < 7; ++i) {
    sprintf(name, "h_%s_%s", varname[var].c_str(), names[i].c_str());
    TH1D* histo = (TH1D*)file->FindObjectAny(name);
    if (histo) {
      histArr.AddLast(histo);
      labels.push_back(namefull[i]);
      color.push_back(colors[i]);
      int ibin = histo->GetMaximumBin();
      if (histo->GetBinContent(ibin) > ymx0)
        ymx0 = histo->GetBinContent(ibin);
    }
  }
  if (histArr.GetEntries() > 0) {
    char cname[50];
    sprintf(cname, "c_%s", varname[var].c_str());
    sprintf(name, "%s", vartitle[var].c_str());
    return plotHisto(cname, HLT, histArr, labels, color, name, ymx0, logy, pos, 0.10, 0.05, -1, varbin, rebin, approve);
  } else {
    return 0;
  }
}

TCanvas* plotIsolation(
    std::string fname, std::string HLT, int var, bool varbin, int rebin, bool approve, bool logy, int pos) {
  TFile* file = TFile::Open(fname.c_str());
  char name[100];
  TObjArray histArr;
  std::vector<std::string> labels;
  std::vector<int> color;
  double ymx0(0);
  for (int i = 0; i < 2; ++i) {
    sprintf(name, "h_%s_%s", varnameC[var].c_str(), nameC[i].c_str());
    TH1D* histo = (TH1D*)file->FindObjectAny(name);
    if (histo) {
      histArr.AddLast(histo);
      labels.push_back(nameCF[i]);
      color.push_back(colors[i]);
      int ibin = histo->GetMaximumBin();
      if (histo->GetBinContent(ibin) > ymx0)
        ymx0 = histo->GetBinContent(ibin);
    }
  }
  if (histArr.GetEntries() > 0) {
    char cname[50];
    sprintf(cname, "c_%s", varnameC[var].c_str());
    sprintf(name, "%s (GeV)", vartitlC[var].c_str());
    return plotHisto(cname, HLT, histArr, labels, color, name, ymx0, logy, pos, 0.10, 0.05, -1, varbin, rebin, approve);
  } else {
    return 0;
  }
}

TCanvas* plotHLT(std::string fname, std::string HLT, int run, bool varbin, int rebin, bool approve, bool logy, int pos) {
  TFile* file = TFile::Open(fname.c_str());
  char name[100];
  TObjArray histArr;
  std::vector<std::string> labels;
  std::vector<int> color;
  double ymx0(0);
  if (run > 0)
    sprintf(name, "h_HLTAccepts_%d", run);
  else
    sprintf(name, "h_HLTAccept");
  TH1D* histo = (TH1D*)file->FindObjectAny(name);
  if (histo) {
    histArr.AddLast(histo);
    labels.push_back(HLT);
    color.push_back(colors[3]);
    int ibin = histo->GetMaximumBin();
    ymx0 = histo->GetBinContent(ibin);
  }
  if (histArr.GetEntries() > 0) {
    char cname[50], hname[50];
    if (run > 0) {
      sprintf(cname, "c_HLT_%d", run);
      sprintf(name, "Run %d", run);
    } else {
      sprintf(cname, "c_HLTs");
      sprintf(name, "All runs");
    }
    sprintf(hname, " ");
    return plotHisto(cname, "", histArr, labels, color, hname, ymx0, logy, pos, 0.40, 0.01, -1, varbin, rebin, approve);
  } else {
    return 0;
  }
}

TCanvas* plotHisto(char* cname,
                   std::string HLT,
                   TObjArray& histArr,
                   std::vector<std::string>& labels,
                   std::vector<int>& color,
                   char* name,
                   double ymx0,
                   bool logy,
                   int pos,
                   double yloff,
                   double yhoff,
                   double xmax,
                   bool varbin,
                   int rebin0,
                   bool approve) {
  int nentry = histArr.GetEntries();
  double ymax = 10.;
  for (int i = 0; i < 10; ++i) {
    if (ymx0 < ymax)
      break;
    ymax *= 10.;
  }
  double ystep = ymax * 0.1;
  for (int i = 0; i < 9; ++i) {
    if (ymax - ystep < ymx0)
      break;
    ymax -= ystep;
  }
  double ymin(0);
  if (logy)
    ymin = 1;

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(kFALSE);
  gStyle->SetPadBorderMode(0);
  gStyle->SetCanvasBorderMode(0);
  if (approve)
    gStyle->SetOptStat(0);
  else
    gStyle->SetOptStat(1110);
  TCanvas* canvas = new TCanvas(cname, cname, 500, 500);
  gPad->SetTopMargin(yhoff);
  gPad->SetLeftMargin(0.15);
  gPad->SetRightMargin(0.025);
  gPad->SetBottomMargin(yloff);
  if (logy)
    canvas->SetLogy();
  double height = 0.08;
  double dx = (nentry > 2) ? 0.50 : 0.30;
  double dy = (nentry > 2) ? 0.12 : 0.04;
  double dy2 = 0.035;
  double xmin1 = (pos > 1) ? 0.375 : 0.75 - dx;
  double xmin2 = (pos > 1) ? 0.12 : 0.65;
  double xmin3 = (pos > 1) ? 0.15 : 0.75;
  double ymin1 = (pos % 2 == 0) ? (1.0 - yhoff - dy) : (yloff + 0.02);
  double ymin2 = (pos % 2 == 0) ? (0.96 - yhoff - nentry * height) : (yloff + 0.025 + nentry * height);
  double ymin3 = (pos % 2 == 0) ? (ymin2 - dy2) : (ymin2 + dy2);
  double dx2 = (approve) ? dx : 0.32;
  double dx3 = (approve) ? dx : 0.22;
  if (approve) {
    xmin1 = xmin2 = xmin3 = 0.975 - dx;
    ymin1 = (1.0 - yhoff - dy);
    ymin2 = ymin1 - dy2 - 0.01;
    ymin3 = ymin2 - dy2;
  }
  TLegend* legend = new TLegend(xmin1, ymin1, xmin1 + dx, ymin1 + dy);
  TPaveText *text, *text2;
  if (varbin || rebin0 == 1) {
    text = new TPaveText(xmin2, ymin2, xmin2 + dx2, ymin2 + dy2, "brNDC");
    text2 = new TPaveText(xmin3, ymin3, xmin3 + dx3, ymin3 + dy2, "brNDC");
  } else {
    text = new TPaveText(0.10, 0.95, dx2 + 0.10, dy2 + 0.95, "brNDC");
    text2 = new TPaveText(dx2 + 0.10, 0.95, dx2 + dx3 + 0.10, dy2 + 0.95, "brNDC");
  }
  legend->SetBorderSize(1);
  legend->SetFillColor(kWhite);
  char texts[200];
  sprintf(texts, cmsP.c_str());
  text2->AddText(texts);
  THStack* Hs = new THStack("hs2", " ");
  for (int i = 0; i < nentry; i++) {
    TH1D* h = (varbin) ? rebin((TH1D*)histArr[i], i) : (TH1D*)((TH1D*)histArr[i])->Rebin(rebin0);
    h->SetLineColor(color[i]);
    h->SetLineStyle(lstyle[i]);
    h->SetLineWidth(2);
    h->SetMarkerSize(1.0);
    double ymax0 = (varbin) ? ymax : rebin0 * ymax;
    h->GetYaxis()->SetRangeUser(ymin, ymax0);
    if (xmax > 0 && (!varbin))
      h->GetXaxis()->SetRangeUser(0, xmax);
    Hs->Add(h, "hist sames");
    legend->AddEntry(h, labels[i].c_str(), "l");
  }
  Hs->Draw("nostack");
  canvas->Update();
  Hs->GetHistogram()->GetXaxis()->SetTitle(name);
  Hs->GetHistogram()->GetXaxis()->SetLabelSize(0.035);
  Hs->GetHistogram()->GetYaxis()->SetTitleOffset(1.6);
  if (varbin) {
    Hs->GetHistogram()->GetYaxis()->SetTitle("Tracks/0.01");
  } else {
    Hs->GetHistogram()->GetYaxis()->SetTitle("Tracks");
    if (xmax > 0)
      Hs->GetHistogram()->GetXaxis()->SetRangeUser(0, xmax);
  }
  canvas->Modified();

  canvas->Update();
  if (!approve) {
    for (int i = 0; i < nentry; i++) {
      TH1D* h = (TH1D*)histArr[i];
      if (h != NULL) {
        TPaveStats* st1 = (TPaveStats*)h->GetListOfFunctions()->FindObject("stats");
        if (st1 != NULL) {
          if (pos % 2 == 0) {
            st1->SetY1NDC(1.0 - yhoff - (i + 1) * height);
            st1->SetY2NDC(1.0 - yhoff - i * height);
          } else {
            st1->SetY1NDC(yloff + 0.02 + i * height);
            st1->SetY2NDC(yloff + 0.02 + (i + 1) * height);
          }
          if (pos > 1) {
            st1->SetX1NDC(0.15);
            st1->SetX2NDC(.375);
          } else {
            st1->SetX1NDC(0.75);
            st1->SetX2NDC(.975);
          }
          st1->SetTextColor(colors[i]);
        }
      }
    }
  }
  legend->Draw("");
  if (HLT != "") {
    text->AddText(HLT.c_str());
    text->Draw("same");
  }
  text2->Draw("same");

  return canvas;
}

void getHistStats(TH1D* h,
                  int& entries,
                  int& integral,
                  double& mean,
                  double& meanE,
                  double& rms,
                  double& rmsE,
                  int& uflow,
                  int& oflow) {
  entries = h->GetEntries();
  integral = h->Integral();
  mean = h->GetMean();
  meanE = h->GetMeanError();
  rms = h->GetRMS();
  rmsE = h->GetRMSError();
  uflow = h->GetBinContent(0);
  oflow = h->GetBinContent(h->GetNbinsX() + 1);
}

TFitResultPtr getHistFitStats(
    TH1F* h, const char* formula, double xlow, double xup, unsigned int& nPar, double* par, double* epar) {
  TFitResultPtr fit = h->Fit(formula, "+qRB0", "", xlow, xup);
  nPar = fit->NPar();
  const std::vector<double> errors = fit->Errors();
  for (unsigned int i = 0; i < nPar; i++) {
    par[i] = fit->Value(i);
    epar[i] = errors[i];
  }
  return fit;
}

void setHistAttr(TH1F* h, int icol, int lwid, int ltype) {
  h->SetLineColor(icol);
  h->SetLineStyle(ltype);
  h->SetLineWidth(lwid);
  TF1* f = h->GetFunction("gaus");
  if (!f->IsZombie()) {
    f->SetLineColor(icol);
    f->SetLineStyle(2);
  }
}

double getWeightedMean(int npt, int Start, std::vector<double>& mean, std::vector<double>& emean) {
  double sumDen = 0, sumNum = 0;
  for (int i = Start; i < npt; i++) {
    if (mean[i] == 0.0 || emean[i] == 0.0) {
      sumNum += 0;
      sumDen += 0;
    } else {
      sumNum += mean[i] / (emean[i] * emean[i]);
      sumDen += 1.0 / (emean[i] * emean[i]);
    }
  }
  double WeightedMean = sumNum / sumDen;
  return WeightedMean;
}

TH1D* rebin(TH1D* histin, int indx) {
  std::string nameIn(histin->GetName());
  char name[200];
  sprintf(name, "%sRebin%d", nameIn.c_str(), indx);
  TH1D* hist = new TH1D(name, histin->GetXaxis()->GetTitle(), nbins, xbins);
  std::vector<double> cont;
  for (int i = 1; i <= histin->GetNbinsX(); ++i) {
    double value = histin->GetBinContent(i);
    cont.push_back(value);
  }
  for (int i = 0; i < nbins; ++i) {
    double totl = 0;
    int kount = 0;
    int klow = ibins[i];
    int khigh = ibins[i + 1];
    for (int k = klow; k < khigh; ++k) {
      totl += cont[k - 1];
      kount++;
    }
    if (kount > 0)
      totl /= kount;
    hist->SetBinContent(i + 1, totl);
  }
  return hist;
}
