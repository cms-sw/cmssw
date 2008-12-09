#include "TFile.h"
#include "TChain.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TRegexp.h"
#include "TGraphErrors.h"
#include "iostream"
#include "fstream"
#include "vector"
#include "map"
#include "TTreeFormula.h"

struct sample {
  TString name;
  TChain *chain;
  Double_t nEvents;
  Double_t xSection;
  Bool_t isSignal;
  Bool_t doPresel;
  Int_t nElecs;
  Double_t etPresel;
  Double_t effPresel;
};

struct path {
  TString name;
  Int_t nCandsCut;
};

void DoOptimization() {
  /* Function: DoOptimization()
     Performs the optimization algorithm as described in Jan. 10 TSG meeting.  Luminosity can be edited at the top.  The optimal cuts are written to myfile.
  */

  /* Set Luminosity to use.  Conversion factor to get rate in Hz. */
  Double_t luminosity = 1.0E32; // in cm^-2 s^-1
  Double_t conversion = 1.0E-27;

  /* Set output file name.  This file will contain a list of the optimal cuts on the various variables. */
  ofstream myfile;
  myfile.open("OptimalCutsNewVars.txt");

  /* Set initial cuts on the variables */
  Double_t cutEt = 15.;
  Double_t cutIHcal = 3.;
  Double_t cutEoverpBarrel = 1.5;
  Double_t cutEoverpEndcap = 2.45;
  Double_t cutItrack = 0.06;

  /* IMPORTANT: the size of this array must be 3^N, where N is the number of cuts that are varying.  This should way of doing things should be changed.  However,
     it is relatively complicated an I have not had a chance yet.  Keep in mind that if you change N, you MUST change many other things in the code. */
  Double_t newEff[243];

  /* Set threshold on rate. */
  Double_t rateLimit = 20.;

  /* Set the initial step sizes.  These sizes will be divided by 2 every time an optimal point is reached until they reach some threshold on the first step (also below) */
  Double_t stepSizes[5];
  stepSizes[0] = 1.;
  stepSizes[1] = 0.02;
  stepSizes[2] = 0.2;
  stepSizes[3] = 0.2;
  stepSizes[4] = 0.2;
  Double_t stepThresh = 0.25;

  /* Set the HLT paths to study.  Choices are (Relaxed)SingleElecs, (Relaxed)DoubleElecs.  The only difference between these is what L1 was used and whether L1 Non-Iso electrons are included
     For details on what cuts were used, see http://cmslxr.fnal.gov/lxr/source/HLTrigger/Egamma/data/EgammaHLTLocal_1e32.cff?v=CMSSW_1_6_7 */
  vector<path> paths;
  path thisPath;
  thisPath.name = "SingleElecs";
  thisPath.nCandsCut = 1; // Set to 2 for double paths
  paths.push_back(thisPath);

  /* More parameters (including samples) set below */

  /* Variable definitions */

  TChain *thisChain;

  vector<sample> samples;
  sample thisSample;

  Double_t pass = 0.;
  Double_t eff = 0., bestEff;
  Double_t rate = 0., rateL1 = 0., rateTot = 0.;
  Double_t thisCutEt = cutEt;
  Double_t thisCutIHcal = cutIHcal;
  Double_t thisCutEoverpBarrel = cutEoverpBarrel;
  Double_t thisCutEoverpEndcap = cutEoverpEndcap;
  Double_t thisCutItrack = cutItrack;

  Int_t pathNum, sampleNum;

  Int_t direction = 0, bestDir = 0;

  TString cutText = "";

  Bool_t isBetter = false;

  Int_t xSecNum = 0, fileNum = 0;

  /* Set samples as follows:
     thisChain = new TChain("Events");
     thisChain->Add("fileN.root"); where fileN.root is one of the root files contributing to the samples
     ...
     thisSample.name = name for display purposes if necessary;                                                                                                                                                   
     thisSample.chain = thisChain; always necessary                                                                                                                                                             
     thisSample.nEvents = number of events in sample BEFORE ANY CUTS; used to calculate rate                                                                                                                    
     thisSample.xSection = cross section for this sample in mb;                                                                                                                     
     thisSample.isSignal = set to true for signal sample(s);                                                                                                                                               
     thisSample.doPresel = set to true to do MC preselection on electrons (only used to mimic previous studies);                                                   
     thisSample.nElecs = number of electrons in MC sample (1 for W->ev, 2 for Z->ee, etc.) for preselection;                                                                               
     thisSample.etPresel = MC Et cut for preselection;                                                                                                      
     thisSample.effPresel = Currently, a pre-calculated efficiency for this cut is needed to do preselection.  See Lorenzo's note for theoretical values;                                     
     samples.push_back(thisSample); always needed at the end.
  */                     

  thisChain = new TChain("Events");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-0.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-1.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-2.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-3.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-4.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-5.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-6.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-7.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-8.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-9.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-10.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-11.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-12.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-13.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-14.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-15.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-16.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-17.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-18.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-19.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-20.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-21.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-22.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-23.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-24.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-25.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-26.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-27.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-28.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-29.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-30.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-31.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-32.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-33.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-34.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-35.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-36.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-37.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-38.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-39.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-40.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-41.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-42.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-43.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-44.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-45.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-46.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-47.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-48.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-49.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-50.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-51.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-52.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-53.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-54.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-55.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-56.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-57.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-58.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-59.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-60.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-61.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-62.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-63.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-64.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-65.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-66.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-67.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-68.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-69.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-70.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-71.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-72.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-73.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-74.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-75.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-76.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-77.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-78.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-79.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-80.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-81.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-82.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-83.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-84.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-85.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-86.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-87.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-88.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-89.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-90.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-91.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-92.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-93.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-94.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-95.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-96.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-97.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-98.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-99.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-100.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-101.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-102.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-103.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-104.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-105.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-106.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-107.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-108.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-109.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-110.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-111.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-112.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-113.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-114.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-115.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-116.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-117.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-118.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-119.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-120.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-121.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-122.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-123.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-124.root");
  thisChain->Add("../test/HLTStudyData/QCD-15-20/QCD-15-20-HLTVars-125.root");
  thisSample.name = "QCD 15-20";
  thisSample.chain = thisChain;
  thisSample.nEvents = 1260000;
  thisSample.xSection = 1.46;
  thisSample.isSignal = false;
  thisSample.doPresel = false;
  thisSample.nElecs = 0;
  thisSample.etPresel = 0.;
  thisSample.effPresel = 0.;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-20-30-HLTVars.root");
  thisSample.name = "QCD 20-30";
  thisSample.chain = thisChain;
  thisSample.nEvents = 100000;
  thisSample.xSection = 6.32E-1;
  thisSample.isSignal = false;
  thisSample.doPresel = false;
  thisSample.nElecs = 0;
  thisSample.etPresel = 0.;
  thisSample.effPresel = 0.;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-30-50-HLTVars.root");
  thisSample.name = "QCD 30-50";
  thisSample.chain = thisChain;
  thisSample.nEvents = 100000;
  thisSample.xSection = 1.63E-1;
  thisSample.isSignal = false;
  thisSample.doPresel = false;
  thisSample.nElecs = 0;
  thisSample.etPresel = 0.;
  thisSample.effPresel = 0.;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-HLTVars-1.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-HLTVars-2.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-HLTVars-3.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-HLTVars-4.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-HLTVars-5.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-HLTVars-6.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-HLTVars-7.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-HLTVars-8.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-HLTVars-9.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-HLTVars-10.root");
  thisSample.name = "QCD 50-80";
  thisSample.chain = thisChain;
  thisSample.nEvents = 100000;
  thisSample.xSection = 2.16E-2;
  thisSample.isSignal = false;
  thisSample.doPresel = false;
  thisSample.nElecs = 0;
  thisSample.etPresel = 0.;
  thisSample.effPresel = 0.;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");;
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-cfgs/QCD-HLTVars80120-0.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-cfgs/QCD-HLTVars80120-1.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-cfgs/QCD-HLTVars80120-2.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-cfgs/QCD-HLTVars80120-3.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-cfgs/QCD-HLTVars80120-4.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-cfgs/QCD-HLTVars80120-5.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-cfgs/QCD-HLTVars80120-6.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-cfgs/QCD-HLTVars80120-7.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-cfgs/QCD-HLTVars80120-8.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/QCD-cfgs/QCD-HLTVars80120-9.root");
  thisSample.name = "QCD 80-120";
  thisSample.chain = thisChain;
  thisSample.nEvents = 239993;
  thisSample.xSection = 3.08E-3;
  thisSample.isSignal = false;
  thisSample.doPresel = false;
  thisSample.nElecs = 0;
  thisSample.etPresel = 0.;
  thisSample.effPresel = 0.;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");;
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-120-170-HLTVars-2.root");
  thisSample.name = "QCD 120-170";
  thisSample.chain = thisChain;
  thisSample.nEvents = 11092;
  thisSample.xSection = 4.94E-4;
  thisSample.isSignal = false;
  thisSample.doPresel = false;
  thisSample.nElecs = 0;
  thisSample.etPresel = 0.;
  thisSample.effPresel = 0.;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");;
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-170-230-HLTVars-0.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-170-230-HLTVars-1.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-170-230-HLTVars-2.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-170-230-HLTVars-3.root");
  thisSample.name = "QCD 170-230";
  thisSample.chain = thisChain;
  thisSample.nEvents = 400000;
  thisSample.xSection = 1.01E-4;
  thisSample.isSignal = false;
  thisSample.doPresel = false;
  thisSample.nElecs = 0;
  thisSample.etPresel = 0.;
  thisSample.effPresel = 0.;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");;
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-230-300-HLTVars-0.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-230-300-HLTVars-1.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-230-300-HLTVars-2.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-230-300-HLTVars-3.root");
  thisSample.name = "QCD 230-300";
  thisSample.chain = thisChain;
  thisSample.nEvents = 400000;
  thisSample.xSection = 2.45E-5;
  thisSample.isSignal = false;
  thisSample.doPresel = false;
  thisSample.nElecs = 0;
  thisSample.etPresel = 0.;
  thisSample.effPresel = 0.;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");;
  thisChain->Add("../test/HLTStudyData/ZEE-HLTVars-NoPresel.root");
  thisSample.name = "Z->ee";
  thisSample.chain = thisChain;
  thisSample.nEvents = 3800;
  thisSample.xSection = 1.62E-6;
  thisSample.isSignal = false;
  thisSample.doPresel = true;
  thisSample.nElecs = 2;
  thisSample.etPresel = 5.;
  thisSample.effPresel = 0.504;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");
  thisChain->Add("../test/HLTStudyData/WENU-HLTVars-NoPresel.root");
  thisSample.name = "W->ev";
  thisSample.chain = thisChain;
  thisSample.nEvents = 2000;
  thisSample.xSection = 1.72E-5;
  thisSample.isSignal = true;
  thisSample.doPresel = true;
  thisSample.nElecs = 1;
  thisSample.etPresel = 20.;
  thisSample.effPresel = 0.463;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");;
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/TTbar-HLTVars-1e.root");
  thisSample.name = "t-tbar";
  thisSample.chain = thisChain;
  thisSample.nEvents = 100000;
  thisSample.xSection = 8.33E-7;
  thisSample.isSignal = false;
  thisSample.doPresel = false;
  thisSample.nElecs = 1;
  thisSample.etPresel = 0.;
  thisSample.effPresel = 0.;
  samples.push_back(thisSample);

  /* Now do the initial cut to set a base rate and efficiency */
  /* For each path, look at each sample and apply the cuts to get the efficiency and rate.  DOES NOT WORK WITH MULTIPLE PATHS AT ONCE YET */
  for (pathNum = 0; pathNum < paths.size(); pathNum++) {  
    rate = 0.;
    for (sampleNum = 0; sampleNum < samples.size(); sampleNum++) {
      /* Set the cut... Messy for now until update to add H/E, |1/E-1/p| */
      cutText = "Sum$(";
      cutText += paths[pathNum].name;
      cutText += ".l1Match && ";
      cutText += paths[pathNum].name;
      cutText += ".Et > ";
      cutText += thisCutEt;
      cutText += " && ";
      cutText += paths[pathNum].name;
      cutText += ".IHcal < ";
      cutText += thisCutIHcal;
      cutText += " && ";
      cutText += paths[pathNum].name;
      cutText += ".pixMatch > 0 && ((";
      cutText += paths[pathNum].name;  
      cutText += ".Eoverp < ";                                                                                                                                                                  
      cutText += thisCutEoverpBarrel;
      cutText += " && fabs(";                                                                                                                                                                            
      cutText += paths[pathNum].name;                                                                                                                                                                       
      cutText += ".eta) < 1.5) || (";                                                                                                                                                                  
      cutText += paths[pathNum].name;                                                                                                                                                                        
      cutText += ".Eoverp < ";                                                                                                                                                                  
      cutText += thisCutEoverpEndcap;                                                                                                             
      cutText += " && fabs(";                                                                                                                                                        
      cutText += paths[pathNum].name;                                                                                                                                          
      cutText += ".eta) > 1.5 && fabs(";
      cutText += paths[pathNum].name;
      cutText += ".eta) < 2.5)) && ";
      cutText += paths[pathNum].name;
      cutText += ".Itrack < ";
      cutText += thisCutItrack;
      cutText += ") >= ";
      cutText += paths[pathNum].nCandsCut;
      if (samples[sampleNum].doPresel) {
	cutText += " && Sum$(CaloVarss_hltCutVars_mcSingleElecs_EGAMMAHLT.obj.Et > ";
	cutText += samples[pathNum].etPresel;
	cutText += " && fabs(CaloVarss_hltCutVars_mcSingleElecs_EGAMMAHLT.obj.eta) < 2.7) >= ";
	cutText += samples[sampleNum].nElecs;
      }

      pass = (Double_t)samples[sampleNum].chain->Draw("", cutText);
      
      rate += pass / samples[sampleNum].nEvents * samples[sampleNum].xSection * luminosity * conversion;

      if (samples[sampleNum].isSignal) {
	bestEff = pass;
      }
    }
    if (rate > maxRate) {
      cout<<"Invalid initial choice!"<<endl;
      cout<<"Rate is "<<rate<<endl;
    }
    while (stepSizes[0] > stepThresh) {
      isBetter = false;
      cout<<cutEt<<", "<<cutIHcal<<", "<<cutEoverpBarrel<<". "<<cutEoverpEndcap<<", "<<cutItrack<<endl;
      for (direction = 0; direction < 241; direction++) {
	rate = 0.;
	if (direction % 3 == 0) thisCutEt = cutEt + stepSizes[0];
	else if  (direction % 3 == 1) thisCutEt = cutEt;
	else thisCutEt = cutEt - stepSizes[0];
	if ((direction / 3) % 3 == 0) thisCutIHcal = cutIHcal + stepSizes[1];
	else if ((direction / 3) % 3 == 1) thisCutIHcal = cutIHcal;
	else thisCutIHcal = cutIHcal - stepSizes[1];
	if ((direction / 9) % 3 == 0) thisCutEoverpBarrel = cutEoverpBarrel + stepSizes[2];
	else if ((direction / 9) % 3 == 1) thisCutEoverpBarrel = cutEoverpBarrel;
	else thisCutEoverpBarrel = cutEoverpBarrel - stepSizes[2];
	if ((direction / 27) % 3 == 0) thisCutEoverpEndcap = cutEoverpEndcap + stepSizes[3];
	else if ((direction / 27) % 3 == 1) thisCutEoverpEndcap = cutEoverpEndcap;
	else thisCutEoverpEndcap = cutEoverpEndcap - stepSizes[3];
	if ((direction / 81) % 3 == 0) thisCutItrack = cutItrack + stepSizes[4];
	else if ((direction / 81) % 3 == 1) thisCutItrack = cutItrack;
	else thisCutItrack = cutItrack - stepSizes[4];
	cout<<thisCutEt<<", "<<thisCutIHcal<<", "<<thisCutEoverpBarrel<<", "<<thisCutEoverpEndcap<<", "<<thisCutItrack<<endl;
	for (sampleNum = 0; sampleNum < samples.size(); sampleNum++) {
	  cutText = "Sum$(";
	  cutText += paths[pathNum].name;
	  cutText += ".l1Match && ";
	  cutText += paths[pathNum].name;
	  cutText += ".Et > ";
	  cutText += thisCutEt;
	  cutText += " && ";
	  cutText += paths[pathNum].name;
	  cutText += ".IHcal < ";
	  cutText += thisCutIHcal;
	  cutText += " && ";
	  cutText += paths[pathNum].name;
	  cutText += ".pixMatch > 0 && ((";
	  cutText += paths[pathNum].name;  
	  cutText += ".Eoverp < ";                                                                                                                                                                  
	  cutText += thisCutEoverpBarrel;
	  cutText += " && fabs(";                                                                                                                                                                            
	  cutText += paths[pathNum].name;                                                                                                                                                                       
	  cutText += ".eta) < 1.5) || (";                                                                                                                                                                  
	  cutText += paths[pathNum].name;                                                                                                                                                                        
	  cutText += ".Eoverp < ";                                                                                                                                                                  
	  cutText += thisCutEoverpEndcap;                                                                                                             
	  cutText += " && fabs(";                                                                                                                                                        
	  cutText += paths[pathNum].name;                                                                                                                                          
	  cutText += ".eta) > 1.5 && fabs(";
	  cutText += paths[pathNum].name;
	  cutText += ".eta) < 2.5)) && ";
	  cutText += paths[pathNum].name;
	  cutText += ".Itrack < ";
	  cutText += thisCutItrack;
	  cutText += ") >= ";
	  cutText += paths[pathNum].nCandsCut;
	  if (samples[sampleNum].doPresel) {
	    cutText += " && Sum$(CaloVarss_hltCutVars_mcSingleElecs_EGAMMAHLT.obj.Et > ";
	    cutText += samples[pathNum].etPresel;
	    cutText += " && fabs(CaloVarss_hltCutVars_mcSingleElecs_EGAMMAHLT.obj.eta) < 2.7) >= ";
	    cutText += samples[sampleNum].nElecs;
	  }

	  pass = (Double_t)samples[sampleNum].chain->Draw("", cutText);
	  
	  rate += pass / samples[sampleNum].nEvents * samples[sampleNum].xSection * luminosity * conversion;

	  if (samples[sampleNum].isSignal) {
	    newEff[direction] = pass;
	  }
	}	
	cout<<"Pass: "<<newEff[direction]<<", Rate: "<<rate<<endl; // To keep track of progress during output
	  
	if (rate < rateLimit && newEff[direction] > bestEff) {
	  bestEff = newEff[direction];
	  bestDir = direction;
	  isBetter = true;
	}
      }
      /* THIS PART SHOULD BE CHANGED.  Right now, if you change the number of cuts varying, you must edit this part non-trivially.  The Nth cut should have:
	 if ((bestDir / 3^(N-1)) % 3 == 0) cut = cut + stepSizes[N-1];
	 else if ((bestDir / 3^(N-1)) % 3 == 1) cut = cut;
	 else cut = cut - stepSizes[N-1];
      */
      if (isBetter) {
	cout<<"There is a better cut!"<<endl;
	if (bestDir % 3 == 0) cutEt = cutEt + stepSizes[0];
	else if  (bestDir % 3 == 1) cutEt = cutEt;
	else cutEt = cutEt - stepSizes[0];
	if ((bestDir / 3) % 3 == 0) cutIHcal = cutIHcal + stepSizes[1];
	else if ((bestDir / 3) % 3 == 1) cutIHcal = cutIHcal;
	else cutIHcal = cutIHcal - stepSizes[1];
	if ((bestDir / 9) % 3 == 0) cutEoverpBarrel = cutEoverpBarrel + stepSizes[2];
	else if ((bestDir / 9) % 3 == 1) cutEoverpBarrel = cutEoverpBarrel;
	else cutEoverpBarrel = cutEoverpBarrel - stepSizes[2];
	if ((bestDir / 27) % 3 == 0) cutEoverpEndcap = cutEoverpEndcap + stepSizes[3];
	else if ((bestDir / 27) % 3 == 1) cutEoverpEndcap = cutEoverpEndcap;
	else cutEoverpEndcap = cutEoverpEndcap - stepSizes[3];
	if ((bestDir / 81) % 3 == 0) cutItrack = cutItrack + stepSizes[4];
	else if ((bestDir / 81) % 3 == 1) cutItrack = cutItrack;
	else cutItrack = cutItrack - stepSizes[4];
      }
      else {
	cout<<"Nothing better... Shrinking window"<<endl;
	stepSizes[0] = stepSizes[0] / 2.;
	stepSizes[1] = stepSizes[1] / 2.;
	stepSizes[2] = stepSizes[2] / 2.;
	stepSizes[3] = stepSizes[3] / 2.;
	stepSizes[4] = stepSizes[4] / 2.;
      }
    }
  
    myfile<<"Optimal cuts: "<<endl;
    myfile<<"Et = "<<cutEt<<endl;
    myfile<<"IHcal = "<<cutIHcal<<endl;
    myfile<<"EoverpBarrel = "<<cutEoverpBarrel<<endl;
    myfile<<"EoverpEndcap = "<<cutEoverpEndcap<<endl;
    myfile<<"Itrack = "<<cutItrack<<endl;
    
  }

  myfile.close();

}
