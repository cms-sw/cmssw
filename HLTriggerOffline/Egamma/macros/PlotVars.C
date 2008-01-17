#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TString.h"
#include "TStyle.h"
#include "TEventList.h"
#include "TH1F.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TArrow.h"
#include "TTreeFormula.h"

struct filter {
  TString name;
  TString cut;
  Int_t pathNum;
  Int_t direction; // -1: <, 0: bool, 1: >
  Double_t hltBarrelCut;
  Double_t hltEndcapCut;
  Double_t maxCut;
};

struct path {
  TString name;
  Int_t nCandsCut;
  std::vector<filter> filters;
};

struct sample {
  TString name;
  TChain *chain;
  Double_t nEvents;
  Double_t xSection;
};

void PlotVars() {
  /* Set desired luminosity.  Conversion gives rate in Hz */
  Double_t luminosity = 1.0E32;
  Double_t conversion  = 1.0E-27;

  /* Set name to prefix image filenames with */
  TString name = "WenuOpt";

  /* Set the running mode: 0 - basic variable plot, 1 - electrons with optimal value of variable in given event, 2 - N-1 plots */
  Int_t mode = 1;

  /* Variable declarations */
  struct path thisPath;
  TString pathName;
  struct filter thisFilter;
  std::vector<filter> filters;

  std::vector<sample> samples;
  sample thisSample;
  TChain *thisChain;

  Int_t cutNum = 0;

  TTree *tempTree;
  TTreeFormula *thisFormula;
  Double_t best;
  Long64_t entry = 0, part = 0;

  /* Set the values of the cuts */
  Double_t cutEt = 17.;
  Double_t cutHoE = 0.07;
  Double_t cutEpMatch = 0.9;
  Double_t cutItrack = 0.15;

  /* Choose samples */
  thisChain = new TChain("Events");

  /*
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
  samples.push_back(thisSample);

  thisChain = new TChain("Events");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-20-30-HLTVars.root");
  thisSample.name = "QCD 20-30";
  thisSample.chain = thisChain;
  thisSample.nEvents = 100000;
  thisSample.xSection = 6.32E-1;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-30-50-HLTVars.root");
  thisSample.name = "QCD 30-50";
  thisSample.chain = thisChain;
  thisSample.nEvents = 100000;
  thisSample.xSection = 1.63E-1;
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
  samples.push_back(thisSample);

  thisChain = new TChain("Events");;
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-120-170-HLTVars-2.root");
  thisSample.name = "QCD 120-170";
  thisSample.chain = thisChain;
  thisSample.nEvents = 11092;
  thisSample.xSection = 4.94E-4;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-170-230-HLTVars-0.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-170-230-HLTVars-1.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-170-230-HLTVars-2.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-170-230-HLTVars-3.root");
  thisSample.name = "QCD 170-230";
  thisSample.chain = thisChain;
  thisSample.nEvents = 400000;
  thisSample.xSection = 1.01E-4;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-230-300-HLTVars-0.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-230-300-HLTVars-1.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-230-300-HLTVars-2.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-230-300-HLTVars-3.root");
  thisSample.name = "QCD 230-300";
  thisSample.chain = thisChain;
  thisSample.nEvents = 400000;
  thisSample.xSection = 2.45E-5;
  samples.push_back(thisSample);
  */

  /*
  thisChain = new TChain("Events");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/ZEE-HLTVars.root");
  thisSample.name = "Z->ee";
  thisSample.chain = thisChain;
  thisSample.nEvents = 3800;
  thisSample.xSection = 1.62E-6;
  samples.push_back(thisSample);
  */

  thisChain = new TChain("Events");
  thisChain->Add("../test/HLTStudyData/WENU-HLTVars-NoPresel.root");
  thisSample.name = "W->ev";
  thisSample.chain = thisChain;
  thisSample.nEvents = 2000;
  thisSample.xSection = 1.72E-5;
  samples.push_back(thisSample);

  /*
  thisChain = new TChain("Events");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/TTbar-HLTVars-1e.root");
  thisSample.name = "TTBar";
  thisSample.chain = thisChain;
  thisSample.nEvents = 100000;
  thisSample.xSection = 8.33E-7;
  samples.push_back(thisSample);
  */

  /* Selec path: (Relaxed)SingleElecs, (Relaxed)DoubleElecs */
  pathName = "SingleElecs.";
  thisPath.name = pathName;
  thisPath.nCandsCut = 1;
  /* Define the filters that go into the HLT.  hltBarrelCut, hltEndcapCut now redundant but still necessary */
  thisFilter.name = "Et";
  thisFilter.cut = thisPath.name; // this sets the text to go with the cut for each filter
  thisFilter.cut += "Et > ";
  thisFilter.cut += cutEt;
  thisFilter.pathNum = 0;
  thisFilter.direction = 1;
  thisFilter.hltBarrelCut = 17.; // Still being used to position arrows; should be removed (or cut* variables should be removed)
  thisFilter.hltEndcapCut = 17.;
  thisFilter.maxCut = 60.;
  filters.push_back(thisFilter);
  thisFilter.name = "HOE";
  thisFilter.cut = thisPath.name;
  thisFilter.cut += "IHcal / ";
  thisFilter.cut += thisPath.name;
  thisFilter.cut += "Et < ";
  thisFilter.cut += cutHoE;
  thisFilter.pathNum = 0;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 0.07;
  thisFilter.hltEndcapCut = 0.07;
  thisFilter.maxCut = 0.15;
  filters.push_back(thisFilter);
  thisFilter.name = "EpMatch";
  thisFilter.cut = "fabs((1. - ";
  thisFilter.cut += thisPath.name;
  thisFilter.cut += "Eoverp) * sin(2.*atan(exp(-";
  thisFilter.cut += thisPath.name;
  thisFilter.cut += "eta))) / ";
  thisFilter.cut += thisPath.name;
  thisFilter.cut += "Et) < ";
  thisFilter.cut += cutEpMatch;
  thisFilter.pathNum = 1;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 0.9;
  thisFilter.hltEndcapCut = 0.9;
  thisFilter.maxCut = 2.;
  filters.push_back(thisFilter);
  thisFilter.name = "Itrack";
  thisFilter.cut = thisPath.name;
  thisFilter.cut += "Itrack < ";
  thisFilter.cut += cutItrack;
  cout<<thisFilter.cut<<endl;
  thisFilter.pathNum = 1;
  thisFilter.direction = -1;
  thisFilter.hltBarrelCut = 0.15;
  thisFilter.hltEndcapCut = 0.15;
  thisFilter.maxCut = 0.5;
  filters.push_back(thisFilter);
  thisPath.filters = filters;
  filters.clear();

  TString cutText;

  TString plotText = "", plotFileName = "", plotTitle, thisHName = "";
  Double_t thisBCut = 0., thisECut = 0.;
  TString filterName;
  Long64_t sampleNum = 0, filterNum = 0;
  Double_t nPass = 0, oldNPass = 0, errNPass = 0, eff = 0, errEff = 0;
  vector<TH1F*> hists;
  TCanvas *myCanvas = new TCanvas("myCanvas", "Cut Variable", 700, 300);
  TArrow *barrelCutArr;
  TArrow *endcapCutArr;
  gStyle->SetOptStat(0);
  for (filterNum = 0; filterNum < (thisPath.filters).size(); filterNum++) {
    filterName = (thisPath.filters)[filterNum].name;
    //    thisHist = new TH1F("thisHist", filterName, 50, 0, (thisPath.filters)[filterNum].maxCut);
    THStack thisHist("thisHist", filterName);
    cutText = "";
    /* First set any cuts we would like on the plots */
    if (mode < 2) {
      cutText = pathName;
      cutText += "mcEt > 0."; // Check if MC elec matches reco Elec... optional I suppose
    }
    else if (mode == 2) {
      cutText = pathName;
      cutText += "l1Match && ";
      cutText += pathName;
      cutText += "pixMatch > 0";
      for (cutNum = 0; cutNum < (thisPath.filters).size(); cutNum++) {
	if (cutNum != filterNum) {
	  cutText += " && ";
	  cutText += filters[cutNum].cut;
	}
      }
    }
    
    hists.resize(samples.size());
    /* Now add each sample to the stack */
    for (sampleNum = 0; sampleNum < samples.size(); sampleNum++) {
      thisHName = "thisH";
      thisHName += sampleNum;
      if (filterName == "HOE") {
	plotText = pathName;
	plotText += "IHcal / ";
	plotText += pathName;
	plotText += "Et";
      }
      else if (filterName == "EpMatch") {
	plotText ="fabs(1. / ";
	plotText += pathName;
	plotText += "Et - ";
	plotText += pathName;
	plotText += "Eoverp / ";
	plotText += pathName;
	plotText += "Et) * sin(2*atan(exp(-";
	plotText += pathName;
	plotText += "eta)))";
      }
      else {
	plotText = pathName;
	plotText += filterName;
      }
      hists[sampleNum] = new TH1F(thisHName, filterName, 50, 0, (thisPath.filters)[filterNum].maxCut);
      if (mode == 2) {
	plotText += ">>";
	plotText += thisHName;
	samples[sampleNum].chain->Draw(plotText, cutText);
      }
      else if (mode == 1) {
	tempTree = samples[sampleNum].chain; // Must work with tree (as opposed to chain for this method... ROOT quirk
	thisFormula = new TTreeFormula("thisFormula", plotText, tempTree);
	for (entry = 0; entry < tempTree->GetEntries(); entry++) {
	  tempTree->LoadTree(entry);
	  if (thisFormula->GetNdata() == 0) {
	    thisFormula->EvalInstance(0); // Do eval instance even if nothing is there... another ROOT quirk
	  }
	  if (filters[filterNum].direction == 1) {
	    best = 0.;
	  }
	  else {
	    best = 100000.;
	  }
	  for (part = 0; part < thisFormula->GetNdata(); part++) {
	    if (filters[filterNum].direction == 1) {
	      if (thisFormula->EvalInstance(part) > best) {
		best = thisFormula->EvalInstance(part);                                                                                      
	      }                                                                                                                                                          
	    }                                                  
	    else if (filters[filterNum].direction == -1) {
	      if (thisFormula->EvalInstance(part) < best) {
		best = thisFormula->EvalInstance(part);
	      }
	    }
	  }
	  hists[sampleNum]->Fill(best);
	}
      delete thisFormula;
      }
      hists[sampleNum]->Scale(samples[sampleNum].xSection * luminosity * conversion / samples[sampleNum].nEvents);
      hists[sampleNum]->SetFillColor(sampleNum+1);
      plotTitle = filterName;
      plotTitle += ";";
      plotTitle += filterName;
      plotTitle += ";";
      plotTitle += "Rate (Hz)";
      hists[sampleNum]->SetTitle(plotTitle);
      thisHist.Add(hists[sampleNum]);
    }
    thisHist.Draw();
    
    /* Now just add the arrows and we're done */
    barrelCutArr = new TArrow((thisPath.filters)[filterNum].hltBarrelCut, -0.1, (thisPath.filters)[filterNum].hltBarrelCut, 0.);
    barrelCutArr->SetLineColor(2);
    barrelCutArr->Draw();
    if ( (thisPath.filters)[filterNum].hltBarrelCut !=  (thisPath.filters)[filterNum].hltEndcapCut) {
      endcapCutArr = new  TArrow((thisPath.filters)[filterNum].hltEndcapCut, -0.1, (thisPath.filters)[filterNum].hltEndcapCut, 0.);
      endcapCutArr->SetLineColor(4);
      endcapCutArr->Draw();
    }
    plotFileName = "images/";
    plotFileName += name;
    plotFileName += filterName;
    plotFileName += pathName;
    plotFileName += "gif";
    myCanvas->Print(plotFileName);
    /* Clean-up a little */ 
    delete barrelCutArr;
    for (sampleNum = 0; sampleNum < samples.size(); sampleNum++) {
      delete hists[sampleNum];
    }
    hists.clear();
  }
}
