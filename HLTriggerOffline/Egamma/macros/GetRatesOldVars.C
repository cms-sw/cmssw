#include "TChain.h"
#include "iostream"
#include "fstream"
#include "iomanip"
#include "vector"
#include "map"

struct sample {
  TString name;
  TChain *chain;
  Double_t nEvents;
  Double_t xSection;
};

void GetRates() {
  Double_t luminosity = 1.0E32; // in cm^-2 s^-1
  Double_t conversion = 1.0E-27; // mb -> cm^2

  /* Set the cuts for which we calculate the rates */
  Double_t cutEt = 15.; // 19.
  Double_t cutIHcal = 3.; // 0.06
  Double_t cutEoverpBarrel = 1.5; // 0.45
  Double_t cutEoverpEndcap = 2.45; // 9.5
  Double_t cutItrack = 0.06; // 0.45
  /* ********************************************* */

  TString pathName = "SingleElecs"; // Egamma path to analyze: SingleElecs, RelaxedSingleElecs, DoubleElecs, RelaxedDoubleElecs, SinglePhots, etc.
                                // Note: High and Very High EM start with the same L1 as RelaxedSingleElecs

  ofstream myfile;
  myfile.open("RatesTableOldVars.txt"); // Output file name

  std::vector<sample> samples;
  sample thisSample;
  TString thisName;
  TChain *thisChain;
  Int_t sampleNum = 0;

  /* Set properties of input files */
  
  /* Sample input:
     thisChain = new TChain("Events");
     thisChain->Add("<path-to-root-file-1>");
     ...
     thisChain->Add("<path-to-root-file-N>");
     thisSample.name = "Name to be displayed in table";
     thisSample.chain=  thisChain;
     thisSample.nEvents = number of events in all files in chain combined;
     thisSample.xSection = cross section for process contributing to this chain;
     samples.push_back(thisSample);
  */
  
  thisChain = new TChain("Events");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_1.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_2.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_3.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_4.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_5.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_6.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_7.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_8.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_9.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_10.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_11.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_12.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_13.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_14.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_15.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_16.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_17.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_18.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_19.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_20.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_21.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_22.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_23.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_24.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_25.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_26.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_27.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_28.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_29.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_30.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_31.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_32.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_33.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_34.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_35.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_36.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_37.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_38.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_39.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_40.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_41.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_42.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_43.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_44.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_45.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_46.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_47.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_48.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_49.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_50.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_51.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_52.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_53.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_54.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_55.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_56.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_57.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_58.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_59.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_60.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_61.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_62.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_63.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_64.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_65.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_66.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_67.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_68.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_69.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_70.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_71.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_72.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_73.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_74.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_75.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_76.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_77.root");
  // thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_78.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_79.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_80.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_81.root");
  thisChain->Add("../test/HLTStudyData/crab_0_071206_144326/res/QCD-0-15-HLTVars_82.root");
  thisSample.name = "QCD 0-15";
  thisSample.chain = thisChain;
  thisSample.nEvents = 810000;
  thisSample.xSection = 52.0;
  samples.push_back(thisSample);

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

  thisChain = new TChain("Events");;
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-170-230-HLTVars-0.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-170-230-HLTVars-1.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-170-230-HLTVars-2.root");
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/QCD-170-230-HLTVars-3.root");
  thisSample.name = "QCD 170-230";
  thisSample.chain = thisChain;
  thisSample.nEvents = 400000;
  thisSample.xSection = 1.01E-4;
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
  samples.push_back(thisSample);

  thisChain = new TChain("Events");;
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/ZEE-HLTVars.root");
  thisSample.name = "Z->ee";
  thisSample.chain = thisChain;
  thisSample.nEvents = 3800;
  thisSample.xSection = 1.62E-6;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");;
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/WENU-HLTVars.root");
  thisSample.name = "W->ev";
  thisSample.chain = thisChain;
  thisSample.nEvents = 2000;
  thisSample.xSection = 1.72E-5;
  samples.push_back(thisSample);

  thisChain = new TChain("Events");;
  thisChain->Add("../../../../../CMSSW_1_6_0/src/HLTriggerOffline/Egamma/test/HLTStudyData/TTbar-HLTVars-1e.root");
  thisSample.name = "t-tbar";
  thisSample.chain = thisChain;
  thisSample.nEvents = 100000;
  thisSample.xSection = 8.33E-7;
  samples.push_back(thisSample);
   
  /* ****************************** */

  TString cutText = "", l1CutText = "";

  Double_t thisPass = 0., thisPassL1 = 0.;
  Double_t thisEff = 0., thisEffL1 = 0.;
  Double_t thisRate = 0., thisRateL1 = 0., rate = 0., rateL1 = 0.;
  Double_t thisEffErr = 0.;
  Double_t thisRateErr = 0., thisRateErrL1 = 0., rateErr = 0., rateErrL1 = 0.;

  /* Set the formula to apply cuts */
  cutText = "Sum$(";
  cutText += pathName;
  cutText += ".l1Match && ";
  cutText += pathName;
  cutText += ".Et > ";
  cutText += cutEt;
  cutText += " && ";
  cutText += pathName;
  cutText += ".IHcal < ";
  cutText += cutIHcal;
  cutText += " && ";
  cutText += pathName;
  cutText += ".pixMatch > 0 && ((";
  cutText += pathName;
  cutText += ".Eoverp < ";
  cutText += cutEoverpBarrel;
  cutText += " && fabs(";
  cutText += pathName;
  cutText += ".eta) < 1.5) || (";
  cutText += pathName;
  cutText += ".Eoverp < ";
  cutText += cutEoverpEndcap;
  cutText += " && fabs(";
  cutText += pathName;
  cutText += ".eta) > 1.5 && fabs(";
  cutText += pathName;
  cutText += ".eta) < 2.5)) && ";
  cutText += pathName;
  cutText += ".Itrack < ";
  cutText += cutItrack;
  //  cutText += " && SingleElecs.mcEt > 0.) >= 1"; Uncomment to require MC electron
  cutText += ") >= 1"; // Change to >= 2 for double cuts

  l1CutText = "Sum$(SingleElecsPT.Et > -999.)";

  /* Calculate raes and fill table */
  myfile<<"Rates and Efficiencies for Input Samples"<<endl;
  myfile<<"Name           X-Section (mb) Efficiency     Uncert.        L1 Rate (Hz)   Uncert. (Hz)   Rate (Hz)      Uncert. (Hz)"<<endl;
  myfile<<"--"<<endl;
  myfile.setf(ios::left);
  for (sampleNum = 0; sampleNum < (Int_t)samples.size(); sampleNum++) {
    thisPass = (Double_t)samples[sampleNum].chain->Draw("", cutText);
    thisPassL1 = (Double_t)samples[sampleNum].chain->Draw("", l1CutText);
    thisEff = thisPass / samples[sampleNum].nEvents;
    thisEffL1 = thisPassL1 / samples[sampleNum].nEvents;
    thisEffErr = sqrt(thisEff * (1. - thisEff) / samples[sampleNum].nEvents); // Simple binomial distribution uncertainty
    thisRate = thisEff * samples[sampleNum].xSection * luminosity * conversion;
    thisRateL1 = thisEffL1 * samples[sampleNum].xSection * luminosity * conversion;
    thisRateErr = thisEffErr * samples[sampleNum].xSection * luminosity * conversion;
    thisRateErrL1 = sqrt(thisEffL1 * (1. - thisEffL1) / samples[sampleNum].nEvents) * samples[sampleNum].xSection * luminosity * conversion;
    rate += thisRate;
    rateL1 += thisRateL1;
    rateErr = sqrt(rateErr*rateErr + thisRateErr*thisRateErr);
    rateErrL1 = sqrt(rateErrL1*rateErrL1 + thisRateErrL1*thisRateErrL1);
    thisName = samples[sampleNum].name;
    myfile<<setw(15)<<thisName<<setw(15)<<samples[sampleNum].xSection<<setw(15)<<setprecision(3)<<thisEff<<setw(15)<<setprecision(3)<<thisEffErr<<setw(15)<<setprecision(3)<<thisRateL1<<setw(15)<<setprecision(3)<<thisRateErrL1<<setw(15)<<setprecision(3)<<thisRate<<setw(15)<<setprecision(3)<<thisRateErr<<endl;
  }
  myfile<<"--"<<endl;
  myfile<<"Total                                                                                     "<<setw(15)<<setprecision(3)<<rate<<setw(15)<<setprecision(3)<<rateErr<<endl;
  myfile.close();
}
