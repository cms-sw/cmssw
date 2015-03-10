#include "TFile.h"
#include "TProfile2D.h"
#include "TH1.h"

#include <iostream>
#include <string>
#include <sstream>
#include <stdlib.h>

// ***************************************************************************************
// Program to read timing histograms in TT bins from DQM input files from two runs
// Creates output root file with timing histograms for EB/EE for each run and the diff.
// 
// Seth Cooper, October 2010
// $Id: CompareEcalFETimingDQM.cpp,v 1.1 2011/09/26 12:43:09 scooper Exp $


void usage()
{
  std::cout << "Usage:"
    << "  CompareEcalFETimingDQM Run1DQMFileBarrel Run1DQMFileEndcap Run2DQMFileBarrel Run2DQMFileEndcap; output is Run2-Run1."
    << std::endl;
  return;
}

void moveBinsTProfile2D(TProfile2D* myprof)
{
  int nxbins = myprof->GetNbinsX();
  int nybins = myprof->GetNbinsY();

  for(int i=0; i<=(nxbins+2)*(nybins+2); i++ )
  {
    Double_t binents = myprof->GetBinEntries(i);
    if(binents == 0)
    {
      myprof->SetBinEntries(i,1);
      myprof->SetBinContent(i,-1000);
    }
  }
  return;
}

std::string intToString(int num)
{
  using namespace std;
  ostringstream myStream;
  myStream << num << flush;
  return(myStream.str());
}

// Fish run number out of file name
int getRunNumber(std::string fileName)
{
  using namespace std;

  int runNumPos = fileName.find(".root");
  int Rpos = fileName.find("_R");
  if(runNumPos <= 0 || Rpos <= 0)
    return -1;

  string runNumString = fileName.substr(Rpos+2,runNumPos-Rpos-2);
  stringstream convert(runNumString);
  int runNumber;
  if(!(convert >> runNumber))
    runNumber = -1;
  return runNumber;
}


// Use DQM root files as input and make TT timing averages
int main(int argc, char* argv[])
{
  using namespace std;

  char* infile1 = argv[1];
  char* infile2 = argv[2];
  char* infile3 = argv[3];
  char* infile4 = argv[4];
  if(!infile1 || !infile2 || !infile3 || !infile4)
  {
    cout << "Error: Must specify 4 input files." << endl;
    usage();
    return -1;
  }

  int runNum1 = getRunNumber(string(infile1));
  if(runNum1 <= 0)
  {
    cout << "Error: Unusual root file name." << endl;
    return -2;
  }
  int runNum1again = getRunNumber(string(infile2));
  if(runNum1again != runNum1)
  {
    cout << "Error: Run numbers in first two files given are mismatched: "
      << runNum1 << " vs. " << runNum1again << endl;
    usage();
    return -3;
  }
  int runNum2 = getRunNumber(string(infile3));
  if(runNum2 <= 0)
  {
    cout << "Error: Unusual root file name (3rd file)." << endl;
    return -2;
  }
  int runNum2again = getRunNumber(string(infile4));
  if(runNum2again != runNum2)
  {
    cout << "Error: Run numbers in second two files given are mismatched: "
      << runNum2 << " vs. " << runNum2again << endl;
    usage();
    return -3;
  }
  
  cout << "Run number 1: " << runNum1 << endl;
  cout << "Run number 2: " << runNum2 << endl;

  TFile* file1 = TFile::Open(infile1);
  TFile* file2 = TFile::Open(infile2);
  TFile* file3 = TFile::Open(infile3);
  TFile* file4 = TFile::Open(infile4);
  // Look in file1/file2 and try to get the timing maps
  string firstPartOfPath = "DQMData/Run ";
  string path1 = firstPartOfPath;
  path1+=intToString(runNum1);
  string path1EB=path1;
  string path1EEP=path1;
  string path1EEM=path1;
  path1EB+="/EcalBarrel/Run summary/EBTimingTask/EBTMT timing map";
  path1EEP+="/EcalEndcap/Run summary/EETimingTask/EETMT timing map EE +";
  path1EEM+="/EcalEndcap/Run summary/EETimingTask/EETMT timing map EE -";
  TProfile2D* ttMapEB1orig = (TProfile2D*) file1->Get(path1EB.c_str());
  if(!ttMapEB1orig)
  {
    cout << "Error: EB plot not found in first input file, " << infile1 << endl;
    usage();
    return -4;
  }
  TProfile2D* ttMapEEP1orig = (TProfile2D*) file2->Get(path1EEP.c_str());
  TProfile2D* ttMapEEM1orig = (TProfile2D*) file2->Get(path1EEM.c_str());
  if(!ttMapEEP1orig || !ttMapEEM1orig)
  {
    cout << "Error: EE plot not found in second input file, " << infile2 << endl;
    usage();
    return -4;
  }
  // Look in file3/file4 and try to get the timing maps
  string path2 = firstPartOfPath;
  path2+=intToString(runNum2);
  string path2EB=path2;
  string path2EEP=path2;
  string path2EEM=path2;
  path2EB+="/EcalBarrel/Run summary/EBTimingTask/EBTMT timing map";
  path2EEP+="/EcalEndcap/Run summary/EETimingTask/EETMT timing map EE +";
  path2EEM+="/EcalEndcap/Run summary/EETimingTask/EETMT timing map EE -";
  TProfile2D* ttMapEB2orig = (TProfile2D*) file3->Get(path2EB.c_str());
  if(!ttMapEB2orig)
  {
    cout << "Error: EB plot not found in third input file, " << infile3 << endl;
    usage();
    return -5;
  }
  TProfile2D* ttMapEEP2orig = (TProfile2D*) file4->Get(path2EEP.c_str());
  TProfile2D* ttMapEEM2orig = (TProfile2D*) file4->Get(path2EEM.c_str());
  if(!ttMapEEP2orig || !ttMapEEM2orig)
  {
    cout << "Error: EE plot not found in fourth input file, " << infile4 << endl;
    usage();
    return -5;
  }

  // We should have good maps at this point; construct the TT differences
  string runNum1str = intToString(runNum1);
  string runNum2str = intToString(runNum2);
  string filename = "compareEcalFETimingDQM_";
  filename+=runNum2str;
  filename+="_";
  filename+=runNum1str;
  filename+=".root";
  TFile* output = new TFile(filename.c_str(),"recreate");
  output->cd();
  TH1F* timingTTDiffEBHist = new TH1F("timingTTDiffEB","Difference of trigger tower timing EB;ns",100,-5,5);
  TH1F* timingTTDiffEEPHist = new TH1F("timingTTDiffEEP","Difference of trigger tower timing EEP;ns",100,-5,5);
  TH1F* timingTTDiffEEMHist = new TH1F("timingTTDiffEEM","Difference of trigger tower timing EEM;ns",100,-5,5);
  TH1F* timingTTrun1EBHist = new TH1F("timingRun1TTEB","trigger tower timing EB;ns",100,-5,5);
  TH1F* timingTTrun1EEPHist = new TH1F("timingRun1TTEEP","trigger tower timing EEP;ns",100,-5,5);
  TH1F* timingTTrun1EEMHist = new TH1F("timingRun1TTEEM","trigger tower timing EEM;ns",100,-5,5);
  TH1F* timingTTrun2EBHist = new TH1F("timingRun2TTEB","trigger tower timing EB;ns",100,-5,5);
  TH1F* timingTTrun2EEPHist = new TH1F("timingRun2TTEEP","trigger tower timing EEP;ns",100,-5,5);
  TH1F* timingTTrun2EEMHist = new TH1F("timingRun2TTEEM","trigger tower timing EEM;ns",100,-5,5);
  TProfile2D* ttMapEB1 =  (TProfile2D*) ttMapEB1orig->Clone();
  TProfile2D* ttMapEB2 =  (TProfile2D*) ttMapEB2orig->Clone();
  TProfile2D* ttMapEEP1 = (TProfile2D*) ttMapEEP1orig->Clone();
  TProfile2D* ttMapEEP2 = (TProfile2D*) ttMapEEP2orig->Clone();
  TProfile2D* ttMapEEM1 = (TProfile2D*) ttMapEEM1orig->Clone();
  TProfile2D* ttMapEEM2 = (TProfile2D*) ttMapEEM2orig->Clone();
  // Run 1
  string ebMap1str = "TT Timing EB Run ";
  ebMap1str+=runNum1str;
  ttMapEB1->SetNameTitle(ebMap1str.c_str(),ebMap1str.c_str());
  timingTTrun1EBHist->SetTitle(ebMap1str.c_str());
  string eepMap1str = "TT Timing EE+ Run ";
  eepMap1str+=runNum1str;
  ttMapEEP1->SetNameTitle(eepMap1str.c_str(),eepMap1str.c_str());
  timingTTrun1EEPHist->SetTitle(eepMap1str.c_str());
  string eemMap1str = "TT Timing EE- Run ";
  eemMap1str+=runNum1str;
  ttMapEEM1->SetNameTitle(eemMap1str.c_str(),eemMap1str.c_str());
  timingTTrun1EEMHist->SetTitle(eepMap1str.c_str());
  // Run 2
  string ebMap2str = "TT Timing EB Run ";
  ebMap2str+=runNum2str;
  ttMapEB2->SetNameTitle(ebMap2str.c_str(),ebMap2str.c_str());
  timingTTrun2EBHist->SetTitle(ebMap2str.c_str());
  string eepMap2str = "TT Timing EE+ Run ";
  eepMap2str+=runNum2str;
  ttMapEEP2->SetNameTitle(eepMap2str.c_str(),eepMap2str.c_str());
  timingTTrun2EEPHist->SetTitle(eepMap2str.c_str());
  string eemMap2str = "TT Timing EE- Run ";
  eemMap2str+=runNum2str;
  ttMapEEM2->SetNameTitle(eemMap2str.c_str(),eemMap2str.c_str());
  timingTTrun2EEMHist->SetTitle(eepMap2str.c_str());

  //double ttEtaBins[36] = {-85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86 };
  //double ttPhiBins[73];
  //for (int i = 0; i < 73; ++i)
  //{
  //  ttPhiBins[i]=1+5*i;
  //}
  //TProfile2D* timingTTDiffMapEB = new TProfile2D("timingTTDiffMapEB","Difference of trigger tower timing EB;i#phi;i#eta",72,ttPhiBins,35,ttEtaBins);
  TProfile2D* timingTTDiffMapEB = new TProfile2D("timingTTDiffMapEB","Difference of trigger tower timing EB;i#phi;i#eta",72,0,360,34,-85, 85);
  TProfile2D* timingTTDiffMapEEP = new TProfile2D("timingTTDiffMapEEP","Difference of trigger tower timing EEP;ix;iy",100/5,0,100,100/5,0,100);  
  TProfile2D* timingTTDiffMapEEM = new TProfile2D("timingTTDiffMapEEM","Difference of trigger tower timing EEM;ix;iy",100/5,0,100,100/5,0,100);  

  // Create differences for EB
  int nxbins = ttMapEB1->GetNbinsX();
  int nybins = ttMapEB1->GetNbinsY();
  for(int x=1; x < nxbins+1; ++x)
  {
    for(int y=1; y < nybins+1; ++y)
    {
      int i = ttMapEB1->GetBin(x,y);
      double eb1 = ttMapEB1->GetBinContent(i)-50;
      double eb2 = ttMapEB2->GetBinContent(i)-50;
      double diff = eb2-eb1;
      if(ttMapEB1->GetBinEntries(i) > 0)
      {
        ttMapEB1->SetBinContent(i,eb1);
        ttMapEB1->SetBinEntries(i,1);
        timingTTrun1EBHist->Fill(eb1);
      }
      if(ttMapEB2->GetBinEntries(i) > 0)
      {
        ttMapEB2->SetBinContent(i,eb2);
        ttMapEB2->SetBinEntries(i,1);
        timingTTrun2EBHist->Fill(eb2);
      }
      if(ttMapEB1->GetBinEntries(i)==0 || ttMapEB2->GetBinEntries(i)==0)
        continue;
      timingTTDiffEBHist->Fill(diff);
      timingTTDiffMapEB->SetBinContent(i,diff);
      timingTTDiffMapEB->SetBinEntries(i,1);
    }
  }
  // Create differences for EE+
  nxbins = ttMapEEP1->GetNbinsX();
  nybins = ttMapEEP1->GetNbinsY();
  for(int i=0; i<=(nxbins+2)*(nybins+2); i++)
  {
    double eep1 = ttMapEEP1->GetBinContent(i)-50;
    double eep2 = ttMapEEP2->GetBinContent(i)-50;
    double diff = eep2-eep1;
    if(ttMapEEP1->GetBinEntries(i) > 0)
    {
      ttMapEEP1->SetBinContent(i,eep1);
      ttMapEEP1->SetBinEntries(i,1);
      timingTTrun1EEPHist->Fill(eep1);
    }
    if(ttMapEEP2->GetBinEntries(i) > 0)
    {
      ttMapEEP2->SetBinContent(i,eep2);
      ttMapEEP2->SetBinEntries(i,1);
      timingTTrun2EEPHist->Fill(eep2);
    }
    if(ttMapEEP1->GetBinEntries(i)==0 || ttMapEEP2->GetBinEntries(i)==0)
      continue;
    timingTTDiffEEPHist->Fill(diff);
    timingTTDiffMapEEP->SetBinContent(i,diff);
    timingTTDiffMapEEP->SetBinEntries(i,1);
  }
  // Create differences for EE-
  nxbins = ttMapEEM1->GetNbinsX();
  nybins = ttMapEEM1->GetNbinsY();
  for(int i=0; i<=(nxbins+2)*(nybins+2); i++)
  {
    double eem1 = ttMapEEM1->GetBinContent(i)-50;
    double eem2 = ttMapEEM2->GetBinContent(i)-50;
    double diff = eem2-eem1;
    if(ttMapEEM1->GetBinEntries(i) > 0)
    {
      ttMapEEM1->SetBinContent(i,eem1);
      ttMapEEM1->SetBinEntries(i,1);
      timingTTrun1EEMHist->Fill(eem1);
    }
    if(ttMapEEM1->GetBinEntries(i) > 0)
    {
      ttMapEEM2->SetBinContent(i,eem2);
      ttMapEEM2->SetBinEntries(i,1);
      timingTTrun2EEMHist->Fill(eem2);
    }
    if(ttMapEEM1->GetBinEntries(i)==0 || ttMapEEM2->GetBinEntries(i)==0)
      continue;
    timingTTDiffEEMHist->Fill(diff);
    timingTTDiffMapEEM->SetBinContent(i,diff);
    timingTTDiffMapEEM->SetBinEntries(i,1);
  }

  // Move bins without entries away from zero and set range
  moveBinsTProfile2D(timingTTDiffMapEB);
  moveBinsTProfile2D(timingTTDiffMapEEP);
  moveBinsTProfile2D(timingTTDiffMapEEM);
  moveBinsTProfile2D(ttMapEB1);
  moveBinsTProfile2D(ttMapEEP1);
  moveBinsTProfile2D(ttMapEEM1);
  moveBinsTProfile2D(ttMapEB2);
  moveBinsTProfile2D(ttMapEEP2);
  moveBinsTProfile2D(ttMapEEM2);
  timingTTDiffMapEB->SetMinimum(-1);
  timingTTDiffMapEEP->SetMinimum(-1);
  timingTTDiffMapEEM->SetMinimum(-1);
  ttMapEB1->SetMinimum(-1);
  ttMapEEP1->SetMinimum(-1);
  ttMapEEM1->SetMinimum(-1);
  ttMapEB2->SetMinimum(-1);
  ttMapEEP2->SetMinimum(-1);
  ttMapEEM2->SetMinimum(-1);
  timingTTDiffMapEB->SetMaximum(1);
  timingTTDiffMapEEP->SetMaximum(1);
  timingTTDiffMapEEM->SetMaximum(1);
  ttMapEB1->SetMaximum(1);
  ttMapEEP1->SetMaximum(1);
  ttMapEEM1->SetMaximum(1);
  ttMapEB2->SetMaximum(1);
  ttMapEEP2->SetMaximum(1);
  ttMapEEM2->SetMaximum(1);

  // Write
  timingTTDiffEBHist->Write();
  timingTTDiffEEPHist->Write();
  timingTTDiffEEMHist->Write();
  timingTTDiffMapEB->Write();
  timingTTDiffMapEEP->Write();
  timingTTDiffMapEEM->Write();
  ttMapEB1->Write();
  ttMapEEP1->Write();
  ttMapEEM1->Write();
  ttMapEB2->Write();
  ttMapEEP2->Write();
  ttMapEEM2->Write();
  timingTTrun1EBHist->Write();
  timingTTrun1EEPHist->Write();
  timingTTrun1EEMHist->Write();
  timingTTrun2EBHist->Write();
  timingTTrun2EEPHist->Write();
  timingTTrun2EEMHist->Write();


  output->Close();
}
