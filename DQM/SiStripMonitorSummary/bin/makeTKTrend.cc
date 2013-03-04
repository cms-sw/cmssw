#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <sstream>

#include "TFile.h"
#include "TVectorT.h"
#include "TGraph.h"
#include "TGaxis.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TFile.h"

void makeTKTrend(const char* inFileName, const char* outFileName, std::string subDetName, std::string partName, const unsigned int partNumber);

int main(int argc , char *argv[]) {

  if(argc==6) {
    char* inFileName = argv[1];
    char* outFileName = argv[2];
    char* subDetName = argv[3];
    char* partName = argv[4];
    int partNumber = atoi(argv[5]);

    std::cout << "ready to make trend plots from " 
	      << inFileName << " to " << outFileName << " for " 
	      << subDetName << " " << partName << " " << partNumber << std::endl;

    
    int returncode = 0;
    makeTKTrend(inFileName,outFileName,subDetName,partName,partNumber);

    return  returncode;

  }
  else {std::cout << "Too few arguments: " << argc << std::endl; return -1; }

  return -9;

}

void makeTKTrend(const char* inFileName, const char* outFileName, std::string subDetName, std::string partName, const unsigned int partNumber)
{
  // Maps <Run number, nBad>
  std::map<unsigned int, unsigned int> badModulesTK;
  std::map<unsigned int, unsigned int> badFibersTK;
  std::map<unsigned int, unsigned int> badAPVsTK;
  std::map<unsigned int, unsigned int> badStripsTK;
  std::map<unsigned int, unsigned int> badStripsFromAPVsTK;
  std::map<unsigned int, unsigned int> allBadStripsTK;
  
  std::ostringstream oss;
  
  if(partName=="All") partName="";

//   // Number of modules, fibers, APVs, strips for each tracker part
//   std::vector< std::string > tkParts;
//   tkParts.push_back("Tracker");
//   tkParts.push_back("TIB");
//   tkParts.push_back("TID");
//   tkParts.push_back("TOB");
//   tkParts.push_back("TEC");
//   tkParts.push_back("TIB Layer 1");
//   tkParts.push_back("TIB Layer 2");
//   tkParts.push_back("TIB Layer 3");
//   tkParts.push_back("TIB Layer 4");
//   tkParts.push_back("TID+ Disk 1");
//   tkParts.push_back("TID+ Disk 2");
//   tkParts.push_back("TID+ Disk 3");
//   tkParts.push_back("TID- Disk 1");
//   tkParts.push_back("TID- Disk 2");
//   tkParts.push_back("TID- Disk 3");
//   tkParts.push_back("TOB Layer 1");
//   tkParts.push_back("TOB Layer 2");
//   tkParts.push_back("TOB Layer 3");
//   tkParts.push_back("TOB Layer 4");
//   tkParts.push_back("TOB Layer 5");
//   tkParts.push_back("TOB Layer 6");
//   tkParts.push_back("TEC+ Disk 1");
//   tkParts.push_back("TEC+ Disk 2");
//   tkParts.push_back("TEC+ Disk 3");
//   tkParts.push_back("TEC+ Disk 4");
//   tkParts.push_back("TEC+ Disk 5");
//   tkParts.push_back("TEC+ Disk 6");
//   tkParts.push_back("TEC+ Disk 7");
//   tkParts.push_back("TEC+ Disk 8");
//   tkParts.push_back("TEC+ Disk 9");
//   tkParts.push_back("TEC- Disk 1");
//   tkParts.push_back("TEC- Disk 2");
//   tkParts.push_back("TEC- Disk 3");
//   tkParts.push_back("TEC- Disk 4");
//   tkParts.push_back("TEC- Disk 5");
//   tkParts.push_back("TEC- Disk 6");
//   tkParts.push_back("TEC- Disk 7");
//   tkParts.push_back("TEC- Disk 8");
//   tkParts.push_back("TEC- Disk 9");
//   
//   std::vector<unsigned int> nModules;
//   nModules.push_back(15148);
//   nModules.push_back(2724);
//   nModules.push_back(816);
//   nModules.push_back(5208);
//   nModules.push_back(6400);
//   nModules.push_back(672);
//   nModules.push_back(864);
//   nModules.push_back(540);
//   nModules.push_back(648);
//   nModules.push_back(136);
//   nModules.push_back(136);
//   nModules.push_back(136);
//   nModules.push_back(136);
//   nModules.push_back(136);
//   nModules.push_back(136);
//   nModules.push_back(1008);
//   nModules.push_back(1152);
//   nModules.push_back(648);
//   nModules.push_back(720);
//   nModules.push_back(792);
//   nModules.push_back(888);
//   nModules.push_back(408);
//   nModules.push_back(408);
//   nModules.push_back(408);
//   nModules.push_back(360);
//   nModules.push_back(360);
//   nModules.push_back(360);
//   nModules.push_back(312);
//   nModules.push_back(312);
//   nModules.push_back(272);
//   nModules.push_back(408);
//   nModules.push_back(408);
//   nModules.push_back(408);
//   nModules.push_back(360);
//   nModules.push_back(360);
//   nModules.push_back(360);
//   nModules.push_back(312);
//   nModules.push_back(312);
//   nModules.push_back(272);
//   
//   std::vector<unsigned int> nFibers;
//   nFibers.push_back(36392);
//   nFibers.push_back(6984);
//   nFibers.push_back(2208);
//   nFibers.push_back(12096);
//   nFibers.push_back(15104);
//   nFibers.push_back(2016);
//   nFibers.push_back(2592);
//   nFibers.push_back(1080);
//   nFibers.push_back(1296);
//   nFibers.push_back(368);
//   nFibers.push_back(368);
//   nFibers.push_back(368);
//   nFibers.push_back(368);
//   nFibers.push_back(368);
//   nFibers.push_back(368);
//   nFibers.push_back(2016);
//   nFibers.push_back(2304);
//   nFibers.push_back(1296);
//   nFibers.push_back(1440);
//   nFibers.push_back(2376);
//   nFibers.push_back(2664);
//   nFibers.push_back(992);
//   nFibers.push_back(992);
//   nFibers.push_back(992);
//   nFibers.push_back(848);
//   nFibers.push_back(848);
//   nFibers.push_back(848);
//   nFibers.push_back(704);
//   nFibers.push_back(704);
//   nFibers.push_back(624);
//   nFibers.push_back(992);
//   nFibers.push_back(992);
//   nFibers.push_back(992);
//   nFibers.push_back(848);
//   nFibers.push_back(848);
//   nFibers.push_back(848);
//   nFibers.push_back(704);
//   nFibers.push_back(704);
//   nFibers.push_back(624);
//   
//   std::vector<unsigned int> nAPVs;
//   nAPVs.push_back(72784);
//   nAPVs.push_back(13968);
//   nAPVs.push_back(4416);
//   nAPVs.push_back(24192);
//   nAPVs.push_back(30208);
//   nAPVs.push_back(4032);
//   nAPVs.push_back(5184);
//   nAPVs.push_back(2160);
//   nAPVs.push_back(2592);
//   nAPVs.push_back(736);
//   nAPVs.push_back(736);
//   nAPVs.push_back(736);
//   nAPVs.push_back(736);
//   nAPVs.push_back(736);
//   nAPVs.push_back(736);
//   nAPVs.push_back(4032);
//   nAPVs.push_back(4608);
//   nAPVs.push_back(2592);
//   nAPVs.push_back(2880);
//   nAPVs.push_back(4752);
//   nAPVs.push_back(5328);
//   nAPVs.push_back(1984);
//   nAPVs.push_back(1984);
//   nAPVs.push_back(1984);
//   nAPVs.push_back(1696);
//   nAPVs.push_back(1696);
//   nAPVs.push_back(1696);
//   nAPVs.push_back(1408);
//   nAPVs.push_back(1408);
//   nAPVs.push_back(1248);
//   nAPVs.push_back(1984);
//   nAPVs.push_back(1984);
//   nAPVs.push_back(1984);
//   nAPVs.push_back(1696);
//   nAPVs.push_back(1696);
//   nAPVs.push_back(1696);
//   nAPVs.push_back(1408);
//   nAPVs.push_back(1408);
//   nAPVs.push_back(1248);
//   
//   std::vector<unsigned int> nStrips;
//   nStrips.push_back(9316352);
//   nStrips.push_back(1787904);
//   nStrips.push_back(565248);
//   nStrips.push_back(3096576);
//   nStrips.push_back(3866624);
//   nStrips.push_back(516096);
//   nStrips.push_back(663552);
//   nStrips.push_back(276480);
//   nStrips.push_back(331776);
//   nStrips.push_back(94208);
//   nStrips.push_back(94208);
//   nStrips.push_back(94208);
//   nStrips.push_back(94208);
//   nStrips.push_back(94208);
//   nStrips.push_back(94208);
//   nStrips.push_back(516096);
//   nStrips.push_back(589824);
//   nStrips.push_back(331776);
//   nStrips.push_back(368640);
//   nStrips.push_back(608256);
//   nStrips.push_back(681984);
//   nStrips.push_back(253952);
//   nStrips.push_back(253952);
//   nStrips.push_back(253952);
//   nStrips.push_back(217088);
//   nStrips.push_back(217088);
//   nStrips.push_back(217088);
//   nStrips.push_back(180224);
//   nStrips.push_back(180224);
//   nStrips.push_back(159744);
//   nStrips.push_back(253952);
//   nStrips.push_back(253952);
//   nStrips.push_back(253952);
//   nStrips.push_back(217088);
//   nStrips.push_back(217088);
//   nStrips.push_back(217088);
//   nStrips.push_back(180224);
//   nStrips.push_back(180224);
//   nStrips.push_back(159744);
//   
//   // Map with <name of tracker part, count of channels in the part>
//   std::map<std::string, unsigned int> allModulesTK;
//   std::map<std::string, unsigned int> allFibersTK;
//   std::map<std::string, unsigned int> allAPVsTK;
//   std::map<std::string, unsigned int> allStripsTK;
//   for(unsigned int i = 0; i < tkParts.size(); i++)
//   {
//     allModulesTK[tkParts[i].c_str()] = nModules[i];
//     allFibersTK[tkParts[i].c_str()] = nFibers[i];
//     allAPVsTK[tkParts[i].c_str()] = nAPVs[i];
//     allStripsTK[tkParts[i].c_str()] = nStrips[i];
//   }
//   
// //   for(std::map< std::string, unsigned int>::iterator it = allStripsTK.begin(); it != allStripsTK.end(); it++)
// //   {
// //     std::cout << it->first.c_str() << " " << it->second << std::endl;
// //   }
//   
//   // Finds number of channels from above map
//   std::string completePartName = subDetName;
//   if(partName.compare("") != 0)
//     completePartName += " " + partName;
//   if(partNumber != 0)
//   {
//     oss.str("");
//     oss << partNumber;
//     completePartName += " " + oss.str();
//   }
  
//   // Total number of channels in currently processed map
//   const unsigned int nModulesInPart = allModulesTK[completePartName.c_str()];
//   const unsigned int nFibersInPart = allFibersTK[completePartName.c_str()];
//   const unsigned int nAPVsInPart = allAPVsTK[completePartName.c_str()];
//   const unsigned int nStripsInPart = allStripsTK[completePartName.c_str()];
  
  // Read input file
  std::ifstream resultsFile(inFileName);
  unsigned int runIOV;
  unsigned int values[6];
  do
  {
    resultsFile >> runIOV;
    resultsFile >> values[0] >> values[1] >> values[2] >> values[3] >> values[4] >> values[5];
    //    std::cout << runIOV << " " << values[0] << " " << values[1] << " " << values[2] << " " << values[3] << " " << values[4] << " " << values[5] << std::endl;
    badModulesTK[runIOV]=values[0];
    badFibersTK[runIOV]=values[1];
    badAPVsTK[runIOV]=values[2];
    badStripsTK[runIOV]=values[3];
    badStripsFromAPVsTK[runIOV]=values[4];
    allBadStripsTK[runIOV]=values[5];
  }
  while(!resultsFile.eof());
  
  const unsigned int IOVSize = badStripsTK.size();
  
  // Create histograms
  std::string histoName;
  std::string histoTitle;
  
  oss.str("");
  histoName = "hBadModules" + subDetName + partName;
  if(partNumber!=0)
  {
    oss << partNumber;
    histoName += oss.str();
  }
  oss.str("");
  histoTitle = "Bad modules in " + subDetName;
  if(partName!="")
  {
    histoTitle += " " + partName;
  }
  if(partNumber!=0)
  {
    oss << partNumber;
    histoTitle += " " + oss.str();
  }
  TH1F* hBadModulesTK = new TH1F(histoName.c_str(), histoTitle.c_str(), IOVSize, 0.5, IOVSize+0.5);

  oss.str("");
  histoName = "hBadFibers" + subDetName + partName;
  if(partNumber!=0)
  {
    oss << partNumber;
    histoName += oss.str();
  }
  oss.str("");
  histoTitle = "Bad fibers in " + subDetName;
  if(partName!="")
  {
    histoTitle += " " + partName;
  }
  if(partNumber!=0)
  {
    oss << partNumber;
    histoTitle += " " + oss.str();
  }
  TH1F* hBadFibersTK = new TH1F(histoName.c_str(), histoTitle.c_str(), IOVSize, 0.5, IOVSize+0.5);
  
  oss.str("");
  histoName = "hBadAPVs" + subDetName + partName;
  if(partNumber!=0)
  {
    oss << partNumber;
    histoName += oss.str();
  }
  oss.str("");
  histoTitle = "Bad APVs in " + subDetName;
  if(partName!="")
  {
    histoTitle += " " + partName;
  }
  if(partNumber!=0)
  {
    oss << partNumber;
    histoTitle += " " + oss.str();
  }
  TH1F* hBadAPVsTK = new TH1F(histoName.c_str(), histoTitle.c_str(), IOVSize, 0.5, IOVSize+0.5);
  
  oss.str("");
  histoName = "hBadStrips" + subDetName + partName;
  if(partNumber!=0)
  {
    oss << partNumber;
    histoName += oss.str();
  }
  oss.str("");
  histoTitle = "Bad strips in " + subDetName;
  if(partName!="")
  {
    histoTitle += " " + partName;
  }
  if(partNumber!=0)
  {
    oss << partNumber;
    histoTitle += " " + oss.str();
  }
  TH1F* hBadStripsTK = new TH1F(histoName.c_str(), histoTitle.c_str(), IOVSize, 0.5, IOVSize+0.5);
  
  oss.str("");
  histoName = "hBadStripsFromAPVs" + subDetName + partName;
  if(partNumber!=0)
  {
    oss << partNumber;
    histoName += oss.str();
  }
  oss.str("");
  histoTitle = "Bad strips from APVs in " + subDetName;
  if(partName!="")
  {
    histoTitle += " " + partName;
  }
  if(partNumber!=0)
  {
    oss << partNumber;
    histoTitle += " " + oss.str();
  }
  TH1F* hBadStripsFromAPVsTK = new TH1F(histoName.c_str(), histoTitle.c_str(), IOVSize, 0.5, IOVSize+0.5);
  
  oss.str("");
  histoName = "hAllBadStrips" + subDetName + partName;
  if(partNumber!=0)
  {
    oss << partNumber;
    histoName += oss.str();
  }
  oss.str("");
  histoTitle = "All bad strips in " + subDetName;
  if(partName!="")
  {
    histoTitle += " " + partName;
  }
  if(partNumber!=0)
  {
    oss << partNumber;
    histoTitle += " " + oss.str();
  }
  TH1F* hAllBadStripsTK = new TH1F(histoName.c_str(), histoTitle.c_str(), IOVSize, 0.5, IOVSize+0.5);
  
  unsigned int j = 0;
  for(std::map<unsigned int, unsigned int>::iterator iMap=badModulesTK.begin(); iMap!=badModulesTK.end(); iMap++ )
  {
    hBadModulesTK->SetBinContent(++j,/*(double)*/iMap->second/*/(double)nModulesInPart*/);
    oss.str("");
    oss << iMap->first;
    hBadModulesTK->GetXaxis()->SetBinLabel(j,oss.str().c_str());
    //    std::cout << hBadModulesTK->GetBinContent(j) << std::endl;
  }
  TCanvas* cBadModulesTK = new TCanvas();
  hBadModulesTK->Draw();
  cBadModulesTK->Update();
  
  j = 0;
  for(std::map<unsigned int, unsigned int>::iterator iMap=badFibersTK.begin(); iMap!=badFibersTK.end(); iMap++ )
  {
    hBadFibersTK->SetBinContent(++j,/*(double)*/iMap->second/*/(double)nFibersInPart*/);
    oss.str("");
    oss << iMap->first;
    hBadFibersTK->GetXaxis()->SetBinLabel(j,oss.str().c_str());
    //    std::cout << hBadFibersTK->GetBinContent(j) << std::endl;
  }
  TCanvas* cBadFibersTK = new TCanvas();
  hBadFibersTK->Draw();
  cBadFibersTK->Update();
  
  j = 0;
  for(std::map<unsigned int, unsigned int>::iterator iMap=badAPVsTK.begin(); iMap!=badAPVsTK.end(); iMap++ )
  {
    hBadAPVsTK->SetBinContent(++j,/*(double)*/iMap->second/*/(double)nAPVsInPart*/);
    oss.str("");
    oss << iMap->first;
    hBadAPVsTK->GetXaxis()->SetBinLabel(j,oss.str().c_str());
    //    std::cout << hBadAPVsTK->GetBinContent(j) << std::endl;
  }
  TCanvas* cBadAPVsTK = new TCanvas();
  hBadAPVsTK->Draw();
  cBadAPVsTK->Update();

  j = 0;
  for(std::map<unsigned int, unsigned int>::iterator iMap=badStripsTK.begin(); iMap!=badStripsTK.end(); iMap++ )
  {
    hBadStripsTK->SetBinContent(++j,/*(double)*/iMap->second/*/(double)nStripsInPart*/);
    oss.str("");
    oss << iMap->first;
    hBadStripsTK->GetXaxis()->SetBinLabel(j,oss.str().c_str());
    //    std::cout << hBadStripsTK->GetBinContent(j) << std::endl;
  }
  TCanvas* cBadStripsTK = new TCanvas();
  hBadStripsTK->Draw();
  cBadStripsTK->Update();
  
  j = 0;
  for(std::map<unsigned int, unsigned int>::iterator iMap=badStripsFromAPVsTK.begin(); iMap!=badStripsFromAPVsTK.end(); iMap++ )
  {
    hBadStripsFromAPVsTK->SetBinContent(++j,/*(double)*/iMap->second/*/(double)nStripsInPart*/);
    oss.str("");
    oss << iMap->first;
    hBadStripsFromAPVsTK->GetXaxis()->SetBinLabel(j,oss.str().c_str());
    //    std::cout << hBadStripsTK->GetBinContent(j) << std::endl;
  }
  TCanvas* cBadStripsFromAPVsTK = new TCanvas();
  hBadStripsFromAPVsTK->Draw();
  cBadStripsFromAPVsTK->Update();

  j = 0;
  for(std::map<unsigned int, unsigned int>::iterator iMap=allBadStripsTK.begin(); iMap!=allBadStripsTK.end(); iMap++ )
  {
    hAllBadStripsTK->SetBinContent(++j,/*(double)*/iMap->second/*/(double)nStripsInPart*/);
    oss.str("");
    oss << iMap->first;
    hAllBadStripsTK->GetXaxis()->SetBinLabel(j,oss.str().c_str());
    //    std::cout << hAllBadStripsTK->GetBinContent(j) << std::endl;
  }
  TCanvas* cAllBadStripsTK = new TCanvas();
  hAllBadStripsTK->Draw();
  cAllBadStripsTK->Update();
  
  // Write histograms to output file
  
  //  std::cout << "Ready to open the file " << outFileName << std::endl;

  TFile* outFile = new TFile(outFileName, "UPDATE");

  //  std::cout << "File opened: " << outFileName << std::endl;

  outFile->cd();
  hBadModulesTK->Write();
  hBadFibersTK->Write();
  hBadAPVsTK->Write();
  hBadStripsTK->Write();
  hBadStripsFromAPVsTK->Write();
  hAllBadStripsTK->Write();

  //  std::cout << "histograms written in " << outFileName << std::endl;

  delete outFile;
  delete hBadModulesTK;
  delete hBadFibersTK;
  delete hBadAPVsTK;
  delete hBadStripsTK;
  delete hBadStripsFromAPVsTK;
  delete hAllBadStripsTK;
  delete cBadModulesTK;
  delete cBadFibersTK;
  delete cBadAPVsTK;
  delete cBadStripsTK;
  delete cBadStripsFromAPVsTK;
  delete cAllBadStripsTK;

}
