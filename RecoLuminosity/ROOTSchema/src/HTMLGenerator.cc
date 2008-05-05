#include "RecoLuminosity/ROOTSchema/interface/HTMLGenerator.h"

// STL Headers
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

// ROOT Headers
#include <TH1F.h>
#include <TCanvas.h>
#include <TChain.h>

// Lumi Headers
#include "RecoLuminosity/HLXReadOut/CoreUtils/include/ICTypeDefs.hh"
#include "RecoLuminosity/HLXReadOut/HLXCoreLibs/include/LumiStructures.hh"

// mkdir
#include <sys/types.h>
#include <sys/stat.h>

HCAL_HLX::HTMLGenerator::HTMLGenerator():ROOTFileReader(){

#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  // TODO: read config file

  plotExt_ = "gif";                                 // config file
  outputDir_ = "./";                                // config file
  baseURL = "http://cmsmon.cern.ch/lumi/dqmhtml/";  // config file
  writeMode = 0777;

#ifdef DEBUG
  std::cout << "baseURL: " << baseURL << std::endl;
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

HCAL_HLX::HTMLGenerator::~HTMLGenerator(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  //Nothing to do ....

#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

std::string HCAL_HLX::HTMLGenerator::GetRunDir(){
  // convert from up to 9 digit integer to XXX/YYY/ZZZ 
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  std::stringstream dirName;
  const unsigned int runNumber = GetRunNumber();
  
  dirName.str(std::string());
  dirName << std::setw(3) << std::setfill('0') << (runNumber / 1000000)
	  << "/" << std::setw(3) << std::setfill('0') << (runNumber % 1000000) / 1000
	  << "/" << std::setw(3) << std::setfill('0') << (runNumber % 1000)
	  << "/";
  
#ifdef DEBUG
  std::cout << "Directory: " << dirName.str()  << std::endl;
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif

  return dirName.str();
}

int HCAL_HLX::HTMLGenerator::MakeRunDir(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  std::stringstream dirName;
  const unsigned int runNumber = GetRunNumber();
  
  // Improve by using tokenizer or some other method.

  dirName.str(std::string());
  dirName << outputDir_; // Must assume that at least this exists!!
  dirName << std::setw(3) << std::setfill('0') << (runNumber / 1000000);
  mkdir(dirName.str().c_str(),writeMode);
  dirName << "/" << std::setw(3) << std::setfill('0') << (runNumber % 1000000) / 1000;
  mkdir(dirName.str().c_str(),writeMode);
  dirName << "/" << std::setw(3) << std::setfill('0') << (runNumber % 1000);
  mkdir(dirName.str().c_str(),writeMode);

#ifdef DEBUG
  std::cout << "Directory: " << dirName.str()  << std::endl;
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif

  return 0;
}

std::string HCAL_HLX::HTMLGenerator::GetLSDir(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  std::stringstream dirName;
  dirName.str(std::string());    
  dirName << std::setw(4) << std::setfill('0') << GetSectionNumber() << "/";

#ifdef DEBUG
  std::cout << "Directory: " << dirName.str()  << std::endl;
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif

  return dirName.str();
}

int HCAL_HLX::HTMLGenerator::MakeLSDir(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  std::string dirName = outputDir_ + GetRunDir() + GetLSDir();
  int errCode = mkdir(dirName.c_str(), writeMode);

#ifdef DEBUG
  std::cout << "Directory: " << dirName  << std::endl;
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif

  return errCode;
}


std::string HCAL_HLX::HTMLGenerator::GetWedgeDir(const unsigned short int &wedgeNum){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  std::stringstream dirName;
  dirName.str(std::string());
  dirName << "Wedge" << std::setw(2) << std::setfill('0') << wedgeNum << "/";

#ifdef DEBUG
  std::cout << "Directory: " << dirName.str()  << std::endl;
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
  return dirName.str();
}

int HCAL_HLX::HTMLGenerator::MakeWedgeDir(const unsigned short int &wedgeNum){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
  
  std::string dirName = outputDir_ + GetRunDir() + GetLSDir() + GetWedgeDir(wedgeNum);
  int errCode = mkdir(dirName.c_str(), writeMode);

#ifdef DEBUG
  std::cout << "Directory: " << dirName << std::endl; 
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
  return errCode;
}

std::string HCAL_HLX::HTMLGenerator::GetWedgePicDir(const unsigned short int &wedgeNum){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  std::stringstream dirName;
  dirName.str(std::string());
  dirName << "Pics/";

#ifdef DEBUG
  std::cout << "Directory: " << dirName.str() << std::endl;
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
  return dirName.str();
}


int HCAL_HLX::HTMLGenerator::MakeWedgePicDir(const unsigned short int &wedgeNum){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
  
  std::string dirName = outputDir_ + GetRunDir() + GetLSDir() + GetWedgeDir(wedgeNum) + "Pics/"; 
  int errCode = mkdir(dirName.c_str(), writeMode);
  std::cout << errCode << std::endl;

#ifdef DEBUG
  std::cout << "Directory: " << dirName << std::endl;
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
  return errCode;
}

void HCAL_HLX::HTMLGenerator::CreateWebPage(){

#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif
  
  // Assume This directory exists.
  GenerateIndexPage();

  MakeRunDir();
  GenerateRunPage();

  MakeLSDir();
  GenerateSectionPage();

  for(int wedgeNum = 0; wedgeNum < 36; wedgeNum++){
    MakeWedgeDir(wedgeNum);
    GenerateWedgePage(wedgeNum);
    MakeWedgePicDir(wedgeNum);
    GenerateWedgePlots(wedgeNum);
  }

  GenerateAveragePlots();
  GenerateAveragePage();

  GenerateComparePlots();
  GenerateComparePage();

#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void HCAL_HLX::HTMLGenerator::GenerateIndexPage(){
  // list all the runs that have data
  // if index.html page does not exist, then create it
  // if index.html page does exist, then append it with the new run number.
  // reminder: time stamp

#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  std::ofstream fileStr;
  std::stringstream fileName;

  fileName.str(std::string());
  fileName << outputDir_ << "index.html";
  fileStr.open(fileName.str().c_str());
    
  fileStr << "<html>" << std::endl;

  fileStr << "<title>" << std::endl; 
  fileStr << "Luminosity Monitoring System" << std::endl;
  fileStr << "</title>" << std::endl; 
  fileStr << "<H1>" << std::endl;
  fileStr << "Luminosity Monitoring System </br>" << std::endl;
  fileStr << "</H1>" << std::endl;

  fileStr << "<a href = \"" << GetRunDir() << "index.html\"> Run - " << GetRunNumber() << "</a>" << std::endl;

  fileStr << "</html>" << std::endl;

  fileStr.close();

#ifdef DEBUG
  std::cout << "**** " << fileName.str() << " ****" << std::endl;
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif

}

void HCAL_HLX::HTMLGenerator::GenerateRunPage(){
  // list all the lumi secitons in a run
  // if outputdDir/XXX/YYY/ZZZ/index.html does not exist, 
  // create a new one.
  // if outputDir/XXX/YYY/ZZZ/index.html exists, 
  // append it with the new section number.
  // must make dir structure

#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  std::ofstream fileStr;

  const unsigned int runNumber = GetRunNumber();
  const unsigned int sectionNumber = GetSectionNumber();
  
  std::stringstream fileName;

  fileName.str(std::string());
  fileName << outputDir_ << GetRunDir() << "index.html";  
  fileStr.open(fileName.str().c_str());

  fileStr << "<html>" << std::endl;
  fileStr << "<title>" << std::endl; 
  fileStr << "Luminosity File Reader - " 
	  << "Run " << runNumber 
	  << " Lumi Section " << sectionNumber << std::endl;
  fileStr << "</title>" << std::endl; 
  fileStr << "<body>" << std::endl;
  
  fileStr << "<a href = \"" <<  GetLSDir() <<  "index.html\"> Run - " << runNumber << " Section " << sectionNumber << "</a>" << std::endl;

  //figure out a way of appending this section

  fileStr << "</body>" << std::endl;
  fileStr << "</html>" << std::endl;

  fileStr.close();

#ifdef DEBUG
  std::cout << "**** " << fileName.str() << " ****" << std::endl;
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void HCAL_HLX::HTMLGenerator::GenerateSectionPage(){
  // list all the individual wedge pages
  // make section directory
  
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  std::ofstream fileStr;
  const unsigned int runNumber     = GetRunNumber();
  const unsigned int sectionNumber = GetSectionNumber();
  std::stringstream fileName;
  
  fileName.str(std::string());
  fileName << outputDir_ << GetRunDir() << GetLSDir() << "index.html";  
  fileStr.open(fileName.str().c_str());

  fileStr << "<html>" << std::endl;
  fileStr << "<title>" << std::endl; 
  fileStr << "Luminosity File Reader - " 
	  << "Run " << runNumber 
	  << " Lumi Section " << sectionNumber << std::endl;
  fileStr << "</title>" << std::endl; 

  fileStr << "<body>" << std::endl;
  fileStr << "HLX by Wedge </br>" << std::endl;
  fileStr << "<hline>" << std::endl;
 
  for(int wedgeNum = 0; wedgeNum < 36; wedgeNum++){
    fileStr << "<a href=\"Wedge" << std::setw(2) << std::setfill('0') << wedgeNum << "/index.html\" >" 
	    <<  "Wedge " << std::setw(2) << std::setfill('0') << wedgeNum 
	    << "</a></br>" << std::endl;
  }
  fileStr << "</body>" << std::endl;
  fileStr << "</html>" << std::endl;
  fileStr.close();

#ifdef DEBUG
  std::cout << "**** " << fileName.str() << " ****" << std::endl;
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void HCAL_HLX::HTMLGenerator::GenerateWedgePage(const unsigned short int &wedgeNum){

  // make wedge directory

#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  std::stringstream fileName;
  std::stringstream pictureDir;
  std::stringstream wedgeDir;
  
  const unsigned int runNumber     = GetRunNumber();
  const unsigned int sectionNumber = GetSectionNumber();

  std::vector< std::string > HistoNames;
  
  HistoNames.push_back("ETSum");
  HistoNames.push_back("OccBelowSet1");
  HistoNames.push_back("OccBetweenSet1");
  HistoNames.push_back("OccAboveSet1");
  HistoNames.push_back("OccBelowSet2");
  HistoNames.push_back("OccBetweenSet2");
  HistoNames.push_back("OccAboveSet2");
  HistoNames.push_back("LHC");

  fileName.str(std::string());
  fileName << outputDir_ << GetRunDir() << GetLSDir() << GetWedgeDir(wedgeNum)  << "index.html";

  std::ofstream fileStr;
  fileStr.open(fileName.str().c_str());

  fileStr << "<html>" << std::endl;
  fileStr << "<title>" << std::endl; 
  fileStr << "Luminosity File Reader - " 
	  << "Run " << runNumber 
	  << " Lumi Section " << sectionNumber 
	  << " Wedge " << wedgeNum
	  << std::endl;
  fileStr << "</title>" << std::endl; 
  fileStr << "<body>" << std::endl;

  fileStr << "<H1>" << std::endl;
  fileStr << "Luminosity File Reader - " 
	  << "Run " << runNumber 
	  << " Lumi Section " << sectionNumber 
	  << " Wedge " << wedgeNum
	  << std::endl;
  fileStr << "</H1>" << std::endl;
  fileStr << "<hline>" << std::endl;

  fileStr << "<a href=\"Pics/"<< HistoNames[0] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[0] << "." << plotExt_ << "\"></a>"      << std::endl; 
  fileStr << "<a href=\"Pics/"<< HistoNames[7] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[7] << "." << plotExt_ << "\"></a></br>" << std::endl; 

  fileStr << "<a href=\"Pics/"<< HistoNames[1] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[1] << "." << plotExt_ << "\"></a>"      << std::endl;
  fileStr << "<a href=\"Pics/"<< HistoNames[2] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[2] << "." << plotExt_ << "\"></a>"    	 << std::endl; 
  fileStr << "<a href=\"Pics/"<< HistoNames[3] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[3] << "." << plotExt_ << "\"></a></br>" << std::endl;

  fileStr << "<a href=\"Pics/"<< HistoNames[4] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[4] << "." << plotExt_ << "\"></a>"      << std::endl;
  fileStr << "<a href=\"Pics/"<< HistoNames[5] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[5] << "." << plotExt_ << "\"></a>"      << std::endl;
  fileStr << "<a href=\"Pics/"<< HistoNames[6] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[6] << "." << plotExt_ << "\"></a></br>" << std::endl;

  fileStr << "</body>" << std::endl;
  fileStr << "</html>" << std::endl;  
  fileStr.close();

#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void HCAL_HLX::HTMLGenerator::GenerateWedgePlots(const unsigned short int & wedgeNum){

#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  HCAL_HLX::LUMI_SECTION lumiSection;

  const unsigned short int NumBX = 3564;

  std::vector< std::string > HistoNames;
  
  HistoNames.push_back("ETSum");
  HistoNames.push_back("OccBelowSet1");
  HistoNames.push_back("OccBetweenSet1");
  HistoNames.push_back("OccAboveSet1");
  HistoNames.push_back("OccBelowSet2");
  HistoNames.push_back("OccBetweenSet2");
  HistoNames.push_back("OccAboveSet2");
  HistoNames.push_back("LHC");

  // Create Histograms

  TH1F* Histos[8];

  const unsigned int XBinWidth = 1;
  const unsigned int XMax = NumBX;
  const unsigned int XMin = 0;
  const unsigned int XBins = (XMax - XMin)/XBinWidth;
  
  for(int histoNum = 0; histoNum < 8; histoNum++){
    std::cout << histoNum << ": " <<  HistoNames[histoNum] << std::endl;
    Histos[histoNum] = new TH1F(HistoNames[histoNum].c_str(),"", XBins, XMin, XMax);
  }
  
  // Fill histograms
#ifdef DEBUG
  std::cout << "Filling Histograms" << std::endl;
#endif

  GetLumiSection(lumiSection);

  for(unsigned int BXNum = 0; BXNum < NumBX; BXNum++){ 
    Histos[0]->SetBinContent(BXNum, lumiSection.etSum[wedgeNum].data[BXNum-1]);
    Histos[7]->SetBinContent(BXNum, lumiSection.lhc[wedgeNum].data[BXNum-1]);
    for(int k = 0; k < 6; k++)
      Histos[k+1]->SetBinContent(BXNum, lumiSection.occupancy[wedgeNum].data[k][BXNum-1]);
  }

  const std::string wedgePicsDir = outputDir_ + GetRunDir() + GetLSDir() + GetWedgeDir(wedgeNum) + GetWedgePicDir(wedgeNum); 

  std::string plotFileName;

  TCanvas* c1 = new TCanvas("c1","c1",700,500);
  //c1->SetLogy();
  c1->SetTicks(1,1);

#ifdef DEBUG
  std::cout << "Drawing and saving histograms" << std::endl;
#endif
  
  for(int histoNum = 0; histoNum < 8; histoNum++){
    std::cout << histoNum << ": " <<  HistoNames[histoNum] << std::endl;
    Histos[histoNum]->Draw();
    plotFileName =  wedgePicsDir +  HistoNames[histoNum] + "." + plotExt_;
    std::cout << histoNum << ": " <<  plotFileName << std::endl;
    c1->SaveAs(plotFileName.c_str());
    std::cout << histoNum << ": " <<  HistoNames[histoNum] << std::endl;
  }

  delete c1;
  delete Histos[0];
  delete Histos[1];
  delete Histos[2];
  delete Histos[3];
  delete Histos[4];
  delete Histos[5];
  delete Histos[6];
  delete Histos[7];

#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void HCAL_HLX::HTMLGenerator::GenerateComparePlots(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif


#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void HCAL_HLX::HTMLGenerator::GenerateComparePage(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void HCAL_HLX::HTMLGenerator::GenerateAveragePlots(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}

void HCAL_HLX::HTMLGenerator::GenerateAveragePage(){
#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

#ifdef DEBUG
  std::cout << "End " << __PRETTY_FUNCTION__ << std::endl;
#endif
}
