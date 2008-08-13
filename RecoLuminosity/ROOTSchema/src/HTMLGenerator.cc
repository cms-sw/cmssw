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
#include "RecoLuminosity/TCPReceiver/interface/ICTypeDefs.hh"
#include "RecoLuminosity/TCPReceiver/interface/LumiStructures.hh"

// mkdir
#include <sys/types.h>
#include <sys/stat.h>

HCAL_HLX::HTMLGenerator::HTMLGenerator():ROOTFileReader(){

  // TODO: read config file

  plotExt_   = "png";     // config file
  outputDir_ = "./";      // config file
  writeMode  = 0777;
  
  previousRunNumber = 0;

  HistoNames.push_back("ETSum");
  HistoNames.push_back("OccBelowSet1");
  HistoNames.push_back("OccBetweenSet1");
  HistoNames.push_back("OccAboveSet1");
  HistoNames.push_back("OccBelowSet2");
  HistoNames.push_back("OccBetweenSet2");
  HistoNames.push_back("OccAboveSet2");
  HistoNames.push_back("LHC");

}

HCAL_HLX::HTMLGenerator::~HTMLGenerator(){}

// ******** Main function ******

void HCAL_HLX::HTMLGenerator::CreateWebPage(){
  
  GenerateIndexPage();

  MakeDir( outputDir_ + GetRunDir(), writeMode );
  GenerateRunPage();

  MakeDir( outputDir_ + GetRunDir() + GetLSDir(), writeMode);
  GenerateSectionPage();

  for(int HLXID = 0; HLXID < 36; HLXID++){
    MakeDir(outputDir_ + GetRunDir() + GetLSDir() + GetHLXDir(HLXID), writeMode);
    GenerateHLXPage(HLXID);
    MakeDir(outputDir_ + GetRunDir() + GetLSDir() + GetHLXDir(HLXID) + GetHLXPicDir(HLXID), writeMode);
    GenerateHLXPlots(HLXID);
  }

  GenerateAveragePlots();
  GenerateAveragePage();

  GenerateComparePlots();
  GenerateComparePage();

  for(unsigned int iHistos = 0; iHistos < 8; ++iHistos) 
    GenerateHistoGroupPage(HistoNames[iHistos]);
}

// ************* Get Directory name functions ****************

std::string HCAL_HLX::HTMLGenerator::GetRunDir(){
  // convert from up to 9 digit integer to XXX/YYY/ZZZ 

  std::stringstream dirName;
  const unsigned int runNumber = GetRunNumber();
  
  dirName.str(std::string());
  dirName << std::setw(3) << std::setfill('0') << (runNumber / 1000000)
	  << "/" << std::setw(3) << std::setfill('0') << (runNumber % 1000000) / 1000
	  << "/" << std::setw(3) << std::setfill('0') << (runNumber % 1000)
	  << "/";

  return dirName.str();
}

std::string HCAL_HLX::HTMLGenerator::GetLSDir(){

  std::stringstream dirName;
  dirName.str(std::string());    
  dirName << std::setw(4) << std::setfill('0') << GetSectionNumber() << "/";

  return dirName.str();
}

std::string HCAL_HLX::HTMLGenerator::GetHLXDir(const unsigned short int &HLXID){

  std::stringstream dirName;
  dirName.str(std::string());
  dirName << "HLX" << std::setw(2) << std::setfill('0') << HLXID << "/";

  return dirName.str();
}

std::string HCAL_HLX::HTMLGenerator::GetHLXPicDir( const unsigned short int &HLXID ){

  return "Pics/";
}

// ************ Generate Pages *******

void HCAL_HLX::HTMLGenerator::GenerateIndexPage(){

  std::string fileName;
  std::stringstream runLine;
  
  const unsigned int runNumber = GetRunNumber();

  fileName = outputDir_ + "index.html";

  if( !fileExists(fileName) ){
    std::string fileTitle = "Luminosity Monitoring System";
    
    MakeEmptyWebPage( fileName, fileTitle);
    InsertLineAfter( fileName, "<H1>\nLuminosity Monitoring System\n</H1>", "<body>");
  }

  if(runNumber != previousRunNumber  ){
    runLine << "<a href = \"" 
	    << GetRunDir() 
	    << "index.html\"> Run - " 
	    << runNumber << "</a>  " 
	    << TimeStampLong() 
	    << "</br>"; 
    previousRunNumber = runNumber;
  }
  InsertLineBefore( fileName, runLine.str(), "</body>"); 
}

void HCAL_HLX::HTMLGenerator::GenerateRunPage(){

  using std::stringstream;

  const unsigned int runNumber = GetRunNumber();
  const unsigned int sectionNumber = GetSectionNumber();
  
  std::string fileName;

  fileName = outputDir_ + GetRunDir() + "index.html";  
  
  if(!fileExists(fileName)){
    stringstream fileTitle;

    fileTitle << "Luminosity File Reader - Run " << runNumber;
    MakeEmptyWebPage(fileName, fileTitle.str());
  }

  stringstream sectionLine;

  sectionLine << "<a href = \"" 
	      <<  GetLSDir() 
	      <<  "index.html\"> Run - " 
	      << runNumber << " Section " 
	      << sectionNumber << "</a>  " 
	      << TimeStampLong() << "</br>";

  InsertLineBefore( fileName, sectionLine.str(), "</body>");
}

void HCAL_HLX::HTMLGenerator::GenerateSectionPage(){
  // list all the individual HLX pages
  // make section directory

  std::ofstream fileStr;
  const unsigned int runNumber     = GetRunNumber();
  const unsigned int sectionNumber = GetSectionNumber();
  std::string fileName;
  
  fileName = outputDir_ + GetRunDir() + GetLSDir() + "index.html";  
  fileStr.open(fileName.c_str());

  fileStr << "<html>" << std::endl;
  fileStr << "<title>" << std::endl; 
  fileStr << "Luminosity File Reader - " 
	  << "Run " << runNumber << " - " 
	  << " Lumi Section - " << sectionNumber << std::endl;
  fileStr << "</title>" << std::endl; 

  fileStr << "<body>" << std::endl;
  
  for(int i = 0; i < 8; i++){
    fileStr << "<a href=\"" 
	    << HistoNames[i] 
	    << "/index.html\" >" 
	    << HistoNames[i] <<  "</a></br>" << std::endl;
  }

  fileStr << "</br>" << std::endl;
  fileStr << "</br>" << std::endl;
  
  fileStr << "<H2>" << std::endl;
  fileStr << "HLX ID </br>" << std::endl;
  fileStr << "<hr>" << std::endl;
  fileStr << "</H2>" << std::endl;
  fileStr << "<H3>" << std::endl;

  for(int HLXID = 0; HLXID < 36; HLXID++){
    fileStr << "<a href=\"HLX" 
	    << std::setw(2) << std::setfill('0') << HLXID 
	    << "/index.html\" >" 
	    <<  "HLX " 
	    << std::setw(2) << std::setfill('0') << HLXID 
	    << "</a>       " << std::endl;
    if(HLXID % 6 == 5) fileStr << "</br>";
  }
  
  fileStr << "</H3>" << std::endl;
  fileStr << "</body>" << std::endl;
  fileStr << "</html>" << std::endl;
  fileStr.close();

}

void HCAL_HLX::HTMLGenerator::GenerateHLXPage(const unsigned short int &HLXID){

  // make HLX directory

  std::stringstream fileName;
  std::stringstream pictureDir;
  std::stringstream HLXDir;
  
  const unsigned int runNumber     = GetRunNumber();
  const unsigned int sectionNumber = GetSectionNumber();

  fileName.str(std::string());
  fileName << outputDir_ << GetRunDir() << GetLSDir() << GetHLXDir(HLXID)  << "index.html";

  std::ofstream fileStr;
  fileStr.open(fileName.str().c_str());

  fileStr << "<html>" << std::endl;
  fileStr << "<title>" << std::endl; 
  fileStr << "Luminosity File Reader - " 
	  << "Run " << runNumber 
	  << " Lumi Section " << sectionNumber 
	  << " HLX " << HLXID
	  << std::endl;
  fileStr << "</title>" << std::endl; 
  fileStr << "<body>" << std::endl;

  fileStr << "<H1>" << std::endl;
  fileStr << "Luminosity File Reader - " 
	  << "Run " << runNumber << " - " 
	  << " Lumi Section " << sectionNumber << " - " 
	  << " HLX " << HLXID
	  << std::endl;
  fileStr << "</H1>" << std::endl;
  fileStr << "<hr>" << std::endl;

  fileStr << "<a href=\"Pics/"<< HistoNames[0] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[0] << "." << plotExt_ << "\" width=\"30%\"></a>"     << std::endl; 
  fileStr << "<a href=\"Pics/"<< HistoNames[7] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[7] << "." << plotExt_ << "\" width=\"30%\"></a></br>"<< std::endl; 

  fileStr << "<a href=\"Pics/"<< HistoNames[1] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[1] << "." << plotExt_ << "\" width=\"30%\"></a>"     << std::endl;
  fileStr << "<a href=\"Pics/"<< HistoNames[2] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[2] << "." << plotExt_ << "\" width=\"30%\"></a>"     << std::endl; 
  fileStr << "<a href=\"Pics/"<< HistoNames[3] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[3] << "." << plotExt_ << "\" width=\"30%\"></a></br>" << std::endl;

  fileStr << "<a href=\"Pics/"<< HistoNames[4] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[4] << "." << plotExt_ << "\" width=\"30%\"></a>"     << std::endl;
  fileStr << "<a href=\"Pics/"<< HistoNames[5] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[5] << "." << plotExt_ << "\" width=\"30%\"></a>"     << std::endl;
  fileStr << "<a href=\"Pics/"<< HistoNames[6] << "." << plotExt_ << "\"><img src=\"Pics/" << HistoNames[6] << "." << plotExt_ << "\" width=\"30%\"></a></br>"<< std::endl;

  fileStr << "</body>" << std::endl;
  fileStr << "</html>" << std::endl;  
  fileStr.close();

}

void HCAL_HLX::HTMLGenerator::SetHistoBins(unsigned int NBins, double XMin, double XMax){

  NBins_ = NBins;

  if( XMax < XMin){
    XMax_ = XMin;
    XMin_ = XMax;
  }else{
    XMax_ = XMax;
    XMin_ = XMin;
  }
}

void HCAL_HLX::HTMLGenerator::GenerateHLXPlots(const unsigned short int & HLXID){

#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  HCAL_HLX::LUMI_SECTION lumiSection;

  const unsigned short int NumBX = 3564;

  // Create Histograms

  TH1F* Histos[8];

  std::stringstream HistoTitle;

  for(int histoNum = 0; histoNum < 8; histoNum++){
    Histos[histoNum] = new TH1F(HistoNames[histoNum].c_str(),"", NBins_, XMin_, XMax_);
    Histos[histoNum]->GetXaxis()->SetTitle("Bunch Crossing");
    HistoTitle.str(std::string());
    HistoTitle << HistoNames[histoNum] << " - HLX " << HLXID;
    Histos[histoNum]->SetTitle(HistoTitle.str().c_str());
    Histos[histoNum]->GetYaxis()->SetTitleOffset(1.3);
  }
  
  Histos[0]->GetYaxis()->SetTitle("Et Sum (1 LS)");
  Histos[7]->GetYaxis()->SetTitle("LHC (1 LS)");
  for(int k = 0; k < 6; k++){
    Histos[k+1]->GetYaxis()->SetTitle("Occupancy (1 LS)");
  }

  GetLumiSection(lumiSection);

  for(unsigned int BXNum = 0; BXNum < NumBX; BXNum++){ 
    Histos[0]->SetBinContent(BXNum, lumiSection.etSum[HLXID].data[BXNum-1]);
    Histos[7]->SetBinContent(BXNum, lumiSection.lhc[HLXID].data[BXNum-1]);
    for(int k = 0; k < 6; k++)
      Histos[k+1]->SetBinContent(BXNum, lumiSection.occupancy[HLXID].data[k][BXNum-1]);
  }

  const std::string HLXPicsDir = outputDir_ + GetRunDir() + GetLSDir() + GetHLXDir(HLXID) + GetHLXPicDir(HLXID); 

  std::string plotFileName;

  TCanvas* c1 = new TCanvas("c1","c1",700,500);
  //c1->SetLogy();
  c1->SetTicks(1,1);
  
  for(int histoNum = 0; histoNum < 8; histoNum++){
    Histos[histoNum]->Draw();
    plotFileName =  HLXPicsDir +  HistoNames[histoNum] + "." + plotExt_;
    c1->SaveAs(plotFileName.c_str());
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

void HCAL_HLX::HTMLGenerator::GenerateHistoGroupPage(const std::string &HistoName){

  std::stringstream fileName;
  std::stringstream pageDir;
  
  const unsigned int runNumber     = GetRunNumber();
  const unsigned int sectionNumber = GetSectionNumber();

  pageDir.str(std::string());
  pageDir << outputDir_ << GetRunDir() << GetLSDir() << HistoName;
  mkdir(pageDir.str().c_str(), writeMode);

  fileName.str(std::string());
  fileName << outputDir_ << GetRunDir() << GetLSDir() << HistoName << "/"  << "index.html";

  std::ofstream fileStr;
  fileStr.open(fileName.str().c_str());

  fileStr << "<html>" << std::endl;
  fileStr << "<title>" << std::endl; 
  fileStr << "Luminosity File Reader - " 
	  << "Run " << runNumber 
	  << " Lumi Section " << sectionNumber 
	  << " - " << HistoName
	  << std::endl;
  fileStr << "</title>" << std::endl; 
  fileStr << "<body>" << std::endl;

  fileStr << "<H1>" << std::endl;
  fileStr << "Luminosity File Reader - " 
	  << "Run " << runNumber 
	  << " Lumi Section " << sectionNumber 
	  << " - " << HistoName
	  << std::endl;
  fileStr << "</H1>" << std::endl;
  fileStr << "<hr>" << std::endl;

  for(int HLXID = 0; HLXID < 36; HLXID++){
    fileStr << "<a href=\"../HLX" << std::setw(2) << std::setfill('0') << HLXID 
	    << "/Pics/" << HistoName << "." << plotExt_ << "\"><img src=\"../HLX" 
	    << std::setw(2) << std::setfill('0') << HLXID << "/Pics/" << HistoName << "." 
	    << plotExt_ << "\" width=\"15%\" ></a>" << std::endl; 
  }
  fileStr << "</body>" << std::endl;
  fileStr << "</html>" << std::endl;  
  fileStr.close();
}
