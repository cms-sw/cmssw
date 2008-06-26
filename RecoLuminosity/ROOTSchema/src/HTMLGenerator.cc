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

  GenerateLumiPage();
  
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

  fileStr << "<a href=\"Luminosity/index.html\">Luminosity</a>" << std::endl;

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

  const unsigned short int NumBX = 3564;

  TH1F* Histos[8];

  for(int histoNum = 0; histoNum < 8; histoNum++){
    Histos[histoNum] = new TH1F(HistoNames[histoNum].c_str(),"", NBins_, XMin_, XMax_);
    Histos[histoNum]->GetXaxis()->SetTitle("Bunch Crossing");

    std::stringstream HistoTitle;

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

  HCAL_HLX::LUMI_SECTION lumiSection;
  GetLumiSection(lumiSection);

  for(unsigned int BXNum = 0; BXNum < NumBX; BXNum++){ 
    Histos[0]->Fill(BXNum, lumiSection.etSum[HLXID].data[BXNum-1]);
    Histos[7]->Fill(BXNum, lumiSection.lhc[HLXID].data[BXNum-1]);
    for(int k = 0; k < 6; k++)
      Histos[k+1]->Fill(BXNum, lumiSection.occupancy[HLXID].data[k][BXNum-1]);
  }


  const std::string HLXPicsDir = outputDir_ + GetRunDir() + GetLSDir() + GetHLXDir(HLXID) + GetHLXPicDir(HLXID); 
  TCanvas* c1 = new TCanvas("c1","c1",700,500);
  //c1->SetLogy();
  c1->SetTicks(1,1);

  for(int histoNum = 0; histoNum < 8; histoNum++){
    Histos[histoNum]->Draw();
    std::string plotFileName =  HLXPicsDir +  HistoNames[histoNum] + "." + plotExt_;
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

  std::string fileName;
  std::string  pageDir;
  
  const unsigned int runNumber     = GetRunNumber();
  const unsigned int sectionNumber = GetSectionNumber();

  pageDir = outputDir_ + GetRunDir() + GetLSDir() + HistoName;
  MakeDir(pageDir, writeMode);

  fileName = pageDir + "/index.html";

  std::ofstream fileStr;
  fileStr.open(fileName.c_str());

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

void HCAL_HLX::HTMLGenerator::GenerateLumiPage(){

   std::string fileName;
   std::string  pageDir;
   
   const unsigned int runNumber     = GetRunNumber();
   const unsigned int sectionNumber = GetSectionNumber();
   
   pageDir = outputDir_ + GetRunDir() + GetLSDir() + "Luminosity/";
   MakeDir(pageDir + "/Pics" , writeMode);
   
   fileName = pageDir + "index.html";
   std::cout << "********* " << fileName << " **********" << std::endl;


   TCanvas* c1 = new TCanvas("c1","c1",700,500);
   //c1->SetLogy();
   c1->SetTicks(1,1);
   
   TH1F* ETLumiHisto      = new TH1F("ETLumi",      "E_{T} Lumi",          NBins_, XMin_, XMax_);
   TH1F* OccLumiSet1Histo = new TH1F("OccLumiSet1", "Occupancy Lumi Set1", NBins_, XMin_, XMax_);
   TH1F* OccLumiSet2Histo = new TH1F("OccLumiSet2", "Occupancy Lumi Set2", NBins_, XMin_, XMax_);
   
   ETLumiHisto->GetXaxis()->SetTitle("Bunch Crossing");
   OccLumiSet1Histo->GetXaxis()->SetTitle("Bunch Crossing");
   OccLumiSet2Histo->GetXaxis()->SetTitle("Bunch Crossing");
   
   HCAL_HLX::LUMI_SECTION lumiSection;
  
   GetLumiSection(lumiSection);

   for( unsigned int iBX = 0; iBX < 3564; ++iBX ){
      ETLumiHisto->     SetBinContent(iBX, lumiSection.lumiDetail.ETLumi[iBX]);
      ETLumiHisto->     SetBinError(  iBX, lumiSection.lumiDetail.ETLumiErr[iBX]);
      OccLumiSet1Histo->SetBinContent(iBX, lumiSection.lumiDetail.OccLumi[0][iBX]);
      OccLumiSet1Histo->SetBinError(  iBX, lumiSection.lumiDetail.OccLumiErr[0][iBX]);
      OccLumiSet2Histo->SetBinContent(iBX, lumiSection.lumiDetail.OccLumi[1][iBX]);
      OccLumiSet2Histo->SetBinError(  iBX, lumiSection.lumiDetail.OccLumiErr[1][iBX]);
   }

   ETLumiHisto->Draw();
   c1->SaveAs( (pageDir + "/Pics/EtSumLumi.png").c_str() );
   OccLumiSet1Histo->Draw();
   c1->SaveAs( (pageDir + "/Pics/OccLumiSet1.png").c_str() ); 
   OccLumiSet2Histo->Draw();
   c1->SaveAs( (pageDir + "/Pics/OccLumiSet2.png").c_str() );

   delete c1;
   delete ETLumiHisto;
   delete OccLumiSet1Histo;
   delete OccLumiSet2Histo;
   
   fstream fileStr;
   
   fileStr.open(fileName.c_str());
   
   fileStr << "<html>" << std::endl;
   fileStr << "<title>" << std::endl; 
   fileStr << "Luminosity File Reader - " 
	   << "Run " << runNumber 
	   << " Lumi Section " << sectionNumber 
	   << " Luminosity" 
	   << std::endl;
  fileStr << "</title>" << std::endl; 
  fileStr << "<body>" << std::endl;

  fileStr << "<H1>" << std::endl;
  fileStr << "Luminosity File Reader - " 
	  << "Run " << runNumber 
	  << " Lumi Section " << sectionNumber 
	  << " - Luminosity "
	  << std::endl;
  fileStr << "</H1>" << std::endl;
  fileStr << "<hr>" << std::endl;

  fileStr << "<a href=\"Pics/EtSumLumi.png\"\"><img src=\"Pics/EtSumLumi.png\" width=\"30%\" ></a>" << std::endl; 
  fileStr << "<a href=\"Pics/OccLumiSet1.png\"\"><img src=\"Pics/OccLumiSet1.png\" width=\"30%\" ></a>" << std::endl; 
  fileStr << "<a href=\"Pics/OccLumiSet2.png\"\"><img src=\"Pics/OccLumiSet2.png\" width=\"30%\" ></a>" << std::endl; 
  
  fileStr.close();

}

