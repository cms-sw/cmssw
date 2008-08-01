#include "RecoLuminosity/ROOTSchema/interface/HTMLGenerator.h"

// STL Headers
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

// ROOT Headers
#include <TH1F.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TStyle.h>


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

  std::vector< std::string > HLXToHFMap;
  
  HLXToHFMap.push_back("HLX  1 - hf- s26,27 - iPhi 51");
  HLXToHFMap.push_back("HLX  2 - hf- s28,29 - iPhi 55");
  HLXToHFMap.push_back("HLX  3 - hf- s30,31 - iPhi 59");
  HLXToHFMap.push_back("HLX  4 - hf- s32,33 - iPhi 63");
  HLXToHFMap.push_back("HLX  5 - hf- s34,35 - iPhi 67");
  HLXToHFMap.push_back("HLX  6 - hf- s36,1  - iPhi 71");
  HLXToHFMap.push_back("HLX  7 - hf+ s26,27 - iPhi 51");
  HLXToHFMap.push_back("HLX  8 - hf+ s28,29 - iPhi 55");
  HLXToHFMap.push_back("HLX  9 - hf+ s30,31 - iPhi 59");
  HLXToHFMap.push_back("HLX 10 - hf+ s32,33 - iPhi 63");
  HLXToHFMap.push_back("HLX 11 - hf+ s34,35 - iPhi 67");
  HLXToHFMap.push_back("HLX 12 - hf+ s36,1  - iPhi 71");
  HLXToHFMap.push_back("HLX 13 - hf- s14,15 - iPhi 27");
  HLXToHFMap.push_back("HLX 14 - hf- s16,17 - iPhi 31");
  HLXToHFMap.push_back("HLX 15 - hf- s18,19 - iPhi 35");
  HLXToHFMap.push_back("HLX 16 - hf- s20,21 - iPhi 39");
  HLXToHFMap.push_back("HLX 17 - hf- s22,23 - iPhi 43");
  HLXToHFMap.push_back("HLX 18 - hf- s24,25 - iPhi 47");
  HLXToHFMap.push_back("HLX 19 - hf+ s14,15 - iPhi 27");
  HLXToHFMap.push_back("HLX 20 - hf+ s16,17 - iPhi 31");
  HLXToHFMap.push_back("HLX 21 - hf+ s18,19 - iPhi 35");
  HLXToHFMap.push_back("HLX 22 - hf+ s20,21 - iPhi 39");
  HLXToHFMap.push_back("HLX 23 - hf+ s22,23 - iPhi 43");
  HLXToHFMap.push_back("HLX 24 - hf+ s24,25 - iPhi 47");
  HLXToHFMap.push_back("HLX 25 - hf- s2,3   - iPhi  3");
  HLXToHFMap.push_back("HLX 26 - hf- s4,5   - iPhi  7");
  HLXToHFMap.push_back("HLX 27 - hf- s6,7   - iPhi 11");
  HLXToHFMap.push_back("HLX 28 - hf- s8,9   - iPhi 15");
  HLXToHFMap.push_back("HLX 29 - hf- s10,11 - iPhi 19");
  HLXToHFMap.push_back("HLX 30 - hf- s12,13 - iPhi 23");
  HLXToHFMap.push_back("HLX 31 - hf+ s2,3   - iPhi  3");
  HLXToHFMap.push_back("HLX 32 - hf+ s4,5   - iPhi  7");
  HLXToHFMap.push_back("HLX 33 - hf+ s6,7   - iPhi 11");
  HLXToHFMap.push_back("HLX 34 - hf+ s8,9   - iPhi 15");
  HLXToHFMap.push_back("HLX 35 - hf+ s10,11 - iPhi 19");
  HLXToHFMap.push_back("HLX 36 - hf+ s12,13 - iPhi 23");

  int iEta[] = {-41,-41,-41,-41,-41,-41,
	      29, 29, 29, 29, 29, 29,
	      -41,-41,-41,-41,-41,-41,
	      29, 29, 29, 29, 29, 29,
	      -41,-41,-41,-41,-41,-41,
	      29, 29, 29, 29, 29, 29};

  int iPhi[] = { 51,55,59,63,67,71,
	       51,55,59,63,67,71,
	       27,31,35,39,43,47,
	       27,31,35,39,43,47,
	       3,7,11,15,19,23,
	       3,7,11,15,19,23};

  for( unsigned int iHLX = 0; iHLX < 36; ++iHLX ){
    HLXToHFMap_[iHLX] = HLXToHFMap[iHLX];
    iEta_[iHLX] = iEta[iHLX];
    iPhi_[iHLX] = iPhi[iHLX];
  }

}

HCAL_HLX::HTMLGenerator::~HTMLGenerator(){}

// ******** Main function ******

void HCAL_HLX::HTMLGenerator::CreateWebPage(){

  GetLumiSection(lumiSection_);
    
  MakeDir( outputDir_ + GetRunDir(), writeMode );
  MakeDir( outputDir_ + GetRunDir() + GetLSDir(), writeMode);

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

  //  GenerateLumiPage();

  for(unsigned int iHistos = 0; iHistos < 8; ++iHistos) 
    GenerateHistoGroupPage(HistoNames[iHistos]);

  GenerateSectionPage();
  GenerateRunPage();
  GenerateIndexPage();

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

  fileStr << "<hr>" << std::endl;
  fileStr << "Summary" << std::endl;
  fileStr << "<hr>" << std::endl;



  TCanvas* c1 = new TCanvas("c1","c1",700,500);
  c1->SetTicks(1,1);

  // Et Sum Summary

  TH2F* EtSummary = new TH2F("EtSummary", "Et Sum - Summary", 12, -42, 42, 18, 0, 72);
  EtSummary->GetXaxis()->SetTitle("i #eta");
  EtSummary->GetYaxis()->SetTitle("i #phi");

  float MaxEt = 0.0;
  float MinEt = 1000000000.0;

  for( int iHLX = 0; iHLX < 36; ++iHLX){
    float AvgEtSum = 0.0;
    for( int iBX = 100; iBX < 3500; ++iBX){ 
      AvgEtSum += lumiSection_.etSum[iHLX].data[iBX];
    }

    AvgEtSum /= (3400.0*lumiSection_.hdr.numOrbits);

    if( AvgEtSum > MaxEt ){
      MaxEt = AvgEtSum;
    }
    if( AvgEtSum < MinEt ){
      MinEt = AvgEtSum;
    }
    EtSummary->Fill( iEta_[iHLX], iPhi_[iHLX], AvgEtSum );
    EtSummary->Fill( iEta_[iHLX] +7, iPhi_[iHLX], AvgEtSum );
    std::cout << "iEta: " << iEta_[iHLX] << " iPhi: " << iPhi_[iHLX] << " AvgEtSum: " << AvgEtSum << std::endl;
  }

  std::string picName = outputDir_ + GetRunDir() + GetLSDir() + "EtSummary.png";
  gStyle->SetOptStat(0);
  gStyle->SetPalette(1);

  EtSummary->GetZaxis()->SetRangeUser( MinEt, MaxEt );

  EtSummary->Draw("colz");

  c1->SaveAs(picName.c_str());

  // Occupancy Summary

  TH2F* OccSummary = new TH2F("OccSummary", "Occupancy - Summary", 12, -42, 42, 18, 0, 72);
  OccSummary->GetXaxis()->SetTitle("i #eta");
  OccSummary->GetYaxis()->SetTitle("i #phi");

  float MaxOcc = 0.0;
  float MinOcc = 1000000000.0;

  for( int iHLX = 0; iHLX < 36; ++iHLX){
    float AvgOccSet1 = 0.0;
    float AvgOccSet2 = 0.0;
    
    for( int iBX = 100; iBX < 3500; ++iBX){ 
      AvgOccSet1 += lumiSection_.occupancy[iHLX].data[0][iBX];
      AvgOccSet2 += lumiSection_.occupancy[iHLX].data[3][iBX];
    }

    AvgOccSet1 /= (3400.0*lumiSection_.hdr.numOrbits);
    AvgOccSet2 /= (3400.0*lumiSection_.hdr.numOrbits);

    if( AvgOccSet1 > MaxOcc ){
      MaxOcc = AvgOccSet1;
    }
    if( AvgOccSet1 < MinOcc ){
      MinOcc = AvgOccSet1;
    }

    if( AvgOccSet2 > MaxOcc ){
      MaxOcc = AvgOccSet2;
    }
    if( AvgOccSet1 < MinOcc ){
      MinOcc = AvgOccSet1;
    }

    if( iEta_[iHLX] > 0 ){
      OccSummary->Fill( iEta_[iHLX] +7 , iPhi_[iHLX], AvgOccSet1 );
      OccSummary->Fill( iEta_[iHLX],     iPhi_[iHLX], AvgOccSet2 );
    }

    if( iEta_[iHLX] < 0 ){
      OccSummary->Fill( iEta_[iHLX],     iPhi_[iHLX], AvgOccSet1 );
      OccSummary->Fill( iEta_[iHLX] + 7, iPhi_[iHLX], AvgOccSet2 );
    }

    std::cout << "iEta: " << iEta_[iHLX] << " iPhi: " << iPhi_[iHLX] << " AvgOccSet1: " << AvgOccSet1 << " AvgOccSet2: " << AvgOccSet2 << std::endl;
  }

  picName = outputDir_ + GetRunDir() + GetLSDir() + "OccSummary.png";
  
  OccSummary->GetZaxis()->SetRangeUser( MinOcc, MaxOcc );

  OccSummary->Draw("colz");

  c1->SaveAs(picName.c_str());

  fileStr << "<a href=\"EtSummary.png\"><img src=\"EtSummary.png\" width=\"45%\"></a>" << std::endl; 
  fileStr << "<a href=\"OccSummary.png\"><img src=\"OccSummary.png\" width=\"45%\"></a>" << std::endl; 

  fileStr << "<hr>" << std::endl;  
  fileStr << "<H2>" << std::endl;
  fileStr << "HLX ID </br>" << std::endl;
  fileStr << "<hr>" << std::endl;
  fileStr << "</H2>" << std::endl;
  fileStr << "<H3>" << std::endl;

  for(int HLXID = 0; HLXID < 36; HLXID++){
    fileStr << "<a href=\"HLX" 
	    << std::setw(2) << std::setfill('0') << HLXID 
	    << "/index.html\" >" 
	    << HLXToHFMap_[HLXID]
	    << "</a> </br>" << std::endl;
  }
  
  fileStr << "</H3>" << std::endl;
  fileStr << "</body>" << std::endl;
  fileStr << "</html>" << std::endl;

  fileStr.close();

  delete EtSummary;
  delete OccSummary;

  delete c1;
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
	  << HLXToHFMap_[HLXID]
	  << std::endl;
  fileStr << "</title>" << std::endl; 
  fileStr << "<body>" << std::endl;

  fileStr << "<H1>" << std::endl;
  fileStr << "Luminosity File Reader - " 
	  << "Run " << runNumber << " - " 
	  << " Lumi Section " << sectionNumber << " - " 
	  << HLXToHFMap_[HLXID]
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
    HistoTitle << HistoNames[histoNum] << " - " << HLXToHFMap_[HLXID];
    Histos[histoNum]->SetTitle(HistoTitle.str().c_str());
    Histos[histoNum]->GetYaxis()->SetTitleOffset(1.3);
  }
  
  Histos[0]->GetYaxis()->SetTitle("Et Sum (1 LS)");
  Histos[7]->GetYaxis()->SetTitle("LHC (1 LS)");
  for(int k = 0; k < 6; k++){
    Histos[k+1]->GetYaxis()->SetTitle("Occupancy (1 LS)");
  }

  double EtSumNoise[4];
  
  EtSumNoise[0] = 0;
  EtSumNoise[1] = 0;
  EtSumNoise[2] = 0;
  EtSumNoise[3] = 0;
  
  for( unsigned int iBX = 2750; iBX < 3250; ++iBX ){
    EtSumNoise[iBX % 4] += lumiSection_.etSum[HLXID].data[iBX];
  }
  
  EtSumNoise[0] /= 125.0;
  EtSumNoise[1] /= 125.0;
  EtSumNoise[2] /= 125.0;
  EtSumNoise[3] /= 125.0;

  
  for(unsigned int BXNum = 0; BXNum < NumBX; BXNum++){
   
    Histos[7]->Fill(BXNum, lumiSection_.lhc[HLXID].data[BXNum]);
   
    if( lumiSection_.hdr.numOrbits > 0){
      Histos[0]->Fill(BXNum, (lumiSection_.etSum[HLXID].data[BXNum] - EtSumNoise[BXNum % 4]) / (float)(lumiSection_.hdr.numOrbits));
       
      for(int k = 0; k < 6; k++){
	Histos[k+1]->Fill(BXNum, ((float)(lumiSection_.occupancy[HLXID].data[k][BXNum]) / (float)(lumiSection_.hdr.numOrbits)));
      }
    }
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
   
   TCanvas* c1 = new TCanvas("c1","c1",700,500);
   //c1->SetLogy();
   c1->SetTicks(1,1);
   
   TH1F* ETLumiHisto      = new TH1F("ETLumi",      "E_{T} Lumi",          NBins_, XMin_, XMax_);
   TH1F* OccLumiSet1Histo = new TH1F("OccLumiSet1", "Occupancy Lumi Set1", NBins_, XMin_, XMax_);
   TH1F* OccLumiSet2Histo = new TH1F("OccLumiSet2", "Occupancy Lumi Set2", NBins_, XMin_, XMax_);
   
   ETLumiHisto->GetXaxis()->SetTitle("Bunch Crossing");
   OccLumiSet1Histo->GetXaxis()->SetTitle("Bunch Crossing");
   OccLumiSet2Histo->GetXaxis()->SetTitle("Bunch Crossing");
   
   for( unsigned int iBX = 0; iBX < 3564; ++iBX ){
      ETLumiHisto->     SetBinContent(iBX, lumiSection_.lumiDetail.ETLumi[iBX]);
      ETLumiHisto->     SetBinError(  iBX, lumiSection_.lumiDetail.ETLumiErr[iBX]);
      OccLumiSet1Histo->SetBinContent(iBX, lumiSection_.lumiDetail.OccLumi[0][iBX]);
      OccLumiSet1Histo->SetBinError(  iBX, lumiSection_.lumiDetail.OccLumiErr[0][iBX]);
      OccLumiSet2Histo->SetBinContent(iBX, lumiSection_.lumiDetail.OccLumi[1][iBX]);
      OccLumiSet2Histo->SetBinError(  iBX, lumiSection_.lumiDetail.OccLumiErr[1][iBX]);
   }

   ETLumiHisto->Draw();
   std::cout << "Saving ETLumi Histo" << std::endl;
   c1->SaveAs( (pageDir + "/Pics/EtSumLumi.png").c_str() );
   OccLumiSet1Histo->Draw();
   std::cout << "Saving Occ Lumi 1 Histo" << std::endl;
   c1->SaveAs( (pageDir + "/Pics/OccLumiSet1.png").c_str() ); 
   OccLumiSet2Histo->Draw();
   std::cout << "Saving Occ Lumi 2 Histo" << std::endl;
   c1->SaveAs( (pageDir + "/Pics/OccLumiSet2.png").c_str() );

   std::cout << "Deleting Histograms" << std::endl;

   delete c1;
   delete ETLumiHisto;
   delete OccLumiSet1Histo;
   delete OccLumiSet2Histo;
   
}
