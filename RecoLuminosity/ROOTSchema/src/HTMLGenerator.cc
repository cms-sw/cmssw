// STL Headers
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>


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

#include "RecoLuminosity/ROOTSchema/interface/HTMLGenerator.hh"

HCAL_HLX::HTMLGenerator::HTMLGenerator():ROOTFileReader(){

  // TODO: read config file

  plotExt_   = "png";     // config file
  outputDir_ = "./";      // config file
  writeMode_  = 0777;
  
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

  // allocate memory for histograms

  c1_ = new TCanvas("c1","c1",700,500);    
  c1_->SetTicks(1,1);

  NBins_ = 3400;
  XMin_ = 100;
  XMax_ = 3500;
  BinWidth_ = (unsigned int)(XMax_ - XMin_)/NBins_;
  
  for( unsigned int iHisto = 0; iHisto < 8; ++iHisto ){
    HLXHistos_[iHisto] = new TH1F(HistoNames[iHisto].c_str(),"", NBins_, XMin_, XMax_);
    HLXHistos_[iHisto]->GetXaxis()->SetTitle("Bunch Crossing");
    HLXHistos_[iHisto]->GetYaxis()->SetTitleOffset(1.3); 
    HLXHistos_[iHisto]->SetFillColor(kBlue);
  }
  
  HLXHistos_[0]->GetYaxis()->SetTitle("Et Sum (1 LS)");
  HLXHistos_[7]->GetYaxis()->SetTitle("LHC (1 LS)");
  for( unsigned int iHisto = 0; iHisto < 6; ++iHisto ){
    HLXHistos_[iHisto+1]->GetYaxis()->SetTitle("Occupancy (1 LS)");
  }

  EtSummary_     = new TH2F("EtSummary", "Avg Et Sum Per Wedge", 12, -42, 42, 18, 0, 72);
  EtSummary_->GetXaxis()->SetTitle("i #eta");
  EtSummary_->GetYaxis()->SetTitle("i #phi");
  OccSummary_    = new TH2F("OccSummary", "Avg Occupancy Per Wedge", 12, -42, 42, 18, 0, 72);
  OccSummary_->GetXaxis()->SetTitle("i #eta");
  OccSummary_->GetYaxis()->SetTitle("i #phi");

  ETLumiHisto_      = new TH1F("ETLumi",      "E_{T} Lumi",          NBins_, XMin_, XMax_);
  ETLumiHisto_->GetXaxis()->SetTitle("Bunch Crossing");
  OccLumiSet1Histo_ = new TH1F("OccLumiSet1", "Occupancy Lumi Set1", NBins_, XMin_, XMax_);
  OccLumiSet1Histo_->GetXaxis()->SetTitle("Bunch Crossing");
  OccLumiSet2Histo_ = new TH1F("OccLumiSet2", "Occupancy Lumi Set2", NBins_, XMin_, XMax_);
  OccLumiSet2Histo_->GetXaxis()->SetTitle("Bunch Crossing");

}

HCAL_HLX::HTMLGenerator::~HTMLGenerator(){
  
  // deallocate

  delete c1_;

  delete HLXHistos_[0];
  delete HLXHistos_[1];
  delete HLXHistos_[2];
  delete HLXHistos_[3];
  delete HLXHistos_[4];
  delete HLXHistos_[5];
  delete HLXHistos_[6];
  delete HLXHistos_[7];

  delete ETLumiHisto_;
  delete OccLumiSet1Histo_;
  delete OccLumiSet2Histo_;

  delete EtSummary_;
  delete OccSummary_;

}

// ******** Main function ******

void HCAL_HLX::HTMLGenerator::CreateWebPage(){

  GetLumiSection(lumiSection_);
    
  MakeDir( outputDir_ + GetRunDir(), writeMode_ );
  MakeDir( outputDir_ + GetRunDir() + GetLSDir(), writeMode_ );

  for(unsigned int iHLX = 0; iHLX < 36; ++iHLX){
    MakeDir(outputDir_ + GetRunDir() + GetLSDir() + GetHLXDir(iHLX), writeMode_ );
    GenerateHLXPage(iHLX);
    MakeDir(outputDir_ + GetRunDir() + GetLSDir() + GetHLXDir(iHLX) + GetHLXPicDir(iHLX), writeMode_ );
    GenerateHLXPlots(iHLX);
  }

  GenerateAveragePlots();
  GenerateAveragePage();

  GenerateComparePlots();
  GenerateComparePage();

  GenerateLumiPage();

  for(unsigned int iHisto = 0; iHisto < 8; ++iHisto){
    GenerateHistoGroupPage(HistoNames[iHisto]);
  }

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
  fileName = outputDir_ + "index.html";

  if( !fileExists(fileName) ){
    std::string fileTitle = "Luminosity Monitoring System";
    
    MakeEmptyWebPage( fileName, fileTitle);
    InsertLineAfter( fileName, "<H1>\nLuminosity Monitoring System\n</H1>", "<body>");
  }

  const unsigned int runNumber = GetRunNumber();
  if( runNumber != previousRunNumber  ){
    std::stringstream runLine;
    runLine << "<a href = \"" 
	    << GetRunDir() 
	    << "index.html\"> Run - " 
	    << runNumber << "</a>  " 
	    << TimeStampLong() 
	    << "</br>"; 
    previousRunNumber = runNumber;
    InsertLineBefore( fileName, runLine.str(), "</body>"); 
  }

}

void HCAL_HLX::HTMLGenerator::GenerateRunPage(){

  const unsigned int runNumber = GetRunNumber();
  const unsigned int sectionNumber = GetSectionNumber();
  
  std::string fileName;

  fileName = outputDir_ + GetRunDir() + "index.html";  
  
  if(!fileExists(fileName)){
    std::stringstream fileTitle;

    fileTitle << "Luminosity File Reader - Run " << runNumber;
    MakeEmptyWebPage(fileName, fileTitle.str());
  }

  std::stringstream sectionLine;

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
  fileStr << "Luminosity File Reader - " << "Run " << runNumber << " - " 
	  << " Lumi Section - " << sectionNumber << std::endl;
  fileStr << "</title>" << std::endl; 
  fileStr << "<body>" << std::endl;
  
  for(int unsigned iHisto = 0; iHisto < 8; ++iHisto){
    fileStr << "<a href=\"" 
	    << HistoNames[iHisto] 
	    << "/index.html\" >" 
	    << HistoNames[iHisto] <<  "</a></br>" << std::endl;
  }

  fileStr << "<a href=\"Luminosity/index.html\">Luminosity</a>" << std::endl;

  fileStr << "</br>" << std::endl;
  fileStr << "</br>" << std::endl;

  fileStr << "<hr>" << std::endl;
  fileStr << "Summary" << std::endl;
  fileStr << "<hr>" << std::endl;

  // Summary Plots
  // Et Sum Summary
  c1_->cd();

  EtSummary_->Reset();
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
    EtSummary_->Fill( iEta_[iHLX], iPhi_[iHLX], AvgEtSum );
    EtSummary_->Fill( iEta_[iHLX] +7, iPhi_[iHLX], AvgEtSum );
    //    std::cout << "iEta: " << iEta_[iHLX] << " iPhi: " << iPhi_[iHLX] << " AvgEtSum: " << AvgEtSum << std::endl;
  }

  std::string picName = outputDir_ + GetRunDir() + GetLSDir() + "EtSummary.png";
  gStyle->SetOptStat(0);
  gStyle->SetPalette(1);

  EtSummary_->GetZaxis()->SetRangeUser( MinEt, MaxEt );
  EtSummary_->Draw("colz");
  c1_->SaveAs(picName.c_str());

  // Occupancy Summary
  OccSummary_->Reset();
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
      OccSummary_->Fill( iEta_[iHLX] +7 , iPhi_[iHLX], AvgOccSet1 );
      OccSummary_->Fill( iEta_[iHLX],     iPhi_[iHLX], AvgOccSet2 );
    }

    if( iEta_[iHLX] < 0 ){
      OccSummary_->Fill( iEta_[iHLX],     iPhi_[iHLX], AvgOccSet1 );
      OccSummary_->Fill( iEta_[iHLX] + 7, iPhi_[iHLX], AvgOccSet2 );
    }

    //    std::cout << "iEta: " << iEta_[iHLX] << " iPhi: " << iPhi_[iHLX] << " AvgOccSet1: " << AvgOccSet1 << " AvgOccSet2: " << AvgOccSet2 << std::endl;
  }

  picName = outputDir_ + GetRunDir() + GetLSDir() + "OccSummary.png";
  OccSummary_->GetZaxis()->SetRangeUser( MinOcc, MaxOcc );
  OccSummary_->Draw("colz");
  c1_->SaveAs(picName.c_str());

  fileStr << "<a href=\"EtSummary.png\"><img src=\"EtSummary.png\" width=\"45%\"></a>" << std::endl; 
  fileStr << "<a href=\"OccSummary.png\"><img src=\"OccSummary.png\" width=\"45%\"></a>" << std::endl; 

  fileStr << "<hr>" << std::endl;  
  fileStr << "<H2>" << std::endl;
  fileStr << "HLX ID </br>" << std::endl;
  fileStr << "<hr>" << std::endl;
  fileStr << "</H2>" << std::endl;
  fileStr << "<H3>" << std::endl;

  for(unsigned int iHLX = 0; iHLX < 36; ++iHLX){
    fileStr << "<a href=\"HLX" 
	    << std::setw(2) << std::setfill('0') << iHLX 
	    << "/index.html\" >" 
	    << HLXToHFMap_[iHLX]
	    << "</a> </br>" << std::endl;
  }
  
  fileStr << "</H3>" << std::endl;
  fileStr << "</body>" << std::endl;
  fileStr << "</html>" << std::endl;

  fileStr.close();

}

void HCAL_HLX::HTMLGenerator::GenerateHLXPage(const unsigned short int &HLXID){

  // make HLX directory

  const unsigned int runNumber     = GetRunNumber();
  const unsigned int sectionNumber = GetSectionNumber();

  std::stringstream fileName;
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

  for( unsigned int iHisto = 0; iHisto < 8; ++iHisto){
    HLXHistos_[iHisto]->SetBins(NBins_, XMin_, XMax_);
  }
  ETLumiHisto_->SetBins(NBins_, XMin_, XMax_);
  OccLumiSet1Histo_->SetBins(NBins_, XMin_, XMax_);
  OccLumiSet2Histo_->SetBins(NBins_, XMin_, XMax_);

  BinWidth_ = (unsigned int )(XMax_ - XMin_)/NBins_;

}

void HCAL_HLX::HTMLGenerator::GenerateHLXPlots(const unsigned short int & HLXID){

#ifdef DEBUG
  std::cout << "Begin " << __PRETTY_FUNCTION__ << std::endl;
#endif

  for(unsigned int iHisto = 0; iHisto < 8; ++iHisto ){
    HLXHistos_[iHisto]->Reset();
    std::stringstream HistoTitle;
    HistoTitle.str(std::string());
    HistoTitle << HistoNames[iHisto] << " - " << HLXToHFMap_[HLXID];
    HLXHistos_[iHisto]->SetTitle(HistoTitle.str().c_str());
  }

  double EtSumNoise[4];
  
  EtSumNoise[0] = 0;
  EtSumNoise[1] = 0;
  EtSumNoise[2] = 0;
  EtSumNoise[3] = 0;

  // The noise range is arbitrary
  for( unsigned int iBX = 2750; iBX < 3250; ++iBX ){
    EtSumNoise[iBX % 4] += lumiSection_.etSum[HLXID].data[iBX];
  }
  
  EtSumNoise[0] /= 125.0;
  EtSumNoise[1] /= 125.0;
  EtSumNoise[2] /= 125.0;
  EtSumNoise[3] /= 125.0;

  for(unsigned int iBX = 0; iBX < 3564; ++iBX ){
   
    HLXHistos_[7]->Fill( iBX, lumiSection_.lhc[HLXID].data[iBX]);
   
    if( lumiSection_.hdr.numOrbits > 0){
      HLXHistos_[0]->Fill( iBX, (lumiSection_.etSum[HLXID].data[iBX] - EtSumNoise[iBX % 4]) / (float)(lumiSection_.hdr.numOrbits));
       
      for(int k = 0; k < 6; k++){
	HLXHistos_[k+1]->Fill(iBX, ((float)(lumiSection_.occupancy[HLXID].data[k][iBX]) / (float)(lumiSection_.hdr.numOrbits)));
      }
    }
  }

  const std::string HLXPicsDir = outputDir_ + GetRunDir() + GetLSDir() + GetHLXDir(HLXID) + GetHLXPicDir(HLXID); 

  //c1->SetLogy();
  // Draw Histograms and make pngs
  c1_->cd();
  for(unsigned int iHisto = 0; iHisto < 8; ++iHisto ){
    HLXHistos_[iHisto]->Draw();
    std::string plotFileName =  HLXPicsDir +  HistoNames[ iHisto ] + "." + plotExt_;
    c1_->SaveAs(plotFileName.c_str());
  }

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
  MakeDir(pageDir, writeMode_);

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
   MakeDir(pageDir + "/Pics" , writeMode_ );
   
   fileName = pageDir + "index.html";
   std::cout << "********* " << fileName << " **********" << std::endl;

   std::ofstream fileStr;
   
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
   
   //c1->SetLogy();
   c1_->cd();
     
   for( unsigned int iBX = 0; iBX < 3564; ++iBX ){
     int Bin = (int)(iBX - XMin_)/BinWidth_ + 1;
     if( Bin < 1 ) Bin = 0;
     if( Bin > (int)NBins_ ) Bin = NBins_ + 1;

     ETLumiHisto_->    Fill(iBX, lumiSection_.lumiDetail.ETLumi[iBX]);
     ETLumiHisto_->    SetBinError(  Bin, lumiSection_.lumiDetail.ETLumiErr[iBX]);
     OccLumiSet1Histo_->Fill(iBX, lumiSection_.lumiDetail.OccLumi[0][iBX]);
     OccLumiSet1Histo_->SetBinError(  Bin, lumiSection_.lumiDetail.OccLumiErr[0][iBX]);
     OccLumiSet2Histo_->Fill(iBX, lumiSection_.lumiDetail.OccLumi[1][iBX]);
     OccLumiSet2Histo_->SetBinError(  Bin, lumiSection_.lumiDetail.OccLumiErr[1][iBX]);
   }

   ETLumiHisto_->Draw();
   std::cout << "Saving ETLumi Histo" << std::endl;
   c1_->SaveAs( (pageDir + "/Pics/EtSumLumi.png").c_str() );
   OccLumiSet1Histo_->Draw();
   std::cout << "Saving Occ Lumi 1 Histo" << std::endl;
   c1_->SaveAs( (pageDir + "/Pics/OccLumiSet1.png").c_str() ); 
   OccLumiSet2Histo_->Draw();
   std::cout << "Saving Occ Lumi 2 Histo" << std::endl;
   c1_->SaveAs( (pageDir + "/Pics/OccLumiSet2.png").c_str() );

   std::cout << "Deleting Histograms" << std::endl;

}
