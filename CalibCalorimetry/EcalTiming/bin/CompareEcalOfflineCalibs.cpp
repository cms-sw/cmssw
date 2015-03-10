#include "CompareEcalOfflineCalibs.h"

#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TProfile2D.h"
#include "TStyle.h"
#include "TDirectory.h"
#include "TError.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"



// ****************************************************************
float getBarrelEtaZPos(int ieta)
{
    return barrelEtaZPos[ieta+85];
}

// ****************************************************************
std::string intToString(int num)
{
  using namespace std;
  ostringstream myStream;
  myStream << num << flush;
  return(myStream.str()); //returns the string form of the stringstream object
}

// ****************************************************************
void SetEStyle() {
  TStyle* EStyle = new TStyle("EStyle", "E's not Style");

  //set the background color to white
  EStyle->SetFillColor(10);
  EStyle->SetFrameFillColor(10);
  EStyle->SetFrameFillStyle(0);
  EStyle->SetFillStyle(0);
  EStyle->SetCanvasColor(10);
  EStyle->SetPadColor(10);
  EStyle->SetTitleFillColor(0);
  EStyle->SetStatColor(10);

  //dont put a colored frame around the plots
  EStyle->SetFrameBorderMode(0);
  EStyle->SetCanvasBorderMode(0);
  EStyle->SetPadBorderMode(0);

  //use the primary color palette
  EStyle->SetPalette(1,0);
  EStyle->SetNumberContours(255);

  //set the default line color for a histogram to be black
  EStyle->SetHistLineColor(kBlack);

  //set the default line color for a fit function to be red
  EStyle->SetFuncColor(kRed);

  //make the axis labels black
  EStyle->SetLabelColor(kBlack,"xyz");

  //set the default title color to be black
  EStyle->SetTitleColor(kBlack);
   
  // Sizes

  //For Small Plot needs
  //set the margins
 //  EStyle->SetPadBottomMargin(.2);
//   EStyle->SetPadTopMargin(0.08);
//   EStyle->SetPadLeftMargin(0.12);
//   EStyle->SetPadRightMargin(0.12);

//   //set axis label and title text sizes
//   EStyle->SetLabelSize(0.06,"xyz");
//   EStyle->SetTitleSize(0.06,"xyz");
//   EStyle->SetTitleOffset(1.,"x");
//   EStyle->SetTitleOffset(.9,"yz");
//   EStyle->SetStatFontSize(0.04);
//   EStyle->SetTextSize(0.06);
//   EStyle->SetTitleBorderSize(0.5);
  

  //set the margins
  EStyle->SetPadBottomMargin(.15);
  EStyle->SetPadTopMargin(0.08);
  EStyle->SetPadLeftMargin(0.14);
  EStyle->SetPadRightMargin(0.12);
  
  //set axis label and title text sizes
  EStyle->SetLabelSize(0.04,"xyz");
  EStyle->SetTitleSize(0.06,"xyz");
  EStyle->SetTitleOffset(1.,"x");
  EStyle->SetTitleOffset(1.1,"yz");
  EStyle->SetStatFontSize(0.04);
  EStyle->SetTextSize(0.04);
  EStyle->SetTitleBorderSize(size_t(0.5));
  //EStyle->SetTitleY(0.5);
  
  //set line widths
  EStyle->SetHistLineWidth(1);
  EStyle->SetFrameLineWidth(2);
  EStyle->SetFuncWidth(2);

  //Paper Size
  EStyle->SetPaperSize(TStyle::kUSLetter);

  // Misc

  //align the titles to be centered
  //Style->SetTextAlign(22);

  //set the number of divisions to show
  EStyle->SetNdivisions(506, "xy");

  //turn off xy grids
  EStyle->SetPadGridX(0);
  EStyle->SetPadGridY(0);

  //set the tick mark style
  //EStyle->SetPadTickX(1);
  //EStyle->SetPadTickY(1);

  //show the fit parameters in a box
  EStyle->SetOptFit(111111);

  //turn on all other stats
   //EStyle->SetOptStat(0000000);
  //EStyle->SetOptStat(1111111);
  //With errors
  EStyle->SetOptStat(1112211);

  //Move stats box
  //EStyle->SetStatX(0.85);

  //marker settings
  EStyle->SetMarkerStyle(8);
  EStyle->SetMarkerSize(0.8);
   
  // Fonts
   EStyle->SetStatFont(42);
   EStyle->SetLabelFont(42,"xyz");
   EStyle->SetTitleFont(42,"xyz");
   EStyle->SetTextFont(42);
//  EStyle->SetStatFont(82);
//   EStyle->SetLabelFont(82,"xyz");
//   EStyle->SetTitleFont(82,"xyz");
//   EStyle->SetTextFont(82);


  //done
  EStyle->cd();
}


//XXX: Main
int main(int argc, char* argv[])
{
  using namespace std;
  char* infile1 = argv[1];
  char* infile2 = argv[2];

  if(!infile1 && !infile2)
  {
    std::cout << "Error: 2 input files were not specified." << std::endl;
    std::cout << "Usage: compareCalibs correctedCalibFileRun1.txt" <<
     " correctedCalibFileRun2.txt" << std::endl;
    return -1;
  }

  //Set TStyle
  SetEStyle();

  ifstream calibFile1, calibFile2;
  calibFile1.open(infile1,ifstream::in);
  calibFile2.open(infile2,ifstream::in);
  map<int,double> timingCalibMapEB1;
  map<int,double> timingCalibErrMapEB1;
  map<int,double> timingCalibMapEB2;
  map<int,double> timingCalibErrMapEB2;
  map<int,double> timingCalibMapEE1;
  map<int,double> timingCalibErrMapEE1;
  map<int,double> timingCalibMapEE2;
  map<int,double> timingCalibErrMapEE2;

  //Open and read calibFile1
  if(calibFile1.good())
  {
    int hashIndex;
    double calibConst, calibConstErr;
    string subDet;
    while(calibFile1.good())
    {
      calibFile1 >> subDet >> hashIndex >> calibConst >> calibConstErr;
      //cout << "(Map1) Hash: " << hashIndex << "\t calib: " << calibConst << endl;
      if(subDet=="EB")
      {
        timingCalibMapEB1.insert(std::make_pair(hashIndex,calibConst));
        timingCalibErrMapEB1.insert(std::make_pair(hashIndex,calibConstErr));
      }
      else if(subDet=="EE")
      {
        timingCalibMapEE1.insert(std::make_pair(hashIndex,calibConst));
        timingCalibErrMapEE1.insert(std::make_pair(hashIndex,calibConstErr));
      }
      else
        std::cout << "ERROR: (calibFile1) subDetector: " << subDet << "not understood" << std::endl;
    }
    calibFile1.close();
  }
  else
  {
    std::cout << "ERROR: Calib file1 not opened." << std::endl;
    return -1;
  }
  //Open and read calibFile2
  if(calibFile2.good())
  {
    int hashIndex;
    double calibConst, calibConstErr;
    string subDet;
    while(calibFile2.good())
    {
      calibFile2 >> subDet >> hashIndex >> calibConst >> calibConstErr;
      //cout << "(Map2) Hash: " << hashIndex << "\t calib: " << calibConst << endl;
      if(subDet=="EB")
      {
        timingCalibMapEB2.insert(std::make_pair(hashIndex,calibConst));
        timingCalibErrMapEB2.insert(std::make_pair(hashIndex,calibConstErr));
      }
      else if(subDet=="EE")
      {
        timingCalibMapEE2.insert(std::make_pair(hashIndex,calibConst));
        timingCalibErrMapEE2.insert(std::make_pair(hashIndex,calibConstErr));
      }
      else
        std::cout << "ERROR: (calibFile2) subDetector: " << subDet << "not understood" << std::endl;
    }
    calibFile2.close();
  }
  else
  {
    std::cout << "ERROR: Calib file2 not opened." << std::endl;
    return -1;
  }
  
// ***
  cout << "Doing " << infile1 << " - " << infile2 << endl;

  TFile* outputTFile = new TFile("compareCalibs.root","RECREATE");

  TCanvas can;
  can.cd();
  TH1F* differenceHist = new TH1F("relativeDifferenceBetweenCalibs","2(calib1-calib2)/(calib1+calib2) [ns]",100,-0.1,0.1);
  TCanvas can2;
  can2.cd();
  TH2F* scatterHist = new TH2F("calibConst1VsCalibConst2","calib1 vs. calib2 [ns]",150,-75,75,150,-75,75);
  TCanvas can3;
  can3.cd();
  TH1F* straightDifferenceHist = new TH1F("differenceBetweenCalibs","calib1-calib2 [ns]",10000,-50,50);
  TCanvas can4;
  can4.cd();
  TProfile* differenceByEtaProfile = new TProfile("differenceByEta","calib1-calib2 [ns]",172,-86,86);
  TH2F* calibDifferenceMapEB = new TH2F("calibDiffMapEB","calib1-calib2 [ns]",360,0,361,172,-86,86);
  TH2F* calibDifferenceMapEEM = new TH2F("calibDiffMapEEM","calib1-calib2 [ns]",100,1,101,100,1,101);
  TH2F* calibDifferenceMapEEP = new TH2F("calibDiffMapEEP","calib1-calib2 [ns]",100,1,101,100,1,101);
  TH2F* calibDifferenceVsCalib1EB = new TH2F("calibDiffVsCalib1EB","#Delta(calib) vs. calib 1 [ns]",150,-75,75,200,-50,50);
  TH2F* calibDifferenceVsCalib2EB = new TH2F("calibDiffVsCalib2EB","#Delta(calib) vs. calib 2 [ns]",150,-75,75,200,-50,50);

  TH2F* calib1Map = new TH2F("calib1Map","Map of 1st calib consts [ns]",360,0,361,172,-86,86);
  TH2F* calib2Map = new TH2F("calib2Map","Map of 2nd calib consts [ns]",360,0,361,172,-86,86);

  TDirectory* calibDiffHists = gDirectory->mkdir("calibDiffHistsInEta");
  calibDiffHists->cd();

  TProfile* calibDiffByEta = new TProfile("calibDiffIeta","#Delta(calib) by i#eta",172,-86,86);
  TProfile* calibDiffSigmaByEta = new TProfile("calibDiffSigmaIeta","Sigma of #Delta(calib) by i#eta",172,-86,86);

  map<int,double> cryHashToDeltaCalibEB;
  //TODO: Implement EE
  map<int,double> cryHashToDeltaCalibEE;

  
  // *** EB
  cout << "INFO: size of EBmap1: " << timingCalibMapEB1.size() <<
     " size of EBmap2: " << timingCalibMapEB2.size() << endl;

  double deltaCalibInIEta[171]; // make ieta==-85 --> 0 here; then ieta==85-->170
  int numCrysInIEta[171];
  for(int i=0; i<171; ++i)
  {
    deltaCalibInIEta[i]=0;
    numCrysInIEta[i]=0;
  }

  // Loop over calibTimingMap1EB and find the corresponding entries in map2
  for(std::map<int,double>::const_iterator map1Itr = timingCalibMapEB1.begin();
      map1Itr != timingCalibMapEB1.end(); ++map1Itr)
  {
    std::map<int,double>::const_iterator map2Itr = timingCalibMapEB2.find(map1Itr->first);
    if(map2Itr==timingCalibMapEB2.end())
    {
      //std::cout << "Could not find crystal: " << map1Itr->first << " in EBmap2." << std::endl;
      continue;
    }
    differenceHist->Fill(2*(map1Itr->second-map2Itr->second)/(map1Itr->second+map2Itr->second));
    scatterHist->Fill(map2Itr->second,map1Itr->second);
    straightDifferenceHist->Fill(map1Itr->second-map2Itr->second);
    if(!EBDetId::validHashIndex(map1Itr->first))
    {
      cout << "ERROR: invalid EB hashIndex (in map1): " << map1Itr->first << endl;
      continue;
    }
    int ieta = (EBDetId::unhashIndex(map1Itr->first)).ieta();
    int iphi = (EBDetId::unhashIndex(map1Itr->first)).iphi();
    double deltaCalib = map1Itr->second-map2Itr->second;
    differenceByEtaProfile->Fill(ieta,deltaCalib);
    calibDifferenceMapEB->Fill(iphi,ieta,deltaCalib);
    calibDifferenceVsCalib1EB->Fill(-1*map1Itr->second,deltaCalib);
    calibDifferenceVsCalib2EB->Fill(-1*map2Itr->second,deltaCalib);
    cryHashToDeltaCalibEB.insert(std::make_pair(map1Itr->first,deltaCalib));
    calib1Map->Fill(iphi,ieta,map1Itr->second);
    calib2Map->Fill(iphi,ieta,map2Itr->second);
    deltaCalibInIEta[ieta+85]+=deltaCalib;
    numCrysInIEta[ieta+85]++;
  }

  for(int i=0;i<171;++i)
  {
    if(numCrysInIEta[i]>0)
      calibDiffByEta->Fill(i-85,deltaCalibInIEta[i]/numCrysInIEta[i]);
  }
  
  // *** EE
  cout << "INFO: size of EEmap1: " << timingCalibMapEE1.size() <<
     " size of EEmap2: " << timingCalibMapEE2.size() << endl;
  // Loop over calibTimingMap1EE and find the corresponding entries in map2
  for(std::map<int,double>::const_iterator map1Itr = timingCalibMapEE1.begin();
      map1Itr != timingCalibMapEE1.end(); ++map1Itr)
  {
    std::map<int,double>::const_iterator map2Itr = timingCalibMapEE2.find(map1Itr->first);
    if(map2Itr==timingCalibMapEE2.end())
    {
      //std::cout << "Could not find crystal: " << map1Itr->first << " in EEmap2." << std::endl;
      continue;
    }
    differenceHist->Fill(2*(map1Itr->second-map2Itr->second)/(map1Itr->second+map2Itr->second));
    scatterHist->Fill(map1Itr->second,map2Itr->second);
    straightDifferenceHist->Fill(map1Itr->second-map2Itr->second);
    if(!EEDetId::validHashIndex(map1Itr->first))
    {
      cout << "ERROR: invalid EE hashIndex (in map1): " << map1Itr->first << endl;
      continue;
    }
    int ix = (EEDetId::unhashIndex(map1Itr->first)).ix();
    int iy = (EEDetId::unhashIndex(map1Itr->first)).iy();
    int zside = (EEDetId::unhashIndex(map1Itr->first)).zside();
    double deltaCalib = map1Itr->second-map2Itr->second;
    if(zside==1)
      calibDifferenceMapEEP->Fill(ix,iy,deltaCalib);
    else if(zside==-1)
      calibDifferenceMapEEM->Fill(ix,iy,deltaCalib);
    else
      cout << "WARNING: Strange zside found for ix: " << ix << " iy: " << iy
        << " : " << zside << endl;
  }

  //sigmaDifferenceHistEB->Write();

  can.cd();
  differenceHist->Fit("gaus","Q");
  differenceHist->Draw();
  //can.Print("compareCalibsDifferenceHist.png");
  differenceHist->Write();

  can3.cd();
  straightDifferenceHist->Fit("gaus","Q");
  straightDifferenceHist->SetXTitle("#Delta(calib) [ns]");
  straightDifferenceHist->Draw();
  //can3.Print("compareCalibsStraightDifferenceHist.png");
  straightDifferenceHist->Write();

  can2.cd();
  gStyle->SetOptStat(111111);
  gStyle->SetStatX(0.9);
  scatterHist->Draw("colz");
  //can2.Print("compareCalibsScatterHist.png");
  scatterHist->Write();

  can4.cd();
  differenceByEtaProfile->SetXTitle("i#eta");
  differenceByEtaProfile->Draw();
  //can4.Print("differenceByEtaEB.png");
  differenceByEtaProfile->Write();

  gStyle->SetOptStat(11);
  calibDifferenceMapEB->SetXTitle("i#phi");
  calibDifferenceMapEB->SetYTitle("i#eta");
  calibDifferenceMapEB->Draw("colz");
  calibDifferenceMapEB->Write();

  calibDifferenceMapEEM->SetXTitle("ix");
  calibDifferenceMapEEM->SetYTitle("iy");
  calibDifferenceMapEEM->Draw("colz");
  calibDifferenceMapEEM->Write();

  calibDifferenceMapEEP->SetXTitle("ix");
  calibDifferenceMapEEP->SetYTitle("iy");
  calibDifferenceMapEEP->Draw("colz");
  calibDifferenceMapEEP->Write();

  calibDifferenceVsCalib1EB->SetXTitle("calib1 [ns]");
  calibDifferenceVsCalib2EB->SetXTitle("calib2 [ns]");
  calibDifferenceVsCalib1EB->SetYTitle("calib_-calib_ [ns]");
  calibDifferenceVsCalib2EB->SetYTitle("calib_-calib_ [ns]");
  calibDifferenceVsCalib1EB->Write();
  calibDifferenceVsCalib2EB->Write();

  calib1Map->SetXTitle("i#phi");
  calib1Map->SetYTitle("i#eta");
  calib1Map->Write();
  calib2Map->SetXTitle("i#phi");
  calib2Map->SetYTitle("i#eta");
  calib2Map->Write();

  calibDiffSigmaByEta->SetXTitle("i#eta");
  calibDiffSigmaByEta->Write();
  calibDiffByEta->SetXTitle("i#eta");
  calibDiffByEta->Write();

  //outputTFile->Write();
  outputTFile->Close();
  return 0;
}
