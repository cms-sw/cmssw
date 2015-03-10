// -*- C++ -*-
//
//
//
/*
Description: adds endcap crytal time histograms produced from EcalCreateTimeCalibrations from a given region

Implementation:
execute as a root macro with arguments:
1) fileName is the name of a root file produced by EcalCreateTimeCalibrations code
2) zside is +1 for EEP or -1 for EEM
3) ixMin is the minimum crystal ix-value
4) ixMax is the maximum crystal ix-value
5) iyMin is the minimum crystal iy-value
6) iyMax is the maximum crystal iy-value
*/
// Authors:                    Jared Turkewitz (Minnesota)
//          Created: Mon Aug 8 18:02 CEST 2011
//

#include <string.h>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
#include "TKey.h"
#include "Riostream.h"

void viewCalibrationHistsFromEERegion(char *fileName, int zside, int ixMin, int ixMax, int iyMin, int iyMax)
{
  std::cout << "zside: " << zside << std::endl;
  std::cout << "ixMin: " << ixMin << " ixMax: " << ixMax << std::endl;
  std::cout << "iyMin: " << iyMin << " iyMax: " << iyMax << std::endl;
  if(ixMin > ixMax)
  {
    std::cout << "ixMin " << ixMin << " > ixMax " << ixMax << " please fix this" << std::endl;
    return;
  }
  if(iyMin > iyMax)
  {
    std::cout << "iyMin " << iyMin << " > iyMax " << iyMax << " please fix this" << std::endl;
    return; 
  }
  if (zside != 1 && zside != -1)
  {
    std::cout << "zside " << zside << " should either be -1 or 1 please fix this" << std::endl;
    return;
  }
  char *zsideName;
  if (zside == 1)
  {
    zsideName = "P";
  }
  else
  {
    zsideName = "M";
  }
  char summedHistName[100];
  sprintf(summedHistName,"EE%s_cryTiming_ix%d-%d_iy%d-%d",zsideName,ixMin,ixMax,iyMin,iyMax);
  TH1F* summedHist = new TH1F("summedHist",summedHistName,200,-10,10);
  summedHist->GetXaxis()->SetTitle("Reco Time [ns]");
  summedHist->GetYaxis()->SetTitle("Counts / 0.1 [ns]");
  summedHist->Sumw2();
  TFile *theFile = new TFile(fileName);
  for (int ix = ixMin ; ix<=ixMax ; ix++)
  {
    for (int iy = iyMin ; iy<=iyMax ; iy++ )
    {
      char histName[100];
      sprintf(histName,"createTimeCalibs/crystalTimingHistsEE%s/EE%s_cryTiming_ix%d_iy%d",zsideName,zsideName,ix,iy);
      TH1F *crystalHist = (TH1F*) theFile->Get(histName);
      summedHist->Add(crystalHist);
    }
  }


  TH2F *calibDiffMapEEM   = (TH2F*) theFile->Get("createTimeCalibs/calibDiffMapEEM");
  TH2F *calibDiffMapEEMFromSelectedRegion   = (TH2F*) theFile->Get("createTimeCalibs/calibDiffMapEEM");
  TH2F *calibDiffMapEEP   = (TH2F*) theFile->Get("createTimeCalibs/calibDiffMapEEP");
  TH2F *calibDiffMapEEPFromSelectedRegion   = (TH2F*) theFile->Get("createTimeCalibs/calibDiffMapEEP");
  
  char *dataType = fileName;
  string fileNameString = dataType;
  int lastSlash = fileNameString.find("-");
  int lastDot = fileNameString.rfind(".");
  string pictureNameStart;
  pictureNameStart = fileNameString.substr(lastSlash+1,lastDot-lastSlash-1);
  string canvasName = "";
  canvasName+=pictureNameStart.c_str();
//  std::cout << "pictureNameStart " << pictureNameStart.c_str() << std::endl;
  float minTimeAverage = -1.5;//Min time for z axis on timing maps hists
  float maxTimeAverage = 1.5;


  gStyle->SetOptStat(2222210);
  TCanvas *c1 = new TCanvas((canvasName+summedHistName).c_str(), summedHistName);
  c1->cd();
  c1->SetLogy();
  summedHist->Fit("gaus","Q");
  summedHist->Draw();
  TImage *img = TImage::Create();
  img->FromPad(c1);
//  img->WriteImage((canvasName+summedHistName+".png").c_str());
  delete img;

  if(zside == -1)
  {

    TCanvas *c2 = new TCanvas((canvasName+"_calibDiffMapEEM").c_str(), "c2");
    c2->cd();
    gStyle->SetOptStat(10);
    calibDiffMapEEM->SetTitle("Mean Time EEM [ns]");
    calibDiffMapEEM->GetXaxis()->SetTitle("ix");
    calibDiffMapEEM->GetYaxis()->SetTitle("iy");
    calibDiffMapEEM->SetMinimum(minTimeAverage);
    calibDiffMapEEM->SetMaximum(maxTimeAverage);
    calibDiffMapEEM->Draw("colz");
    TImage *img = TImage::Create();
    img->FromPad(c2);
//    img->WriteImage((canvasName+"_calibDiffMapEEM"+".png").c_str());
    delete img;

    TCanvas *c3 = new TCanvas((canvasName+"_calibDiffMapEEMFromSelectedRegion").c_str(), "c3");
    c3->cd();
    calibDiffMapEEMFromSelectedRegion->SetTitle("Mean Time EEM [ns]");
    calibDiffMapEEMFromSelectedRegion->GetXaxis()->SetTitle("ix");
    calibDiffMapEEMFromSelectedRegion->GetYaxis()->SetTitle("iy");
    calibDiffMapEEMFromSelectedRegion->GetXaxis()->SetRangeUser(ixMin,ixMax);
    calibDiffMapEEMFromSelectedRegion->GetYaxis()->SetRangeUser(iyMin,iyMax);
    calibDiffMapEEMFromSelectedRegion->SetMinimum(minTimeAverage);
    calibDiffMapEEMFromSelectedRegion->SetMaximum(maxTimeAverage);
    calibDiffMapEEMFromSelectedRegion->Draw("colz");
    TImage *img = TImage::Create();
    img->FromPad(c3);
//    img->WriteImage((canvasName+"_calibDiffMapEEMFromSelectedRegion"+".png").c_str());
    delete img;
  }

  if (zside == 1)
  {
    TCanvas *c4 = new TCanvas((canvasName+"_calibDiffMapEEP").c_str(), "c4");
    c4->cd();
    calibDiffMapEEP->SetTitle("Mean Time EEP [ns]");
    calibDiffMapEEP->GetXaxis()->SetTitle("ix");
    calibDiffMapEEP->GetYaxis()->SetTitle("iy");
    calibDiffMapEEP->SetMinimum(minTimeAverage);
    calibDiffMapEEP->SetMaximum(maxTimeAverage);
    calibDiffMapEEP->Draw("colz");
    TImage *img = TImage::Create();
    img->FromPad(c4);
//  img->WriteImage((canvasName+"_calibDiffMapEEP"+".png").c_str());
    delete img;

    TCanvas *c5 = new TCanvas((canvasName+"_calibDiffMapEEPFromSelectedRegion").c_str(), "c5");
    c5->cd();
    calibDiffMapEEPFromSelectedRegion->SetTitle("Mean Time EEP [ns]");
    calibDiffMapEEPFromSelectedRegion->GetXaxis()->SetTitle("ix");
    calibDiffMapEEPFromSelectedRegion->GetYaxis()->SetTitle("iy");
    calibDiffMapEEPFromSelectedRegion->GetXaxis()->SetRangeUser(ixMin,ixMax);
    calibDiffMapEEPFromSelectedRegion->GetYaxis()->SetRangeUser(iyMin,iyMax);
    calibDiffMapEEPFromSelectedRegion->SetMinimum(minTimeAverage);
    calibDiffMapEEPFromSelectedRegion->SetMaximum(maxTimeAverage);
    calibDiffMapEEPFromSelectedRegion->Draw("colz");
    TImage *img = TImage::Create();
    img->FromPad(c5);
//  img->WriteImage((canvasName+"_calibDiffMapEEPFromSelectedRegion"+".png").c_str());
    delete img;
  }
}
