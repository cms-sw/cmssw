// -*- C++ -*-
//
//
//
/*
Description: adds barrel crytal time histograms produced from EcalCreateTimeCalibrations from a given region

Implementation:
execute as a root macro with arguments:
1) fileName is the name of a root file produced by EcalCreateTimeCalibrations code
2) iphiMin is the minimum crystal iphi-value
3) iphiMax is the maximum crystal iphi-value
4) ietaMin is the minimum crystal ieta-value
5) ietaMax is the maximum crystal ieta-value
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

void viewCalibrationHistsFromEBRegion(char *fileName, int iphiMin, int iphiMax, int ietaMin, int ietaMax)
{
  std::cout << "iphiMin: " << iphiMin << " iphiMax: " << iphiMax << std::endl;
  std::cout << "ietaMin: " << ietaMin << " ietaMax: " << ietaMax << std::endl;
  if(iphiMin > iphiMax)
  {
    std::cout << "iphiMin " << iphiMin << " > iphiMax " << iphiMax << " please fix this" << std::endl;
    return;
  }
  if(ietaMin > ietaMax)
  {
    std::cout << "ietaMin " << ietaMin << " > ietaMax " << ietaMax << " please fix this" << std::endl;
    return; 
  }
  
  char summedHistName[100];
  sprintf(summedHistName,"EB_cryTiming_iphi%d-%d_ieta%d-%d",iphiMin,iphiMax,ietaMin,ietaMax);
  TH1F* summedHist = new TH1F("summedHist",summedHistName,200,-10,10);
  summedHist->GetXaxis()->SetTitle("Reco Time [ns]");
  summedHist->GetYaxis()->SetTitle("Counts / 0.1 [ns]");
  summedHist->Sumw2();
  TFile *theFile = new TFile(fileName);
  for (int iphi = iphiMin ; iphi<=iphiMax ; iphi++)
  {
    for (int ieta = ietaMin ; ieta<=ietaMax ; ieta++ )
    {
      char histName[100];
      sprintf(histName,"createTimeCalibs/crystalTimingHistsEB/EB_cryTiming_ieta%d_iphi%d",ieta,iphi);
//      std::cout << histName << std::endl;
      TH1F *crystalHist = (TH1F*) theFile->Get(histName);
      summedHist->Add(crystalHist);
    }
  }

  TH2F *calibDiffMapEB   = (TH2F*) theFile->Get("createTimeCalibs/calibDiffMapEB");
  TH2F *calibDiffMapEBFromSelectedRegion = (TH2F*) calibDiffMapEB->Clone("calibDiffMapEBFromSelectedRegion");
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

  TCanvas *c2 = new TCanvas((canvasName+"_calibDiffMapEB").c_str(), "c2");
//  gStyle->SetOptStat(2000010);
  gStyle->SetOptStat(10);
  c2->cd();
  calibDiffMapEB->SetTitle("Mean Time EB [ns]");
  calibDiffMapEB->SetMinimum(minTimeAverage);
  calibDiffMapEB->SetMaximum(maxTimeAverage);
  calibDiffMapEB->Draw("colz");
  TImage *img = TImage::Create();
  img->FromPad(c2);
//  img->WriteImage((canvasName+"_calibDiffMapEB"+".png").c_str());
  delete img;

  TCanvas *c3 = new TCanvas((canvasName+"_calibDiffMapEBFromSelectedRegion").c_str(), "c3");
  c3->cd();
  calibDiffMapEBFromSelectedRegion->SetTitle("Mean Time EB [ns]");
  calibDiffMapEBFromSelectedRegion->SetMinimum(minTimeAverage);
  calibDiffMapEBFromSelectedRegion->SetMaximum(maxTimeAverage);
  calibDiffMapEBFromSelectedRegion->GetXaxis()->SetRangeUser(iphiMin,iphiMax);
  calibDiffMapEBFromSelectedRegion->GetYaxis()->SetRangeUser(ietaMin,ietaMax);
  calibDiffMapEBFromSelectedRegion->Draw("colz");
  TImage *img = TImage::Create();
  img->FromPad(c3);
//  img->WriteImage((canvasName+"_calibDiffMapEBFromSelectedRegion"+".png").c_str());
  delete img;
}
