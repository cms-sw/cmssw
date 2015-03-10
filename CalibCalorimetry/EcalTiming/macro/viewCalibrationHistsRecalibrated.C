#include <string.h>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
#include "TKey.h"
#include "Riostream.h"

void viewCalibrationHistsRecalibrated(char *fileName)
{
  std::cout << fileName << std::endl;
  TFile *theFile = new TFile(fileName);

  TH1F *timingCalibDiffEB = (TH1F*) theFile->Get("recalibratedTimeHists/timingCalibDiffEB");
  TH1F *timingCalibDiffsEEM = (TH1F*) theFile->Get("recalibratedTimeHists/timingCalibDiffsEEM");
  TH1F *timingCalibDiffsEEP = (TH1F*) theFile->Get("recalibratedTimeHists/timingCalibDiffsEEP");
  TH2F *hitsPerCryMapEB = (TH2F*) theFile->Get("recalibratedTimeHists/hitsPerCryMapEB");
  TH2F *hitsPerCryMapEEM = (TH2F*) theFile->Get("recalibratedTimeHists/hitsPerCryMapEEM");
  TH2F *hitsPerCryMapEEP = (TH2F*) theFile->Get("recalibratedTimeHists/hitsPerCryMapEEP");
  TH2F *calibDiffMapEB   = (TH2F*) theFile->Get("recalibratedTimeHists/calibDiffMapEB");
  TH2F *calibDiffMapEEM   = (TH2F*) theFile->Get("recalibratedTimeHists/calibDiffMapEEM");
  TH2F *calibDiffMapEEP   = (TH2F*) theFile->Get("recalibratedTimeHists/calibDiffMapEEP");
  
  TH1F *sigmaCalibsEB = (TH1F*) theFile->Get("recalibratedTimeHists/sigmaCalibsEB");
  TH1F *sigmaCalibsEE = (TH1F*) theFile->Get("recalibratedTimeHists/sigmaCalibDiffsEE");

  TH2F *sigmaMapEB   = (TH2F*) theFile->Get("recalibratedTimeHists/sigmaDiffMapEB");
  TH2F *sigmaMapEEM   = (TH2F*) theFile->Get("recalibratedTimeHists/sigmaMapEEM");
  TH2F *sigmaMapEEP   = (TH2F*) theFile->Get("recalibratedTimeHists/sigmaMapEEP");

  float minTimeAverageDistribution = -3.0;//Min time for x axis on time distribution hists
  float maxTimeAverageDistribution = 3.0;
  float minTimeAverage = -1.0;//Min time for z axis on timing maps hists
  float maxTimeAverage = 1.0;
  
  char *dataType = fileName;
  string fileNameString = dataType;
  int lastSlash = fileNameString.find("-");
  int lastDot = fileNameString.rfind(".");
  string pictureNameStart;
  pictureNameStart = fileNameString.substr(lastSlash+1,lastDot-lastSlash-1);
  string canvasName;
  canvasName+=pictureNameStart.c_str();
  std::cout << "pictureNameStart " << pictureNameStart.c_str() << std::endl;

  float crystalsUsedEB = timingCalibDiffEB->GetEntries();
  float crystalsUsedEEM = timingCalibDiffsEEM->GetEntries();
  float crystalsUsedEEP = timingCalibDiffsEEP->GetEntries();
  float hitsUsedEB = hitsPerCryMapEB->Integral();
  float hitsUsedEEM = hitsPerCryMapEEM->Integral();
  float hitsUsedEEP = hitsPerCryMapEEP->Integral();
  float hitsPerCrystalEB = hitsUsedEB / crystalsUsedEB;
  float hitsPerCrystalEEM = hitsUsedEEM / crystalsUsedEEM;
  float hitsPerCrystalEEP = hitsUsedEEP / crystalsUsedEEP;
  
  if(crystalsUsedEB > 50000)
  {
    std::cout << "crystalsUsedEB: " << crystalsUsedEB << " hitsPerCrystalEB: " << hitsPerCrystalEB << "\ncrystalsUsedEEM: " << crystalsUsedEEM  << " hitsPerCrystalEEM: " << hitsPerCrystalEEM  << "\ncrystalsUsedEEP: " << crystalsUsedEEP  << " hitsPerCrystalEEP: " << hitsPerCrystalEEP << std::endl;
  }
  if(crystalsUsedEB <= 50000)
  {
    std::cout << "Less than 50,000 crystals in EB - Using total number of crystals in each region to calculate hits per crystal\n" << "crystalsUsedEB: " << crystalsUsedEB << " hitsPerCrystalEB: " << hitsUsedEB / 61200 << "\ncrystalsUsedEEM: " << crystalsUsedEEM  << " hitsPerCrystalEEM: " << hitsUsedEEM/7324  << "\ncrystalsUsedEEP: " << crystalsUsedEEP  << " hitsPerCrystalEEP: " << hitsUsedEEP/7324 << std::endl;
  }

  gStyle->SetOptStat(2222210);
  TCanvas *c1 = new TCanvas((canvasName+"_timingCalibDiffEB").c_str(), "timingCalibDiffEB");
  c1->cd();
  c1->SetLogy();
  timingCalibDiffEB->GetXaxis()->SetRangeUser(minTimeAverageDistribution,maxTimeAverageDistribution);
  timingCalibDiffEB->GetXaxis()->SetTitle("Mean Time [ns]");
  timingCalibDiffEB->GetYaxis()->SetTitle("Crystals / 50 [ps]");
  timingCalibDiffEB->SetTitle("Time Mean Distribution EB [ns]");
  timingCalibDiffEB->Fit("gaus","Q");
  float timingMeanEB = gaus->GetParameter(1);
  float timingSigmaEB = gaus->GetParameter(2);
  timingCalibDiffEB->Draw();
  TImage *img = TImage::Create();
  img->FromPad(c1);
  img->WriteImage((canvasName+"_timingCalibDiffEB"+".png").c_str());
  delete img;

  TCanvas *c2 = new TCanvas((canvasName+"_timingCalibDiffsEEM").c_str(), "c2");
  c2->cd();
  c2->SetLogy();
  timingCalibDiffsEEM->SetTitle("Time Mean Distribution EEM [ns]");
  timingCalibDiffsEEM->GetXaxis()->SetRangeUser(minTimeAverageDistribution,maxTimeAverageDistribution);
  timingCalibDiffsEEM->GetXaxis()->SetTitle("Mean Time [ns]");
  timingCalibDiffsEEM->GetYaxis()->SetTitle("Crystals / 50 [ps]");
  timingCalibDiffsEEM->Fit("gaus","Q");
  float timingMeanEEM = gaus->GetParameter(1);
  float timingSigmaEEM = gaus->GetParameter(2);
  timingCalibDiffsEEM->Draw();
  TImage *img = TImage::Create();
  img->FromPad(c2);
  img->WriteImage((canvasName+"_timingCalibDiffsEEM"+".png").c_str());
  delete img;

  TCanvas *c3 = new TCanvas((canvasName+"_timingCalibDiffsEEP").c_str(), "c3");
  c3->cd();
  c3->SetLogy();
  timingCalibDiffsEEP->SetTitle("Time Mean Distribution EEP [ns]");
  timingCalibDiffsEEP->GetXaxis()->SetRangeUser(minTimeAverageDistribution,maxTimeAverageDistribution);
  timingCalibDiffsEEP->GetXaxis()->SetTitle("Mean Time [ns]");
  timingCalibDiffsEEP->GetYaxis()->SetTitle("Crystals / 50 [ps]");
  timingCalibDiffsEEP->Fit("gaus","Q");
  float timingMeanEEP = gaus->GetParameter(1);
  float timingSigmaEEP = gaus->GetParameter(2);
  timingCalibDiffsEEP->Draw();
  TImage *img = TImage::Create();
  img->FromPad(c3);
  img->WriteImage((canvasName+"_timingCalibDiffsEEP"+".png").c_str());
  delete img;

  std::cout << "timingMeanEB: " << timingMeanEB*1000.0 << " timingSigmaEB: " << timingSigmaEB*1000.0 << std::endl;
  std::cout << "timingMeanEEM: " << timingMeanEEM*1000.0 << " timingSigmaEEM: " << timingSigmaEEM*1000.0 << std::endl;
  std::cout << "timingMeanEEP: " << timingMeanEEP*1000.0 << " timingSigmaEEP: " << timingSigmaEEP*1000.0 << std::endl;

  TCanvas *c4 = new TCanvas((canvasName+"_calibDiffMapEB").c_str(), "c4");
  gStyle->SetOptStat(2000010);
  gStyle->SetOptStat(10);
  c4->cd();
  calibDiffMapEB->SetTitle("Mean Time EB [ns]");
  calibDiffMapEB->SetMinimum(minTimeAverage);
  calibDiffMapEB->SetMaximum(maxTimeAverage);
  calibDiffMapEB->Draw("colz");
  TImage *img = TImage::Create();
  img->FromPad(c4);
  img->WriteImage((canvasName+"_calibDiffMapEB"+".png").c_str());
  delete img;

  
  TCanvas *c5 = new TCanvas((canvasName+"_calibDiffMapEEM").c_str(), "c5");
  c5->cd();
  calibDiffMapEEM->SetTitle("Mean Time EEM [ns]");
  calibDiffMapEEM->GetXaxis()->SetTitle("ix");
  calibDiffMapEEM->GetYaxis()->SetTitle("iy");
  calibDiffMapEEM->SetMinimum(minTimeAverage);
  calibDiffMapEEM->SetMaximum(maxTimeAverage);
  calibDiffMapEEM->Draw("colz");
  TImage *img = TImage::Create();
  img->FromPad(c5);
  img->WriteImage((canvasName+"_calibDiffMapEEM"+".png").c_str());
  delete img;

  TCanvas *c6 = new TCanvas((canvasName+"_calibDiffMapEEP").c_str(), "c6");
  c6->cd();
  calibDiffMapEEP->SetTitle("Mean Time EEP [ns]");
  calibDiffMapEEP->GetXaxis()->SetTitle("ix");
  calibDiffMapEEP->GetYaxis()->SetTitle("iy");
  calibDiffMapEEP->SetMinimum(minTimeAverage);
  calibDiffMapEEP->SetMaximum(maxTimeAverage);
  calibDiffMapEEP->Draw("colz");
  TImage *img = TImage::Create();
  img->FromPad(c6);
  img->WriteImage((canvasName+"_calibDiffMapEEP"+".png").c_str());
  delete img;

  TCanvas *c7 = new TCanvas((canvasName+"_hitsPerCryMapEB").c_str(), "c7");
  c7->cd();
  hitsPerCryMapEB->Draw("colz");
  TImage *img = TImage::Create();
  img->FromPad(c7);
  img->WriteImage((canvasName+"_hitsPerCryMapEB"+".png").c_str());
  delete img;

  TCanvas *c8 = new TCanvas((canvasName+"_hitsPerCryMapEEM").c_str(), "c8");
  c8->cd();
  hitsPerCryMapEEM->Draw("colz");
  TImage *img = TImage::Create();
  img->FromPad(c8);
  img->WriteImage((canvasName+"_hitsPerCryMapEEM"+".png").c_str());
  delete img;

  TCanvas *c9 = new TCanvas((canvasName+"_hitsPerCryMapEEP").c_str(), "c9");
  c9->cd();
  hitsPerCryMapEEP->Draw("colz");
  TImage *img = TImage::Create();
  img->FromPad(c9);
  img->WriteImage((canvasName+"_hitsPerCryMapEEP"+".png").c_str());
  delete img;

  TCanvas *c10 = new TCanvas((canvasName+"_sigmaMapEB").c_str(), "c10");
  c10->cd();
  sigmaMapEB->Draw("colz");
  sigmaMapEB->SetTitle("Standard Deviation of Time EB [ns]");
  sigmaMapEB->SetMinimum(0);
  sigmaMapEB->SetMaximum(2);
  TImage *img = TImage::Create();
  img->FromPad(c10);
  img->WriteImage((canvasName+"_sigmaMapEB"+".png").c_str());
  delete img;

  TCanvas *c11 = new TCanvas((canvasName+"_sigmaMapEEM").c_str(), "c11");
  c11->cd();
  sigmaMapEEM->SetTitle("Standard Deviation Of Time EEM [ns]");
  sigmaMapEEM->GetXaxis()->SetTitle("ix");
  sigmaMapEEM->GetYaxis()->SetTitle("iy");
  sigmaMapEEM->SetMinimum(0);
  sigmaMapEEM->SetMaximum(2);
  sigmaMapEEM->Draw("colz");
  TImage *img = TImage::Create();
  img->FromPad(c11);
  img->WriteImage((canvasName+"_sigmaMapEEM"+".png").c_str());
  delete img;

  TCanvas *c12 = new TCanvas((canvasName+"_sigmaMapEEP").c_str(), "c12");
  c12->cd();
  sigmaMapEEP->SetTitle("Standard Deviation of Time EEP [ns]");
  sigmaMapEEP->GetXaxis()->SetTitle("ix");
  sigmaMapEEP->GetYaxis()->SetTitle("iy");
  sigmaMapEEP->SetMinimum(0);
  sigmaMapEEP->SetMaximum(2);
  sigmaMapEEP->Draw("colz");
  TImage *img = TImage::Create();
  img->FromPad(c12);
  img->WriteImage((canvasName+"_sigmaMapEEP"+".png").c_str());
  delete img;

  TCanvas *c13 = new TCanvas((canvasName+"_sigmaCalibsEB").c_str(), "c13");
  c13->cd();
  gStyle->SetOptStat(2222210);
  sigmaCalibsEB->Draw();
  TImage *img = TImage::Create();
  img->FromPad(c13);
  img->WriteImage((canvasName+"_sigmaCalibsEB"+".png").c_str());
  delete img;

  TCanvas *c14 = new TCanvas((canvasName+"_sigmaCalibsEE").c_str(), "c14");
  c14->cd();
  sigmaCalibsEE->Draw();
  TImage *img = TImage::Create();
  img->FromPad(c14);
  img->WriteImage((canvasName+"_sigmaCalibsEE"+".png").c_str());
  delete img;
}
