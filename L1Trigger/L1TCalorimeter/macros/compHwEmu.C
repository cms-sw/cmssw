#include "TFile.h"
#include "TH1D.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLine.h"

#include <sstream>
#include <string>
#include <iostream>

using std::string;
using std::cout;
using std::endl;
using std::stringstream;

void create_plot(
  TH1D * hw,
  TH1D * emu,
  int runNo,
  const char * dataset,
  const char * xLabel,
  const char * exportPath,
  int rebin = 1,
  int energy = 13,
  long rangeLow = 0,
  long rangeHigh = 0
  ) {

  // define latex container to hold the title
  TLatex n;
  n.SetNDC();
  n.SetTextFont(52);
  n.SetTextSize(0.05);

  // create canvas that will hold each plot
  TCanvas* canv = new TCanvas("canv","canvas");

  // top pad (comparison)
  TPad* pad1 = new TPad("mainPad","mainPad",0,0.3,1,1);
  // bottom (resuduals) pad
  TPad* pad2 = new TPad("ratioPad","ratioPad",0,0.05,1,0.3);

  // pad to contain trendline for hw-emu ratio of 1
  TPad* overlayPad = new TPad("pInv","pInv", 0,0.05,1,0.3);
  overlayPad->SetFillStyle(0);

  // create legend that will describe appearance of data points
  TLegend * leg = new TLegend(0.65,0.85,0.91,1);
  leg->SetFillColor(0);
  leg->SetNColumns(2);
  leg->AddEntry(hw,"Hardware", "p");//"l");
  leg->AddEntry(emu,"Emulator", "l");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);

  // optionally reduce segmentation in x to improve visibility of some plots
  hw->Rebin(rebin);
  emu->Rebin(rebin);

  hw->SetMarkerStyle(21);
  hw->SetMarkerColor(1);
  hw->SetMarkerSize(0.5);
  emu->SetLineColor(kRed);
  hw->GetYaxis()->SetTitle("Number of candidates");
  hw->GetYaxis()->SetTitleSize(0.062);
  hw->GetYaxis()->SetTitleOffset(0.80);
  hw->GetYaxis()->SetLabelSize(0.045);
  hw->GetYaxis()->SetTickSize(0.01);

  hw->GetXaxis()->SetLabelSize(0);

  emu->GetXaxis()->SetLabelSize(0);

  pad1->SetBottomMargin(0.02);
  pad1->SetGridx();
  //pad1->SetLogy();

  pad1->Draw();
  pad1->cd();

  hw->SetStats(0);
  emu->SetStats(0);

  if (emu->GetMaximum() > hw->GetMaximum()) {
    hw->SetMaximum(1.1*emu->GetMaximum());
  }

  if (rangeLow != 0 || rangeHigh != 0) {
    emu->GetXaxis()->SetRangeUser(rangeLow, rangeHigh);
    hw->GetXaxis()->SetRangeUser(rangeLow, rangeHigh);
  }
  hw->DrawCopy("p");
  emu->Draw("same");

  if (rangeLow != 0 || rangeHigh != 0) {
    emu->GetXaxis()->SetRangeUser(rangeLow, rangeHigh);
    hw->GetXaxis()->SetRangeUser(rangeLow, rangeHigh);
  }

  leg->Draw();
  stringstream caption;
  caption << "#bf{CMS Preliminary}: " << dataset;
  /*
    caption << "#bf{CMS Preliminary, 2016 Data}:" << ", #sqrt{s} = " << energy
    << " TeV ";
  */
  n.DrawLatex(0.1, 0.915, caption.str().c_str());

  canv->cd();
  pad2->SetTopMargin(0);
  pad2->SetBottomMargin(0.39);
  pad2->Draw();
  pad2->cd();
  pad2->SetGridy();
  pad2->SetGridx();
  hw->Divide(emu);
  hw->GetYaxis()->SetTitle("HW/EM");
  hw->GetYaxis()->CenterTitle();

  stringstream labelText;
  labelText << "Level-1 Trigger " << xLabel;
  hw->GetYaxis()->CenterTitle();
  hw->GetXaxis()->SetTitle(labelText.str().c_str());

  hw->GetYaxis()->SetTitleSize(0.15);
  hw->GetYaxis()->SetTitleOffset(0.3);
  hw->GetYaxis()->SetLabelSize(0.12);

  hw->GetYaxis()->SetLabelOffset(0.006);
  hw->GetYaxis()->SetNdivisions(40407);

  hw->GetXaxis()->SetTitleSize(0.15);
  hw->GetXaxis()->SetTitleOffset(1.1);
  hw->GetXaxis()->SetLabelOffset(0.04);
  hw->GetXaxis()->SetLabelSize(0.12);

  hw->SetMinimum(0.76);
  hw->SetMaximum(1.24);
  hw->Draw("p");
  canv->cd();
  overlayPad->Draw();
  overlayPad->cd();

  if (rangeLow != 0 || rangeHigh != 0) {
    hw->GetXaxis()->SetRangeUser(rangeLow, rangeHigh);
  }
  
  TLine* unity = new TLine(0.1,0.695,0.9,0.695);
  unity->SetLineColor(kBlue);
  unity->Draw();

  stringstream pathStream;
  pathStream << "compHwEmu/" << exportPath;
  canv->SaveAs(pathStream.str().c_str());

  delete canv;
}


void compHwEmu (
  int runNo, const char * dataset, bool useEventDisplay = false, bool presentationMode = false
  ) {

  const unsigned int evtToDisplay = 18;
  
  stringstream filename;
  stringstream title;

  const char * rootFilename = "l1tCalo_2016_simHistos.root";

  TFile* inFileHw = new TFile(rootFilename);
  TFile* inFileEm = new TFile(rootFilename);

  if (!inFileHw->IsOpen() || !inFileEm->IsOpen()) {
    cout << "Failed to open " << rootFilename << " file." << endl;
    exit(0);
  } 

  // Jets

  // Jet Et
  TH1D* hwMPJetEt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpjet/et");
  TH1D* emMPJetEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpjet/et");
  TH1D* hwMPJetEtSat = (TH1D*) new TH1D(*hwMPJetEt);
  TH1D* emMPJetEtSat = (TH1D*) new TH1D(*emMPJetEt);
  TH1D* hwJetEt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/jet/et");
  TH1D* emJetEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/jet/et");

  // jet eta
  TH1D* hwMPJetEta = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpjet/eta");
  TH1D* emMPJetEta = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpjet/eta");
  TH1D* hwJetEta = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/jet/eta");
  TH1D* emJetEta = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/jet/eta");

  // jet phi
  TH1D* hwMPJetPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpjet/phi");
  TH1D* emMPJetPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpjet/phi");
  TH1D* hwJetPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/jet/phi");
  TH1D* emJetPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/jet/phi");

  // MP sums

  // ETT 
  TH1D* hwMPSumEt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsumet/et");
  TH1D* emMPSumEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsumet/et");

  TH1D* hwMPSumEtSat = (TH1D*) new TH1D(*hwMPSumEt);
  TH1D* emMPSumEtSat = (TH1D*) new TH1D(*emMPSumEt);

  // ETTHF
  TH1D* hwMPSumEtHF = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsumethf/et");
  TH1D* emMPSumEtHF = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsumethf/et");

  TH1D* hwMPSumEtHFSat = (TH1D*) new TH1D(*hwMPSumEtHF);
  TH1D* emMPSumEtHFSat = (TH1D*) new TH1D(*emMPSumEtHF);

 
  // ETTEM
  TH1D* hwMPSumEtEM = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsumetem/et");
  TH1D* emMPSumEtEM = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsumetem/et");
  
   // ETTEM
  TH1D* hwMPSumEtEMSat = (TH1D*) new TH1D(*hwMPSumEtEM);
  TH1D* emMPSumEtEMSat = (TH1D*) new TH1D(*emMPSumEtEM);
 
  // ETx
  TH1D* hwMPSumEtx = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsummetx/et");
  TH1D* emMPSumEtx = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummetx/et");

  // ETx Sat
  TH1D* hwMPSumEtxSat = (TH1D*) new TH1D(*hwMPSumEtx);
  TH1D* emMPSumEtxSat = (TH1D*) new TH1D(*emMPSumEtx);

  // ETxHF
  TH1D* hwMPSumEtxHF = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsummetxhf/et");
  TH1D* emMPSumEtxHF = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummetxhf/et");

  // ETx Sat
  TH1D* hwMPSumEtxHFSat = (TH1D*) new TH1D(*hwMPSumEtxHF);
  TH1D* emMPSumEtxHFSat = (TH1D*) new TH1D(*emMPSumEtxHF);
  

  // ETy
  TH1D* hwMPSumEty = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsummety/et");
  TH1D* emMPSumEty = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummety/et");

  // ETy Sat
  TH1D* hwMPSumEtySat = (TH1D*) new TH1D(*hwMPSumEty);
  TH1D* emMPSumEtySat = (TH1D*) new TH1D(*emMPSumEty);

  // ETyHF
  TH1D* hwMPSumEtyHF = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsummetyhf/et");
  TH1D* emMPSumEtyHF = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummetyhf/et");

  // ETy Sat
  TH1D* hwMPSumEtyHFSat = (TH1D*) new TH1D(*hwMPSumEtyHF);
  TH1D* emMPSumEtyHFSat = (TH1D*) new TH1D(*emMPSumEtyHF);

  // HTT
  TH1D* hwMPSumHt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsumht/et");
  TH1D* emMPSumHt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsumht/et");

  // HTT
  TH1D* hwMPSumHtSat = (TH1D*) new TH1D(*hwMPSumHt);
  TH1D* emMPSumHtSat = (TH1D*) new TH1D(*emMPSumHt);

  // HTTHF
  TH1D* hwMPSumHtHF = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsumhthf/et");
  TH1D* emMPSumHtHF = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsumhthf/et");

  // HTT
  TH1D* hwMPSumHtHFSat = (TH1D*) new TH1D(*hwMPSumHtHF);
  TH1D* emMPSumHtHFSat = (TH1D*) new TH1D(*emMPSumHtHF);

  // HTx
  TH1D* hwMPSumHtx = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsummhtx/et");
  TH1D* emMPSumHtx = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummhtx/et");

  // HTx Sat
  TH1D* hwMPSumHtxSat = (TH1D*) new TH1D(*hwMPSumHtx);
  TH1D* emMPSumHtxSat = (TH1D*) new TH1D(*emMPSumHtx);

  // HTxHF
  TH1D* hwMPSumHtxHF = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsummhtxhf/et");
  TH1D* emMPSumHtxHF = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummhtxhf/et");

  // HTxHF Sat
  TH1D* hwMPSumHtxHFSat = (TH1D*) new TH1D(*hwMPSumHtxHF);
  TH1D* emMPSumHtxHFSat = (TH1D*) new TH1D(*emMPSumHtxHF);


  // HTy
  TH1D* hwMPSumHty = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsummhty/et");
  TH1D* emMPSumHty = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummhty/et");

  // HTy Sat
  TH1D* hwMPSumHtySat = (TH1D*) new TH1D(*hwMPSumHty);
  TH1D* emMPSumHtySat = (TH1D*) new TH1D(*emMPSumHty);

  // HTyHF
  TH1D* hwMPSumHtyHF = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsummhtyhf/et");
  TH1D* emMPSumHtyHF = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummhtyhf/et");

  // HTyHF Sat
  TH1D* hwMPSumHtyHFSat = (TH1D*) new TH1D(*hwMPSumHtyHF);
  TH1D* emMPSumHtyHFSat = (TH1D*) new TH1D(*emMPSumHtyHF);

  
  // HITowerCount
  TH1D* hwMPSumHITowerCount = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsumhitowercount/et");
  TH1D* emMPSumHITowerCount = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsumhitowercount/et");

  // Demux sums

  // ETT
  TH1D* hwSumEt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sumet/et");
  TH1D* emSumEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumet/et");

  /*
  // ETT
  TH1D* hwSumEtHF = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sumet/et");
  TH1D* emSumEtHF = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumet/et");
  */

  // ETTEM
  TH1D* hwSumEtEM = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sumetem/et");
  TH1D* emSumEtEM = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumetem/et");

  // MET
  TH1D* hwSumMet = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/summet/et");
  TH1D* emSumMet = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summet/et");

  // METHF
  TH1D* hwSumMetHF = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/summethf/et");
  TH1D* emSumMetHF = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summethf/et");

  // HTT
  TH1D* hwSumHt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sumht/et");
  TH1D* emSumHt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumht/et");


  // MHT
  TH1D* hwSumMht = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/summht/et");
  TH1D* emSumMht = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summht/et");

  // MHTHF
  TH1D* hwSumMhtHF = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/summhthf/et");
  TH1D* emSumMhtHF = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summhthf/et");

  // MET phi
  TH1D* hwMetPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/summet/phi");
  TH1D* emMetPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summet/phi");

  // METHF phi
  TH1D* hwMetHFPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/summethf/phi");
  TH1D* emMetHFPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summethf/phi");

  // MHT phi
  TH1D* hwMhtPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/summht/phi");
  TH1D* emMhtPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summht/phi");

  // MHTHF phi
  TH1D* hwMhtHFPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/summhthf/phi");
  TH1D* emMhtHFPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summhthf/phi");

   // HI Tower count
  TH1D* hwHITowerCount = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sumhitowercount/et");
  TH1D* emHITowerCount = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumhitowercount/et");
  
  // HI centrality
  TH1D* hwHICentrality = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sumcentrality/et");
  TH1D* emHICentrality = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumcentrality/et");

  // ET Asym
  TH1D* hwEtAsymSum = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sumasymet/et");
  TH1D* emEtAsymSum = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumasymet/et");

  // HT Asym
  TH1D* hwHtAsymSum = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sumasymht/et");
  TH1D* emHtAsymSum = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumasymht/et");

  // ET HF Asym
  TH1D* hwEtAsymSumHF = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sumasymethf/et");
  TH1D* emEtAsymSumHF = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumasymethf/et");

  // HT HF Asym
  TH1D* hwHtAsymSumHF = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sumasymhthf/et");
  TH1D* emHtAsymSumHF = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumasymhthf/et");


  // Sorts
  TH1D* hwSortMP = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sortMP");
  TH1D* emSortMP = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sortMP");
  TH1D* hwSort = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sort");
  TH1D* emSort = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sort");

  // EG Et
  TH1D* hwMPEgEt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpeg/et");
  TH1D* emMPEgEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpeg/et");
  TH1D* hwEgEt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/eg/et");
  TH1D* emEgEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/eg/et");

  // EG eta
  TH1D* hwMPEgEta = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpeg/eta");
  TH1D* emMPEgEta = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpeg/eta");
  TH1D* hwEgEta = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/eg/eta");
  TH1D* emEgEta = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/eg/eta");

  // EG phi
  TH1D* hwMPEgPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpeg/phi");
  TH1D* emMPEgPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpeg/phi");
  TH1D* hwEgPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/eg/phi");
  TH1D* emEgPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/eg/phi");

  // EG iso
  TH1D* hwMPEgIso = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpeg/iso");
  TH1D* emMPEgIso = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpeg/iso");
  TH1D* hwEgIso = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/eg/iso");
  TH1D* emEgIso = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/eg/iso");
  

  // Tau Et
  TH1D* hwMPTauEt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mptau/et");
  TH1D* emMPTauEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mptau/et");
  TH1D* hwTauEt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/tau/et");
  TH1D* emTauEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/tau/et");

  // Tau eta
  TH1D* hwMPTauEta = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mptau/eta");
  TH1D* emMPTauEta = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mptau/eta");
  TH1D* hwTauEta = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/tau/eta");
  TH1D* emTauEta = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/tau/eta");

  // Tau phi
  TH1D* hwMPTauPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mptau/phi");
  TH1D* emMPTauPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mptau/phi");
  TH1D* hwTauPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/tau/phi");
  TH1D* emTauPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/tau/phi");

  // HF feature bits Demux
  TH1D* hwMinBiasHFp0 = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/minbiashfp0/et");
  TH1D* emMinBiasHFp0 = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/minbiashfp0/et");
  TH1D* hwMinBiasHFm0 = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/minbiashfm0/et");
  TH1D* emMinBiasHFm0 = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/minbiashfm0/et");

  TH1D* hwMinBiasHFp1 = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/minbiashfp1/et");
  TH1D* emMinBiasHFp1 = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/minbiashfp1/et");
  TH1D* hwMinBiasHFm1 = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/minbiashfm1/et");
  TH1D* emMinBiasHFm1 = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/minbiashfm1/et");

  // HF feature bits MP
  TH1D* hwMPMinBiasHFp0 = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpminbiashfp0/et");
  TH1D* emMPMinBiasHFp0 = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/mpminbiashfp0/et");
  TH1D* hwMPMinBiasHFm0 = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpminbiashfm0/et");
  TH1D* emMPMinBiasHFm0 = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/mpminbiashfm0/et");

  TH1D* hwMPMinBiasHFp1 = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpminbiashfp1/et");
  TH1D* emMPMinBiasHFp1 = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/mpminbiashfp1/et");
  TH1D* hwMPMinBiasHFm1 = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpminbiashfm1/et");
  TH1D* emMPMinBiasHFm1 = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/mpminbiashfm1/et");

  // HF feature bits MP
  create_plot(
    hwMPMinBiasHFp0, emMPMinBiasHFp0, runNo, dataset,
    "HF tower HCAL flags (p0)", "MPSums/HFSumP0.pdf"
    );

  create_plot(
    hwMPMinBiasHFm0, emMPMinBiasHFm0, runNo, dataset,
    "HF tower HCAL flags (m0)", "MPSums/HFSumM0.pdf"
    );

  create_plot(
    hwMPMinBiasHFp1, emMPMinBiasHFp1, runNo, dataset,
    "HF tower HCAL flags (p1)", "MPSums/HFSumP1.pdf"
    );

  create_plot(
    hwMPMinBiasHFm1, emMPMinBiasHFm1, runNo, dataset,
    "HF tower HCAL flags (m1)", "MPSums/HFSumM1.pdf"
    );

  // HF feature bits Demux
  create_plot(
    hwMinBiasHFp0, emMinBiasHFp0, runNo, dataset,
    "HF tower HCAL flags (p0)", "DemuxSums/HFSumP0.pdf"
    );

  create_plot(
    hwMinBiasHFm0, emMinBiasHFm0, runNo, dataset,
    "HF tower HCAL flags (m0)", "DemuxSums/HFSumM0.pdf"
    );

  create_plot(
    hwMinBiasHFp1, emMinBiasHFp1, runNo, dataset,
    "HF tower HCAL flags (p1)", "DemuxSums/HFSumP1.pdf"
    );

  create_plot(
    hwMinBiasHFm1, emMinBiasHFm1, runNo, dataset,
    "HF tower HCAL flags (m1)", "DemuxSums/HFSumM1.pdf"
    );

// ========================== jets start ========================
  if (presentationMode) {
    // plot MP jet Et
    create_plot(
      hwMPJetEt, emMPJetEt, runNo, dataset,
      "Jet iE_{T}", "Jets/JetEt.pdf", 2, 13, 0, 200
      );

    // plot MP jet eta
    create_plot(
      hwMPJetEta, emMPJetEta, runNo, dataset,
      "Jet i#eta", "Jets/JetEta.pdf", 2
      );

    // plot MP jet phi
    create_plot(
      hwMPJetPhi, emMPJetPhi, runNo, dataset,
      "Jet i#phi", "Jets/JetPhi.pdf", 2
      );

    // plot demux jet Et
    create_plot(
      hwJetEt, emJetEt, runNo, dataset,
      "Jet iE_{T}", "DemuxJets/JetEt.pdf", 2, 13, 0, 200
      );

    // plot demux jet eta
    create_plot(
      hwJetEta, emJetEta, runNo, dataset,
      "Jet i#eta", "DemuxJets/JetEta.pdf", 2
      );

    // plot demux jet phi
    create_plot(
      hwJetPhi, emJetPhi, runNo, dataset,
      "Jet i#phi", "DemuxJets/JetPhi.pdf", 4
      );
  } else {
    // plot MP jet Et
    create_plot(
      hwMPJetEt, emMPJetEt, runNo, dataset,
      "Jet iE_{T}", "Jets/JetEt.pdf",  1, 13, 0, 1500
      );

     create_plot(
      hwMPJetEtSat, emMPJetEtSat, runNo, dataset,
      "Jet iE_{T} Full", "Jets/JetEtSat.pdf"
      );

    // plot MP jet eta
    create_plot(
      hwMPJetEta, emMPJetEta, runNo, dataset,
      "Jet i#eta", "Jets/JetEta.pdf"
      );

    // plot MP jet phi
    create_plot(
      hwMPJetPhi, emMPJetPhi, runNo, dataset,
      "Jet i#phi", "Jets/JetPhi.pdf"
      );

    // plot demux jet Et
    create_plot(
      hwJetEt, emJetEt, runNo, dataset,
      "Jet iE_{T}", "DemuxJets/JetEt.pdf",  1, 13, 0, 2500
      );

    // plot demux jet eta
    create_plot(
      hwJetEta, emJetEta, runNo, dataset,
      "Jet i#eta", "DemuxJets/JetEta.pdf"
      );

    // plot demux jet phi
    create_plot(
      hwJetPhi, emJetPhi, runNo, dataset,
      "Jet i#phi", "DemuxJets/JetPhi.pdf"
      );
  }
// =========================== jets end =========================
// ======================== MP sums start ========================
  if (presentationMode) {
    // plot MP sum Et
    create_plot(
      hwMPSumEt, emMPSumEt, runNo, dataset,
      "Sum iE_{T}", "MPSums/MPSumEt.pdf", 2, 13, 0, 600
      );

    // plot MP sum Et with HF
    create_plot(
      hwMPSumEtHF, emMPSumEtHF, runNo, dataset,
      "Sum iE_{T}", "MPSums/MPSumEtHF.pdf", 2, 13, 0, 600
      );
  } else {
    // plot MP sum Et
    create_plot(
      hwMPSumEt, emMPSumEt, runNo, dataset,
      "Sum iE_{T}", "MPSums/MPSumEt.pdf", 1, 13, 0, 10000
      );

     // plot MP sum Et
    create_plot(
      hwMPSumEtSat, emMPSumEtSat, runNo, dataset,
      "Sum iE_{T}", "MPSums/MPSumEtSat.pdf", 1, 13, 0, 100000
      );
    

    // plot MP sum Et with HF
    create_plot(
      hwMPSumEtHF, emMPSumEtHF, runNo, dataset,
      "Sum iE_{T}", "MPSums/MPSumEtHF.pdf",  1, 13, 0, 10000
      );

    // plot MP sum Et with HF
    create_plot(
      hwMPSumEtHFSat, emMPSumEtHFSat, runNo, dataset,
      "Sum iE_{T}", "MPSums/MPSumEtHFSat.pdf", 1, 13, 0, 100000
      );

    

    
    // plot MP sum ETEm
    create_plot(
      hwMPSumEtEM, emMPSumEtEM, runNo, dataset,
      "Sum iE_{T}", "MPSums/MPSumEtEM.pdf", 1, 13, 0, 5000
      );
    
     
  

  // plot MP sum ETTEM
    create_plot(
      hwMPSumEtEMSat, emMPSumEtEMSat, runNo, dataset,
      "Sum iE_{T}", "MPSums/MPSumEtEMSat.pdf", 1, 13, 0, 100000
      );
    
 

  // plot MP sum Etx
  create_plot(
    hwMPSumEtx, emMPSumEtx, runNo, dataset,
    "Sum iE_{T,x}", "MPSums/MPSumEtx.pdf", 1, 13, -200000, 200000
    );

  create_plot(
    hwMPSumEtxSat, emMPSumEtxSat, runNo, dataset,
    "Sum iE_{T,x}", "MPSums/MPSumEtxSat.pdf", 1000, 13, -2200000000, 2200000000
    );

  // plot MP sum Etx with HF
  create_plot(
    hwMPSumEtxHF, emMPSumEtxHF, runNo, dataset,
    "Sum iE_{T,x}", "MPSums/MPSumEtxHF.pdf", 1, 13, -200000, 200000 
    );

    // plot MP sum Etx with HF sat
  create_plot(
    hwMPSumEtxHFSat, emMPSumEtxHFSat, runNo, dataset,
    "Sum iE_{T,x}", "MPSums/MPSumEtxHFSat.pdf", 1000, 13, -2200000000, 2200000000
    );


  // plot MP sum Ety
  create_plot(
    hwMPSumEty, emMPSumEty, runNo, dataset, 
    "Sum iE_{T,y}", "MPSums/MPSumEty.pdf", 1, 13, -200000, 200000
    );

   create_plot(
    hwMPSumEtySat, emMPSumEtySat, runNo, dataset,
    "Sum iE_{T,y}", "MPSums/MPSumEtySat.pdf", 1000, 13, -2200000000, 2200000000
    );

  // plot MP sum Ety with HF
  create_plot(
    hwMPSumEtyHF, emMPSumEtyHF, runNo, dataset,
    "Sum iE_{T,y}", "MPSums/MPSumEtyHF.pdf", 1, 13, -200000, 200000
    );

  // plot MP sum Ety with HF
  create_plot(
    hwMPSumEtyHFSat, emMPSumEtyHFSat, runNo, dataset,
    "Sum iE_{T,y}", "MPSums/MPSumEtyHFSat.pdf",  1000, 13, -2200000000, 2200000000 
    );
  }

  if (presentationMode) {
    // plot MP sum Ht
    create_plot(
      hwMPSumHt, emMPSumHt, runNo, dataset,
      "Sum iH_{T}", "MPSums/MPSumHt.pdf", 1, 13, 0, 1000
      );

    // plot MP sum Ht (with HF)
    create_plot(
      hwMPSumHtHF, emMPSumHtHF, runNo, dataset,
      "Sum iH_{T}", "MPSums/MPSumHtHF.pdf", 1, 13, 0, 1000
      );

    // plot MP sum Htx
    create_plot(
      hwMPSumHtx, emMPSumHtx, runNo, dataset,
      "Sum iH_{T,x}", "MPSums/MPSumHtx.pdf", 1, 13, -20000, 20000
      );

    // plot MP sum Htx (with HF)
    create_plot(
      hwMPSumHtxHF, emMPSumHtxHF, runNo, dataset,
      "Sum iH_{T,x}", "MPSums/MPSumHtxHF.pdf", 1, 13, -20000, 20000
      );

    // plot MP sum Hty
    create_plot(
      hwMPSumHty, emMPSumHty, runNo, dataset,
      "Sum iH_{T,y}", "MPSums/MPSumHty.pdf", 1, 13, -20000, 20000
      );

    // plot MP sum Hty (with HF)
    create_plot(
      hwMPSumHtyHF, emMPSumHtyHF, runNo, dataset,
      "Sum iH_{T,y}", "MPSums/MPSumHtyHF.pdf", 1, 13, -20000, 20000
      );

      // plot HI tower count
    create_plot(
      hwMPSumHITowerCount, emMPSumHITowerCount, runNo, dataset,
      "# Towers", "MPSums/MPSumHITowerCount.pdf", 1, 13, 0, 5904
      );
  } else {

    // plot MP sum Ht
    create_plot(
      hwMPSumHt, emMPSumHt, runNo, dataset,
      "Sum iH_{T}", "MPSums/MPSumHt.pdf", 1, 13, 0, 10000
      );

    // plot MP sum Ht (with HF)
    create_plot(
      hwMPSumHtHF, emMPSumHtHF, runNo, dataset,
      "Sum iH_{T}", "MPSums/MPSumHtHF.pdf", 1, 13, 0, 10000
      );


      // plot MP sum Ht
    create_plot(
      hwMPSumHtSat, emMPSumHtSat, runNo, dataset,
      "Sum iH_{T}", "MPSums/MPSumHtSat.pdf", 1, 13, 0, 100000
      );

    // plot MP sum Ht (with HF)
    create_plot(
      hwMPSumHtHFSat, emMPSumHtHFSat, runNo, dataset,
      "Sum iH_{T}", "MPSums/MPSumHtHFSat.pdf", 1, 13, 0, 100000
      );



    // plot MP sum Htx
    create_plot(
      hwMPSumHtx, emMPSumHtx, runNo, dataset,
      "Sum iH_{T,x}", "MPSums/MPSumHtx.pdf", 1, 13, -200000, 200000
      );

    // plot MP sum Htx (with HF)
    create_plot(
      hwMPSumHtxHF, emMPSumHtxHF, runNo, dataset,
      "Sum iH_{T,x}", "MPSums/MPSumHtxHF.pdf", 1, 13, -200000, 200000
      );


      // plot MP sum Htx (with HF) sat
    create_plot(
      hwMPSumHtxSat, emMPSumHtxSat, runNo, dataset,
      "Sum iH_{T,x}", "MPSums/MPSumHtxSat.pdf", 1000, 13, -2200000000, 2200000000
      );

    // plot MP sum Htx (with HF) sat
    create_plot(
      hwMPSumHtxHFSat, emMPSumHtxHFSat, runNo, dataset,
      "Sum iH_{T,x}", "MPSums/MPSumHtxHFSat.pdf", 1000, 13, -2200000000, 2200000000
      );

    // plot MP sum Hty
    create_plot(
      hwMPSumHty, emMPSumHty, runNo, dataset,
      "Sum iH_{T,y}", "MPSums/MPSumHty.pdf", 1, 13, -200000, 200000
      );

    // plot MP sum Hty (with HF)
    create_plot(
      hwMPSumHtyHF, emMPSumHtyHF, runNo, dataset,
      "Sum iH_{T,y}", "MPSums/MPSumHtyHF.pdf", 1, 13, -200000, 200000
      );

       // plot MP sum Hty sat
    create_plot(
      hwMPSumHtySat, emMPSumHtySat, runNo, dataset,
      "Sum iH_{T,y}", "MPSums/MPSumHtySat.pdf", 1000, 13, -2200000000, 2200000000
      );

    // plot MP sum Hty (with HF) sat
    create_plot(
      hwMPSumHtyHFSat, emMPSumHtyHFSat, runNo, dataset,
      "Sum iH_{T,y}", "MPSums/MPSumHtyHFSat.pdf", 1000, 13, -2200000000, 2200000000
      );
  

    // plot HI tower count
    create_plot(
      hwMPSumHITowerCount, emMPSumHITowerCount, runNo, dataset,
      "# Towers", "MPSums/MPSumHITowerCount.pdf"
      );

  }
// ========================= MP sums end ========================
// ======================== demux sums start ========================

  if (presentationMode) {
    // plot demux sum Et
    create_plot(
      hwSumEt,
      emSumEt,
      runNo, dataset, "iE_{T}", "DemuxSums/DemSumEt.pdf", 10, 13, 0, 800
      );

    // plot demux sum EtEM
    create_plot(
      hwSumEtEM,
      emSumEtEM,
      runNo, dataset, "iE_{T}", "DemuxSums/DemSumEtEM.pdf", 10, 13, 0, 800
      );

    /*
    // plot demux sum Et with HF
    create_plot(
    hwSumEtHF,
    emSumEtHF,
    runNo, dataset, "iE_{T}", "DemuxSums/DemSumEtHF.pdf", 10, 13, 0, 800
    );
    */

    // plot demux sum Met
    create_plot(
      hwSumMet,
      emSumMet,
      runNo, dataset, "iMET", "DemuxSums/DemSumMet.pdf", 5, 13, 0, 5000
      );

    // plot demux sum Met with HF
    create_plot(
      hwSumMetHF,
      emSumMetHF,
      runNo, dataset, "iMET", "DemuxSums/DemSumMetHF.pdf", 5, 13, 0, 5000
      );

    // plot demux sum Met phi
    create_plot(
      hwMetPhi,
      emMetPhi,
      runNo, dataset, "MET i#phi", "DemuxSums/DemMetPhi.pdf", 1, 13, 0, 143
      );

    // plot demux sum Met phi (with HF)
    create_plot(
      hwMetHFPhi,
      emMetHFPhi,
      runNo, dataset, "MET i#phi", "DemuxSums/DemMetHFPhi.pdf", 1, 13, 0, 143
      );

    // plot demux sum Ht
    create_plot(
      hwSumHt,
      emSumHt,
      runNo, dataset, "iH_{T}", "DemuxSums/DemSumHt.pdf", 10, 13, 0, 800
      );

      // plot demux hi tower count
    create_plot(
      hwHITowerCount,
      emHITowerCount,
      runNo, dataset, "# Towers", "DemuxSums/DemHITowCount.pdf", 1, 13, 0, 5904
      );

    // plot demux sum Mht
    create_plot(
      hwSumMht,
      emSumMht,
      runNo, dataset, "iMHT", "DemuxSums/DemSumMht.pdf", 5, 13, 0, 200
      );

    // plot demux sum Mht (with HF)
    create_plot(
      hwSumMhtHF,
      emSumMhtHF,
      runNo, dataset, "iMHT", "DemuxSums/DemSumMhtHF.pdf", 5, 13, 0, 200
      );

    // plot demux sum Mht phi
    create_plot(
      hwMhtPhi,
      emMhtPhi,
      runNo, dataset, "MHT i#phi", "DemuxSums/DemMhtPhi.pdf", 1, 13, 0, 143
      );

    // plot demux sum Mht phi (with HF)
    create_plot(
      hwMhtHFPhi,
      emMhtHFPhi,
      runNo, dataset, "MHT i#phi", "DemuxSums/DemMhtHFPhi.pdf", 1, 13, 0, 143
      );
  } else {
    // plot demux sum Et
    create_plot(
      hwSumEt,
      emSumEt,
      runNo, dataset, "iE_{T}", "DemuxSums/DemSumEt.pdf", 1, 13, 0, 5000
      );

    // plot demux sum EtEM
    create_plot(
      hwSumEtEM,
      emSumEtEM,
      runNo, dataset, "iE_{T}", "DemuxSums/DemSumEtEM.pdf", 1, 13, 0, 5000
      );

    /*
    // plot demux sum Et with HF
    create_plot(
    hwSumEtHF,
    emSumEtHF,
    runNo, dataset, "iE_{T}", "DemuxSums/DemSumEtHF.pdf"
    );
    */

    // plot demux sum Met with HF
    create_plot(
      hwSumMetHF,
      emSumMetHF,
      runNo, dataset, "iMET", "DemuxSums/DemSumMetHF.pdf", 1, 13, 0, 5000
      );

    // plot demux sum Met
    create_plot(
      hwSumMet,
      emSumMet,
      runNo, dataset, "iMET", "DemuxSums/DemSumMet.pdf", 1, 13, 0, 5000
      );

    // plot demux sum Met phi
    create_plot(
      hwMetPhi,
      emMetPhi,
      runNo, dataset, "MET i#phi", "DemuxSums/DemMetPhi.pdf", 1, 13, 0, 143
      );

    // plot demux sum Met phi (with HF)
    create_plot(
      hwMetHFPhi,
      emMetHFPhi,
      runNo, dataset, "MET i#phi", "DemuxSums/DemMetHFPhi.pdf", 1, 13, 0, 143
      );

    // plot demux sum Ht
    create_plot(
      hwSumHt,
      emSumHt,
      runNo, dataset, "iH_{T}", "DemuxSums/DemSumHt.pdf", 1, 13, 0, 5000
      );

      // plot demux hi tower count
    create_plot(
      hwHITowerCount,
      emHITowerCount,
      runNo, dataset, "# Towers", "DemuxSums/DemHITowCount.pdf", 1, 13, 0, 5904
      );

    // plot demux sum Mht
    create_plot(
      hwSumMht,
      emSumMht,
      runNo, dataset, "iMHT", "DemuxSums/DemSumMht.pdf", 1, 13, 1, 5000
      );

    // plot demux sum Mht (with HF)
    create_plot(
      hwSumMhtHF,
      emSumMhtHF,
      runNo, dataset, "iMHT", "DemuxSums/DemSumMhtHF.pdf", 1, 13, 1, 5000
      );

    // plot demux sum Mht phi
    create_plot(
      hwMhtPhi,
      emMhtPhi,
      runNo, dataset, "MHT i#phi", "DemuxSums/DemMhtPhi.pdf", 1, 13, 0, 143
      );
    
    // plot demux sum Mht phi (with HF)
    create_plot(
      hwMhtHFPhi,
      emMhtHFPhi,
      runNo, dataset, "MHT i#phi", "DemuxSums/DemMhtHFPhi.pdf", 1, 13, 0, 143
      );

    // plot demux sum Centrality
    create_plot(
      hwHICentrality,
      emHICentrality,
      runNo, dataset, "Centrality", "DemuxSums/DemSumCentrality.pdf", 1, 13, 0, 300
      );

    // plot demux sum Et Asym
    create_plot(
      hwEtAsymSum,
      emEtAsymSum,
      runNo, dataset, "iE_{T} Asym", "DemuxSums/DemSumEtAsym.pdf", 1, 13, 0, 300
      );

    // plot demux sum Ht Asym
    create_plot(
      hwHtAsymSum,
      emHtAsymSum,
      runNo, dataset, "iH_{T} Asym", "DemuxSums/DemSumHtAsym.pdf", 1, 13, 0, 300
      );

    // plot demux sum Et Asym HF
    create_plot(
      hwEtAsymSumHF,
      emEtAsymSumHF,
      runNo, dataset, "iE_{T} Asym", "DemuxSums/DemSumEtAsymHF.pdf", 1, 13, 0, 300
      );

    // plot demux sum Ht Asym HF
    create_plot(
      hwHtAsymSumHF,
      emHtAsymSumHF,
      runNo, dataset, "iH_{T} Asym", "DemuxSums/DemSumHtAsymHF.pdf", 1, 13, 0, 300
      );

    



  }
// ========================= demux sums end ========================

// ======================== e/gamma start ======================

// plot MP e/g Et
  create_plot(
    hwMPEgEt,
    emMPEgEt,
    runNo, dataset, "e/#gamma iE_{T}", "Egs/EgEt.pdf", 1, 13, 0, 1000
    );

// plot MP e/g eta
  create_plot(
    hwMPEgEta,
    emMPEgEta,
    runNo, dataset, "e/#gamma i#eta", "Egs/EgEta.pdf"
    );
  
// plot MP e/g phi
  create_plot(
    hwMPEgPhi,
    emMPEgPhi,
    runNo, dataset, "e/#gamma i#phi", "Egs/EgPhi.pdf"
    );


  // plot MP e/g iso
  create_plot(
    hwMPEgIso,
    emMPEgIso,
    runNo, dataset, "e/#gamma i#iso", "Egs/EgIso.pdf"
    );



// plot demux e/g Et
  create_plot(
    hwEgEt,
    emEgEt,
    runNo, dataset, "e/#gamma iE_{T}", "DemuxEgs/EgEt.pdf", 1, 13, 0, 1000
    );

// plot demux e/g eta
  create_plot(
    hwEgEta,
    emEgEta,
    runNo, dataset, "e/#gamma i#eta", "DemuxEgs/EgEta.pdf"
    );

// plot demux e/g phi
  create_plot(
    hwEgPhi,
    emEgPhi,
    runNo, dataset, "e/#gamma i#phi", "DemuxEgs/EgPhi.pdf"
    );

  // plot demux e/g iso
  create_plot(
    hwEgIso,
    emEgIso,
    runNo, dataset, "e/#gamma i#iso", "DemuxEgs/EgIso.pdf"
    );


// ======================== e/gamma end ========================

// ========================= tau start =========================

// plot MP tau Et
  create_plot(
    hwMPTauEt,
    emMPTauEt,
    runNo, dataset, "#tau iE_{T}", "Taus/TauEt.pdf", 1, 13, 0, 1500
    );

// plot MP tau eta
  create_plot(
    hwMPTauEta,
    emMPTauEta,
    runNo, dataset, "#tau i#eta", "Taus/TauEta.pdf"
    );

// plot MP tau phi
  create_plot(
    hwMPTauPhi,
    emMPTauPhi,
    runNo, dataset, "#tau i#phi", "Taus/TauPhi.pdf"
    );
  
// plot demux tau Et
  create_plot(
    hwTauEt,
    emTauEt,
    runNo, dataset, "#tau iE_{T}", "DemuxTaus/TauEt.pdf", 1, 13, 0, 1500
    );

// plot demux tau eta
  create_plot(
    hwTauEta,
    emTauEta,
    runNo, dataset, "#tau i#eta", "DemuxTaus/TauEta.pdf"
    );
  
// plot demux tau phi
  create_plot(
    hwTauPhi,
    emTauPhi,
    runNo, dataset, "#tau i#phi", "DemuxTaus/TauPhi.pdf"
    );
// ========================== tau end ==========================--------------

};
