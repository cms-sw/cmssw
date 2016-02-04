 void compHwEmu(){

   bool doSave=true;
   bool doRatio=true;

   bool doEgamma=false;
   bool doTau=false;
   bool doJets=true;
   bool doSums=true;
   bool doSorts=false;

   bool doDemux=true;

   int nbins=0;

   TLatex n1;
   n1.SetNDC();
   n1.SetTextFont(42);
   n1.SetTextSize(0.04);
   
   TLatex n2;
   n2.SetNDC();
   n2.SetLineWidth(2);
   n2.SetTextFont(61);
   n2.SetTextSize(0.05);
   
   TLatex n3;
   n3.SetNDC();
   n3.SetTextFont(52);
   n3.SetTextSize(0.04);

   TLatex n4;
   n4.SetNDC();
   n4.SetTextFont(52);
   n4.SetTextSize(0.04);
 
 

   TFile* inFileHw = new TFile("l1tCalo_2016_simHistos.root");
   TFile* inFileEm = new TFile("l1tCalo_2016_simHistos.root");
  

  //TH2D* towEtaPhi = (TH2D*)inFile->Get("rawPlots/tower/etaphi");

  //TH2D* rawEtaPhi = (TH2D*)inFile->Get("rawPlots/jet/etaphi");
  //TH2D* emuEtaPhi = (TH2D*)inFile->Get("simPlots/jet/etaphi");

  // Jets
  
  TH1D* hwMPJetEt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpjet/et");
  TH1D* emMPJetEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpjet/et");
  TH1D* hwJetEt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/jet/et");
  TH1D* emJetEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/jet/et");

  TH1D* hwMPJetEta = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpjet/eta");
  TH1D* emMPJetEta = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpjet/eta");
  TH1D* hwJetEta = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/jet/eta");
  TH1D* emJetEta = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/jet/eta");

  TH1D* hwMPJetPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpjet/phi");
  TH1D* emMPJetPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpjet/phi");
  TH1D* hwJetPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/jet/phi");
  TH1D* emJetPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/jet/phi");

  // MP sums
  
  TH1D* hwMPSumEt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsumet/et");
  TH1D* emMPSumEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsumet/et");

  TH1D* hwMPSumEtx = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsummetx/et");
  TH1D* emMPSumEtx = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummetx/et");

  TH1D* hwMPSumEty = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsummety/et");
  TH1D* emMPSumEty = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummety/et");

  TH1D* hwMPSumHt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsumht/et");
  TH1D* emMPSumHt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsumht/et");

  TH1D* hwMPSumHtx = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsummhtx/et");
  TH1D* emMPSumHtx = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummhtx/et");

  TH1D* hwMPSumHty = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/mpsummhty/et");
  TH1D* emMPSumHty = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummhty/et");
  
  // Demux sums
  
  TH1D* hwSumEt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sumet/et");
  TH1D* emSumEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumet/et");

  TH1D* hwSumMet = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/summet/et");
  TH1D* emSumMet = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summet/et");

  TH1D* hwSumHt = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/sumht/et");
  TH1D* emSumHt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumht/et");

  TH1D* hwSumMht = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/summht/et");
  TH1D* emSumMht = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summht/et");
  
  // Sum phi's
  
  TH1D* hwMetPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/summet/phi");
  TH1D* emMetPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summet/phi");

  TH1D* hwMhtPhi = (TH1D*)inFileHw->Get("l1tCaloStage2HwHistos/summht/phi");
  TH1D* emMhtPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summht/phi");
  

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





  ////////////////////////////////////////////////////////////////////////////////////

  TLine* unity = new TLine(0.1,0.525,0.9,0.525);
  unity->SetLineColor(kBlue);

  TLegend* leg = new TLegend(0.6,0.75,0.85,0.85);
  leg->SetFillColor(0);
  leg->AddEntry(hwEgEt,"Upgrade hardware", "p");
  leg->AddEntry(emEgEt,"Upgrade emulator", "l");

  /*
    TLegend* leg = new TLegend(0.6,0.7,0.8,0.9);
    leg->SetFillColor(0);
    leg->AddEntry(towEtaPhi,"Towers");

    c1->cd();
    towEtaPhi->GetXaxis()->SetTitle("iEta");
    towEtaPhi->GetYaxis()->SetTitle("iPhi");
    towEtaPhi->GetZaxis()->SetTitle("E_{T}");
    towEtaPhi->SetAxisRange(0.,40.);
    towEtaPhi->Draw("text");
    leg->Draw();
    c1->Print("compHwEmu.pdf(","pdf");
  */




// //================ egamma ====================

if(doEgamma){

  TCanvas* cEgEt = new TCanvas("cEgEt","EgEt");
  TPad* pEgEt = new TPad("pEgEt","pEgEt",0,0.0,1,1); 

  if(doRatio) pEgEt = new TPad("pEgEt","pEgEt",0,0.3,1,1); 
 
  TPad* pEgEtRatio = new TPad("pEgEtratio","pEgEtratio",0,0,1,0.3);
  
  TPad* pInvEgEtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
  pInvEgEtRatio->SetFillStyle(0);

  leg = new TLegend(0.6,0.75,0.85,0.85);
  leg->SetFillColor(0);
  leg->AddEntry(hwMPEgEt,"Upgrade hardware", "p");//"l");
  //leg->AddEntry(hwEgEt,"Upgrade hardware", "p");
  leg->AddEntry(emMPEgEt,"Upgrade emulator", "l");
  //leg->AddEntry(emEgEt,"Upgrade emulator", "p");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  
  //hwMPEgEt->Rebin(2);
  //emMPEgEt->Rebin(2);

  hwMPEgEt->SetStats(0);
  //hwMPEgEt->SetLineColor(kBlue);
  hwMPEgEt->SetMarkerStyle(21);
  hwMPEgEt->SetMarkerColor(1);
  hwMPEgEt->SetMarkerSize(0.4);
  emMPEgEt->SetLineColor(kRed);
  hwMPEgEt->GetXaxis()->SetRange(0,100);
  hwMPEgEt->GetXaxis()->SetTitle("Level-1 Trigger EGamma iE_{T}");
  hwMPEgEt->GetYaxis()->SetTitle("# EGammas");
  hwMPEgEt->GetYaxis()->SetTitleSize(0.05);
  hwMPEgEt->GetYaxis()->SetTitleOffset(0.66);
  hwMPEgEt->GetXaxis()->SetTitleSize(0.04);
  hwMPEgEt->GetXaxis()->SetTitleOffset(1.0);
  pEgEt->SetBottomMargin(0.12);
  pEgEt->Draw();
  pEgEt->cd();

  TH1D* EgEtRatio = (TH1D*)hwMPEgEt->DrawCopy("p");
  EgEtRatio->SetMinimum(0);
  emMPEgEt->Draw("same");//"");
  leg->Draw();
  n2.DrawLatex(0.7, 0.65, "CMS");
  n3.DrawLatex(0.7, 0.6, "Run 260627    #sqrt{s} = 13 TeV");
  n3.DrawLatex(0.7, 0.55, "Preliminary");
  n4.DrawLatex(0.7, 0.45, "Single Muon stream");

    

  if(doRatio){

  cEgEt->cd();
  pEgEtRatio->SetTopMargin(0.05);
  pEgEtRatio->Draw();
  pEgEtRatio->cd();
  hwMPEgEt->Divide(emMPEgEt);
  hwMPEgEt->GetYaxis()->SetTitle("Ratio HW/EM");
  hwMPEgEt->GetYaxis()->SetTitleSize(0.09);
  hwMPEgEt->GetYaxis()->SetLabelSize(0.05);
  hwMPEgEt->GetXaxis()->SetLabelSize(0.07);
  hwMPEgEt->GetXaxis()->SetTitleSize(0.0);
  hwMPEgEt->GetYaxis()->SetTitleOffset(0.35);
  hwMPEgEt->SetMinimum(0.8);
  hwMPEgEt->SetMaximum(1.2);
  hwMPEgEt->Draw("p");
  cEgEt->cd();
  pInvEgEtRatio->Draw();
  pInvEgEtRatio->cd();
  unity->Draw();

  }
  
  if(doSave) cEgEt->SaveAs("compHwEmu/Egs/EgEt.pdf");
  

  if(doDemux){


 TCanvas* cDEgEt = new TCanvas("cDEgEt","DEgEt");
 TPad* pDEgEt = new TPad("pEgEt","pEgEt",0,0.0,1,1); 
 if(doRatio) pDEgEt = new TPad("pEgEt","pEgEt",0,0.3,1,1); 
 TPad* pDEgEtRatio = new TPad("pEgEtratio","pEgEtratio",0,0,1,0.3);
 
 TPad* pInvDEgEtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDEgEtRatio->SetFillStyle(0);
  

 //hwEgEt->Rebin(2);
 //emEgEt->Rebin(2);

 hwEgEt->SetStats(0);
 hwEgEt->SetMarkerStyle(21);
 hwEgEt->SetMarkerColor(1);
 hwEgEt->SetMarkerSize(0.4);
 emEgEt->SetLineColor(kRed);
 hwEgEt->GetXaxis()->SetRange(0,30);
 hwEgEt->GetXaxis()->SetTitle("Level-1 Trigger EGamma iE_{T}");
 hwEgEt->GetYaxis()->SetTitle("# EGamma");
 hwEgEt->GetYaxis()->SetTitleSize(0.05);
 hwEgEt->GetYaxis()->SetTitleOffset(0.77);
 hwEgEt->GetXaxis()->SetTitleSize(0.04);
 hwEgEt->GetXaxis()->SetTitleOffset(1.0);
 pDEgEt->SetBottomMargin(0.12);
 pDEgEt->Draw();
 pDEgEt->cd();
 
 TH1D* DEgEtRatio = (TH1D*)hwEgEt->DrawCopy("p");
 DEgEtRatio->SetMinimum(0);
 emEgEt->Draw("same");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");


 if(doRatio){

 cDEgEt->cd();
 pDEgEtRatio->SetTopMargin(0.05);
 pDEgEtRatio->Draw();
 pDEgEtRatio->cd();
 hwEgEt->Divide(emEgEt);
 hwEgEt->GetYaxis()->SetTitle("Ratio HW/EM");
 hwEgEt->GetYaxis()->SetTitleSize(0.09);
 hwEgEt->GetYaxis()->SetLabelSize(0.05);
 hwEgEt->GetXaxis()->SetLabelSize(0.07);
 hwEgEt->GetXaxis()->SetTitleSize(0.0);
 hwEgEt->GetYaxis()->SetTitleOffset(0.35);
 hwEgEt->SetMinimum(0.8);
 hwEgEt->SetMaximum(1.2);
 hwEgEt->Draw("p");
 cDEgEt->cd();
 pInvDEgEtRatio->Draw();
 pInvDEgEtRatio->cd();
 unity->Draw();
 pDEgEtRatio->Update();

 }

  if(doSave) cDEgEt->SaveAs("compHwEmu/DemuxEgs/EgEt.pdf");
 
  }

 TCanvas* cEgEta = new TCanvas("cEgEta","EgEta");

 TPad* pEgEta = new TPad("pEgEt","pEgEt",0,0.0,1,1); 
 if(doRatio) pEgEta = new TPad("pEgEt","pEgEt",0,0.3,1,1); 
 
 TPad* pEgEtaRatio = new TPad("pEgEtratio","pEgEtratio",0,0,1,0.3);
  
 TPad* pInvEgEtaRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvEgEtaRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.75,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPEgEta,"Upgrade hardware", "p");//"l");
 //leg->AddEntry(hwEgEta,"Upgrade hardware", "p");
 leg->AddEntry(emMPEgEta,"Upgrade emulator", "l");
 //leg->AddEntry(emEgEta,"Upgrade emulator", "p");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);

 hwMPEgEta->SetStats(0);
 //hwMPEgEta->SetLineColor(kBlue);
 hwMPEgEta->SetMarkerStyle(21);
 hwMPEgEta->SetMarkerColor(1);
 hwMPEgEta->SetMarkerSize(0.4);
 emMPEgEta->SetLineColor(kRed);
 emMPEgEta->SetMarkerStyle(20);
 emMPEgEta->SetMarkerColor(kRed);
 emMPEgEta->SetMarkerSize(0.4);
 hwMPEgEta->GetXaxis()->SetRange(10,74);
 hwMPEgEta->GetXaxis()->SetTitle("EGamma i#eta");
 hwMPEgEta->GetYaxis()->SetTitle("# EGammas");
 hwMPEgEta->GetYaxis()->SetTitleSize(0.05);
 hwMPEgEta->GetYaxis()->SetTitleOffset(0.77);
 hwMPEgEta->GetXaxis()->SetTitleSize(0.04);
 hwMPEgEta->GetXaxis()->SetTitleOffset(1.0);
 pEgEta->SetBottomMargin(0.12);
 pEgEta->Draw();
 pEgEta->cd();

 TH1D* EgEtaRatio = (TH1D*)hwMPEgEta->DrawCopy("p");
 EgEtaRatio->SetMinimum(0);
 emMPEgEta->Draw("same");//"");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");


 if(doRatio){

 cEgEta->cd();
 pEgEtaRatio->SetTopMargin(0.05);
 pEgEtaRatio->Draw();
 pEgEtaRatio->cd();
 hwMPEgEta->Divide(emMPEgEta);
 hwMPEgEta->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPEgEta->GetYaxis()->SetTitleSize(0.09);
 hwMPEgEta->GetYaxis()->SetLabelSize(0.05);
 hwMPEgEta->GetXaxis()->SetLabelSize(0.07);
 hwMPEgEta->GetXaxis()->SetTitleSize(0.0);
 hwMPEgEta->GetYaxis()->SetTitleOffset(0.35);
 hwMPEgEta->SetMinimum(0.8);
 hwMPEgEta->SetMaximum(1.2);
 hwMPEgEta->Draw("p");
 cEgEta->cd();
 pInvEgEtaRatio->Draw();
 pInvEgEtaRatio->cd();
 unity->Draw();
 //c1->Print("compHwEmu.pdf","pdf");

 }

 if(doSave) cEgEta->SaveAs("compHwEmu/Egs/EgEta.pdf");
 

 if(doDemux){
 
 TCanvas* cDEgEta = new TCanvas("cDEgEta","DEgEta");
 
 TPad* pDEgEta = new TPad("pEgEta","pEgEta",0,0.0,1,1); 
 if(doRatio) pDEgEta = new TPad("pEgEta","pEgEta",0,0.3,1,1); 

 TPad* pDEgEtaRatio = new TPad("pEgEtaratio","pEgEtaratio",0,0,1,0.3);
 
 TPad* pInvDEgEtaRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDEgEtaRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.75,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwEgEta,"Upgrade hardware", "p");
 leg->AddEntry(emEgEta,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwEgEta->SetStats(0);
 hwEgEta->SetMarkerStyle(21);
 hwEgEta->SetMarkerColor(1);
 hwEgEta->SetMarkerSize(0.4);
 emEgEta->SetLineColor(kRed);
 hwEgEta->GetXaxis()->SetTitle("EGamma i#eta");
 hwEgEta->GetXaxis()->SetRange(82,146);
 hwEgEta->GetYaxis()->SetTitle("# EGamma");
 hwEgEta->GetYaxis()->SetTitleSize(0.05);
 hwEgEta->GetYaxis()->SetTitleOffset(0.77);
 hwEgEta->GetXaxis()->SetTitleSize(0.04);
 hwEgEta->GetXaxis()->SetTitleOffset(1.0);
 pDEgEta->SetBottomMargin(0.12);
 pDEgEta->Draw();
 pDEgEta->cd();
 
 TH1D* DEgEtaRatio = (TH1D*)hwEgEta->DrawCopy("p");
 DEgEtaRatio->SetMinimum(0);
 emEgEta->Draw("same");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");

 if(doRatio){

 cDEgEta->cd();
 pDEgEtaRatio->SetTopMargin(0.05);
 pDEgEtaRatio->Draw();
 pDEgEtaRatio->cd();
 hwEgEta->Divide(emEgEta);
 hwEgEta->GetYaxis()->SetTitle("Ratio HW/EM");
 hwEgEta->GetYaxis()->SetTitleSize(0.09);
 hwEgEta->GetYaxis()->SetLabelSize(0.05);
 hwEgEta->GetXaxis()->SetLabelSize(0.07);
 hwEgEta->GetXaxis()->SetTitleSize(0.0);
 hwEgEta->GetYaxis()->SetTitleOffset(0.35);
 hwEgEta->SetMinimum(0.8);
 hwEgEta->SetMaximum(1.2);
 hwEgEta->Draw("p");
 cDEgEta->cd();
 pInvDEgEtaRatio->Draw();
 pInvDEgEtaRatio->cd();
 unity->Draw();
 pDEgEtaRatio->Update();

 }

 if(doSave) cDEgEta->SaveAs("compHwEmu/DemuxEgs/EgEta.pdf");
 
 }


 TCanvas* cEgPhi = new TCanvas("cEgPhi","EgPhi");

 TPad* pEgPhi = new TPad("pEgPhi","pEgPhi",0,0.0,1,1); 
 if(doRatio)  pEgPhi = new TPad("pEgPhi","pEgPhi",0,0.3,1,1); 
 TPad* pEgPhiRatio = new TPad("pEgPhiratio","pEgPhiratio",0,0,1,0.3);
 
 TPad* pInvEgPhiRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvEgPhiRatio->SetFillStyle(0);


 leg = new TLegend(0.4,0.75,0.65,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPEgPhi,"Upgrade hardware", "p");//"l");
 //leg->AddEntry(hwEgPhi,"Upgrade hardware", "p");
 leg->AddEntry(emMPEgPhi,"Upgrade emulator", "l");
 //leg->AddEntry(emEgPhi,"Upgrade emulator", "p");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwMPEgPhi->SetStats(0);
 //hwMPEgPhi->SetLineColor(kBlue);
 hwMPEgPhi->SetMarkerStyle(21);
 hwMPEgPhi->SetMarkerColor(1);
 hwMPEgPhi->SetMarkerSize(0.4);
 emMPEgPhi->SetLineColor(kRed);
 emMPEgPhi->SetMarkerStyle(20);
 emMPEgPhi->SetMarkerColor(kRed);
 emMPEgPhi->SetMarkerSize(0.4);
 emMPEgPhi->GetXaxis()->SetRange(0,100);
 emMPEgPhi->GetYaxis()->SetRange(0,70);
 hwMPEgPhi->GetYaxis()->SetRange(0,70);
 hwMPEgPhi->GetXaxis()->SetTitle("EGamma i#phi");
 hwMPEgPhi->GetYaxis()->SetTitle("# EGammas");
 hwMPEgPhi->GetYaxis()->SetTitleSize(0.05);
 hwMPEgPhi->GetYaxis()->SetTitleOffset(0.77);
 hwMPEgPhi->GetXaxis()->SetTitleSize(0.04);
 hwMPEgPhi->GetXaxis()->SetTitleOffset(1.0);
 pEgPhi->SetBottomMargin(0.12);
 pEgPhi->Draw();
 pEgPhi->cd();

 TH1D* EgPhiRatio = (TH1D*)hwMPEgPhi->DrawCopy("p");
 EgPhiRatio->SetMinimum(0);
 emMPEgPhi->Draw("same");//"");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");
 

 if(doRatio){

 cEgPhi->cd();
 pEgPhiRatio->SetTopMargin(0.05);
 pEgPhiRatio->Draw();
 pEgPhiRatio->cd();
 hwMPEgPhi->Divide(emMPEgPhi);
 hwMPEgPhi->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPEgPhi->GetYaxis()->SetTitleSize(0.09);
 hwMPEgPhi->GetYaxis()->SetLabelSize(0.05);
 hwMPEgPhi->GetXaxis()->SetLabelSize(0.07);
 hwMPEgPhi->GetXaxis()->SetTitleSize(0.0);
 hwMPEgPhi->GetYaxis()->SetTitleOffset(0.35);
 hwMPEgPhi->SetMinimum(0.8);
 hwMPEgPhi->SetMaximum(1.2);
 hwMPEgPhi->Draw("p");
 cEgPhi->cd();
 pInvEgPhiRatio->Draw();
 pInvEgPhiRatio->cd();
 unity->Draw();

 }

 if(doSave) cEgPhi->SaveAs("compHwEmu/Egs/EgPhi.pdf");
 

 if(doDemux){

 TCanvas* cDEgPhi = new TCanvas("cDEgPhi","DEgPhi");
 
 TPad* pDEgPhi = new TPad("pEgPhi","pEgPhi",0,0.0,1,1); 
 if(doRatio) pDEgPhi = new TPad("pEgPhi","pEgPhi",0,0.3,1,1); 
   
 TPad* pDEgPhiRatio = new TPad("pEgPhiratio","pEgPhiratio",0,0,1,0.3);
  
 TPad* pInvDEgPhiRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDEgPhiRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.75,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwEgPhi,"Upgrade hardware", "p");
 leg->AddEntry(emEgPhi,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwEgPhi->SetStats(0);
 hwEgPhi->SetMarkerStyle(21);
 hwEgPhi->SetMarkerColor(1);
 hwEgPhi->SetMarkerSize(0.4);
 emEgPhi->SetLineColor(kRed);
 emEgPhi->GetYaxis()->SetRange(0,70);
 hwEgPhi->GetXaxis()->SetTitle("EGamma i#phi");
 hwEgPhi->GetXaxis()->SetRange(0,73);
 hwEgPhi->GetYaxis()->SetRange(0,70);
 hwEgPhi->GetYaxis()->SetTitle("# EGammas");
 hwEgPhi->GetYaxis()->SetTitleSize(0.05);
 hwEgPhi->GetYaxis()->SetTitleOffset(0.77);
 hwEgPhi->GetXaxis()->SetTitleSize(0.04);
 hwEgPhi->GetXaxis()->SetTitleOffset(1.0);
 pDEgPhi->SetBottomMargin(0.12);
 pDEgPhi->Draw();
 pDEgPhi->cd();

 TH1D* DEgPhiRatio = (TH1D*)hwEgPhi->DrawCopy("p");
 DEgPhiRatio->SetMinimum(0);
 emEgPhi->Draw("same");//"");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");



 if(doRatio){

 cDEgPhi->cd();
 pDEgPhiRatio->SetTopMargin(0.05);
 pDEgPhiRatio->Draw();
 pDEgPhiRatio->cd();
 hwEgPhi->Divide(emEgPhi);
 hwEgPhi->GetYaxis()->SetTitle("Ratio HW/EM");
 hwEgPhi->GetYaxis()->SetTitleSize(0.09);
 hwEgPhi->GetYaxis()->SetLabelSize(0.05);
 hwEgPhi->GetXaxis()->SetLabelSize(0.07);
 hwEgPhi->GetXaxis()->SetTitleSize(0.0);
 hwEgPhi->GetYaxis()->SetTitleOffset(0.35);
 hwEgPhi->SetMinimum(0.8);
 hwEgPhi->SetMaximum(1.2);
 hwEgPhi->Draw("p");
 cDEgPhi->cd();
 pInvDEgPhiRatio->Draw();
 pInvDEgPhiRatio->cd();
 unity->Draw();

 }
 
 if(doSave) cDEgPhi->SaveAs("compHwEmu/DemuxEgs/EgPhi.pdf");
 
 }


 }


if(doTau){

   TCanvas* cTauEt = new TCanvas("cTauEt","TauEt");

  TPad* pTauEt = new TPad("pTauEt","pTauEt",0,0.3,1,1); 
  TPad* pTauEtRatio = new TPad("pTauEtratio","pTauEtratio",0,0,1,0.3);
  
  TPad* pInvTauEtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
  pInvTauEtRatio->SetFillStyle(0);

  leg = new TLegend(0.6,0.65,0.85,0.85);
  leg->SetFillColor(0);
  leg->AddEntry(hwMPTauEt,"Hardware MP", "p");//"l");
  leg->AddEntry(emMPTauEt,"Emulator MP", "l");
    
  //hwMPTauEt->Rebin(2);
  //emMPTauEt->Rebin(2);

  hwMPTauEt->SetStats(0);
  hwMPTauEt->SetLineColor(kBlue);
  hwMPTauEt->SetMarkerStyle(21);
  hwMPTauEt->SetMarkerColor(1);
  hwMPTauEt->SetMarkerSize(0.4);
  emMPTauEt->SetLineColor(kRed);
  hwMPTauEt->GetXaxis()->SetRange(0,30);
  hwMPTauEt->GetXaxis()->SetTitle("Tau iET");
  hwMPTauEt->GetYaxis()->SetTitle("# Taus");
  hwMPTauEt->GetYaxis()->SetTitleSize(0.05);
  hwMPTauEt->GetYaxis()->SetTitleOffset(0.66);
  hwMPTauEt->GetXaxis()->SetTitleSize(0.04);
  hwMPTauEt->GetXaxis()->SetTitleOffset(0.9);
  pTauEt->SetBottomMargin(0.08);
  pTauEt->Draw();
  pTauEt->cd();

  TH1D* TauEtRatio = (TH1D*)hwMPTauEt->DrawCopy("p");
  TauEtRatio->SetMinimum(0);
  emMPTauEt->Draw("same");//"");
  leg->Draw();
  cTauEt->cd();
  pTauEtRatio->SetTopMargin(0.05);
  pTauEtRatio->Draw();
  pTauEtRatio->cd();
  hwMPTauEt->Divide(emMPTauEt);
  hwMPTauEt->GetYaxis()->SetTitle("Ratio HW/EM");
  hwMPTauEt->GetYaxis()->SetTitleSize(0.09);
  hwMPTauEt->GetYaxis()->SetLabelSize(0.05);
  hwMPTauEt->GetXaxis()->SetLabelSize(0.07);
  hwMPTauEt->GetXaxis()->SetTitleSize(0.0);
  hwMPTauEt->GetYaxis()->SetTitleOffset(0.35);
  hwMPTauEt->SetMinimum(0.8);
  hwMPTauEt->SetMaximum(1.2);
  hwMPTauEt->Draw("p");
  cTauEt->cd();
  pInvTauEtRatio->Draw();
  pInvTauEtRatio->cd();
  unity->Draw();


 if(doSave) cTauEt->SaveAs("compHwEmu/Taus/TauEt.pdf");


 if(doDemux){

TCanvas* cDTauEt = new TCanvas("cDTauEt","DTauEt");

TPad* pDTauEt = new TPad("pTauEt","pTauEt",0,0.3,1,1); 
TPad* pDTauEtRatio = new TPad("pTauEtratio","pTauEtratio",0,0,1,0.3);

TPad* pInvDTauEtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
pInvDTauEtRatio->SetFillStyle(0);


//hwTauEt->Rebin(2);
//emTauEt->Rebin(2);

hwTauEt->SetStats(0);
hwTauEt->SetMarkerStyle(21);
hwTauEt->SetMarkerColor(1);
hwTauEt->SetMarkerSize(0.4);
emTauEt->SetLineColor(kRed);
hwTauEt->GetXaxis()->SetRange(0,30);
hwTauEt->GetXaxis()->SetTitle("Tau iET");
hwTauEt->GetYaxis()->SetTitle("# Tau");
hwTauEt->GetYaxis()->SetTitleSize(0.05);
hwTauEt->GetYaxis()->SetTitleOffset(0.66);
hwTauEt->GetXaxis()->SetTitleSize(0.04);
hwTauEt->GetXaxis()->SetTitleOffset(0.9);
pDTauEt->SetBottomMargin(0.08);
pDTauEt->Draw();
pDTauEt->cd();

TH1D* DTauEtRatio = (TH1D*)hwTauEt->DrawCopy("p");
DTauEtRatio->SetMinimum(0);
emTauEt->Draw("same");
leg->Draw();
cDTauEt->cd();
pDTauEtRatio->SetTopMargin(0.05);
pDTauEtRatio->Draw();
pDTauEtRatio->cd();
hwTauEt->Divide(emTauEt);
hwTauEt->GetYaxis()->SetTitle("Ratio HW/EM");
hwTauEt->GetYaxis()->SetTitleSize(0.09);
hwTauEt->GetYaxis()->SetLabelSize(0.05);
hwTauEt->GetXaxis()->SetLabelSize(0.07);
hwTauEt->GetXaxis()->SetTitleSize(0.0);
hwTauEt->GetYaxis()->SetTitleOffset(0.35);
hwTauEt->SetMinimum(0.8);
hwTauEt->SetMaximum(1.2);
hwTauEt->Draw("p");
cDTauEt->cd();
pInvDTauEtRatio->Draw();
pInvDTauEtRatio->cd();
unity->Draw();
pDTauEtRatio->Update();


 if(doSave)  cDTauEt->SaveAs("compHwEmu/DemuxTaus/TauEt.pdf");


 }

 TCanvas* cTauEta = new TCanvas("cTauEta","TauEta");

 TPad* pTauEta = new TPad("pTauEt","pTauEt",0,0.3,1,1); 
 TPad* pTauEtaRatio = new TPad("pTauEtratio","pTauEtratio",0,0,1,0.3);
  
 TPad* pInvTauEtaRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvTauEtaRatio->SetFillStyle(0);

 leg = new TLegend(0.4,0.65,0.65,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPTauEta,"Hardware MP", "p");//"l");
 leg->AddEntry(emMPTauEta,"Emulator MP", "l");
 

 hwMPTauEta->SetStats(0);
 hwMPTauEta->SetLineColor(kBlue);
 hwMPTauEta->SetMarkerStyle(21);
 hwMPTauEta->SetMarkerColor(1);
 hwMPTauEta->SetMarkerSize(0.4);
 emMPTauEta->SetLineColor(kRed);
 emMPTauEta->SetMarkerStyle(20);
 emMPTauEta->SetMarkerColor(kRed);
 emMPTauEta->SetMarkerSize(0.4);
 hwMPTauEta->GetXaxis()->SetRange(10,74);
 hwMPTauEta->GetXaxis()->SetTitle("Tau i#eta");
 hwMPTauEta->GetYaxis()->SetTitle("# Taus");
 hwMPTauEta->GetYaxis()->SetTitleSize(0.05);
 hwMPTauEta->GetYaxis()->SetTitleOffset(0.66);
 hwMPTauEta->GetXaxis()->SetTitleSize(0.04);
 hwMPTauEta->GetXaxis()->SetTitleOffset(0.9);
 pTauEta->SetBottomMargin(0.08);
 pTauEta->Draw();
 pTauEta->cd();

 TH1D* TauEtaRatio = (TH1D*)hwMPTauEta->DrawCopy("p");
 TauEtaRatio->SetMinimum(0);
 emMPTauEta->Draw("same");//"");
 leg->Draw();
 cTauEta->cd();
 pTauEtaRatio->SetTopMargin(0.05);
 pTauEtaRatio->Draw();
 pTauEtaRatio->cd();
 hwMPTauEta->Divide(emMPTauEta);
 hwMPTauEta->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPTauEta->GetYaxis()->SetTitleSize(0.09);
 hwMPTauEta->GetYaxis()->SetLabelSize(0.05);
 hwMPTauEta->GetXaxis()->SetLabelSize(0.07);
 hwMPTauEta->GetXaxis()->SetTitleSize(0.0);
 hwMPTauEta->GetYaxis()->SetTitleOffset(0.35);
 hwMPTauEta->SetMinimum(0.8);
 hwMPTauEta->SetMaximum(1.2);
 hwMPTauEta->Draw("p");
 cTauEta->cd();
 pInvTauEtaRatio->Draw();
 pInvTauEtaRatio->cd();
 unity->Draw();
 

 if(doSave)  cTauEta->SaveAs("compHwEmu/Taus/TauEta.pdf");


 if(doDemux){
 
 TCanvas* cDTauEta = new TCanvas("cDTauEta","DTauEta");
 TPad* pDTauEta = new TPad("pTauEta","pTauEta",0,0.3,1,1); 
 TPad* pDTauEtaRatio = new TPad("pTauEtaratio","pTauEtaratio",0,0,1,0.3);
 
 TPad* pInvDTauEtaRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDTauEtaRatio->SetFillStyle(0);

 leg = new TLegend(0.4,0.65,0.65,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwTauEta,"Hardware Demux", "p");
 leg->AddEntry(emTauEta,"Emulator Demux", "l");

 hwTauEta->SetStats(0);
 hwTauEta->SetMarkerStyle(21);
 hwTauEta->SetMarkerColor(1);
 hwTauEta->SetMarkerSize(0.4);
 emTauEta->SetLineColor(kRed);
 hwTauEta->GetXaxis()->SetTitle("Tau i#eta");
 hwTauEta->GetXaxis()->SetRange(82,146);
 hwTauEta->GetYaxis()->SetTitle("# Tau");
 hwTauEta->GetYaxis()->SetTitleSize(0.05);
 hwTauEta->GetYaxis()->SetTitleOffset(0.66);
 hwTauEta->GetXaxis()->SetTitleSize(0.04);
 hwTauEta->GetXaxis()->SetTitleOffset(0.9);
 pDTauEta->SetBottomMargin(0.08);
 pDTauEta->Draw();
 pDTauEta->cd();
 
 TH1D* DTauEtaRatio = (TH1D*)hwTauEta->DrawCopy("p");
 DTauEtaRatio->SetMinimum(0);
 emTauEta->Draw("same");
 leg->Draw();
 cDTauEta->cd();
 pDTauEtaRatio->SetTopMargin(0.05);
 pDTauEtaRatio->Draw();
 pDTauEtaRatio->cd();
 hwTauEta->Divide(emTauEta);
 hwTauEta->GetYaxis()->SetTitle("Ratio HW/EM");
 hwTauEta->GetYaxis()->SetTitleSize(0.09);
 hwTauEta->GetYaxis()->SetLabelSize(0.05);
 hwTauEta->GetXaxis()->SetLabelSize(0.07);
 hwTauEta->GetXaxis()->SetTitleSize(0.0);
 hwTauEta->GetYaxis()->SetTitleOffset(0.35);
 hwTauEta->SetMinimum(0.8);
 hwTauEta->SetMaximum(1.2);
 hwTauEta->Draw("p");
 cDTauEta->cd();
 pInvDTauEtaRatio->Draw();
 pInvDTauEtaRatio->cd();
 unity->Draw();
 pDTauEtaRatio->Update();
 

 if(doSave)  cDTauEta->SaveAs("compHwEmu/DemuxTaus/TauEta.pdf");

 }

 TCanvas* cTauPhi = new TCanvas("cTauPhi","TauPhi");

 TPad* pTauPhi = new TPad("pTauPhi","pTauPhi",0,0.3,1,1); 
 TPad* pTauPhiRatio = new TPad("pTauPhiratio","pTauPhiratio",0,0,1,0.3);
 
 TPad* pInvTauPhiRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvTauPhiRatio->SetFillStyle(0);


 leg = new TLegend(0.75,0.75,0.9,0.9);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPTauPhi,"Hardware MP", "p");//"l");
 leg->AddEntry(emMPTauPhi,"Emulator MP", "l");
 
 hwMPTauPhi->SetStats(0);
 hwMPTauPhi->SetLineColor(kBlue);
 hwMPTauPhi->SetMarkerStyle(21);
 hwMPTauPhi->SetMarkerColor(1);
 hwMPTauPhi->SetMarkerSize(0.4);
 emMPTauPhi->SetLineColor(kRed);
 emMPTauPhi->SetMarkerStyle(20);
 emMPTauPhi->SetMarkerColor(kRed);
 emMPTauPhi->SetMarkerSize(0.4);
 emMPTauPhi->GetXaxis()->SetRange(0,100);
 hwMPTauPhi->GetXaxis()->SetTitle("Tau i#phi");
 hwMPTauPhi->GetYaxis()->SetTitle("# Taus");
 hwMPTauPhi->GetYaxis()->SetTitleSize(0.05);
 hwMPTauPhi->GetYaxis()->SetTitleOffset(0.66);
 hwMPTauPhi->GetXaxis()->SetTitleSize(0.04);
 hwMPTauPhi->GetXaxis()->SetTitleOffset(0.9);
 pTauPhi->SetBottomMargin(0.08);
 pTauPhi->Draw();
 pTauPhi->cd();

 TH1D* TauPhiRatio = (TH1D*)hwMPTauPhi->DrawCopy("p");
 TauPhiRatio->SetMinimum(0);
 emMPTauPhi->Draw("same");//"");
 leg->Draw();
 cTauPhi->cd();
 pTauPhiRatio->SetTopMargin(0.05);
 pTauPhiRatio->Draw();
 pTauPhiRatio->cd();
 hwMPTauPhi->Divide(emMPTauPhi);
 hwMPTauPhi->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPTauPhi->GetYaxis()->SetTitleSize(0.09);
 hwMPTauPhi->GetYaxis()->SetLabelSize(0.05);
 hwMPTauPhi->GetXaxis()->SetLabelSize(0.07);
 hwMPTauPhi->GetXaxis()->SetTitleSize(0.0);
 hwMPTauPhi->GetYaxis()->SetTitleOffset(0.35);
 hwMPTauPhi->SetMinimum(0.8);
 hwMPTauPhi->SetMaximum(1.2);
 hwMPTauPhi->Draw("p");
 cTauPhi->cd();
 pInvTauPhiRatio->Draw();
 pInvTauPhiRatio->cd();
 unity->Draw();
 


 if(doSave)  cTauPhi->SaveAs("compHwEmu/Taus/TauPhi.pdf");


 if(doDemux){

 TCanvas* cDTauPhi = new TCanvas("cDTauPhi","DTauPhi");

 TPad* pDTauPhi = new TPad("pTauPhi","pTauPhi",0,0.3,1,1); 
 TPad* pDTauPhiRatio = new TPad("pTauPhiratio","pTauPhiratio",0,0,1,0.3);
 
 TPad* pInvDTauPhiRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDTauPhiRatio->SetFillStyle(0);

 leg = new TLegend(0.75,0.75,0.9,0.9);
 leg->SetFillColor(0);
 leg->AddEntry(hwTauPhi,"Hardware Demux", "p");
 leg->AddEntry(emTauPhi,"Emulator Demux", "l");

 hwTauPhi->SetStats(0);
 hwTauPhi->SetMarkerStyle(21);
 hwTauPhi->SetMarkerColor(1);
 hwTauPhi->SetMarkerSize(0.4);
 emTauPhi->SetLineColor(kRed);
 hwTauPhi->GetXaxis()->SetTitle("Tau i#phi");
 hwTauPhi->GetXaxis()->SetRange(0,73);
 hwTauPhi->GetYaxis()->SetTitle("# Taus");
 hwTauPhi->GetYaxis()->SetTitleSize(0.05);
 hwTauPhi->GetYaxis()->SetTitleOffset(0.66);
 hwTauPhi->GetXaxis()->SetTitleSize(0.04);
 hwTauPhi->GetXaxis()->SetTitleOffset(0.9);
 pDTauPhi->SetBottomMargin(0.08);
 pDTauPhi->Draw();
 pDTauPhi->cd();

 TH1D* DTauPhiRatio = (TH1D*)hwTauPhi->DrawCopy("p");
 DTauPhiRatio->SetMinimum(0);
 emTauPhi->Draw("same");//"");
 leg->Draw();
 cDTauPhi->cd();
 pDTauPhiRatio->SetTopMargin(0.05);
 pDTauPhiRatio->Draw();
 pDTauPhiRatio->cd();
 hwTauPhi->Divide(emTauPhi);
 hwTauPhi->GetYaxis()->SetTitle("Ratio HW/EM");
 hwTauPhi->GetYaxis()->SetTitleSize(0.09);
 hwTauPhi->GetYaxis()->SetLabelSize(0.05);
 hwTauPhi->GetXaxis()->SetLabelSize(0.07);
 hwTauPhi->GetXaxis()->SetTitleSize(0.0);
 hwTauPhi->GetYaxis()->SetTitleOffset(0.35);
 hwTauPhi->SetMinimum(0.8);
 hwTauPhi->SetMaximum(1.2);
 hwTauPhi->Draw("p");
 cDTauPhi->cd();
 pInvDTauPhiRatio->Draw();
 pInvDTauPhiRatio->cd();
 unity->Draw();
 
 


 if(doSave)  cDTauPhi->SaveAs("compHwEmu/DemuxTaus/TauPhi.pdf");


 }

 }

//   //--- jet et ---//

if(doJets){

  TCanvas* cJetEt = new TCanvas("cJetEt","JetEt");

  TPad* pJetEt = new TPad("pJetEt","pJetEt",0,0.0,1,1); 
  if(doRatio) pJetEt = new TPad("pJetEt","pJetEt",0,0.3,1,1); 

  TPad* pJetEtRatio = new TPad("pJetEtratio","pJetEtratio",0,0,1,0.3);
  
  TPad* pInvJetEtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
  pInvJetEtRatio->SetFillStyle(0);

  leg = new TLegend(0.6,0.75,0.85,0.85);
  leg->SetFillColor(0);
  leg->AddEntry(hwMPJetEt,"Upgrade hardware", "p");//"l");
  //leg->AddEntry(hwJetEt,"Upgrade hardware", "p");
  leg->AddEntry(emMPJetEt,"Upgrade emulator", "l");
  //leg->AddEntry(emJetEt,"Upgrade emulator", "p");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  //hwMPJetEt->Rebin(2);
  //emMPJetEt->Rebin(2);

  hwMPJetEt->SetStats(0);
  //hwMPJetEt->SetLineColor(kBlue);
  hwMPJetEt->SetMarkerStyle(21);
  hwMPJetEt->SetMarkerColor(1);
  hwMPJetEt->SetMarkerSize(0.4);
  hwMPJetEt->GetXaxis()->SetRange(0,200);
  emMPJetEt->SetLineColor(kRed);
  hwMPJetEt->GetXaxis()->SetTitle("Level-1 Trigger Jet iE_{T}");
  hwMPJetEt->GetYaxis()->SetTitle("Number of candidates");
  hwMPJetEt->GetYaxis()->SetTitleSize(0.05);
  hwMPJetEt->GetYaxis()->SetTitleOffset(0.77);
  hwMPJetEt->GetXaxis()->SetTitleSize(0.04);
  hwMPJetEt->GetXaxis()->SetTitleOffset(1.0);
  pJetEt->SetBottomMargin(0.12);
  pJetEt->Draw();
  pJetEt->cd();

  TH1D* JetEtRatio = (TH1D*)hwMPJetEt->DrawCopy("p");
  JetEtRatio->SetMinimum(0);
  emMPJetEt->Draw("same");//"");
  leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");



  if(doRatio){

  cJetEt->cd();
  pJetEtRatio->SetTopMargin(0.05);
  pJetEtRatio->Draw();
  pJetEtRatio->cd();
  hwMPJetEt->Divide(emMPJetEt);
  hwMPJetEt->GetYaxis()->SetTitle("Ratio HW/EM");
  hwMPJetEt->GetYaxis()->SetTitleSize(0.09);
  hwMPJetEt->GetYaxis()->SetLabelSize(0.05);
  hwMPJetEt->GetXaxis()->SetLabelSize(0.07);
  hwMPJetEt->GetXaxis()->SetTitleSize(0.0);
  hwMPJetEt->GetYaxis()->SetTitleOffset(0.35);
  hwMPJetEt->SetMinimum(0.8);
  hwMPJetEt->SetMaximum(1.2);
  hwMPJetEt->Draw("p");
  cJetEt->cd();
  pInvJetEtRatio->Draw();
  pInvJetEtRatio->cd();
  unity->Draw();

  }

 if(doSave)  cJetEt->SaveAs("compHwEmu/Jets/JetEt.pdf");

  
 if(doDemux){

 TCanvas* cDJetEt = new TCanvas("cDJetEt","DJetEt");

 TPad* pDJetEt = new TPad("pJetEt","pJetEt",0,0.0,1,1); 
 if(doRatio) pDJetEt = new TPad("pJetEt","pJetEt",0,0.3,1,1); 

 TPad* pDJetEtRatio = new TPad("pJetEtratio","pJetEtratio",0,0,1,0.3);
 
 TPad* pInvDJetEtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDJetEtRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.75,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwJetEt,"Upgrade hardware", "p");
 leg->AddEntry(emJetEt,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);

 //hwJetEt->Rebin(2);
 //emJetEt->Rebin(2);

 hwJetEt->SetStats(0);
 hwJetEt->SetMarkerStyle(21);
 hwJetEt->SetMarkerColor(1);
 hwJetEt->SetMarkerSize(0.4);
 hwJetEt->GetXaxis()->SetRange(0,200);
 emJetEt->SetLineColor(kRed);
 hwJetEt->GetXaxis()->SetTitle("Level-1 Trigger Jet iE_{T}");
 hwJetEt->GetYaxis()->SetTitle("Number of candidates");
 hwJetEt->GetYaxis()->SetTitleSize(0.05);
 hwJetEt->GetYaxis()->SetTitleOffset(0.77);
 hwJetEt->GetXaxis()->SetTitleSize(0.04);
 hwJetEt->GetXaxis()->SetTitleOffset(1.0);
 pDJetEt->SetBottomMargin(0.12);
 pDJetEt->Draw();
 pDJetEt->cd();
 
 TH1D* DJetEtRatio = (TH1D*)hwJetEt->DrawCopy("p");
 DJetEtRatio->SetMinimum(0);
 emJetEt->Draw("same");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");



 if(doRatio){

 cDJetEt->cd();
 pDJetEtRatio->SetTopMargin(0.05);
 pDJetEtRatio->Draw();
 pDJetEtRatio->cd();
 hwJetEt->Divide(emJetEt);
 hwJetEt->GetYaxis()->SetTitle("Ratio HW/EM");
 hwJetEt->GetYaxis()->SetTitleSize(0.09);
 hwJetEt->GetYaxis()->SetLabelSize(0.05);
 hwJetEt->GetXaxis()->SetLabelSize(0.07);
 hwJetEt->GetXaxis()->SetTitleSize(0.0);
 hwJetEt->GetYaxis()->SetTitleOffset(0.35);
 hwJetEt->SetMinimum(0.8);
 hwJetEt->SetMaximum(1.2);
 hwJetEt->Draw("p");
 cDJetEt->cd();
 pInvDJetEtRatio->Draw();
 pInvDJetEtRatio->cd();
 unity->Draw();
 pDJetEtRatio->Update();

 }

 if(doSave) cDJetEt->SaveAs("compHwEmu/DemuxJets/JetEt.pdf");

 }


 // //--- jet eta ---//

 TCanvas* cJetEta = new TCanvas("cJetEta","JetEta");

 TPad* pJetEta = new TPad("pJetEt","pJetEt",0,0.0,1,1); 
 if(doRatio) pJetEta = new TPad("pJetEt","pJetEt",0,0.3,1,1); 
 
 TPad* pJetEtaRatio = new TPad("pJetEtratio","pJetEtratio",0,0,1,0.3);
  
 TPad* pInvJetEtaRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvJetEtaRatio->SetFillStyle(0);

 leg = new TLegend(0.4,0.77,0.65,0.87);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPJetEta,"Upgrade hardware", "p");//"l");
 //leg->AddEntry(hwJetEta,"Upgrade hardware", "p");
 leg->AddEntry(emMPJetEta,"Upgrade emulator", "l");
 //leg->AddEntry(emJetEta,"Upgrade emulator", "p");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwMPJetEta->SetStats(0);
 //hwMPJetEta->SetLineColor(kBlue);
 hwMPJetEta->SetMarkerStyle(21);
 hwMPJetEta->SetMarkerColor(1);
 hwMPJetEta->SetMarkerSize(0.4);
 emMPJetEta->SetLineColor(kRed);
 emMPJetEta->SetMarkerStyle(20);
 emMPJetEta->SetMarkerColor(kRed);
 emMPJetEta->SetMarkerSize(0.4);
 //hwMPJetEta->GetXaxis()->SetRange(10,74);
 hwMPJetEta->GetXaxis()->SetTitle("Level-1 Trigger Jet i#eta");
 hwMPJetEta->GetYaxis()->SetTitle("Number of candidates");
 hwMPJetEta->GetYaxis()->SetTitleSize(0.05);
 hwMPJetEta->GetYaxis()->SetTitleOffset(0.77);
 hwMPJetEta->GetXaxis()->SetTitleSize(0.04);
 hwMPJetEta->GetXaxis()->SetTitleOffset(1.0);
 pJetEta->SetBottomMargin(0.12);
 pJetEta->Draw();
 pJetEta->cd();

 TH1D* JetEtaRatio = (TH1D*)hwMPJetEta->DrawCopy("p");
 JetEtaRatio->SetMinimum(0);
 emMPJetEta->Draw("same");//"");
 leg->Draw();
 n2.DrawLatex(0.4, 0.65, "CMS");
 n3.DrawLatex(0.4, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.4, 0.55, "Preliminary");
 n4.DrawLatex(0.4, 0.45, "Single Muon stream");


 if(doRatio){

 cJetEta->cd();
 pJetEtaRatio->SetTopMargin(0.05);
 pJetEtaRatio->Draw();
 pJetEtaRatio->cd();
 hwMPJetEta->Divide(emMPJetEta);
 hwMPJetEta->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPJetEta->GetYaxis()->SetTitleSize(0.09);
 hwMPJetEta->GetYaxis()->SetLabelSize(0.05);
 hwMPJetEta->GetXaxis()->SetLabelSize(0.07);
 hwMPJetEta->GetXaxis()->SetTitleSize(0.0);
 hwMPJetEta->GetYaxis()->SetTitleOffset(0.35);
 hwMPJetEta->SetMinimum(0.8);
 hwMPJetEta->SetMaximum(1.2);
 hwMPJetEta->Draw("p");
 cJetEta->cd();
 pInvJetEtaRatio->Draw();
 pInvJetEtaRatio->cd();
 unity->Draw();

 }

 if(doSave) cJetEta->SaveAs("compHwEmu/Jets/JetEta.pdf");


 if(doDemux){

 TCanvas* cDJetEta = new TCanvas("cDJetEta","DJetEta");

 TPad* pDJetEta = new TPad("pJetEta","pJetEta",0,0.0,1,1); 
 if(doRatio) pDJetEta = new TPad("pJetEta","pJetEta",0,0.3,1,1); 

 TPad* pDJetEtaRatio = new TPad("pJetEtaratio","pJetEtaratio",0,0,1,0.3);
 
 TPad* pInvDJetEtaRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDJetEtaRatio->SetFillStyle(0);

 leg = new TLegend(0.65,0.77,0.9,0.87);
 leg->SetFillColor(0);
 leg->AddEntry(hwJetEta,"Upgrade hardware", "p");
 leg->AddEntry(emJetEta,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);

 hwJetEta->SetStats(0);
 hwJetEta->SetMarkerStyle(21);
 hwJetEta->SetMarkerColor(1);
 hwJetEta->SetMarkerSize(0.4);
 emJetEta->SetLineColor(kRed);
 hwJetEta->GetXaxis()->SetTitle("Level-1 Trigger Jet i#eta");
 // hwJetEta->GetXaxis()->SetRange(82,146);
 hwJetEta->GetYaxis()->SetTitle("Number of candidates");
 hwJetEta->GetYaxis()->SetTitleSize(0.05);
 hwJetEta->GetYaxis()->SetTitleOffset(0.77);
 hwJetEta->GetXaxis()->SetTitleSize(0.04);
 hwJetEta->GetXaxis()->SetTitleOffset(1.0);
 pDJetEta->SetBottomMargin(0.12);
 pDJetEta->Draw();
 pDJetEta->cd();
 
 TH1D* DJetEtaRatio = (TH1D*)hwJetEta->DrawCopy("p");
 DJetEtaRatio->SetMinimum(0);
 emJetEta->Draw("same");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");


 if(doRatio){

 cDJetEta->cd();
 pDJetEtaRatio->SetTopMargin(0.05);
 pDJetEtaRatio->Draw();
 pDJetEtaRatio->cd();
 hwJetEta->Divide(emJetEta);
 hwJetEta->GetYaxis()->SetTitle("Ratio HW/EM");
 hwJetEta->GetYaxis()->SetTitleSize(0.09);
 hwJetEta->GetYaxis()->SetLabelSize(0.05);
 hwJetEta->GetXaxis()->SetLabelSize(0.07);
 hwJetEta->GetXaxis()->SetTitleSize(0.0);
 hwJetEta->GetYaxis()->SetTitleOffset(0.35);
 hwJetEta->SetMinimum(0.8);
 hwJetEta->SetMaximum(1.2);
 hwJetEta->Draw("p");
 cDJetEta->cd();
 pInvDJetEtaRatio->Draw();
 pInvDJetEtaRatio->cd();
 unity->Draw();
 pDJetEtaRatio->Update();

 }

 if(doSave) cDJetEta->SaveAs("compHwEmu/DemuxJets/JetEta.pdf","pdf");


 }

//--- jet phi ---//

 TCanvas* cJetPhi = new TCanvas("cJetPhi","JetPhi");

 TPad* pJetPhi = new TPad("pJetPhi","pJetPhi",0,0.0,1,1); 
 if(doRatio) pJetPhi = new TPad("pJetPhi","pJetPhi",0,0.3,1,1); 

 TPad* pJetPhiRatio = new TPad("pJetPhiratio","pJetPhiratio",0,0,1,0.3);
 
 TPad* pInvJetPhiRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvJetPhiRatio->SetFillStyle(0);


 leg = new TLegend(0.6,0.25,0.85,0.35);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPJetPhi,"Upgrade hardware", "p");//"l");
 //leg->AddEntry(hwJetPhi,"Upgrade hardware", "p");
 leg->AddEntry(emMPJetPhi,"Upgrade emulator", "l");
 //leg->AddEntry(emJetPhi,"Upgrade emulator", "p");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwMPJetPhi->Rebin(2);
 emMPJetPhi->Rebin(2);
     

 hwMPJetPhi->SetStats(0);
 //hwMPJetPhi->SetLineColor(kBlue);
 hwMPJetPhi->SetMarkerStyle(21);
 hwMPJetPhi->SetMarkerColor(1);
 hwMPJetPhi->SetMarkerSize(0.4);
 emMPJetPhi->SetLineColor(kRed);
 emMPJetPhi->SetMarkerStyle(20);
 emMPJetPhi->SetMarkerColor(kRed);
 emMPJetPhi->SetMarkerSize(0.4);
 //emMPJetPhi->GetXaxis()->SetRange(0,100);
 //emMPJetPhi->SetMaximum(220);
 hwMPJetPhi->GetXaxis()->SetTitle("Level-1 Trigger Jet i#phi");
 hwMPJetPhi->GetYaxis()->SetTitle("Number of candidates");
 hwMPJetPhi->GetYaxis()->SetTitleSize(0.05);
 hwMPJetPhi->GetYaxis()->SetTitleOffset(0.77);
 hwMPJetPhi->GetXaxis()->SetTitleSize(0.04);
 hwMPJetPhi->GetXaxis()->SetTitleOffset(1.0);
 //hwMPJetPhi->SetMaximum(220);
 pJetPhi->SetBottomMargin(0.12);
 pJetPhi->Draw();
 pJetPhi->cd();

 TH1D* JetPhiRatio = (TH1D*)hwMPJetPhi->DrawCopy("p");
 JetPhiRatio->SetMinimum(0);
 emMPJetPhi->Draw("same");//"");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");


 if(doRatio){

 cJetPhi->cd();
 pJetPhiRatio->SetTopMargin(0.05);
 pJetPhiRatio->Draw();
 pJetPhiRatio->cd();
 hwMPJetPhi->Divide(emMPJetPhi);
 hwMPJetPhi->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPJetPhi->GetYaxis()->SetTitleSize(0.09);
 hwMPJetPhi->GetYaxis()->SetLabelSize(0.05);
 hwMPJetPhi->GetXaxis()->SetLabelSize(0.07);
 hwMPJetPhi->GetXaxis()->SetTitleSize(0.0);
 hwMPJetPhi->GetYaxis()->SetTitleOffset(0.35);
 hwMPJetPhi->SetMinimum(0.8);
 hwMPJetPhi->SetMaximum(1.2);
 hwMPJetPhi->Draw("p");
 cJetPhi->cd();
 pInvJetPhiRatio->Draw();
 pInvJetPhiRatio->cd();
 unity->Draw();

 }

 if(doSave) cJetPhi->SaveAs("compHwEmu/Jets/JetPhi.pdf","pdf");


 if(doDemux){


 TCanvas* cDJetPhi = new TCanvas("cDJetPhi","DJetPhi");

 TPad* pDJetPhi = new TPad("pJetPhi","pJetPhi",0,0.0,1,1); 
 if(doRatio) pDJetPhi = new TPad("pJetPhi","pJetPhi",0,0.3,1,1); 

 TPad* pDJetPhiRatio = new TPad("pJetPhiratio","pJetPhiratio",0,0,1,0.3);

 TPad* pInvDJetPhiRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDJetPhiRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.78,0.85,0.88);
 leg->SetFillColor(0);
 leg->AddEntry(hwJetPhi,"Upgrade hardware", "p");
 leg->AddEntry(emJetPhi,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwJetPhi->Rebin(4);
 emJetPhi->Rebin(4);


 hwJetPhi->SetStats(0);
 hwJetPhi->SetMarkerStyle(21);
 hwJetPhi->SetMarkerColor(1);
 hwJetPhi->SetMarkerSize(0.4);
 emJetPhi->SetMarkerStyle(20);
 emJetPhi->SetLineColor(kRed);
 //emJetPhi->GetXaxis()->SetRange(0,100);
 //emJetPhi->SetMaximum(220);
 hwJetPhi->GetXaxis()->SetTitle("Level-1 Trigger Jet i#phi");
 hwJetPhi->GetYaxis()->SetTitle("Number of candidates");
 hwJetPhi->GetYaxis()->SetTitleSize(0.05);
 hwJetPhi->GetYaxis()->SetTitleOffset(0.77);
 hwJetPhi->GetXaxis()->SetTitleSize(0.04);
 hwJetPhi->GetXaxis()->SetTitleOffset(1.0);
 //hwJetPhi->SetMaximum(220);
 pDJetPhi->SetBottomMargin(0.12);
 pDJetPhi->Draw();
 pDJetPhi->cd();

 TH1D* DJetPhiRatio = (TH1D*)hwJetPhi->DrawCopy("p");
 DJetPhiRatio->SetMinimum(0);
 emJetPhi->Draw("same");//"");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");



 if(doRatio){

 cDJetPhi->cd();
 pDJetPhiRatio->SetTopMargin(0.05);
 pDJetPhiRatio->Draw();
 pDJetPhiRatio->cd();
 hwJetPhi->Divide(emJetPhi);
 hwJetPhi->GetYaxis()->SetTitle("Ratio HW/EM");
 hwJetPhi->GetYaxis()->SetTitleSize(0.09);
 hwJetPhi->GetYaxis()->SetLabelSize(0.05);
 hwJetPhi->GetXaxis()->SetLabelSize(0.07);
 hwJetPhi->GetXaxis()->SetTitleSize(0.0);
 hwJetPhi->GetYaxis()->SetTitleOffset(0.35);
 hwJetPhi->SetMinimum(0.8);
 hwJetPhi->SetMaximum(1.2);
 hwJetPhi->Draw("p");
 cDJetPhi->cd();
 pInvDJetPhiRatio->Draw();
 pInvDJetPhiRatio->cd();
 unity->Draw();

 }

 if(doSave)  cDJetPhi->SaveAs("compHwEmu/DemuxJets/JetPhi.pdf","pdf");

 }

 }

//--- MP sum et ---//

 if(doSums){

 TCanvas* cMPSumEt = new TCanvas("cMPSumEt","MPSumEt");

 TPad* pSumEt = new TPad("pSumEt","pSumEt",0,0.0,1,1); 
 if(doRatio) pSumEt = new TPad("pSumEt","pSumEt",0,0.3,1,1); 

 TPad* pSumEtRatio = new TPad("pSumEtratio","pSumEtratio",0,0,1,0.3);

 TPad* pInvSumEtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSumEtRatio->SetFillStyle(0);

 leg = new TLegend(0.65,0.75,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPSumEt,"Upgrade hardware", "p");
 leg->AddEntry(emMPSumEt,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwMPSumEt->Rebin(40);
 emMPSumEt->Rebin(40);

 hwMPSumEt->SetStats(0);
 hwMPSumEt->SetMarkerStyle(21);
 hwMPSumEt->SetMarkerColor(1);
 hwMPSumEt->SetMarkerSize(0.4);
 emMPSumEt->SetLineColor(kRed);
 //hwMPSumEt->GetXaxis()->SetRange(0,100);
 hwMPSumEt->GetXaxis()->SetTitle("Level-1 Trigger Jet iE_{T}");
 hwMPSumEt->GetYaxis()->SetTitle("Number of events");
 hwMPSumEt->GetYaxis()->SetTitleSize(0.05);
 hwMPSumEt->GetYaxis()->SetTitleOffset(0.77);
 hwMPSumEt->GetXaxis()->SetTitleSize(0.04);
 hwMPSumEt->GetXaxis()->SetTitleOffset(1.0);
 pSumEt->SetBottomMargin(0.12);
 pSumEt->Draw();
 pSumEt->cd();
 
 TH1D* SumEtRatio = (TH1D*)hwMPSumEt->DrawCopy("p");
 SumEtRatio->SetMinimum(0);
 emMPSumEt->Draw("same");//""); 
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");

 if(doRatio){

 cMPSumEt->cd();
 pSumEtRatio->SetTopMargin(0.05);
 pSumEtRatio->Draw();
 pSumEtRatio->cd();
 hwMPSumEt->Divide(emMPSumEt);
 hwMPSumEt->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPSumEt->GetYaxis()->SetTitleSize(0.09);
 hwMPSumEt->GetYaxis()->SetLabelSize(0.05);
 hwMPSumEt->GetXaxis()->SetLabelSize(0.07);
 hwMPSumEt->GetXaxis()->SetTitleSize(0.0);
 hwMPSumEt->GetYaxis()->SetTitleOffset(0.35);
 hwMPSumEt->SetMinimum(0.8);
 hwMPSumEt->SetMaximum(1.2);
 hwMPSumEt->Draw("p");
 cMPSumEt->cd();
 pInvSumEtRatio->Draw();
 pInvSumEtRatio->cd();
 unity->Draw();

 }

 if(doSave)  cMPSumEt->SaveAs("compHwEmu/MPSums/MPSumEt.pdf","pdf");


 // //--- MP sum etx ---//

 TCanvas* cMPSumEtx = new TCanvas("cMPSumEtx","MPSumEtx");

 TPad* pSumEtx = new TPad("pSumEtx","pSumEtx",0,0.0,1,1); 
 if(doRatio) pSumEtx = new TPad("pSumEtx","pSumEtx",0,0.3,1,1); 

 TPad* pSumEtxRatio = new TPad("pSumEtxratio","pSumEtxratio",0,0,1,0.3);
 
 TPad* pInvSumEtxRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSumEtxRatio->SetFillStyle(0);

 leg = new TLegend(0.65,0.75,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPSumEtx,"Upgrade hardware", "p");
 leg->AddEntry(emMPSumEtx,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwMPSumEtx->Rebin(10);
 emMPSumEtx->Rebin(10);

 hwMPSumEtx->SetStats(0);
 hwMPSumEtx->SetMarkerStyle(21);
 hwMPSumEtx->SetMarkerColor(1);
 hwMPSumEtx->SetMarkerSize(0.4);
 emMPSumEtx->SetLineColor(kRed);
 nbins = hwMPSumEtx->GetXaxis()->GetNbins();
 //hwMPSumEtx->GetXaxis()->SetRange((nbins/2)-40,(nbins/2)+40);
 hwMPSumEtx->GetXaxis()->SetTitle("Level-1 Trigger iE_{T,x}");
 hwMPSumEtx->GetYaxis()->SetTitle("Number of events");
 hwMPSumEtx->GetYaxis()->SetTitleSize(0.05);
 hwMPSumEtx->GetYaxis()->SetTitleOffset(0.77);
 hwMPSumEtx->GetXaxis()->SetTitleSize(0.04);
 hwMPSumEtx->GetXaxis()->SetTitleOffset(1.0);
 pSumEtx->SetBottomMargin(0.12);
 pSumEtx->Draw();
 pSumEtx->cd();
 
 TH1D* SumEtxRatio = (TH1D*)hwMPSumEtx->DrawCopy("p");
 SumEtxRatio->SetMinimum(0);
 emMPSumEtx->Draw("same");//"");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");
 
 if(doRatio){

 cMPSumEtx->cd();
 pSumEtxRatio->SetTopMargin(0.05);
 pSumEtxRatio->Draw();
 pSumEtxRatio->cd();
 hwMPSumEtx->Divide(emMPSumEtx);
 hwMPSumEtx->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPSumEtx->GetYaxis()->SetTitleSize(0.09);
 hwMPSumEtx->GetYaxis()->SetLabelSize(0.05);
 hwMPSumEtx->GetXaxis()->SetLabelSize(0.07);
 hwMPSumEtx->GetXaxis()->SetTitleSize(0.0);
 hwMPSumEtx->GetYaxis()->SetTitleOffset(0.35);
 hwMPSumEtx->SetMinimum(0.8);
 hwMPSumEtx->SetMaximum(1.2);
 hwMPSumEtx->Draw("p");
 cMPSumEtx->cd();
 pInvSumEtxRatio->Draw();
 pInvSumEtxRatio->cd();
 unity->Draw();
 
 }

 if(doSave)  cMPSumEtx->SaveAs("compHwEmu/MPSums/MPSumEtx.pdf","pdf");


 // //--- MP sum ety ---//

 TCanvas* cMPSumEty = new TCanvas("cMPSumEty","MPSumEty");

  TPad* pSumEty = new TPad("pSumEty","pSumEty",0,0.0,1,1); 
 if(doRatio) pSumEty = new TPad("pSumEty","pSumEty",0,0.3,1,1); 

 TPad* pSumEtyRatio = new TPad("pSumEtyratio","pSumEtyratio",0,0,1,0.3);
  
 TPad* pInvSumEtyRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSumEtyRatio->SetFillStyle(0);


 leg = new TLegend(0.65,0.75,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPSumEty,"Upgrade hardware", "p");
 leg->AddEntry(emMPSumEty,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwMPSumEty->Rebin(10);
 emMPSumEty->Rebin(10);

 hwMPSumEty->SetStats(0);
 hwMPSumEty->SetMarkerStyle(21);
 hwMPSumEty->SetMarkerColor(1);
 hwMPSumEty->SetMarkerSize(0.4);
 emMPSumEty->SetLineColor(kRed);
 nbins = hwMPSumEty->GetXaxis()->GetNbins();
 //hwMPSumEty->GetXaxis()->SetRange((nbins/2)-40,(nbins/2)+40);
 hwMPSumEty->GetXaxis()->SetTitle("Level-1 Trigger iE_{T,y}");
 hwMPSumEty->GetYaxis()->SetTitle("Number of events");
 hwMPSumEty->GetYaxis()->SetTitleSize(0.05);
 hwMPSumEty->GetYaxis()->SetTitleOffset(0.77);
 hwMPSumEty->GetXaxis()->SetTitleSize(0.04);
 hwMPSumEty->GetXaxis()->SetTitleOffset(1.0);
 pSumEty->SetBottomMargin(0.12);
 pSumEty->Draw();
 pSumEty->cd();
 
 TH1D* SumEtyRatio = (TH1D*)hwMPSumEty->DrawCopy("p");
 SumEtyRatio->SetMinimum(0);
 emMPSumEty->Draw("same");//"");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");
 

 if(doRatio){

 cMPSumEty->cd();
 pSumEtyRatio->SetTopMargin(0.05);
 pSumEtyRatio->Draw();
 pSumEtyRatio->cd();
 hwMPSumEty->Divide(emMPSumEty);
 hwMPSumEty->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPSumEty->GetYaxis()->SetTitleSize(0.09);
 hwMPSumEty->GetYaxis()->SetLabelSize(0.05);
 hwMPSumEty->GetXaxis()->SetLabelSize(0.07);
 hwMPSumEty->GetXaxis()->SetTitleSize(0.0);
 hwMPSumEty->GetYaxis()->SetTitleOffset(0.35);
 hwMPSumEty->SetMinimum(0.8);
 hwMPSumEty->SetMaximum(1.2);
 hwMPSumEty->Draw("p");
 cMPSumEty->cd();
 pInvSumEtyRatio->Draw();
 pInvSumEtyRatio->cd();
 unity->Draw();

 }

 if(doSave)  cMPSumEty->SaveAs("compHwEmu/MPSums/MPSumEty.pdf","pdf");


 //--- MP sum ht ---//

 TCanvas* cMPSumHt = new TCanvas("cMPSumHt","MPSumHt");

 TPad* pSumHt = new TPad("pSumHt","pSumHt",0,0.0,1,1); 
 if(doRatio) pSumHt = new TPad("pSumHt","pSumHt",0,0.3,1,1); 

 TPad* pSumHtRatio = new TPad("pSumHtratio","pSumHtratio",0,0,1,0.3);
 
 TPad* pInvSumHtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSumHtRatio->SetFillStyle(0);

 leg = new TLegend(0.65,0.75,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPSumHt,"Upgrade hardware", "p");
 leg->AddEntry(emMPSumHt,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwMPSumHt->Rebin(40);
 emMPSumHt->Rebin(40);

 hwMPSumHt->SetStats(0);
 hwMPSumHt->SetMarkerStyle(21);
 hwMPSumHt->SetMarkerColor(1);
 hwMPSumHt->SetMarkerSize(0.4);
 emMPSumHt->SetLineColor(kRed);
 //hwMPSumHt->GetXaxis()->SetRange(0,100);
 hwMPSumHt->GetXaxis()->SetTitle("Level-1 Trigger iH_{T}");
 hwMPSumHt->GetYaxis()->SetTitle("Number of events");
 hwMPSumHt->GetYaxis()->SetTitleSize(0.05);
 hwMPSumHt->GetYaxis()->SetTitleOffset(0.77);
 hwMPSumHt->GetXaxis()->SetTitleSize(0.04);
 hwMPSumHt->GetXaxis()->SetTitleOffset(1.0);
 pSumHt->SetBottomMargin(0.12);
 pSumHt->Draw();
 pSumHt->cd();

 TH1D* SumHtRatio = (TH1D*)hwMPSumHt->DrawCopy("p");
 SumHtRatio->SetMinimum(0);
 emMPSumHt->Draw("same");//"");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");


 if(doRatio){

 cMPSumHt->cd();
 pSumHtRatio->SetTopMargin(0.05);
 pSumHtRatio->Draw();
 pSumHtRatio->cd();
 hwMPSumHt->Divide(emMPSumHt);
 hwMPSumHt->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPSumHt->GetYaxis()->SetTitleSize(0.09);
 hwMPSumHt->GetYaxis()->SetLabelSize(0.05);
 hwMPSumHt->GetXaxis()->SetLabelSize(0.07);
 hwMPSumHt->GetXaxis()->SetTitleSize(0.0);
 hwMPSumHt->GetYaxis()->SetTitleOffset(0.35);
 hwMPSumHt->SetMinimum(0.8);
 hwMPSumHt->SetMaximum(1.2);
 hwMPSumHt->Draw("p");
 cMPSumHt->cd();
 pInvSumHtRatio->Draw();
 pInvSumHtRatio->cd();
 unity->Draw();

 }

 if(doSave)  cMPSumHt->SaveAs("compHwEmu/MPSums/MPSumHt.pdf","pdf");


 // //--- MP sum htx ---//

 TCanvas* cMPSumHtx = new TCanvas("cMPSumHtx","MPSumHtx");

 TPad* pSumHtx = new TPad("pSumHtx","pSumHtx",0,0.0,1,1); 
 if(doRatio) pSumHtx = new TPad("pSumHtx","pSumHtx",0,0.3,1,1); 

 TPad* pSumHtxRatio = new TPad("pSumHtxratio","pSumHtxratio",0,0,1,0.3);
 
 TPad* pInvSumHtxRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSumHtxRatio->SetFillStyle(0);

 leg = new TLegend(0.65,0.75,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPSumHtx,"Upgrade hardware", "p");
 leg->AddEntry(emMPSumHtx,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwMPSumHtx->Rebin(10);
 emMPSumHtx->Rebin(10);

 hwMPSumHtx->SetStats(0);
 hwMPSumHtx->SetMarkerStyle(21);
 hwMPSumHtx->SetMarkerColor(1);
 hwMPSumHtx->SetMarkerSize(0.4);
 emMPSumHtx->SetLineColor(kRed);
 nbins = hwMPSumHtx->GetXaxis()->GetNbins();
 //hwMPSumHtx->GetXaxis()->SetRange((nbins/2)-40,(nbins/2)+40);
 hwMPSumHtx->GetXaxis()->SetTitle("Level-1 Trigger iH_{T,x}");
 hwMPSumHtx->GetYaxis()->SetTitle("Number of events");
 hwMPSumHtx->GetYaxis()->SetTitleSize(0.05);
 hwMPSumHtx->GetYaxis()->SetTitleOffset(0.77);
 hwMPSumHtx->GetXaxis()->SetTitleSize(0.04);
 hwMPSumHtx->GetXaxis()->SetTitleOffset(1.0);
 pSumHtx->SetBottomMargin(0.12);
 pSumHtx->Draw();
 pSumHtx->cd();
 
 TH1D* SumHtxRatio = (TH1D*)hwMPSumHtx->DrawCopy("p");
 SumHtxRatio->SetMinimum(0);
 emMPSumHtx->Draw("same");//"");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");


 if(doRatio){

 cMPSumHtx->cd();
 pSumHtxRatio->SetTopMargin(0.05);
 pSumHtxRatio->Draw();
 pSumHtxRatio->cd();
 hwMPSumHtx->Divide(emMPSumHtx);
 hwMPSumHtx->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPSumHtx->GetYaxis()->SetTitleSize(0.09);
 hwMPSumHtx->GetYaxis()->SetLabelSize(0.05);
 hwMPSumHtx->GetXaxis()->SetLabelSize(0.07);
 hwMPSumHtx->GetXaxis()->SetTitleSize(0.0);
 hwMPSumHtx->GetYaxis()->SetTitleOffset(0.35);
 hwMPSumHtx->SetMinimum(0.8);
 hwMPSumHtx->SetMaximum(1.2);
 hwMPSumHtx->Draw("p");
 cMPSumHtx->cd();
 pInvSumHtxRatio->Draw();
 pInvSumHtxRatio->cd();
 unity->Draw();

 }

 if(doSave)  cMPSumHtx->SaveAs("compHwEmu/MPSums/MPSumHtx.pdf","pdf");


 // //--- MP sum hty ---//

 TCanvas* cMPSumHty = new TCanvas("cMPSumHty","MPSumHty");

 TPad* pSumHty = new TPad("pSumHty","pSumHty",0,0.0,1,1); 
 if(doRatio) pSumHty = new TPad("pSumHty","pSumHty",0,0.3,1,1); 

 TPad* pSumHtyRatio = new TPad("pSumHtyratio","pSumHtyratio",0,0,1,0.3);
 
 TPad* pInvSumHtyRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSumHtyRatio->SetFillStyle(0);
  
 leg = new TLegend(0.65,0.75,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPSumHty,"Upgrade hardware", "p");
 leg->AddEntry(emMPSumHty,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwMPSumHty->Rebin(10);
 emMPSumHty->Rebin(10);

 hwMPSumHty->SetStats(0);
 hwMPSumHty->SetMarkerStyle(21);
 hwMPSumHty->SetMarkerColor(1);
 hwMPSumHty->SetMarkerSize(0.4);
 emMPSumHty->SetLineColor(kRed);
 nbins = hwMPSumHty->GetXaxis()->GetNbins();
 //hwMPSumHty->GetXaxis()->SetRange((nbins/2)-40,(nbins/2)+40);
 hwMPSumHty->GetXaxis()->SetTitle("Level-1 Trigger iH_{T,y}");
 hwMPSumHty->GetYaxis()->SetTitle("Number of events");
 hwMPSumHty->GetYaxis()->SetTitleSize(0.05);
 hwMPSumHty->GetYaxis()->SetTitleOffset(0.77);
 hwMPSumHty->GetXaxis()->SetTitleSize(0.04);
 hwMPSumHty->GetXaxis()->SetTitleOffset(1.0);
 pSumHty->SetBottomMargin(0.12);
 pSumHty->Draw();
 pSumHty->cd();

 TH1D* SumHtyRatio = (TH1D*)hwMPSumHty->DrawCopy("p");
 SumHtyRatio->SetMinimum(0);
 emMPSumHty->Draw("same");//"");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");
 

 if(doRatio){

 cMPSumHty->cd();
 pSumHtyRatio->SetTopMargin(0.05);
 pSumHtyRatio->Draw();
 pSumHtyRatio->cd();
 hwMPSumHty->Divide(emMPSumHty);
 hwMPSumHty->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPSumHty->GetYaxis()->SetTitleSize(0.09);
 hwMPSumHty->GetYaxis()->SetLabelSize(0.05);
 hwMPSumHty->GetXaxis()->SetLabelSize(0.07);
 hwMPSumHty->GetXaxis()->SetTitleSize(0.0);
 hwMPSumHty->GetYaxis()->SetTitleOffset(0.35);
 hwMPSumHty->SetMinimum(0.8);
 hwMPSumHty->SetMaximum(1.2);
 hwMPSumHty->Draw("p");
 cMPSumHty->cd();
 pInvSumHtyRatio->Draw();
 pInvSumHtyRatio->cd();
 unity->Draw();

 }

 if(doSave)  cMPSumHty->SaveAs("compHwEmu/MPSums/MPSumHty.pdf","pdf");


 if(doDemux){

 // //--- demux sum et ---//

 TCanvas* cSumEt = new TCanvas("cSumEt","SumEt");

 TPad* pDSumEt = new TPad("pSumEt","pSumEt",0,0.0,1,1); 
 if(doRatio) pDSumEt = new TPad("pSumEt","pSumEt",0,0.3,1,1); 

 TPad* pDSumEtRatio = new TPad("pSumEtratio","pSumEtratio",0,0,1,0.3);
 
 TPad* pInvDSumEtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDSumEtRatio->SetFillStyle(0);

 leg = new TLegend(0.65,0.75,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwSumEt,"Upgrade hardware", "p");
 leg->AddEntry(emSumEt,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwSumEt->Rebin(40);
 emSumEt->Rebin(40);

 hwSumEt->SetStats(0);
 hwSumEt->SetMarkerStyle(21);
 hwSumEt->SetMarkerColor(1);
 hwSumEt->SetMarkerSize(0.4);
 emSumEt->SetLineColor(kRed);
 //hwSumEt->GetXaxis()->SetRange(0,50);
 hwSumEt->GetXaxis()->SetTitle("Level-1 Trigger iE_{T}");
 hwSumEt->GetYaxis()->SetTitle("Number of events");
 hwSumEt->GetYaxis()->SetTitleSize(0.05);
 hwSumEt->GetYaxis()->SetTitleOffset(0.77);
 hwSumEt->GetXaxis()->SetTitleSize(0.04);
 hwSumEt->GetXaxis()->SetTitleOffset(1.0);
 pDSumEt->SetBottomMargin(0.12);
 pDSumEt->Draw();
 pDSumEt->cd();
 
 TH1D* DSumEtRatio = (TH1D*)hwSumEt->DrawCopy("p");
 DSumEtRatio->SetMinimum(0);
 emSumEt->Draw("same");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");


 if(doRatio){

 cSumEt->cd();
 pDSumEtRatio->SetTopMargin(0.05);
 pDSumEtRatio->Draw();
 pDSumEtRatio->cd();
 hwSumEt->Divide(emSumEt);
 hwSumEt->GetYaxis()->SetTitle("Ratio HW/EM");
 hwSumEt->GetYaxis()->SetTitleSize(0.09);
 hwSumEt->GetYaxis()->SetLabelSize(0.05);
 hwSumEt->GetXaxis()->SetLabelSize(0.07);
 hwSumEt->GetXaxis()->SetTitleSize(0.0);
 hwSumEt->GetYaxis()->SetTitleOffset(0.35);
 hwSumEt->SetMinimum(0.8);
 hwSumEt->SetMaximum(1.2);
 hwSumEt->Draw("p");
 cSumEt->cd();
 pInvDSumEtRatio->Draw();
 pInvDSumEtRatio->cd();
 unity->Draw();
 pDSumEtRatio->Update();

 }

 if(doSave)  cSumEt->SaveAs("compHwEmu/DemuxSums/DemSumEt.pdf","pdf");


 //--- demux sum met ---//

 TCanvas* cSumMet = new TCanvas("cSumMet","SumMet");

 TPad* pDSumMet = new TPad("pSumMet","pSumMet",0,0.0,1,1); 
 if(doRatio) pDSumMet = new TPad("pSumMet","pSumMet",0,0.3,1,1); 

 TPad* pDSumMetRatio = new TPad("pSumMetratio","pSumMetratio",0,0,1,0.3);
 
 TPad* pInvDSumMetRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDSumMetRatio->SetFillStyle(0);

 leg = new TLegend(0.65,0.75,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwSumMet,"Upgrade hardware", "p");
 leg->AddEntry(emSumMet,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwSumMet->Rebin(10);
 emSumMet->Rebin(10);

 hwSumMet->SetStats(0);
 hwSumMet->SetMarkerStyle(21);
 hwSumMet->SetMarkerColor(1);
 hwSumMet->SetMarkerSize(0.4);
 emSumMet->SetLineColor(kRed);
 //hwSumMet->GetXaxis()->SetRange(0,50);
 hwSumMet->GetXaxis()->SetTitle("Level-1 Trigger iMET");
 hwSumMet->GetYaxis()->SetTitle("Number of events");
 hwSumMet->GetYaxis()->SetTitleSize(0.05);
 hwSumMet->GetYaxis()->SetTitleOffset(0.77);
 hwSumMet->GetXaxis()->SetTitleSize(0.04);
 hwSumMet->GetXaxis()->SetTitleOffset(1.0);
 pDSumMet->SetBottomMargin(0.12);
 pDSumMet->Draw();
 pDSumMet->cd();

 TH1D* DSumMetRatio = (TH1D*)hwSumMet->DrawCopy("p");
 DSumMetRatio->SetMinimum(0);
 emSumMet->Draw("same");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");


 if(doRatio){

 cSumMet->cd();
 pDSumMetRatio->SetTopMargin(0.05);
 pDSumMetRatio->Draw();
 pDSumMetRatio->cd();
 hwSumMet->Divide(emSumMet);
 hwSumMet->GetYaxis()->SetTitle("Ratio HW/EM");
 hwSumMet->GetYaxis()->SetTitleSize(0.09);
 hwSumMet->GetYaxis()->SetLabelSize(0.05);
 hwSumMet->GetXaxis()->SetLabelSize(0.07);
 hwSumMet->GetXaxis()->SetTitleSize(0.0);
 hwSumMet->GetYaxis()->SetTitleOffset(0.35);
 hwSumMet->SetMinimum(0.8);
 hwSumMet->SetMaximum(1.2);
 hwSumMet->Draw("p");
 cSumMet->cd();
 pInvDSumMetRatio->Draw();
 pInvDSumMetRatio->cd();
 unity->Draw();
 pDSumMetRatio->Update();

 }

 if(doSave) cSumMet->SaveAs("compHwEmu/DemuxSums/DemSumMet.pdf","pdf");


 //--- demux sum ht ---//

 TCanvas* cSumHt = new TCanvas("cSumHt","SumHt");

 TPad* pDSumHt = new TPad("pSumHt","pSumHt",0,0.0,1,1); 
 if(doRatio) pDSumHt = new TPad("pSumHt","pSumHt",0,0.3,1,1); 

 TPad* pDSumHtRatio = new TPad("pSumHtratio","pSumHtratio",0,0,1,0.3);
 
 TPad* pInvDSumHtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDSumHtRatio->SetFillStyle(0);

 leg = new TLegend(0.65,0.75,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwSumHt,"Upgrade hardware", "p");
 leg->AddEntry(emSumHt,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwSumHt->Rebin(40);
 emSumHt->Rebin(40);

 hwSumHt->SetStats(0);
 hwSumHt->SetMarkerStyle(21);
 hwSumHt->SetMarkerColor(1);
 hwSumHt->SetMarkerSize(0.4);
 emSumHt->SetLineColor(kRed);
 //hwSumHt->GetXaxis()->SetRange(0,50);
 hwSumHt->GetXaxis()->SetTitle("Level-1 Trigger iH_{T}");
 hwSumHt->GetYaxis()->SetTitle("Number of events");
 hwSumHt->GetYaxis()->SetTitleSize(0.05);
 hwSumHt->GetYaxis()->SetTitleOffset(0.77);
 hwSumHt->GetXaxis()->SetTitleSize(0.04);
 hwSumHt->GetXaxis()->SetTitleOffset(1.0);
 pDSumHt->SetBottomMargin(0.12);
 pDSumHt->Draw();
 pDSumHt->cd();

 TH1D* DSumHtRatio = (TH1D*)hwSumHt->DrawCopy("p");
 DSumHtRatio->SetMinimum(0);
 emSumHt->Draw("same");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");


 if(doRatio){

 cSumHt->cd();
 pDSumHtRatio->SetTopMargin(0.05);
 pDSumHtRatio->Draw();
 pDSumHtRatio->cd();
 hwSumHt->Divide(emSumHt);
 hwSumHt->GetYaxis()->SetTitle("Ratio HW/EM");
 hwSumHt->GetYaxis()->SetTitleSize(0.09);
 hwSumHt->GetYaxis()->SetLabelSize(0.05);
 hwSumHt->GetXaxis()->SetLabelSize(0.07);
 hwSumHt->GetXaxis()->SetTitleSize(0.0);
 hwSumHt->GetYaxis()->SetTitleOffset(0.35);
 hwSumHt->SetMinimum(0.8);
 hwSumHt->SetMaximum(1.2);
 hwSumHt->Draw("p");
 cSumHt->cd();
 pInvDSumHtRatio->Draw();
 pInvDSumHtRatio->cd();
 unity->Draw();
 pDSumHtRatio->Update();

 }

 if(doSave) cSumHt->SaveAs("compHwEmu/DemuxSums/DemSumHt.pdf","pdf");


 //--- demux sum mht ---//

 TCanvas* cSumMht = new TCanvas("cSumMht","SumMht");

 TPad* pDSumMht = new TPad("pSumMht","pSumMht",0,0.0,1,1); 
 if(doRatio) pDSumMht = new TPad("pSumMht","pSumMht",0,0.3,1,1); 

 TPad* pDSumMhtRatio = new TPad("pSumMhtratio","pSumMhtratio",0,0,1,0.3);
 
 TPad* pInvDSumMhtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDSumMhtRatio->SetFillStyle(0);

 leg = new TLegend(0.65,0.75,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwSumMht,"Upgrade hardware", "p");
 leg->AddEntry(emSumMht,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwSumMht->Rebin(10);
 emSumMht->Rebin(10);

 hwSumMht->SetStats(0);
 hwSumMht->SetMarkerStyle(21);
 hwSumMht->SetMarkerColor(1);
 hwSumMht->SetMarkerSize(0.4);
 emSumMht->SetLineColor(kRed);
 //hwSumMht->GetXaxis()->SetRange(0,20);
 hwSumMht->GetXaxis()->SetTitle("Level-1 Trigger iH_{T}^{miss}");
 hwSumMht->GetYaxis()->SetTitle("Number of events");
 hwSumMht->GetYaxis()->SetTitleSize(0.05);
 hwSumMht->GetYaxis()->SetTitleOffset(0.77);
 hwSumMht->GetXaxis()->SetTitleSize(0.04);
 hwSumMht->GetXaxis()->SetTitleOffset(1.0);
 pDSumMht->SetBottomMargin(0.12);
 pDSumMht->Draw();
 pDSumMht->cd();

 TH1D* DSumMhtRatio = (TH1D*)hwSumMht->DrawCopy("p");
 DSumMhtRatio->SetMinimum(0);
 emSumMht->Draw("same");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");

 if(doRatio){

 cSumMht->cd();
 pDSumMhtRatio->SetTopMargin(0.05);
 pDSumMhtRatio->Draw();
 pDSumMhtRatio->cd();
 hwSumMht->Divide(emSumMht);
 hwSumMht->GetYaxis()->SetTitle("Ratio HW/EM");
 hwSumMht->GetYaxis()->SetTitleSize(0.09);
 hwSumMht->GetYaxis()->SetLabelSize(0.05);
 hwSumMht->GetXaxis()->SetLabelSize(0.07);
 hwSumMht->GetXaxis()->SetTitleSize(0.0);
 hwSumMht->GetYaxis()->SetTitleOffset(0.35);
 hwSumMht->SetMinimum(0.8);
 hwSumMht->SetMaximum(1.2);
 hwSumMht->Draw("p");
 cSumMht->cd();
 pInvDSumMhtRatio->Draw();
 pInvDSumMhtRatio->cd();
 unity->Draw();
 pDSumMhtRatio->Update();

 }

 if(doSave)  cSumMht->SaveAs("compHwEmu/DemuxSums/DemSumMht.pdf","pdf");


 //--- met phi ---//

 TCanvas* cMetPhi = new TCanvas("cMetPhi","MetPhi");

 TPad* pDMetPhi = new TPad("pMetPhi","pMetPhi",0,0.0,1,1); 
 if(doRatio)  pDMetPhi = new TPad("pMetPhi","pMetPhi",0,0.3,1,1); 

 TPad* pDMetPhiRatio = new TPad("pMetPhiratio","pMetPhiratio",0,0,1,0.3);
 
 TPad* pInvDMetPhiRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDMetPhiRatio->SetFillStyle(0);

 leg = new TLegend(0.65,0.75,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMetPhi,"Upgrade hardware", "p");
 leg->AddEntry(emMetPhi,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwMetPhi->Rebin(8);
 emMetPhi->Rebin(8);

 hwMetPhi->SetStats(0);
 hwMetPhi->SetMarkerStyle(21);
 hwMetPhi->SetMarkerColor(1);
 hwMetPhi->SetMarkerSize(0.4);
 emMetPhi->SetLineColor(kRed);
 //hwMetPhi->GetXaxis()->SetRange(0,40);
 hwMetPhi->GetXaxis()->SetTitle("Level-1 Trigger MET i#phi");
 hwMetPhi->GetYaxis()->SetTitle("Number of events");
 hwMetPhi->GetYaxis()->SetTitleSize(0.05);
 hwMetPhi->GetYaxis()->SetTitleOffset(0.77);
 hwMetPhi->GetXaxis()->SetTitleSize(0.04);
 hwMetPhi->GetXaxis()->SetTitleOffset(1.0);
 pDMetPhi->SetBottomMargin(0.12);
 pDMetPhi->Draw();
 pDMetPhi->cd();

 TH1D* DMetPhiRatio = (TH1D*)hwMetPhi->DrawCopy("p");
 DMetPhiRatio->SetMinimum(0);
 emMetPhi->Draw("same");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");


 if(doRatio){

 cMetPhi->cd();
 pDMetPhiRatio->SetTopMargin(0.05);
 pDMetPhiRatio->Draw();
 pDMetPhiRatio->cd();
 hwMetPhi->Divide(emMetPhi);
 hwMetPhi->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMetPhi->GetYaxis()->SetTitleSize(0.09);
 hwMetPhi->GetYaxis()->SetLabelSize(0.05);
 hwMetPhi->GetXaxis()->SetLabelSize(0.07);
 hwMetPhi->GetXaxis()->SetTitleSize(0.0);
 hwMetPhi->GetYaxis()->SetTitleOffset(0.35);
 hwMetPhi->SetMinimum(0.8);
 hwMetPhi->SetMaximum(1.2);
 hwMetPhi->Draw("p");
 cMetPhi->cd();
 pInvDMetPhiRatio->Draw();
 pInvDMetPhiRatio->cd();
 unity->Draw();
 pDMetPhiRatio->Update();

 }

 if(doSave)  cMetPhi->SaveAs("compHwEmu/DemuxSums/DemMetPhi.pdf","pdf");


 //--- mht phi ---//

 TCanvas* cMhtPhi = new TCanvas("cMhtPhi","MhtPhi");

 TPad* pDMhtPhi = new TPad("pMhtPhi","pMhtPhi",0,0.0,1,1); 
 if(doRatio) pDMhtPhi = new TPad("pMhtPhi","pMhtPhi",0,0.3,1,1); 

 TPad* pDMhtPhiRatio = new TPad("pMhtPhiratio","pMhtPhiratio",0,0,1,0.3);
 
 TPad* pInvDMhtPhiRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDMhtPhiRatio->SetFillStyle(0);

 leg = new TLegend(0.65,0.75,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMhtPhi,"Upgrade hardware", "p");
 leg->AddEntry(emMhtPhi,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwMhtPhi->Rebin(8);
 emMhtPhi->Rebin(8);

 hwMhtPhi->SetStats(0);
 hwMhtPhi->SetMarkerStyle(21);
 hwMhtPhi->SetMarkerColor(1);
 hwMhtPhi->SetMarkerSize(0.4);
 emMhtPhi->SetLineColor(kRed);
 //hwMhtPhi->GetXaxis()->SetRange(0,40);
 hwMhtPhi->GetXaxis()->SetTitle("Level-1 Trigger iH_{T}^{miss} i#phi");
 hwMhtPhi->GetYaxis()->SetTitle("Number of events");
 hwMhtPhi->GetYaxis()->SetTitleSize(0.05);
 hwMhtPhi->GetYaxis()->SetTitleOffset(0.77);
 hwMhtPhi->GetXaxis()->SetTitleSize(0.04);
 hwMhtPhi->GetXaxis()->SetTitleOffset(1.0);
 pDMhtPhi->SetBottomMargin(0.12);
 pDMhtPhi->Draw();
 pDMhtPhi->cd();
 
 TH1D* DMhtPhiRatio = (TH1D*)hwMhtPhi->DrawCopy("p");
 DMhtPhiRatio->SetMinimum(0);
 emMhtPhi->Draw("same");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");


 if(doRatio){

 cMhtPhi->cd();
 pDMhtPhiRatio->SetTopMargin(0.05);
 pDMhtPhiRatio->Draw();
 pDMhtPhiRatio->cd();
 hwMhtPhi->Divide(emMhtPhi);
 hwMhtPhi->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMhtPhi->GetYaxis()->SetTitleSize(0.09);
 hwMhtPhi->GetYaxis()->SetLabelSize(0.05);
 hwMhtPhi->GetXaxis()->SetLabelSize(0.07);
 hwMhtPhi->GetXaxis()->SetTitleSize(0.0);
 hwMhtPhi->GetYaxis()->SetTitleOffset(0.35);
 hwMhtPhi->SetMinimum(0.8);
 hwMhtPhi->SetMaximum(1.2);
 hwMhtPhi->Draw("p");
 cMhtPhi->cd();
 pInvDMhtPhiRatio->Draw();
 pInvDMhtPhiRatio->cd();
 unity->Draw();
 pDMhtPhiRatio->Update();

 }

 if(doSave)  cMhtPhi->SaveAs("compHwEmu/DemuxSums/DemMhtPhi.pdf","pdf");

 }

}

 

 //--- sorts ---//

if(doSorts){

 TCanvas* cSortMP = new TCanvas("cSortMP","SortMP");

 TPad* pSortMP = new TPad("pSortMP","pSortMP",0,0.0,1,1); 
 if(doRatio) pSortMP = new TPad("pSortMP","pSortMP",0,0.3,1,1); 

 TPad* pSortMPRatio = new TPad("pSortMPratio","pSortMPratio",0,0,1,0.3);
 
 TPad* pInvSortMPRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSortMPRatio->SetFillStyle(0);
  
 leg = new TLegend(0.6,0.75,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwSortMP,"Upgrade hardware", "p");
 leg->AddEntry(emSortMP,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 hwSortMP->SetStats(0);
 hwSortMP->SetMarkerStyle(21);
 hwSortMP->SetMarkerColor(1);
 hwSortMP->SetMarkerSize(0.4);
 emSortMP->SetLineColor(kRed);
 //hwSortMP->GetXaxis()->SetRange(40,80);
 hwSortMP->GetXaxis()->SetTitle("Level-1 Trigger iH{T}");
 hwSortMP->GetYaxis()->SetTitle("Number of events");
 hwSortMP->GetYaxis()->SetTitleSize(0.05);
 hwSortMP->GetYaxis()->SetTitleOffset(0.77);
 hwSortMP->GetXaxis()->SetTitleSize(0.04);
 hwSortMP->GetXaxis()->SetTitleOffset(1.0);
 pSortMP->SetBottomMargin(0.12);
 pSortMP->Draw();
 pSortMP->cd();


 TH1D* SortMPRatio = (TH1D*)hwSortMP->DrawCopy("p");
 SortMPRatio->SetMinimum(0);
 emSortMP->Draw("same");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");
 

 if(doRatio){

 cSortMP->cd();
 pSortMPRatio->SetTopMargin(0.05);
 pSortMPRatio->Draw();
 pSortMPRatio->cd();
 hwSortMP->Divide(emSortMP);
 hwSortMP->GetYaxis()->SetTitle("Ratio HW/EM");
 hwSortMP->GetYaxis()->SetTitleSize(0.09);
 hwSortMP->GetYaxis()->SetLabelSize(0.05);
 hwSortMP->GetXaxis()->SetLabelSize(0.07);
 hwSortMP->GetXaxis()->SetTitleSize(0.0);
 hwSortMP->GetYaxis()->SetTitleOffset(0.35);
 hwSortMP->SetMinimum(0.8);
 hwSortMP->SetMaximum(1.2);
 hwSortMP->Draw("p");
 cSortMP->cd();
 pInvSortMPRatio->Draw();
 pInvSortMPRatio->cd();
 unity->Draw();
 pSortMPRatio->Update();

 }

 if(doSave)  cSortMP->SaveAs("compHwEmu/Sorts/MPSort.pdf","pdf");


 TCanvas* cSort = new TCanvas("cSort","Sort");

 TPad* pDSort = new TPad("pSort","pSort",0,0.0,1,1); 
 if(doRatio) pDSort = new TPad("pSort","pSort",0,0.3,1,1); 

 TPad* pDSortRatio = new TPad("pSortratio","pSortratio",0,0,1,0.3);
 
 TPad* pInvDSortRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDSortRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.75,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwSort,"Upgrade hardware", "p");
 leg->AddEntry(emSort,"Upgrade emulator", "l");
 leg->SetBorderSize(0);
 leg->SetFillStyle(0);
 //hwSort->Rebin(3);
 //emSort->Rebin(3);

 hwSort->SetStats(0);
 hwSort->SetMarkerStyle(21);
 hwSort->SetMarkerColor(1);
 hwSort->SetMarkerSize(0.4);
 emSort->SetLineColor(kRed);
 //hwSort->GetXaxis()->SetRange(40,80);
 hwSort->GetXaxis()->SetTitle("Level-1 Trigger iH_{T}");
 hwSort->GetYaxis()->SetTitle("Number of events");
 hwSort->GetYaxis()->SetTitleSize(0.05);
 hwSort->GetYaxis()->SetTitleOffset(0.77);
 hwSort->GetXaxis()->SetTitleSize(0.04);
 hwSort->GetXaxis()->SetTitleOffset(1.0);
 pDSort->SetBottomMargin(0.12);
 pDSort->Draw();
 pDSort->cd();

 TH1D* DSortRatio = (TH1D*)hwSort->DrawCopy("p");
 DSortRatio->SetMinimum(0);
 emSort->Draw("same");
 leg->Draw();
 n2.DrawLatex(0.7, 0.65, "CMS");
 n3.DrawLatex(0.7, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
 n3.DrawLatex(0.7, 0.55, "Preliminary");
 n4.DrawLatex(0.7, 0.45, "Single Muon stream");
 

 if(doRatio){

 cSort->cd();
 pDSortRatio->SetTopMargin(0.05);
 pDSortRatio->Draw();
 pDSortRatio->cd();
 hwSort->Divide(emSort);
 hwSort->GetYaxis()->SetTitle("Ratio HW/EM");
 hwSort->GetYaxis()->SetTitleSize(0.09);
 hwSort->GetYaxis()->SetLabelSize(0.05);
 hwSort->GetXaxis()->SetLabelSize(0.07);  
 hwSort->GetXaxis()->SetTitleSize(0.0);
 hwSort->GetYaxis()->SetTitleOffset(0.35);
 hwSort->SetMinimum(0.8);
 hwSort->SetMaximum(1.2);
 hwSort->Draw("p");
 cSort->cd();
 pInvDSortRatio->Draw();
 pInvDSortRatio->cd();
 unity->Draw();
 pDSortRatio->Update();

 }

 if(doSave) cSort->SaveAs("compHwEmu/Sorts/DemuxSort.pdf","pdf");


 }

 }
 
