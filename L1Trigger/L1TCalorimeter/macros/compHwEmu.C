void compHwEmu(){

  TFile* inFileHw = new TFile("l1tCalo_2016_histosHw.root");
  TFile* inFileEm = new TFile("l1tCalo_2016_histosEm.root");


  //TH2D* towEtaPhi = (TH2D*)inFile->Get("rawPlots/tower/etaphi");

  //TH2D* rawEtaPhi = (TH2D*)inFile->Get("rawPlots/jet/etaphi");
  //TH2D* emuEtaPhi = (TH2D*)inFile->Get("simPlots/jet/etaphi");

  // Jets
  
  TH1D* hwMPJetEt = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/mpjet/et");
  TH1D* emMPJetEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpjet/et");
  TH1D* hwJetEt = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/jet/et");
  TH1D* emJetEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/jet/et");

  TH1D* hwMPJetEta = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/mpjet/eta");
  TH1D* emMPJetEta = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpjet/eta");
  TH1D* hwJetEta = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/jet/eta");
  TH1D* emJetEta = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/jet/eta");

  TH1D* hwMPJetPhi = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/mpjet/phi");
  TH1D* emMPJetPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpjet/phi");
  TH1D* hwJetPhi = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/jet/phi");
  TH1D* emJetPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/jet/phi");

  // MP sums
  
  TH1D* hwMPSumEt = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/mpsumet/et");
  TH1D* emMPSumEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsumet/et");

  TH1D* hwMPSumEtx = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/mpsummetx/et");
  TH1D* emMPSumEtx = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummetx/et");

  TH1D* hwMPSumEty = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/mpsummety/et");
  TH1D* emMPSumEty = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummety/et");

  TH1D* hwMPSumHt = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/mpsumht/et");
  TH1D* emMPSumHt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsumht/et");

  TH1D* hwMPSumHtx = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/mpsummhtx/et");
  TH1D* emMPSumHtx = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummhtx/et");

  TH1D* hwMPSumHty = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/mpsummhty/et");
  TH1D* emMPSumHty = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/mpsummhty/et");
  
  // Demux sums
  
  TH1D* hwSumEt = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/sumet/et");
  TH1D* emSumEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumet/et");

  TH1D* hwSumMet = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/summet/et");
  TH1D* emSumMet = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summet/et");

  TH1D* hwSumHt = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/sumht/et");
  TH1D* emSumHt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sumht/et");

  TH1D* hwSumMht = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/summht/et");
  TH1D* emSumMht = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summht/et");
  
  // Sum phi's
  
  TH1D* hwMetPhi = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/summet/phi");
  TH1D* emMetPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summet/phi");

  TH1D* hwMhtPhi = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/summht/phi");
  TH1D* emMhtPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/summht/phi");
  

  // Sorts

  TH1D* hwSortMP = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/sortMP");
  TH1D* emSortMP = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sortMP");

  TH1D* hwSort = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/sort");
  TH1D* emSort = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/sort");

  // EG Et

  TH1D* hwEgEt = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/eg/et");
  TH1D* emEgEt = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/eg/et");


  // EG eta

  TH1D* hwEgEta = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/eg/eta");
  TH1D* emEgEta = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/eg/eta");

  // EG phi

  TH1D* hwEgPhi = (TH1D*)inFileHw->Get("l1tStage2CaloAnalyzer/eg/phi");
  TH1D* emEgPhi = (TH1D*)inFileEm->Get("l1tStage2CaloAnalyzer/eg/phi");


  ////////////////////////////////////////////////////////////////////////////////////

  TLine* unity = new TLine(0.1,0.525,0.9,0.525);
  unity->SetLineColor(kBlue);

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


  //--- jet et ---//

  TCanvas* cJetEt = new TCanvas("cJetEt","JetEt");

  TPad* pJetEt = new TPad("pJetEt","pJetEt",0,0.3,1,1); 
  TPad* pJetEtRatio = new TPad("pJetEtratio","pJetEtratio",0,0,1,0.3);
  
  TPad* pInvJetEtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
  pInvJetEtRatio->SetFillStyle(0);

  TLegend* leg = new TLegend(0.6,0.65,0.85,0.85);
  leg->SetFillColor(0);
  leg->AddEntry(hwMPJetEt,"Hardware MP", "p");//"l");
  //leg->AddEntry(hwJetEt,"Hardware Demux", "p");
  leg->AddEntry(emMPJetEt,"Emulator MP", "l");
  //leg->AddEntry(emJetEt,"Emulator Demux", "p");
  
  hwMPJetEt->Rebin(10);
  emMPJetEt->Rebin(10);

  hwMPJetEt->SetStats(0);
  //hwMPJetEt->SetLineColor(kBlue);
  hwMPJetEt->SetMarkerStyle(21);
  hwMPJetEt->SetMarkerColor(1);
  hwMPJetEt->SetMarkerSize(0.4);
  emMPJetEt->SetLineColor(kRed);
  hwMPJetEt->GetXaxis()->SetTitle("Jet iET");
  hwMPJetEt->GetYaxis()->SetTitle("# Jets");
  hwMPJetEt->GetYaxis()->SetTitleSize(0.07);
  hwMPJetEt->GetYaxis()->SetTitleOffset(0.48);
  hwMPJetEt->GetXaxis()->SetTitleSize(0.04);
  hwMPJetEt->GetXaxis()->SetTitleOffset(0.9);
  pJetEt->SetBottomMargin(0.08);
  pJetEt->Draw();
  pJetEt->cd();

  TH1D* JetEtRatio = (TH1D*)hwMPJetEt->DrawCopy("p");
  JetEtRatio->SetMinimum(0);
  emMPJetEt->Draw("same");//"");
  leg->Draw();
  cJetEt->cd();
  pJetEtRatio->SetTopMargin(0.05);
  pJetEtRatio->Draw();
  pJetEtRatio->cd();
  hwMPJetEt->Divide(emMPJetEt);
  hwMPJetEt->GetYaxis()->SetTitle("Ratio HW/EM");
  hwMPJetEt->GetYaxis()->SetTitleSize(0.09);
  hwMPJetEt->GetYaxis()->SetLabelSize(0.07);
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
  cJetEt->SaveAs("compHwEmu/Jets/JetEt.png");

  


 TCanvas* cDJetEt = new TCanvas("cDJetEt","DJetEt");

 TPad* pDJetEt = new TPad("pJetEt","pJetEt",0,0.3,1,1); 
 TPad* pDJetEtRatio = new TPad("pJetEtratio","pJetEtratio",0,0,1,0.3);
 
 TPad* pInvDJetEtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDJetEtRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwJetEt,"Hardware Demux", "p");
 leg->AddEntry(emJetEt,"Emulator Demux", "l");

 hwJetEt->Rebin(10);
 emJetEt->Rebin(10);

 hwJetEt->SetStats(0);
 hwJetEt->SetMarkerStyle(21);
 hwJetEt->SetMarkerColor(1);
 hwJetEt->SetMarkerSize(0.4);
 emJetEt->SetLineColor(kRed);
 hwJetEt->GetXaxis()->SetTitle("Jet iET");
 hwJetEt->GetYaxis()->SetTitle("# Jets");
 hwJetEt->GetYaxis()->SetTitleSize(0.07);
 hwJetEt->GetYaxis()->SetTitleOffset(0.48);
 hwJetEt->GetXaxis()->SetTitleSize(0.04);
 hwJetEt->GetXaxis()->SetTitleOffset(0.9);
 pDJetEt->SetBottomMargin(0.08);
 pDJetEt->Draw();
 pDJetEt->cd();
 
 TH1D* DJetEtRatio = (TH1D*)hwJetEt->DrawCopy("p");
 DJetEtRatio->SetMinimum(0);
 emJetEt->Draw("same");
 leg->Draw();
 cDJetEt->cd();
 pDJetEtRatio->SetTopMargin(0.05);
 pDJetEtRatio->Draw();
 pDJetEtRatio->cd();
 hwJetEt->Divide(emJetEt);
 hwJetEt->GetYaxis()->SetTitle("Ratio HW/EM");
 hwJetEt->GetYaxis()->SetTitleSize(0.09);
 hwJetEt->GetYaxis()->SetLabelSize(0.07);
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
 cDJetEt->SaveAs("compHwEmu/DemuxJets/JetEt.png");



 // //--- jet eta ---//

 TCanvas* cJetEta = new TCanvas("cJetEta","JetEta");

 TPad* pJetEta = new TPad("pJetEt","pJetEt",0,0.3,1,1); 
 TPad* pJetEtaRatio = new TPad("pJetEtratio","pJetEtratio",0,0,1,0.3);
  
 TPad* pInvJetEtaRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvJetEtaRatio->SetFillStyle(0);

 leg = new TLegend(0.4,0.65,0.65,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPJetEta,"Hardware MP", "p");//"l");
 //leg->AddEntry(hwJetEta,"Hardware Demux", "p");
 leg->AddEntry(emMPJetEta,"Emulator MP", "l");
 //leg->AddEntry(emJetEta,"Emulator Demux", "p");


 hwMPJetEta->SetStats(0);
 //hwMPJetEta->SetLineColor(kBlue);
 hwMPJetEta->SetMarkerStyle(21);
 hwMPJetEta->SetMarkerColor(1);
 hwMPJetEta->SetMarkerSize(0.4);
 emMPJetEta->SetLineColor(kRed);
 emMPJetEta->SetMarkerStyle(20);
 emMPJetEta->SetMarkerColor(kRed);
 emMPJetEta->SetMarkerSize(0.4);
 hwMPJetEta->GetXaxis()->SetRange(10,74);
 hwMPJetEta->GetXaxis()->SetTitle("Jet i#eta");
 hwMPJetEta->GetYaxis()->SetTitle("# Jets");
 hwMPJetEta->GetYaxis()->SetTitleSize(0.07);
 hwMPJetEta->GetYaxis()->SetTitleOffset(0.48);
 hwMPJetEta->GetXaxis()->SetTitleSize(0.04);
 hwMPJetEta->GetXaxis()->SetTitleOffset(0.9);
 pJetEta->SetBottomMargin(0.08);
 pJetEta->Draw();
 pJetEta->cd();

 TH1D* JetEtaRatio = (TH1D*)hwMPJetEta->DrawCopy("p");
 JetEtaRatio->SetMinimum(0);
 emMPJetEta->Draw("same");//"");
 leg->Draw();
 cJetEta->cd();
 pJetEtaRatio->SetTopMargin(0.05);
 pJetEtaRatio->Draw();
 pJetEtaRatio->cd();
 hwMPJetEta->Divide(emMPJetEta);
 hwMPJetEta->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPJetEta->GetYaxis()->SetTitleSize(0.09);
 hwMPJetEta->GetYaxis()->SetLabelSize(0.07);
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
 cJetEta->SaveAs("compHwEmu/Jets/JetEta.png");
 //c1->Print("compHwEmu.pdf","pdf");


 TCanvas* cDJetEta = new TCanvas("cDJetEta","DJetEta");
 TPad* pDJetEta = new TPad("pJetEta","pJetEta",0,0.3,1,1); 
 TPad* pDJetEtaRatio = new TPad("pJetEtaratio","pJetEtaratio",0,0,1,0.3);
 
 TPad* pInvDJetEtaRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDJetEtaRatio->SetFillStyle(0);

 leg = new TLegend(0.4,0.65,0.65,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwJetEta,"Hardware Demux", "p");
 leg->AddEntry(emJetEta,"Emulator Demux", "l");

 hwJetEta->SetStats(0);
 hwJetEta->SetMarkerStyle(21);
 hwJetEta->SetMarkerColor(1);
 hwJetEta->SetMarkerSize(0.4);
 emJetEta->SetLineColor(kRed);
 hwJetEta->GetXaxis()->SetTitle("Jet i#eta");
 hwJetEta->GetXaxis()->SetRange(82,146);
 hwJetEta->GetYaxis()->SetTitle("# Jets");
 hwJetEta->GetYaxis()->SetTitleSize(0.07);
 hwJetEta->GetYaxis()->SetTitleOffset(0.48);
 hwJetEta->GetXaxis()->SetTitleSize(0.04);
 hwJetEta->GetXaxis()->SetTitleOffset(0.9);
 pDJetEta->SetBottomMargin(0.08);
 pDJetEta->Draw();
 pDJetEta->cd();
 
 TH1D* DJetEtaRatio = (TH1D*)hwJetEta->DrawCopy("p");
 DJetEtaRatio->SetMinimum(0);
 emJetEta->Draw("same");
 leg->Draw();
 cDJetEta->cd();
 pDJetEtaRatio->SetTopMargin(0.05);
 pDJetEtaRatio->Draw();
 pDJetEtaRatio->cd();
 hwJetEta->Divide(emJetEta);
 hwJetEta->GetYaxis()->SetTitle("Ratio HW/EM");
 hwJetEta->GetYaxis()->SetTitleSize(0.09);
 hwJetEta->GetYaxis()->SetLabelSize(0.07);
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
 cDJetEta->SaveAs("compHwEmu/DemuxJets/JetEta.png");


//--- jet phi ---//

 TCanvas* cJetPhi = new TCanvas("cJetPhi","JetPhi");

 TPad* pJetPhi = new TPad("pJetPhi","pJetPhi",0,0.3,1,1); 
 TPad* pJetPhiRatio = new TPad("pJetPhiratio","pJetPhiratio",0,0,1,0.3);
 
 TPad* pInvJetPhiRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvJetPhiRatio->SetFillStyle(0);


 leg = new TLegend(0.75,0.75,0.9,0.9);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPJetPhi,"Hardware MP", "p");//"l");
 //leg->AddEntry(hwJetPhi,"Hardware Demux", "p");
 leg->AddEntry(emMPJetPhi,"Emulator MP", "l");
 //leg->AddEntry(emJetPhi,"Emulator Demux", "p");

 hwMPJetPhi->SetStats(0);
 //hwMPJetPhi->SetLineColor(kBlue);
 hwMPJetPhi->SetMarkerStyle(21);
 hwMPJetPhi->SetMarkerColor(1);
 hwMPJetPhi->SetMarkerSize(0.4);
 emMPJetPhi->SetLineColor(kRed);
 emMPJetPhi->SetMarkerStyle(20);
 emMPJetPhi->SetMarkerColor(kRed);
 emMPJetPhi->SetMarkerSize(0.4);
 emMPJetPhi->GetXaxis()->SetRange(0,100);
 hwMPJetPhi->GetXaxis()->SetTitle("Jet i#phi");
 hwMPJetPhi->GetYaxis()->SetTitle("# Jets");
 hwMPJetPhi->GetYaxis()->SetTitleSize(0.07);
 hwMPJetPhi->GetYaxis()->SetTitleOffset(0.48);
 hwMPJetPhi->GetXaxis()->SetTitleSize(0.04);
 hwMPJetPhi->GetXaxis()->SetTitleOffset(0.9);
 pJetPhi->SetBottomMargin(0.08);
 pJetPhi->Draw();
 pJetPhi->cd();

 TH1D* JetPhiRatio = (TH1D*)hwMPJetPhi->DrawCopy("p");
 JetPhiRatio->SetMinimum(0);
 emMPJetPhi->Draw("same");//"");
 leg->Draw();
 cJetPhi->cd();
 pJetPhiRatio->SetTopMargin(0.05);
 pJetPhiRatio->Draw();
 pJetPhiRatio->cd();
 hwMPJetPhi->Divide(emMPJetPhi);
 hwMPJetPhi->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPJetPhi->GetYaxis()->SetTitleSize(0.09);
 hwMPJetPhi->GetYaxis()->SetLabelSize(0.07);
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
 cJetPhi->SaveAs("compHwEmu/Jets/JetPhi.png");
 //c1->Print("compHwEmu.pdf","pdf");


 TCanvas* cDJetPhi = new TCanvas("cDJetPhi","DJetPhi");

 TPad* pDJetPhi = new TPad("pJetPhi","pJetPhi",0,0.3,1,1); 
 TPad* pDJetPhiRatio = new TPad("pJetPhiratio","pJetPhiratio",0,0,1,0.3);
  
 TPad* pInvDJetPhiRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDJetPhiRatio->SetFillStyle(0);

 leg = new TLegend(0.75,0.75,0.9,0.9);
 leg->SetFillColor(0);
 leg->AddEntry(hwJetPhi,"Hardware Demux", "p");
 leg->AddEntry(emJetPhi,"Emulator Demux", "l");

 hwJetPhi->SetStats(0);
 hwJetPhi->SetMarkerStyle(21);
 hwJetPhi->SetMarkerColor(1);
 hwJetPhi->SetMarkerSize(0.4);
 emJetPhi->SetLineColor(kRed);
 hwJetPhi->GetXaxis()->SetTitle("Jet i#phi");
 hwJetPhi->GetXaxis()->SetRange(0,73);
 hwJetPhi->GetYaxis()->SetTitle("# Jets");
 hwJetPhi->GetYaxis()->SetTitleSize(0.07);
 hwJetPhi->GetYaxis()->SetTitleOffset(0.48);
 hwJetPhi->GetXaxis()->SetTitleSize(0.04);
 hwJetPhi->GetXaxis()->SetTitleOffset(0.9);
 pDJetPhi->SetBottomMargin(0.08);
 pDJetPhi->Draw();
 pDJetPhi->cd();

 TH1D* DJetPhiRatio = (TH1D*)hwJetPhi->DrawCopy("p");
 DJetPhiRatio->SetMinimum(0);
 emJetPhi->Draw("same");//"");
 leg->Draw();
 cDJetPhi->cd();
 pDJetPhiRatio->SetTopMargin(0.05);
 pDJetPhiRatio->Draw();
 pDJetPhiRatio->cd();
 hwJetPhi->Divide(emJetPhi);
 hwJetPhi->GetYaxis()->SetTitle("Ratio HW/EM");
 hwJetPhi->GetYaxis()->SetTitleSize(0.09);
 hwJetPhi->GetYaxis()->SetLabelSize(0.07);
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
 cDJetPhi->SaveAs("compHwEmu/DemuxJets/JetPhi.png");


//--- MP sum et ---//

 TCanvas* cMPSumEt = new TCanvas("cMPSumEt","MPSumEt");

 TPad* pSumEt = new TPad("pSumEt","pSumEt",0,0.3,1,1); 
 TPad* pSumEtRatio = new TPad("pSumEtratio","pSumEtratio",0,0,1,0.3);

 TPad* pInvSumEtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSumEtRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPSumEt,"Hardware MP", "p");
 leg->AddEntry(emMPSumEt,"Emulator MP", "l");

 hwMPSumEt->Rebin(50);
 emMPSumEt->Rebin(50);

 hwMPSumEt->SetStats(0);
 hwMPSumEt->SetMarkerStyle(21);
 hwMPSumEt->SetMarkerColor(1);
 hwMPSumEt->SetMarkerSize(0.4);
 emMPSumEt->SetLineColor(kRed);
 hwMPSumEt->GetXaxis()->SetTitle("iET");
 hwMPSumEt->GetYaxis()->SetTitle("# Events");
 hwMPSumEt->GetYaxis()->SetTitleSize(0.07);
 hwMPSumEt->GetYaxis()->SetTitleOffset(0.48);
 hwMPSumEt->GetXaxis()->SetTitleSize(0.04);
 hwMPSumEt->GetXaxis()->SetTitleOffset(0.9);
 pSumEt->SetBottomMargin(0.08);
 pSumEt->Draw();
 pSumEt->cd();
 
 TH1D* SumEtRatio = (TH1D*)hwMPSumEt->DrawCopy("p");
 SumEtRatio->SetMinimum(0);
 emMPSumEt->Draw("same");//""); 
 leg->Draw();
 cMPSumEt->cd();
 pSumEtRatio->SetTopMargin(0.05);
 pSumEtRatio->Draw();
 pSumEtRatio->cd();
 hwMPSumEt->Divide(emMPSumEt);
 hwMPSumEt->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPSumEt->GetYaxis()->SetTitleSize(0.09);
 hwMPSumEt->GetYaxis()->SetLabelSize(0.07);
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
 cMPSumEt->SaveAs("compHwEmu/MPSums/MPSumEt.png");


 // //--- MP sum etx ---//

 TCanvas* cMPSumEtx = new TCanvas("cMPSumEtx","MPSumEtx");

 TPad* pSumEtx = new TPad("pSumEtx","pSumEtx",0,0.3,1,1); 
 TPad* pSumEtxRatio = new TPad("pSumEtxratio","pSumEtxratio",0,0,1,0.3);
 
 TPad* pInvSumEtxRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSumEtxRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPSumEtx,"Hardware MP", "p");
 leg->AddEntry(emMPSumEtx,"Emulator MP", "l");

 hwMPSumEtx->Rebin(20);
 emMPSumEtx->Rebin(20);

 hwMPSumEtx->SetStats(0);
 hwMPSumEtx->SetMarkerStyle(21);
 hwMPSumEtx->SetMarkerColor(1);
 hwMPSumEtx->SetMarkerSize(0.4);
 emMPSumEtx->SetLineColor(kRed);
 hwMPSumEtx->GetXaxis()->SetTitle("iETx");
 hwMPSumEtx->GetYaxis()->SetTitle("# Events");
 hwMPSumEtx->GetYaxis()->SetTitleSize(0.07);
 hwMPSumEtx->GetYaxis()->SetTitleOffset(0.48);
 hwMPSumEtx->GetXaxis()->SetTitleSize(0.04);
 hwMPSumEtx->GetXaxis()->SetTitleOffset(0.9);
 pSumEtx->SetBottomMargin(0.08);
 pSumEtx->Draw();
 pSumEtx->cd();
 
 TH1D* SumEtxRatio = (TH1D*)hwMPSumEtx->DrawCopy("p");
 SumEtxRatio->SetMinimum(0);
 emMPSumEtx->Draw("same");//"");
 leg->Draw();
 cMPSumEtx->cd();
 pSumEtxRatio->SetTopMargin(0.05);
 pSumEtxRatio->Draw();
 pSumEtxRatio->cd();
 hwMPSumEtx->Divide(emMPSumEtx);
 hwMPSumEtx->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPSumEtx->GetYaxis()->SetTitleSize(0.09);
 hwMPSumEtx->GetYaxis()->SetLabelSize(0.07);
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
 
 cMPSumEtx->SaveAs("compHwEmu/MPSums/MPSumEtx.png");


 // //--- MP sum ety ---//

 TCanvas* cMPSumEty = new TCanvas("cMPSumEty","MPSumEty");

 TPad* pSumEty = new TPad("pSumEty","pSumEty",0,0.3,1,1); 
 TPad* pSumEtyRatio = new TPad("pSumEtyratio","pSumEtyratio",0,0,1,0.3);
  
 TPad* pInvSumEtyRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSumEtyRatio->SetFillStyle(0);


 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPSumEty,"Hardware MP", "p");
 leg->AddEntry(emMPSumEty,"Emulator MP", "l");

 hwMPSumEty->Rebin(20);
 emMPSumEty->Rebin(20);

 hwMPSumEty->SetStats(0);
 hwMPSumEty->SetMarkerStyle(21);
 hwMPSumEty->SetMarkerColor(1);
 hwMPSumEty->SetMarkerSize(0.4);
 emMPSumEty->SetLineColor(kRed);
 hwMPSumEty->GetXaxis()->SetTitle("iETy");
 hwMPSumEty->GetYaxis()->SetTitle("# Events");
 hwMPSumEty->GetYaxis()->SetTitleSize(0.07);
 hwMPSumEty->GetYaxis()->SetTitleOffset(0.48);
 hwMPSumEty->GetXaxis()->SetTitleSize(0.04);
 hwMPSumEty->GetXaxis()->SetTitleOffset(0.9);
 pSumEty->SetBottomMargin(0.08);
 pSumEty->Draw();
 pSumEty->cd();
 
 TH1D* SumEtyRatio = (TH1D*)hwMPSumEty->DrawCopy("p");
 SumEtyRatio->SetMinimum(0);
 emMPSumEty->Draw("same");//"");
 leg->Draw();
 cMPSumEty->cd();
 pSumEtyRatio->SetTopMargin(0.05);
 pSumEtyRatio->Draw();
 pSumEtyRatio->cd();
 hwMPSumEty->Divide(emMPSumEty);
 hwMPSumEty->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPSumEty->GetYaxis()->SetTitleSize(0.09);
 hwMPSumEty->GetYaxis()->SetLabelSize(0.07);
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
 cMPSumEty->SaveAs("compHwEmu/MPSums/MPSumEty.png");


 //--- MP sum ht ---//

 TCanvas* cMPSumHt = new TCanvas("cMPSumHt","MPSumHt");

 TPad* pSumHt = new TPad("pSumHt","pSumHt",0,0.3,1,1); 
 TPad* pSumHtRatio = new TPad("pSumHtratio","pSumHtratio",0,0,1,0.3);
 
 TPad* pInvSumHtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSumHtRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPSumHt,"Hardware MP", "p");
 leg->AddEntry(emMPSumHt,"Emulator MP", "l");

 hwMPSumHt->Rebin(50);
 emMPSumHt->Rebin(50);

 hwMPSumHt->SetStats(0);
 hwMPSumHt->SetMarkerStyle(21);
 hwMPSumHt->SetMarkerColor(1);
 hwMPSumHt->SetMarkerSize(0.4);
 emMPSumHt->SetLineColor(kRed);
 hwMPSumHt->GetXaxis()->SetTitle("iHT");
 hwMPSumHt->GetYaxis()->SetTitle("# Events");
 hwMPSumHt->GetYaxis()->SetTitleSize(0.07);
 hwMPSumHt->GetYaxis()->SetTitleOffset(0.48);
 hwMPSumHt->GetXaxis()->SetTitleSize(0.04);
 hwMPSumHt->GetXaxis()->SetTitleOffset(0.9);
 pSumHt->SetBottomMargin(0.08);
 pSumHt->Draw();
 pSumHt->cd();

 TH1D* SumHtRatio = (TH1D*)hwMPSumHt->DrawCopy("p");
 SumHtRatio->SetMinimum(0);
 emMPSumHt->Draw("same");//"");
 leg->Draw();
 cMPSumHt->cd();
 pSumHtRatio->SetTopMargin(0.05);
 pSumHtRatio->Draw();
 pSumHtRatio->cd();
 hwMPSumHt->Divide(emMPSumHt);
 hwMPSumHt->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPSumHt->GetYaxis()->SetTitleSize(0.09);
 hwMPSumHt->GetYaxis()->SetLabelSize(0.07);
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

 cMPSumHt->SaveAs("compHwEmu/MPSums/MPSumHt.png");


 // //--- MP sum htx ---//

 TCanvas* cMPSumHtx = new TCanvas("cMPSumHtx","MPSumHtx");

 TPad* pSumHtx = new TPad("pSumHtx","pSumHtx",0,0.3,1,1); 
 TPad* pSumHtxRatio = new TPad("pSumHtxratio","pSumHtxratio",0,0,1,0.3);
 
 TPad* pInvSumHtxRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSumHtxRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPSumHtx,"Hardware MP", "p");
 leg->AddEntry(emMPSumHtx,"Emulator MP", "l");
 
 hwMPSumHtx->Rebin(20);
 emMPSumHtx->Rebin(20);

 hwMPSumHtx->SetStats(0);
 hwMPSumHtx->SetMarkerStyle(21);
 hwMPSumHtx->SetMarkerColor(1);
 hwMPSumHtx->SetMarkerSize(0.4);
 emMPSumHtx->SetLineColor(kRed);
 hwMPSumHtx->GetXaxis()->SetTitle("iHTx");
 hwMPSumHtx->GetYaxis()->SetTitle("# Events");
 hwMPSumHtx->GetYaxis()->SetTitleSize(0.07);
 hwMPSumHtx->GetYaxis()->SetTitleOffset(0.48);
 hwMPSumHtx->GetXaxis()->SetTitleSize(0.04);
 hwMPSumHtx->GetXaxis()->SetTitleOffset(0.9);
 pSumHtx->SetBottomMargin(0.08);
 pSumHtx->Draw();
 pSumHtx->cd();
 
 TH1D* SumHtxRatio = (TH1D*)hwMPSumHtx->DrawCopy("p");
 SumHtxRatio->SetMinimum(0);
 emMPSumHtx->Draw("same");//"");
 leg->Draw();
 cMPSumHtx->cd();
 pSumHtxRatio->SetTopMargin(0.05);
 pSumHtxRatio->Draw();
 pSumHtxRatio->cd();
 hwMPSumHtx->Divide(emMPSumHtx);
 hwMPSumHtx->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPSumHtx->GetYaxis()->SetTitleSize(0.09);
 hwMPSumHtx->GetYaxis()->SetLabelSize(0.07);
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
 cMPSumHtx->SaveAs("compHwEmu/MPSums/MPSumHtx.png");


 // //--- MP sum hty ---//

 TCanvas* cMPSumHty = new TCanvas("cMPSumHty","MPSumHty");

 TPad* pSumHty = new TPad("pSumHty","pSumHty",0,0.3,1,1); 
 TPad* pSumHtyRatio = new TPad("pSumHtyratio","pSumHtyratio",0,0,1,0.3);
 
 TPad* pInvSumHtyRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSumHtyRatio->SetFillStyle(0);
  
 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMPSumHty,"Hardware MP", "p");
 leg->AddEntry(emMPSumHty,"Emulator MP", "l");

 hwMPSumHty->Rebin(20);
 emMPSumHty->Rebin(20);

 hwMPSumHty->SetStats(0);
 hwMPSumHty->SetMarkerStyle(21);
 hwMPSumHty->SetMarkerColor(1);
 hwMPSumHty->SetMarkerSize(0.4);
 emMPSumHty->SetLineColor(kRed);
 hwMPSumHty->GetXaxis()->SetTitle("iHTy");
 hwMPSumHty->GetYaxis()->SetTitle("# Events");
 hwMPSumHty->GetYaxis()->SetTitleSize(0.07);
 hwMPSumHty->GetYaxis()->SetTitleOffset(0.48);
 hwMPSumHty->GetXaxis()->SetTitleSize(0.04);
 hwMPSumHty->GetXaxis()->SetTitleOffset(0.9);
 pSumHty->SetBottomMargin(0.08);
 pSumHty->Draw();
 pSumHty->cd();

 TH1D* SumHtyRatio = (TH1D*)hwMPSumHty->DrawCopy("p");
 SumHtyRatio->SetMinimum(0);
 emMPSumHty->Draw("same");//"");
 leg->Draw();
 cMPSumHty->cd();
 pSumHtyRatio->SetTopMargin(0.05);
 pSumHtyRatio->Draw();
 pSumHtyRatio->cd();
 hwMPSumHty->Divide(emMPSumHty);
 hwMPSumHty->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMPSumHty->GetYaxis()->SetTitleSize(0.09);
 hwMPSumHty->GetYaxis()->SetLabelSize(0.07);
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

 cMPSumHty->SaveAs("compHwEmu/MPSums/MPSumHty.png");


 // //--- demux sum et ---//

 TCanvas* cSumEt = new TCanvas("cSumEt","SumEt");

 TPad* pDSumEt = new TPad("pSumEt","pSumEt",0,0.3,1,1); 
 TPad* pDSumEtRatio = new TPad("pSumEtratio","pSumEtratio",0,0,1,0.3);
 
 TPad* pInvDSumEtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDSumEtRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwSumEt,"Hardware Demux", "p");
 leg->AddEntry(emSumEt,"Emulator Demux", "l");

 hwSumEt->Rebin(50);
 emSumEt->Rebin(50);

 hwSumEt->SetStats(0);
 hwSumEt->SetMarkerStyle(21);
 hwSumEt->SetMarkerColor(1);
 hwSumEt->SetMarkerSize(0.4);
 emSumEt->SetLineColor(kRed);
 hwSumEt->GetXaxis()->SetTitle("iET");
 hwSumEt->GetYaxis()->SetTitle("# Jets");
 hwSumEt->GetYaxis()->SetTitleSize(0.07);
 hwSumEt->GetYaxis()->SetTitleOffset(0.48);
 hwSumEt->GetXaxis()->SetTitleSize(0.04);
 hwSumEt->GetXaxis()->SetTitleOffset(0.9);
 pDSumEt->SetBottomMargin(0.08);
 pDSumEt->Draw();
 pDSumEt->cd();
 
 TH1D* DSumEtRatio = (TH1D*)hwSumEt->DrawCopy("p");
 DSumEtRatio->SetMinimum(0);
 emSumEt->Draw("same");
 leg->Draw();
 cSumEt->cd();
 pDSumEtRatio->SetTopMargin(0.05);
 pDSumEtRatio->Draw();
 pDSumEtRatio->cd();
 hwSumEt->Divide(emSumEt);
 hwSumEt->GetYaxis()->SetTitle("Ratio HW/EM");
 hwSumEt->GetYaxis()->SetTitleSize(0.09);
 hwSumEt->GetYaxis()->SetLabelSize(0.07);
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
 cSumEt->SaveAs("compHwEmu/DemuxSums/DemSumEt.png");


 //--- demux sum met ---//

 TCanvas* cSumMet = new TCanvas("cSumMet","SumMet");

 TPad* pDSumMet = new TPad("pSumMet","pSumMet",0,0.3,1,1); 
 TPad* pDSumMetRatio = new TPad("pSumMetratio","pSumMetratio",0,0,1,0.3);
 
 TPad* pInvDSumMetRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDSumMetRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwSumMet,"Hardware Demux", "p");
 leg->AddEntry(emSumMet,"Emulator Demux", "l");

 hwSumMet->Rebin(20);
 emSumMet->Rebin(20);

 hwSumMet->SetStats(0);
 hwSumMet->SetMarkerStyle(21);
 hwSumMet->SetMarkerColor(1);
 hwSumMet->SetMarkerSize(0.4);
 emSumMet->SetLineColor(kRed);
 hwSumMet->GetXaxis()->SetTitle("iMET");
 hwSumMet->GetYaxis()->SetTitle("# Events");
 hwSumMet->GetYaxis()->SetTitleSize(0.07);
 hwSumMet->GetYaxis()->SetTitleOffset(0.48);
 hwSumMet->GetXaxis()->SetTitleSize(0.04);
 hwSumMet->GetXaxis()->SetTitleOffset(0.9);
 pDSumMet->SetBottomMargin(0.08);
 pDSumMet->Draw();
 pDSumMet->cd();

 TH1D* DSumMetRatio = (TH1D*)hwSumMet->DrawCopy("p");
 DSumMetRatio->SetMinimum(0);
 emSumMet->Draw("same");
 leg->Draw();
 cSumMet->cd();
 pDSumMetRatio->SetTopMargin(0.05);
 pDSumMetRatio->Draw();
 pDSumMetRatio->cd();
 hwSumMet->Divide(emSumMet);
 hwSumMet->GetYaxis()->SetTitle("Ratio HW/EM");
 hwSumMet->GetYaxis()->SetTitleSize(0.09);
 hwSumMet->GetYaxis()->SetLabelSize(0.07);
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
 cSumMet->SaveAs("compHwEmu/DemuxSums/DemSumMet.png");


 //--- demux sum ht ---//

 TCanvas* cSumHt = new TCanvas("cSumHt","SumHt");

 TPad* pDSumHt = new TPad("pSumHt","pSumHt",0,0.3,1,1); 
 TPad* pDSumHtRatio = new TPad("pSumHtratio","pSumHtratio",0,0,1,0.3);
 
 TPad* pInvDSumHtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDSumHtRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwSumHt,"Hardware Demux", "p");
 leg->AddEntry(emSumHt,"Emulator Demux", "l");

 hwSumHt->Rebin(50);
 emSumHt->Rebin(50);

 hwSumHt->SetStats(0);
 hwSumHt->SetMarkerStyle(21);
 hwSumHt->SetMarkerColor(1);
 hwSumHt->SetMarkerSize(0.4);
 emSumHt->SetLineColor(kRed);
 hwSumHt->GetXaxis()->SetTitle("iHT");
 hwSumHt->GetYaxis()->SetTitle("# Events");
 hwSumHt->GetYaxis()->SetTitleSize(0.07);
 hwSumHt->GetYaxis()->SetTitleOffset(0.48);
 hwSumHt->GetXaxis()->SetTitleSize(0.04);
 hwSumHt->GetXaxis()->SetTitleOffset(0.9);
 pDSumHt->SetBottomMargin(0.08);
 pDSumHt->Draw();
 pDSumHt->cd();

 TH1D* DSumHtRatio = (TH1D*)hwSumHt->DrawCopy("p");
 DSumHtRatio->SetMinimum(0);
 emSumHt->Draw("same");
 leg->Draw();
 cSumHt->cd();
 pDSumHtRatio->SetTopMargin(0.05);
 pDSumHtRatio->Draw();
 pDSumHtRatio->cd();
 hwSumHt->Divide(emSumHt);
 hwSumHt->GetYaxis()->SetTitle("Ratio HW/EM");
 hwSumHt->GetYaxis()->SetTitleSize(0.09);
 hwSumHt->GetYaxis()->SetLabelSize(0.07);
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
 cSumHt->SaveAs("compHwEmu/DemuxSums/DemSumHt.png");


 //--- demux sum mht ---//

 TCanvas* cSumMht = new TCanvas("cSumMht","SumMht");

 TPad* pDSumMht = new TPad("pSumMht","pSumMht",0,0.3,1,1); 
 TPad* pDSumMhtRatio = new TPad("pSumMhtratio","pSumMhtratio",0,0,1,0.3);
 
 TPad* pInvDSumMhtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDSumMhtRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwSumMht,"Hardware Demux", "p");
 leg->AddEntry(emSumMht,"Emulator Demux", "l");

 hwSumMht->Rebin(20);
 emSumMht->Rebin(20);

 hwSumMht->SetStats(0);
 hwSumMht->SetMarkerStyle(21);
 hwSumMht->SetMarkerColor(1);
 hwSumMht->SetMarkerSize(0.4);
 emSumMht->SetLineColor(kRed);
 hwSumMht->GetXaxis()->SetTitle("iMHT");
 hwSumMht->GetYaxis()->SetTitle("# Events");
 hwSumMht->GetYaxis()->SetTitleSize(0.07);
 hwSumMht->GetYaxis()->SetTitleOffset(0.48);
 hwSumMht->GetXaxis()->SetTitleSize(0.04);
 hwSumMht->GetXaxis()->SetTitleOffset(0.9);
 pDSumMht->SetBottomMargin(0.08);
 pDSumMht->Draw();
 pDSumMht->cd();

 TH1D* DSumMhtRatio = (TH1D*)hwSumMht->DrawCopy("p");
 DSumMhtRatio->SetMinimum(0);
 emSumMht->Draw("same");
 leg->Draw();
 cSumMht->cd();
 pDSumMhtRatio->SetTopMargin(0.05);
 pDSumMhtRatio->Draw();
 pDSumMhtRatio->cd();
 hwSumMht->Divide(emSumMht);
 hwSumMht->GetYaxis()->SetTitle("Ratio HW/EM");
 hwSumMht->GetYaxis()->SetTitleSize(0.09);
 hwSumMht->GetYaxis()->SetLabelSize(0.07);
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
 cSumMht->SaveAs("compHwEmu/DemuxSums/DemSumMht.png");

 //--- met phi ---//

 TCanvas* cMetPhi = new TCanvas("cMetPhi","MetPhi");

 TPad* pDMetPhi = new TPad("pMetPhi","pMetPhi",0,0.3,1,1); 
 TPad* pDMetPhiRatio = new TPad("pMetPhiratio","pMetPhiratio",0,0,1,0.3);
 
 TPad* pInvDMetPhiRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDMetPhiRatio->SetFillStyle(0);

 leg = new TLegend(0.7,0.65,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMetPhi,"Hardware Demux", "p");
 leg->AddEntry(emMetPhi,"Emulator Demux", "l");

 hwMetPhi->Rebin(6);
 emMetPhi->Rebin(6);

 hwMetPhi->SetStats(0);
 hwMetPhi->SetMarkerStyle(21);
 hwMetPhi->SetMarkerColor(1);
 hwMetPhi->SetMarkerSize(0.4);
 emMetPhi->SetLineColor(kRed);
 hwMetPhi->GetXaxis()->SetRange(0,40);
 hwMetPhi->GetXaxis()->SetTitle("MET i#phi");
 hwMetPhi->GetYaxis()->SetTitle("# Jets");
 hwMetPhi->GetYaxis()->SetTitleSize(0.07);
 hwMetPhi->GetYaxis()->SetTitleOffset(0.48);
 hwMetPhi->GetXaxis()->SetTitleSize(0.04);
 hwMetPhi->GetXaxis()->SetTitleOffset(0.9);
 pDMetPhi->SetBottomMargin(0.08);
 pDMetPhi->Draw();
 pDMetPhi->cd();

 TH1D* DMetPhiRatio = (TH1D*)hwMetPhi->DrawCopy("p");
 DMetPhiRatio->SetMinimum(0);
 emMetPhi->Draw("same");
 leg->Draw();
 cMetPhi->cd();
 pDMetPhiRatio->SetTopMargin(0.05);
 pDMetPhiRatio->Draw();
 pDMetPhiRatio->cd();
 hwMetPhi->Divide(emMetPhi);
 hwMetPhi->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMetPhi->GetYaxis()->SetTitleSize(0.09);
 hwMetPhi->GetYaxis()->SetLabelSize(0.07);
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
 cMetPhi->SaveAs("compHwEmu/DemuxSums/DemMetPhi.png");

 //--- mht phi ---//

 TCanvas* cMhtPhi = new TCanvas("cMhtPhi","MhtPhi");

 TPad* pDMhtPhi = new TPad("pMhtPhi","pMhtPhi",0,0.3,1,1); 
 TPad* pDMhtPhiRatio = new TPad("pMhtPhiratio","pMhtPhiratio",0,0,1,0.3);
 
 TPad* pInvDMhtPhiRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDMhtPhiRatio->SetFillStyle(0);

 leg = new TLegend(0.7,0.65,0.9,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwMhtPhi,"Hardware Demux", "p");
 leg->AddEntry(emMhtPhi,"Emulator Demux", "l");

 hwMhtPhi->Rebin(6);
 emMhtPhi->Rebin(6);

 hwMhtPhi->SetStats(0);
 hwMhtPhi->SetMarkerStyle(21);
 hwMhtPhi->SetMarkerColor(1);
 hwMhtPhi->SetMarkerSize(0.4);
 emMhtPhi->SetLineColor(kRed);
 hwMhtPhi->GetXaxis()->SetRange(0,40);
 hwMhtPhi->GetXaxis()->SetTitle("MHT i#phi");
 hwMhtPhi->GetYaxis()->SetTitle("# Events");
 hwMhtPhi->GetYaxis()->SetTitleSize(0.07);
 hwMhtPhi->GetYaxis()->SetTitleOffset(0.48);
 hwMhtPhi->GetXaxis()->SetTitleSize(0.04);
 hwMhtPhi->GetXaxis()->SetTitleOffset(0.9);
 pDMhtPhi->SetBottomMargin(0.08);
 pDMhtPhi->Draw();
 pDMhtPhi->cd();

 TH1D* DMhtPhiRatio = (TH1D*)hwMhtPhi->DrawCopy("p");
 DMhtPhiRatio->SetMinimum(0);
 emMhtPhi->Draw("same");
 leg->Draw();
 cMhtPhi->cd();
 pDMhtPhiRatio->SetTopMargin(0.05);
 pDMhtPhiRatio->Draw();
 pDMhtPhiRatio->cd();
 hwMhtPhi->Divide(emMhtPhi);
 hwMhtPhi->GetYaxis()->SetTitle("Ratio HW/EM");
 hwMhtPhi->GetYaxis()->SetTitleSize(0.09);
 hwMhtPhi->GetYaxis()->SetLabelSize(0.07);
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
 cMhtPhi->SaveAs("compHwEmu/DemuxSums/DemMhtPhi.png");



 //--- sorts ---//

 TCanvas* cSortMP = new TCanvas("cSortMP","SortMP");

 TPad* pSortMP = new TPad("pSortMP","pSortMP",0,0.3,1,1); 
 TPad* pSortMPRatio = new TPad("pSortMPratio","pSortMPratio",0,0,1,0.3);
 
 TPad* pInvSortMPRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvSortMPRatio->SetFillStyle(0);
  
 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwSortMP,"Hardware MP", "p");
 leg->AddEntry(emSortMP,"Emulator MP", "l");

 hwSortMP->SetStats(0);
 hwSortMP->SetMarkerStyle(21);
 hwSortMP->SetMarkerColor(1);
 hwSortMP->SetMarkerSize(0.4);
 emSortMP->SetLineColor(kRed);
 hwSortMP->GetXaxis()->SetTitle("iHT");
 hwSortMP->GetYaxis()->SetTitle("# Events");
 hwSortMP->GetYaxis()->SetTitleSize(0.07);
 hwSortMP->GetYaxis()->SetTitleOffset(0.48);
 hwSortMP->GetXaxis()->SetTitleSize(0.04);
 hwSortMP->GetXaxis()->SetTitleOffset(0.9);
 pSortMP->SetBottomMargin(0.08);
 pSortMP->Draw();
 pSortMP->cd();


 TH1D* SortMPRatio = (TH1D*)hwSortMP->DrawCopy("p");
 SortMPRatio->SetMinimum(0);
 emSortMP->Draw("same");
 leg->Draw();
 cSortMP->cd();
 pSortMPRatio->SetTopMargin(0.05);
 pSortMPRatio->Draw();
 pSortMPRatio->cd();
 hwSortMP->Divide(emSortMP);
 hwSortMP->GetYaxis()->SetTitle("Ratio HW/EM");
 hwSortMP->GetYaxis()->SetTitleSize(0.09);
 hwSortMP->GetYaxis()->SetLabelSize(0.07);
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

 cSortMP->SaveAs("compHwEmu/Sorts/MPSort.png");


 TCanvas* cSort = new TCanvas("cSort","Sort");

 TPad* pDSort = new TPad("pSort","pSort",0,0.3,1,1); 
 TPad* pDSortRatio = new TPad("pSortratio","pSortratio",0,0,1,0.3);
 
 TPad* pInvDSortRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDSortRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwSort,"Hardware Demux", "p");
 leg->AddEntry(emSort,"Emulator Demux", "l");

 hwSort->Rebin(3);
 emSort->Rebin(3);

 hwSort->SetStats(0);
 hwSort->SetMarkerStyle(21);
 hwSort->SetMarkerColor(1);
 hwSort->SetMarkerSize(0.4);
 emSort->SetLineColor(kRed);
 hwSort->GetXaxis()->SetTitle("iHT");
 hwSort->GetYaxis()->SetTitle("# Events");
 hwSort->GetYaxis()->SetTitleSize(0.07);
 hwSort->GetYaxis()->SetTitleOffset(0.48);
 hwSort->GetXaxis()->SetTitleSize(0.04);
 hwSort->GetXaxis()->SetTitleOffset(0.9);
 pDSort->SetBottomMargin(0.08);
 pDSort->Draw();
 pDSort->cd();

 TH1D* DSortRatio = (TH1D*)hwSort->DrawCopy("p");
 DSortRatio->SetMinimum(0);
 emSort->Draw("same");
 leg->Draw();
 cSort->cd();
 pDSortRatio->SetTopMargin(0.05);
 pDSortRatio->Draw();
 pDSortRatio->cd();
 hwSort->Divide(emSort);
 hwSort->GetYaxis()->SetTitle("Ratio HW/EM");
 hwSort->GetYaxis()->SetTitleSize(0.09);
 hwSort->GetYaxis()->SetLabelSize(0.07);
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
 cSort->SaveAs("compHwEmu/Sorts/DemuxSort.png");


 //================ egamma ====================


 TCanvas* cDEgEt = new TCanvas("cDEgEt","DEgEt");

 TPad* pDEgEt = new TPad("pEgEt","pEgEt",0,0.3,1,1); 
 TPad* pDEgEtRatio = new TPad("pEgEtratio","pEgEtratio",0,0,1,0.3);
 
 TPad* pInvDEgEtRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDEgEtRatio->SetFillStyle(0);

 leg = new TLegend(0.6,0.65,0.85,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwEgEt,"Hardware Demux", "p");
 leg->AddEntry(emEgEt,"Emulator Demux", "l");

 hwEgEt->Rebin(10);
 emEgEt->Rebin(10);

 hwEgEt->SetStats(0);
 hwEgEt->SetMarkerStyle(21);
 hwEgEt->SetMarkerColor(1);
 hwEgEt->SetMarkerSize(0.4);
 emEgEt->SetLineColor(kRed);
 hwEgEt->GetXaxis()->SetTitle("Egamma iET");
 hwEgEt->GetYaxis()->SetTitle("# Egamma");
 hwEgEt->GetYaxis()->SetTitleSize(0.07);
 hwEgEt->GetYaxis()->SetTitleOffset(0.48);
 hwEgEt->GetXaxis()->SetTitleSize(0.04);
 hwEgEt->GetXaxis()->SetTitleOffset(0.9);
 pDEgEt->SetBottomMargin(0.08);
 pDEgEt->Draw();
 pDEgEt->cd();
 
 TH1D* DEgEtRatio = (TH1D*)hwEgEt->DrawCopy("p");
 DEgEtRatio->SetMinimum(0);
 emEgEt->Draw("same");
 leg->Draw();
 cDEgEt->cd();
 pDEgEtRatio->SetTopMargin(0.05);
 pDEgEtRatio->Draw();
 pDEgEtRatio->cd();
 hwEgEt->Divide(emEgEt);
 hwEgEt->GetYaxis()->SetTitle("Ratio HW/EM");
 hwEgEt->GetYaxis()->SetTitleSize(0.09);
 hwEgEt->GetYaxis()->SetLabelSize(0.07);
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
 cDEgEt->SaveAs("compHwEmu/DemuxJets/EgEt.png");


 
 TCanvas* cDEgEta = new TCanvas("cDEgEta","DEgEta");
 TPad* pDEgEta = new TPad("pEgEta","pEgEta",0,0.3,1,1); 
 TPad* pDEgEtaRatio = new TPad("pEgEtaratio","pEgEtaratio",0,0,1,0.3);
 
 TPad* pInvDEgEtaRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDEgEtaRatio->SetFillStyle(0);

 leg = new TLegend(0.4,0.65,0.65,0.85);
 leg->SetFillColor(0);
 leg->AddEntry(hwEgEta,"Hardware Demux", "p");
 leg->AddEntry(emEgEta,"Emulator Demux", "l");

 hwEgEta->SetStats(0);
 hwEgEta->SetMarkerStyle(21);
 hwEgEta->SetMarkerColor(1);
 hwEgEta->SetMarkerSize(0.4);
 emEgEta->SetLineColor(kRed);
 hwEgEta->GetXaxis()->SetTitle("Jet i#eta");
 hwEgEta->GetXaxis()->SetRange(82,146);
 hwEgEta->GetYaxis()->SetTitle("# Jets");
 hwEgEta->GetYaxis()->SetTitleSize(0.07);
 hwEgEta->GetYaxis()->SetTitleOffset(0.48);
 hwEgEta->GetXaxis()->SetTitleSize(0.04);
 hwEgEta->GetXaxis()->SetTitleOffset(0.9);
 pDEgEta->SetBottomMargin(0.08);
 pDEgEta->Draw();
 pDEgEta->cd();
 
 TH1D* DEgEtaRatio = (TH1D*)hwEgEta->DrawCopy("p");
 DEgEtaRatio->SetMinimum(0);
 emEgEta->Draw("same");
 leg->Draw();
 cDEgEta->cd();
 pDEgEtaRatio->SetTopMargin(0.05);
 pDEgEtaRatio->Draw();
 pDEgEtaRatio->cd();
 hwEgEta->Divide(emEgEta);
 hwEgEta->GetYaxis()->SetTitle("Ratio HW/EM");
 hwEgEta->GetYaxis()->SetTitleSize(0.09);
 hwEgEta->GetYaxis()->SetLabelSize(0.07);
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
 cDEgEta->SaveAs("compHwEmu/DemuxJets/EgEta.png");


 TCanvas* cDEgPhi = new TCanvas("cDEgPhi","DEgPhi");

 TPad* pDEgPhi = new TPad("pEgPhi","pEgPhi",0,0.3,1,1); 
 TPad* pDEgPhiRatio = new TPad("pEgPhiratio","pEgPhiratio",0,0,1,0.3);
  
 TPad* pInvDEgPhiRatio = new TPad("pInv","pInv", 0,0,1,0.3);
 pInvDEgPhiRatio->SetFillStyle(0);

 leg = new TLegend(0.75,0.75,0.9,0.9);
 leg->SetFillColor(0);
 leg->AddEntry(hwEgPhi,"Hardware Demux", "p");
 leg->AddEntry(emEgPhi,"Emulator Demux", "l");

 hwEgPhi->SetStats(0);
 hwEgPhi->SetMarkerStyle(21);
 hwEgPhi->SetMarkerColor(1);
 hwEgPhi->SetMarkerSize(0.4);
 emEgPhi->SetLineColor(kRed);
 hwEgPhi->GetXaxis()->SetTitle("Jet i#phi");
 hwEgPhi->GetXaxis()->SetRange(0,73);
 hwEgPhi->GetYaxis()->SetTitle("# Jets");
 hwEgPhi->GetYaxis()->SetTitleSize(0.07);
 hwEgPhi->GetYaxis()->SetTitleOffset(0.48);
 hwEgPhi->GetXaxis()->SetTitleSize(0.04);
 hwEgPhi->GetXaxis()->SetTitleOffset(0.9);
 pDEgPhi->SetBottomMargin(0.08);
 pDEgPhi->Draw();
 pDEgPhi->cd();

 TH1D* DEgPhiRatio = (TH1D*)hwEgPhi->DrawCopy("p");
 DEgPhiRatio->SetMinimum(0);
 emEgPhi->Draw("same");//"");
 leg->Draw();
 cDEgPhi->cd();
 pDEgPhiRatio->SetTopMargin(0.05);
 pDEgPhiRatio->Draw();
 pDEgPhiRatio->cd();
 hwEgPhi->Divide(emEgPhi);
 hwEgPhi->GetYaxis()->SetTitle("Ratio HW/EM");
 hwEgPhi->GetYaxis()->SetTitleSize(0.09);
 hwEgPhi->GetYaxis()->SetLabelSize(0.07);
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
 cDEgPhi->SaveAs("compHwEmu/DemuxJets/EgPhi.png");

}
