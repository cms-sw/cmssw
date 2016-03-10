#include "TMath.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TMultiGraph.h"

#include <iostream>

#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeDataFormat.h"

void rates(const char * rootfile="L1Ntuple.root",const char * treepath="l1UpgradeEmuTree/L1UpgradeTree"){
  nevents=10000;

  // make trees
  TFile * file = new TFile(rootfile);
  TTree * treeL1Up  = (TTree*) file->Get(treepath);
  if (! treeL1Up){
    cout << "ERROR: could not open tree\n";
    return;
  }


  treeL1Up->Print();

  // set branch addresses
  L1Analysis::L1AnalysisL1UpgradeDataFormat    *upgrade_ = new L1Analysis::L1AnalysisL1UpgradeDataFormat();
  treeL1Up->SetBranchAddress("L1Upgrade", &upgrade_);

  // mu bins
  int nMuBins = 50;
  float muLo = 10.;
  float muHi = 50.;
  float muBinWidth = (muHi-muLo)/nMuBins;

  // eg bins
  int nEgBins = 50;
  float egLo = 10.;
  float egHi = 260.;
  float egBinWidth = (egHi-egLo)/nEgBins;

  // tau bins
  int nTauBins = 50;
  float tauLo = 10.;
  float tauHi = 260.;
  float tauBinWidth = (tauHi-tauLo)/nTauBins;
 
  // jet bins
  int nJetBins = 50;
  float jetLo = 10.;
  float jetHi = 260.;
  float jetBinWidth = (jetHi-jetLo)/nJetBins;

  // etSum bins
  int nEtSumBins = 100;
  float etSumLo = 10.;
  float etSumHi = 510.;
  float etSumBinWidth = (etSumHi-etSumLo)/nEtSumBins;

  // htSum bins
  int nHtSumBins = 100;
  float htSumLo = 10.;
  float htSumHi = 510.;
  float htSumBinWidth = (htSumHi-htSumLo)/nHtSumBins;

  // metSum bins
  int nMetSumBins = 50;
  float metSumLo = 10.;
  float metSumHi = 260.;
  float metSumBinWidth = (metSumHi-metSumLo)/nMetSumBins;

  // mhtSum bins
  int nMhtSumBins = 50;
  float mhtSumLo = 10.;
  float mhtSumHi = 260.;
  float mhtSumBinWidth = (mhtSumHi-mhtSumLo)/nMhtSumBins;

  //make histos
  TH1D* muRates = new TH1D("muRates", "", nMuBins, muLo-2.5, muHi-2.5);
  TH1D* egRates = new TH1D("egRates", "", nEgBins, egLo-2.5, egHi-2.5);
  TH1D* tauRates = new TH1D("tauRates", "", nTauBins, tauLo-2.5, tauHi-2.5);
  TH1D* jetRates = new TH1D("jetRates", "", nJetBins, jetLo-2.5, jetHi-2.5);
  TH1D* etSumRates = new TH1D("etSumRates","", nEtSumBins, etSumLo-2.5, etSumHi-2.5);
  TH1D* htSumRates = new TH1D("htSumRates","", nHtSumBins, htSumLo-2.5, htSumHi-2.5);
  TH1D* metSumRates = new TH1D("metSumRates","", nMetSumBins, metSumLo-2.5, metSumHi-2.5);
  TH1D* mhtSumRates = new TH1D("mhtSumRates","", nMhtSumBins, mhtSumLo-2.5, mhtSumHi-2.5);

  TH1D* hMuEt = new TH1D("muEt", "", 400, 0.,400.);
  TH1D* hEgEt = new TH1D("egEt", "", 400, 0.,400.);
  TH1D* hTauEt = new TH1D("tauEt","",400, 0.,400.);
  TH1D* hJetEt = new TH1D("jetEt","",400, 0.,400.);
  TH1D* hEtSum = new TH1D("etSum","",800, 0.,800.);
  TH1D* hHtSum = new TH1D("htSum","",800, 0.,800.);

  // get entries
  Long64_t nentries = treeL1Up->GetEntriesFast();
  if (nevents>nentries) nevents=nentries;

  std::cout << "Running over " << nevents << ", nentries = " << nentries << std::endl;

  for (Long64_t jentry=0; jentry<nevents;jentry++){

    if((jentry%1000)==0) std::cout << "Done " << jentry  << " events..." << std::endl;

    treeL1Up->GetEntry(jentry);
    
    //cout << upgrade_->nJets << "\n";

    // get Mu rates
    double muEt(0);
    for(uint it=0; it<upgrade_->nMuons; ++it){
      // work around a muon bug:
      int offset = upgrade_->muonQual.size() - upgrade_->nMuons;
      //cout << "INFO:  " << upgrade_->nMuons << "\n";
      //cout << "INFO:  " << upgrade_->muonEt.size() << "\n";
      //cout << "INFO:  " << upgrade_->muonQual.size() << "\n";
      if (upgrade_->muonQual[it+offset]<8) continue;
      hMuEt->Fill(upgrade_->muonEt[it]);
      muEt = upgrade_->muonEt[it] > muEt ?  upgrade_->muonEt[it]  : muEt;
    }
    for(int bin=0; bin<nMuBins; bin++)
      if(muEt >= muLo + (bin*muBinWidth) ) muRates->Fill(muLo+(bin*muBinWidth)); //GeV

    // get Eg rates
    int egEt(0);
    for(uint it=0; it<upgrade_->nEGs; ++it){
      hEgEt->Fill(0.5*upgrade_->egIEt[it]);
      egEt = upgrade_->egIEt[it] > egEt ?  upgrade_->egIEt[it]  : egEt;
    }
    for(int bin=0; bin<nEgBins; bin++)
      if(egEt*0.5 >= egLo + (bin*egBinWidth) ) egRates->Fill(egLo+(bin*egBinWidth)); //GeV
    
    // get Tau rates
    int tauEt(0);
    for(uint it=0; it<upgrade_->nTaus; ++it){
      hTauEt->Fill(0.5*upgrade_->tauIEt[it]);
      tauEt = upgrade_->tauIEt[it] > tauEt ? upgrade_->tauIEt[it] : tauEt;
    }
    for(int bin=0; bin<nTauBins; bin++)
      if( (tauEt*0.5) >= tauLo + (bin*tauBinWidth) ) tauRates->Fill(tauLo+(bin*tauBinWidth)); //GeV
        
    // get Jet rates
    int jetEt(0);
    for(uint it=0; it<upgrade_->nJets; ++it){
      hJetEt->Fill(0.5*upgrade_->jetIEt[it]);
      jetEt =  upgrade_->jetIEt[it] > jetEt ? upgrade_->jetIEt[it] : jetEt;
    }
    for(int bin=0; bin<nJetBins; bin++)
      if( (jetEt*0.5) >= jetLo + (bin*jetBinWidth) ) jetRates->Fill(jetLo+(bin*jetBinWidth));  //GeV

    double etSum  = -1.0;
    double htSum  = -1.0;
    double metSum = -1.0;
    double mhtSum = -1.0;
    for(uint it=0; it<upgrade_->nSums; ++it){
      double et = upgrade_->sumEt[it];
      if (upgrade_->sumType[it] == L1Analysis::kTotalEt)   etSum  = et;
      if (upgrade_->sumType[it] == L1Analysis::kTotalHt)   htSum  = et;
      if (upgrade_->sumType[it] == L1Analysis::kMissingEt) metSum = et;
      if (upgrade_->sumType[it] == L1Analysis::kMissingHt) mhtSum = et;
    }
    //std::cout << "mht:  " << mhtSum << "\n";
    //std::cout << "ht sum:  " << htSum << "\n";

    hEtSum->Fill(0.5*etSum);
    //std::cout << "et sum = " << etSum << std::endl;
    for(int bin=0; bin<nEtSumBins; bin++)
      if( (etSum*0.5) >= etSumLo+(bin*etSumBinWidth) ) etSumRates->Fill(etSumLo+(bin*etSumBinWidth)); //GeV
    
    hHtSum->Fill(0.5*htSum);
    //std::cout << "ht sum = " << htSum << std::endl;
    for(int bin=0; bin<nHtSumBins; bin++){
      //std::cout << "Ht? " << upgrade_->sumEt[1]->getType() << std::endl;
      if( (htSum*0.5) >= htSumLo+(bin*htSumBinWidth) ) htSumRates->Fill(htSumLo+(bin*htSumBinWidth)); //GeV
    }

    //hMetSum->Fill(0.5*metSum);
    //std::cout << "met sum = " << metSum << std::endl;
    for(int bin=0; bin<nMetSumBins; bin++)
      if( (metSum*0.5) >= metSumLo+(bin*metSumBinWidth) ) metSumRates->Fill(metSumLo+(bin*metSumBinWidth)); //GeV
        
    //hMhtSum->Fill(0.5*mhtSum]);
    //std::cout << "mht sum = " << mhtSum << std::endl;
    for(int bin=0; bin<nMhtSumBins; bin++){
      //std::cout << "Mht? " << upgrade_->sumEt[1]->getType() << std::endl;
      if( (mhtSum*0.5) >= mhtSumLo+(bin*mhtSumBinWidth) ) mhtSumRates->Fill(mhtSumLo+(bin*mhtSumBinWidth)); //GeV
    }

  }


  //normalisation factor
  double avrgInstLumi = 4.5e33; 
  double sigmaPP = 6.9e-26;
  //double norm = (avrgInstLumi*sigmaPP)/(nevents*1000); //kHz
  double norm = (11.*2244.)/nevents; // zb rate = n_colliding * 11 kHz 
  std::cout << "norm = " << norm << std::endl;

  //make TGraphs
  TGraph* gMuRate = new TGraph(nMuBins);
  TGraph* gEgRate = new TGraph(nEgBins);
  TGraph* gTauRate = new TGraph(nTauBins);
  TGraph* gJetRate = new TGraph(nJetBins);
  TGraph* gEtSumRate = new TGraph(nEtSumBins);
  TGraph* gHtSumRate = new TGraph(nHtSumBins);
  TGraph* gMetSumRate = new TGraph(nMetSumBins);
  TGraph* gMhtSumRate = new TGraph(nMhtSumBins);

  //norm=1;
  for(int bin=0;bin<nMuBins;bin++) gMuRate->SetPoint(bin,muLo+muBinWidth*bin,muRates->GetBinContent(bin+1)*norm);
  for(int bin=0;bin<nEgBins;bin++) gEgRate->SetPoint(bin,egLo+egBinWidth*bin,egRates->GetBinContent(bin+1)*norm);
  for(int bin=0;bin<nTauBins;bin++) gTauRate->SetPoint(bin,tauLo+tauBinWidth*bin,tauRates->GetBinContent(bin+1)*norm);
  for(int bin=0;bin<nJetBins;bin++) gJetRate->SetPoint(bin,jetLo+jetBinWidth*bin,jetRates->GetBinContent(bin+1)*norm);
  for(int bin=0;bin<nEtSumBins;bin++) gEtSumRate->SetPoint(bin,etSumLo+etSumBinWidth*bin,etSumRates->GetBinContent(bin+1)*norm);
  for(int bin=0;bin<nHtSumBins;bin++) gHtSumRate->SetPoint(bin,htSumLo+htSumBinWidth*bin,htSumRates->GetBinContent(bin+1)*norm);
  for(int bin=0;bin<nMetSumBins;bin++) gMetSumRate->SetPoint(bin,metSumLo+metSumBinWidth*bin,metSumRates->GetBinContent(bin+1)*norm);
  for(int bin=0;bin<nMhtSumBins;bin++) gMhtSumRate->SetPoint(bin,mhtSumLo+mhtSumBinWidth*bin,mhtSumRates->GetBinContent(bin+1)*norm);


  TCanvas* c1 = new TCanvas;
  c1->SetLogy();

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

  gEgRate->SetLineWidth(2);
  gEgRate->SetLineColor(kRed);
  gEgRate->GetXaxis()->SetTitle("Threshold [GeV]");
  gEgRate->GetYaxis()->SetTitle("Rate");
  gEgRate->SetMarkerStyle(23);
  gEgRate->SetMarkerColor(kRed);
  gEgRate->GetYaxis()->SetRangeUser(1, 1e7);
  //gEgRate->Draw("APL");
  
  gTauRate->SetLineWidth(2);
  gTauRate->SetLineColor(kBlue);
  gTauRate->GetXaxis()->SetTitle("Threshold [GeV]");
  gTauRate->GetYaxis()->SetTitle("Rate");
  gTauRate->SetMarkerStyle(23);
  gTauRate->SetMarkerColor(kBlue);
  gTauRate->GetYaxis()->SetRangeUser(1, 1e7);
  //gTauRate->Draw("sameAPL");

  gJetRate->SetLineWidth(2);
  gJetRate->SetLineColor(kGreen);
  gJetRate->GetXaxis()->SetTitle("Threshold [GeV]");
  gJetRate->GetYaxis()->SetTitle("Rate");
  gJetRate->SetMarkerStyle(23);
  gJetRate->SetMarkerColor(kGreen);
  gJetRate->GetYaxis()->SetRangeUser(1, 1e7);
  gJetRate->SetTitle("");
  //gJetRate->Draw("sameAPL");

  TMultiGraph *mg = new TMultiGraph();
  mg->Add(gEgRate);
  mg->Add(gTauRate);
  mg->Add(gJetRate);
  mg->SetMinimum(0.1);
  mg->SetMaximum(3E3);
  mg->Draw("APL");
  mg->GetXaxis()->SetTitle("Threshold [GeV]");
  mg->GetYaxis()->SetTitle("Rate [kHz]");
  gPad->Modified();

  TLegend* leg1 = new TLegend(0.5,0.73,0.7,0.88);
  leg1->SetFillColor(0);
  leg1->AddEntry(gEgRate,"EGamma","lp");
  leg1->AddEntry(gTauRate,"Tau","lp");
  leg1->AddEntry(gJetRate,"Jets","lp");
  leg1->SetBorderSize(0);
  leg1->SetFillStyle(0);
  leg1->Draw();


  n3.DrawLatex(0.5, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
  n4.DrawLatex(0.5, 0.55, "Zero Bias");
 
  c1->SaveAs("ratesJetEgTau.pdf");


  TCanvas* c2 = new TCanvas;
  c2->SetLogy();

  gMuRate->SetTitle("");
  gMuRate->SetLineWidth(2);
  gMuRate->SetLineColor(kOrange);
  gMuRate->GetXaxis()->SetTitle("Threshold [GeV]");
  gMuRate->GetYaxis()->SetTitle("Rate");
  gMuRate->SetMarkerStyle(23);
  gMuRate->SetMarkerColor(kOrange);
  gMuRate->GetYaxis()->SetRangeUser(1, 1e2);

  gMuRate->Draw("APL");
  gMuRate->GetXaxis()->SetTitle("Threshold [GeV]");
  gMuRate->GetYaxis()->SetTitle("Rate [kHz]");
  gPad->Modified();


  TLegend* leg2 = new TLegend(0.5,0.73,0.7,0.88);
  leg2->SetFillColor(0);
  leg2->AddEntry(gMuRate,"Muon","lp");
  leg2->SetBorderSize(0);
  leg2->SetFillStyle(0);
  leg2->Draw();
  leg2->Draw();
  n3.DrawLatex(0.5, 0.6, "Run 260627 #sqrt{s} = 13 TeV");
  n4.DrawLatex(0.5, 0.55, "Zero Bias");

  c2->SaveAs("ratesMuon.pdf");

  TCanvas* c3 = new TCanvas;
  c3->SetLogy();

  gEtSumRate->SetLineWidth(2);
  gEtSumRate->SetLineColor(kMagenta);
  gEtSumRate->GetXaxis()->SetTitle("Threshold [GeV]");
  gEtSumRate->GetYaxis()->SetTitle("Rate");
  gEtSumRate->SetMarkerStyle(23);
  gEtSumRate->SetMarkerColor(kMagenta);
  gEtSumRate->GetYaxis()->SetRangeUser(1, 1e7);
  //gEtSumRate->Draw("APL");

  gHtSumRate->SetLineWidth(2);
  gHtSumRate->SetLineColor(kTeal);
  gHtSumRate->GetXaxis()->SetTitle("Threshold [GeV]");
  gHtSumRate->GetYaxis()->SetTitle("Rate");
  gHtSumRate->SetMarkerStyle(23);
  gHtSumRate->SetMarkerColor(kTeal);
  gHtSumRate->GetYaxis()->SetRangeUser(1, 1e7);
  gHtSumRate->SetTitle("");
  //gHtSumRate->Draw("sameAPL");

  TMultiGraph *mgSums = new TMultiGraph();
  mgSums->Add(gEtSumRate);
  mgSums->Add(gHtSumRate);
  mgSums->Draw("APL");
  mgSums->GetXaxis()->SetTitle("Threshold [GeV]");
  mgSums->GetYaxis()->SetTitle("Rate [kHz]");
  gPad->Modified();
  
  TLegend* leg3 = new TLegend(0.7,0.78,0.9,0.88);
  leg3->SetFillColor(0);
  leg3->AddEntry(gEtSumRate,"E_{T}^{total}","lp");
  leg3->AddEntry(gHtSumRate,"H_{T}","lp");
  leg3->SetBorderSize(0);
  leg3->SetFillStyle(0);
  leg3->Draw("same");

  n3.DrawLatex(0.6, 0.4, "Run 260627 #sqrt{s} = 13 TeV");
  n4.DrawLatex(0.6, 0.25, "Zero Bias");
  
  c3->SaveAs("ratesSums.pdf");





  TCanvas* c4 = new TCanvas;
  c4->SetLogy();

  gMetSumRate->SetLineWidth(2);
  gMetSumRate->SetLineColor(kViolet);
  gMetSumRate->GetXaxis()->SetTitle("Threshold [GeV]");
  gMetSumRate->GetYaxis()->SetTitle("Rate");
  gMetSumRate->SetMarkerStyle(23);
  gMetSumRate->SetMarkerColor(kViolet);
  gMetSumRate->GetYaxis()->SetRangeUser(1, 1e7);
  //gEtSumRate->Draw("APL");

  gMhtSumRate->SetLineWidth(2);
  gMhtSumRate->SetLineColor(kOrange);
  gMhtSumRate->GetXaxis()->SetTitle("Threshold [GeV]");
  gMhtSumRate->GetYaxis()->SetTitle("Rate");
  gMhtSumRate->SetMarkerStyle(23);
  gMhtSumRate->SetMarkerColor(kOrange);
  gMhtSumRate->GetYaxis()->SetRangeUser(1, 1e7);
  gMhtSumRate->SetTitle("");
  //gMhtSumRate->Draw("sameAPL");

  TMultiGraph *mgMsums = new TMultiGraph();
  mgMsums->Add(gMetSumRate);
  mgMsums->Add(gMhtSumRate);
  mgMsums->Draw("APL");
  mgMsums->GetXaxis()->SetTitle("Threshold [GeV]");
  mgMsums->GetYaxis()->SetTitle("Rate [kHz]");
  gPad->Modified();
  
  TLegend* leg4 = new TLegend(0.7,0.78,0.9,0.88);
  leg4->SetFillColor(0);
  leg4->AddEntry(gMetSumRate,"E_{T}^{miss}","lp");
  leg4->AddEntry(gMhtSumRate,"H_{T}^{miss}","lp");
  leg4->SetBorderSize(0);
  leg4->SetFillStyle(0);
  leg4->Draw("same");

  n3.DrawLatex(0.3, 0.4, "Run 260627 #sqrt{s} = 13 TeV");
  n4.DrawLatex(0.3, 0.25, "Zero Bias");
  
 
  
  c4->SaveAs("ratesMETMHT.pdf");




  // hJetEt->GetXaxis()->SetTitle("Jet E_{T} [GeV]");
  // hJetEt->GetYaxis()->SetTitle("Rate");
  // hJetEt->SetMarkerStyle(23);
  // hJetEt->SetMarkerColor(kGreen);
  // hJetEt->Scale(norm);
  // hJetEt->Draw("P");
  // c1.Print("hJetEt.pdf");

  // hEgEt->GetXaxis()->SetTitle("EG E_{T} [GeV]");
  // hEgEt->GetYaxis()->SetTitle("Rate");
  // hEgEt->SetMarkerStyle(23);
  // hEgEt->SetMarkerColor(kGreen);
  // hEgEt->Scale(norm);
  // hEgEt->Draw("P");
  // c1.Print("hEgEt.pdf");
  

  // hTauEt->GetXaxis()->SetTitle("Tau E_{T} [GeV]");
  // hTauEt->GetYaxis()->SetTitle("Rate");
  // hTauEt->SetMarkerStyle(23);
  // hTauEt->SetMarkerColor(kGreen);
  // hTauEt->Scale(norm);
  // hTauEt->Draw("P");
  // c1.Print("hTauEt.pdf");

  // hEtSum->GetXaxis()->SetTitle("E^{sum}_{T} [GeV]");
  // hEtSum->GetYaxis()->SetTitle("Rate");
  // hEtSum->SetMarkerStyle(23);
  // hEtSum->SetMarkerColor(kGreen);
  // hEtSum->Scale(norm);
  // hEtSum->Draw("P");
  // c1.Print("hEtSum.pdf");

  // hHtSum->GetXaxis()->SetTitle("H_{T} [GeV]");
  // hHtSum->GetYaxis()->SetTitle("Rate");
  // hHtSum->SetMarkerStyle(23);
  // hHtSum->SetMarkerColor(kGreen);
  // hHtSum->Scale(norm);
  // hHtSum->Draw("P");
  // c1.Print("hHtSum.pdf");





  // jetRates->GetXaxis()->SetTitle("E_{T} [GeV]");
  // jetRates->GetYaxis()->SetTitle("Rate");
  // jetRates->SetMarkerStyle(23);
  // jetRates->SetMarkerColor(kGreen);
  // jetRates->GetXaxis()->SetRangeUser(10,110);
  // jetRates->Scale(norm);
  // jetRates->Draw("P");
 

  // egRates->GetXaxis()->SetTitle("E_{T} [GeV]");
  // egRates->GetYaxis()->SetTitle("Rate");
  // egRates->SetMarkerStyle(23);
  // egRates->SetMarkerColor(kRed);
  // egRates->GetXaxis()->SetRangeUser(10,110);
  // egRates->Scale(norm);
  // egRates->Draw("sameP");

  // tauRates->GetXaxis()->SetTitle("E_{T} [GeV]");
  // tauRates->GetYaxis()->SetTitle("Rate");
  // tauRates->SetMarkerStyle(23);
  // tauRates->SetMarkerColor(kBlue);
  // tauRates->GetXaxis()->SetRangeUser(10,110);
  // tauRates->Scale(norm);
  // tauRates->Draw("sameP");

  // etSumRates->GetXaxis()->SetTitle("E_{T} [GeV]");
  // etSumRates->GetYaxis()->SetTitle("Rate");
  // etSumRates->SetMarkerStyle(23);
  // etSumRates->SetMarkerColor(kViolet);
  // etSumRates->GetXaxis()->SetRangeUser(10,110);
  // etSumRates->Scale(norm);
  // etSumRates->Draw("sameP");

  // htSumRates->GetXaxis()->SetTitle("E_{T} [GeV]");
  // htSumRates->GetYaxis()->SetTitle("Rate");
  // htSumRates->SetMarkerStyle(23);
  // htSumRates->SetMarkerColor(kTeal);
  // htSumRates->GetXaxis()->SetRangeUser(10,110);
  // htSumRates->Scale(norm);
  // htSumRates->Draw("sameP");

  
  // c1.SetLogy();
  // c1.Print("rates.pdf");
 


  //individual plots


  // jetRates->Draw("P");
  // c1.SetLogy();
  // c1.Print("jetRates.pdf"); 
 
  // egRates->Draw("P");
  // c1.SetLogy();
  // c1.Print("egRates.pdf");

  // tauRates->Draw("P");
  // c1.SetLogy();
  // c1.Print("tauRates.pdf");

  // etSumRates->Draw("P");
  // c1.SetLogy();
  // c1.Print("etSumRates.pdf");

  // htSumRates->Draw("P");
  // c1.SetLogy();
  // c1.Print("htSumRates.pdf");


  //c1.SetLogy();
  //c1.Print("rates.pdf");

  
}
