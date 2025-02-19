void plotL1RelVal(bool printCanvas = false, bool printIndividual = false, bool normHists = true)
{
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);
  gStyle->SetErrorX(0);
  gStyle->SetOptTitle(kFALSE);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetFrameBorderMode(0);
  gStyle->SetPadRightMargin(0.12);
  gStyle->SetCanvasDefH(820);
  gStyle->SetCanvasDefW(760);
  gStyle->SetLabelOffset(0.001,"Y");

  TString dir = "";
  TString l1branch = "l1tau/histos/";

  TString ver1 = "CMSSW_1_6_9";
  //TString ver2 = "CMSSW_1_8_0_pre9";
  TString ver2 = "CMSSW_2_0_0_pre2";

  
  TFile *f1 = new TFile("/uscms/home/chinhan/lpctau/CMSSW_1_6_9/src/UserCode/L1TauAnalyzer/test/relValZTT.root"); 
  TFile *f2 = new TFile("relValZTT.root"); 
  //TFile *f1 = new TFile("/uscms/home/chinhan/lpctau/CMSSW_1_8_0_pre9/src/UserCode/L1TauAnalyzer/test/relValZTT.root"); 
  //TFile *f1 = new TFile("relValZTT.root"); 
  //TFile *f2 = new TFile("relValZTT.root");
  //TFile *f1 = new TFile("test.root"); 
  //TFile *f2 = new TFile("test.root"); 
  
  TCanvas* ca1 = new TCanvas();
  ca1->Divide(3,3); 

  ca1->cd(1);
  //__________________________________________________________________________________________________
  TString lstr = "L1TauEt";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("Incl. L1 #tau E_{T} (GeV)");
  hl1->GetYaxis()->SetTitle("Entries");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  if (normHists) {
    double integ1 = (double)hl1->Integral();
    hl1->Scale(1./integ1);
    double integ2 = (double)hl2->Integral();
    hl2->Scale(1./integ2);
  }
  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));
  
  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca1->cd(2);
  //__________________________________________________________________________________________________
  TString lstr = "L1TauEta";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("Incl. L1 #tau #eta");
  hl1->GetYaxis()->SetTitle("Entries");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  if (normHists) {
    double integ1 = (double)hl1->Integral();
    hl1->Scale(1./integ1);
    double integ2 = (double)hl2->Integral();
    hl2->Scale(1./integ2);
  }
  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca1->cd(3);
  //__________________________________________________________________________________________________
  TString lstr = "L1TauPhi";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("Incl. L1 #tau #phi");
  hl1->GetYaxis()->SetTitle("Entries");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  if (normHists) {
    double integ1 = (double)hl1->Integral();
    hl1->Scale(1./integ1);
    double integ2 = (double)hl2->Integral();
    hl2->Scale(1./integ2);
  }
  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;


  ca1->cd(4);
  //__________________________________________________________________________________________________
  TString lstr = "L1Tau1Et";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("Leading L1 #tau E_{T} (GeV)");
  hl1->GetYaxis()->SetTitle("Entries");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  if (normHists) {
    double integ1 = (double)hl1->Integral();
    hl1->Scale(1./integ1);
    double integ2 = (double)hl2->Integral();
    hl2->Scale(1./integ2);
  }
  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca1->cd(5);
  //__________________________________________________________________________________________________
  TString lstr = "L1Tau1Eta";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("Leading L1 #tau #eta");
  hl1->GetYaxis()->SetTitle("Entries");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  if (normHists) {
    double integ1 = (double)hl1->Integral();
    hl1->Scale(1./integ1);
    double integ2 = (double)hl2->Integral();
    hl2->Scale(1./integ2);
  }
  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca1->cd(6);
  //__________________________________________________________________________________________________
  TString lstr = "L1Tau1Phi";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("Leading L1 #tau #phi");
  hl1->GetYaxis()->SetTitle("Entries");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  if (normHists) {
    double integ1 = (double)hl1->Integral();
    hl1->Scale(1./integ1);
    double integ2 = (double)hl2->Integral();
    hl2->Scale(1./integ2);
  }
  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca1->cd(7);
  //__________________________________________________________________________________________________
  TString lstr = "L1Tau2Et";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("2. L1 #tau E_{T} (GeV)");
  hl1->GetYaxis()->SetTitle("Entries");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  if (normHists) {
    double integ1 = (double)hl1->Integral();
    hl1->Scale(1./integ1);
    double integ2 = (double)hl2->Integral();
    hl2->Scale(1./integ2);
  }
  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca1->cd(8);
  //__________________________________________________________________________________________________
  TString lstr = "L1Tau2Eta";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("2. L1 #tau #eta");
  hl1->GetYaxis()->SetTitle("Entries");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  if (normHists) {
    double integ1 = (double)hl1->Integral();
    hl1->Scale(1./integ1);
    double integ2 = (double)hl2->Integral();
    hl2->Scale(1./integ2);
  }
  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca1->cd(9);
  //__________________________________________________________________________________________________
  TString lstr = "L1Tau2Phi";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("2. L1 #tau #phi");
  hl1->GetYaxis()->SetTitle("Entries");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  if (normHists) {
    double integ1 = (double)hl1->Integral();
    hl1->Scale(1./integ1);
    double integ2 = (double)hl2->Integral();
    hl2->Scale(1./integ2);
  }
  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;


  ////////////////////////////////////////////////////////////////
  
  ca1->cd();
  if (printCanvas) {
    gPad->SaveAs("L1ValidationPlots_c1.gif");
  }

  ////////////////////////////////////////////////////////////////


  TCanvas* ca2 = new TCanvas();
  ca2->Divide(3,3); 


  ca2->cd(1);
  //__________________________________________________________________________________________________
  
  TLegend *legend = new TLegend(0.25, 0.4, 0.87, 0.75);
  legend->SetTextSize(0.045);
  legend->SetFillColor(0);
  legend->SetBorderSize(1);
  //legend->SetHeader("Inclusive L1 Tau Et");
  legend->AddEntry(hl1,ver1,"l");
  legend->AddEntry(hl2,ver2,"pl");
  legend->Draw();
  gPad->Update();


  ca2->cd(2);
  //__________________________________________________________________________________________________
  TString lstr = "L1minusMCTauEt";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("L1 #tau E_{T} - MC Tau_{had}^{vis} E_{T} (GeV)");
  hl1->GetYaxis()->SetTitle("Entries");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  if (normHists) {
    double integ1 = (double)hl1->Integral();
    hl1->Scale(1./integ1);
    double integ2 = (double)hl2->Integral();
    hl2->Scale(1./integ2);
  }
  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca2->cd(3);
  //__________________________________________________________________________________________________
  TString lstr = "L1minusMCoverMCTauEt";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("(L1 #tau E_{T} - MC Tau_{had}^{vis} E_{T})/ MC Tau_{had}^{vis} E_{T}");
  hl1->GetYaxis()->SetTitle("Entries");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  if (normHists) {
    double integ1 = (double)hl1->Integral();
    hl1->Scale(1./integ1);
    double integ2 = (double)hl2->Integral();
    hl2->Scale(1./integ2);
  }
  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;


  ca2->cd(4);
  //__________________________________________________________________________________________________
  TString lstr = "EffMCTauEt";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("(MC Tau_{had}^{vis} E_{T} GeV");
  hl1->GetYaxis()->SetTitle("MCMatch Efficiency");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  //hl1->SetMaximum(1.05*max(hl1->GetMaximum(),hl2->GetMaximum()));
  hl1->SetMaximum(1.01);
  hl1->SetMinimum(0.);

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca2->cd(5);
  //__________________________________________________________________________________________________
  TString lstr = "EffMCTauEta";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("(MC Tau_{had}^{vis} #eta");
  hl1->GetYaxis()->SetTitle("MCMatch Efficiency");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  //  hl1->SetMaximum(1.05*max(hl1->GetMaximum(),hl2->GetMaximum()));
  hl1->SetMaximum(1.01);
  hl1->SetMinimum(0.);

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca2->cd(6);
  //__________________________________________________________________________________________________
  TString lstr = "EffMCTauPhi";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("(MC Tau_{had}^{vis} #phi");
  hl1->GetYaxis()->SetTitle("MCMatch Efficiency");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  //hl1->SetMaximum(1.05*max(hl1->GetMaximum(),hl2->GetMaximum()));
  hl1->SetMaximum(1.01);
  hl1->SetMinimum(0.);

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca2->cd(7);
  //__________________________________________________________________________________________________
  TString lstr = "EffMCPFTauEt";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("(MC Tau_{had}^{vis} E_{T} GeV");
  hl1->GetYaxis()->SetTitle("PFTau-MCMatch Efficiency");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  //hl1->SetMaximum(1.05*max(hl1->GetMaximum(),hl2->GetMaximum()));
  hl1->SetMaximum(1.01);
  hl1->SetMinimum(0.);

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca2->cd(8);
  //__________________________________________________________________________________________________
  TString lstr = "EffMCPFTauEta";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("(MC Tau_{had}^{vis} #eta");
  hl1->GetYaxis()->SetTitle("PFTau-MCMatch Efficiency");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  //  hl1->SetMaximum(1.05*max(hl1->GetMaximum(),hl2->GetMaximum()));
  hl1->SetMaximum(1.01);
  hl1->SetMinimum(0.);

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca2->cd(9);
  //__________________________________________________________________________________________________
  TString lstr = "EffMCPFTauPhi";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("(MC Tau_{had}^{vis} #phi");
  hl1->GetYaxis()->SetTitle("PFTau-MCMatch Efficiency");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  //hl1->SetMaximum(1.05*max(hl1->GetMaximum(),hl2->GetMaximum()));
  hl1->SetMaximum(1.01);
  hl1->SetMinimum(0.);

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  
  ////////////////////////////////////////////////////////////////
  
  ca2->cd();
  if (printCanvas) {
    gPad->SaveAs("L1ValidationPlots_c2.gif");
  }

  ////////////////////////////////////////////////////////////////


  TCanvas* ca3 = new TCanvas();
  ca3->Divide(2,3); 

  ca3->cd(1);
  //__________________________________________________________________________________________________
  TString lstr = "L1SingleTauEffEt";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("L1 Single #tau E_{T} threshold (GeV)");
  hl1->GetYaxis()->SetTitle("Integrated Global Efficiency");
  hl1->SetLineColor(kBlue);

  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca3->cd(2);
  //__________________________________________________________________________________________________
  TString lstr = "L1DoubleTauEffEt";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("L1 Double #tau E_{T} threshold (GeV)");
  hl1->GetYaxis()->SetTitle("Integrated Global Efficiency");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;




  ca3->cd(3);
  //__________________________________________________________________________________________________
  TString lstr = "L1SingleTauEffMCMatchEt";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("L1 Single #tau E_{T} threshold (GeV)");
  hl1->GetYaxis()->SetTitle("Integrated MCMatch Efficiency");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca3->cd(4);
  //__________________________________________________________________________________________________
  TString lstr = "L1DoubleTauEffMCMatchEt";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("L1 Double #tau E_{T} threshold (GeV)");
  hl1->GetYaxis()->SetTitle("Integrated MCMatch Efficiency");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca3->cd(5);
  //__________________________________________________________________________________________________
  TString lstr = "L1SingleTauEffPFMCMatchEt";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("L1 Single #tau E_{T} threshold (GeV)");
  hl1->GetYaxis()->SetTitle("Integrated PFTau-MCMatch Efficiency");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;

  ca3->cd(6);
  //__________________________________________________________________________________________________
  TString lstr = "L1DoubleTauEffPFMCMatchEt";
  TString l1str = l1branch+lstr;
  TH1* hl1 = (TH1F *) (f1->Get(l1str));
  //hl1->SetAxisRange(0.,120.,"X");
  hl1->GetXaxis()->SetTitle("L1 Double #tau E_{T} threshold (GeV)");
  hl1->GetYaxis()->SetTitle("Integrated PFTau-MCMatch Efficiency");
  hl1->SetLineColor(kBlue);
  TH1* hl2 = (TH1F *) (f2->Get(l1str));
  hl2->SetMarkerColor(kBlack);
  hl2->SetMarkerSize(1.);
  hl2->SetLineWidth(1);
  hl2->SetLineColor(kBlack);
  hl2->SetMarkerStyle(kFullDotLarge);

  hl1->SetMaximum(1.1*max(hl1->GetMaximum(),hl2->GetMaximum()));

  hl1->Draw("hist");
  hl2->Draw("e same");

  gPad->Update();
  //gPad->SetLogy();
  if (printIndividual) {
    TString plotFileName = dir+l1str;
    gPad->SaveAs(plotFileName+".gif");
  }
  //return;


  ////////////////////////////////////////////////////////////////
  
  ca3->cd();
  if (printCanvas) {
    gPad->SaveAs("L1ValidationPlots_c3.gif");
  }

  ////////////////////////////////////////////////////////////////

}
