{
  gROOT->Reset();
  
 #include "TH1F.h"
 #include "TNtuple.h"
 #include "TLegend.h"
 #include "TCanvas.h"
 #include "TCutG.h"
 #include "TFile.h"
 #include "TH2.h"
 #include "TPad.h"

 using namespace ROOT::Math;
 
 //  gSystem->Load("libRooFit");
 //  using namespace RooFit;
 // preamble
 TPaveText *cmsprel = new TPaveText(0.19, 0.95, 0.95, 0.99, "NDC");
 cmsprel->SetTextSize(0.03);
 cmsprel->SetTextFont(42);
 cmsprel->SetFillColor(0);
 cmsprel->SetBorderSize(0);
 cmsprel->SetMargin(0.01);
 cmsprel->SetTextAlign(12); // align left
 TString text = "CMS Preliminary 2012";
 cmsprel->AddText(0.01,0.5,text);  
 TString text2 = "#sqrt{s} = 8 TeV  Run2012A TkAlZMuMu |#eta_{#mu}|<2.4";               
 cmsprel->AddText(0.4, 0.6, text2);

 TFile *file1  = new TFile("./BiasCheck.root");
 TFile *file2  = new TFile("./BiasCheck_Reference.root");

 bool switchONfit= false;
 bool switchONfitEta= false;
  
 gROOT->LoadMacro("tdrstyle.C"); 
 setTDRStyle();

 TCanvas* c0 = new TCanvas("c0", "c0",50, 20, 800,600);
 TCanvas* c1 = new TCanvas("c1", "c1",50, 20, 800,600);
 TCanvas* c2 = new TCanvas("c2", "c2",50, 20, 800,600);
 TCanvas* c3 = new TCanvas("c3", "c3",50, 20, 800,600);
 TCanvas* c4 = new TCanvas("c4", "c4",50, 20, 800,600);
 TCanvas* c5 = new TCanvas("c5", "c5",50, 20, 800,600);
 TCanvas* c6 = new TCanvas("c6", "c6",50, 20, 800,600);


 leg = new TLegend(0.65,0.8,0.93,0.9);  // upper right
 leg->SetBorderSize(1);
 leg->SetFillColor(0);
 leg->SetTextFont(42);

 //----------------- CANVAS C0 --------------//
 c0->SetFillColor(0);  
 c0->cd();

// Mass VS muon phi plus -------------------------------
 histo1=(TH1D*)file1->Get("MassVsPhiPlus/allHistos/meanHisto");
 histo1->SetLineColor(1);
 histo1->SetMarkerColor(1);
 histo1->SetMarkerStyle(20); //r.c.
 histo1->SetMarkerSize(1.5);
 histo1->GetXaxis()->SetTitle("positive muon #phi (rad) ");
 histo1->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
 histo1->GetYaxis()->SetRangeUser(89.5,92.0);
 histo1->GetXaxis()->SetRangeUser(-3.14,3.14);
 histo1->Draw();
 leg->AddEntry(histo1,"this validation","L");  
 //--- fit ----------------------------------------------
 TF1 * f1 = 0;
 f1 = new TF1("cosinusoidal1", "[0]+[1]*cos(x+[2])", -3.14, 3.14);
 f1->SetParameter(0, 90.5);
 f1->SetParameter(1, 1.);
 f1->SetParameter(2, 1.);
 f1->SetLineColor(1);  
 if (switchONfit){
   histo1->Fit(f1,"R","same", -3.14, 3.14);
   f1->Draw("same");
 }

 //---- 2-------------------------------
 histo2=(TH1D*)file2->Get("MassVsPhiPlus/allHistos/meanHisto");
 histo2->SetLineColor(2);
 histo2->SetMarkerColor(2);
 histo2->Draw("same");
 leg->AddEntry(histo2,"reference","L");   
 //--- fit ----------------------------------------------
 TF1 * f2 = 0;
 f2 = new TF1("cosinusoidal2", "[0]+[1]*cos(x+[2])", -3.14, 3.14);
 f2->SetParameter(0, 90.5);
 f2->SetParameter(1, 1.);
 f2->SetParameter(2, 1.);
 f2->SetLineColor(2);
 if (switchONfit){
   histo2->Fit(f2,"R","same", -3.14, 3.14);
   f2->Draw("same");
 }


 cmsprel->Draw("same");
 leg->Draw("same");

 c0->SaveAs("MassVsPhiPlus_ALL.png"); 

 //----------------- CANVAS C1 --------------//
 c1->SetFillColor(0);
 c1->cd();

// Mass VS muon phi minus -------------------------------
 histo1=(TH1D*)file1->Get("MassVsPhiMinus/allHistos/meanHisto");
 histo1->SetLineColor(1);
 histo1->SetMarkerColor(1);
 histo1->SetMarkerStyle(20); //r.c.
 histo1->SetMarkerSize(1.5); //r.c.
 histo1->GetXaxis()->SetTitle("negative muon #phi");
 histo1->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
 histo1->GetYaxis()->SetRangeUser(89.5,92.0);
 histo1->GetXaxis()->SetRangeUser(-3.14,3.14);
 histo1->Draw();
 //--- fit ----------------------------------------------
 TF1 * f1 = 0;
 f1 = new TF1("cosinusoidal1", "[0]+[1]*cos(x+[2])", -3.14, 3.14);
 f1->SetParameter(0, 90.5);
 f1->SetParameter(1, 1.);
 f1->SetParameter(2, 1.);
 f1->SetLineColor(1);  
 if (switchONfit){
   histo1->Fit(f1,"R","same", -3.14, 3.14);
   f1->Draw("same");
 }


 //---- 2-------------------------------
 histo2=(TH1D*)file2->Get("MassVsPhiMinus/allHistos/meanHisto");
 histo2->SetLineColor(2);
 histo2->SetMarkerColor(2);
 histo2->Draw("same");
 //--- fit ----------------------------------------------
 TF1 * f2 = 0;
 f2 = new TF1("cosinusoidal2", "[0]+[1]*cos(x+[2])", -3.14, 3.14);
 f2->SetParameter(0, 90.5);
 f2->SetParameter(1, 1.);
 f2->SetParameter(2, 1.);
 f2->SetLineColor(2);
 if (switchONfit){
   histo2->Fit(f2,"R","same", -3.14, 3.14);
   f2->Draw("same");
 }

 cmsprel->Draw("same");
 leg->Draw("same");

 c1->SaveAs("MassVsPhiMinus_ALL.png"); 

 //----------------- CANVAS C2 --------------//
 c2->SetFillColor(0);
 c2->cd();

// Mass VS muon eta plus -------------------------------
 histo1=(TH1D*)file1->Get("MassVsEtaPlus/allHistos/meanHisto");
 histo1->SetLineColor(1);
 histo1->SetMarkerColor(1);
 histo1->SetMarkerStyle(20); //r.c.
 histo1->SetMarkerSize(1.5); //r.c.
 histo1->GetXaxis()->SetTitle("positive muon #eta");
 histo1->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
 histo1->GetYaxis()->SetRangeUser(89.5,92.0);
 histo1->GetXaxis()->SetRangeUser(-2.6,2.6);
 histo1->Draw();
 //--- fit ----------------------------------------------
 TF1 * f1 = 0;
 f1 = new TF1("linear1", "[0]+[1]*x", -2.6, 2.6);
 f1->SetParameter(0, 90.5);
 f1->SetParameter(1, 1.);
 f1->SetLineColor(1);  
 if (switchONfitEta){
   histo1->Fit(f1,"R","same", -2.6, 2.6);
   f1->Draw("same");
 }


 //---- 2-------------------------------
 histo2=(TH1D*)file2->Get("MassVsEtaPlus/allHistos/meanHisto");
 histo2->SetLineColor(2);
 histo2->SetMarkerColor(2);
 histo2->Draw("same");
 //--- fit ----------------------------------------------
 TF1 * f2 = 0;
 f2 = new TF1("linear2", "[0]+[1]*x", -2.6, 2.6);
 f2->SetParameter(0, 90.5);
 f2->SetParameter(1, 1.);
 f2->SetLineColor(2);
 if (switchONfitEta){
   histo2->Fit(f2,"R","same", -2.6, 2.6);
   f2->Draw("same");
 }

 cmsprel->Draw("same");
 leg->Draw("same");

 c2->SaveAs("MassVsEtaPlus_ALL.png"); 

 //----------------- CANVAS C3 --------------//
 c3->SetFillColor(0);
 c3->cd();

// Mass VS muon eta minus  -------------------------------
 histo1=(TH1D*)file1->Get("MassVsEtaMinus/allHistos/meanHisto");  
 histo1->SetLineColor(1);
 histo1->SetMarkerColor(1);
 histo1->SetMarkerStyle(20);
 histo1->SetMarkerSize(1.5);
 histo1->GetXaxis()->SetTitle("negative muon #eta"); 
 histo1->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
 histo1->GetYaxis()->SetRangeUser(89.5,92.0);
 histo1->GetXaxis()->SetRangeUser(-2.6,2.6);
 histo1->Draw();
 //--- fit ----------------------------------------------
 TF1 * f1 = 0;
 f1 = new TF1("linear1", "[0]+[1]*x", -2.6, 2.6);
 f1->SetParameter(0, 0.);
 f1->SetParameter(1, 0.);
 f1->SetLineColor(1);  
 if (switchONfitEta){
   histo1->Fit(f1,"R","same", -2.6, 2.6);
   f1->Draw("same");
 }


 //---- 2-------------------------------
 histo2=(TH1D*)file2->Get("MassVsEtaMinus/allHistos/meanHisto");  
 histo2->SetLineColor(2);
 histo2->SetMarkerColor(2);
 histo2->Draw("same");
 //--- fit ----------------------------------------------
 TF1 * f2 = 0;
 f2 = new TF1("linear2", "[0]+[1]*x", -2.6, 2.6);
 f2->SetParameter(0, 0.);
 f2->SetParameter(1, 0.);
 f2->SetLineColor(2);
 if (switchONfitEta){
   histo2->Fit(f2,"R","same", -2.6, 2.6);
   f2->Draw("same");
 }

 cmsprel->Draw("same");
 leg->Draw("same");

 c3->SaveAs("MassVsEtaMinus_ALL.png"); 

 //----------------- CANVAS C4 --------------//
 c4->SetFillColor(0);
 c4->cd();

// Mass VS muon eta plus - eta minus  -------------------------------
 histo1=(TH1D*)file1->Get("MassVsEtaPlusMinusDiff/allHistos/meanHisto");  
 histo1->SetLineColor(1);
 histo1->SetMarkerColor(1);
 histo1->SetMarkerStyle(20);
 histo1->SetMarkerSize(1.5);
 histo1->GetXaxis()->SetTitle("#eta pos - #eta neg"); 
 histo1->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
 histo1->GetYaxis()->SetRangeUser(89.5,92.0);
 histo1->GetXaxis()->SetRangeUser(-3.2,3.2);
 histo1->Draw();
 //--- fit ----------------------------------------------
 TF1 * f1 = 0;
 f1 = new TF1("linear1", "[0]+[1]*x", -3.2, 3.2);
 f1->SetParameter(0, 0.);
 f1->SetParameter(1, 0.);
 f1->SetLineColor(1);  
 if (switchONfitEta){
   histo1->Fit(f1,"R","same", -3.2, 3.2);
   f1->Draw("same");
 }


 //---- 2-------------------------------
 histo2=(TH1D*)file2->Get("MassVsEtaPlusMinusDiff/allHistos/meanHisto");  
 histo2->SetLineColor(2);
 histo2->SetMarkerColor(2);
 histo2->Draw("same");
 //--- fit ----------------------------------------------
 TF1 * f2 = 0;
 f2 = new TF1("linear1", "[0]+[1]*x", -3.2, 3.2);
 f2->SetParameter(0, 0.);
 f2->SetParameter(1, 0.);
 f2->SetLineColor(2);
 if (switchONfitEta){
   histo2->Fit(f2,"R","same", -3.2, 3.2);
   f2->Draw("same");
 }

 cmsprel->Draw("same");
 leg->Draw("same");

 c4->SaveAs("MassVsDeltaEta_ALL.png"); 


 //----------------- CANVAS C5 --------------//
 c5->SetFillColor(0);
 c5->cd();

// Mass VS muon cos(theta_CS)  -------------------------------
 histo1=(TH1D*)file1->Get("MassVsCosThetaCS/allHistos/meanHisto");  
 histo1->SetLineColor(1);
 histo1->SetMarkerColor(1);
 histo1->SetMarkerStyle(20);
 histo1->SetMarkerSize(1.5);
 histo1->GetXaxis()->SetTitle("cos#theta_{CS}"); 
 histo1->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
 histo1->GetYaxis()->SetRangeUser(89.5,92);
 histo1->GetXaxis()->SetRangeUser(-1.1,1.1);
 histo1->Draw();
 //--- fit ----------------------------------------------
 TF1 * f1 = 0; 
 f1 = new TF1("cosinusoidal1", "[0]+[1]*cos(x+[2])", -1.1, 1.1);
 f1->SetParameter(0, 90.5);
 f1->SetParameter(1, 1.);
 f1->SetParameter(2, 1.);
 f1->SetLineColor(6);
 if (switchONfit){
   histo1->Fit(f1,"R","same", -1.1, 1.1); 
   f1->Draw("same");
 }


 //---- 2-------------------------------
 histo2=(TH1D*)file2->Get("MassVsCosThetaCS/allHistos/meanHisto");  
 histo2->SetLineColor(2);
 histo2->SetMarkerColor(2);
 histo2->Draw("same");
 //--- fit ----------------------------------------------
 TF1 * f2 = 0; 
 f2 = new TF1("cosinusoidal2", "[0]+[1]*cos(x+[2])", -1.1, 1.1);
 f2->SetParameter(0, 90.5);
 f2->SetParameter(1, 1.);
 f2->SetParameter(2, 1.);
 f2->SetLineColor(2);
 if (switchONfit){
   histo2->Fit(f2,"R","same", -1.1, 1.1); 
   f2->Draw("same");
 }

 cmsprel->Draw("same");
 leg->Draw("same");

 c5->SaveAs("MassVsCosThetaCS_ALL.png"); 

 //----------------- CANVAS C6 --------------//
 c6->SetFillColor(0);
 c6->cd();

// Mass VS muon cos(theta_CS)  -------------------------------
 histo1=(TH1D*)file1->Get("MassVsPhiCS/allHistos/meanHisto");  
 histo1->SetLineColor(1);
 histo1->SetMarkerColor(1);
 histo1->SetMarkerStyle(20);
 histo1->SetMarkerSize(1.5);
 histo1->GetXaxis()->SetTitle("#phi_{CS}"); 
 histo1->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
 histo1->GetYaxis()->SetRangeUser(89.5,92);
 histo1->GetXaxis()->SetRangeUser(-3.14,3.14);
 histo1->Draw();
 //--- fit ----------------------------------------------
 TF1 * f1 = 0; 
 f1 = new TF1("cosinusoidal1", "[0]+[1]*cos(x+[2])", -3.14, 3.14);
 f1->SetParameter(0, 90.5);
 f1->SetParameter(1, 1.);
 f1->SetParameter(2, 1.);
 f1->SetLineColor(6);
 if (switchONfit){
   histo1->Fit(f1,"R","same", -3.14, 3.14); 
   f1->Draw("same");
 }


 //---- 2-------------------------------
 histo2=(TH1D*)file2->Get("MassVsPhiCS/allHistos/meanHisto");  
 histo2->SetLineColor(2);
 histo2->SetMarkerColor(2);
 histo2->Draw("same");
 //--- fit ----------------------------------------------
 TF1 * f2 = 0; 
 f2 = new TF1("cosinusoidal2", "[0]+[1]*cos(x+[2])", -3.14, 3.14);
 f2->SetParameter(0, 90.5);
 f2->SetParameter(1, 1.);
 f2->SetParameter(2, 1.);
 f2->SetLineColor(2);
 if (switchONfit){
   histo2->Fit(f2,"R","same", -3.14, 3.14); 
   f2->Draw("same");
 }

 cmsprel->Draw("same");
 leg->Draw("same");

 c6->SaveAs("MassVsPhiCS_ALL.png"); 

}
