#include "TROOT.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TF1.h"
#include "TMath.h"
#include "TNtuple.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TCutG.h"
#include "TFile.h"
#include "TString.h"
#include "TH2.h"
#include "TPad.h"
#include "TPaveText.h"
#include "Alignment/OfflineValidation/plugins/TkAlStyle.cc"


using namespace ROOT::Math;

void MultiHistoOverlap_Z(bool switchONfitEta = false, bool switchONfit = false){
  gROOT->Reset();
  if (TkAlStyle::status() == NO_STATUS)
    TkAlStyle::set(INTERNAL);
  gROOT->ForceStyle();

  TString strValidation_label = "Run 2015B (251604-251642)";
  TString strReference_label = "DY 2012 Ideal";

  TFile* file[2];
  file[0] = new TFile("./BiasCheck.root", "read");
  file[1] = new TFile("./BiasCheck_Reference.root", "read");

  int pIndex;
  TH1D* histo[2][7];
  TF1* hfit[2][7];
  TCanvas* c[7];
  for (int i=0; i<7; i++){
    TString cname = Form("c%i", i);
    c[i] = new TCanvas(cname, cname, 8, 30, 800, 800);
  }

  float lxmin = 0.22, lxwidth = 0.38;
  float lymax = 0.9, lywidth = 0.15;
  float lxmax = lxmin + lxwidth;
  float lymin = lymax - lywidth;
  TLegend* leg = TkAlStyle::legend("topleft", 2);

  //----------------- CANVAS C0 --------------//
  pIndex=0;
  c[pIndex]->cd();

  TString histoName[7] ={
    "MassVsPhiPlus/allHistos/meanHisto",
    "MassVsPhiMinus/allHistos/meanHisto",
    "MassVsEtaPlus/allHistos/meanHisto",
    "MassVsEtaMinus/allHistos/meanHisto",
    "MassVsEtaPlusMinusDiff/allHistos/meanHisto",
    "MassVsCosThetaCS/allHistos/meanHisto",
    "MassVsPhiCS/allHistos/meanHisto"
  };
  double minmax_plot[7][2]={ { 0 } };
  for (int iP=0; iP<7; iP++){
    double absMin=9e9;
    double absMax = -9e9;
    for (int f=0; f<2; f++){
      histo[f][iP]=(TH1D*)file[f]->Get(histoName[iP]);

      histo[f][iP]->SetTitle("");

      histo[f][iP]->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
      histo[f][iP]->SetLineWidth(1);
      histo[f][iP]->SetMarkerSize(1.2);
      if (f==0){
        histo[f][iP]->SetLineColor(kBlack);
        histo[f][iP]->SetMarkerColor(kBlack);
        histo[f][iP]->SetMarkerStyle(20);
      }
      else{
        histo[f][iP]->SetLineColor(kRed);
        histo[f][iP]->SetMarkerColor(kRed);
        histo[f][iP]->SetMarkerStyle(1);
      }
      for (int bin=1; bin<=histo[f][iP]->GetNbinsX(); bin++){
        double bincontent = histo[f][iP]->GetBinContent(bin);
        double binerror = histo[f][iP]->GetBinError(bin);
        if (binerror==0 && bincontent==0) continue;
        absMin = min(absMin, bincontent - binerror);
        absMax = max(absMax, bincontent + binerror);
      }
    }
    minmax_plot[iP][0] = absMin/1.1;
    minmax_plot[iP][1] = absMax*1.1;
    for (int f=0; f<2; f++) histo[f][iP]->GetYaxis()->SetRangeUser(minmax_plot[iP][0], minmax_plot[iP][1]);
  }

  // Mass VS muon phi plus -------------------------------
  histo[0][pIndex]->GetXaxis()->SetTitle("#phi_{#mu+}");
  histo[0][pIndex]->GetXaxis()->SetRangeUser(-TMath::Pi(), TMath::Pi());
  histo[0][pIndex]->Draw();
  leg->AddEntry(histo[0][pIndex], strValidation_label, "l");
  //--- fit ----------------------------------------------
  hfit[0][pIndex] = new TF1("cosinusoidal1", "[0]+[1]*cos(x+[2])", -TMath::Pi(), TMath::Pi());
  hfit[0][pIndex]->SetParameter(0, 90.5);
  hfit[0][pIndex]->SetParameter(1, 1.);
  hfit[0][pIndex]->SetParameter(2, 1.);
  hfit[0][pIndex]->SetLineColor(1);
  if (switchONfit){
    histo[0][pIndex]->Fit(hfit[0][pIndex], "R", "same", -TMath::Pi(), TMath::Pi());
    hfit[0][pIndex]->Draw("same");
  }

  //---- 2-------------------------------
  histo[1][pIndex]->Draw("same");
  leg->AddEntry(histo[1][pIndex], strReference_label, "l");
  //--- fit ----------------------------------------------
  hfit[1][pIndex] = new TF1("cosinusoidal2", "[0]+[1]*cos(x+[2])", -TMath::Pi(), TMath::Pi());
  hfit[1][pIndex]->SetParameter(0, 90.5);
  hfit[1][pIndex]->SetParameter(1, 1.);
  hfit[1][pIndex]->SetParameter(2, 1.);
  hfit[1][pIndex]->SetLineColor(2);
  if (switchONfit){
    histo[1][pIndex]->Fit(hfit[1][pIndex], "R", "same", -TMath::Pi(), TMath::Pi());
    hfit[1][pIndex]->Draw("same");
  }

  leg->Draw("same");
  c[pIndex]->RedrawAxis();
  c[pIndex]->Modified();
  c[pIndex]->Update();
  c[pIndex]->SaveAs("MassVsPhiPlus_ALL.png");

  //----------------- CANVAS C1 --------------//
  pIndex=1;
  c[pIndex]->cd();

  // Mass VS muon phi minus -------------------------------
  histo[0][pIndex]->GetXaxis()->SetTitle("#phi_{#mu-}");
  histo[0][pIndex]->GetXaxis()->SetRangeUser(-TMath::Pi(), TMath::Pi());
  histo[0][pIndex]->Draw();
  //--- fit ----------------------------------------------
  hfit[0][pIndex] = new TF1("cosinusoidal1", "[0]+[1]*cos(x+[2])", -TMath::Pi(), TMath::Pi());
  hfit[0][pIndex]->SetParameter(0, 90.5);
  hfit[0][pIndex]->SetParameter(1, 1.);
  hfit[0][pIndex]->SetParameter(2, 1.);
  hfit[0][pIndex]->SetLineColor(1);
  if (switchONfit){
    histo[0][pIndex]->Fit(hfit[0][pIndex], "R", "same", -TMath::Pi(), TMath::Pi());
    hfit[0][pIndex]->Draw("same");
  }


  //---- 2-------------------------------
  histo[1][pIndex]->Draw("same");
  //--- fit ----------------------------------------------
  hfit[1][pIndex] = new TF1("cosinusoidal2", "[0]+[1]*cos(x+[2])", -TMath::Pi(), TMath::Pi());
  hfit[1][pIndex]->SetParameter(0, 90.5);
  hfit[1][pIndex]->SetParameter(1, 1.);
  hfit[1][pIndex]->SetParameter(2, 1.);
  hfit[1][pIndex]->SetLineColor(2);
  if (switchONfit){
    histo[1][pIndex]->Fit(hfit[1][pIndex], "R", "same", -TMath::Pi(), TMath::Pi());
    hfit[1][pIndex]->Draw("same");
  }

  leg->Draw("same");
  c[pIndex]->RedrawAxis();
  c[pIndex]->Modified();
  c[pIndex]->Update();
  c[pIndex]->SaveAs("MassVsPhiMinus_ALL.png");

  //----------------- CANVAS C2 --------------//
  pIndex=2;
  c[pIndex]->cd();

  // Mass VS muon eta plus -------------------------------
  histo[0][pIndex]->GetXaxis()->SetTitle("#eta_{#mu+}");
  histo[0][pIndex]->GetXaxis()->SetRangeUser(-2.6, 2.6);
  histo[0][pIndex]->Draw();
  //--- fit ----------------------------------------------
  hfit[0][pIndex] = new TF1("linear1", "[0]+[1]*x", -2.6, 2.6);
  hfit[0][pIndex]->SetParameter(0, 90.5);
  hfit[0][pIndex]->SetParameter(1, 1.);
  hfit[0][pIndex]->SetLineColor(1);
  if (switchONfitEta){
    histo[0][pIndex]->Fit(hfit[0][pIndex], "R", "same", -2.6, 2.6);
    hfit[0][pIndex]->Draw("same");
  }


  //---- 2-------------------------------
  histo[1][pIndex]->Draw("same");
  //--- fit ----------------------------------------------
  hfit[1][pIndex] = new TF1("linear2", "[0]+[1]*x", -2.6, 2.6);
  hfit[1][pIndex]->SetParameter(0, 90.5);
  hfit[1][pIndex]->SetParameter(1, 1.);
  hfit[1][pIndex]->SetLineColor(2);
  if (switchONfitEta){
    histo[1][pIndex]->Fit(hfit[1][pIndex], "R", "same", -2.6, 2.6);
    hfit[1][pIndex]->Draw("same");
  }

  leg->Draw("same");
  c[pIndex]->RedrawAxis();
  c[pIndex]->Modified();
  c[pIndex]->Update();
  c[pIndex]->SaveAs("MassVsEtaPlus_ALL.png");

  //----------------- CANVAS C3 --------------//
  pIndex=3;
  c[pIndex]->cd();

  // Mass VS muon eta minus  -------------------------------
  histo[0][pIndex]->GetXaxis()->SetTitle("#eta_{#mu-}");
  histo[0][pIndex]->GetXaxis()->SetRangeUser(-2.6, 2.6);
  histo[0][pIndex]->Draw();
  //--- fit ----------------------------------------------
  hfit[0][pIndex] = new TF1("linear1", "[0]+[1]*x", -2.6, 2.6);
  hfit[0][pIndex]->SetParameter(0, 0.);
  hfit[0][pIndex]->SetParameter(1, 0.);
  hfit[0][pIndex]->SetLineColor(1);
  if (switchONfitEta){
    histo[0][pIndex]->Fit(hfit[0][pIndex], "R", "same", -2.6, 2.6);
    hfit[0][pIndex]->Draw("same");
  }


  //---- 2-------------------------------
  histo[1][pIndex]->Draw("same");
  //--- fit ----------------------------------------------
  hfit[1][pIndex] = new TF1("linear2", "[0]+[1]*x", -2.6, 2.6);
  hfit[1][pIndex]->SetParameter(0, 0.);
  hfit[1][pIndex]->SetParameter(1, 0.);
  hfit[1][pIndex]->SetLineColor(2);
  if (switchONfitEta){
    histo[1][pIndex]->Fit(hfit[1][pIndex], "R", "same", -2.6, 2.6);
    hfit[1][pIndex]->Draw("same");
  }

  leg->Draw("same");
  c[pIndex]->RedrawAxis();
  c[pIndex]->Modified();
  c[pIndex]->Update();
  c[pIndex]->SaveAs("MassVsEtaMinus_ALL.png");

  //----------------- CANVAS C4 --------------//
  pIndex=4;
  c[pIndex]->cd();

  // Mass VS muon eta plus - eta minus  -------------------------------
  histo[0][pIndex]->GetXaxis()->SetTitle("#eta_{#mu+} - #eta_{#mu-}");
  histo[0][pIndex]->GetXaxis()->SetRangeUser(-3.2, 3.2);
  histo[0][pIndex]->Draw();
  //--- fit ----------------------------------------------
  hfit[0][pIndex] = new TF1("linear1", "[0]+[1]*x", -3.2, 3.2);
  hfit[0][pIndex]->SetParameter(0, 0.);
  hfit[0][pIndex]->SetParameter(1, 0.);
  hfit[0][pIndex]->SetLineColor(1);
  if (switchONfitEta){
    histo[0][pIndex]->Fit(hfit[0][pIndex], "R", "same", -3.2, 3.2);
    hfit[0][pIndex]->Draw("same");
  }


  //---- 2-------------------------------
  histo[1][pIndex]->Draw("same");
  //--- fit ----------------------------------------------
  hfit[1][pIndex] = new TF1("linear1", "[0]+[1]*x", -3.2, 3.2);
  hfit[1][pIndex]->SetParameter(0, 0.);
  hfit[1][pIndex]->SetParameter(1, 0.);
  hfit[1][pIndex]->SetLineColor(2);
  if (switchONfitEta){
    histo[1][pIndex]->Fit(hfit[1][pIndex], "R", "same", -3.2, 3.2);
    hfit[1][pIndex]->Draw("same");
  }

  leg->Draw("same");
  c[pIndex]->RedrawAxis();
  c[pIndex]->Modified();
  c[pIndex]->Update();
  c[pIndex]->SaveAs("MassVsDeltaEta_ALL.png");


  //----------------- CANVAS C5 --------------//
  pIndex=5;
  c[pIndex]->cd();

  // Mass VS muon cos(theta_CS)  -------------------------------
  histo[0][pIndex]->GetXaxis()->SetTitle("cos #theta_{CS}");
  histo[0][pIndex]->GetXaxis()->SetRangeUser(-1.1, 1.1);
  histo[0][pIndex]->Draw();
  //--- fit ----------------------------------------------
  hfit[0][pIndex] = new TF1("cosinusoidal1", "[0]+[1]*cos(x+[2])", -1.1, 1.1);
  hfit[0][pIndex]->SetParameter(0, 90.5);
  hfit[0][pIndex]->SetParameter(1, 1.);
  hfit[0][pIndex]->SetParameter(2, 1.);
  hfit[0][pIndex]->SetLineColor(6);
  if (switchONfit){
    histo[0][pIndex]->Fit(hfit[0][pIndex], "R", "same", -1.1, 1.1);
    hfit[0][pIndex]->Draw("same");
  }


  //---- 2-------------------------------
  histo[1][pIndex]->Draw("same");
  //--- fit ----------------------------------------------
  hfit[1][pIndex] = new TF1("cosinusoidal2", "[0]+[1]*cos(x+[2])", -1.1, 1.1);
  hfit[1][pIndex]->SetParameter(0, 90.5);
  hfit[1][pIndex]->SetParameter(1, 1.);
  hfit[1][pIndex]->SetParameter(2, 1.);
  hfit[1][pIndex]->SetLineColor(2);
  if (switchONfit){
    histo[1][pIndex]->Fit(hfit[1][pIndex], "R", "same", -1.1, 1.1);
    hfit[1][pIndex]->Draw("same");
  }

  leg->Draw("same");
  c[pIndex]->RedrawAxis();
  c[pIndex]->Modified();
  c[pIndex]->Update();
  c[pIndex]->SaveAs("MassVsCosThetaCS_ALL.png");

  //----------------- CANVAS C6 --------------//
  pIndex=6;
  c[pIndex]->cd();

  // Mass VS muon cos(theta_CS)  -------------------------------
  histo[0][pIndex]->GetXaxis()->SetTitle("#phi_{CS}");
  histo[0][pIndex]->GetXaxis()->SetRangeUser(-TMath::Pi(), TMath::Pi());
  histo[0][pIndex]->Draw();
  //--- fit ----------------------------------------------
  hfit[0][pIndex] = new TF1("cosinusoidal1", "[0]+[1]*cos(x+[2])", -TMath::Pi(), TMath::Pi());
  hfit[0][pIndex]->SetParameter(0, 90.5);
  hfit[0][pIndex]->SetParameter(1, 1.);
  hfit[0][pIndex]->SetParameter(2, 1.);
  hfit[0][pIndex]->SetLineColor(6);
  if (switchONfit){
    histo[0][pIndex]->Fit(hfit[0][pIndex], "R", "same", -TMath::Pi(), TMath::Pi());
    hfit[0][pIndex]->Draw("same");
  }


  //---- 2-------------------------------
  histo[1][pIndex]->Draw("same");
  //--- fit ----------------------------------------------
  hfit[1][pIndex] = new TF1("cosinusoidal2", "[0]+[1]*cos(x+[2])", -TMath::Pi(), TMath::Pi());
  hfit[1][pIndex]->SetParameter(0, 90.5);
  hfit[1][pIndex]->SetParameter(1, 1.);
  hfit[1][pIndex]->SetParameter(2, 1.);
  hfit[1][pIndex]->SetLineColor(2);
  if (switchONfit){
    histo[1][pIndex]->Fit(hfit[1][pIndex], "R", "same", -TMath::Pi(), TMath::Pi());
    hfit[1][pIndex]->Draw("same");
  }

  leg->Draw("same");
  c[pIndex]->RedrawAxis();
  c[pIndex]->Modified();
  c[pIndex]->Update();
  c[pIndex]->SaveAs("MassVsPhiCS_ALL.png");

  for (int i=0; i<7; i++){
    c[i]->Close();
    for (int j=0; j<2; j++) delete hfit[j][i];
  }
  delete leg;
  for(int f=1;f>=0;f--) file[f]->Close();
}
