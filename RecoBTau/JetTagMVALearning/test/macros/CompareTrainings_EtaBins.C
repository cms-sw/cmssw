// general
#include "TH1.h"
#include "TH2F.h"
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TF1.h"
#include "TKey.h"
#include "TH1F.h"
#include "TStyle.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TLine.h"
#include "TArrow.h"
#include "TLatex.h"
#include "TMinuit.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TFormula.h"
#include "TAxis.h"

#include <iostream.h>
#include <stdio.h>
#include <fstream.h>
#include <vector.h>
#include "/user/pvmulder/tdrstyle.C"

TFile *file1, *file2;
TLegend *leg;
string plotname;

void CompareTrainings_EtaBins()
{
  using namespace std;

  setTDRStyle();
  gStyle->SetOptFit(0);
  gStyle->SetOptStat(0);
  
	string dir = "Plots_CompareTrainings";
	
	gSystem->mkdir(dir.c_str());

	TString fileName1 = "DQM_V0001_R000000001__POG__BTAG__categories.root";
	file1 = TFile::Open(fileName1);
	//TString fileName2 = "../ProcLinear_fixed/DQM_V0001_R000000001__POG__BTAG__categories.root";
	TString fileName2 = "../Reweighting_default_fixcfg/DQM_V0001_R000000001__POG__BTAG__categories.root";
	file2 = TFile::Open(fileName2);

	if (!file1 || !file2) abort();

	//TH1F * CSV1_FlavEffVsBEff_DUSG = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/FlavEffVsBEff_DUSG_discr_CSV_GLOBAL");
	//TH1F * CSV2_FlavEffVsBEff_DUSG = (TH1F*) file2->Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/FlavEffVsBEff_DUSG_discr_CSV_GLOBAL");
	
	TH1F * CSV2dflight_FlavEffVsBEff_DUSG_ETA_0_1v4 = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/CSV_ETA_0-1v4/FlavEffVsBEff_DUSG_discr_CSV_ETA_0-1v4");
	TH1F * CSV2dflight_FlavEffVsBEff_DUSG_ETA_1v4_2v4 = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/CSV_ETA_1v4-2v4/FlavEffVsBEff_DUSG_discr_CSV_ETA_1v4-2v4");
	TH1F * CSV3dflight_FlavEffVsBEff_DUSG_ETA_0_1v4 = (TH1F*) file2->Get("DQMData/Run 1/Btag/Run summary/CSV_ETA_0-1v4/FlavEffVsBEff_DUSG_discr_CSV_ETA_0-1v4");
	TH1F * CSV3dflight_FlavEffVsBEff_DUSG_ETA_1v4_2v4 = (TH1F*) file2->Get("DQMData/Run 1/Btag/Run summary/CSV_ETA_1v4-2v4/FlavEffVsBEff_DUSG_discr_CSV_ETA_1v4-2v4");
	

	CSV2dflight_FlavEffVsBEff_DUSG_ETA_0_1v4->GetXaxis()->SetTitle("B efficiency");
  //CSV2dflight_FlavEffVsBEff_DUSG_ETA_0_1v4->GetXaxis()->SetTitleOffset(1.2);
  CSV2dflight_FlavEffVsBEff_DUSG_ETA_0_1v4->GetYaxis()->SetTitle("DUSG efficiency");
  CSV2dflight_FlavEffVsBEff_DUSG_ETA_0_1v4->GetYaxis()->SetTitleOffset(1.2);
  TCanvas * Plots1 = new TCanvas("Plots1","");
	Plots1->SetLogy();
	//Plots1->SetGridx();
	//Plots1->SetGridy();
	CSV2dflight_FlavEffVsBEff_DUSG_ETA_0_1v4->SetTitle("");
	CSV2dflight_FlavEffVsBEff_DUSG_ETA_0_1v4->SetMarkerStyle(2);
	CSV2dflight_FlavEffVsBEff_DUSG_ETA_0_1v4->SetMarkerColor(4);
  CSV2dflight_FlavEffVsBEff_DUSG_ETA_0_1v4->Draw();
	CSV2dflight_FlavEffVsBEff_DUSG_ETA_1v4_2v4->SetMarkerColor(4);
  CSV2dflight_FlavEffVsBEff_DUSG_ETA_1v4_2v4->Draw("same");
  CSV3dflight_FlavEffVsBEff_DUSG_ETA_0_1v4->SetMarkerColor(2);
  CSV3dflight_FlavEffVsBEff_DUSG_ETA_0_1v4->SetMarkerStyle(2);
  CSV3dflight_FlavEffVsBEff_DUSG_ETA_0_1v4->Draw("same");
	CSV3dflight_FlavEffVsBEff_DUSG_ETA_1v4_2v4->SetMarkerColor(2);
  CSV3dflight_FlavEffVsBEff_DUSG_ETA_1v4_2v4->Draw("same");  
  
	leg = new TLegend(0.2,0.6,0.6,0.9);
  leg->SetFillColor(0);
  leg->AddEntry(CSV2dflight_FlavEffVsBEff_DUSG_ETA_0_1v4,"CSV LR Eta 0-1.4, default","p");
  leg->AddEntry(CSV2dflight_FlavEffVsBEff_DUSG_ETA_1v4_2v4,"CSV LR Eta 1v4_2v4, default","p");
	leg->AddEntry(CSV3dflight_FlavEffVsBEff_DUSG_ETA_0_1v4,"CSV LR Eta 0-1.4, minor fix","p");
  leg->AddEntry(CSV3dflight_FlavEffVsBEff_DUSG_ETA_1v4_2v4,"CSV LR Eta 1v4_2v4, minor fix","p");

  leg->Draw();
	plotname = dir+"/FlavEffVsBEff_DUSG_cfgfix.png";
	Plots1->Print(plotname.c_str());

	TH1F * CSV2dflight_FlavEffVsBEff_C_ETA_0_1v4 = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/CSV_ETA_0-1v4/FlavEffVsBEff_C_discr_CSV_ETA_0-1v4");
	TH1F * CSV2dflight_FlavEffVsBEff_C_ETA_1v4_2v4 = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/CSV_ETA_1v4-2v4/FlavEffVsBEff_C_discr_CSV_ETA_1v4-2v4");
	TH1F * CSV3dflight_FlavEffVsBEff_C_ETA_0_1v4 = (TH1F*) file2->Get("DQMData/Run 1/Btag/Run summary/CSV_ETA_0-1v4/FlavEffVsBEff_C_discr_CSV_ETA_0-1v4");
	TH1F * CSV3dflight_FlavEffVsBEff_C_ETA_1v4_2v4 = (TH1F*) file2->Get("DQMData/Run 1/Btag/Run summary/CSV_ETA_1v4-2v4/FlavEffVsBEff_C_discr_CSV_ETA_1v4-2v4");
	

	CSV2dflight_FlavEffVsBEff_C_ETA_0_1v4->GetXaxis()->SetTitle("B efficiency");
  //CSV2dflight_FlavEffVsBEff_C_ETA_0_1v4->GetXaxis()->SetTitleOffset(1.2);
  CSV2dflight_FlavEffVsBEff_C_ETA_0_1v4->GetYaxis()->SetTitle("C efficiency");
  CSV2dflight_FlavEffVsBEff_C_ETA_0_1v4->GetYaxis()->SetTitleOffset(1.2);
  TCanvas * Plots2 = new TCanvas("Plots2","");
	Plots2->SetLogy();
	//Plots1->SetGridx();
	//Plots1->SetGridy();
	CSV2dflight_FlavEffVsBEff_C_ETA_0_1v4->SetTitle("");
	CSV2dflight_FlavEffVsBEff_C_ETA_0_1v4->SetMarkerStyle(2);
	CSV2dflight_FlavEffVsBEff_C_ETA_0_1v4->SetMarkerColor(4);
  CSV2dflight_FlavEffVsBEff_C_ETA_0_1v4->Draw();
	CSV2dflight_FlavEffVsBEff_C_ETA_1v4_2v4->SetMarkerColor(4);
  CSV2dflight_FlavEffVsBEff_C_ETA_1v4_2v4->Draw("same");
  CSV3dflight_FlavEffVsBEff_C_ETA_0_1v4->SetMarkerColor(2);
  CSV3dflight_FlavEffVsBEff_C_ETA_0_1v4->SetMarkerStyle(2);
  CSV3dflight_FlavEffVsBEff_C_ETA_0_1v4->Draw("same");
	CSV3dflight_FlavEffVsBEff_C_ETA_1v4_2v4->SetMarkerColor(2);
  CSV3dflight_FlavEffVsBEff_C_ETA_1v4_2v4->Draw("same");  
  
	leg = new TLegend(0.2,0.6,0.6,0.9);
  leg->SetFillColor(0);
  leg->AddEntry(CSV2dflight_FlavEffVsBEff_C_ETA_0_1v4,"CSV LR Eta 0-1.4, default","p");
  leg->AddEntry(CSV2dflight_FlavEffVsBEff_C_ETA_1v4_2v4,"CSV LR Eta 1v4_2v4, default","p");
	leg->AddEntry(CSV3dflight_FlavEffVsBEff_C_ETA_0_1v4,"CSV LR Eta 0-1.4, minor fix","p");
  leg->AddEntry(CSV3dflight_FlavEffVsBEff_C_ETA_1v4_2v4,"CSV LR Eta 1v4_2v4, minor fix","p");

  leg->Draw();
	plotname = dir+"/FlavEffVsBEff_C_cfgfix.png";
	Plots2->Print(plotname.c_str());
}
