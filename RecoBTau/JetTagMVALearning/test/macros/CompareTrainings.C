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
#include "tdrstyle.C"

TFile *file1, *file2;
TLegend *leg;
string plotname;

void CompareTrainings()
{
  using namespace std;

  setTDRStyle();
  gStyle->SetOptFit(0);
  gStyle->SetOptStat(0);
  
	string dir = "Plots_CompareTrainings";
	
	gSystem->mkdir(dir.c_str());

	TString fileName1 = "../skeleton/DQM_V0001_R000000001__POG__BTAG__Andrea.root";
	file1 = TFile::Open(fileName1);
	TString fileName2 = "DQM_V0001_R000000001__POG__BTAG__ProcLinearFixed.root";
	file2 = TFile::Open(fileName2);

	if (!file1 || !file2) abort();

	TH1F * CSV1_FlavEffVsBEff_DUSG = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/FlavEffVsBEff_DUSG_discr_CSV_GLOBAL");
	TH1F * CSV2_FlavEffVsBEff_DUSG = (TH1F*) file2->Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/FlavEffVsBEff_DUSG_discr_CSV_GLOBAL");


	CSV1_FlavEffVsBEff_DUSG->GetXaxis()->SetTitle("B efficiency");
  //CSV1_FlavEffVsBEff_DUSG->GetXaxis()->SetTitleOffset(1.2);
  CSV1_FlavEffVsBEff_DUSG->GetYaxis()->SetTitle("DUSG efficiency");
  CSV1_FlavEffVsBEff_DUSG->GetYaxis()->SetTitleOffset(1.2);
  TCanvas * Plots1 = new TCanvas("Plots1","");
	Plots1->SetLogy();
	//Plots1->SetGridx();
	//Plots1->SetGridy();
	CSV1_FlavEffVsBEff_DUSG->SetTitle("");
  CSV1_FlavEffVsBEff_DUSG->Draw();
  CSV2_FlavEffVsBEff_DUSG->SetMarkerColor(4);
  CSV2_FlavEffVsBEff_DUSG->Draw("same");  
  
	leg = new TLegend(0.2,0.6,0.45,0.9);
  leg->SetFillColor(0);
  leg->AddEntry(CSV1_FlavEffVsBEff_DUSG,"CSV Andrea","p");
  leg->AddEntry(CSV2_FlavEffVsBEff_DUSG,"CSV ProcLinear fixed","p");
  
  leg->Draw();
	plotname = dir+"/FlavEffVsBEff_DUSG.png";
	Plots1->Print(plotname.c_str());

	TH1F * CSV1_FlavEffVsBEff_C = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/FlavEffVsBEff_C_discr_CSV_GLOBAL");
	TH1F * CSV2_FlavEffVsBEff_C = (TH1F*) file2->Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/FlavEffVsBEff_C_discr_CSV_GLOBAL");

	CSV1_FlavEffVsBEff_C->GetXaxis()->SetTitle("B efficiency");
  //CSV1_FlavEffVsBEff_C->GetXaxis()->SetTitleOffset(1.2);
  CSV1_FlavEffVsBEff_C->GetYaxis()->SetTitle("C efficiency");
  CSV1_FlavEffVsBEff_C->GetYaxis()->SetTitleOffset(1.2);
  TCanvas * Plots2 = new TCanvas("Plots2","");
	Plots2->SetLogy();
	//Plots2->SetGridx();
	//Plots2->SetGridy();
	CSV1_FlavEffVsBEff_C->SetTitle("");
  CSV1_FlavEffVsBEff_C->Draw();
  CSV2_FlavEffVsBEff_C->SetMarkerColor(4);
  CSV2_FlavEffVsBEff_C->Draw("same");  
  
	leg = new TLegend(0.2,0.6,0.45,0.9);
  leg->SetFillColor(0);
  leg->AddEntry(CSV1_FlavEffVsBEff_C,"CSV Andrea","p");
  leg->AddEntry(CSV2_FlavEffVsBEff_C,"CSV ProcLinear fixed","p");
  
  leg->Draw();
	plotname = dir+"/FlavEffVsBEff_C.png";
	Plots2->Print(plotname.c_str());


}
