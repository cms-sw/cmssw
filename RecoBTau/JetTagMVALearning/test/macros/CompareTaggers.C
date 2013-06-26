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
#include "./tdrstyle.C"

TFile *file1;
TLegend *leg;
string plotname;

void CompareTaggers()
{
  using namespace std;

  setTDRStyle();
  gStyle->SetOptFit(0);
  gStyle->SetOptStat(0);
  
	string dir = "PlotsEvaluation_categories";
	
	gSystem->mkdir(dir.c_str());

	TString fileName1 = "DQM_V0001_R000000001__POG__BTAG__categories.root";
	file1 = TFile::Open(fileName1);

	if (!file1) abort();

/*	TH1F * TagInfo_Sip3DSig2nd_all = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/IPTag_GLOBAL/ips2_3D_IPTag_GLOBALALL");
	TH1F * TCHE_Sip3DSig2nd_all = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/TCHE_GLOBAL/discr_TCHE_GLOBALALL");
//	TH1F * MVA_Sip3DSig2nd_all = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/trackSip3dSig_2nd_GLOBAL/discr_trackSip3dSig_2nd_GLOBALALL");
	TH1F * MVA_Sip3DSig2nd_all = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/discr_CSV_GLOBALALL");

	TagInfo_Sip3DSig2nd_all->SetTitle("Sip3DSig2nd all jets");
	TagInfo_Sip3DSig2nd_all->GetXaxis()->SetTitle("discriminator");
//  TagInfo_Sip3DSig2nd_all->GetXaxis()->SetTitleOffset(1.2);
  TagInfo_Sip3DSig2nd_all->GetYaxis()->SetTitle("");
  TagInfo_Sip3DSig2nd_all->GetYaxis()->SetTitleOffset(1.2);
  TagInfo_Sip3DSig2nd_all->SetMaximum(7000);
	TagInfo_Sip3DSig2nd_all->GetXaxis()->SetRangeUser(-10,35);
	TCanvas * Plotsa = new TCanvas("Plotsa","");
  TCHE_Sip3DSig2nd_all->SetLineColor(4);
  MVA_Sip3DSig2nd_all->SetLineColor(2);
	TagInfo_Sip3DSig2nd_all->SetLineWidth(2);
  TCHE_Sip3DSig2nd_all->SetLineWidth(2);
  MVA_Sip3DSig2nd_all->SetLineWidth(2);
  TagInfo_Sip3DSig2nd_all->Draw();
  TCHE_Sip3DSig2nd_all->Draw("same");
  MVA_Sip3DSig2nd_all->Draw("same");
	leg = new TLegend(0.65,0.78,0.92,0.9);
  leg->SetFillColor(0);
  leg->AddEntry(TCHE_Sip3DSig2nd_all,"TCHE","l");
  leg->AddEntry(TagInfo_Sip3DSig2nd_all,"TagInfo","l");
  leg->AddEntry(MVA_Sip3DSig2nd_all,"MVA","l");
  leg->Draw();
	plotname = dir+"/discriminator.png";
	Plotsa->Print(plotname.c_str());

	TH1F * TagInfo_Sip3DSig2nd_b = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/IPTag_GLOBAL/ips2_3D_IPTag_GLOBALB");
	TH1F * TCHE_Sip3DSig2nd_b = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/TCHE_GLOBAL/discr_TCHE_GLOBALB");
//	TH1F * MVA_Sip3DSig2nd_b = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/trackSip3dSig_2nd_GLOBAL/discr_trackSip3dSig_2nd_GLOBALB");
	TH1F * MVA_Sip3DSig2nd_b = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/discr_CSV_GLOBALB");

	TagInfo_Sip3DSig2nd_b->SetTitle("Sip3DSig2nd B jets");
	TagInfo_Sip3DSig2nd_b->GetXaxis()->SetTitle("discriminator");
//  TagInfo_Sip3DSig2nd_b->GetXaxis()->SetTitleOffset(1.2);
  TagInfo_Sip3DSig2nd_b->GetYaxis()->SetTitle("");
  TagInfo_Sip3DSig2nd_b->GetYaxis()->SetTitleOffset(1.2);
  TagInfo_Sip3DSig2nd_b->SetMaximum(1000);
	TagInfo_Sip3DSig2nd_b->GetXaxis()->SetRangeUser(-10,35);
	TCanvas * Plotsb = new TCanvas("Plotsb","");
  TCHE_Sip3DSig2nd_b->SetLineColor(4);
  MVA_Sip3DSig2nd_b->SetLineColor(2);
	TagInfo_Sip3DSig2nd_b->SetLineWidth(2);
  TCHE_Sip3DSig2nd_b->SetLineWidth(2);
  MVA_Sip3DSig2nd_b->SetLineWidth(2);
  TagInfo_Sip3DSig2nd_b->Draw();
  TCHE_Sip3DSig2nd_b->Draw("same");
  MVA_Sip3DSig2nd_b->Draw("same");
	leg = new TLegend(0.65,0.78,0.92,0.9);
  leg->SetFillColor(0);
  leg->AddEntry(TCHE_Sip3DSig2nd_b,"TCHE","l");
  leg->AddEntry(TagInfo_Sip3DSig2nd_b,"TagInfo","l");
  leg->AddEntry(MVA_Sip3DSig2nd_b,"MVA","l");
  leg->Draw();
	plotname = dir+"/discriminator_b.png";
	Plotsb->Print(plotname.c_str());

	TH1F * TagInfo_Sip3DSig2nd_c = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/IPTag_GLOBAL/ips2_3D_IPTag_GLOBALC");
	TH1F * TCHE_Sip3DSig2nd_c = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/TCHE_GLOBAL/discr_TCHE_GLOBALC");
//	TH1F * MVA_Sip3DSig2nd_c = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/trackSip3dSig_2nd_GLOBAL/discr_trackSip3dSig_2nd_GLOBALC");
	TH1F * MVA_Sip3DSig2nd_c = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/discr_CSV_GLOBALC");

	TagInfo_Sip3DSig2nd_c->SetTitle("Sip3DSig2nd C jets");
	TagInfo_Sip3DSig2nd_c->GetXaxis()->SetTitle("discriminator");
//  TagInfo_Sip3DSig2nd_c->GetXaxis()->SetTitleOffset(1.2);
  TagInfo_Sip3DSig2nd_c->GetYaxis()->SetTitle("");
  TagInfo_Sip3DSig2nd_c->GetYaxis()->SetTitleOffset(1.2);
  TagInfo_Sip3DSig2nd_c->SetMaximum(1000);
	TagInfo_Sip3DSig2nd_c->GetXaxis()->SetRangeUser(-10,35);
	TCanvas * Plotsc = new TCanvas("Plotsc","");
  TCHE_Sip3DSig2nd_c->SetLineColor(4);
  MVA_Sip3DSig2nd_c->SetLineColor(2);
	TagInfo_Sip3DSig2nd_c->SetLineWidth(2);
  TCHE_Sip3DSig2nd_c->SetLineWidth(2);
  MVA_Sip3DSig2nd_c->SetLineWidth(2);
  TagInfo_Sip3DSig2nd_c->Draw();
  TCHE_Sip3DSig2nd_c->Draw("same");
  MVA_Sip3DSig2nd_c->Draw("same");
	leg = new TLegend(0.65,0.78,0.92,0.9);
  leg->SetFillColor(0);
  leg->AddEntry(TCHE_Sip3DSig2nd_c,"TCHE","l");
  leg->AddEntry(TagInfo_Sip3DSig2nd_c,"TagInfo","l");
  leg->AddEntry(MVA_Sip3DSig2nd_c,"MVA","l");
  leg->Draw();
	plotname = dir+"/discriminator_c.png";
	Plotsc->Print(plotname.c_str());

	TH1F * TagInfo_Sip3DSig2nd_dusg = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/IPTag_GLOBAL/ips2_3D_IPTag_GLOBALDUSG");
	TH1F * TCHE_Sip3DSig2nd_dusg = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/TCHE_GLOBAL/discr_TCHE_GLOBALDUSG");
//	TH1F * MVA_Sip3DSig2nd_dusg = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/trackSip3dSig_2nd_GLOBAL/discr_trackSip3dSig_2nd_GLOBALDUSG");
	TH1F * MVA_Sip3DSig2nd_dusg = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/discr_CSV_GLOBALDUSG");

	TagInfo_Sip3DSig2nd_dusg->SetTitle("Sip3DSig2nd DUSG jets");
	TagInfo_Sip3DSig2nd_dusg->GetXaxis()->SetTitle("discriminator");
//  TagInfo_Sip3DSig2nd_dusg->GetXaxis()->SetTitleOffset(1.2);
  TagInfo_Sip3DSig2nd_dusg->GetYaxis()->SetTitle("");
  TagInfo_Sip3DSig2nd_dusg->GetYaxis()->SetTitleOffset(1.2);
  TagInfo_Sip3DSig2nd_dusg->SetMaximum(5000);
	TagInfo_Sip3DSig2nd_dusg->GetXaxis()->SetRangeUser(-10,35);
	TCanvas * Plotsd = new TCanvas("Plotsd","");
  TCHE_Sip3DSig2nd_dusg->SetLineColor(4);
  MVA_Sip3DSig2nd_dusg->SetLineColor(2);
	TagInfo_Sip3DSig2nd_dusg->SetLineWidth(2);
  TCHE_Sip3DSig2nd_dusg->SetLineWidth(2);
  MVA_Sip3DSig2nd_dusg->SetLineWidth(2);
  TagInfo_Sip3DSig2nd_dusg->Draw();
  TCHE_Sip3DSig2nd_dusg->Draw("same");
  MVA_Sip3DSig2nd_dusg->Draw("same");
	leg = new TLegend(0.65,0.78,0.92,0.9);
  leg->SetFillColor(0);
  leg->AddEntry(TCHE_Sip3DSig2nd_dusg,"TCHE","l");
  leg->AddEntry(TagInfo_Sip3DSig2nd_dusg,"TagInfo","l");
  leg->AddEntry(MVA_Sip3DSig2nd_dusg,"MVA","l");
  leg->Draw();
	plotname = dir+"/discriminator_dusg.png";
	Plotsd->Print(plotname.c_str());
*/
	
	TH1F * TCHE_FlavEffVsBEff_DUSG = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/TCHE_GLOBAL/FlavEffVsBEff_DUSG_discr_TCHE_GLOBAL");
	TH1F * TCHP_FlavEffVsBEff_DUSG = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/TCHP_GLOBAL/FlavEffVsBEff_DUSG_discr_TCHP_GLOBAL");
	TH1F * MVA_FlavEffVsBEff_DUSG = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/FlavEffVsBEff_DUSG_discr_CSV_GLOBAL");
	TH1F * SSVHE_FlavEffVsBEff_DUSG = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/SSVHE_GLOBAL/FlavEffVsBEff_DUSG_discr_SSVHE_GLOBAL");
	TH1F * SSVHP_FlavEffVsBEff_DUSG = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/SSVHP_GLOBAL/FlavEffVsBEff_DUSG_discr_SSVHP_GLOBAL");


	TCHE_FlavEffVsBEff_DUSG->GetXaxis()->SetTitle("B efficiency");
  //TCHE_FlavEffVsBEff_DUSG->GetXaxis()->SetTitleOffset(1.2);
  TCHE_FlavEffVsBEff_DUSG->GetYaxis()->SetTitle("DUSG efficiency");
  TCHE_FlavEffVsBEff_DUSG->GetYaxis()->SetTitleOffset(1.2);
  TCanvas * Plots1 = new TCanvas("Plots1","");
	Plots1->SetLogy();
	//Plots1->SetGridx();
	//Plots1->SetGridy();
	TCHE_FlavEffVsBEff_DUSG->SetTitle("");

  TCHE_FlavEffVsBEff_DUSG->Draw();
  MVA_FlavEffVsBEff_DUSG->SetMarkerColor(4);
  MVA_FlavEffVsBEff_DUSG->Draw("same");  
  TCHP_FlavEffVsBEff_DUSG->SetMarkerColor(2);
  TCHP_FlavEffVsBEff_DUSG->Draw("same");  
  SSVHE_FlavEffVsBEff_DUSG->SetMarkerColor(3);
  SSVHE_FlavEffVsBEff_DUSG->Draw("same");  
  SSVHP_FlavEffVsBEff_DUSG->SetMarkerColor(5);
  SSVHP_FlavEffVsBEff_DUSG->Draw("same");
  
	leg = new TLegend(0.2,0.6,0.45,0.9);
  leg->SetFillColor(0);
  leg->AddEntry(TCHE_FlavEffVsBEff_DUSG,"TCHE","p");
  leg->AddEntry(TCHP_FlavEffVsBEff_DUSG,"TCHP","p");
  leg->AddEntry(SSVHE_FlavEffVsBEff_DUSG,"SSVHE","p");
  leg->AddEntry(SSVHP_FlavEffVsBEff_DUSG,"SSVHP","p");
  leg->AddEntry(MVA_FlavEffVsBEff_DUSG,"MVA","p");
  
  leg->Draw();
	plotname = dir+"/FlavEffVsBEff_DUSG.png";
	Plots1->Print(plotname.c_str());


	
	TH1F * TCHE_FlavEffVsBEff_C = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/TCHE_GLOBAL/FlavEffVsBEff_C_discr_TCHE_GLOBAL");
	TH1F * TCHP_FlavEffVsBEff_C = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/TCHP_GLOBAL/FlavEffVsBEff_C_discr_TCHP_GLOBAL");
	TH1F * MVA_FlavEffVsBEff_C = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/CSV_GLOBAL/FlavEffVsBEff_C_discr_CSV_GLOBAL");
	TH1F * SSVHE_FlavEffVsBEff_C = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/SSVHE_GLOBAL/FlavEffVsBEff_C_discr_SSVHE_GLOBAL");
	TH1F * SSVHP_FlavEffVsBEff_C = (TH1F*) file1->Get("DQMData/Run 1/Btag/Run summary/SSVHP_GLOBAL/FlavEffVsBEff_C_discr_SSVHP_GLOBAL");


	TCHE_FlavEffVsBEff_C->GetXaxis()->SetTitle("B efficiency");
  //TCHE_FlavEffVsBEff_C->GetXaxis()->SetTitleOffset(1.2);
  TCHE_FlavEffVsBEff_C->GetYaxis()->SetTitle("C efficiency");
  TCHE_FlavEffVsBEff_C->GetYaxis()->SetTitleOffset(1.2);
  TCanvas * Plots2 = new TCanvas("Plots2","");
	Plots2->SetLogy();
	//Plots2->SetGridx();
	//Plots2->SetGridy();
	TCHE_FlavEffVsBEff_C->SetTitle("");

  TCHE_FlavEffVsBEff_C->Draw();
  MVA_FlavEffVsBEff_C->SetMarkerColor(4);
  MVA_FlavEffVsBEff_C->Draw("same");  
  TCHP_FlavEffVsBEff_C->SetMarkerColor(2);
  TCHP_FlavEffVsBEff_C->Draw("same");  
  SSVHE_FlavEffVsBEff_C->SetMarkerColor(3);
  SSVHE_FlavEffVsBEff_C->Draw("same");  
  SSVHP_FlavEffVsBEff_C->SetMarkerColor(5);
  SSVHP_FlavEffVsBEff_C->Draw("same");
  
	leg = new TLegend(0.6,0.2,0.85,0.5);
  leg->SetFillColor(0);
  leg->AddEntry(TCHE_FlavEffVsBEff_C,"TCHE","p");
  leg->AddEntry(TCHP_FlavEffVsBEff_C,"TCHP","p");
  leg->AddEntry(SSVHE_FlavEffVsBEff_C,"SSVHE","p");
  leg->AddEntry(SSVHP_FlavEffVsBEff_C,"SSVHP","p");
  leg->AddEntry(MVA_FlavEffVsBEff_C,"MVA","p");
  leg->Draw();
	plotname = dir+"/FlavEffVsBEff_C.png";
	Plots2->Print(plotname.c_str());


}
