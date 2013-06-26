#include "TF1.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TTree.h"
#include "TGraphErrors.h"
#include "TGraph.h"
#include "TROOT.h"
#include "Riostream.h"
#include "TBox.h"
#include "TLine.h"
#include "TLatex.h"
#include "TText.h"
#include "TStyle.h"


void LAFits(double RANGE,int PHICUT, int SHOW, int SAVE)
{
gROOT->Reset();
gStyle->SetHistLineWidth(2.5);
gStyle->SetLineWidth(2.5);
gStyle->SetLineStyleString(2,"[12 12]"); 
gROOT->SetStyle("Plain");

TFile* f = new TFile("LA_TEST.root");

TTree* tree = (TTree*)f->Get("LorentzAngle_Plots/Rootple/ModuleTree");

TCanvas *tib_graph = new TCanvas("tib_graph","TIB Graph",1);
TCanvas *tob_graph = new TCanvas("tob_graph","TOB Graph",1);
TCanvas *tib_graph_rms = new TCanvas("tib_graph_rms","TIB Graph (RMS)",1);
TCanvas *tob_graph_rms = new TCanvas("tob_graph_rms","TOB Graph (RMS)",1);
TCanvas *Ratios = new TCanvas("Ratios","Ratios",1);

int showplots = SHOW;

int save_plots=SAVE;
int phicut = PHICUT;
double range = RANGE;

double histo_range = 0.2;

float TIB_expected = 0.024;
float TOB_expected = 0.027;

TH1F *plot_TIB1= new TH1F("plot_TIB1","TIB1",600,-histo_range,histo_range);
TH1F *plot_TIB2= new TH1F("plot_TIB2","TIB2",600,-histo_range,histo_range);
TH1F *plot_TIB3= new TH1F("plot_TIB3","TIB3",600,-histo_range,histo_range);
TH1F *plot_TIB4= new TH1F("plot_TIB4","TIB4",600,-histo_range,histo_range);

TH1F *plot_TOB1= new TH1F("plot_TOB1","TOB1",600,-histo_range,histo_range);
TH1F *plot_TOB2= new TH1F("plot_TOB2","TOB2",600,-histo_range,histo_range);
TH1F *plot_TOB3= new TH1F("plot_TOB3","TOB3",600,-histo_range,histo_range);
TH1F *plot_TOB4= new TH1F("plot_TOB4","TOB4",600,-histo_range,histo_range);
TH1F *plot_TOB5= new TH1F("plot_TOB5","TOB5",600,-histo_range,histo_range);
TH1F *plot_TOB6= new TH1F("plot_TOB6","TOB6",600,-histo_range,histo_range);

if(phicut==0){
tree->Draw("muH >> plot_TIB1","TIB==1 && Layer==1 && goodFit==1");
tree->Draw("muH >> plot_TIB2","TIB==1 && Layer==2 && goodFit==1");
tree->Draw("muH >> plot_TIB3","TIB==1 && Layer==3 && goodFit==1");
tree->Draw("muH >> plot_TIB4","TIB==1 && Layer==4 && goodFit==1");

tree->Draw("muH >> plot_TOB1","TOB==1 && Layer==1 && goodFit==1");
tree->Draw("muH >> plot_TOB2","TOB==1 && Layer==2 && goodFit==1");
tree->Draw("muH >> plot_TOB3","TOB==1 && Layer==3 && goodFit==1");
tree->Draw("muH >> plot_TOB4","TOB==1 && Layer==4 && goodFit==1");
tree->Draw("muH >> plot_TOB5","TOB==1 && Layer==5 && goodFit==1");
tree->Draw("muH >> plot_TOB6","TOB==1 && Layer==6 && goodFit==1");
}

if(phicut==1){
tree->Draw("muH >> plot_TIB1","TIB==1 && Layer==1 && goodFit==1 && ((-2.2<gphi && gphi<-1) || (1<gphi && gphi<2.2))");
tree->Draw("muH >> plot_TIB2","TIB==1 && Layer==2 && goodFit==1 && ((-2.2<gphi && gphi<-1) || (1<gphi && gphi<2.2))");
tree->Draw("muH >> plot_TIB3","TIB==1 && Layer==3 && goodFit==1 && ((-2.2<gphi && gphi<-1) || (1<gphi && gphi<2.2))");
tree->Draw("muH >> plot_TIB4","TIB==1 && Layer==4 && goodFit==1 && ((-2.2<gphi && gphi<-1) || (1<gphi && gphi<2.2))");

tree->Draw("muH >> plot_TOB1","TOB==1 && Layer==1 && goodFit==1 && ((-2.2<gphi && gphi<-1) || (1<gphi && gphi<2.2))");
tree->Draw("muH >> plot_TOB2","TOB==1 && Layer==2 && goodFit==1 && ((-2.2<gphi && gphi<-1) || (1<gphi && gphi<2.2))");
tree->Draw("muH >> plot_TOB3","TOB==1 && Layer==3 && goodFit==1 && ((-2.2<gphi && gphi<-1) || (1<gphi && gphi<2.2))");
tree->Draw("muH >> plot_TOB4","TOB==1 && Layer==4 && goodFit==1 && ((-2.2<gphi && gphi<-1) || (1<gphi && gphi<2.2))");
tree->Draw("muH >> plot_TOB5","TOB==1 && Layer==5 && goodFit==1 && ((-2.2<gphi && gphi<-1) || (1<gphi && gphi<2.2))");
tree->Draw("muH >> plot_TOB6","TOB==1 && Layer==6 && goodFit==1 && ((-2.2<gphi && gphi<-1) || (1<gphi && gphi<2.2))");
} 

ofstream out;
out.open("Layer_MuH.txt");
out<<endl;
  
plot_TIB1->Fit("gaus","","",-range,range);
TF1 *FitFunc1 = plot_TIB1->GetFunction("gaus");
double TIB1_muh = (FitFunc1->GetParameter(1));
double TIB1_muh_err = (FitFunc1->GetParError(1));
double TIB1_muh_rms = (FitFunc1->GetParameter(2));
out<<"TIB1 = "<<TIB1_muh<<" +- "<<TIB1_muh_err<<" Sigma = "<<TIB1_muh_rms<<endl;
if(showplots==1){
TCanvas *a1=new TCanvas;
a1->cd();
gStyle->SetOptStat(11);
gStyle->SetOptFit(111);
plot_TIB1->Draw();
}

plot_TIB2->Fit("gaus","","",-range,range);
TF1 *FitFunc2 = plot_TIB2->GetFunction("gaus");
double TIB2_muh = (FitFunc2->GetParameter(1));
double TIB2_muh_err = (FitFunc2->GetParError(1));
double TIB2_muh_rms = (FitFunc2->GetParameter(2));
out<<"TIB2 = "<<TIB2_muh<<" +- "<<TIB2_muh_err<<" Sigma = "<<TIB2_muh_rms<<endl;
if(showplots==1){
TCanvas *a2 = new TCanvas;
a2->cd();
gStyle->SetOptStat(11);
gStyle->SetOptFit(111);
plot_TIB2->Draw();
if(save_plots==1)a2->SaveAs("TIB2.eps");
}

plot_TIB3->Fit("gaus","","",-range,range);
TF1 *FitFunc3 = plot_TIB3->GetFunction("gaus");
double TIB3_muh = (FitFunc3->GetParameter(1));
double TIB3_muh_err = (FitFunc3->GetParError(1));
double TIB3_muh_rms = (FitFunc3->GetParameter(2));
out<<"TIB3 = "<<TIB3_muh<<" +- "<<TIB3_muh_err<<" Sigma = "<<TIB3_muh_rms<<endl;
if(showplots==1){
TCanvas *a3 = new TCanvas;
a3->cd();
gStyle->SetOptStat(11);
gStyle->SetOptFit(111);
plot_TIB3->Draw();
if(save_plots==1)a3->SaveAs("TIB3.eps");
}

plot_TIB4->Fit("gaus","","",-range,range);
TF1 *FitFunc4 = plot_TIB4->GetFunction("gaus");
double TIB4_muh = (FitFunc4->GetParameter(1));
double TIB4_muh_err = (FitFunc4->GetParError(1));
double TIB4_muh_rms = (FitFunc4->GetParameter(2));
out<<"TIB4 = "<<TIB4_muh<<" +- "<<TIB4_muh_err<<" Sigma = "<<TIB4_muh_rms<<endl;
if(showplots==1){
TCanvas *a4 = new TCanvas;
a4->cd();
gStyle->SetOptStat(11);
gStyle->SetOptFit(111);
plot_TIB4->Draw();
if(save_plots==1)a4->SaveAs("TIB4.eps");
}

plot_TOB1->Fit("gaus","","",-range,range);
TF1 *FitFunc5 = plot_TOB1->GetFunction("gaus");
double TOB1_muh = (FitFunc5->GetParameter(1));
double TOB1_muh_err = (FitFunc5->GetParError(1));
double TOB1_muh_rms = (FitFunc5->GetParameter(2));
out<<"TOB1 = "<<TOB1_muh<<" +- "<<TOB1_muh_err<<" Sigma = "<<TOB1_muh_rms<<endl;
if(showplots==1){
TCanvas *c1=new TCanvas;
c1->cd();
gStyle->SetOptStat(11);
gStyle->SetOptFit(111);
plot_TOB1->Draw();
if(save_plots==1)c1->SaveAs("TOB1.eps");
}

plot_TOB2->Fit("gaus","","",-range,range);
TF1 *FitFunc6 = plot_TOB2->GetFunction("gaus");
double TOB2_muh = (FitFunc6->GetParameter(1));
double TOB2_muh_err = (FitFunc6->GetParError(1));
double TOB2_muh_rms = (FitFunc6->GetParameter(2));
out<<"TOB2 = "<<TOB2_muh<<" +- "<<TOB2_muh_err<<" Sigma = "<<TOB2_muh_rms<<endl;
if(showplots==1){
TCanvas *c2=new TCanvas;
c2->cd();
gStyle->SetOptStat(11);
gStyle->SetOptFit(111);
plot_TOB2->Draw();
if(save_plots==1)c2->SaveAs("TOB2.eps");
}

plot_TOB3->Fit("gaus","","",-range,range);
TF1 *FitFunc7 = plot_TOB3->GetFunction("gaus");
double TOB3_muh = (FitFunc7->GetParameter(1));
double TOB3_muh_err = (FitFunc7->GetParError(1));
double TOB3_muh_rms = (FitFunc7->GetParameter(2));
out<<"TOB3 = "<<TOB3_muh<<" +- "<<TOB3_muh_err<<" Sigma = "<<TOB3_muh_rms<<endl;
if(showplots==1){
TCanvas *c3=new TCanvas;
c3->cd();
gStyle->SetOptStat(11);
gStyle->SetOptFit(111);
plot_TOB3->Draw();
if(save_plots==1)c3->SaveAs("TOB3.eps");
}

plot_TOB4->Fit("gaus","","",-range,range);
TF1 *FitFunc8 = plot_TOB4->GetFunction("gaus");
double TOB4_muh = (FitFunc8->GetParameter(1));
double TOB4_muh_err = (FitFunc8->GetParError(1));
double TOB4_muh_rms = (FitFunc8->GetParameter(2));
out<<"TOB4 = "<<TOB4_muh<<" +- "<<TOB4_muh_err<<" Sigma = "<<TOB4_muh_rms<<endl;
if(showplots==1){
TCanvas *c4=new TCanvas;
c4->cd();
gStyle->SetOptStat(11);
gStyle->SetOptFit(111);
plot_TOB4->Draw();
if(save_plots==1)c4->SaveAs("TOB4.eps");
}

plot_TOB5->Fit("gaus","","",-range,range);
TF1 *FitFunc9 = plot_TOB5->GetFunction("gaus");
double TOB5_muh = (FitFunc9->GetParameter(1));
double TOB5_muh_err = (FitFunc9->GetParError(1));
double TOB5_muh_rms = (FitFunc9->GetParameter(2));
out<<"TOB5 = "<<TOB5_muh<<" +- "<<TOB5_muh_err<<" Sigma = "<<TOB5_muh_rms<<endl;
if(showplots==1){
TCanvas *c5=new TCanvas;
c5->cd();
gStyle->SetOptStat(11);
gStyle->SetOptFit(111);
plot_TOB5->Draw();
if(save_plots==1)c5->SaveAs("TOB5.eps");
}

plot_TOB6->Fit("gaus","","",-range,range);
TF1 *FitFunc10 = plot_TOB6->GetFunction("gaus");
double TOB6_muh = (FitFunc10->GetParameter(1));
double TOB6_muh_err = (FitFunc10->GetParError(1));
double TOB6_muh_rms = (FitFunc10->GetParameter(2));
out<<"TOB6 = "<<TOB6_muh<<" +- "<<TOB6_muh_err<<" Sigma = "<<TOB6_muh_rms<<endl;
if(showplots==1){
TCanvas *c6=new TCanvas;
c6->cd();
gStyle->SetOptStat(11);
gStyle->SetOptFit(111);
plot_TOB6->Draw();
if(save_plots==1)c6->SaveAs("TOB6.eps");
}

if(showplots==1){
TCanvas *a5=new TCanvas;
a5->cd();
gStyle->SetOptStat(11);
gStyle->SetOptFit(111);
plot_TIB1->Draw();
if(save_plots==1)a5->SaveAs("TIB1.eps");
}

int n_tib = 4;
int n_tob = 6;

//TIB graph

float TIBx[4]={1,2,3,4};
float TIBex[4]={0,0,0,0};
float TIBy[4]={TIB1_muh, TIB2_muh, TIB3_muh, TIB4_muh};
float TIBey[4]={TIB1_muh_err, TIB2_muh_err, TIB3_muh_err, TIB4_muh_err};
TGraphErrors *TIB_graph = new TGraphErrors(n_tib,TIBx,TIBy,TIBex,TIBey);
TIB_graph->SetTitle("TIB MuH Values per Layer");
TIB_graph->SetMarkerStyle(20);
TIB_graph->GetXaxis()->SetTitle("Layer number");
TIB_graph->GetXaxis()->SetNdivisions(4);
TIB_graph->GetYaxis()->SetTitle("#mu_{H} (m^{2}/Vs)");
TIB_graph->GetYaxis()->CenterTitle();
TIB_graph->GetYaxis()->SetTitleOffset(1.25);
TIB_graph->GetYaxis()->SetRangeUser(0.015,0.021);
TF1 *TIBFit = new TF1("TIBFit","[0]",0,5);
TIB_graph->Fit("TIBFit","E","", 1, 4);
tib_graph->cd();
gStyle->SetOptStat(11);
gStyle->SetOptFit(111);
TIB_graph->Draw("AP");

if(save_plots==1)tib_graph->SaveAs("TIB_Graph.eps");

//TOB graph

float TOBx[6]={1,2,3,4,5,6};
float TOBex[6]={0,0,0,0,0,0};
float TOBy[6]={TOB1_muh, TOB2_muh, TOB3_muh, TOB4_muh, TOB5_muh, TOB6_muh};
float TOBey[6]={TOB1_muh_err, TOB2_muh_err, TOB3_muh_err, TOB4_muh_err, TOB5_muh_err,TOB6_muh_err};
TGraphErrors *TOB_graph = new TGraphErrors(n_tob,TOBx,TOBy,TOBex,TOBey);
TOB_graph->SetTitle("TOB MuH Values per Layer");
TOB_graph->SetMarkerStyle(20);
TOB_graph->GetXaxis()->SetTitle("Layer number");
TOB_graph->GetXaxis()->SetNdivisions(6);
TOB_graph->GetYaxis()->SetTitle("#mu_{H} (m^{2}/Vs)");
TOB_graph->GetYaxis()->CenterTitle();
TOB_graph->GetYaxis()->SetTitleOffset(1.25);
TOB_graph->GetYaxis()->SetRangeUser(0.02,0.025);
TF1 *TOBFit = new TF1("TOBFit","[0]",0,7);
TOB_graph->Fit("TOBFit","E","", 1, 6);
tob_graph->cd();
gStyle->SetOptStat(11);
gStyle->SetOptFit(111);
TOB_graph->Draw("AP");
if(save_plots==1)tob_graph->SaveAs("TOB_Graph.eps");

//TIB graph

float rmsTIBey[4]={TIB1_muh_rms, TIB2_muh_rms, TIB3_muh_rms, TIB4_muh_rms};
TGraphErrors *TIB_graph_rms = new TGraphErrors(n_tib,TIBx,TIBy,TIBex,rmsTIBey);
TIB_graph_rms->SetMarkerStyle(20);
TIB_graph_rms->SetTitle("TIB MuH Values per Layer (RMS)");
TIB_graph_rms->GetXaxis()->SetTitle("Layer number");
TIB_graph_rms->GetXaxis()->SetNdivisions(4);
TIB_graph_rms->GetYaxis()->SetTitle("#mu_{H} (m^{2}/Vs)");
TIB_graph_rms->GetYaxis()->CenterTitle();
TIB_graph_rms->GetYaxis()->SetTitleOffset(1.25);
TIB_graph_rms->GetYaxis()->SetRangeUser(0.01,0.03);
tib_graph_rms->cd();
TIB_graph_rms->Draw("AP");

if(save_plots==1)tib_graph_rms->SaveAs("TIB_Graph_rms.eps");

//TOB graph

float rmsTOBey[6]={TOB1_muh_rms, TOB2_muh_rms, TOB3_muh_rms, TOB4_muh_rms, TOB5_muh_rms,TOB6_muh_rms};
TGraphErrors *TOB_graph_rms = new TGraphErrors(n_tob,TOBx,TOBy,TOBex,rmsTOBey);
TOB_graph_rms->SetMarkerStyle(20);
TOB_graph_rms->SetTitle("TOB MuH Values per Layer (RMS)");
TOB_graph_rms->GetXaxis()->SetTitle("Layer number");
TOB_graph_rms->GetXaxis()->SetNdivisions(6);
TOB_graph_rms->GetYaxis()->SetTitle("#mu_{H} (m^{2}/Vs)");
TOB_graph_rms->GetYaxis()->CenterTitle();
TOB_graph_rms->GetYaxis()->SetTitleOffset(1.25);
TOB_graph_rms->GetYaxis()->SetRangeUser(0.01,0.03);
tob_graph_rms->cd();
TOB_graph_rms->Draw("AP");

if(save_plots==1)tob_graph_rms->SaveAs("TOB_Graph_rms.eps");

float rTIB1 = TIB1_muh/TIB_expected;
float rTIB2 = TIB2_muh/TIB_expected; 
float rTIB3 = TIB3_muh/TIB_expected; 
float rTIB4 = TIB4_muh/TIB_expected; 
float rTOB1 = TOB1_muh/TOB_expected; 
float rTOB2 = TOB2_muh/TOB_expected; 
float rTOB3 = TOB3_muh/TOB_expected; 
float rTOB4 = TOB4_muh/TOB_expected; 
float rTOB5 = TOB5_muh/TOB_expected; 
float rTOB6 = TOB6_muh/TOB_expected;

out<<endl<<endl<<"------ Ratios ------"<<endl<<endl;
out<<"TIB1 (meas/exp) = "<<rTIB1<<endl;
out<<"TIB2 (meas/exp) = "<<rTIB2<<endl;
out<<"TIB3 (meas/exp) = "<<rTIB3<<endl;
out<<"TIB4 (meas/exp) = "<<rTIB4<<endl;
out<<"TOB1 (meas/exp) = "<<rTOB1<<endl;
out<<"TOB2 (meas/exp) = "<<rTOB2<<endl;
out<<"TOB3 (meas/exp) = "<<rTOB3<<endl;
out<<"TOB4 (meas/exp) = "<<rTOB4<<endl;
out<<"TOB5 (meas/exp) = "<<rTOB5<<endl;
out<<"TOB6 (meas/exp) = "<<rTOB6<<endl;
out<<endl;

int r = 10;
float RatiosX[10]={1,2,3,4,5,6,7,8,9,10};
float RatiosEX[10]={0,0,0,0,0,0,0,0,0,0};
float RatiosY[10]={rTIB1,rTIB2,rTIB3,rTIB4,rTOB1,rTOB2,rTOB3,rTOB4,rTOB5,rTOB6};
float RatiosEY[10]={TIB1_muh_err/TIB_expected, TIB2_muh_err/TIB_expected, TIB3_muh_err/TIB_expected, TIB4_muh_err/TIB_expected, TOB1_muh_err/TOB_expected,TOB2_muh_err/TOB_expected, TOB3_muh_err/TOB_expected, TOB4_muh_err/TOB_expected, TOB5_muh_err/TOB_expected,TOB6_muh_err/TOB_expected};
TGraphErrors *Ratios_graph = new TGraphErrors(r,RatiosX,RatiosY,RatiosEX,RatiosEY);
Ratios_graph->SetMarkerStyle(20);
Ratios_graph->SetTitle("MuH Ratios (meas/exp)");
Ratios_graph->GetXaxis()->SetNdivisions(10);
Ratios_graph->GetXaxis()->SetTitle("Layers");
Ratios_graph->GetYaxis()->SetTitle("muH");
Ratios_graph->GetYaxis()->CenterTitle();
Ratios_graph->GetYaxis()->SetTitleOffset(1.1);
Ratios->cd();
Ratios_graph ->Draw("AP");

out.close();

}

