#include <fstream>
#include <stdio.h>
#include <string>
#include <stdlib.h>

#include "TROOT.h"
#include "TFile.h"
#include "TGraph.h"
#include "TPostScript.h"
#include "TLine.h"
#include "TText.h"
#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TProfile.h"
#include "TNtuple.h"
#include "TRandom.h"
#include "TCanvas.h"

void silvia(){

gStyle->SetPalette(1,0);
gStyle->SetOptFit(111111);
 gStyle->SetOptStat(111111);


  gSystem->Load("libRooFit");
  using namespace RooFit;

 TFile *fileSL1 = TFile::Open("../test/dt_phase2.root");

 TH1F *hResSlope4h = (TH1F*)fileSL1->Get("selected_chamber_segment_vs_jm_tanPhi_gauss");
 TH1F *hResSlope3h = (TH1F*)fileSL1->Get("selected_chamber_segment_vs_jm_tanPhi_gauss");
 TH1F *hResPos4h = (TH1F*)fileSL1->Get("selected_chamber_segment_vs_jm_x_gauss");
 TH1F *hResPos3h = (TH1F*)fileSL1->Get("selected_chamber_segment_vs_jm_x_gauss");



RooRealVar x("x","",-0.1,0.1);
RooRealVar mean("mean",",mean of Gaussian",0.,-0.1,0.1);
RooRealVar sigma1("sigma1",",width of narrow Gaussian",0.008,-0.1,0.1);
RooRealVar sigma2("sigma2",",width of wide Gaussian",0.010,-0.1,0.1);
RooRealVar fraction("fraction",",fraction of narrow Gaussian",2./3.,0.,1.);

RooGaussian gauss1("gauss1","Narow Gaussian",x, mean, sigma1);
RooGaussian gauss2("gauss2","Wide Gaussian",x, mean, sigma2);

RooAddPdf twogauss("twogauss","Two Gaussians pdf",RooArgList(gauss1,gauss2),fraction);

RooDataHist data("data","data",x,hResSlope4h);

twogauss.fitTo(data,RooFit::Extended());

RooPlot* xframe=x.frame();
data.plotOn(xframe);
twogauss.plotOn(xframe);
twogauss.plotOn(xframe,Components("gauss1"),LineColor(kRed));
twogauss.plotOn(xframe,Components("gauss2"),LineStyle(kDashed)); 

xframe->SetTitle("Slope Resolution for In Time 4h fits, SL1");
xframe->SetXTitle(" Segment Slope - Fit Slope (rad)");

TCanvas *canvas1 = new TCanvas();
xframe->Draw();

TPaveText *pave1 = new TPaveText(0.05,8000,0.07,10000);
   TText *t16=pave1->AddText("Sigma1: 9.07 +- 0.04 mrad");
   TText *t26=pave1->AddText("Sigma2: 29.4 +- 0.6 mrad");
   TText *t36=pave1->AddText("Bias: 1.01+-0.03 mrad");
 pave1->SetTextSize(0.03); 
 pave1->SetFillColor(0);
 pave1->Draw("same");

 gPad->SaveAs("ResSlope4h_Lin.png");

 canvas1->SetLogy();
 gPad->SaveAs("ResSlope4h_Log.png");


//POSITION

RooRealVar xpos("xpos","",-0.1,0.1);
RooRealVar meanpos("meanpos",",mean of Gaussian",0.,-0.1,0.1);
RooRealVar sigma1pos("sigma1pos",",width of narrow Gaussian",0.0033,-0.1,0.1);
RooRealVar sigma2pos("sigma2pos",",width of wide Gaussian",0.0150,-0.1,0.1);
RooRealVar fractionpos("fractionpos",",fraction of narrow Gaussian",2./3.,0.,1.);

RooGaussian gauss1pos("gauss1","Narow Gaussian",xpos, meanpos, sigma1pos);
RooGaussian gauss2pos("gauss2","Wide Gaussian",xpos, meanpos, sigma2pos);

RooAddPdf twogausspos("twogausspos","Two Gaussians pdf",RooArgList(gauss1pos,gauss2pos),fractionpos);

RooDataHist datapos("datapos","datapos",xpos,hResPos4h);
twogausspos.fitTo(datapos,RooFit::Extended());

RooPlot* xposframe=xpos.frame();
datapos.plotOn(xposframe);

twogausspos.plotOn(xposframe);
twogausspos.plotOn(xposframe,Components("gauss1"),LineColor(kRed));
twogausspos.plotOn(xposframe,Components("gauss2"),LineStyle(kDashed)); 

xposframe->SetTitle("Position Resolution for In Time 4h fits, SL1");
xposframe->SetXTitle(" Segment Position - Fit Position (cm)");

TCanvas *canvas5 = new TCanvas();

xposframe->Draw();

 pave1=new TPaveText(0.04,1200,0.06,1500);
 TText *t18=pave1->AddText("Sigma1: 30.3 +- 0.2 u");
 TText *t28=pave1->AddText("Sigma2: 548 +- 8 u");
 TText *t38=pave1->AddText("Bias: 1.9+-0.2 u");
 pave1->SetTextSize(0.03); 
 pave1->SetFillColor(0);
 pave1->Draw("same");

 gPad->SaveAs("ResPos4h_Lin.png");

 canvas5->SetLogy();
 gPad->SaveAs("ResPos4h_Log.png");

 exit(0);

}
