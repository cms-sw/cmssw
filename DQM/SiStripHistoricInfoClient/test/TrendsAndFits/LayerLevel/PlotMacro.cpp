#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TGraphErrors.h"
#include "TAxis.h"

//#include<Riostream>	       
#include "TCanvas.h"
#include "TObject.h"

using namespace std;

void DrawClus(bool Flag_err,TTree* tree,Double_t *errx,char* varToPlot, char* cond, Int_t kColor, Int_t kMarker, char* Title,char* xTitle,char* yTitle, TLegend *leg, char* cLeg,Double_t downlim,Double_t uplim);
void DrawSame(bool Flag_err,TTree *tree,Double_t *errx, char* varToPlot,char* cond,Int_t kColor, Int_t kMarker,TLegend* leg,char* cLeg);

void PlotMacro(char* input, char* output){
  TFile *out = new TFile(output,"recreate"); 
  TFile *f = new TFile(input); 
  TObjArray DetList(0);
  TTree *t = (TTree*) f->Get("DataTree");

  Double_t errx[t->GetSelectedRows()];
  for(int i=0;i<t->GetSelectedRows();i++){
    errx[i]=0;
  }

  //=== S/N Corr MPV 
  //TOB
  TCanvas *C1=new TCanvas("TOB_mpv","TOB_mpv",10,10,600,400); 
  DetList.Add(C1);
  C1->cd();
  TLegend *legend1 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(1,t,errx,"Run.number:TOB.cStoNCorr.L1.On.FitPar.mp:TOB.cStoNCorr.L1.On.FitPar.emp","",1,20,"TOB, MPV S/N Corrected for the track angle","Run Number","MPV",legend1,"L1",20,35);
  DrawSame(1,t,errx,"Run.number:TOB.cStoNCorr.L2.On.FitPar.mp:TOB.cStoNCorr.L2.On.FitPar.emp","",2,21,legend1,"L2");
  DrawSame(1,t,errx,"Run.number:TOB.cStoNCorr.L3.On.FitPar.mp:TOB.cStoNCorr.L3.On.FitPar.emp","",3,22,legend1,"L3");
  DrawSame(1,t,errx,"Run.number:TOB.cStoNCorr.L4.On.FitPar.mp:TOB.cStoNCorr.L4.On.FitPar.emp","",4,23,legend1,"L4");
  DrawSame(1,t,errx,"Run.number:TOB.cStoNCorr.L5.On.FitPar.mp:TOB.cStoNCorr.L5.On.FitPar.emp","",1,24,legend1,"L5");
  DrawSame(1,t,errx,"Run.number:TOB.cStoNCorr.L6.On.FitPar.mp:TOB.cStoNCorr.L6.On.FitPar.emp","",6,29,legend1,"L6");
  //TIB
  TCanvas *C2=new TCanvas("TIB_mpv","TIB_mpv",10,10,600,400); 
  DetList.Add(C2);
  C2->cd();
  TLegend *legend2 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(1,t,errx,"Run.number:TIB.cStoNCorr.L1.On.FitPar.mp:TIB.cStoNCorr.L1.On.FitPar.emp","",1,20,"TIB, MPV S/N Corrected for the track angle","Run Number","MPV",legend2,"L1",7,25);
  DrawSame(1,t,errx,"Run.number:TIB.cStoNCorr.L2.On.FitPar.mp:TIB.cStoNCorr.L2.On.FitPar.emp","",2,21,legend2,"L2");
  DrawSame(1,t,errx,"Run.number:TIB.cStoNCorr.L3.On.FitPar.mp:TIB.cStoNCorr.L3.On.FitPar.emp","",3,22,legend2,"L3");
  DrawSame(1,t,errx,"Run.number:TIB.cStoNCorr.L4.On.FitPar.mp:TIB.cStoNCorr.L4.On.FitPar.emp","",4,23,legend2,"L4");

  //=== cCharge off Track mean
  //TOB
  TCanvas *C3=new TCanvas("TOB_chOff","TOB_chOff",10,10,600,400); 
  DetList.Add(C3);
  C3->cd();
  TLegend *legend3 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TOB.cCharge.L1.Off.HistoPar.mean","",1,20,"TOB, Charge of the clusters off track - Mean","Run Number","Mean [ADC]",legend3,"L1",0,250);
  DrawSame(0,t,errx,"Run.number:TOB.cCharge.L2.Off.HistoPar.mean","",2,21,legend3,"L2");
  DrawSame(0,t,errx,"Run.number:TOB.cCharge.L3.Off.HistoPar.mean","",3,22,legend3,"L3");
  DrawSame(0,t,errx,"Run.number:TOB.cCharge.L4.Off.HistoPar.mean","",4,23,legend3,"L4");
  DrawSame(0,t,errx,"Run.number:TOB.cCharge.L5.Off.HistoPar.mean","",1,24,legend3,"L5");
  DrawSame(0,t,errx,"Run.number:TOB.cCharge.L6.Off.HistoPar.mean","",6,29,legend3,"L6");

  //TIB
  TCanvas *C4=new TCanvas("TIB_chOff","TIB_chOff",10,10,600,400); 
  DetList.Add(C4);
  C4->cd();
  TLegend *legend4 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(1,t,errx,"Run.number:TIB.cCharge.L1.Off.HistoPar.mean","",1,20,"TIB, Charge of the clusters off track - Mean","Run Number","Mean [ADC]",legend4,"L1",50,300);
  DrawSame(1,t,errx,"Run.number:TIB.cCharge.L2.Off.HistoPar.mean","",2,21,legend4,"L2");
  DrawSame(1,t,errx,"Run.number:TIB.cCharge.L3.Off.HistoPar.mean","",3,22,legend4,"L3");
  DrawSame(1,t,errx,"Run.number:TIB.cCharge.L4.Off.HistoPar.mean","",4,23,legend4,"L4");

  //=== cCharge off Track RMS
  //TOB
  TCanvas *C5=new TCanvas("TOB_chOffRMS","TOB_chOffRMS",10,10,600,400); 
  DetList.Add(C5);
  C5->cd();
  TLegend *legend5 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TOB.cCharge.L1.Off.HistoPar.rms","",1,20,"TOB, Charge of the clusters off track - RMS","Run Number","RMS [ADC]",legend5,"L1",0,130);
  DrawSame(0,t,errx,"Run.number:TOB.cCharge.L2.Off.HistoPar.rms","",2,21,legend5,"L2");
  DrawSame(0,t,errx,"Run.number:TOB.cCharge.L3.Off.HistoPar.rms","",3,22,legend5,"L3");
  DrawSame(0,t,errx,"Run.number:TOB.cCharge.L4.Off.HistoPar.rms","",4,23,legend5,"L4");
  DrawSame(0,t,errx,"Run.number:TOB.cCharge.L5.Off.HistoPar.rms","",1,24,legend5,"L5");
  DrawSame(0,t,errx,"Run.number:TOB.cCharge.L6.Off.HistoPar.rms","",6,29,legend5,"L6");

  //TIB
  TCanvas *C6=new TCanvas("TIB_chOffRMS","TIB_chOffRMS",10,10,600,400); 
  DetList.Add(C6);
  C6->cd();
  TLegend *legend6 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TIB.cCharge.L1.Off.HistoPar.rms","",1,20,"TIB, Charge of the clusters off track - RMS","Run Number","RMS [ADC]",legend6,"L1",0,120);
  DrawSame(0,t,errx,"Run.number:TIB.cCharge.L2.Off.HistoPar.rms","",2,21,legend6,"L2");
  DrawSame(0,t,errx,"Run.number:TIB.cCharge.L3.Off.HistoPar.rms","",3,22,legend6,"L3");
  DrawSame(0,t,errx,"Run.number:TIB.cCharge.L4.Off.HistoPar.rms","",4,23,legend6,"L4");

  //=== Noise on Track Gauss Fit
  //TOB
  TCanvas *C7=new TCanvas("TOB_NoiseOn","TOB_NoiseOn",10,10,600,400); 
  DetList.Add(C7);
  C7->cd();
  TLegend *legend7 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(1,t,errx,"Run.number:TOB.cNoise.L1.On.FitNoisePar.fitmean:TOB.cNoise.L1.On.FitNoisePar.efitmean","",1,20,"TOB, Mean Noise of the clusters on track","Run Number","Mean Noise [ADC]",legend7,"L1",4.3,4.7);
  DrawSame(1,t,errx,"Run.number:TOB.cNoise.L2.On.FitNoisePar.fitmean:TIB.cNoise.L2.On.FitNoisePar.efitmean","",2,21,legend7,"L2");
  DrawSame(1,t,errx,"Run.number:TOB.cNoise.L3.On.FitNoisePar.fitmean:TID.cNoise.L3.On.FitNoisePar.efitmean","",3,22,legend7,"L3");
  DrawSame(1,t,errx,"Run.number:TOB.cNoise.L4.On.FitNoisePar.fitmean:TIB.cNoise.L4.On.FitNoisePar.efitmean","",4,23,legend7,"L4");
  DrawSame(1,t,errx,"Run.number:TOB.cNoise.L5.On.FitNoisePar.fitmean:TID.cNoise.L5.On.FitNoisePar.efitmean","",1,24,legend7,"L5");
  DrawSame(1,t,errx,"Run.number:TOB.cNoise.L6.On.FitNoisePar.fitmean:TID.cNoise.L6.On.FitNoisePar.efitmean","",6,29,legend7,"L6");
  //TIB
  TCanvas *C8=new TCanvas("TIB_NoiseOn","TIB_NoiseOn",10,10,600,400); 
  DetList.Add(C8);
  C8->cd();
  TLegend *legend8 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(1,t,errx,"Run.number:TIB.cNoise.L1.On.FitNoisePar.fitmean:TIB.cNoise.L1.On.FitNoisePar.efitmean","",1,20,"TIB, Mean Noise of the clusters on track","Run Number","Mean Noise [ADC]",legend8,"L1",1,5);
  DrawSame(1,t,errx,"Run.number:TIB.cNoise.L2.On.FitNoisePar.fitmean:TIB.cNoise.L2.On.FitNoisePar.efitmean","",2,21,legend8,"L2");
  DrawSame(1,t,errx,"Run.number:TIB.cNoise.L3.On.FitNoisePar.fitmean:TIB.cNoise.L3.On.FitNoisePar.efitmean","",3,22,legend8,"L3");
  DrawSame(1,t,errx,"Run.number:TIB.cNoise.L4.On.FitNoisePar.fitmean:TIB.cNoise.L4.On.FitNoisePar.efitmean","",4,23,legend8,"L4");

  //=== Noise off Track mean
  //TOB
  TCanvas *C9=new TCanvas("TOB_NoiseOff","TOB_NoiseOff",10,10,600,400); 
  DetList.Add(C9);
  C9->cd();
  TLegend *legend9 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TOB.cNoise.L1.Off.HistoPar.mean","",1,20,"TOB, Mean noise of the clusters off track","Run Number","Noise Mean [ADC]",legend9,"L1",0,8);
  DrawSame(0,t,errx,"Run.number:TOB.cNoise.L2.Off.HistoPar.mean","",2,21,legend9,"L2");
  DrawSame(0,t,errx,"Run.number:TOB.cNoise.L3.Off.HistoPar.mean","",3,22,legend9,"L3");
  DrawSame(0,t,errx,"Run.number:TOB.cNoise.L4.Off.HistoPar.mean","",4,23,legend9,"L4");
  DrawSame(0,t,errx,"Run.number:TOB.cNoise.L5.Off.HistoPar.mean","",1,24,legend9,"L5");
  DrawSame(0,t,errx,"Run.number:TOB.cNoise.L6.Off.HistoPar.mean","",6,29,legend9,"L6");

  //TIB
  TCanvas *C10=new TCanvas("TIB_NoiseOff","TIB_NoiseOff",10,10,600,400); 
  DetList.Add(C10);
  C10->cd();
  TLegend *legend10 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TIB.cNoise.L1.Off.HistoPar.mean","",1,20,"TIB, Noise of the clusters off track - Mean","Run Number","Noise Mean [ADC]",legend10,"L1",3,6);
  DrawSame(0,t,errx,"Run.number:TIB.cNoise.L2.Off.HistoPar.mean","",2,21,legend10,"L2");
  DrawSame(0,t,errx,"Run.number:TIB.cNoise.L3.Off.HistoPar.mean","",3,22,legend10,"L3");
  DrawSame(0,t,errx,"Run.number:TIB.cNoise.L4.Off.HistoPar.mean","",4,23,legend10,"L4");

  //=== Noise off Track RMS
  //TOB
  TCanvas *C15=new TCanvas("TOB_NoiseOff_Rms","TOB_NoiseOff_Rms",10,10,600,400); 
  DetList.Add(C15);
  C15->cd();
  TLegend *legend15 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TOB.cNoise.L1.Off.HistoPar.rms","",1,20,"TOB, Noise of the clusters off track - RMS","Run Number","Noise RMS [ADC]",legend15,"L1",0,4);
  DrawSame(0,t,errx,"Run.number:TOB.cNoise.L2.Off.HistoPar.rms","",2,21,legend15,"L2");
  DrawSame(0,t,errx,"Run.number:TOB.cNoise.L3.Off.HistoPar.rms","",3,22,legend15,"L3");
  DrawSame(0,t,errx,"Run.number:TOB.cNoise.L4.Off.HistoPar.rms","",4,23,legend15,"L4");
  DrawSame(0,t,errx,"Run.number:TOB.cNoise.L5.Off.HistoPar.rms","",1,24,legend15,"L5");
  DrawSame(0,t,errx,"Run.number:TOB.cNoise.L6.Off.HistoPar.rms","",6,29,legend15,"L6");

  //TIB
  TCanvas *C16=new TCanvas("TIB_NoiseOff_Rms","TIB_NoiseOff_Rms",10,10,600,400); 
  DetList.Add(C16);
  C16->cd();
  TLegend *legend16 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TIB.cNoise.L1.Off.HistoPar.rms","",1,20,"TIB, Noise of the clusters off track - RMS","Run Number","Noise RMS [ADC]",legend16,"L1",0,4);
  DrawSame(0,t,errx,"Run.number:TIB.cNoise.L2.Off.HistoPar.rms","",2,21,legend16,"L2");
  DrawSame(0,t,errx,"Run.number:TIB.cNoise.L3.Off.HistoPar.rms","",3,22,legend16,"L3");
  DrawSame(0,t,errx,"Run.number:TIB.cNoise.L4.Off.HistoPar.rms","",4,23,legend16,"L4");

  //=== Width on Track mean
  //TOB
  TCanvas *C11=new TCanvas("TOB_WidthOn","TOB_WidthOn",10,10,600,400); 
  DetList.Add(C11);
  C11->cd();
  TLegend *legend11 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TOB.cWidth.L1.On.HistoPar.mean","",1,20,"TOB, Mean width of the clusters on track","Run Number","Mean width [#strips]",legend11,"L1",1,4);
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L2.On.HistoPar.mean","",2,21,legend11,"L2");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L3.On.HistoPar.mean","",3,22,legend11,"L3");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L4.On.HistoPar.mean","",4,23,legend11,"L4");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L5.On.HistoPar.mean","",1,24,legend11,"L5");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L6.On.HistoPar.mean","",6,29,legend11,"L6");

  //TIB
  TCanvas *C12=new TCanvas("TIB_WidthOn","TIB_WidthOn",10,10,600,400); 
  DetList.Add(C12);
  C12->cd();
  TLegend *legend12 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TIB.cWidth.L1.On.HistoPar.mean","",1,20,"TIB, Mean width of the clusters on track","Run Number","Mean width [#strips]",legend12,"L1",0,5);
  DrawSame(0,t,errx,"Run.number:TIB.cWidth.L2.On.HistoPar.mean","",2,21,legend12,"L2");
  DrawSame(0,t,errx,"Run.number:TIB.cWidth.L3.On.HistoPar.mean","",3,22,legend12,"L3");
  DrawSame(0,t,errx,"Run.number:TIB.cWidth.L4.On.HistoPar.mean","",4,23,legend12,"L4");

 //=== Width on Track rms
  //TOB
  TCanvas *C17=new TCanvas("TOB_WidthOn_Rms","TOB_WidthOn_Rms",10,10,600,400); 
  DetList.Add(C17);
  C17->cd();
  TLegend *legend17 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TOB.cWidth.L1.On.HistoPar.rms","",1,20,"TOB, Rms width of the clusters on track","Run Number","Rms width [#strips]",legend17,"L1",1,4);
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L2.On.HistoPar.rms","",2,21,legend17,"L2");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L3.On.HistoPar.rms","",3,22,legend17,"L3");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L4.On.HistoPar.rms","",4,23,legend17,"L4");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L5.On.HistoPar.rms","",1,24,legend17,"L5");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L6.On.HistoPar.rms","",6,29,legend17,"L6");

  //TIB
  TCanvas *C18=new TCanvas("TIB_WidthOn_Rms","TIB_WidthOn_Rms",10,10,600,400); 
  DetList.Add(C18);
  C18->cd();
  TLegend *legend18 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TIB.cWidth.L1.On.HistoPar.rms","",1,20,"TIB, Rms width of the clusters on track","Run Number","Rms width [#strips]",legend18,"L1",0,5);
  DrawSame(0,t,errx,"Run.number:TIB.cWidth.L2.On.HistoPar.rms","",2,21,legend18,"L2");
  DrawSame(0,t,errx,"Run.number:TIB.cWidth.L3.On.HistoPar.rms","",3,22,legend18,"L3");
  DrawSame(0,t,errx,"Run.number:TIB.cWidth.L4.On.HistoPar.rms","",4,23,legend18,"L4");

  //=== Width off Track mean
  //TOB
  TCanvas *C13=new TCanvas("TOB_WidthOff","TOB_WidthOff",10,10,600,400); 
  DetList.Add(C13);
  C13->cd();
  TLegend *legend13 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TOB.cWidth.L1.Off.HistoPar.mean","",1,20,"TOB, Mean width of the clusters off track","Run Number","Mean width [#strips]",legend13,"L1",0,7);
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L2.Off.HistoPar.mean","",2,21,legend13,"L2");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L3.Off.HistoPar.mean","",3,22,legend13,"L3");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L4.Off.HistoPar.mean","",4,23,legend13,"L4");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L5.Off.HistoPar.mean","",1,24,legend13,"L5");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L6.Off.HistoPar.mean","",6,29,legend13,"L6");

  //TIB
  TCanvas *C14=new TCanvas("TIB_WidthOff","TIB_WidthOff",10,10,600,400); 
  DetList.Add(C14);
  C14->cd();
  TLegend *legend14 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TIB.cWidth.L1.Off.HistoPar.mean","",1,20,"TIB, Mean width of the clusters off track","Run Number","Mean width [#strips]",legend14,"L1",0,7);
  DrawSame(0,t,errx,"Run.number:TIB.cWidth.L2.Off.HistoPar.mean","",2,21,legend14,"L2");
  DrawSame(0,t,errx,"Run.number:TIB.cWidth.L3.Off.HistoPar.mean","",3,22,legend14,"L3");
  DrawSame(0,t,errx,"Run.number:TIB.cWidth.L4.Off.HistoPar.mean","",4,23,legend14,"L4");

  //=== Width off Track rms
  //TOB
  TCanvas *C20=new TCanvas("TOB_WidthOff_Rms","TOB_WidthOff_Rms",10,10,600,400); 
  DetList.Add(C20);
  C20->cd();
  TLegend *legend20 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TOB.cWidth.L1.Off.HistoPar.rms","",1,20,"TOB, Rms width of the clusters off track","Run Number","Rms width [#strips]",legend20,"L1",0,7);
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L2.Off.HistoPar.rms","",2,21,legend20,"L2");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L3.Off.HistoPar.rms","",3,22,legend20,"L3");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L4.Off.HistoPar.rms","",4,23,legend20,"L4");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L5.Off.HistoPar.rms","",1,24,legend20,"L5");
  DrawSame(0,t,errx,"Run.number:TOB.cWidth.L6.Off.HistoPar.rms","",6,29,legend20,"L6");

  //TIB
  TCanvas *C21=new TCanvas("TIB_WidthOff_Rms","TIB_WidthOff_Rms",10,10,600,400); 
  DetList.Add(C21);
  C21->cd();
  TLegend *legend21 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TIB.cWidth.L1.Off.HistoPar.rms","",1,20,"TIB, Rms width of the clusters off track","Run Number","Rms width [#strips]",legend21,"L1",0,7);
  DrawSame(0,t,errx,"Run.number:TIB.cWidth.L2.Off.HistoPar.rms","",2,21,legend21,"L2");
  DrawSame(0,t,errx,"Run.number:TIB.cWidth.L3.Off.HistoPar.rms","",3,22,legend21,"L3");
  DrawSame(0,t,errx,"Run.number:TIB.cWidth.L4.Off.HistoPar.rms","",4,23,legend21,"L4");


  out->cd();
  DetList.Write();
  f->cd();
   
}

void DrawClus(bool Flag_err,TTree* tree,Double_t *errx,char* varToPlot, char* cond, Int_t kColor, Int_t kMarker, char* Title,char* xTitle,char* yTitle, TLegend *leg, char* cLeg,Double_t downlim,Double_t uplim){
  TGraphErrors* g;
  
  tree->Draw(varToPlot, cond,"goff");
  cout << tree->GetSelectedRows() << endl;
  if(tree->GetSelectedRows()){
    if(Flag_err)
      g=new TGraphErrors(tree->GetSelectedRows(),tree->GetV1(),tree->GetV2(),errx,tree->GetV3());
    else
      g=new TGraphErrors(tree->GetSelectedRows(),tree->GetV1(),tree->GetV2(),errx,errx);
    g->SetMarkerStyle(kMarker);
    g->SetMarkerSize(0.9);
    g->SetMarkerColor(kColor);
    g->SetTitle(Title);
    g->GetXaxis()->SetTitle(xTitle);
    g->GetXaxis()->CenterTitle();
    g->GetYaxis()->SetTitle(yTitle);
    g->GetYaxis()->CenterTitle();
    g->GetYaxis()->SetRangeUser(downlim,uplim);
    g->Draw("Ap");
    leg->AddEntry(g, cLeg,"p");
  }else{
    cout << "NO rows selected for leave " << varToPlot << endl;
  }
}

void DrawSame(bool Flag_err,TTree *tree,Double_t *errx, char* varToPlot, char* cond, Int_t kColor,Int_t kMarker,TLegend* leg,char* cLeg){
  TGraphErrors *g;
  tree->Draw(varToPlot,cond,"goff");
  
  if (tree->GetSelectedRows()) {
    if(Flag_err)    
      g = new TGraphErrors(tree->GetSelectedRows(), tree->GetV1(), tree->GetV2(), errx, tree->GetV3());
    else
      g=new TGraphErrors(tree->GetSelectedRows(),tree->GetV1(),tree->GetV2(),errx,errx);

    g->SetMarkerStyle(kMarker);
    g->SetMarkerSize(0.9);
    g->SetMarkerColor(kColor);
    g->Draw("SP"); //draw graph in current pad
    
    leg->AddEntry(g,cLeg,"p");
    leg->Draw();
  }else{
    cout << "NO rows selected for leave " << varToPlot << endl;
  }
}
