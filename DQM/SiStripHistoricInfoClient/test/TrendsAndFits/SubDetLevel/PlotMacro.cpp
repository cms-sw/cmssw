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

void PlotMacro(char *inputFile, char *outputFile){
  TFile *out = new TFile(outputFile,"recreate"); 
  TFile *f = new TFile(inputFile); 
  TObjArray DetList(0);
  TTree *t = (TTree*) f->Get("DataTree");

  Double_t errx[t->GetSelectedRows()];
  for(int i=0;i<t->GetSelectedRows();i++){
    errx[i]=0;
  }

  //=== S/N Corr MPV 
  TCanvas *C1=new TCanvas("SN_MPV","SN_MPV",10,10,600,400); 
  DetList.Add(C1);
  C1->cd();
  TLegend *legend1 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(1,t,errx,"Run.number:TOB.StoNCorr.On.FitPar.mp:TOB.StoNCorr.On.FitPar.emp","",2,22,"MPV S/N Corrected for the track angle","Run Number","MPV",legend1,"TOB",5,35);
  DrawSame(1,t,errx,"Run.number:TIB.StoNCorr.On.FitPar.mp:TIB.StoNCorr.On.FitPar.emp","",3,23,legend1,"TIB");
  DrawSame(1,t,errx,"Run.number:TID.StoNCorr.On.FitPar.mp:TID.StoNCorr.On.FitPar.emp","",4,24,legend1,"TID");

  //=== Charge off Track mean
  TCanvas *C2=new TCanvas("chOffmean","chOffmean",10,10,600,400); 
  DetList.Add(C2);
  C2->cd();
  TLegend *legend2 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TOB.Charge.Off.HistoPar.mean","",2,22,"Charge of the clusters off track - Mean","Run Number","Mean Charge [ADC]",legend2,"TOB",50,250);
  DrawSame(0,t,errx,"Run.number:TIB.Charge.Off.HistoPar.mean","",3,23,legend2,"TIB");
  DrawSame(0,t,errx,"Run.number:TID.Charge.Off.HistoPar.mean","",4,24,legend2,"TID");

  //=== Charge off Track RMS
  TCanvas *C3=new TCanvas("chOffRms","chOffRms",10,10,600,400); 
  DetList.Add(C3);
  C3->cd();
  TLegend *legend3 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TOB.Charge.Off.HistoPar.rms","",2,22,"Charge of the clusters off track - RMS","Run Number","RMS [ADC]",legend3,"TOB",50,110);
  DrawSame(0,t,errx,"Run.number:TIB.Charge.Off.HistoPar.rms","",3,23,legend3,"TIB");
  DrawSame(0,t,errx,"Run.number:TID.Charge.Off.HistoPar.rms","",4,24,legend3,"TID");

  //=== Noise on Track Gauss Fit
  TCanvas *C4=new TCanvas("NoiseOn","NoiseOn",10,10,600,400); 
  DetList.Add(C4);
  C4->cd();
  TLegend *legend4 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(1,t,errx,"Run.number:TOB.Noise.On.FitNoisePar.fitmean:TOB.Noise.On.FitNoisePar.efitmean","",2,22,"Mean Noise of the clusters on track","Run Number","Mean Noise [ADC]",legend4,"TOB",2,6);
  DrawSame(1,t,errx,"Run.number:TIB.Noise.On.FitNoisePar.fitmean:TIB.Noise.On.FitNoisePar.efitmean","",3,23,legend4,"TIB");
  DrawSame(1,t,errx,"Run.number:TID.Noise.On.FitNoisePar.fitmean:TID.Noise.On.FitNoisePar.efitmean","",4,24,legend4,"TID");

  //=== Noise off Track mean
  TCanvas *C5=new TCanvas("NoiseOff","NoiseOff",10,10,600,400); 
  DetList.Add(C5);
  C5->cd();
  TLegend *legend5 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TOB.Noise.Off.HistoPar.mean","",2,22,"Mean noise of the clusters off track","Run Number","Mean noise [ADC]",legend5,"TOB",1,6);
  DrawSame(0,t,errx,"Run.number:TIB.Noise.Off.HistoPar.mean","",3,23,legend5,"TIB");
  DrawSame(0,t,errx,"Run.number:TID.Noise.Off.HistoPar.mean","",4,24,legend5,"TID");

  //=== Width on Track mean
  TCanvas *C6=new TCanvas("WidthOn","WidthOn",10,10,600,400); 
  DetList.Add(C6);
  C6->cd();
  TLegend *legend6 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TOB.Width.On.HistoPar.mean","",2,22,"Mean width of the clusters on track","Run Number","Mean width [#strips]",legend6,"TOB",0,7);
  DrawSame(0,t,errx,"Run.number:TIB.Width.On.HistoPar.mean","",3,23,legend6,"TIB");
  DrawSame(0,t,errx,"Run.number:TID.Width.On.HistoPar.mean","",4,24,legend6,"TID");

  //=== Width off Track mean
  TCanvas *C7=new TCanvas("WidthOff","WidthOff",10,10,600,400); 
  DetList.Add(C7);
  C7->cd();
  TLegend *legend7 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TOB.Width.Off.HistoPar.mean","",2,22,"Mean width of the clusters off track","Run Number","Mean width [#strips]",legend7,"TOB",0,7);
  DrawSame(0,t,errx,"Run.number:TIB.Width.Off.HistoPar.mean","",3,23,legend7,"TIB");
  DrawSame(0,t,errx,"Run.number:TID.Width.Off.HistoPar.mean","",4,24,legend7,"TID");

  //=== Number of Events 
  TCanvas *C8= new TCanvas("nEvents","nEvents",10,10,600,400);
  DetList.Add(C8);
  C8->cd();
  TLegend *legend8 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TrAndClus.nTracks.entries","",1,22,"Number of processed Events","Run Number", "# Events",legend8,"",40000,1000000);

  //=== Number of Events with at least one track
  TCanvas *C9= new TCanvas("nTracks","nTracks",10,10,600,400);
  DetList.Add(C9);
  C9->cd();
  TLegend *legend9 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:(TrAndClus.nTracks.nonzero/TrAndClus.nTracks.entries)","",1,22,"Fraction of Events with at least one track","Run Number", "# Events",legend9,"",0,1);

  //=== Number of clusters on track
  TCanvas *C10= new TCanvas("nClusOn","nClusOn",10,10,600,400);
  DetList.Add(C10);
  C10->cd();
  TLegend *legend10 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:(TIB.nClusters.On.HistoPar.nonzero/TrAndClus.nTracks.nonzero)","",2,22,"Fraction of clusters associated to a track","Run Number", "Fraction of Cluster On Track",legend10,"TIB",0,1);
  DrawSame(0,t,errx,"Run.number:(TOB.nClusters.On.HistoPar.nonzero/TrAndClus.nTracks.nonzero)","",3,23,legend10,"TOB");
  DrawSame(0,t,errx,"Run.number:(TID.nClusters.On.HistoPar.nonzero/TrAndClus.nTracks.nonzero)","",4,24,legend10,"TID");

  //=== Mean number of clusters off track
  TCanvas *C12= new TCanvas("nClusOff","nClusOff",10,10,600,400);
  DetList.Add(C12);
  C12->cd();
  TLegend *legend12 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TrAndClus.OffClus.mean","",1,21,"Mean number of clusters off track","Run Number", "# Cluster Off Track",legend12,"TOT",0,25);
  DrawSame(0,t,errx,"Run.number:TIB.nClusters.Off.HistoPar.mean","",2,22,legend12,"TIB");
  DrawSame(0,t,errx,"Run.number:TOB.nClusters.Off.HistoPar.mean","",3,23,legend12,"TOB");
  DrawSame(0,t,errx,"Run.number:TID.nClusters.Off.HistoPar.mean","",4,24,legend12,"TID");

  //=== Mean Number of Tracks
  TCanvas *C11= new TCanvas("MeanTracks","MeanTracks",10,10,600,400);
  DetList.Add(C11);
  C11->cd();
  TLegend *legend11 = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(0,t,errx,"Run.number:TrAndClus.nTracks.mean","",1,22,"Mean number of tracks per event","Run Number", "# Tracks",legend11,"",0.4,0.7);

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
