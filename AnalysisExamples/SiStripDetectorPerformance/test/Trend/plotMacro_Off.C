#include<Riostream>	       
#include "TCanvas.h"
#include "TObject.h"
using namespace std;

//void DrawFunction(TCanvas *c, char* canvas, char* varToPlot, char* cond, Int_t kColor, char* Title);
void DrawFunction(TCanvas *c,char* varToPlot, char* cond, Int_t kColor, char* Title,TLegend* leg,char* cLeg);
void DrawSame(char* varToPlot, char* cond, Int_t kColor,TLegend* leg,char* cLeg);
void DrawGlobal(TObjArray List,TFile *in,TFile *out);

void PlotMacro(char* input="", char* outputFile=""){

  gStyle->SetOptStat(1110);
  TObjArray GlobList(0),DetList(0);

  TFile *out = new TFile(outputFile,"recreate");  
  TFile *f = new TFile(input);
  TIFTree->Scan("number:Clusters.entries_all:Tracks.mean:Clusters.mean_corr:RecHits.mean");
  TCanvas *Events; 
  TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
 //==========Global Plots:Tracks, RecHits and nClusters========//

  DrawGlobal(GlobList,f,out);

  //============ Global Signal off Track=========//

  TCanvas *cSignal_Off=new TCanvas("cSignal_Off","cSignal_Off_Track",10,10,900,500);
  DetList.Add(cSignal_Off);
  DrawFunction(cSignal_Off,"number:TIB.Signal.Off.FitPar.mp:TIB.Signal.Off.FitPar.emp","Clusters.entries_all>2000 && TIB.Signal.Off.FitPar.mp>0 && TIB.Signal.Off.FitPar.emp>0",1,"Cluster Signal ected Off Track (Fit MP) [ADC]",legend,"TIB");
  DrawSame("number:TOB.Signal.Off.FitPar.mp:TOB.Signal.Off.FitPar.emp","Clusters.entries_all>2000 && TOB.Signal.Off.FitPar.mp>0 && TOB.Signal.Off.FitPar.emp>0",2,legend,"TOB");
  DrawSame("number:TEC.Signal.Off.FitPar.mp:TEC.Signal.Off.FitPar.emp","Clusters.entries_all>2000 && TEC.Signal.Off.FitPar.mp>0 && TEC.Signal.Off.FitPar.emp>0",3,legend,"TEC");
  DrawSame("number:TID.Signal.Off.FitPar.mp:TID.Signal.Off.FitPar.emp","Clusters.entries_all>2000 && TID.Signal.Off.FitPar.mp>0 && TID.Signal.Off.FitPar.emp>0",4,legend,"TID");

  //=========== Global StoN off Track ==========//
  TCanvas *cStoN_Off=new TCanvas("cStoN_Off","cStoN_Off_Track",10,10,900,500);
  DetList.Add(cStoN_Off);
  DrawFunction(cStoN_Off,"number:TIB.StoN.Off.FitPar.mp:TIB.StoN.Off.FitPar.emp","Clusters.entries_all>2000 && TIB.StoN.Off.FitPar.mp>0 && TIB.StoN.Off.FitPar.emp>0",1,"StoN ected Off Track (MP)",legend,"TIB");
  DrawSame("number:TOB.StoN.Off.FitPar.mp:TOB.StoN.Off.FitPar.emp","Clusters.entries_all>2000 && TOB.StoN.Off.FitPar.mp>0 && TOB.StoN.Off.FitPar.emp>0",2,legend,"TOB");
  DrawSame("number:TEC.StoN.Off.FitPar.mp:TEC.StoN.Off.FitPar.emp","Clusters.entries_all>2000 && TEC.StoN.Off.FitPar.mp>0 && TEC.StoN.Off.FitPar.emp>0",3,legend,"TEC");
  DrawSame("number:TID.StoN.Off.FitPar.mp:TID.StoN.Off.FitPar.emp","Clusters.entries_all>2000 && TID.StoN.Off.FitPar.mp>0 && TID.StoN.Off.FitPar.emp>0",4,legend,"TID");

  //============Per Layer==============//
  //============Signal ected MP==========//
  
 //=============TIB================//
  TCanvas *cSignalTIB=new TCanvas("cSignalTIB","cSignalTIB",10,10,900,500);
  DetList.Add(cSignalTIB);
  DrawFunction(cSignalTIB,"number:TIB.cSignal.L1.FitPar.mp:TIB.cSignal.L1.FitPar.emp","Clusters.entries_all>2000 && TIB.cSignal.L1.FitPar.mp>0 && TIB.cSignal.L1.FitPar.emp>0",1,"TIB, Signal ected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TIB.cSignal.L2.FitPar.mp:TIB.cSignal.L2.FitPar.emp","Clusters.entries_all>2000 && TIB.cSignal.L2.FitPar.mp>0 && TIB.cSignal.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TIB.cSignal.L3.FitPar.mp:TIB.cSignal.L3.FitPar.emp","Clusters.entries_all>2000 && TIB.cSignal.L3.FitPar.mp>0 && TIB.cSignal.L3.FitPar.emp>0",3,legend,"L3");
  DrawSame("number:TIB.cSignal.L4.FitPar.mp:TIB.cSignal.L4.FitPar.emp","Clusters.entries_all>2000 && TIB.cSignal.L4.FitPar.mp>0 && TIB.cSignal.L4.FitPar.emp>0",4,legend,"L4"); 

 //==========TOB=============//
  TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
  TCanvas *cSignalTOB=new TCanvas("cSignalTOB","cSignalTOB",10,10,900,500); 
  DetList.Add(cSignalTOB);

  DrawFunction(cSignalTOB,"number:TOB.cSignal.L1.FitPar.mp:TOB.cSignal.L1.FitPar.emp","Clusters.entries_all>2000 && TOB.cSignal.L1.FitPar.mp>0 && TOB.cSignal.L1.FitPar.emp>0",1,"TOB, Signal ected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TOB.cSignal.L2.FitPar.mp:TOB.cSignal.L2.FitPar.emp","Clusters.entries_all>2000 && TOB.cSignal.L2.FitPar.mp>0 && TOB.cSignal.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TOB.cSignal.L3.FitPar.mp:TOB.cSignal.L3.FitPar.emp","Clusters.entries_all>2000 && TOB.cSignal.L3.FitPar.mp>0 && TOB.cSignal.L3.FitPar.emp>0",3,legend,"L3");
  DrawSame("number:TOB.cSignal.L4.FitPar.mp:TOB.cSignal.L4.FitPar.emp","Clusters.entries_all>2000 && TOB.cSignal.L4.FitPar.mp>0 && TOB.cSignal.L4.FitPar.emp>0",4,legend,"L4");
  DrawSame("number:TOB.cSignal.L5.FitPar.mp:TOB.cSignal.L5.FitPar.emp","Clusters.entries_all>2000 && TOB.cSignal.L5.FitPar.mp>0 && TOB.cSignal.L5.FitPar.emp>0",5,legend,"L5");
  DrawSame("number:TOB.cSignal.L6.FitPar.mp:TOB.cSignal.L6.FitPar.emp","Clusters.entries_all>2000 && TOB.cSignal.L6.FitPar.mp>0 && TOB.cSignal.L6.FitPar.emp>0",6,legend,"L6");

  //===========TEC==============//
 TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233);   
  TCanvas *cSignalTEC=new TCanvas("cSignalTEC","cSignalTEC",10,10,900,500); 
  DetList.Add(cSignalTEC);
  //===========TEC==============//
  DrawFunction(cSignalTEC,"number:TEC.cSignal.L1.FitPar.mp:TEC.cSignal.L1.FitPar.emp","TEC.cSignal.L1.FitPar.mp>0 && TEC.cSignal.L1.FitPar.emp>0",1,"TEC, Signal ected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TEC.cSignal.L2.FitPar.mp:TEC.cSignal.L2.FitPar.emp","TEC.cSignal.L2.FitPar.mp>0 && TEC.cSignal.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TEC.cSignal.L3.FitPar.mp:TEC.cSignal.L3.FitPar.emp","TEC.cSignal.L3.FitPar.mp>0 && TEC.cSignal.L3.FitPar.emp>0",3,legend,"L3");
  DrawSame("number:TEC.cSignal.L4.FitPar.mp:TEC.cSignal.L4.FitPar.emp","TEC.cSignal.L4.FitPar.mp>0 && TEC.cSignal.L4.FitPar.emp>0",4,legend,"L4");
  DrawSame("number:TEC.cSignal.L5.FitPar.mp:TEC.cSignal.L5.FitPar.emp","TEC.cSignal.L5.FitPar.mp>0 && TEC.cSignal.L5.FitPar.emp>0",5,legend,"L5");
  DrawSame("number:TEC.cSignal.L6.FitPar.mp:TEC.cSignal.L6.FitPar.emp","TEC.cSignal.L6.FitPar.mp>0 && TEC.cSignal.L6.FitPar.emp>0",6,legend,"L6");
  DrawSame("number:TEC.cSignal.L7.FitPar.mp:TEC.cSignal.L7.FitPar.emp","TEC.cSignal.L7.FitPar.mp>0 && TEC.cSignal.L7.FitPar.emp>0",7,legend,"L7");
  DrawSame("number:TEC.cSignal.L8.FitPar.mp:TEC.cSignal.L8.FitPar.emp","TEC.cSignal.L8.FitPar.mp>0 && TEC.cSignal.L8.FitPar.emp>0",8,legend,"L8");
  DrawSame("number:TEC.cSignal.L9.FitPar.mp:TEC.cSignal.L9.FitPar.emp","TEC.cSignal.L9.FitPar.mp>0 && TEC.cSignal.L9.FitPar.emp>0",28,legend,"L9");
  
  //==========TID===========//
  TCanvas *cSignalTID=new TCanvas("cSignalTID","cSignalTID",10,10,900,500); 
  TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
  DetList.Add(cSignalTID);

  DrawFunction(cSignalTID,"number:TID.cSignal.L1.FitPar.mp:TID.cSignal.L1.FitPar.emp","Clusters.entries_all>2000 && TID.cSignal.L1.FitPar.mp>0 && TID.cSignal.L1.FitPar.emp>0",1,"TID, Signal ected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TID.cSignal.L2.FitPar.mp:TID.cSignal.L2.FitPar.emp","Clusters.entries_all>2000 && TID.cSignal.L2.FitPar.mp>0 && TID.cSignal.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TID.cSignal.L3.FitPar.mp:TID.cSignal.L3.FitPar.emp","Clusters.entries_all>2000 && TID.cSignal.L3.FitPar.mp>0 && TID.cSignal.L3.FitPar.emp>0",3,legend,"L3");

//   //============StoN ected MP=========//
  
  //=============TIB================//
  TCanvas *cStoNTIB=new TCanvas("cStoNTIB","cStoNTIB",10,10,900,500); 
  DetList.Add(cStoNTIB);
  TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
  DrawFunction(cStoNTIB, "number:TIB.cStoN.L1.FitPar.mp:TIB.cStoN.L1.FitPar.emp","Clusters.entries_all>2000 && TIB.cStoN.L1.FitPar.mp>0 && TIB.cStoN.L1.FitPar.emp>0",1,"TIB, StoN ected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TIB.cStoN.L2.FitPar.mp:TIB.cStoN.L2.FitPar.emp","Clusters.entries_all>2000 && TIB.cStoN.L2.FitPar.mp>0 && TIB.cStoN.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TIB.cStoN.L3.FitPar.mp:TIB.cStoN.L3.FitPar.emp","Clusters.entries_all>2000 && TIB.cStoN.L3.FitPar.mp>0 && TIB.cStoN.L3.FitPar.emp>0",3,legend,"L3");
  DrawSame("number:TIB.cStoN.L4.FitPar.mp:TIB.cStoN.L4.FitPar.emp","Clusters.entries_all>2000 && TIB.cStoN.L4.FitPar.mp>0 && TIB.cStoN.L4.FitPar.emp>0",4,legend,"L4");
  
  //==========TOB=============//

  TCanvas *cStoNTOB=new TCanvas("cStoNTOB","cStoNTOB",10,10,900,500); 
  DetList.Add(cStoNTOB);
  TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
  DrawFunction(cStoNTOB,"number:TOB.cStoN.L1.FitPar.mp:TOB.cStoN.L1.FitPar.emp","Clusters.entries_all>2000 && TOB.cStoN.L1.FitPar.mp>0 && TOB.cStoN.L1.FitPar.emp>0",1,"TOB, StoN ected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TOB.cStoN.L2.FitPar.mp:TOB.cStoN.L2.FitPar.emp","Clusters.entries_all>2000 && TOB.cStoN.L2.FitPar.mp>0 && TOB.cStoN.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TOB.cStoN.L3.FitPar.mp:TOB.cStoN.L3.FitPar.emp","Clusters.entries_all>2000 && TOB.cStoN.L3.FitPar.mp>0 && TOB.cStoN.L3.FitPar.emp>0",3,legend,"L3");
  DrawSame("number:TOB.cStoN.L4.FitPar.mp:TOB.cStoN.L4.FitPar.emp","Clusters.entries_all>2000 && TOB.cStoN.L4.FitPar.mp>0 && TOB.cStoN.L4.FitPar.emp>0",4,legend,"L4");
  DrawSame("number:TOB.cStoN.L5.FitPar.mp:TOB.cStoN.L5.FitPar.emp","Clusters.entries_all>2000 && TOB.cStoN.L5.FitPar.mp>0 && TOB.cStoN.L5.FitPar.emp>0",5,legend,"L5");
  DrawSame("number:TOB.cStoN.L6.FitPar.mp:TOB.cStoN.L6.FitPar.emp","Clusters.entries_all>2000 && TOB.cStoN.L6.FitPar.mp>0 && TOB.cStoN.L6.FitPar.emp>0",6,legend,"L6");

  //===========TEC==============//
  TCanvas *cStoNTEC=new TCanvas("cStoNTEC","cStoNTEC",10,10,900,500); 
  DetList.Add(cStoNTEC); 
 TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
  DrawFunction(cStoNTEC,"number:TEC.cStoN.L1.FitPar.mp:TEC.cStoN.L1.FitPar.emp","TEC.cStoN.L1.FitPar.mp>0 && TEC.cStoN.L1.FitPar.emp>0",1,"TEC, StoN ected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TEC.cStoN.L2.FitPar.mp:TEC.cStoN.L2.FitPar.emp","TEC.cStoN.L2.FitPar.mp>0 && TEC.cStoN.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TEC.cStoN.L3.FitPar.mp:TEC.cStoN.L3.FitPar.emp","TEC.cStoN.L3.FitPar.mp>0 && TEC.cStoN.L3.FitPar.emp>0",3,legend,"L3");
  DrawSame("number:TEC.cStoN.L4.FitPar.mp:TEC.cStoN.L4.FitPar.emp","TEC.cStoN.L4.FitPar.mp>0 && TEC.cStoN.L4.FitPar.emp>0",4,legend,"L4");
  DrawSame("number:TEC.cStoN.L5.FitPar.mp:TEC.cStoN.L5.FitPar.emp","TEC.cStoN.L5.FitPar.mp>0 && TEC.cStoN.L5.FitPar.emp>0",5,legend,"L5");
  DrawSame("number:TEC.cStoN.L6.FitPar.mp:TEC.cStoN.L6.FitPar.emp","TEC.cStoN.L6.FitPar.mp>0 && TEC.cStoN.L6.FitPar.emp>0",6,legend,"L6");
  DrawSame("number:TEC.cStoN.L7.FitPar.mp:TEC.cStoN.L7.FitPar.emp","TEC.cStoN.L7.FitPar.mp>0 && TEC.cStoN.L7.FitPar.emp>0",7,legend,"L7");
  DrawSame("number:TEC.cStoN.L8.FitPar.mp:TEC.cStoN.L8.FitPar.emp","TEC.cStoN.L8.FitPar.mp>0 && TEC.cStoN.L8.FitPar.emp>0",8,legend,"L8");
  DrawSame("number:TEC.cStoN.L9.FitPar.mp:TEC.cStoN.L9.FitPar.emp","TEC.cStoN.L9.FitPar.mp>0 && TEC.cStoN.L9.FitPar.emp>0",28,legend,"L9");
  
  //==========TID===========//
  TCanvas *cStoNTID=new TCanvas("cStoNTID","cStoNTID",10,10,900,500); 
  DetList.Add(cStoNTID); 
  TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
  DrawFunction(cStoNTID,"number:TID.cStoN.L1.FitPar.mp:TID.cStoN.L1.FitPar.emp","Clusters.entries_all>2000 && TID.cStoN.L1.FitPar.mp>0 && TID.cStoN.L1.FitPar.emp>0",1,"TID, StoN ected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TID.cStoN.L2.FitPar.mp:TID.cStoN.L2.FitPar.emp","Clusters.entries_all>2000 && TID.cStoN.L2.FitPar.mp>0 && TID.cStoN.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TID.cStoN.L3.FitPar.mp:TID.cStoN.L3.FitPar.emp","Clusters.entries_all>2000 && TID.cStoN.L3.FitPar.mp>0 && TID.cStoN.L3.FitPar.emp>0",3,legend,"L3");
 
  //============Noise on Track=========//
  
  //=============TIB================//
  TCanvas *cNoiseTIB=new TCanvas("cNoiseTIB","cNoiseTIB",10,10,900,500); 
  TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 

  DetList.Add(cNoiseTIB); 

  DrawFunction(cNoiseTIB,"number:TIB.cNoise.L1.FitNoisePar.fitmean:TIB.cNoise.L1.FitNoisePar.fitrms","Clusters.entries_all>2000 && TIB.cNoise.L1.FitNoisePar.fitmean>0 && TIB.cNoise.L1.FitNoisePar.fitrms>0",1,"TIB, Noise (Gauss Fit Mean Value)",legend,"L1");
  DrawSame("number:TIB.cNoise.L2.FitNoisePar.fitmean:TIB.cNoise.L2.FitNoisePar.fitrms","Clusters.entries_all>2000 && TIB.cNoise.L2.FitNoisePar.fitmean>0 && TIB.cNoise.L2.FitNoisePar.fitrms>0",2,legend,"L2");
  DrawSame("number:TIB.cNoise.L3.FitNoisePar.fitmean:TIB.cNoise.L3.FitNoisePar.fitrms","Clusters.entries_all>2000 && TIB.cNoise.L3.FitNoisePar.fitmean>0 && TIB.cNoise.L3.FitNoisePar.fitrms>0",3,legend,"L3");
  DrawSame("number:TIB.cNoise.L4.FitNoisePar.fitmean:TIB.cNoise.L4.FitNoisePar.fitrms","Clusters.entries_all>2000 && TIB.cNoise.L4.FitNoisePar.fitmean>0 && TIB.cNoise.L4.FitNoisePar.fitrms>0",4,legend,"L4");

  //==========TOB=============//
  TCanvas *cNoiseTOB=new TCanvas("cNoiseTOB","cNoiseTOB",10,10,900,500); 
  TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 

  DetList.Add(cNoiseTOB); 

  DrawFunction(cNoiseTOB,"number:TOB.cNoise.L1.FitNoisePar.fitmean:TOB.cNoise.L1.FitNoisePar.fitrms","Clusters.entries_all>2000 && TOB.cNoise.L1.FitNoisePar.fitmean>0 && TOB.cNoise.L1.FitNoisePar.fitrms>0",1,"TOB, Noise (Gauss Fit Mean Value)",legend,"L1");
  DrawSame("number:TOB.cNoise.L2.FitNoisePar.fitmean:TOB.cNoise.L2.FitNoisePar.fitrms","Clusters.entries_all>2000 && TOB.cNoise.L2.FitNoisePar.fitmean>0 && TOB.cNoise.L2.FitNoisePar.fitrms>0",2,legend,"L2");
  DrawSame("number:TOB.cNoise.L3.FitNoisePar.fitmean:TOB.cNoise.L3.FitNoisePar.fitrms","Clusters.entries_all>2000 && TOB.cNoise.L3.FitNoisePar.fitmean>0 && TOB.cNoise.L3.FitNoisePar.fitrms>0",3,legend,"L3");
  DrawSame("number:TOB.cNoise.L4.FitNoisePar.fitmean:TOB.cNoise.L4.FitNoisePar.fitrms","Clusters.entries_all>2000 && TOB.cNoise.L4.FitNoisePar.fitmean>0 && TOB.cNoise.L4.FitNoisePar.fitrms>0",4,legend,"L4");
  DrawSame("number:TOB.cNoise.L5.FitNoisePar.fitmean:TOB.cNoise.L5.FitNoisePar.fitrms","Clusters.entries_all>2000 && TOB.cNoise.L5.FitNoisePar.fitmean>0 && TOB.cNoise.L5.FitNoisePar.fitrms>0",5,legend,"L5");
  DrawSame("number:TOB.cNoise.L6.FitNoisePar.fitmean:TOB.cNoise.L6.FitNoisePar.fitrms","Clusters.entries_all>2000 && TOB.cNoise.L6.FitNoisePar.fitmean>0 && TOB.cNoise.L6.FitNoisePar.fitrms>0",6,legend,"L6");

 //===========TEC==============//
  TCanvas *cNoiseTEC=new TCanvas("cNoiseTEC","cNoiseTEC",10,10,900,500); 
  DetList.Add(cNoiseTEC);
 TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233);  
  DrawFunction(cNoiseTEC,"number:TEC.cNoise.L1.FitNoisePar.fitmean:TEC.cNoise.L1.FitNoisePar.fitrms","TEC.cNoise.L1.FitNoisePar.fitmean>0 && TEC.cNoise.L1.FitNoisePar.fitrms>0",1,"TEC, Noise (Gauss Fit Mean Value)",legend,"L1");
  DrawSame("number:TEC.cNoise.L2.FitNoisePar.fitmean:TEC.cNoise.L2.FitNoisePar.fitrms","TEC.cNoise.L2.FitNoisePar.fitmean>0 && TEC.cNoise.L2.FitNoisePar.fitrms>0",2,legend,"L2");
  DrawSame("number:TEC.cNoise.L3.FitNoisePar.fitmean:TEC.cNoise.L3.FitNoisePar.fitrms","TEC.cNoise.L3.FitNoisePar.fitmean>0 && TEC.cNoise.L3.FitNoisePar.fitrms>0",3,legend,"L3");
  DrawSame("number:TEC.cNoise.L4.FitNoisePar.fitmean:TEC.cNoise.L4.FitNoisePar.fitrms","TEC.cNoise.L4.FitNoisePar.fitmean>0 && TEC.cNoise.L4.FitNoisePar.fitrms>0",4,legend,"L4");
  DrawSame("number:TEC.cNoise.L5.FitNoisePar.fitmean:TEC.cNoise.L5.FitNoisePar.fitrms","TEC.cNoise.L5.FitNoisePar.fitmean>0 && TEC.cNoise.L5.FitNoisePar.fitrms>0",5,legend,"L5");
  DrawSame("number:TEC.cNoise.L6.FitNoisePar.fitmean:TEC.cNoise.L6.FitNoisePar.fitrms","TEC.cNoise.L6.FitNoisePar.fitmean>0 && TEC.cNoise.L6.FitNoisePar.fitrms>0",6,legend,"L6");
  DrawSame("number:TEC.cNoise.L7.FitNoisePar.fitmean:TEC.cNoise.L7.FitNoisePar.fitrms","TEC.cNoise.L7.FitNoisePar.fitmean>0 && TEC.cNoise.L7.FitNoisePar.fitrms>0",7,legend,"L7");
  DrawSame("number:TEC.cNoise.L8.FitNoisePar.fitmean:TEC.cNoise.L8.FitNoisePar.fitrms","TEC.cNoise.L8.FitNoisePar.fitmean>0 && TEC.cNoise.L8.FitNoisePar.fitrms>0",8,legend,"L8");
  DrawSame("number:TEC.cNoise.L9.FitNoisePar.fitmean:TEC.cNoise.L9.FitNoisePar.fitrms","TEC.cNoise.L9.FitNoisePar.fitmean>0 && TEC.cNoise.L9.FitNoisePar.fitrms>0",28,legend,"L9");
      
   //==========TID===========//
  TCanvas *cNoiseTID=new TCanvas("cNoiseTID","cNoiseTID",10,10,900,500); 
  DetList.Add(cNoiseTID); 
 TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
  DrawFunction(cNoiseTID,"number:TID.cNoise.L1.FitNoisePar.fitmean:TID.cNoise.L1.FitNoisePar.fitrms","Clusters.entries_all>2000 && TID.cNoise.L1.FitNoisePar.fitmean>0 && TID.cNoise.L1.FitNoisePar.fitrms>0",1,"TID, Noise (Gauss Fit Mean Value)",legend,"L1");
  DrawSame("number:TID.cNoise.L2.FitNoisePar.fitmean:TID.cNoise.L2.FitNoisePar.fitrms","Clusters.entries_all>2000 && TID.cNoise.L2.FitNoisePar.fitmean>0 && TID.cNoise.L2.FitNoisePar.fitrms>0",2,legend,"L2");
  DrawSame("number:TID.cNoise.L3.FitNoisePar.fitmean:TID.cNoise.L3.FitNoisePar.fitrms","Clusters.entries_all>2000 && TID.cNoise.L3.FitNoisePar.fitmean>0 && TID.cNoise.L3.FitNoisePar.fitrms>0",3,legend,"L3");

  out->cd();
  DetList.Write();
  f->cd();
 
}

void DrawGlobal(TObjArray List,TFile *in, TFile *out){
  TObjArray List;

  TCanvas *MeanTk = new TCanvas("MeanTk","MeanTk",10,10,900,500);

  TIFTree->Project("h0","Tracks.MeanTrack:number","Clusters.entries_all>2000");
  
  h0->SetTitle("Number of Events with at least on Track/Total number of Events");
  h0->GetXaxis()->SetTitle("Run Number");
  h0->GetXaxis()->CenterTitle(1);
  h0->GetYaxis()->SetTitle("Fraction of Events with nTracks non zero");
  h0->GetYaxis()->CenterTitle(1); 
  h0->SetMarkerStyle(20);
  h0->SetStats(0);
  h0->Draw();
  List.Add(MeanTk);

  TCanvas *Events = new TCanvas("Events","Events",10,10,900,500);

  TIFTree->Project("h3","Clusters.entries_all:number","Clusters.entries_all>2000");
  
  h3->SetTitle("Total Number of Events");
  h3->GetXaxis()->SetTitle("Run Number");
  h3->GetXaxis()->CenterTitle(1);
  h3->GetYaxis()->SetTitle("Events");
  h3->GetYaxis()->CenterTitle(1); 
  h3->SetMarkerStyle(20);
  h3->SetStats(0);
  h3->Draw();
  List.Add(Events);

  TCanvas *nTracks = new TCanvas("nTracks","nTracks",10,10,900,500);

  TIFTree->Project("h","Tracks.mean:number","Clusters.entries_all>2000");
  
  h->SetTitle("Mean Track number per Event");
  h->GetXaxis()->SetTitle("Run Number");
  h->GetXaxis()->CenterTitle(1);
  h->GetYaxis()->SetTitle("Mean Track number per Event");
  h->GetYaxis()->CenterTitle(1); 
  h->SetMarkerStyle(20);
  h->SetStats(0);
  h->Draw();
  List.Add(nTracks);
  
  TCanvas *nRecHits= new TCanvas("nRecHits","nRecHits",10,10,900,500);  
  TIFTree->Project("h1","RecHits.mean:number","Clusters.entries_all>2000");
  
  h1->SetTitle("Mean RecHits Number per Event");
  h1->GetXaxis()->SetTitle("Run Number");
  h1->GetXaxis()->CenterTitle(1);
  h1->GetYaxis()->SetTitle("Mean RecHits Number per Event");
  h1->GetYaxis()->CenterTitle(1); 
  h1->SetMarkerStyle(20);
  h1->SetStats(0);
  h1->Draw();
  List.Add(nRecHits);

  TCanvas *nCluster= new TCanvas("nClusters","nClusters",10,10,900,500);
  TIFTree->Project("h2","Clusters.mean_corr:number","Clusters.entries_all>2000");
  
  h2->SetTitle("Mean Number of Cluster per Event");
  h2->GetXaxis()->SetTitle("Run Number");
  h2->GetXaxis()->CenterTitle(1);
  h2->GetYaxis()->SetTitle("Mean Number of Cluster per Event");
  h2->GetYaxis()->CenterTitle(1); 
  h2->SetMarkerStyle(20);
  h2->SetStats(0);
  h2->Draw();
  List.Add(nClusters);

 out->cd();
 List.Write();
 // out->Close();

 in->cd();
}

void DrawFunction(TCanvas *c,char* varToPlot, char* cond, Int_t kColor, char* Title, TLegend *leg, char* cLeg){
  c->cd();
  // TRACE
  TGraphErrors *g;
  TIFTree->Draw(varToPlot,cond,"goff");
  
  Int_t nSel=TIFTree->GetSelectedRows();
  if ( nSel ) {
    
    Double_t *ErrX= new Double_t[nSel];
    for ( Int_t i=0; i<nSel; i++) ErrX[i]=0;
    
    g = new TGraphErrors(TIFTree->GetSelectedRows(), TIFTree->GetV1(),  TIFTree->GetV2(), ErrX, TIFTree->GetV3());
    
    g->SetMarkerStyle(21);
    g->SetMarkerSize(0.5);
    g->SetMarkerColor(kColor);
    g->SetLineColor(kColor);
    g->Draw("Ap"); //draw graph in current pad
    
    g->SetTitle(Title);
    g->GetXaxis()->SetTitle("run");
    g->GetYaxis()->SetTitle(Title);
    //    g->GetYaxis()->SetRangeUser(40., 100.);
    
    leg->AddEntry(g, cLeg);
   delete[] ErrX;
  }
  else {
    cout << "NO rows selected " << endl;
  }

}

void DrawSame(char* varToPlot, char* cond, Int_t kColor,TLegend* leg,char* cLeg)
{
  TGraphErrors *g;
  TIFTree->Draw(varToPlot,cond,"goff");
  
  Int_t nSel=TIFTree->GetSelectedRows();
  if ( nSel ) {
    
    Double_t *ErrX= new Double_t[nSel];
    for ( Int_t i=0; i<nSel; i++) ErrX[i]=0;
    
    g = new TGraphErrors(TIFTree->GetSelectedRows(), TIFTree->GetV1(),  TIFTree->GetV2(), ErrX, TIFTree->GetV3());
    
    g->SetMarkerStyle(21);
    g->SetMarkerSize(0.5);
    g->SetMarkerColor(kColor);
    g->SetLineColor(kColor);
    g->Draw("SP"); //draw graph in current pad
    
    //    g->GetYaxis()->SetRangeUser(40., 100.);
    leg->AddEntry(g,cLeg);
    leg->Draw();
    delete[] ErrX;
  }
  else {
    cout << "NO rows selected " << endl;
  }
}


