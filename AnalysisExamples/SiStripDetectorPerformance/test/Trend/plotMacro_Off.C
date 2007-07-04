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


//   //============Global Noise on Track==========//
//   TCanvas *cNoise_On=new TCanvas("cNoise_On","cNoise_On_Track",10,10,900,500);
//   DetList.Add(cNoise_On);
//   DrawFunction(cNoise_On,"number:TIB.Noise.On.FitNoisePar.fitmean:TIB.Noise.On.FitNoisePar.fitrms","Clusters.entries_all>2000 && TIB.Noise.On.FitNoisePar.fitmean>0 && TIB.Noise.On.FitNoisePar.fitrms>0",1,"Mean Cluster Noise On Track (Fit mean)");
//   DrawSame("number:TOB.Noise.On.FitNoisePar.fitmean:TOB.Noise.On.FitNoisePar.fitrms","Clusters.entries_all>2000 && TOB.Noise.On.FitNoisePar.fitmean>0 && TOB.Noise.On.FitNoisePar.fitrms>0",2);
//   DrawSame("number:TEC.Noise.On.FitNoisePar.fitmean:TEC.Noise.On.FitNoisePar.fitrms","Clusters.entries_all>2000 && TEC.Noise.On.FitNoisePar.fitmean>0 && TEC.Noise.On.FitNoisePar.fitrms>0",3);
//   DrawSame("number:TID.Noise.On.FitNoisePar.fitmean:TID.Noise.On.FitNoisePar.fitrms","Clusters.entries_all>2000 && TID.Noise.On.FitNoisePar.fitmean>0 && TID.Noise.On.FitNoisePar.fitrms>0",4);
 
  //============Signal Corrected MP==========//
  
 //=============TIB================//
  TCanvas *cSignalCorrTIB=new TCanvas("cSignalCorrTIB","cSignalCorrTIB",10,10,900,500);
  DetList.Add(cSignalCorrTIB);
  DrawFunction(cSignalCorrTIB,"number:TIB.cSignalCorr.L1.FitPar.mp:TIB.cSignalCorr.L1.FitPar.emp","Clusters.entries_all>2000 && TIB.cSignalCorr.L1.FitPar.mp>0 && TIB.cSignalCorr.L1.FitPar.emp>0",1,"TIB, Signal Corrected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TIB.cSignalCorr.L2.FitPar.mp:TIB.cSignalCorr.L2.FitPar.emp","Clusters.entries_all>2000 && TIB.cSignalCorr.L2.FitPar.mp>0 && TIB.cSignalCorr.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TIB.cSignalCorr.L3.FitPar.mp:TIB.cSignalCorr.L3.FitPar.emp","Clusters.entries_all>2000 && TIB.cSignalCorr.L3.FitPar.mp>0 && TIB.cSignalCorr.L3.FitPar.emp>0",3,legend,"L3");
  DrawSame("number:TIB.cSignalCorr.L4.FitPar.mp:TIB.cSignalCorr.L4.FitPar.emp","Clusters.entries_all>2000 && TIB.cSignalCorr.L4.FitPar.mp>0 && TIB.cSignalCorr.L4.FitPar.emp>0",4,legend,"L4"); 

 //==========TOB=============//
  TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
  TCanvas *cSignalCorrTOB=new TCanvas("cSignalCorrTOB","cSignalCorrTOB",10,10,900,500); 
  DetList.Add(cSignalCorrTOB);

  DrawFunction(cSignalCorrTOB,"number:TOB.cSignalCorr.L1.FitPar.mp:TOB.cSignalCorr.L1.FitPar.emp","Clusters.entries_all>2000 && TOB.cSignalCorr.L1.FitPar.mp>0 && TOB.cSignalCorr.L1.FitPar.emp>0",1,"TOB, Signal Corrected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TOB.cSignalCorr.L2.FitPar.mp:TOB.cSignalCorr.L2.FitPar.emp","Clusters.entries_all>2000 && TOB.cSignalCorr.L2.FitPar.mp>0 && TOB.cSignalCorr.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TOB.cSignalCorr.L3.FitPar.mp:TOB.cSignalCorr.L3.FitPar.emp","Clusters.entries_all>2000 && TOB.cSignalCorr.L3.FitPar.mp>0 && TOB.cSignalCorr.L3.FitPar.emp>0",3,legend,"L3");
  DrawSame("number:TOB.cSignalCorr.L4.FitPar.mp:TOB.cSignalCorr.L4.FitPar.emp","Clusters.entries_all>2000 && TOB.cSignalCorr.L4.FitPar.mp>0 && TOB.cSignalCorr.L4.FitPar.emp>0",4,legend,"L4");
  DrawSame("number:TOB.cSignalCorr.L5.FitPar.mp:TOB.cSignalCorr.L5.FitPar.emp","Clusters.entries_all>2000 && TOB.cSignalCorr.L5.FitPar.mp>0 && TOB.cSignalCorr.L5.FitPar.emp>0",5,legend,"L5");
  DrawSame("number:TOB.cSignalCorr.L6.FitPar.mp:TOB.cSignalCorr.L6.FitPar.emp","Clusters.entries_all>2000 && TOB.cSignalCorr.L6.FitPar.mp>0 && TOB.cSignalCorr.L6.FitPar.emp>0",6,legend,"L6");

  //===========TEC==============//
 TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233);   
  TCanvas *cSignalCorrTEC=new TCanvas("cSignalCorrTEC","cSignalCorrTEC",10,10,900,500); 
  DetList.Add(cSignalCorrTEC);
  //===========TEC==============//
  DrawFunction(cSignalCorrTEC,"number:TEC.cSignalCorr.L1.FitPar.mp:TEC.cSignalCorr.L1.FitPar.emp","TEC.cSignalCorr.L1.FitPar.mp>0 && TEC.cSignalCorr.L1.FitPar.emp>0",1,"TEC, Signal Corrected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TEC.cSignalCorr.L2.FitPar.mp:TEC.cSignalCorr.L2.FitPar.emp","TEC.cSignalCorr.L2.FitPar.mp>0 && TEC.cSignalCorr.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TEC.cSignalCorr.L3.FitPar.mp:TEC.cSignalCorr.L3.FitPar.emp","TEC.cSignalCorr.L3.FitPar.mp>0 && TEC.cSignalCorr.L3.FitPar.emp>0",3,legend,"L3");
  DrawSame("number:TEC.cSignalCorr.L4.FitPar.mp:TEC.cSignalCorr.L4.FitPar.emp","TEC.cSignalCorr.L4.FitPar.mp>0 && TEC.cSignalCorr.L4.FitPar.emp>0",4,legend,"L4");
  DrawSame("number:TEC.cSignalCorr.L5.FitPar.mp:TEC.cSignalCorr.L5.FitPar.emp","TEC.cSignalCorr.L5.FitPar.mp>0 && TEC.cSignalCorr.L5.FitPar.emp>0",5,legend,"L5");
  DrawSame("number:TEC.cSignalCorr.L6.FitPar.mp:TEC.cSignalCorr.L6.FitPar.emp","TEC.cSignalCorr.L6.FitPar.mp>0 && TEC.cSignalCorr.L6.FitPar.emp>0",6,legend,"L6");
  DrawSame("number:TEC.cSignalCorr.L7.FitPar.mp:TEC.cSignalCorr.L7.FitPar.emp","TEC.cSignalCorr.L7.FitPar.mp>0 && TEC.cSignalCorr.L7.FitPar.emp>0",7,legend,"L7");
  DrawSame("number:TEC.cSignalCorr.L8.FitPar.mp:TEC.cSignalCorr.L8.FitPar.emp","TEC.cSignalCorr.L8.FitPar.mp>0 && TEC.cSignalCorr.L8.FitPar.emp>0",8,legend,"L8");
  DrawSame("number:TEC.cSignalCorr.L9.FitPar.mp:TEC.cSignalCorr.L9.FitPar.emp","TEC.cSignalCorr.L9.FitPar.mp>0 && TEC.cSignalCorr.L9.FitPar.emp>0",28,legend,"L9");
  
  //==========TID===========//
  TCanvas *cSignalCorrTID=new TCanvas("cSignalCorrTID","cSignalCorrTID",10,10,900,500); 
  TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
  DetList.Add(cSignalCorrTID);

  DrawFunction(cSignalCorrTID,"number:TID.cSignalCorr.L1.FitPar.mp:TID.cSignalCorr.L1.FitPar.emp","Clusters.entries_all>2000 && TID.cSignalCorr.L1.FitPar.mp>0 && TID.cSignalCorr.L1.FitPar.emp>0",1,"TID, Signal Corrected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TID.cSignalCorr.L2.FitPar.mp:TID.cSignalCorr.L2.FitPar.emp","Clusters.entries_all>2000 && TID.cSignalCorr.L2.FitPar.mp>0 && TID.cSignalCorr.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TID.cSignalCorr.L3.FitPar.mp:TID.cSignalCorr.L3.FitPar.emp","Clusters.entries_all>2000 && TID.cSignalCorr.L3.FitPar.mp>0 && TID.cSignalCorr.L3.FitPar.emp>0",3,legend,"L3");

//   //============StoN Corrected MP=========//
  
  //=============TIB================//
  TCanvas *cStoNCorrTIB=new TCanvas("cStoNCorrTIB","cStoNCorrTIB",10,10,900,500); 
  DetList.Add(cStoNCorrTIB);
  TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
  DrawFunction(cStoNCorrTIB, "number:TIB.cStoNCorr.L1.FitPar.mp:TIB.cStoNCorr.L1.FitPar.emp","Clusters.entries_all>2000 && TIB.cStoNCorr.L1.FitPar.mp>0 && TIB.cStoNCorr.L1.FitPar.emp>0",1,"TIB, StoN Corrected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TIB.cStoNCorr.L2.FitPar.mp:TIB.cStoNCorr.L2.FitPar.emp","Clusters.entries_all>2000 && TIB.cStoNCorr.L2.FitPar.mp>0 && TIB.cStoNCorr.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TIB.cStoNCorr.L3.FitPar.mp:TIB.cStoNCorr.L3.FitPar.emp","Clusters.entries_all>2000 && TIB.cStoNCorr.L3.FitPar.mp>0 && TIB.cStoNCorr.L3.FitPar.emp>0",3,legend,"L3");
  DrawSame("number:TIB.cStoNCorr.L4.FitPar.mp:TIB.cStoNCorr.L4.FitPar.emp","Clusters.entries_all>2000 && TIB.cStoNCorr.L4.FitPar.mp>0 && TIB.cStoNCorr.L4.FitPar.emp>0",4,legend,"L4");
  
  //==========TOB=============//

  TCanvas *cStoNCorrTOB=new TCanvas("cStoNCorrTOB","cStoNCorrTOB",10,10,900,500); 
  DetList.Add(cStoNCorrTOB);
  TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
  DrawFunction(cStoNCorrTOB,"number:TOB.cStoNCorr.L1.FitPar.mp:TOB.cStoNCorr.L1.FitPar.emp","Clusters.entries_all>2000 && TOB.cStoNCorr.L1.FitPar.mp>0 && TOB.cStoNCorr.L1.FitPar.emp>0",1,"TOB, StoN Corrected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TOB.cStoNCorr.L2.FitPar.mp:TOB.cStoNCorr.L2.FitPar.emp","Clusters.entries_all>2000 && TOB.cStoNCorr.L2.FitPar.mp>0 && TOB.cStoNCorr.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TOB.cStoNCorr.L3.FitPar.mp:TOB.cStoNCorr.L3.FitPar.emp","Clusters.entries_all>2000 && TOB.cStoNCorr.L3.FitPar.mp>0 && TOB.cStoNCorr.L3.FitPar.emp>0",3,legend,"L3");
  DrawSame("number:TOB.cStoNCorr.L4.FitPar.mp:TOB.cStoNCorr.L4.FitPar.emp","Clusters.entries_all>2000 && TOB.cStoNCorr.L4.FitPar.mp>0 && TOB.cStoNCorr.L4.FitPar.emp>0",4,legend,"L4");
  DrawSame("number:TOB.cStoNCorr.L5.FitPar.mp:TOB.cStoNCorr.L5.FitPar.emp","Clusters.entries_all>2000 && TOB.cStoNCorr.L5.FitPar.mp>0 && TOB.cStoNCorr.L5.FitPar.emp>0",5,legend,"L5");
  DrawSame("number:TOB.cStoNCorr.L6.FitPar.mp:TOB.cStoNCorr.L6.FitPar.emp","Clusters.entries_all>2000 && TOB.cStoNCorr.L6.FitPar.mp>0 && TOB.cStoNCorr.L6.FitPar.emp>0",6,legend,"L6");

  //===========TEC==============//
  TCanvas *cStoNCorrTEC=new TCanvas("cStoNCorrTEC","cStoNCorrTEC",10,10,900,500); 
  DetList.Add(cStoNCorrTEC); 
 TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
  DrawFunction(cStoNCorrTEC,"number:TEC.cStoNCorr.L1.FitPar.mp:TEC.cStoNCorr.L1.FitPar.emp","TEC.cStoNCorr.L1.FitPar.mp>0 && TEC.cStoNCorr.L1.FitPar.emp>0",1,"TEC, StoN Corrected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TEC.cStoNCorr.L2.FitPar.mp:TEC.cStoNCorr.L2.FitPar.emp","TEC.cStoNCorr.L2.FitPar.mp>0 && TEC.cStoNCorr.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TEC.cStoNCorr.L3.FitPar.mp:TEC.cStoNCorr.L3.FitPar.emp","TEC.cStoNCorr.L3.FitPar.mp>0 && TEC.cStoNCorr.L3.FitPar.emp>0",3,legend,"L3");
  DrawSame("number:TEC.cStoNCorr.L4.FitPar.mp:TEC.cStoNCorr.L4.FitPar.emp","TEC.cStoNCorr.L4.FitPar.mp>0 && TEC.cStoNCorr.L4.FitPar.emp>0",4,legend,"L4");
  DrawSame("number:TEC.cStoNCorr.L5.FitPar.mp:TEC.cStoNCorr.L5.FitPar.emp","TEC.cStoNCorr.L5.FitPar.mp>0 && TEC.cStoNCorr.L5.FitPar.emp>0",5,legend,"L5");
  DrawSame("number:TEC.cStoNCorr.L6.FitPar.mp:TEC.cStoNCorr.L6.FitPar.emp","TEC.cStoNCorr.L6.FitPar.mp>0 && TEC.cStoNCorr.L6.FitPar.emp>0",6,legend,"L6");
  DrawSame("number:TEC.cStoNCorr.L7.FitPar.mp:TEC.cStoNCorr.L7.FitPar.emp","TEC.cStoNCorr.L7.FitPar.mp>0 && TEC.cStoNCorr.L7.FitPar.emp>0",7,legend,"L7");
  DrawSame("number:TEC.cStoNCorr.L8.FitPar.mp:TEC.cStoNCorr.L8.FitPar.emp","TEC.cStoNCorr.L8.FitPar.mp>0 && TEC.cStoNCorr.L8.FitPar.emp>0",8,legend,"L8");
  DrawSame("number:TEC.cStoNCorr.L9.FitPar.mp:TEC.cStoNCorr.L9.FitPar.emp","TEC.cStoNCorr.L9.FitPar.mp>0 && TEC.cStoNCorr.L9.FitPar.emp>0",28,legend,"L9");
  
  //==========TID===========//
  TCanvas *cStoNCorrTID=new TCanvas("cStoNCorrTID","cStoNCorrTID",10,10,900,500); 
  DetList.Add(cStoNCorrTID); 
  TLegend *legend = new TLegend(0.874414,0.597285,0.997825,0.997233); 
  DrawFunction(cStoNCorrTID,"number:TID.cStoNCorr.L1.FitPar.mp:TID.cStoNCorr.L1.FitPar.emp","Clusters.entries_all>2000 && TID.cStoNCorr.L1.FitPar.mp>0 && TID.cStoNCorr.L1.FitPar.emp>0",1,"TID, StoN Corrected for the Angle (Landau mp)",legend,"L1");
  DrawSame("number:TID.cStoNCorr.L2.FitPar.mp:TID.cStoNCorr.L2.FitPar.emp","Clusters.entries_all>2000 && TID.cStoNCorr.L2.FitPar.mp>0 && TID.cStoNCorr.L2.FitPar.emp>0",2,legend,"L2");
  DrawSame("number:TID.cStoNCorr.L3.FitPar.mp:TID.cStoNCorr.L3.FitPar.emp","Clusters.entries_all>2000 && TID.cStoNCorr.L3.FitPar.mp>0 && TID.cStoNCorr.L3.FitPar.emp>0",3,legend,"L3");
 
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
    g->Draw("S"); //draw graph in current pad
    
    //    g->GetYaxis()->SetRangeUser(40., 100.);
    leg->AddEntry(g,cLeg);
    leg->Draw();
    delete[] ErrX;
  }
  else {
    cout << "NO rows selected " << endl;
  }
}


