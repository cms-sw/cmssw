#include<Riostream>	       
#include "TCanvas.h"
#include "TObject.h"

using namespace std;
void DrawClus(TCanvas *c,char* varToPlot, char* cond, Int_t kColor, Int_t kMarker, char* Title,char* xTitle,char* yTitle, TLegend *leg, char* cLeg,Double_t downlim,Double_t uplim);
void DrawSame(char* varToPlot,char* cond,Int_t kColor, Int_t kMarker,TLegend* leg,char* cLeg);

void PlotMacro(){
  TFile *out = new TFile("/tmp/maborgia/out_Tc_TOBL4.root","recreate"); 
  TFile *f = new TFile("/tmp/maborgia/pass4_8055_ClusThr_TOB_L4.root"); 
  //  TFile *f = new TFile("/tmp/maborgia/ClusterThr/pass4_8055_ClusThr_TOB_L4.root");
  TObjArray DetList(0);

  f->cd("AsciiOutput");
  TCanvas *C3=new TCanvas("cC3","cC3",10,10,900,900);
  DetList.Add(cC3);
  results->Project("h","NTs:Tn","Tc==9.5 && Ts==3.5");
  h->Draw();
  TCanvas *C1=new TCanvas("cC1","cC1",10,10,900,900); 
  DetList.Add(cC1);
  C1->Divide(2,2);

  C1->cd(1);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC1,"Tc:Nb","Ts==2.5 && Tn==1.5",2,20,"TOB L4,Mean number of BG cluster per event","Tc","Mean BG clusters",legend,"Ts2.5:Tn1.5",0.,0.4);
  DrawSame("Tc:Nb","Ts==2.75 && Tn==1.5",3,20,legend,"Ts2.75:Tn1.5");
  DrawSame("Tc:Nb","Ts==3.0 && Tn==1.5",4,20,legend,"Ts3.0:Tn1.5");
  DrawSame("Tc:Nb","Ts==3.25 && Tn==1.5",5,20,legend,"Ts3.25:Tn1.5");
  DrawSame("Tc:Nb","Ts==3.5 && Tn==1.5",6,20,legend,"Ts3.5:Tn1.5");
  DrawSame("Tc:Nb","Ts==3.75 && Tn==1.5",7,20,legend,"Ts3.75:Tn1.5");
  DrawSame("Tc:Nb","Ts==4.0 && Tn==1.5",8,20,legend,"Ts4.0:Tn1.5");
  DrawSame("Tc:Nb","Ts==2.5 && Tn==1.75",2,21,legend,"Ts2.5:Tn1.75");
  DrawSame("Tc:Nb","Ts==2.75 && Tn==1.75",3,21,legend,"Ts2.75:Tn1.75");
  DrawSame("Tc:Nb","Ts==3.0 && Tn==1.75",4,21,legend,"Ts3.0:Tn1.75");
  DrawSame("Tc:Nb","Ts==3.25 && Tn==1.75",5,21,legend,"Ts3.25:Tn1.75");
  DrawSame("Tc:Nb","Ts==3.5 && Tn==1.75",6,21,legend,"Ts3.5:Tn1.75");
  DrawSame("Tc:Nb","Ts==3.75 && Tn==1.75",7,21,legend,"Ts3.75:Tn1.75");
  DrawSame("Tc:Nb","Ts==4.0 && Tn==1.75",8,21,legend,"Ts4.0:Tn1.75");
  DrawSame("Tc:Nb","Ts==2.5 && Tn==2",2,22,legend,"Ts2.5:Tn2");
  DrawSame("Tc:Nb","Ts==2.75 && Tn==2",3,22,legend,"Ts2.75:Tn2");
  DrawSame("Tc:Nb","Ts==3.0 && Tn==2",4,22,legend,"Ts3.0:Tn2");
  DrawSame("Tc:Nb","Ts==3.25 && Tn==2",5,22,legend,"Ts3.25:Tn2");
  DrawSame("Tc:Nb","Ts==3.5 && Tn==2",6,22,legend,"Ts3.5:Tn2");
  DrawSame("Tc:Nb","Ts==3.75 && Tn==2",7,22,legend,"Ts3.75:Tn2");
  DrawSame("Tc:Nb","Ts==4.0 && Tn==2",8,22,legend,"Ts4.0:Tn2");
  DrawSame("Tc:Nb","Ts==2.5 && Tn==2.25",2,23,legend,"Ts2.5:Tn2.25");
  DrawSame("Tc:Nb","Ts==2.75 && Tn==2.25",3,23,legend,"Ts2.75:Tn2.25");
  DrawSame("Tc:Nb","Ts==3.0 && Tn==2.25",4,23,legend,"Ts3.0:Tn2.25");
  DrawSame("Tc:Nb","Ts==3.25 && Tn==2.25",5,23,legend,"Ts3.25:Tn2.25");
  DrawSame("Tc:Nb","Ts==3.5 && Tn==2.25",6,23,legend,"Ts3.5:Tn2.25");
  DrawSame("Tc:Nb","Ts==3.75 && Tn==2.25",7,23,legend,"Ts3.75:Tn2.25");
  DrawSame("Tc:Nb","Ts==4.0 && Tn==2.25",8,23,legend,"Ts4.0:Tn2.25");

  C1->cd(2);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC1,"Tc:Ns","Ts==2.5 && Tn==1.5",2,20,"TOB L4,Mean number of Signal cluster per event","Tc","Mean Signal clusters",legend,"Ts2.5:Tn1.5",1.2,1.38);
  DrawSame("Tc:Ns","Ts==2.75 && Tn==1.5",3,20,legend,"Ts2.75:Tn1.5");
  DrawSame("Tc:Ns","Ts==3.0 && Tn==1.5",4,20,legend,"Ts3.0:Tn1.5");
  DrawSame("Tc:Ns","Ts==3.25 && Tn==1.5",5,20,legend,"Ts3.25:Tn1.5");
  DrawSame("Tc:Ns","Ts==3.5 && Tn==1.5",6,20,legend,"Ts3.5:Tn1.5");
  DrawSame("Tc:Ns","Ts==3.75 && Tn==1.5",7,20,legend,"Ts3.75:Tn1.5");
  DrawSame("Tc:Ns","Ts==4.0 && Tn==1.5",8,20,legend,"Ts4.0:Tn1.5");
  DrawSame("Tc:Ns","Ts==2.5 && Tn==1.75",2,21,legend,"Ts2.5:Tn1.75");
  DrawSame("Tc:Ns","Ts==2.75 && Tn==1.75",3,21,legend,"Ts2.75:Tn1.75");
  DrawSame("Tc:Ns","Ts==3.0 && Tn==1.75",4,21,legend,"Ts3.0:Tn1.75");
  DrawSame("Tc:Ns","Ts==3.25 && Tn==1.75",5,21,legend,"Ts3.25:Tn1.75");
  DrawSame("Tc:Ns","Ts==3.5 && Tn==1.75",6,21,legend,"Ts3.5:Tn1.75");
  DrawSame("Tc:Ns","Ts==3.75 && Tn==1.75",7,21,legend,"Ts3.75:Tn1.75");
  DrawSame("Tc:Ns","Ts==4.0 && Tn==1.75",8,21,legend,"Ts4.0:Tn1.75");
  DrawSame("Tc:Ns","Ts==2.5 && Tn==2",2,22,legend,"Ts2.5:Tn2");
  DrawSame("Tc:Ns","Ts==2.75 && Tn==2",3,22,legend,"Ts2.75:Tn2");
  DrawSame("Tc:Ns","Ts==3.0 && Tn==2",4,22,legend,"Ts3.0:Tn2");
  DrawSame("Tc:Ns","Ts==3.25 && Tn==2",5,22,legend,"Ts3.25:Tn2");
  DrawSame("Tc:Ns","Ts==3.5 && Tn==2",6,22,legend,"Ts3.5:Tn2");
  DrawSame("Tc:Ns","Ts==3.75 && Tn==2",7,22,legend,"Ts3.75:Tn2");
  DrawSame("Tc:Ns","Ts==4.0 && Tn==2",8,22,legend,"Ts4.0:Tn2");
  DrawSame("Tc:Ns","Ts==2.5 && Tn==2.25",2,23,legend,"Ts2.5:Tn2.25");
  DrawSame("Tc:Ns","Ts==2.75 && Tn==2.25",3,23,legend,"Ts2.75:Tn2.25");
  DrawSame("Tc:Ns","Ts==3.0 && Tn==2.25",4,23,legend,"Ts3.0:Tn2.25");
  DrawSame("Tc:Ns","Ts==3.25 && Tn==2.25",5,23,legend,"Ts3.25:Tn2.25");
  DrawSame("Tc:Ns","Ts==3.5 && Tn==2.25",6,23,legend,"Ts3.5:Tn2.25");
  DrawSame("Tc:Ns","Ts==3.75 && Tn==2.25",7,23,legend,"Ts3.75:Tn2.25");
  DrawSame("Tc:Ns","Ts==4.0 && Tn==2.25",8,23,legend,"Ts4.0:Tn2.25");

  C1->cd(3);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC1,"Tc:MPVs","Ts==2.5 && Tn==1.5",2,20,"TOB L4,MPV of S/N","Tc","MPV",legend,"Ts2.5:Tn1.5",38.4,40);
  DrawSame("Tc:MPVs","Ts==2.75 && Tn==1.5",3,20,legend,"Ts2.75:Tn1.5");
  DrawSame("Tc:MPVs","Ts==3.0 && Tn==1.5",4,20,legend,"Ts3.0:Tn1.5");
  DrawSame("Tc:MPVs","Ts==3.25 && Tn==1.5",5,20,legend,"Ts3.25:Tn1.5");
  DrawSame("Tc:MPVs","Ts==3.5 && Tn==1.5",6,20,legend,"Ts3.5:Tn1.5");
  DrawSame("Tc:MPVs","Ts==3.75 && Tn==1.5",7,20,legend,"Ts3.75:Tn1.5");
  DrawSame("Tc:MPVs","Ts==4.0 && Tn==1.5",8,20,legend,"Ts4.0:Tn1.5");
  DrawSame("Tc:MPVs","Ts==2.5 && Tn==1.75",2,21,legend,"Ts2.5:Tn1.75");
  DrawSame("Tc:MPVs","Ts==2.75 && Tn==1.75",3,21,legend,"Ts2.75:Tn1.75");
  DrawSame("Tc:MPVs","Ts==3.0 && Tn==1.75",4,21,legend,"Ts3.0:Tn1.75");
  DrawSame("Tc:MPVs","Ts==3.25 && Tn==1.75",5,21,legend,"Ts3.25:Tn1.75");
  DrawSame("Tc:MPVs","Ts==3.5 && Tn==1.75",6,21,legend,"Ts3.5:Tn1.75");
  DrawSame("Tc:MPVs","Ts==3.75 && Tn==1.75",7,21,legend,"Ts3.75:Tn1.75");
  DrawSame("Tc:MPVs","Ts==4.0 && Tn==1.75",8,21,legend,"Ts4.0:Tn1.75");
  DrawSame("Tc:MPVs","Ts==2.5 && Tn==2",2,22,legend,"Ts2.5:Tn2");
  DrawSame("Tc:MPVs","Ts==2.75 && Tn==2",3,22,legend,"Ts2.75:Tn2");
  DrawSame("Tc:MPVs","Ts==3.0 && Tn==2",4,22,legend,"Ts3.0:Tn2");
  DrawSame("Tc:MPVs","Ts==3.25 && Tn==2",5,22,legend,"Ts3.25:Tn2");
  DrawSame("Tc:MPVs","Ts==3.5 && Tn==2",6,22,legend,"Ts3.5:Tn2");
  DrawSame("Tc:MPVs","Ts==3.75 && Tn==2",7,22,legend,"Ts3.75:Tn2");
  DrawSame("Tc:MPVs","Ts==4.0 && Tn==2",8,22,legend,"Ts4.0:Tn2");
  DrawSame("Tc:MPVs","Ts==2.5 && Tn==2.25",2,23,legend,"Ts2.5:Tn2.25");
  DrawSame("Tc:MPVs","Ts==2.75 && Tn==2.25",3,23,legend,"Ts2.75:Tn2.25");
  DrawSame("Tc:MPVs","Ts==3.0 && Tn==2.25",4,23,legend,"Ts3.0:Tn2.25");
  DrawSame("Tc:MPVs","Ts==3.25 && Tn==2.25",5,23,legend,"Ts3.25:Tn2.25");
  DrawSame("Tc:MPVs","Ts==3.5 && Tn==2.25",6,23,legend,"Ts3.5:Tn2.25");
  DrawSame("Tc:MPVs","Ts==3.75 && Tn==2.25",7,23,legend,"Ts3.75:Tn2.25");
  DrawSame("Tc:MPVs","Ts==4.0 && Tn==2.25",8,23,legend,"Ts4.0:Tn2.25");

  C1->cd(4);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC1,"Tc:FWHMs","Ts==2.5 && Tn==1.5",2,20,"TOB L4,FWHM of S/N","Tc","FWHM",legend,"Ts2.5:Tn1.5",5.5,5.9);
  DrawSame("Tc:FWHMs","Ts==2.75 && Tn==1.5",3,20,legend,"Ts2.75:Tn1.5");
  DrawSame("Tc:FWHMs","Ts==3.0 && Tn==1.5",4,20,legend,"Ts3.0:Tn1.5");
  DrawSame("Tc:FWHMs","Ts==3.25 && Tn==1.5",5,20,legend,"Ts3.25:Tn1.5");
  DrawSame("Tc:FWHMs","Ts==3.5 && Tn==1.5",6,20,legend,"Ts3.5:Tn1.5");
  DrawSame("Tc:FWHMs","Ts==3.75 && Tn==1.5",7,20,legend,"Ts3.75:Tn1.5");
  DrawSame("Tc:FWHMs","Ts==4.0 && Tn==1.5",8,20,legend,"Ts4.0:Tn1.5");
  DrawSame("Tc:FWHMs","Ts==2.5 && Tn==1.75",2,21,legend,"Ts2.5:Tn1.75");
  DrawSame("Tc:FWHMs","Ts==2.75 && Tn==1.75",3,21,legend,"Ts2.75:Tn1.75");
  DrawSame("Tc:FWHMs","Ts==3.0 && Tn==1.75",4,21,legend,"Ts3.0:Tn1.75");
  DrawSame("Tc:FWHMs","Ts==3.25 && Tn==1.75",5,21,legend,"Ts3.25:Tn1.75");
  DrawSame("Tc:FWHMs","Ts==3.5 && Tn==1.75",6,21,legend,"Ts3.5:Tn1.75");
  DrawSame("Tc:FWHMs","Ts==3.75 && Tn==1.75",7,21,legend,"Ts3.75:Tn1.75");
  DrawSame("Tc:FWHMs","Ts==4.0 && Tn==1.75",8,21,legend,"Ts4.0:Tn1.75");
  DrawSame("Tc:FWHMs","Ts==2.5 && Tn==2",2,22,legend,"Ts2.5:Tn2");
  DrawSame("Tc:FWHMs","Ts==2.75 && Tn==2",3,22,legend,"Ts2.75:Tn2");
  DrawSame("Tc:FWHMs","Ts==3.0 && Tn==2",4,22,legend,"Ts3.0:Tn2");
  DrawSame("Tc:FWHMs","Ts==3.25 && Tn==2",5,22,legend,"Ts3.25:Tn2");
  DrawSame("Tc:FWHMs","Ts==3.5 && Tn==2",6,22,legend,"Ts3.5:Tn2");
  DrawSame("Tc:FWHMs","Ts==3.75 && Tn==2",7,22,legend,"Ts3.75:Tn2");
  DrawSame("Tc:FWHMs","Ts==4.0 && Tn==2",8,22,legend,"Ts4.0:Tn2");
  DrawSame("Tc:FWHMs","Ts==2.5 && Tn==2.25",2,23,legend,"Ts2.5:Tn2.25");
  DrawSame("Tc:FWHMs","Ts==2.75 && Tn==2.25",3,23,legend,"Ts2.75:Tn2.25");
  DrawSame("Tc:FWHMs","Ts==3.0 && Tn==2.25",4,23,legend,"Ts3.0:Tn2.25");
  DrawSame("Tc:FWHMs","Ts==3.25 && Tn==2.25",5,23,legend,"Ts3.25:Tn2.25");
  DrawSame("Tc:FWHMs","Ts==3.5 && Tn==2.25",6,23,legend,"Ts3.5:Tn2.25");
  DrawSame("Tc:FWHMs","Ts==3.75 && Tn==2.25",7,23,legend,"Ts3.75:Tn2.25");
  DrawSame("Tc:FWHMs","Ts==4.0 && Tn==2.25",8,23,legend,"Ts4.0:Tn2.25");

  TCanvas *C2=new TCanvas("cC2","cC2",10,10,900,900); 
  DetList.Add(cC2);
  C2->Divide(2,2);

  C2->cd(1);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC2,"Tc:NTb","Ts==2.5 && Tn==1.5",2,20,"TOB L4,Total number of BG cluster","Tc","# clusters",legend,"Ts2.5:Tn1.5",0,4000);
  DrawSame("Tc:NTb","Ts==2.75 && Tn==1.5",3,20,legend,"Ts2.75:Tn1.5");
  DrawSame("Tc:NTb","Ts==3.0 && Tn==1.5",4,20,legend,"Ts3.0:Tn1.5");
  DrawSame("Tc:NTb","Ts==3.25 && Tn==1.5",5,20,legend,"Ts3.25:Tn1.5");
  DrawSame("Tc:NTb","Ts==3.5 && Tn==1.5",6,20,legend,"Ts3.5:Tn1.5");
  DrawSame("Tc:NTb","Ts==3.75 && Tn==1.5",7,20,legend,"Ts3.75:Tn1.5");
  DrawSame("Tc:NTb","Ts==4.0 && Tn==1.5",8,20,legend,"Ts4.0:Tn1.5");
  DrawSame("Tc:NTb","Ts==2.5 && Tn==1.75",2,21,legend,"Ts2.5:Tn1.75");
  DrawSame("Tc:NTb","Ts==2.75 && Tn==1.75",3,21,legend,"Ts2.75:Tn1.75");
  DrawSame("Tc:NTb","Ts==3.0 && Tn==1.75",4,21,legend,"Ts3.0:Tn1.75");
  DrawSame("Tc:NTb","Ts==3.25 && Tn==1.75",5,21,legend,"Ts3.25:Tn1.75");
  DrawSame("Tc:NTb","Ts==3.5 && Tn==1.75",6,21,legend,"Ts3.5:Tn1.75");
  DrawSame("Tc:NTb","Ts==3.75 && Tn==1.75",7,21,legend,"Ts3.75:Tn1.75");
  DrawSame("Tc:NTb","Ts==4.0 && Tn==1.75",8,21,legend,"Ts4.0:Tn1.75");
  DrawSame("Tc:NTb","Ts==2.5 && Tn==2",2,22,legend,"Ts2.5:Tn2");
  DrawSame("Tc:NTb","Ts==2.75 && Tn==2",3,22,legend,"Ts2.75:Tn2");
  DrawSame("Tc:NTb","Ts==3.0 && Tn==2",4,22,legend,"Ts3.0:Tn2");
  DrawSame("Tc:NTb","Ts==3.25 && Tn==2",5,22,legend,"Ts3.25:Tn2");
  DrawSame("Tc:NTb","Ts==3.5 && Tn==2",6,22,legend,"Ts3.5:Tn2");
  DrawSame("Tc:NTb","Ts==3.75 && Tn==2",7,22,legend,"Ts3.75:Tn2");
  DrawSame("Tc:NTb","Ts==4.0 && Tn==2",8,22,legend,"Ts4.0:Tn2");
  DrawSame("Tc:NTb","Ts==2.5 && Tn==2.25",2,23,legend,"Ts2.5:Tn2.25");
  DrawSame("Tc:NTb","Ts==2.75 && Tn==2.25",3,23,legend,"Ts2.75:Tn2.25");
  DrawSame("Tc:NTb","Ts==3.0 && Tn==2.25",4,23,legend,"Ts3.0:Tn2.25");
  DrawSame("Tc:NTb","Ts==3.25 && Tn==2.25",5,23,legend,"Ts3.25:Tn2.25");
  DrawSame("Tc:NTb","Ts==3.5 && Tn==2.25",6,23,legend,"Ts3.5:Tn2.25");
  DrawSame("Tc:NTb","Ts==3.75 && Tn==2.25",7,23,legend,"Ts3.75:Tn2.25");
  DrawSame("Tc:NTb","Ts==4.0 && Tn==2.25",8,23,legend,"Ts4.0:Tn2.25");

  C2->cd(2);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC2,"Tc:NTs","Ts==2.5 && Tn==1.5",2,20,"TOB L4,Total number of Signal cluster","Tc","# clusters",legend,"Ts2.5:Tn1.5",16800,17700);
  DrawSame("Tc:NTs","Ts==2.75 && Tn==1.5",3,20,legend,"Ts2.75:Tn1.5");
  DrawSame("Tc:NTs","Ts==3.0 && Tn==1.5",4,20,legend,"Ts3.0:Tn1.5");
  DrawSame("Tc:NTs","Ts==3.25 && Tn==1.5",5,20,legend,"Ts3.25:Tn1.5");
  DrawSame("Tc:NTs","Ts==3.5 && Tn==1.5",6,20,legend,"Ts3.5:Tn1.5");
  DrawSame("Tc:NTs","Ts==3.75 && Tn==1.5",7,20,legend,"Ts3.75:Tn1.5");
  DrawSame("Tc:NTs","Ts==4.0 && Tn==1.5",8,20,legend,"Ts4.0:Tn1.5");
  DrawSame("Tc:NTs","Ts==2.5 && Tn==1.75",2,21,legend,"Ts2.5:Tn1.75");
  DrawSame("Tc:NTs","Ts==2.75 && Tn==1.75",3,21,legend,"Ts2.75:Tn1.75");
  DrawSame("Tc:NTs","Ts==3.0 && Tn==1.75",4,21,legend,"Ts3.0:Tn1.75");
  DrawSame("Tc:NTs","Ts==3.25 && Tn==1.75",5,21,legend,"Ts3.25:Tn1.75");
  DrawSame("Tc:NTs","Ts==3.5 && Tn==1.75",6,21,legend,"Ts3.5:Tn1.75");
  DrawSame("Tc:NTs","Ts==3.75 && Tn==1.75",7,21,legend,"Ts3.75:Tn1.75");
  DrawSame("Tc:NTs","Ts==4.0 && Tn==1.75",8,21,legend,"Ts4.0:Tn1.75");
  DrawSame("Tc:NTs","Ts==2.5 && Tn==2",2,22,legend,"Ts2.5:Tn2");
  DrawSame("Tc:NTs","Ts==2.75 && Tn==2",3,22,legend,"Ts2.75:Tn2");
  DrawSame("Tc:NTs","Ts==3.0 && Tn==2",4,22,legend,"Ts3.0:Tn2");
  DrawSame("Tc:NTs","Ts==3.25 && Tn==2",5,22,legend,"Ts3.25:Tn2");
  DrawSame("Tc:NTs","Ts==3.5 && Tn==2",6,22,legend,"Ts3.5:Tn2");
  DrawSame("Tc:NTs","Ts==3.75 && Tn==2",7,22,legend,"Ts3.75:Tn2");
  DrawSame("Tc:NTs","Ts==4.0 && Tn==2",8,22,legend,"Ts4.0:Tn2");
  DrawSame("Tc:NTs","Ts==2.5 && Tn==2.25",2,23,legend,"Ts2.5:Tn2.25");
  DrawSame("Tc:NTs","Ts==2.75 && Tn==2.25",3,23,legend,"Ts2.75:Tn2.25");
  DrawSame("Tc:NTs","Ts==3.0 && Tn==2.25",4,23,legend,"Ts3.0:Tn2.25");
  DrawSame("Tc:NTs","Ts==3.25 && Tn==2.25",5,23,legend,"Ts3.25:Tn2.25");
  DrawSame("Tc:NTs","Ts==3.5 && Tn==2.25",6,23,legend,"Ts3.5:Tn2.25");
  DrawSame("Tc:NTs","Ts==3.75 && Tn==2.25",7,23,legend,"Ts3.75:Tn2.25");
  DrawSame("Tc:NTs","Ts==4.0 && Tn==2.25",8,23,legend,"Ts4.0:Tn2.25");

  C2->cd(3);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC2,"Tc:MeanWb","Ts==2.5 && Tn==1.5",2,20,"TOB L4, Mean Width for BG clusters","Tc","Width (# strips)",legend,"Tc2.5:Tn1.5",1.5,3.8);
  DrawSame("Tc:MeanWb","Ts==2.75 && Tn==1.5",3,20,legend,"Ts2.75:Tn1.5");
  DrawSame("Tc:MeanWb","Ts==3.0 && Tn==1.5",4,20,legend,"Ts3.0:Tn1.5");
  DrawSame("Tc:MeanWb","Ts==3.25 && Tn==1.5",5,20,legend,"Ts3.25:Tn1.5");
  DrawSame("Tc:MeanWb","Ts==3.5 && Tn==1.5",6,20,legend,"Ts3.5:Tn1.5");
  DrawSame("Tc:MeanWb","Ts==3.75 && Tn==1.5",7,20,legend,"Ts3.75:Tn1.5");
  DrawSame("Tc:MeanWb","Ts==4.0 && Tn==1.5",8,20,legend,"Ts4.0:Tn1.5");
  DrawSame("Tc:MeanWb","Ts==2.5 && Tn==1.75",2,21,legend,"Ts2.5:Tn1.75");
  DrawSame("Tc:MeanWb","Ts==2.75 && Tn==1.75",3,21,legend,"Ts2.75:Tn1.75");
  DrawSame("Tc:MeanWb","Ts==3.0 && Tn==1.75",4,21,legend,"Ts3.0:Tn1.75");
  DrawSame("Tc:MeanWb","Ts==3.25 && Tn==1.75",5,21,legend,"Ts3.25:Tn1.75");
  DrawSame("Tc:MeanWb","Ts==3.5 && Tn==1.75",6,21,legend,"Ts3.5:Tn1.75");
  DrawSame("Tc:MeanWb","Ts==3.75 && Tn==1.75",7,21,legend,"Ts3.75:Tn1.75");
  DrawSame("Tc:MeanWb","Ts==4.0 && Tn==1.75",8,21,legend,"Ts4.0:Tn1.75");
  DrawSame("Tc:MeanWb","Ts==2.5 && Tn==2",2,22,legend,"Ts2.5:Tn2");
  DrawSame("Tc:MeanWb","Ts==2.75 && Tn==2",3,22,legend,"Ts2.75:Tn2");
  DrawSame("Tc:MeanWb","Ts==3.0 && Tn==2",4,22,legend,"Ts3.0:Tn2");
  DrawSame("Tc:MeanWb","Ts==3.25 && Tn==2",5,22,legend,"Ts3.25:Tn2");
  DrawSame("Tc:MeanWb","Ts==3.5 && Tn==2",6,22,legend,"Ts3.5:Tn2");
  DrawSame("Tc:MeanWb","Ts==3.75 && Tn==2",7,22,legend,"Ts3.75:Tn2");
  DrawSame("Tc:MeanWb","Ts==4.0 && Tn==2",8,22,legend,"Ts4.0:Tn2");
  DrawSame("Tc:MeanWb","Ts==2.5 && Tn==2.25",2,23,legend,"Ts2.5:Tn2.25");
  DrawSame("Tc:MeanWb","Ts==2.75 && Tn==2.25",3,23,legend,"Ts2.75:Tn2.25");
  DrawSame("Tc:MeanWb","Ts==3.0 && Tn==2.25",4,23,legend,"Ts3.0:Tn2.25");
  DrawSame("Tc:MeanWb","Ts==3.25 && Tn==2.25",5,23,legend,"Ts3.25:Tn2.25");
  DrawSame("Tc:MeanWb","Ts==3.5 && Tn==2.25",6,23,legend,"Ts3.5:Tn2.25");
  DrawSame("Tc:MeanWb","Ts==3.75 && Tn==2.25",7,23,legend,"Ts3.75:Tn2.25");
  DrawSame("Tc:MeanWb","Ts==4.0 && Tn==2.25",8,23,legend,"Ts4.0:Tn2.25");

  C2->cd(4);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC2,"Tc:MeanWs","Ts==2.5 && Tn==1.5",2,20,"TOB L4,Mean Width for Signal clusters","Tc","Width (# strips)",legend,"Tc2.5:Tn1.5",3.7,4.3);
  DrawSame("Tc:MeanWs","Ts==2.75 && Tn==1.5",3,20,legend,"Ts2.75:Tn1.5");
  DrawSame("Tc:MeanWs","Ts==3.0 && Tn==1.5",4,20,legend,"Ts3.0:Tn1.5");
  DrawSame("Tc:MeanWs","Ts==3.25 && Tn==1.5",5,20,legend,"Ts3.25:Tn1.5");
  DrawSame("Tc:MeanWs","Ts==3.5 && Tn==1.5",6,20,legend,"Ts3.5:Tn1.5");
  DrawSame("Tc:MeanWs","Ts==3.75 && Tn==1.5",7,20,legend,"Ts3.75:Tn1.5");
  DrawSame("Tc:MeanWs","Ts==4.0 && Tn==1.5",8,20,legend,"Ts4.0:Tn1.5");
  DrawSame("Tc:MeanWs","Ts==2.5 && Tn==1.75",2,21,legend,"Ts2.5:Tn1.75");
  DrawSame("Tc:MeanWs","Ts==2.75 && Tn==1.75",3,21,legend,"Ts2.75:Tn1.75");
  DrawSame("Tc:MeanWs","Ts==3.0 && Tn==1.75",4,21,legend,"Ts3.0:Tn1.75");
  DrawSame("Tc:MeanWs","Ts==3.25 && Tn==1.75",5,21,legend,"Ts3.25:Tn1.75");
  DrawSame("Tc:MeanWs","Ts==3.5 && Tn==1.75",6,21,legend,"Ts3.5:Tn1.75");
  DrawSame("Tc:MeanWs","Ts==3.75 && Tn==1.75",7,21,legend,"Ts3.75:Tn1.75");
  DrawSame("Tc:MeanWs","Ts==4.0 && Tn==1.75",8,21,legend,"Ts4.0:Tn1.75");
  DrawSame("Tc:MeanWs","Ts==2.5 && Tn==2",2,22,legend,"Ts2.5:Tn2");
  DrawSame("Tc:MeanWs","Ts==2.75 && Tn==2",3,22,legend,"Ts2.75:Tn2");
  DrawSame("Tc:MeanWs","Ts==3.0 && Tn==2",4,22,legend,"Ts3.0:Tn2");
  DrawSame("Tc:MeanWs","Ts==3.25 && Tn==2",5,22,legend,"Ts3.25:Tn2");
  DrawSame("Tc:MeanWs","Ts==3.5 && Tn==2",6,22,legend,"Ts3.5:Tn2");
  DrawSame("Tc:MeanWs","Ts==3.75 && Tn==2",7,22,legend,"Ts3.75:Tn2");
  DrawSame("Tc:MeanWs","Ts==4.0 && Tn==2",8,22,legend,"Ts4.0:Tn2");
  DrawSame("Tc:MeanWs","Ts==2.5 && Tn==2.25",2,23,legend,"Ts2.5:Tn2.25");
  DrawSame("Tc:MeanWs","Ts==2.75 && Tn==2.25",3,23,legend,"Ts2.75:Tn2.25");
  DrawSame("Tc:MeanWs","Ts==3.0 && Tn==2.25",4,23,legend,"Ts3.0:Tn2.25");
  DrawSame("Tc:MeanWs","Ts==3.25 && Tn==2.25",5,23,legend,"Ts3.25:Tn2.25");
  DrawSame("Tc:MeanWs","Ts==3.5 && Tn==2.25",6,23,legend,"Ts3.5:Tn2.25");
  DrawSame("Tc:MeanWs","Ts==3.75 && Tn==2.25",7,23,legend,"Ts3.75:Tn2.25");
  DrawSame("Tc:MeanWs","Ts==4.0 && Tn==2.25",8,23,legend,"Ts4.0:Tn2.25");

  out->cd();
  DetList.Write();
  f->cd();
   

}

void DrawClus(TCanvas *c,char* varToPlot, char* cond, Int_t kColor, Int_t kMarker, char* Title,char* xTitle,char* yTitle, TLegend *leg, char* cLeg,Double_t downlim,Double_t uplim){

 TGraph* g;
 results->Draw(varToPlot, cond,"goff");
 cout << results->GetSelectedRows() << endl;
 if(results->GetSelectedRows()){
   g=new TGraph(results->GetSelectedRows(),results->GetV1(),results->GetV2());
   
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

DrawSame(char* varToPlot, char* cond, Int_t kColor,Int_t kMarker,TLegend* leg,char* cLeg){
  TGraph *g;
  results->Draw(varToPlot,cond,"goff");
  
  if (results->GetSelectedRows()) {
    
    g = new TGraph(results->GetSelectedRows(), results->GetV1(), results->GetV2());

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
