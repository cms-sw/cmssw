#include<Riostream>	       
#include "TCanvas.h"
#include "TObject.h"

using namespace std;
void DrawClus(TCanvas *c,char* varToPlot, char* cond, Int_t kColor, Int_t kMarker, char* Title,char* xTitle,char* yTitle, TLegend *leg, char* cLeg,Double_t downlim,Double_t uplim);
void DrawSame(char* varToPlot,char* cond,Int_t kColor, Int_t kMarker,TLegend* leg,char* cLeg);

void PlotMacro(){
  TFile *out = new TFile("/tmp/maborgia/out_Tn_TOBL4.root","recreate");  
  TFile *f = new TFile("/tmp/maborgia/ClusterThr/pass4_8055_ClusThr_TOB_L4.root");
  TObjArray DetList(0);

  f->cd("AsciiOutput");

  TCanvas *C1=new TCanvas("cC1","cC1",10,10,900,900); 
  DetList.Add(cC1);
  C1->Divide(2,2);

  C1->cd(1);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC1,"Tn:Nb","Tc==4.5 && Ts==2.5",2,20,"TOB L4,Mean number of BG cluster per event","Tn","Mean BG clusters",legend,"Tc4.5:Ts2.5",0.15,0.4);
  DrawSame("Tn:Nb","Tc==5.0 && Ts==2.5",3,20,legend,"Tc5.0:Ts2.5");
  DrawSame("Tn:Nb","Tc==5.5 && Ts==2.5",4,20,legend,"Tc5.5:Ts2.5");
  DrawSame("Tn:Nb","Tc==4.5 && Ts==2.75",2,21,legend,"Tc4.5:Ts2.75");
  DrawSame("Tn:Nb","Tc==5.0 && Ts==2.75",3,21,legend,"Tc5.0:Ts2.75");
  DrawSame("Tn:Nb","Tc==5.5 && Ts==2.75",4,21,legend,"Tc5.5:Ts2.75");
  DrawSame("Tn:Nb","Tc==4.5 && Ts==3.0",2,22,legend,"Tc4.5:Ts3.0");
  DrawSame("Tn:Nb","Tc==5.0 && Ts==3.0",3,22,legend,"Tc5.0:Ts3.0");
  DrawSame("Tn:Nb","Tc==5.5 && Ts==3.0",4,22,legend,"Tc5.5:Ts3.0");
  DrawSame("Tn:Nb","Tc==4.5 && Ts==3.25",2,23,legend,"Tc4.5:Ts3.25");
  DrawSame("Tn:Nb","Tc==5.0 && Ts==3.25",3,23,legend,"Tc5.0:Ts3.25");
  DrawSame("Tn:Nb","Tc==5.5 && Ts==3.25",4,23,legend,"Tc5.5:Ts3.25");
  DrawSame("Tn:Nb","Tc==4.5 && Ts==3.5",2,24,legend,"Tc4.5:Ts3.5");
  DrawSame("Tn:Nb","Tc==5.0 && Ts==3.5",3,24,legend,"Tc5.0:Ts3.5");
  DrawSame("Tn:Nb","Tc==5.5 && Ts==3.5",4,24,legend,"Tc5.5:Ts3.5");
  DrawSame("Tn:Nb","Tc==4.5 && Ts==3.75",2,25,legend,"Tc4.5:Ts3.75");
  DrawSame("Tn:Nb","Tc==5.0 && Ts==3.75",3,25,legend,"Tc5.0:Ts3.75");
  DrawSame("Tn:Nb","Tc==5.5 && Ts==3.75",4,25,legend,"Tc5.5:Ts3.75");

  C1->cd(2);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC1,"Tn:Ns","Tc==4.5 && Ts==2.5",2,20,"TOB L4,Mean number of Signal cluster per event","Tn","Mean Signal clusters",legend,"Tc4.5:Ts2.5",1.36,1.52);
  DrawSame("Tn:Ns","Tc==5.0 && Ts==2.5",3,20,legend,"Tc5.0:Ts2.5");
  DrawSame("Tn:Ns","Tc==5.5 && Ts==2.5",4,20,legend,"Tc5.5:Ts2.5");
  DrawSame("Tn:Ns","Tc==4.5 && Ts==2.75",2,21,legend,"Tc4.5:Ts2.75");
  DrawSame("Tn:Ns","Tc==5.0 && Ts==2.75",3,21,legend,"Tc5.0:Ts2.75");
  DrawSame("Tn:Ns","Tc==5.5 && Ts==2.75",4,21,legend,"Tc5.5:Ts2.75");
  DrawSame("Tn:Ns","Tc==4.5 && Ts==3.0",2,22,legend,"Tc4.5:Ts3.0");
  DrawSame("Tn:Ns","Tc==5.0 && Ts==3.0",3,22,legend,"Tc5.0:Ts3.0");
  DrawSame("Tn:Ns","Tc==5.5 && Ts==3.0",4,22,legend,"Tc5.5:Ts3.0");
  DrawSame("Tn:Ns","Tc==4.5 && Ts==3.25",2,23,legend,"Tc4.5:Ts3.25");
  DrawSame("Tn:Ns","Tc==5.0 && Ts==3.25",3,23,legend,"Tc5.0:Ts3.25");
  DrawSame("Tn:Ns","Tc==5.5 && Ts==3.25",4,23,legend,"Tc5.5:Ts3.25");
  DrawSame("Tn:Ns","Tc==4.5 && Ts==3.5",2,24,legend,"Tc4.5:Ts3.5");
  DrawSame("Tn:Ns","Tc==5.0 && Ts==3.5",3,24,legend,"Tc5.0:Ts3.5");
  DrawSame("Tn:Ns","Tc==5.5 && Ts==3.5",4,24,legend,"Tc5.5:Ts3.5");
  DrawSame("Tn:Ns","Tc==4.5 && Ts==3.75",2,25,legend,"Tc4.5:Ts3.75");
  DrawSame("Tn:Ns","Tc==5.0 && Ts==3.75",3,25,legend,"Tc5.0:Ts3.75");
  DrawSame("Tn:Ns","Tc==5.5 && Ts==3.75",4,25,legend,"Tc5.5:Ts3.75");

  C1->cd(3);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC1,"Tn:MPVs","Tc==4.5 && Ts==2.5",2,20,"TOB L4,MPV of S/N","Tn","MPV",legend,"Tc4.5:Ts2.5",36,39);
  DrawSame("Tn:MPVs","Tc==5.0 && Ts==2.5",3,20,legend,"Tc5.0:Ts2.5");
  DrawSame("Tn:MPVs","Tc==5.5 && Ts==2.5",4,20,legend,"Tc5.5:Ts2.5");
  DrawSame("Tn:MPVs","Tc==4.5 && Ts==2.75",2,21,legend,"Tc4.5:Ts2.75");
  DrawSame("Tn:MPVs","Tc==5.0 && Ts==2.75",3,21,legend,"Tc5.0:Ts2.75");
  DrawSame("Tn:MPVs","Tc==5.5 && Ts==2.75",4,21,legend,"Tc5.5:Ts2.75");
  DrawSame("Tn:MPVs","Tc==4.5 && Ts==3.0",2,22,legend,"Tc4.5:Ts3.0");
  DrawSame("Tn:MPVs","Tc==5.0 && Ts==3.0",3,22,legend,"Tc5.0:Ts3.0");
  DrawSame("Tn:MPVs","Tc==5.5 && Ts==3.0",4,22,legend,"Tc5.5:Ts3.0");
  DrawSame("Tn:MPVs","Tc==4.5 && Ts==3.25",2,23,legend,"Tc4.5:Ts3.25");
  DrawSame("Tn:MPVs","Tc==5.0 && Ts==3.25",3,23,legend,"Tc5.0:Ts3.25");
  DrawSame("Tn:MPVs","Tc==5.5 && Ts==3.25",4,23,legend,"Tc5.5:Ts3.25");
  DrawSame("Tn:MPVs","Tc==4.5 && Ts==3.5",2,24,legend,"Tc4.5:Ts3.5");
  DrawSame("Tn:MPVs","Tc==5.0 && Ts==3.5",3,24,legend,"Tc5.0:Ts3.5");
  DrawSame("Tn:MPVs","Tc==5.5 && Ts==3.5",4,24,legend,"Tc5.5:Ts3.5");
  DrawSame("Tn:MPVs","Tc==4.5 && Ts==3.75",2,25,legend,"Tc4.5:Ts3.75");
  DrawSame("Tn:MPVs","Tc==5.0 && Ts==3.75",3,25,legend,"Tc5.0:Ts3.75");
  DrawSame("Tn:MPVs","Tc==5.5 && Ts==3.75",4,25,legend,"Tc5.5:Ts3.75");

  C1->cd(4);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC1,"Tn:FWHMs","Tc==4.5 && Ts==2.5",2,20,"TOB L4,FWHM of S/N","Tn","FWHM",legend,"Tc4.5:Ts2.5",5.4,6.2);
  DrawSame("Tn:FWHMs","Tc==5.0 && Ts==2.5",3,20,legend,"Tc5.0:Ts2.5");
  DrawSame("Tn:FWHMs","Tc==5.5 && Ts==2.5",4,20,legend,"Tc5.5:Ts2.5");
  DrawSame("Tn:FWHMs","Tc==4.5 && Ts==2.75",2,21,legend,"Tc4.5:Ts2.75");
  DrawSame("Tn:FWHMs","Tc==5.0 && Ts==2.75",3,21,legend,"Tc5.0:Ts2.75");
  DrawSame("Tn:FWHMs","Tc==5.5 && Ts==2.75",4,21,legend,"Tc5.5:Ts2.75");
  DrawSame("Tn:FWHMs","Tc==4.5 && Ts==3.0",2,22,legend,"Tc4.5:Ts3.0");
  DrawSame("Tn:FWHMs","Tc==5.0 && Ts==3.0",3,22,legend,"Tc5.0:Ts3.0");
  DrawSame("Tn:FWHMs","Tc==5.5 && Ts==3.0",4,22,legend,"Tc5.5:Ts3.0");
  DrawSame("Tn:FWHMs","Tc==4.5 && Ts==3.25",2,23,legend,"Tc4.5:Ts3.25");
  DrawSame("Tn:FWHMs","Tc==5.0 && Ts==3.25",3,23,legend,"Tc5.0:Ts3.25");
  DrawSame("Tn:FWHMs","Tc==5.5 && Ts==3.25",4,23,legend,"Tc5.5:Ts3.25");
  DrawSame("Tn:FWHMs","Tc==4.5 && Ts==3.5",2,24,legend,"Tc4.5:Ts3.5");
  DrawSame("Tn:FWHMs","Tc==5.0 && Ts==3.5",3,24,legend,"Tc5.0:Ts3.5");
  DrawSame("Tn:FWHMs","Tc==5.5 && Ts==3.5",4,24,legend,"Tc5.5:Ts3.5");
  DrawSame("Tn:FWHMs","Tc==4.5 && Ts==3.75",2,25,legend,"Tc4.5:Ts3.75");
  DrawSame("Tn:FWHMs","Tc==5.0 && Ts==3.75",3,25,legend,"Tc5.0:Ts3.75");
  DrawSame("Tn:FWHMs","Tc==5.5 && Ts==3.75",4,25,legend,"Tc5.5:Ts3.75");

  TCanvas *C2=new TCanvas("cC2","cC2",10,10,900,900); 
  DetList.Add(cC2);
  C2->Divide(2,2);

  C2->cd(1);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC2,"Tn:NTb","Tc==4.5 && Ts==2.5",2,20,"TOB L4,Total number of BG cluster","Tn","# clusters",legend,"Tc4.5:Ts2.5",1500,3700);
  DrawSame("Tn:NTb","Tc==5.0 && Ts==2.5",3,20,legend,"Tc5.0:Ts2.5");
  DrawSame("Tn:NTb","Tc==5.5 && Ts==2.5",4,20,legend,"Tc5.5:Ts2.5");
  DrawSame("Tn:NTb","Tc==4.5 && Ts==2.75",2,21,legend,"Tc4.5:Ts2.75");
  DrawSame("Tn:NTb","Tc==5.0 && Ts==2.75",3,21,legend,"Tc5.0:Ts2.75");
  DrawSame("Tn:NTb","Tc==5.5 && Ts==2.75",4,21,legend,"Tc5.5:Ts2.75");
  DrawSame("Tn:NTb","Tc==4.5 && Ts==3.0",2,22,legend,"Tc4.5:Ts3.0");
  DrawSame("Tn:NTb","Tc==5.0 && Ts==3.0",3,22,legend,"Tc5.0:Ts3.0");
  DrawSame("Tn:NTb","Tc==5.5 && Ts==3.0",4,22,legend,"Tc5.5:Ts3.0");
  DrawSame("Tn:NTb","Tc==4.5 && Ts==3.25",2,23,legend,"Tc4.5:Ts3.25");
  DrawSame("Tn:NTb","Tc==5.0 && Ts==3.25",3,23,legend,"Tc5.0:Ts3.25");
  DrawSame("Tn:NTb","Tc==5.5 && Ts==3.25",4,23,legend,"Tc5.5:Ts3.25");
  DrawSame("Tn:NTb","Tc==4.5 && Ts==3.5",2,24,legend,"Tc4.5:Ts3.5");
  DrawSame("Tn:NTb","Tc==5.0 && Ts==3.5",3,24,legend,"Tc5.0:Ts3.5");
  DrawSame("Tn:NTb","Tc==5.5 && Ts==3.5",4,24,legend,"Tc5.5:Ts3.5");
  DrawSame("Tn:NTb","Tc==4.5 && Ts==3.75",2,25,legend,"Tc4.5:Ts3.75");
  DrawSame("Tn:NTb","Tc==5.0 && Ts==3.75",3,25,legend,"Tc5.0:Ts3.75");
  DrawSame("Tn:NTb","Tc==5.5 && Ts==3.75",4,25,legend,"Tc5.5:Ts3.75");

  C2->cd(2);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC2,"Tn:NTs","Tc==4.5 && Ts==2.5",2,20,"TOB L4,Total number of Signal cluster","Tn","# clusters",legend,"Tc4.5:Ts2.5",17900,19500);
  DrawSame("Tn:NTs","Tc==5.0 && Ts==2.5",3,20,legend,"Tc5.0:Ts2.5");
  DrawSame("Tn:NTs","Tc==5.5 && Ts==2.5",4,20,legend,"Tc5.5:Ts2.5");
  DrawSame("Tn:NTs","Tc==4.5 && Ts==2.75",2,21,legend,"Tc4.5:Ts2.75");
  DrawSame("Tn:NTs","Tc==5.0 && Ts==2.75",3,21,legend,"Tc5.0:Ts2.75");
  DrawSame("Tn:NTs","Tc==5.5 && Ts==2.75",4,21,legend,"Tc5.5:Ts2.75");
  DrawSame("Tn:NTs","Tc==4.5 && Ts==3.0",2,22,legend,"Tc4.5:Ts3.0");
  DrawSame("Tn:NTs","Tc==5.0 && Ts==3.0",3,22,legend,"Tc5.0:Ts3.0");
  DrawSame("Tn:NTs","Tc==5.5 && Ts==3.0",4,22,legend,"Tc5.5:Ts3.0");
  DrawSame("Tn:NTs","Tc==4.5 && Ts==3.25",2,23,legend,"Tc4.5:Ts3.25");
  DrawSame("Tn:NTs","Tc==5.0 && Ts==3.25",3,23,legend,"Tc5.0:Ts3.25");
  DrawSame("Tn:NTs","Tc==5.5 && Ts==3.25",4,23,legend,"Tc5.5:Ts3.25");
  DrawSame("Tn:NTs","Tc==4.5 && Ts==3.5",2,24,legend,"Tc4.5:Ts3.5");
  DrawSame("Tn:NTs","Tc==5.0 && Ts==3.5",3,24,legend,"Tc5.0:Ts3.5");
  DrawSame("Tn:NTs","Tc==5.5 && Ts==3.5",4,24,legend,"Tc5.5:Ts3.5");
  DrawSame("Tn:NTs","Tc==4.5 && Ts==3.75",2,25,legend,"Tc4.5:Ts3.75");
  DrawSame("Tn:NTs","Tc==5.0 && Ts==3.75",3,25,legend,"Tc5.0:Ts3.75");
  DrawSame("Tn:NTs","Tc==5.5 && Ts==3.75",4,25,legend,"Tc5.5:Ts3.75");

  C2->cd(3);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC2,"Tn:MeanWb","Tc==4.5 && Ts==2.5",2,20,"TOB L4, Mean Width for BG clusters","Tn","Width (# strips)",legend,"Tc4.5:Ts2.5",2.0,3.2);
  DrawSame("Tn:MeanWb","Tc==5.0 && Ts==2.5",3,20,legend,"Tc5.0:Ts2.5");
  DrawSame("Tn:MeanWb","Tc==5.5 && Ts==2.5",4,20,legend,"Tc5.5:Ts2.5");
  DrawSame("Tn:MeanWb","Tc==4.5 && Ts==3.75",2,21,legend,"Tc4.5:Ts2.75");
  DrawSame("Tn:MeanWb","Tc==5.0 && Ts==3.75",3,21,legend,"Tc5.0:Ts2.75");
  DrawSame("Tn:MeanWb","Tc==5.5 && Ts==3.75",4,21,legend,"Tc5.5:Ts2.75");
  DrawSame("Tn:MeanWb","Tc==4.5 && Ts==3.0",2,22,legend,"Tc4.5:Ts3.0");
  DrawSame("Tn:MeanWb","Tc==5.0 && Ts==3.0",3,22,legend,"Tc5.0:Ts3.0");
  DrawSame("Tn:MeanWb","Tc==5.5 && Ts==3.0",4,22,legend,"Tc5.5:Ts3.0");
  DrawSame("Tn:MeanWb","Tc==4.5 && Ts==3.25",2,23,legend,"Tc4.5:Ts3.25");
  DrawSame("Tn:MeanWb","Tc==5.0 && Ts==3.25",3,23,legend,"Tc5.0:Ts3.25");
  DrawSame("Tn:MeanWb","Tc==5.5 && Ts==3.25",4,23,legend,"Tc5.5:Ts3.25");
  DrawSame("Tn:MeanWb","Tc==4.5 && Ts==3.5",2,24,legend,"Tc4.5:Ts3.5");
  DrawSame("Tn:MeanWb","Tc==5.0 && Ts==3.5",3,24,legend,"Tc5.0:Ts3.5");
  DrawSame("Tn:MeanWb","Tc==5.5 && Ts==3.5",4,24,legend,"Tc5.5:Ts3.5");
  DrawSame("Tn:MeanWb","Tc==4.5 && Ts==3.75",2,25,legend,"Tc4.5:Ts3.75");
  DrawSame("Tn:MeanWb","Tc==5.0 && Ts==3.75",3,25,legend,"Tc5.0:Ts3.75");
  DrawSame("Tn:MeanWb","Tc==5.5 && Ts==3.75",4,25,legend,"Tc5.5:Ts3.75");

  C2->cd(4);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC2,"Tn:MeanWs","Tc==4.5 && Ts==2.5",2,20,"TOB L4,Mean Width for Signal clusters","Tn","Width (# strips)",legend,"Tc4.5:Ts2.5",3.7,4.5);
  DrawSame("Tn:MeanWs","Tc==5.0 && Ts==2.5",3,20,legend,"Tc5.0:Ts2.5");
  DrawSame("Tn:MeanWs","Tc==5.5 && Ts==2.5",4,20,legend,"Tc5.5:Ts2.5");
  DrawSame("Tn:MeanWs","Tc==4.5 && Ts==3.75",2,21,legend,"Tc4.5:Ts2.75");
  DrawSame("Tn:MeanWs","Tc==5.0 && Ts==3.75",3,21,legend,"Tc5.0:Ts2.75");
  DrawSame("Tn:MeanWs","Tc==5.5 && Ts==3.75",4,21,legend,"Tc5.5:Ts2.75");
  DrawSame("Tn:MeanWs","Tc==4.5 && Ts==3.0",2,22,legend,"Tc4.5:Ts3.0");
  DrawSame("Tn:MeanWs","Tc==5.0 && Ts==3.0",3,22,legend,"Tc5.0:Ts3.0");
  DrawSame("Tn:MeanWs","Tc==5.5 && Ts==3.0",4,22,legend,"Tc5.5:Ts3.0");
  DrawSame("Tn:MeanWs","Tc==4.5 && Ts==3.25",2,23,legend,"Tc4.5:Ts3.25");
  DrawSame("Tn:MeanWs","Tc==5.0 && Ts==3.25",3,23,legend,"Tc5.0:Ts3.25");
  DrawSame("Tn:MeanWs","Tc==5.5 && Ts==3.25",4,23,legend,"Tc5.5:Ts3.25");
  DrawSame("Tn:MeanWs","Tc==4.5 && Ts==3.5",2,24,legend,"Tc4.5:Ts3.5");
  DrawSame("Tn:MeanWs","Tc==5.0 && Ts==3.5",3,24,legend,"Tc5.0:Ts3.5");
  DrawSame("Tn:MeanWs","Tc==5.5 && Ts==3.5",4,24,legend,"Tc5.5:Ts3.5");
  DrawSame("Tn:MeanWs","Tc==4.5 && Ts==3.75",2,25,legend,"Tc4.5:Ts3.75");
  DrawSame("Tn:MeanWs","Tc==5.0 && Ts==3.75",3,25,legend,"Tc5.0:Ts3.75");
  DrawSame("Tn:MeanWs","Tc==5.5 && Ts==3.75",4,25,legend,"Tc5.5:Ts3.75");

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
