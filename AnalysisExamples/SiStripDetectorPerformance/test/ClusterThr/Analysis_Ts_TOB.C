#include<Riostream>	       
#include "TCanvas.h"
#include "TObject.h"

using namespace std;
void DrawClus(TCanvas *c,char* varToPlot, char* cond, Int_t kColor, Int_t kMarker, char* Title,char* xTitle,char* yTitle, TLegend *leg, char* cLeg,Double_t downlim,Double_t uplim);
void DrawSame(char* varToPlot,char* cond,Int_t kColor, Int_t kMarker,TLegend* leg,char* cLeg);

void PlotMacro(){
  TFile *out = new TFile("/tmp/maborgia/out_Ts_TOBL4.root","recreate");  
  TFile *f = new TFile("/tmp/maborgia/pass4_8055_ClusThr_TOB_L4.root");
  TObjArray DetList(0);

  f->cd("AsciiOutput");

  TCanvas *C1=new TCanvas("cC1","cC1",10,10,900,900); 
  DetList.Add(cC1);
  C1->Divide(2,2);

  C1->cd(1);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC1,"Ts:Nb","Tc==4.5 && Tn==1.5",2,20,"TOB L4,Mean number of BG cluster per event","Ts","Mean BG clusters",legend,"Tc4.5:Tn1.5",0.15,0.4);
  DrawSame("Ts:Nb","Tc==5.0 && Tn==1.5",3,20,legend,"Tc5.0:Tn1.5");
  DrawSame("Ts:Nb","Tc==5.5 && Tn==1.5",4,20,legend,"Tc5.5:Tn1.5");
  DrawSame("Ts:Nb","Tc==4.5 && Tn==1.75",2,21,legend,"Tc4.5:Tn1.75");
  DrawSame("Ts:Nb","Tc==5.0 && Tn==1.75",3,21,legend,"Tc5.0:Tn1.75");
  DrawSame("Ts:Nb","Tc==5.5 && Tn==1.75",4,21,legend,"Tc5.5:Tn1.75");
  DrawSame("Ts:Nb","Tc==4.5 && Tn==2.0",2,22,legend,"Tc4.5:Tn2.0");
  DrawSame("Ts:Nb","Tc==5.0 && Tn==2.0",3,22,legend,"Tc5.0:Tn2.0");
  DrawSame("Ts:Nb","Tc==5.5 && Tn==2.0",4,22,legend,"Tc5.5:Tn2.0");
  DrawSame("Ts:Nb","Tc==4.5 && Tn==2.25",2,23,legend,"Tc4.5:Tn2.25");
  DrawSame("Ts:Nb","Tc==5.0 && Tn==2.25",3,23,legend,"Tc5.0:Tn2.25");
  DrawSame("Ts:Nb","Tc==5.5 && Tn==2.25",4,23,legend,"Tc5.5:Tn2.25");

  C1->cd(2);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC1,"Ts:Ns","Tc==4.5 && Tn==1.5",2,20,"TOB L4,Mean number of Signal cluster per event","Ts","Mean Signal clusters",legend,"Tc4.5:Tn1.5",1.36,1.52);
  DrawSame("Ts:Ns","Tc==5.0 && Tn==1.5",3,20,legend,"Tc5.0:Tn1.5");
  DrawSame("Ts:Ns","Tc==5.5 && Tn==1.5",4,20,legend,"Tc5.5:Tn1.5");
  DrawSame("Ts:Ns","Tc==4.5 && Tn==1.75",2,21,legend,"Tc4.5:Tn1.75");
  DrawSame("Ts:Ns","Tc==5.0 && Tn==1.75",3,21,legend,"Tc5.0:Tn1.75");
  DrawSame("Ts:Ns","Tc==5.5 && Tn==1.75",4,21,legend,"Tc5.5:Tn1.75");
  DrawSame("Ts:Ns","Tc==4.5 && Tn==2.0",2,22,legend,"Tc4.5:Tn2.0");
  DrawSame("Ts:Ns","Tc==5.0 && Tn==2.0",3,22,legend,"Tc5.0:Tn2.0");
  DrawSame("Ts:Ns","Tc==5.5 && Tn==2.0",4,22,legend,"Tc5.5:Tn2.0");
  DrawSame("Ts:Ns","Tc==4.5 && Tn==2.25",2,23,legend,"Tc4.5:Tn2.25");
  DrawSame("Ts:Ns","Tc==5.0 && Tn==2.25",3,23,legend,"Tc5.0:Tn2.25");
  DrawSame("Ts:Ns","Tc==5.5 && Tn==2.25",4,23,legend,"Tc5.5:Tn2.25");

  C1->cd(3);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC1,"Ts:MPVs","Tc==4.5 && Tn==1.5",2,20,"TOB L4,MPV of S/N","Ts","MPV",legend,"Tc4.5:Tn1.5",36,39);
  DrawSame("Ts:MPVs","Tc==5.0 && Tn==1.5",3,20,legend,"Tc5.0:Tn1.5");
  DrawSame("Ts:MPVs","Tc==5.5 && Tn==1.5",4,20,legend,"Tc5.5:Tn1.5");
  DrawSame("Ts:MPVs","Tc==4.5 && Tn==1.75",2,21,legend,"Tc4.5:Tn1.75");
  DrawSame("Ts:MPVs","Tc==5.0 && Tn==1.75",3,21,legend,"Tc5.0:Tn1.75");
  DrawSame("Ts:MPVs","Tc==5.5 && Tn==1.75",4,21,legend,"Tc5.5:Tn1.75");
  DrawSame("Ts:MPVs","Tc==4.5 && Tn==2.0",2,22,legend,"Tc4.5:Tn2.0");
  DrawSame("Ts:MPVs","Tc==5.0 && Tn==2.0",3,22,legend,"Tc5.0:Tn2.0");
  DrawSame("Ts:MPVs","Tc==5.5 && Tn==2.0",4,22,legend,"Tc5.5:Tn2.0");
  DrawSame("Ts:MPVs","Tc==4.5 && Tn==2.25",2,23,legend,"Tc4.5:Tn2.25");
  DrawSame("Ts:MPVs","Tc==5.0 && Tn==2.25",3,23,legend,"Tc5.0:Tn2.25");
  DrawSame("Ts:MPVs","Tc==5.5 && Tn==2.25",4,23,legend,"Tc5.5:Tn2.25");

  C1->cd(4);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC1,"Ts:FWHMs","Tc==4.5 && Tn==1.5",2,20,"TOB L4,FWHM of S/N","Ts","FWHM",legend,"Tc4.5:Tn1.5",5.4,6.2);
  DrawSame("Ts:FWHMs","Tc==5.0 && Tn==1.5",3,20,legend,"Tc5.0:Tn1.5");
  DrawSame("Ts:FWHMs","Tc==5.5 && Tn==1.5",4,20,legend,"Tc5.5:Tn1.5");
  DrawSame("Ts:FWHMs","Tc==4.5 && Tn==1.75",2,21,legend,"Tc4.5:Tn1.75");
  DrawSame("Ts:FWHMs","Tc==5.0 && Tn==1.75",3,21,legend,"Tc5.0:Tn1.75");
  DrawSame("Ts:FWHMs","Tc==5.5 && Tn==1.75",4,21,legend,"Tc5.5:Tn1.75");
  DrawSame("Ts:FWHMs","Tc==4.5 && Tn==2.0",2,22,legend,"Tc4.5:Tn2.0");
  DrawSame("Ts:FWHMs","Tc==5.0 && Tn==2.0",3,22,legend,"Tc5.0:Tn2.0");
  DrawSame("Ts:FWHMs","Tc==5.5 && Tn==2.0",4,22,legend,"Tc5.5:Tn2.0");
  DrawSame("Ts:FWHMs","Tc==4.5 && Tn==2.25",2,23,legend,"Tc4.5:Tn2.25");
  DrawSame("Ts:FWHMs","Tc==5.0 && Tn==2.25",3,23,legend,"Tc5.0:Tn2.25");
  DrawSame("Ts:FWHMs","Tc==5.5 && Tn==2.25",4,23,legend,"Tc5.5:Tn2.25");

  TCanvas *C2=new TCanvas("cC2","cC2",10,10,900,900); 
  DetList.Add(cC2);
  C2->Divide(2,2);

  C2->cd(1);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC2,"Ts:NTb","Tc==4.5 && Tn==1.5",2,20,"TOB L4,Total number of BG cluster","Ts","# clusters",legend,"Tc4.5:Tn1.5",1500,3700);
  DrawSame("Ts:NTb","Tc==5.0 && Tn==1.5",3,20,legend,"Tc5.0:Tn1.5");
  DrawSame("Ts:NTb","Tc==5.5 && Tn==1.5",4,20,legend,"Tc5.5:Tn1.5");
  DrawSame("Ts:NTb","Tc==4.5 && Tn==1.75",2,21,legend,"Tc4.5:Tn1.75");
  DrawSame("Ts:NTb","Tc==5.0 && Tn==1.75",3,21,legend,"Tc5.0:Tn1.75");
  DrawSame("Ts:NTb","Tc==5.5 && Tn==1.75",4,21,legend,"Tc5.5:Tn1.75");
  DrawSame("Ts:NTb","Tc==4.5 && Tn==2.0",2,22,legend,"Tc4.5:Tn2.0");
  DrawSame("Ts:NTb","Tc==5.0 && Tn==2.0",3,22,legend,"Tc5.0:Tn2.0");
  DrawSame("Ts:NTb","Tc==5.5 && Tn==2.0",4,22,legend,"Tc5.5:Tn2.0");
  DrawSame("Ts:NTb","Tc==4.5 && Tn==2.25",2,23,legend,"Tc4.5:Tn2.25");
  DrawSame("Ts:NTb","Tc==5.0 && Tn==2.25",3,23,legend,"Tc5.0:Tn2.25");
  DrawSame("Ts:NTb","Tc==5.5 && Tn==2.25",4,23,legend,"Tc5.5:Tn2.25");

  C2->cd(2);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC2,"Ts:NTs","Tc==4.5 && Tn==1.5",2,20,"TOB L4,Total number of Signal cluster","Ts","# clusters",legend,"Tc4.5:Tn1.5",17900,19500);
  DrawSame("Ts:NTs","Tc==5.0 && Tn==1.5",3,20,legend,"Tc5.0:Tn1.5");
  DrawSame("Ts:NTs","Tc==5.5 && Tn==1.5",4,20,legend,"Tc5.5:Tn1.5");
  DrawSame("Ts:NTs","Tc==4.5 && Tn==1.75",2,21,legend,"Tc4.5:Tn1.75");
  DrawSame("Ts:NTs","Tc==5.0 && Tn==1.75",3,21,legend,"Tc5.0:Tn1.75");
  DrawSame("Ts:NTs","Tc==5.5 && Tn==1.75",4,21,legend,"Tc5.5:Tn1.75");
  DrawSame("Ts:NTs","Tc==4.5 && Tn==2.0",2,22,legend,"Tc4.5:Tn2.0");
  DrawSame("Ts:NTs","Tc==5.0 && Tn==2.0",3,22,legend,"Tc5.0:Tn2.0");
  DrawSame("Ts:NTs","Tc==5.5 && Tn==2.0",4,22,legend,"Tc5.5:Tn2.0");
  DrawSame("Ts:NTs","Tc==4.5 && Tn==2.25",2,23,legend,"Tc4.5:Tn2.25");
  DrawSame("Ts:NTs","Tc==5.0 && Tn==2.25",3,23,legend,"Tc5.0:Tn2.25");
  DrawSame("Ts:NTs","Tc==5.5 && Tn==2.25",4,23,legend,"Tc5.5:Tn2.25");

  C2->cd(3);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC2,"Ts:MeanWb","Tc==4.5 && Tn==1.5",2,20,"TOB L4, Mean Width for BG clusters","Ts","Width (# strips)",legend,"Tc4.5:Tn1.5",2.0,3.2);
  DrawSame("Ts:MeanWb","Tc==5.0 && Tn==1.5",3,20,legend,"Tc5.0:Tn1.5");
  DrawSame("Ts:MeanWb","Tc==5.5 && Tn==1.5",4,20,legend,"Tc5.5:Tn1.5");
  DrawSame("Ts:MeanWb","Tc==4.5 && Tn==1.75",2,21,legend,"Tc4.5:Tn1.75");
  DrawSame("Ts:MeanWb","Tc==5.0 && Tn==1.75",3,21,legend,"Tc5.0:Tn1.75");
  DrawSame("Ts:MeanWb","Tc==5.5 && Tn==1.75",4,21,legend,"Tc5.5:Tn1.75");
  DrawSame("Ts:MeanWb","Tc==4.5 && Tn==2.0",2,22,legend,"Tc4.5:Tn2.0");
  DrawSame("Ts:MeanWb","Tc==5.0 && Tn==2.0",3,22,legend,"Tc5.0:Tn2.0");
  DrawSame("Ts:MeanWb","Tc==5.5 && Tn==2.0",4,22,legend,"Tc5.5:Tn2.0");
  DrawSame("Ts:MeanWb","Tc==4.5 && Tn==2.25",2,23,legend,"Tc4.5:Tn2.25");
  DrawSame("Ts:MeanWb","Tc==5.0 && Tn==2.25",3,23,legend,"Tc5.0:Tn2.25");
  DrawSame("Ts:MeanWb","Tc==5.5 && Tn==2.25",4,23,legend,"Tc5.5:Tn2.25");

  C2->cd(4);
  TLegend *legend = new TLegend(0.851008,0.437882,0.995954,1.00035); 
  DrawClus(cC2,"Ts:MeanWs","Tc==4.5 && Tn==1.5",2,20,"TOB L4,Mean Width for Signal clusters","Ts","Width (# strips)",legend,"Tc4.5:Tn1.5",3.7,4.5);
  DrawSame("Ts:MeanWs","Tc==5.0 && Tn==1.5",3,20,legend,"Tc5.0:Tn1.5");
  DrawSame("Ts:MeanWs","Tc==5.5 && Tn==1.5",4,20,legend,"Tc5.5:Tn1.5");
  DrawSame("Ts:MeanWs","Tc==4.5 && Tn==1.75",2,21,legend,"Tc4.5:Tn1.75");
  DrawSame("Ts:MeanWs","Tc==5.0 && Tn==1.75",3,21,legend,"Tc5.0:Tn1.75");
  DrawSame("Ts:MeanWs","Tc==5.5 && Tn==1.75",4,21,legend,"Tc5.5:Tn1.75");
  DrawSame("Ts:MeanWs","Tc==4.5 && Tn==2.0",2,22,legend,"Tc4.5:Tn2.0");
  DrawSame("Ts:MeanWs","Tc==5.0 && Tn==2.0",3,22,legend,"Tc5.0:Tn2.0");
  DrawSame("Ts:MeanWs","Tc==5.5 && Tn==2.0",4,22,legend,"Tc5.5:Tn2.0");
  DrawSame("Ts:MeanWs","Tc==4.5 && Tn==2.25",2,23,legend,"Tc4.5:Tn2.25");
  DrawSame("Ts:MeanWs","Tc==5.0 && Tn==2.25",3,23,legend,"Tc5.0:Tn2.25");
  DrawSame("Ts:MeanWs","Tc==5.5 && Tn==2.25",4,23,legend,"Tc5.5:Tn2.25");

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
