#include "Settings.h"
void DrawResponseVsEta(int ptbin, char s1[1024])
{
  char filename[2][1024];
  sprintf(filename[0],"%s",s1);
  MainProgram(ptbin,1,filename);
}
/////////////////////////////////////////////////////////////////////////////////
void DrawResponseVsEta(int ptbin, char s1[1024],char s2[1024])
{
  char filename[2][1024];
  sprintf(filename[0],"%s",s1);
  sprintf(filename[1],"%s",s2);
  MainProgram(ptbin,2,filename);
}
/////////////////////////////////////////////////////////////////////////////////
void MainProgram(int ptbin, const int N, char filename[][1024])
{
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0000);
  gStyle->SetOptFit(111); 
  gStyle->SetPalette(1);
  TFile *f[N];
  TH1F *hResponse[N];
  int i;
  char name[100];
  sprintf(name,"Response_vs_Eta_RefPt%d",ptbin);
  for(i=0;i<N;i++)
    {
      f[i] = new TFile(filename[i],"r");
      if (f[i]->IsZombie()) break; 
      hResponse[i] = (TH1F*)f[i]->Get(name);
    }
  /////////////////////////////////////////////////////////////////////////////////
  TCanvas *can = new TCanvas("CanResponse","CanResponse",900,600);
  sprintf(name,"%d < RefPt < %d GeV",(int)Pt[ptbin],(int)Pt[ptbin+1]);
  hResponse[0]->SetTitle(name);
  hResponse[0]->GetXaxis()->SetTitle("#eta");
  hResponse[0]->GetYaxis()->SetTitle("Absolute Response");
  hResponse[0]->GetYaxis()->SetNdivisions(505);
  TLegend *leg = new TLegend(0.5,0.6,0.85,0.85);
  for(i=0;i<N;i++)
    {
      hResponse[i]->SetMarkerStyle(20+i);
      hResponse[i]->SetMarkerColor(i+1);
      hResponse[i]->SetLineColor(i+1); 
      hResponse[i]->Draw("same");
      leg->AddEntry(hResponse[i],filename[i],"LP");
    }
  leg->SetFillColor(0);
  leg->SetLineColor(0);
  leg->Draw();  
}
