#include "Settings.h"
void DrawL2(int etabin)
{
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0000);
  gStyle->SetOptFit(000); 
  gStyle->SetPalette(1);
  if (etabin<0 || etabin>=NETA)
    {
      cout<<"Eta bin must be >=0 and <"<<NETA<<endl;
      break;
    }
  TPaveText *pave = new TPaveText(0.3,0.7,0.5,0.85,"NDC");  
  pave->AddText(Version);
  pave->AddText(Algorithm);
  pave->SetLineColor(0);
  pave->SetBorderSize(0);
  pave->SetFillColor(0);
  pave->SetBorderSize(0);
  TGraphErrors *g_EtaCorrection;
  TGraph *g_L2Correction;
  TCanvas *c_Resp;
  TCanvas *c_L2Cor;
  TF1 *L2Fit;
  TF1 *CorFit;
  TFile *rel_f;
  char filename[100],name[100];
  rel_f = new TFile(L2OutputROOTFilename,"r");
  if (!rel_f->IsOpen()) break;
  /////////////////////////////// Correction /////////////////////////
  sprintf(name,"EtaCorrection");
  c_Cor = new TCanvas(name,name,900,700);
  sprintf(name,"Correction_EtaBin%d",etabin);
  g_EtaCorrection = (TGraphErrors*)rel_f->Get(name);
  sprintf(name,"Correction%d",etabin);      
  CorFit = (TF1*)g_EtaCorrection->GetFunction(name); 
  if (CorFit->GetXmax()>200) 
    gPad->SetLogx();
  CorFit->SetLineColor(2);
  g_EtaCorrection->GetXaxis()->SetTitle("Uncorrected jet p_{T} (GeV)");
  g_EtaCorrection->GetYaxis()->SetTitle("Absolute Correction"); 
  sprintf(name,"%1.3f<#eta<%1.3f",eta_boundaries[etabin],eta_boundaries[etabin+1]);
  g_EtaCorrection->SetTitle(name); 
  g_EtaCorrection->SetMarkerStyle(20);
  g_EtaCorrection->Draw("AP"); 
  pave->Draw();
  
  /////////////////////////////// L2 correction ///////////////////////// 
  sprintf(name,"L2Correction");
  c_L2Cor = new TCanvas(name,name,900,700);
  sprintf(name,"L2Correction_EtaBin%d",etabin);
  g_L2Correction = (TGraph*)rel_f->Get(name);
  sprintf(name,"L2Correction%d",etabin);      
  L2Fit = (TF1*)g_L2Correction->GetFunction(name);
  if (L2Fit->GetXmax()>200) 
    gPad->SetLogx(); 
  g_L2Correction->SetMinimum(0.3);
  g_L2Correction->SetMaximum(1.4);
  g_L2Correction->GetXaxis()->SetTitle("Uncorrected jet p_{T} (GeV)");
  g_L2Correction->GetYaxis()->SetTitle("Relative Correction");  
  sprintf(name,"%1.3f<#eta<%1.3f",eta_boundaries[etabin],eta_boundaries[etabin+1]);
  g_L2Correction->SetTitle(name); 
  g_L2Correction->Draw("AP");
  g_L2Correction->SetMarkerStyle(20);
  L2Fit->SetLineColor(2);
  pave->Draw();
}


