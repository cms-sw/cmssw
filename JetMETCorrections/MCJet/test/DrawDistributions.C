#include "Settings.h"
void DrawDistributions()
{
  PrintMessage();
}
void DrawDistributions(TString type, int ptbin)
{
  MainProgram(type,ptbin,0);
}
void DrawDistributions(TString type, int ptbin, int etabin)
{
  MainProgram(type,ptbin,etabin);
}

void MainProgram(TString type, int ptbin, int etabin)
{
  if (ptbin>NPtBins || etabin>NETA)
    {
      cout<<"Choose ptbin < "<<NPtBins<<" and etabin < "<<NETA<<endl;
      break;
    } 
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0000);
  gStyle->SetOptFit(000); 
  gStyle->SetPalette(1);
  TGaxis::SetMaxDigits(3);
  TString fileName,title;
  char responseName[1024],histoName[1024],text[1024];
  TPaveText *GenTitle,*CaloTitle,*ResponseTitle;
  TString Xtitle;
  TH1F *hResponse,*hGen,*hCalo;
  TFile *f;
  GenTitle = new TPaveText(0.65,0.7,0.89,0.85,"NDC");
  GenTitle->SetFillColor(0);
  GenTitle->SetLineColor(0);
  GenTitle->SetBorderSize(0);
  GenTitle->AddText("Particle Jets");
  ResponseTitle = new TPaveText(0.65,0.7,0.89,0.85,"NDC");
  ResponseTitle->SetFillColor(0);
  ResponseTitle->SetLineColor(0);
  ResponseTitle->SetBorderSize(0);
  CaloTitle = new TPaveText(0.65,0.7,0.89,0.85,"NDC");
  CaloTitle->SetFillColor(0);
  CaloTitle->SetLineColor(0);
  CaloTitle->SetBorderSize(0);
  TPaveText *pave = new TPaveText(0.11,0.11,0.89,0.89,"NDC");
  pave->SetFillColor(0);
  pave->SetLineColor(0);
  pave->SetBorderSize(0);
  pave->AddText(ALGO(Algorithm)); 
  sprintf(text,"%d < RefP_{T} < %d GeV",Pt[ptbin],Pt[ptbin+1]);
  pave->AddText(text);
  sprintf(text,"%1.3f < #eta < %1.3f",eta_boundaries[etabin],eta_boundaries[etabin+1]);
  pave->AddText(text);
  /////////////////////////////////////////////////////////////
  if (type=="Raw")
    {   
      if (!UseRatioForResponse)
        {
          title = "CaloJetP_{T}-RefP_{T} (GeV)";
          ResponseTitle->AddText("#DeltaP_{T}");
        }
      else
        {
          title = "CaloJetP_{T}/RefP_{T}";
          ResponseTitle->AddText("CaloJetP_{T}/RefP_{T}");
        }
      CaloTitle->AddText("Uncorrected Jets");
      Xtitle = "CaloJetP_{T} (GeV)";
    }
  else if (type=="Corrected")
    {
      title = "CorJetP_{T}/RefP_{T}";
      ResponseTitle->AddText("Response");
      CaloTitle->AddText("Corrected Jets");
      Xtitle = "CorJetP_{T} (GeV)";
    }
  else
    {
      cout<<"Wrong type!!! Choose \"Raw\" or \"Corrected\""<<endl;
      break;
    }
  ////////////////////////////////////////////////////////////
  f = new TFile(HistoFilename,"R");
  if (f->IsZombie()) break;
  if (NETA>1)
    {   
      sprintf(histoName,"Response_RefPt%d_Eta%d",ptbin,etabin); 
      hResponse = (TH1F*)f->Get(histoName);
      sprintf(histoName,"ptRef_RefPt%d_Eta%d",ptbin,etabin);
      hGen = (TH1F*)f->Get(histoName);
      sprintf(histoName,"ptCalo_RefPt%d_Eta%d",ptbin,etabin);
      hCalo = (TH1F*)f->Get(histoName);
    }
  else
    {    
      sprintf(histoName,"Response_RefPt%d",ptbin); 
      hResponse = (TH1F*)f->Get(histoName);
      sprintf(histoName,"ptRef_RefPt%d",ptbin);
      hGen = (TH1F*)f->Get(histoName);
      sprintf(histoName,"ptCalo_RefPt%d",ptbin);
      hCalo = (TH1F*)f->Get(histoName);
    }
  if (type=="Raw")
    Zoom(hResponse,7.);
  Zoom(hGen,7.);
  Zoom(hCalo,7.);
  ///////////////////////////////////////
  TString cname;
  char aux[100];
  if (etabin>=0)
    sprintf(aux,"_Distributions_%d_%d",ptbin,etabin);
  else
    sprintf(aux,"_Distributions_%d",ptbin);
  cname = type+aux;
  TCanvas *c1 = new TCanvas(cname,cname,900,600);
  c1->Divide(2,2);
  c1->cd(2);
  hResponse->GetXaxis()->SetTitle(title);
  hResponse->GetYaxis()->SetTitle("Matched jets");
  hResponse->SetTitle("");
  hResponse->Draw();
  ResponseTitle->Draw();
  c1->cd(3);
  hCalo->GetXaxis()->SetTitle(Xtitle);
  hCalo->GetYaxis()->SetTitle("Matched jets");
  hCalo->SetTitle("");
  hCalo->Draw();
  CaloTitle->Draw();
  c1->cd(1);
  hGen->GetXaxis()->SetTitle("RefP_{T} (GeV)");
  hGen->GetYaxis()->SetTitle("Matched jets");
  hGen->SetTitle("");
  hGen->Draw();
  GenTitle->Draw();
  c1->cd(4);
  pave->Draw();
}

void Zoom(TH1F* h,double a)
{
  double m,rms;
  m = h->GetMean();
  rms = h->GetRMS();
  h->SetAxisRange(m-a*rms,m+a*rms,"X");
}

TString ALGO(TString algorithm)
{
  TString tmp;
  if (algorithm=="Icone5")
    tmp = "Iterative Cone R = 0.5";
  else if (algorithm=="Scone5")
    tmp = "SIS Cone R = 0.5";
  else if (algorithm=="Scone7")
    tmp = "SIS Cone R = 0.7";
  else if (algorithm=="Kt4")
    tmp = "KT D = 0.4";
  else if (algorithm=="Kt6")
    tmp = "KT D = 0.6";
  else
    tmp = "Unknown algorithm!!!";
  return tmp;
}

void PrintMessage()
{
  cout<<"This ROOT macro draws the basic distributions: RefPt, CaloPt, Response."<<endl;
  cout<<"Usage: .X DrawDistributions.C(\"type\",ptbin,etabin) or .X DrawDistributions.C(\"type\",ptbin)"<<endl;
}
