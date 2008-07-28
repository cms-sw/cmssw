#include "Settings.h"
void DrawResponse()
{
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0000);
  gStyle->SetOptFit(111); 
  gStyle->SetPalette(1);
  TFile *f;
  TH1F *hResponse,*hMeanRefPt,*hMeanCaloPt;
  double yRefPt[NPtBins],eyRefPt[NPtBins],xRefPt[NPtBins],exRefPt[NPtBins];
  double yCaloPt[NPtBins],eyCaloPt[NPtBins],xCaloPt[NPtBins],exCaloPt[NPtBins];
  double x,y,ex,ey,x1,ex1;
  int i,N;
  f = new TFile(FitterFilename,"r");
  if (f->IsZombie()) break; 
  hResponse = (TH1F*)f->Get("Response");
  hMeanRefPt = (TH1F*)f->Get("MeanRefPt");
  hMeanCaloPt = (TH1F*)f->Get("MeanCaloPt");
  N = 0;
  for(i=0;i<NPtBins;i++)
    {
      y = hResponse->GetBinContent(i+1);
      ey = hResponse->GetBinError(i+1);
      x = hMeanRefPt->GetBinContent(i+1);
      ex = hMeanRefPt->GetBinError(i+1);
      x1 = hMeanCaloPt->GetBinContent(i+1);
      ex1 = hMeanCaloPt->GetBinError(i+1); 
      if (y>0 && x>0 && x1>0 && ey>0.000001 && ey<0.2)
	{
          yRefPt[N] = y;
          eyRefPt[N] = ey;
          xRefPt[N] = x;
          exRefPt[N] = ex;
          xCaloPt[N] = x1;
          exCaloPt[N] = ex1;
	  N++;
	}  
    }
  TGraphErrors *gRespRefPt = new TGraphErrors(N,xRefPt,yRefPt,exRefPt,eyRefPt);
  TGraphErrors *gCorrelation = new TGraphErrors(N,xRefPt,xCaloPt,exRefPt,exCaloPt); 
  TF1 *func = new TF1("ideal","x",1,2000);
  /////////////////////////////////////////////////////////////////////////////////
  TCanvas *can = new TCanvas("CanResponse","CanResponse",900,600);
  gPad->SetLogx();
  gPad->SetGridy();
  gRespRefPt->SetTitle("");
  gRespRefPt->GetXaxis()->SetTitle("RefP_{T} (GeV)");
  gRespRefPt->GetYaxis()->SetTitle("Response");
  gRespRefPt->GetYaxis()->SetNdivisions(505);
  gRespRefPt->SetMarkerStyle(20);
  gRespRefPt->Draw("AP");
  /////////////////////////////////////////////////////////////////////////////////
  TCanvas *can1 = new TCanvas("Correlation","Correlation",900,600);
  gPad->SetLogx();
  gPad->SetLogy();
  gCorrelation->SetTitle("");
  gCorrelation->GetXaxis()->SetTitle("<RefP_{T}> (GeV)");
  gCorrelation->GetYaxis()->SetTitle("<CaloP_{T}> (GeV)");
  gCorrelation->GetYaxis()->SetNdivisions(505);
  gCorrelation->SetMarkerStyle(20);
  gCorrelation->Draw("AP");
  func->SetLineColor(2);
  func->Draw("same");
  TLegend *leg = new TLegend(0.65,0.15,0.85,0.35);
  leg->AddEntry(gCorrelation,"measurement","P");
  leg->AddEntry(func,"<CaloP_{T}> = <RefP_{T}>","L"); 
  leg->SetLineColor(0);
  leg->SetFillColor(0);
  leg->Draw();    
}
