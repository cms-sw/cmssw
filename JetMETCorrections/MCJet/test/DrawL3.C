#include "Settings.h"
void DrawL3()
{
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0000);
  gStyle->SetOptFit(000); 
  gStyle->SetPalette(1);
 
  char name[100];
  int i;
  double x,y,e;
  TFile *inf = new TFile(L3OutputROOTFilename,"r");
  TGraphErrors *g_Cor, *g_Resp;
  TF1 *CorFit, *RespFit;
  TMatrixD *COV_Cor, *COV_Resp;
  TH1F *hCorUncertainty, *hRespUncertainty, *hCorFrac, *hRespFrac;
  TPaveText *pave = new TPaveText(0.3,0.7,0.5,0.85,"NDC");  
  pave->AddText(Version);
  pave->AddText(Algorithm);
  pave->SetLineColor(0);
  pave->SetBorderSize(0);
  pave->SetFillColor(0);
  pave->SetBorderSize(0);
  ///////////////////////////////////////////////////////////////
  g_Cor = (TGraphErrors*)inf->Get("Correction_vs_CaloPt");
  COV_Cor = (TMatrixD*)inf->Get("CovMatrix_Correction");
  CorFit = (TF1*)g_Cor->GetFunction("CorFit");
  CorFit->SetRange(1,5000);  
  g_Resp = (TGraphErrors*)inf->Get("Response_vs_RefPt");
  COV_Resp = (TMatrixD*)inf->Get("CovMatrix_Resp");
  RespFit = (TF1*)g_Resp->GetFunction("RespFit");
  RespFit->SetRange(5,5000);
  hCorUncertainty = new TH1F("CorrectionUncertainty","CorrectionUncertainty",5000,1,5001);
  hRespUncertainty = new TH1F("ResponseUncertainty","ResponseUncertainty",5000,5,5005);
  hCorFrac = new TH1F("FractionalCorrectionUncertainty","FractionalCorrectionUncertainty",5000,1,5001);
  hRespFrac = new TH1F("FractionalResponseUncertainty","FractionalResponseUncertainty",5000,5,5005);
  for(i=0;i<5000;i++)
    {
      x = hCorUncertainty->GetBinCenter(i+1);
      y = CorFit->Eval(x);
      e = FitUncertainty(false,CorFit,COV_Cor,x);
      hCorUncertainty->SetBinContent(i+1,y);
      hCorUncertainty->SetBinError(i+1,e);
      hCorFrac->SetBinContent(i+1,100*e/y);
      
      x = hRespUncertainty->GetBinCenter(i+1);
      y = RespFit->Eval(x);
      e = FitUncertainty(true,RespFit,COV_Resp,x);
      hRespUncertainty->SetBinContent(i+1,y);
      hRespUncertainty->SetBinError(i+1,e);
      hRespFrac->SetBinContent(i+1,100*e/y); 
    }
  ////////////////////// Correction ///////////////////////////////////////
  TCanvas *c_Correction = new TCanvas("Correction","Correction",900,600);
  c_Correction->cd(); 
  gPad->SetLogx();
  hCorUncertainty->SetTitle(""); 
  g_Cor->SetMarkerStyle(20);
  g_Cor->SetMarkerColor(1);
  g_Cor->SetLineColor(1);
  hCorUncertainty->SetMaximum(4);
  hCorUncertainty->SetMinimum(1);
  hCorUncertainty->GetXaxis()->SetTitle("p_{T} (GeV)");
  hCorUncertainty->GetYaxis()->SetTitle("L3Correction factor");
  CorFit->SetLineColor(2);
  CorFit->SetLineWidth(2);
  CorFit->SetParNames("a0","a1","a2","a3","a4");
  hCorUncertainty->SetLineColor(5);
  hCorUncertainty->SetFillColor(5);
  hCorUncertainty->SetMarkerColor(5);
  hCorUncertainty->Draw("E3");
  g_Cor->Draw("Psame"); 
  pave->Draw();
  TLegend *leg = new TLegend(0.6,0.65,0.89,0.89);
  leg->AddEntry(g_Cor,"measurement","LP");
  leg->AddEntry(CorFit,"fit","L");
  leg->AddEntry(hCorUncertainty,"fit uncertainty","F");
  leg->SetFillColor(0);
  leg->SetLineColor(0);
  leg->Draw();
  ////////////////////// Response ///////////////////////////////////////
  TCanvas *c_Response = new TCanvas("Response","Response",900,600);
  c_Response->cd(); 
  gPad->SetLogx();
  hRespUncertainty->SetTitle(""); 
  g_Resp->SetMarkerStyle(20);
  g_Resp->SetMarkerColor(1);
  g_Resp->SetLineColor(1);
  hRespUncertainty->SetMaximum(1);
  hRespUncertainty->SetMinimum(0);
  hRespUncertainty->GetXaxis()->SetTitle("p_{T}^{gen} (GeV)");
  hRespUncertainty->GetYaxis()->SetTitle("Response");
  RespFit->SetLineColor(2);
  hRespUncertainty->SetLineColor(5);
  hRespUncertainty->SetFillColor(5);
  hRespUncertainty->SetMarkerColor(5);
  hRespUncertainty->Draw("E3");
  g_Resp->Draw("Psame");
  pave->Draw();
  TLegend *leg = new TLegend(0.6,0.15,0.89,0.39);
  leg->AddEntry(g_Resp,"measurement","LP");
  leg->AddEntry(RespFit,"fit","L");
  leg->AddEntry(hRespUncertainty,"fit uncertainty","F");
  leg->SetFillColor(0);
  leg->SetLineColor(0);
  leg->Draw();  
  ////////////////////// Correction - Response closure ///////////////////////////////////////
  TH1F *hClosure = new TH1F("hClosure","hClosure",1000,5,5005);
  for(int i=0;i<1000;i++)
    {
      double dx = 5;
      double x = 5+dx*i;
      double y = Closure(x,CorFit,RespFit);
      hClosure->SetBinContent(i+1,y);
      hClosure->SetBinError(i+1,0.);
    }
  TCanvas *can = new TCanvas("Closure","Closure",900,600);
  gPad->SetLogx();
  hClosure->SetTitle("");
  hClosure->GetXaxis()->SetTitle("p_{T}");
  hClosure->GetYaxis()->SetTitle("C(p_{T})#times R(p_{T}C(p_{T}))");
  hClosure->Draw();
  ////////////////////// Fractional Correction Fit Uncertainty ///////////////////////////////////////
  TCanvas *can = new TCanvas("FracCorrUnc","FracCorrUnc",900,600);
  gPad->SetLogx();
  hCorFrac->SetTitle("");
  hCorFrac->GetXaxis()->SetTitle("p_{T} (GeV)");
  hCorFrac->GetYaxis()->SetTitle("Fractional Correction Fitting Uncertainty (%)");
  hCorFrac->Draw();
  ////////////////////// Fractional Response Fit Uncertainty ///////////////////////////////////////
  TCanvas *can = new TCanvas("FracRespUnc","FracRespUnc",900,600);
  gPad->SetLogx();
  hRespFrac->SetTitle("");
  hRespFrac->GetXaxis()->SetTitle("p_{T}^{gen} (GeV)");
  hRespFrac->GetYaxis()->SetTitle("Fractional Response Fit Uncertainty (%)");
  hRespFrac->Draw();
}
///////////////////////////////////////////////////////////////////////
double FitUncertainty(bool IsResponse, TF1* f, TMatrixD* COV, double x)
{
  int i,j,dim,N,npar;
  double df,sum,y,z,x;
  double PartialDerivative[10],Parameter[10];
  if (IsResponse)
    npar = 5;
  else
    npar = 4;  
  N = f->GetNumberFreeParameters();
  dim = COV->GetNrows();
  if (dim != npar || N != npar)
    {
      cout<<"ERROR: wrong number of parameters !!!!"<<endl;
      return(-1);
    }  
  for(i=0;i<npar;i++)
    Parameter[i] = f->GetParameter(i);
  z = pow(log10(x),Parameter[2]);  
  PartialDerivative[0] = 1.;
  PartialDerivative[1] = 1./(z+Parameter[3]);
  PartialDerivative[3] = -Parameter[1]/pow(z+Parameter[3],2);
  PartialDerivative[2] = PartialDerivative[3]*log(log10(x))*z;
  if (IsResponse)
    {
      PartialDerivative[1] = -1./(z+Parameter[3]);
      PartialDerivative[3] = Parameter[1]/pow(z+Parameter[3],2);
      PartialDerivative[4] = 1./x;
    }
  sum = 0.;
  for(i=0;i<npar;i++)
    for(j=0;j<npar;j++)
      {
        y = PartialDerivative[i]*PartialDerivative[j]*COV(i,j);
        sum+=y;
      }
  df = sqrt(sum);
  return df;
}
double Closure(double x, TF1 *f1, TF1 *f2)
{
  double y1,y,tmp;
  y1 = f1->Eval(x);
  y = x*y1;
  tmp = y1*f2->Eval(y);
  return tmp;
}


