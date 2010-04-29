#ifndef STACK_COMMON_H
#define STACK_COMMON_H

#include <iostream>

const string zmmName = "Analysis_zmm_2_4.root";     
const string wmnName = "Analysis_wmn_2_4.root";     
const string ttbarName = "Analysis_ttbar_2_4.root"; 
const string qcdName = "Analysis_qcd_2_4_all.root";  
const string dataName = "data133XXX.root";  

const int canvasSizeX = 500;
const int canvasSizeY = 500;

const Color_t zLineColor = kAzure+2;
const Color_t zFillColor = kAzure+6;
const Color_t wLineColor = kAzure+4;
const Color_t wFillColor = kAzure+2;
const Color_t qcdLineColor = kGreen+3;
const Color_t qcdFillColor = kGreen+1;
const Color_t ttFillColor =  kRed+1;
const Color_t ttLineColor = kRed+3;

const double lumi =.001 ;
const double lumiZ = 10. ;
const double lumiW = 10.;
const double lumiQ = 10.;
const double lumiT =10.;

TFile z(zmmName.c_str()) ; 
TFile w( wmnName.c_str()) ; 
TFile tt(ttbarName.c_str()) ; 
TFile qcd(qcdName.c_str()) ; 
TFile data(dataName.c_str()) ; 

TCanvas *c1 = new TCanvas("c1","Stack plot",10,10,canvasSizeX, canvasSizeY);


void setHisto(TH1 * h, Color_t fill, Color_t line, double scale, int rebin) {
  h->SetFillColor(fill);
  h->SetLineColor(line);
  h->Scale(scale);
  h->Rebin(rebin);  
}

void makeStack(TH1 * h1, TH1 * h2, TH1 * h3, TH1 * h4, TH1 * hdata, const char * yTag,
	       double min) {
  THStack * hs = new THStack("hs","");
  hs->Add(h4);
  hs->Add(h3);
  hs->Add(h2);
  hs->Add(h1);

  hs->Draw("HIST");
  if(hdata != 0) {
    hdata->SetMarkerStyle(20);
    hdata->SetMarkerSize(1.4);
    hdata->SetMarkerColor(kBlack);
    hdata->SetLineWidth(2);
    hdata->SetLineColor(kBlack);
    hdata->Draw("epsame"); 
    hdata->GetXaxis()->SetLabelSize(0);
    hdata->GetYaxis()->SetLabelSize(0);
    hs->SetMaximum( hdata->GetMaximum() + 1);
  }
  hs->SetMinimum(min);
  hs->GetXaxis()->SetTitle("m_{#mu #mu} (GeV/c^{2})");

  hs->GetYaxis()->SetTitle(yTag);
  hs->GetXaxis()->SetTitleSize(0.05);
  hs->GetYaxis()->SetTitleSize(0.05);
  hs->GetXaxis()->SetTitleOffset(1.0);
  hs->GetYaxis()->SetTitleOffset(1.3);
  hs->GetYaxis()->SetLabelOffset(0);
  hs->GetXaxis()->SetLabelSize(.05);
  hs->GetYaxis()->SetLabelSize(.05);
  leg = new TLegend(0.7,0.7,0.89,0.89);
  if(hdata != 0)
    leg->AddEntry(hdata,"data");
  leg->AddEntry(h1,"Z#rightarrow#mu #mu","f");
  leg->AddEntry(h2,"W#rightarrow#mu #nu","f");
  leg->AddEntry(h4,"QCD","f");
  leg->AddEntry(h3,"t#bar{t}","f"); 
  leg->SetFillColor(kWhite);
  leg->SetFillColor(kWhite);
  leg->SetShadowColor(kWhite);
  leg->Draw();
  c1->SetLogy();
}

void stat(TH1 * h1, TH1 * h2, TH1 * h3, TH1 * h4, TH1 * hdata) {
  double i1 = h1->Integral(60,120);
  double i2 = h2->Integral(60,120);
  double i3 = h3->Integral(60,120);
  double i4 = h4->Integral(60,120);
  double idata = hdata != 0 ? hdata->Integral(60,120) : 0;
  std::cout<<"zmm (60-120) = " << i1 <<std::endl;
  std::cout<<"w (60-120) = " << i2 <<std::endl;
  std::cout<<"QCD (60-120) = " << i4 <<std::endl;
  std::cout<<"ttbar (60-120) = " << i3 <<std::endl; 
}

void makePlots(const char * name, const char * tag, int rebin, const char * plot,
	       double min = 0.0001,
	       bool doData = false) {
  TH1F *h1 = (TH1F*)z.Get(name);
  setHisto(h1, zFillColor, zLineColor, lumi/lumiZ, rebin);

  TH1F *h2 = (TH1F*)w.Get(name);
  setHisto(h2, wFillColor, wLineColor, lumi/lumiW, rebin); 
  
  TH1F *h3 = (TH1F*)tt.Get(name);
  setHisto(h3, ttFillColor, ttLineColor, lumi/lumiT, rebin);
 
  TH1F *h4 = (TH1F*)qcd.Get(name);
  setHisto(h4, qcdFillColor, qcdLineColor, lumi/lumiQ, rebin);
 
  TH1F *hdata = doData? (TH1F*)data.Get(name) : 0;

  std::cout << "min = " << min << std::endl;
  makeStack(h1, h2, h3, h4, hdata, tag, min);

  stat(h1, h2, h3, h4, hdata);

  c1->SaveAs(plot);
}

#endif
