#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TFile.h"
#include <vector>
#include <iostream>

void fit2DProj(TString name, TString path, int minEntries, int rebin, int fitType);

Double_t gaussian(Double_t *x, Double_t *par);
Double_t lorentzian(Double_t *x, Double_t *par);
Double_t lorentzianPlusLinear(Double_t *x, Double_t *par);

TF1* gaussianFit(TH1* histoY);
TF1* lorentzianFit(TH1* histoY);
TF1* linLorentzianFit(TH1* histoY);


//
// Gaussian function    
// -----------------------
Double_t gaussian(Double_t *x, Double_t *par) {
  return par[0]*exp(-0.5*((x[0]-par[2])/par[1])*((x[0]-par[2])/par[1])) ; }
//
// Lorentzian function
// -----------------------
Double_t lorentzian(Double_t *x, Double_t *par) {
  return (0.5*par[0]*par[1]/TMath::Pi()) /
    TMath::Max( 1.e-10,(x[0]-par[2])*(x[0]-par[2]) + .25*par[1]*par[1]);
}
//
// Linear + Lorentzian function
// -----------------------
Double_t lorentzianPlusLinear(Double_t *x, Double_t *par) {
  return  ((0.5*par[0]*par[1]/TMath::Pi())/
    TMath::Max(1.e-10,((x[0]-par[2])*(x[0]-par[2])+.25*par[1]*par[1])))+par[3]+par[4]*x[0];
}

void fit2DProj(TString name, TString path, int minEntries, int rebin, int fitType) {
  
  //Read the TH2 from file
  TFile *inputFile = new TFile(path);
  TH2 * histo = (TH2*) inputFile->Get(name);; 
  histo->RebinX(rebin);

  //Declare some variables
  TH1 * histoY;
  TString nameY; 
  vector<double> Ftop;
  vector<double> Fwidth;
  vector<double> Fmass;
  vector<double> Etop;
  vector<double> Ewidth;
  vector<double> Emass;
  vector<double> Fchi2;
  vector<double> Xcenter;
  vector<double> Ex;

  //TFile *fileOut=new TFile("fitCompare2.root","update");

  for (int i=1; i<=(histo->GetXaxis()->GetNbins());i++) {

    //Project on Y (set name and title)
    char str[100];
    sprintf(str,"%i",i);
    TString iS(str);
    nameY = name + "_"+ iS;
    std::cout << "nameY  " << nameY << std::endl;
    histoY = histo->ProjectionY(nameY, i, i);
    //histoY->Rebin(2);
    char title[80];
    double x= histo->GetXaxis()-> GetBinCenter(i);
    //sprintf(title,"Projection of bin=%i",i);
    sprintf(title,"Projection of x=%f",x);
    histoY->SetTitle(title);

    if (histoY ->GetEntries() > minEntries) {
      //Make the dirty work!
      TF1 *fit;
      if(fitType == 1){
	fit = gaussianFit(histoY);
      }
      else if(fitType == 2){
	fit = lorentzianFit(histoY);
      }
      else if(fitType == 3)
	fit = linLorentzianFit(histoY);
      else {
	std::cout<<"Wrong fit type: 1=gaussian, 2=lorentzian, 3=lorentzian+linear."<<std::endl;
	abort();
      }

      double *par = fit->GetParameters();
      double *err = fit->GetParErrors();

      if(par[2]>150 || par[2]<50 || err[2]>5)
	continue;
      //Only for check
      TCanvas *c = new TCanvas(nameY, nameY);
      histoY->Draw();
      fit->Draw("same");
      //fileOut->cd();
      //c->Write();

      //Store the fit results
      Ftop.push_back(par[0]);
      Fwidth.push_back(fabs(par[1]));//sometimes the gaussian has negative width (checked with Rene Brun)
      Fmass.push_back(par[2]);
      Etop.push_back(err[0]);
      Ewidth.push_back(err[1]);
      Emass.push_back(err[2]);
	
      Fchi2.push_back(fit->GetChisquare()/fit->GetNDF());
	
      double xx= histo->GetXaxis()->GetBinCenter(i);
      Xcenter.push_back(xx);
      double ex = 0;
      Ex.push_back(ex); 
    }
  }

  //fileOut->Close();

  //Plots the fit results in  TGraphs
  const int nn= Ftop.size();                   
  double x[nn],ym[nn],e[nn],eym[nn];
  double yw[nn],eyw[nn],yc[nn];

  for (int j=0;j<nn;j++){
    x[j]=Xcenter[j];
    ym[j]=Fmass[j];
    eym[j]=Emass[j];
    yw[j]=Fwidth[j];
    eyw[j]=Ewidth[j];
    yc[j]=Fchi2[j];
    e[j]=0;
  }

  TGraphErrors *grM = new TGraphErrors(nn,x,ym,e,eym);
  grM->SetTitle(name+"_M");
  grM->SetName(name+"_M");
  TGraphErrors *grW = new TGraphErrors(nn,x,yw,e,eyw);
  grW->SetTitle(name+"_W");
  grW->SetName(name+"_W");
  TGraphErrors *grC = new TGraphErrors(nn,x,yc,e,e);
  grC->SetTitle(name+"_chi2");
  grC->SetName(name+"_chi2");

  grM->SetMarkerColor(4);
  grM->SetMarkerStyle(20);
  grW->SetMarkerColor(4);
  grW->SetMarkerStyle(20);
  grC->SetMarkerColor(4);
  grC->SetMarkerStyle(20);

  //Draw the graphs
  TCanvas * c1 = new TCanvas(name+"_W",name+"_W");
  grW->Draw("AP");
  TCanvas * c2 = new TCanvas(name+"_M",name+"_M");
  grM->Draw("Ap");
  TCanvas * c3 = new TCanvas(name+"_C",name+"_C");
  grC->Draw("Ap");

  //Save the graphs in a file
  TFile *file = new TFile("filegraph.root","update");
  grW->Write();
  grM->Write();
  grC->Write();
  file->Close();
}


TF1* gaussianFit(TH1* histoY){
  TString name = histoY->GetName() + TString("Fit");

    // Fit slices projected along Y from bins in X 
  TF1 *fit = new TF1(name,gaussian,70,110,3);
  fit->SetParameters(histoY->GetMaximum(),histoY->GetRMS(),histoY->GetMean());
  fit->SetParNames("norm","width","mean");
  fit->SetLineWidth(2);
  histoY -> Fit(name,"0","",85,97);
  return fit;
}

TF1* lorentzianFit(TH1* histoY){
  TString name = histoY->GetName() + TString("Fit");

    // Fit slices projected along Y from bins in X 
    TF1 *fit = new TF1(name,lorentzian,70,110,3);
  fit->SetParameters(histoY->GetMaximum(),histoY->GetRMS(),histoY->GetMean());
  fit->SetParNames("norm","width","mean");
  fit->SetLineWidth(2);
  histoY -> Fit(name,"0","",85,97);
  return fit;
}

TF1* linLorentzianFit(TH1* histoY){
  TString name = histoY->GetName() + TString("Fit");

  // Fit slices projected along Y from bins in X 
  TF1 *fit = new TF1(name,lorentzianPlusLinear,70,110,5);
  fit->SetParameters(histoY->GetMaximum(),histoY->GetRMS(),histoY->GetMean(),10,-0.1);
  fit->SetParNames("norm","width","mean","offset","slope");
  fit->SetParLimits(1,-10,10);
  //fit->SetParLimits(2,85,95);
  fit->SetParLimits(2,90,93);
  fit->SetParLimits(3,0,100);
  fit->SetParLimits(4,-1,0);
  fit->SetLineWidth(2);
  histoY -> Fit(name,"0","",85,97);
  return fit;
}


