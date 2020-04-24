#include <memory>
#include <iostream>
#include <string>
#include <fstream>
#include "TMath.h"
#include "stdlib.h"
#include "stdio.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TObject.h"
#include "TMultiGraph.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TGraphErrors.h"

using namespace std;

void eta_v2_pT()
{
   gStyle->SetOptStat(10000000);
   gStyle->SetStatBorderSize(0);

   TCanvas *c1 = new TCanvas("c1", "c1",364,44,699,499);
   gStyle->SetOptStat(0);
   c1->Range(-24.9362,1.71228,25.0213,4.77534);
   c1->SetFillColor(10);
   c1->SetFillStyle(4000);
   c1->SetBorderSize(2);
   c1->SetFrameFillColor(0);
   c1->SetFrameFillStyle(0);

bool etaAN=1;
bool v2AN=0;
bool ptAN=0;

   TFile *f = new TFile("RunOutput_all.root");
TFile *f2 = new TFile("treefile_all.root");
   TTree *td = (TTree*)f->Get("td");
TTree *td2 = (TTree*)f2->Get("ana/hi");
   Int_t nevents = td->GetEntries(); 
Int_t nevents2 = td2->GetEntries();   
cout<<" Number of events: HYDJET++ - "<<nevents<<", Hydjet2 - "<<nevents2<<endl;

if(etaAN){                  
   TH1D *hy = new TH1D("hy", "hy", 51, -5.1, 5.1);
   TH1D *hy2 = new TH1D("hy2", "hy2", 51, -5.1, 5.1);
   TH1D *hyjets = new TH1D("hyjets", "hyjets", 51, -5.1, 5.1);
   TH1D *hyhydro = new TH1D("hyhydro", "hyhydro", 51, -5.1, 5.1);
   hy->Sumw2();

   td->Draw("(0.5*log((sqrt(Px*Px+Py*Py+Pz*Pz)+Pz)/(sqrt(Px*Px+Py*Py+Pz*Pz)-Pz)))>>hyhydro","final==1&&type==0&&(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");
   td->Draw("(0.5*log((sqrt(Px*Px+Py*Py+Pz*Pz)+Pz)/(sqrt(Px*Px+Py*Py+Pz*Pz)-Pz)))>>hyjets","final==1&&type==1&&(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");


   td->Draw("(0.5*log((sqrt(Px*Px+Py*Py+Pz*Pz)+Pz)/(sqrt(Px*Px+Py*Py+Pz*Pz)-Pz)))>>hy","final==1&&(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)&&(sqrt(Px*Px+Py*Py)<0.5)");//"final==1&&(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");

   //td2->Draw("(pseudoRapidity)>>hy2","(pdg==211)&&(pt<0.5)");//"(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");
   td2->Draw("(0.5*log((sqrt(px*px+py*py+pz*pz)+pz)/(sqrt(px*px+py*py+pz*pz)-pz)))>>hy2","(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)&&(sqrt(px*px+py*py)<0.5)");//"final==1&&(ab

   Int_t nbx=hy->GetNbinsX();
   Double_t binmin=hy->GetXaxis()->GetXmin();
   Double_t binmax=hy->GetXaxis()->GetXmax();
   Double_t delta=binmax-binmin;

   Int_t nbx2=hy2->GetNbinsX();
   Double_t binmin2=hy2->GetXaxis()->GetXmin();
   Double_t binmax2=hy2->GetXaxis()->GetXmax();
   Double_t delta2=binmax2-binmin2;

   hy2->Scale(nbx2/(nevents2*delta2));

   hy->Scale(nbx/(nevents*delta));
   hyjets->Scale(nbx/ (nevents*delta));
   hyhydro->Scale(nbx/ (nevents*delta));

   hy->SetLineWidth(2.);
   hy->SetLineStyle(1);
   hyjets->SetLineWidth(2.);
   hyjets->SetLineStyle(2);
   hyhydro->SetLineWidth(2.);
   hyhydro->SetLineStyle(3);

   hy2->SetLineWidth(2.);
   hy2->SetLineStyle(1);
   hy2->SetLineColor(2);

   hy->Draw("histo");
   hy2->Draw("same:histo");
   //hyjets->Draw("same:histo");
   //hyhydro->Draw("same:histo");
   TLegend *legend=new TLegend(0.6, 0.6, 0.9, 0.9);       
   legend->AddEntry(hy, " HYDJET++: all charged  ", "l");
   legend->AddEntry(hy2, " Hydjet2: all charged  ", "l");
   legend->AddEntry(hyjets, " HYDJET++: jet part ", "l");
   legend->AddEntry(hyhydro, " HYDJET++: hydro part ", "l");
   //legend->Draw();





}else if(v2AN){

   TH1D *hy2v2 = new TH1D("h2v2", "h2v2_histo", 100, 0.0, 10.);
   TH1D *hy2v0 = new TH1D("h2v0", "h2v0_histo", 100, 0., 10.);
   TH1D *hy2v2res1 = new TH1D("h2v2res1", "h2v2res1", 100, 0.0, 10.);

   TH1D *hyv2 = new TH1D("hv2", "hv2_histo", 100, 0.0, 10.);
   TH1D *hyv0 = new TH1D("hv0", "hv0_histo", 100, 0., 10.);
   TH1D *hyv2res1 = new TH1D("hv2res1", "hv2res1", 100, 0.0, 10.);

   hyv2res1->Sumw2();
   hy2v2res1->Sumw2();

//   td2->Draw("(pt)>>h2v2","TMath::Cos(2*phi)","(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");
//   td2->Draw("(pt)>>h2v0","1.","(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");
   td2->Draw("(TMath::Sqrt(px*px+py*py))>>h2v2","TMath::Cos(2*(TMath::ATan2(py,px)))","(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");
   td2->Draw("(TMath::Sqrt(px*px+py*py))>>h2v0","1.","(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");


   td->Draw("(TMath::Sqrt(Px*Px+Py*Py))>>hv2","TMath::Cos(2*(TMath::ATan2(Py,Px)))","final==1&&(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");
   td->Draw("(TMath::Sqrt(Px*Px+Py*Py))>>hv0","1.","final==1&&(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");

   hyv2res1->Divide(hyv2,hyv0,1.,1.);
   hy2v2res1->Divide(hy2v2,hy2v0,1.,1.);

cout<<" V2 "<<endl;

    hy2v2->SetLineWidth(2);
    hy2v2->SetLineStyle(2);

   hyv2res1->SetLineWidth(2.);
   hyv2res1->SetLineStyle(1);

   hy2v2res1->SetLineWidth(2.);
   hy2v2res1->SetLineStyle(1);
   hy2v2res1->SetLineColor(2);
//hy2v2->Draw("histo");
//hy2v0->Draw("same:histo");
hyv2res1->GetXaxis()->SetTitle("pt, GeV/c");
hyv2res1->GetYaxis()->SetTitle(" v2 ");
hyv2res1->Draw("histo");
hy2v2res1->Draw("same:histo");

TLegend *legend=new TLegend(0.6, 0.6, 0.9, 0.9);       
   legend->AddEntry(hyv2res1, " HYDJET++: all charged  ", "l");
   legend->AddEntry(hy2v2res1, " Hydjet2: all charged  ", "l");
//   legend->Draw();

}else if(ptAN){

c1->SetLogy();
   TH1D *hypT = new TH1D("hpT", "hpT_histo", 100, 0.0, 20.);
   TH1D *hypTj = new TH1D("hpTj", "hpTj_histo", 100, 0.0, 20.);
   TH1D *hypTh = new TH1D("hpTh", "hpTh_histo", 100, 0.0, 20.);
   TH1D *hy2pT = new TH1D("h2pT", "h2pT_histo", 100, 0.0, 20.);

   td->Draw("sqrt(Px*Px+Py*Py)>>hpT","(1.0/(sqrt(Px*Px+Py*Py)))*(final==1 && pdg==211 && abs(0.5*log((E+Pz)/(E-Pz)))<1.)");
   td->Draw("sqrt(Px*Px+Py*Py)>>hpTj","(1.0/(sqrt(Px*Px+Py*Py)))*(final==1 && type==1 && pdg==211 && abs(0.5*log((E+Pz)/(E-Pz)))<1.)");
   td->Draw("sqrt(Px*Px+Py*Py)>>hpTh","(1.0/(sqrt(Px*Px+Py*Py)))*(final==1 && type==0 && pdg==211 && abs(0.5*log((E+Pz)/(E-Pz)))<1.)");
   td2->Draw("sqrt(px*px+py*py)>>h2pT","(1.0/(sqrt(px*px+py*py)))*(pdg==211 && abs(0.5*log((e+pz)/(e-pz)))<1.)");
//   td2->Draw("pt>>h2pT","(1.0/(pt))*( pdg==211 && abs(pseudoRapidity)<1.)");

   Int_t nbx=hypT->GetNbinsX();
   Double_t binmin=hypT->GetXaxis()->GetXmin();
   Double_t binmax=hypT->GetXaxis()->GetXmax();
   Double_t delta=binmax-binmin;
   Double_t delta_y=2; //[-1;1]

   Int_t nbx2=hy2pT->GetNbinsX();
   Double_t binmin2=hy2pT->GetXaxis()->GetXmin();
   Double_t binmax2=hy2pT->GetXaxis()->GetXmax();
   Double_t delta2=binmax2-binmin2;
   Double_t delta2_y=2; //[-1;1]

   hypT->Scale(nbx/ (2.0*TMath::Pi()*nevents*delta_y*delta)); 
   hypTj->Scale(nbx/ (2.0*TMath::Pi()*nevents*delta_y*delta));
   hypTh->Scale(nbx/ (2.0*TMath::Pi()*nevents*delta_y*delta)); 
   hy2pT->Scale(nbx2/ (2.0*TMath::Pi()*nevents2*delta2_y*delta2)); 

   hypT->GetXaxis()->SetTitle("p_{t} (GeV/c)");
   hypT->GetYaxis()->SetTitle("1/(2 #pi) d^{2}N/ N p_{t} dp_{t} dY, c^{2}/GeV^{2}");
      
   hypT->SetLineWidth(2.);
   hypTj->SetLineWidth(2.);
   hypTh->SetLineWidth(2.);
   hy2pT->SetLineWidth(2.);

   hypTj->SetLineStyle(2);
   hypTh->SetLineStyle(3);
   hy2pT->SetLineColor(2);

   hypT->Draw("hist");      
   //hypTj->Draw("same::hist");
   //hypTh->Draw("same::hist");
   hy2pT->Draw("same::hist");

   TLegend *legend=new TLegend(0.6, 0.6, 0.9, 0.9);   
   legend->AddEntry(hy2pT, " Hydjet2: all #pi^{+} ", "l"); 
   legend->AddEntry(hypT, " HYDJET++: all #pi^{+}  ", "l");
   legend->AddEntry(hypTj, " HYDJET++: jet part ", "l");
   legend->AddEntry(hypTh, " HYDJET++: hydro part ", "l");
 
   //legend->Draw();

}

}
