#include <memory>
#include <iostream>
#include <string>
#include <fstream>

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

void dNdEta()
{

 //gStyle->SetOptStat(10000000);
  // gStyle->SetStatBorderSize(0);

   TCanvas *c1 = new TCanvas("c1", "c1",364,44,699,499);
   gStyle->SetOptStat(0);
   c1->Range(-24.9362,1.71228,25.0213,4.77534);
   c1->SetFillColor(10);
   c1->SetFillStyle(4000);
   c1->SetBorderSize(2);
   c1->SetFrameFillColor(0);
   c1->SetFrameFillStyle(0);
gStyle->SetOptStat(10001111);

//Hydjet2
float npart;

const int kMax = 200000; 
 
   int   mult;
   int   pdg[kMax];
   int   Mpdg[kMax];
   int   chg[kMax];
   float Px[kMax];
   float Py[kMax];
   float Pz[kMax];
   float E[kMax];   
   float X[kMax];
   float Y[kMax];
   float Z[kMax];
   float T[kMax]; 
   float eta[kMax]; 
   float pseudoRapidity[kMax];
   float NBinaryPart=0.;

   TFile *f = new TFile("treefile.root");
   TTree *td = (TTree*)f->Get("ana/hi");

   TTree *td2 = (TTree*)f->Get("ana/hi");
   TH1D *hy2 = new TH1D("hy2", "hy2", 11, -5.5, 5.5);
   td2->Draw("(0.5*log((sqrt(px*px+py*py+pz*pz)+pz)/(sqrt(px*px+py*py+pz*pz)-pz)))>>hy2","(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");//&&(sqrt(px*px+py*py)<0.5)

   TH1D *hy = new TH1D("hy", "hy", 11, -5.5, 5.5);
int nevents = td2->GetEntries(); 

  td->SetBranchAddress("npart",&npart);
  td->SetBranchAddress("mult",&mult);
  td->SetBranchAddress("pdg",pdg);
  td->SetBranchAddress("eta",eta);
  td->SetBranchAddress("chg",chg);
  td->SetBranchAddress("pseudoRapidity",pseudoRapidity);
  // int nevents = td->GetEntries();  
std::cout<< " Nevents "<< nevents <<std::endl;
   
   
  for (int k=0;k<nevents;k++) 
  {
            
      td->GetEntry(k);  
              
     for (int i=0;i<mult;i++) 
	   {

//all charged
      if( (abs(pdg[i])==211)||(abs(pdg[i])==321)||(abs(pdg[i])==2212) )// !(chg==0) )
      {

      //pt = TMath::Sqrt(Px[i]*Px[i]+Py[i]*Py[i]);      
      //phi = TMath::ATan2(Py[i],Px[i]);
  
      //v2 = TMath::Cos(2*phi); 
      //std::cout<< " eta "<< eta[i]<<std::endl;
      //hv2->Fill(pt,v2);
      //hv0->Fill(pt,1.);
      hy->Fill(pseudoRapidity[i]);
    }     
  } //mult
//NBinaryPart=NBinaryPart+190.5; //exp. value
NBinaryPart=NBinaryPart+(npart/2.);
} 

//hy->Scale(nevents/NBinaryPart);



/*

   //Info("fig_eta_Phobos.C", "Nevents %d ", nevents);
                  
   TH1D *hy = new TH1D("hy", "hy", 10, -2.5, 2.5);
   //TH1D *hyjets = new TH1D("hyjets", "hyjets", 51, -5.1, 5.1);
   //TH1D *hyhydro = new TH1D("hyhydro", "hyhydro", 51, -5.1, 5.1);

   hy->Sumw2();
   //td->Draw("(eta)>>hyhydro","(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");
   //td->Draw("(eta)>>hyjets","(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");
   td->Draw("(eta)>>hy","(abs(pdg)==211||abs(pdg)==321||abs(pdg)==2212)");//,"(!(chg==0))");
hy->Scale(1./190.5);
//std::cout<< " npart "<< npart<<std::endl;
//hy->Scale(1./npart);
*/
  


   hy->GetXaxis()->SetTitle("#eta");
   hy->GetYaxis()->SetTitle("(dN_{ch}/d#eta)/(N^{ev}*delta)");

   int nbx=hy->GetNbinsX();
   double binmin=hy->GetXaxis()->GetXmin();
   double binmax=hy->GetXaxis()->GetXmax();
   double delta=binmax-binmin;

   hy->Scale(1./(nevents));
   hy2->Scale(1./(nevents));
   //hyjets->Scale(nbx/ (nevents*delta));
   //hyhydro->Scale(nbx/ (nevents*delta));

   hy->SetLineWidth(2.);
   hy->SetLineStyle(1);
   hy2->SetLineWidth(3.);
   hy2->SetLineStyle(2);
   //hyjets->SetLineWidth(2.);
   //hyjets->SetLineStyle(2);
   //hyhydro->SetLineWidth(2.);
   //hyhydro->SetLineStyle(3);
    
   hy->Draw("histo");
   hy2->Draw("same:histo");
   //hyhydro->Draw("same:histo");

   cout<< " --- " << hy->GetBinContent(6)<<endl;


}
