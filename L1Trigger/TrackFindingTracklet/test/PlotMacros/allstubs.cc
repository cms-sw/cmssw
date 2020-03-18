#include "TMath.h"
#include "TRint.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TF1.h"
#include "TGaxis.h"
#include <fstream>
#include <iostream>
#include "TMath.h"
#include "TString.h"
#include <string>
#include <map>

#include "plotHist.cc"

void allstubs(TString subname=""){
//
// To see the output of this macro, click here.

//

  int ncut1=72;
  int ncut2=108;

  if (!(subname=="me"||subname=="te"||subname=="")) {
    cout << "Argument to the allstubs macro has to be either 'me' or 'te' or empty"
	 <<endl;
    return;  
  }
  
  gROOT->Reset();

  gROOT->SetStyle("Plain");

  gStyle->SetCanvasColor(kWhite);
  
  gStyle->SetCanvasBorderMode(0);     // turn off canvas borders
  gStyle->SetPadBorderMode(0);
  gStyle->SetOptStat(1111);
  gStyle->SetOptTitle(1);
  
  // For publishing:
  gStyle->SetLineWidth(2);
  gStyle->SetTextSize(1.1);
  gStyle->SetLabelSize(0.06,"xy");
  gStyle->SetTitleSize(0.06,"xy");
  gStyle->SetTitleOffset(1.2,"x");
  gStyle->SetTitleOffset(1.0,"y");
  gStyle->SetPadTopMargin(0.1);
  gStyle->SetPadRightMargin(0.1);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.12);
  

  
  TCanvas* c1 = new TCanvas("c1","Track performance",200,10,700,800);
  c1->Divide(2,3);
  c1->SetFillColor(0);
  c1->SetGrid();
  
  TCanvas* c2 = new TCanvas("c2","Track performance",200,10,700,800);
  c2->Divide(2,3);
  c2->SetFillColor(0);
  c2->SetGrid();


  double max=128.0;
 
  TH1 *hist_L1 = new TH1F("L1","All Stub Occupancy",max+1,-0.5,max+0.5);
  TH1 *hist_L2 = new TH1F("L2","All Stub Occupancy",max+1,-0.5,max+0.5);
  TH1 *hist_L3 = new TH1F("L3","All Stub Occupancy",max+1,-0.5,max+0.5);
  TH1 *hist_L4 = new TH1F("L4","All Stub Occupancy",max+1,-0.5,max+0.5);
  TH1 *hist_L5 = new TH1F("L5","All Stub Occupancy",max+1,-0.5,max+0.5);
  TH1 *hist_L6 = new TH1F("L6","All Stub Occupancy",max+1,-0.5,max+0.5);

  TH1 *hist_D1 = new TH1F("D1","All Stub Occupancy",max+1,-0.5,max+0.5);
  TH1 *hist_D2 = new TH1F("D2","All Stub Occupancy",max+1,-0.5,max+0.5);
  TH1 *hist_D3 = new TH1F("D3","All Stub Occupancy",max+1,-0.5,max+0.5);
  TH1 *hist_D4 = new TH1F("D4","All Stub Occupancy",max+1,-0.5,max+0.5);
  TH1 *hist_D5 = new TH1F("D5","All Stub Occupancy",max+1,-0.5,max+0.5);
  


  ifstream in("allstubs"+subname+".txt");

  int count=0;
  
  std::map<TString, TH1*> hists;
  
  while (in.good()){
    
    TString name;
    int stubs;
  
    in >>name>>stubs;

    if (!in.good()) continue;
    
    if (name.Contains("L1")) hist_L1->Fill(stubs);
    if (name.Contains("L2")) hist_L2->Fill(stubs);
    if (name.Contains("L3")) hist_L3->Fill(stubs);
    if (name.Contains("L4")) hist_L4->Fill(stubs);
    if (name.Contains("L5")) hist_L5->Fill(stubs);
    if (name.Contains("L6")) hist_L6->Fill(stubs);
   
    if (name.Contains("D1")) hist_D1->Fill(stubs);
    if (name.Contains("D2")) hist_D2->Fill(stubs);
    if (name.Contains("D3")) hist_D3->Fill(stubs);
    if (name.Contains("D4")) hist_D4->Fill(stubs);
    if (name.Contains("D5")) hist_D5->Fill(stubs);

    

    //cout << name <<" "<< countall <<" "<< countpass << endl;
    
    if (stubs>max) stubs=max;
    
    std::map<TString, TH1*>::iterator it=hists.find(name);
    
    if (it==hists.end()) {
     TH1 *hist = new TH1F(name,name,max+1,-0.5,max+0.5);
     hist->Fill(stubs);
     hists[name]=hist;
    } else {
      hists[name]->Fill(stubs);
    }
    

    count++;

  }

  cout << "count = "<<count<<endl;
  
  c1->cd(1);
  //gPad->SetLogy();
  plotHist(hist_L1,0.05,ncut1,ncut2);
  
  c1->cd(2);
  //gPad->SetLogy();
  plotHist(hist_L2,0.05,ncut1,ncut2);
  
  c1->cd(3);
  //gPad->SetLogy();
  plotHist(hist_L3,0.05,ncut1,ncut2);
  
  c1->cd(4);
  //gPad->SetLogy();
  plotHist(hist_L4,0.05,ncut1,ncut2);

  c1->cd(5);
  //gPad->SetLogy();
  plotHist(hist_L5,0.05,ncut1,ncut2);
  
  c1->cd(6);
  //gPad->SetLogy();
  plotHist(hist_L6,0.05,ncut1,ncut2);
  
  c1->Print("allstubs"+subname+".pdf(");

  c2->cd(1);
  //gPad->SetLogy();
  plotHist(hist_D1,0.05,ncut1,ncut2);
  
  c2->cd(2);
  //gPad->SetLogy();
  plotHist(hist_D2,0.05,ncut1,ncut2);
  
  c2->cd(3);
  //gPad->SetLogy();
  plotHist(hist_D3,0.05,ncut1,ncut2);
  
  c2->cd(4);
  //gPad->SetLogy();
  plotHist(hist_D4,0.05,ncut1,ncut2);
  
  c2->cd(5);
  //gPad->SetLogy();
  plotHist(hist_D5,0.05,ncut1,ncut2);
  
  c2->Print("allstubs"+subname+".pdf","pdf");
  
  
  int pages=0;
  
  std::map<TString, TH1*>::iterator it=hists.begin();
  
  TCanvas* c=0;
  
  bool first=true;
  
  while(it!=hists.end()) {
    
    if (pages%4==0) {
      
      c = new TCanvas(it->first,"Track performance",200,50,600,700);
      c->Divide(2,2);
      c->SetFillColor(0);
      c->SetGrid();

    }
    
    c->cd(pages%4+1);
    //sgPad->SetLogy();
    plotHist(it->second,0.05,ncut1,ncut2);
    
    pages++;
    
    ++it;

    if (pages%4==0) {
      c->Print("allstubs"+subname+".pdf","pdf");
    }
   
    
  }

  c->Print("allstubs"+subname+".pdf)","pdf");

  
}
