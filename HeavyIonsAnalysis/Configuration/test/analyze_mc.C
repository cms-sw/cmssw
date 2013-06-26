#include "TCanvas.h"
#include "TFile.h"
#include "TH1D.h"
#include "TTree.h"

void analyze_mc(){

TFile* inf = new TFile("/server/02a/yilmaz/data/production/c00to10/central.root");
 TTree* hi = dynamic_cast<TTree*>(inf->Get("ana/hi"));

 int nev = hi->GetEntries();

 const char* selection = "abs(chg > 0)";

TCanvas* c1 = new TCanvas("c1","c1",600,600); 
c1->Divide(2,2);

TCanvas* c2 = new TCanvas("c2","c2",600,600);
c2->Divide(2,2);

TH1D* h1 = new TH1D("h1","Impact Parameter; b[fm]; #_{events}",100,0,20);
TH1D* h2 = new TH1D("h2","Vertex; z[cm]; #_{events}",100,-10,10);
TH1D* h3 = new TH1D("h3","Vertex; r[cm]; #_{events}",100,-0.01,0.01);
TH1D* h4 = new TH1D("h4","Event Ids; id; #_{events}",1050,0,1050);
TH1D* h5 = new TH1D("h5","dN/d#eta;#eta;dN/d#eta",100,-10,10);
TH1D* h6 = new TH1D("h6","dN/dp_{T};p_{T}[GeV];dN/dp_{T}",200,0,20);
TH1D* h7 = new TH1D("h7","dN/d#phi;#phi;dN/d#phi",300,-3.15,3.15);
TH1D* h8 = new TH1D("h8","",1050,0,1050);

 c1->cd(1);
 hi->Draw("b>>h1");
 h1->Draw();
 c1->cd(2);
 hi->Draw("vz>>h2");
 h2->Draw();
 
 c1->cd(3);
 hi->Draw("sqrt(vy*vy+vx*vx)>>h3");
 h3->Draw();
 c1->cd(4);
 hi->Draw("event>>h4");
 h4->Draw();

 c2->cd(1);
 hi->Draw("eta>>h5",selection);
 h5->Scale(1./nev/h5->GetBinWidth(1));
 h5->Draw();
 c2->cd(2);
 c2->GetPad(2)->SetLogy();
 hi->Draw("pt>>h6",selection);
 h6->Scale(1./nev/h6->GetBinWidth(1));
 h6->Draw();
 c2->cd(3);
 hi->Draw("phi>>h7",selection);
 h7->Scale(1./nev/h7->GetBinWidth(1));
 h7->Draw();
 c2->cd(4);
 
 c1->Draw();
 c1->Print("central1.gif");
 c2->Draw();
 c2->Print("central2.gif");


}
