
class Plots{

  public:
  Plots();

};

void Plots::Plots()
{
gROOT->Reset();
gROOT->Clear();

//gStyle->SetOptStat(kFALSE);
// gStyle->SetOptFit(0111);
gStyle->SetStatH(0.1); 
gStyle->SetOptFit(1111);

TCanvas *c1 = new TCanvas("c1","c1",129,17,926,703);
c1->SetBorderSize(2);
c1->SetFrameFillColor(0);
c1->SetLogy(0);
c1->cd(); 


TFile *f[5];
TTree *MyTree[5];

f[2]=new TFile("ValidationMisalignedTracker_zmumu_chiara.root");
MyTree[2]=EffTracks;


////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA ALIGNED
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

char histoname[128];
char name[128];
TH1F *mZmu[5];

Double_t lorentzianPeak(Double_t *x, Double_t *par) {
return (0.5*par[0]*par[1]/TMath::Pi()) /
TMath::Max( 1.e-10,(x[0]-par[2])*(x[0]-par[2]) + .25*par[1]*par[1]);
}

TF1 *fitFcn = new TF1("fitFcn",lorentzianPeak,70,110,3);
fitFcn->SetParameters(3.,4.,90.);
fitFcn->SetParNames("Ftop","Fwidth","Fmass");
fitFcn->SetLineWidth(2);

 int i=2;
//for(int i=2; i<3; i++){
 sprintf(name,"mZmu[%d]",i);
 mZmu[i] = new TH1F(name,name,200,0.,200.);
 sprintf(histoname,"mZmu[%d]",i);
 MyTree[i]->Project(histoname,"recmzmu","eff==1 && recmzmu>0.&& recpt>3. && abs(receta)<2.5");
 
 cout << "Entries " << mZmu[i]->GetEntries() <<endl;
 mZmu[i]->Scale(1/mZmu[i]->GetEntries());
 mZmu[i]->SetTitle("Z->#mu#mu mass from Z->#mu#mu");    
 mZmu[i]->SetXTitle(" GeV/c2");
 mZmu[i]->SetYTitle("arb. units");
 
 mZmu[i]->SetLineColor(i+2);
 // mZmu[i]->SetLineStyle(i+1);
 // mZmu[i]->SetLineWidth(i+2);
 //if (i==0) mZmu[i]->Draw();
 //else mZmu[i]->Draw("same");
 
 mZmu[i]->Fit("fitFcn","M","",60,120);
 //mZmu[i]->Fit("gaus","","",70,110); 
 c1->Update();
  

c1->SaveAs("mZ_mu_fitL1.eps");
gROOT->Reset();
gROOT->Clear();
delete c1;
}


