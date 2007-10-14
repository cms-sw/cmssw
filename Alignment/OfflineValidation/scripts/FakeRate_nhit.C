{
gROOT->Reset();
gROOT->Clear();

gROOT->Reset();
gROOT->Clear();

gStyle->SetNdivisions(5);
gStyle->SetCanvasBorderMode(0); 
gStyle->SetPadBorderMode(1);
gStyle->SetOptTitle(1);
gStyle->SetStatFont(42);
gStyle->SetCanvasColor(10);
gStyle->SetPadColor(0);
gStyle->SetTitleFont(62,"xy");
gStyle->SetLabelFont(62,"xy");
gStyle->SetTitleFontSize(0.05);
gStyle->SetTitleSize(0.045,"xy");
gStyle->SetLabelSize(0.05,"xy");
gStyle->SetHistFillStyle(1001);
gStyle->SetHistFillColor(0);
gStyle->SetHistLineStyle(1);
gStyle->SetHistLineWidth(1);
gStyle->SetHistLineColor(1);
gStyle->SetTitleXOffset(1.1);
gStyle->SetTitleYOffset(1.15);
gStyle->SetOptStat(1110);
// gStyle->SetOptStat(kFALSE);
gStyle->SetOptFit(0111);
gStyle->SetStatH(0.1); 

TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
// c1->SetBorderSize(2);
// c1->SetFrameFillColor(0);
// c1->SetFillColor(0);
c1->SetGrid(1,1);

TFile f0("TotalFakeRate_tth_scen0.root");  
TTree *MyTree=Tracks;

TFile f1("TotalFakeRate_tth_scen1.root");  
TTree *MyTree1=Tracks;

TFile f2("TotalFakeRate_tth_noErr_scen1.root");  
TTree *MyTree2=Tracks;


//TFile f0("FakeRate_tth_ORCA873_scen0.root");  
//TTree *MyTree=Tracks;

//TFile f1("TotalFakeRate_tth_scen1.root");  
//TTree *MyTree1=Tracks;

//TFile f2("TotalFakeRate_tth_noErr_scen1.root");  
//TTree *MyTree2=Tracks;

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS NHIT ALIGNED
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *nhitzero = new TH1F("nhitzero","nhit zero",80,0.,25.); 
TH1F *nhituno = new TH1F("nhituno","nhit uno",80,0.,25.); 
TH1F *nhitdue = new TH1F("nhitdue","nhit due",80,0.,25.); 
TH1F *nhittre = new TH1F("nhittre","nhit tre",80,0.,25.); 

MyTree->Project("nhitzero","recnhit");
MyTree->Project("nhituno","recnhit","eff==1");

nhitdue->Add(nhitzero,nhituno,1,-1);

nhitdue->Divide(nhitzero);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = nhitzero->GetBinContent(k);
  Fake_bin =nhitdue->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Fake_bin*(1.-Fake_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  nhitdue->SetBinError(k,err);
  MC_bin=0.;
  Fake_bin=0.;
  err=0.;
}

nhitdue->SetTitle("Fake rate vs #nhit");
nhitdue->SetMarkerStyle(20);
nhitdue->SetMarkerColor(2);
nhitdue->SetMarkerSize(0.9);
nhitdue->SetLineColor(1);
nhitdue->SetLineWidth(1);
nhitdue->Draw("P"); 

c1->Update();
//c1->WaitPrimitive();

// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS NHIT SCEN 1
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *nhitzero_scen1 = new TH1F("nhitzero_scen1","nhit zero",80,0.,25.); 
TH1F *nhituno_scen1 = new TH1F("nhituno_scen1","nhit uno",80,0.,25.); 
TH1F *nhitdue_scen1 = new TH1F("nhitdue_scen1","nhit due",80,0.,25.); 

MyTree1->Project("nhitzero_scen1","recnhit");
MyTree1->Project("nhituno_scen1","recnhit","eff==1");

nhitdue_scen1->Add(nhitzero_scen1,nhituno_scen1,1,-1);

nhitdue_scen1->Divide(nhitzero_scen1);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = nhitzero_scen1->GetBinContent(k);
  Fake_bin =nhitdue_scen1->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Fake_bin*(1.-Fake_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  nhitdue_scen1->SetBinError(k,err);
  MC_bin=0.;
  Fake_bin=0.;
  err=0.;
}

nhitdue_scen1->SetTitle("Fake rate vs #nhit");
nhitdue_scen1->SetMarkerStyle(21);
nhitdue_scen1->SetMarkerColor(3);
nhitdue_scen1->SetMarkerSize(0.9);
nhitdue_scen1->SetLineColor(1);
nhitdue_scen1->SetLineWidth(1);
nhitdue_scen1->Draw("P"); 

c1->Update();
//c1->WaitPrimitive();

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS NHIT SCEN 1 NO ERR
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *nhitzero_noErr = new TH1F("nhitzero_noErr","nhit zero",80,0.,25.); 
TH1F *nhituno_noErr = new TH1F("nhituno_noErr","nhit uno",80,0.,25.); 
TH1F *nhitdue_noErr = new TH1F("nhitdue_noErr","nhit due",80,0.,25.); 

MyTree2->Project("nhitzero_noErr","recnhit");
MyTree2->Project("nhituno_noErr","recnhit","eff==1");

nhitdue_noErr->Add(nhitzero_noErr,nhituno_noErr,1,-1);

nhitdue_noErr->Divide(nhitzero_noErr);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = nhitzero_noErr->GetBinContent(k);
  Fake_bin =nhitdue_noErr->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Fake_bin*(1.-Fake_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  nhitdue_noErr->SetBinError(k,err);
  MC_bin=0.;
  Fake_bin=0.;
  err=0.;
}

nhitdue_noErr->SetTitle("Fake rate vs #nhit noErr");
nhitdue_noErr->SetMarkerStyle(22);
nhitdue_noErr->SetMarkerColor(4);
nhitdue_noErr->SetMarkerSize(0.9);
nhitdue_noErr->SetLineColor(1);
nhitdue_noErr->SetLineWidth(1);
nhitdue_noErr->Draw("P"); 

c1->Update();
//c1->WaitPrimitive();

// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS NHIT STACKED
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
nhitdue->SetTitle("Fake Rate vs No. of hits");
nhitdue->SetXTitle("No. of hits");
nhitdue->SetYTitle("Fake Rate");
nhitdue->SetAxisRange(-0.01,0.65,"Y");
nhitdue->Draw();
nhitdue_noErr->Draw("same");
nhitdue_scen1->Draw("same");

TLegend *leg1 = new TLegend(0.1,0.81,0.82,0.925);
leg1->SetTextAlign(32);
leg1->SetTextColor(1);
leg1->SetTextSize(0.03);

leg1->AddEntry("nhitdue","Perfect Alignment", "P");
leg1->AddEntry("nhitdue_scen1","Alignment in < 1 fb^{-1}; Alignment error used", "P");
leg1->AddEntry("nhitdue_noErr","Alignment in < 1 fb^{-1}; Alignment error not used", "P");

leg1->Draw();

c1->Update();
c1->SaveAs("Fake_nhit_tth.eps");
c1->WaitPrimitive();

gROOT->Reset();
gROOT->Clear();

delete c1;
}
