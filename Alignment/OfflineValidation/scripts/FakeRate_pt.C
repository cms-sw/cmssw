{
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
gStyle->SetTitleFontSize(0.045);
gStyle->SetTitleSize(0.042,"xy");
gStyle->SetLabelSize(0.05,"xy");
gStyle->SetHistFillStyle(1001);
gStyle->SetHistFillColor(0);
gStyle->SetHistLineStyle(1);
gStyle->SetHistLineWidth(1);
gStyle->SetHistLineColor(1);
gStyle->SetTitleXOffset(1.1);
gStyle->SetTitleYOffset(1.15);
gStyle->SetOptStat(1110);
gStyle->SetOptStat(kFALSE);
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

// TFile f0("FakeRate_h160_ORCA873_scen0.root");  
// TTree *MyTree=Tracks;

// TFile f1("FakeRate_h160_ORCA873_scen1.root");  
// TTree *MyTree1=Tracks;

// TFile f2("FakeRate_h160_ORCA873_scen1_noErr.root");  
//TTree *MyTree2=Tracks;


//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA ALIGNED
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero = new TH1F("etazero","eta zero",20,0.,100.); 
TH1F *etauno = new TH1F("etauno","eta uno",20,0.,100.); 
TH1F *etadue = new TH1F("etadue","eta due",20,0.,100.); 
TH1F *etatre = new TH1F("etatre","eta tre",20,0.,100.); 

MyTree->Project("etazero","abs(recpt)","recnhit>=8");
MyTree->Project("etauno","abs(recpt)","recnhit>=8 && eff==1");

etadue->Add(etazero,etauno,1,-1);

etadue->Divide(etazero);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<21; k++){
  MC_bin = etazero->GetBinContent(k);
  Fake_bin =etadue->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Fake_bin*(1.-Fake_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  etadue->SetBinError(k,err);
  MC_bin=0.;
  Fake_bin=0.;
  err=0.;
}

etadue->SetTitle("Fake rate vs p_{T} for bt03_ttH120_8j");
etadue->SetMarkerStyle(20);
etadue->SetMarkerColor(2);
etadue->SetMarkerSize(0.9);
etadue->SetLineColor(1);
etadue->SetLineWidth(1);
etadue->Draw("P"); 

c1->Update();
c1->WaitPrimitive();

// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS ETA SCEN 1 
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero_scen1 = new TH1F("etazero_scen1","eta zero",20,0.,100.); 
TH1F *etauno_scen1 = new TH1F("etauno_scen1","eta uno",20,0.,100.); 
TH1F *etadue_scen1 = new TH1F("etadue_scen1","eta due",20,0.,100.); 
TH1F *etatre_scen1 = new TH1F("etatre_scen1","eta tre",20,0.,100.); 

MyTree1->Project("etazero_scen1","abs(recpt)","recnhit>=8");
MyTree1->Project("etauno_scen1","abs(recpt)","recnhit>=8 && eff==1");

etadue_scen1->Add(etazero_scen1,etauno_scen1,1,-1);

etadue_scen1->Divide(etazero_scen1);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<21; k++){
  MC_bin = etazero_scen1->GetBinContent(k);
  Fake_bin =etadue_scen1->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Fake_bin*(1.-Fake_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  etadue_scen1->SetBinError(k,err);
  MC_bin=0.;
  Fake_bin=0.;
  err=0.;
}


etadue_scen1->SetTitle("Fake rate vs p_{T} for bt03_ttH120_8j");
etadue_scen1->SetMarkerStyle(21);
etadue_scen1->SetMarkerColor(3);
etadue_scen1->SetMarkerSize(0.9);
etadue_scen1->SetLineColor(1);
etadue_scen1->SetLineWidth(1);
etadue_scen1->Draw("P"); 

c1->Update();
c1->WaitPrimitive();

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA SCEN 1 NO ERR
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero_noErr = new TH1F("etazero_noErr","eta zero",20,0.,100.); 
TH1F *etauno_noErr = new TH1F("etauno_noErr","eta uno",20,0.,100.); 
TH1F *etadue_noErr = new TH1F("etadue_noErr","eta due",20,0.,100.); 
TH1F *etatre_noErr = new TH1F("etatre_noErr","eta tre",20,0.,100.); 

MyTree2->Project("etazero_noErr","abs(recpt)","recnhit>=8");
MyTree2->Project("etauno_noErr","abs(recpt)","recnhit>=8 && eff==1");

etadue_noErr->Add(etazero_noErr,etauno_noErr,1,-1);

etadue_noErr->Divide(etazero_noErr);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<21; k++){
  MC_bin = etazero_noErr->GetBinContent(k);
  Fake_bin =etadue_noErr->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Fake_bin*(1.-Fake_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  etadue_noErr->SetBinError(k,err);
  MC_bin=0.;
  Fake_bin=0.;
  err=0.;
}

etadue_noErr->SetTitle("Fake rate vs p_{T} noErr");
etadue_noErr->SetMarkerStyle(22);
etadue_noErr->SetMarkerColor(4);
etadue_noErr->SetMarkerSize(0.9);
etadue_noErr->SetLineColor(1);
etadue_noErr->SetLineWidth(1);
etadue_noErr->Draw("P"); 

c1->Update();
c1->WaitPrimitive();

// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS ETA STACKED
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

etadue->SetTitle("Fake rate vs p_{T} for bt03_ttH120_8j");
etadue->SetXTitle("p_{T}(GeV/c)");
etadue->SetYTitle("Fake Rate");
etadue->SetAxisRange(-0.005,0.5,"Y");
etadue->Draw();
etadue_noErr->Draw("same");
etadue_scen1->Draw("same");

TLegend *leg1 = new TLegend(0.1,0.81,0.82,0.925);
leg1->SetTextAlign(32);
leg1->SetTextColor(1);
leg1->SetTextSize(0.03);

leg1->AddEntry("etadue","Perfect Alignment", "P");
leg1->AddEntry("etadue_scen1","Alignment in < 1 fb^{-1}; Alignment error used", "P");
leg1->AddEntry("etadue_noErr","Alignment in < 1 fb^{-1}; Alignment error not used", "P");

leg1->Draw();

c1->Update();
c1->SaveAs("Fake_pt_tth.eps");
c1->WaitPrimitive();

gROOT->Reset();
gROOT->Clear();

delete c1;
}
