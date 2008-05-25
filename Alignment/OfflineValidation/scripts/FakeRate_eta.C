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
gStyle->SetTitleFontSize(0.07);
gStyle->SetTitleSize(0.045,"xy");
gStyle->SetLabelSize(0.05,"xy");
gStyle->SetHistFillStyle(1001);
gStyle->SetHistFillColor(0);
gStyle->SetHistLineStyle(1);
gStyle->SetHistLineWidth(1);
gStyle->SetHistLineColor(1);
gStyle->SetTitleXOffset(1.1);
gStyle->SetTitleYOffset(1.25);
//gStyle->SetOptStat(1110);
gStyle->SetOptStat(kFALSE);
gStyle->SetOptFit(0111);
gStyle->SetStatH(0.1); 

TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
// c1->SetBorderSize(2);
// c1->SetFrameFillColor(0);
// c1->SetFillColor(0);
c1->SetGrid(1,1);

TFile f0("/gpfs/defilippis/ValidationMisalignedTracker_ttbar.root");  
TTree *MyTree=FakeTracks;

TFile f1("/gpfs/defilippis/ValidationMisalignedTracker_ttbar_SurveyLAS_merged.root");  
TTree *MyTree1=FakeTracks;

TFile f2("/gpfs/defilippis/ValidationMisalignedTracker_ttbar_SurveyLASCosmics_merged.root");  
TTree *MyTree2=FakeTracks;

TFile f3("/gpfs/defilippis/ValidationMisalignedTracker_ttbar_10pb_merged.root");  
TTree *MyTree3=FakeTracks;

TFile f4("/gpfs/defilippis/ValidationMisalignedTracker_ttbar_100pb_merged.root");  
TTree *MyTree4=FakeTracks;

TFile f5("/gpfs/defilippis/ValidationMisalignedTracker_ttbar_10pb_NOAPE_merged.root");  
TTree *MyTree5=FakeTracks;


//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA ALIGNED
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero = new TH1F("etazero","eta zero",70,0.,2.5); 
TH1F *etauno = new TH1F("etauno","eta uno",70,0.,2.5); 
TH1F *etadue = new TH1F("etadue","eta due",70,0.,2.5); 
TH1F *etatre = new TH1F("etatre","eta tre",70,0.,2.5); 

MyTree->Project("etazero","abs(fakereceta)","fakerecnhit>=8");
MyTree->Project("etauno","abs(fakereceta)","fakerecnhit>=8 && fake==1");

etadue->Add(etazero,etauno,1,-1);

etadue->Divide(etazero);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<71; k++){
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

etadue->SetTitle("Fake rate vs #eta for ttbar events");
etadue->SetMarkerStyle(20);
etadue->SetMarkerColor(2);
etadue->SetMarkerSize(0.9);
etadue->SetLineColor(1);
etadue->SetLineWidth(1);
etadue->Draw("P"); 

c1->Update();
//c1->WaitPrimitive();

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA SCEN 1 
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero_scen1 = new TH1F("etazero_scen1","eta zero",70,0.,2.5); 
TH1F *etauno_scen1 = new TH1F("etauno_scen1","eta uno",70,0.,2.5); 
TH1F *etadue_scen1 = new TH1F("etadue_scen1","eta due",70,0.,2.5); 
TH1F *etatre_scen1 = new TH1F("etatre_scen1","eta tre",70,0.,2.5); 

MyTree1->Project("etazero_scen1","abs(fakereceta)","fakerecnhit>=8");
MyTree1->Project("etauno_scen1","abs(fakereceta)","fakerecnhit>=8 && fake==1");

etadue_scen1->Add(etazero_scen1,etauno_scen1,1,-1);

etadue_scen1->Divide(etazero_scen1);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<71; k++){
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


etadue_scen1->SetTitle("Fake rate vs #eta for ttbar events");
etadue_scen1->SetMarkerStyle(21);
etadue_scen1->SetMarkerColor(3);
etadue_scen1->SetMarkerSize(0.9);
etadue_scen1->SetLineColor(1);
etadue_scen1->SetLineWidth(1);
etadue_scen1->Draw("P"); 

c1->Update();
c1->WaitPrimitive();


//&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA SCEN 1 
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero_scen2 = new TH1F("etazero_scen2","eta zero",70,0.,2.5); 
TH1F *etauno_scen2 = new TH1F("etauno_scen2","eta uno",70,0.,2.5); 
TH1F *etadue_scen2 = new TH1F("etadue_scen2","eta due",70,0.,2.5); 
TH1F *etatre_scen2 = new TH1F("etatre_scen2","eta tre",70,0.,2.5); 

MyTree2->Project("etazero_scen2","abs(fakereceta)","fakerecnhit>=8");
MyTree2->Project("etauno_scen2","abs(fakereceta)","fakerecnhit>=8 && fake==1");

etadue_scen2->Add(etazero_scen2,etauno_scen2,1,-1);

etadue_scen2->Divide(etazero_scen2);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<71; k++){
  MC_bin = etazero_scen2->GetBinContent(k);
  Fake_bin =etadue_scen2->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Fake_bin*(1.-Fake_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  etadue_scen2->SetBinError(k,err);
  MC_bin=0.;
  Fake_bin=0.;
  err=0.;
}


etadue_scen2->SetTitle("Fake rate vs #eta for ttbar events");
etadue_scen2->SetMarkerStyle(22);
etadue_scen2->SetMarkerColor(4);
etadue_scen2->SetMarkerSize(0.9);
etadue_scen2->SetLineColor(1);
etadue_scen2->SetLineWidth(1);
etadue_scen2->Draw("P"); 

c1->Update();
c1->WaitPrimitive();


//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA SCEN 1 NO ERR
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero_scen3 = new TH1F("etazero_scen3","eta zero",70,0.,2.5); 
TH1F *etauno_scen3 = new TH1F("etauno_scen3","eta uno",70,0.,2.5); 
TH1F *etadue_scen3 = new TH1F("etadue_scen3","eta due",70,0.,2.5); 
TH1F *etatre_scen3 = new TH1F("etatre_scen3","eta tre",70,0.,2.5); 

MyTree3->Project("etazero_scen3","abs(fakereceta)","fakerecnhit>=8");
MyTree3->Project("etauno_scen3","abs(fakereceta)","fakerecnhit>=8 && fake==1");

etadue_scen3->Add(etazero_scen3,etauno_scen3,1,-1);

etadue_scen3->Divide(etazero_scen3);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<71; k++){
  MC_bin = etazero_scen3->GetBinContent(k);
  Fake_bin =etadue_scen3->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Fake_bin*(1.-Fake_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  etadue_scen3->SetBinError(k,err);
  MC_bin=0.;
  Fake_bin=0.;
  err=0.;
}

etadue_scen3->SetTitle("Fake rate vs #eta scen3");
etadue_scen3->SetMarkerStyle(23);
etadue_scen3->SetMarkerColor(5);
etadue_scen3->SetMarkerSize(0.9);
etadue_scen3->SetLineColor(1);
etadue_scen3->SetLineWidth(1);
etadue_scen3->Draw("P"); 

c1->Update();
c1->WaitPrimitive();

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA SCEN 1 NO ERR
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero_scen4 = new TH1F("etazero_scen4","eta zero",70,0.,2.5); 
TH1F *etauno_scen4 = new TH1F("etauno_scen4","eta uno",70,0.,2.5); 
TH1F *etadue_scen4 = new TH1F("etadue_scen4","eta due",70,0.,2.5); 
TH1F *etatre_scen4 = new TH1F("etatre_scen4","eta tre",70,0.,2.5); 

MyTree4->Project("etazero_scen4","abs(fakereceta)","fakerecnhit>=8");
MyTree4->Project("etauno_scen4","abs(fakereceta)","fakerecnhit>=8 && fake==1");

etadue_scen4->Add(etazero_scen4,etauno_scen4,1,-1);

etadue_scen4->Divide(etazero_scen4);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<71; k++){
  MC_bin = etazero_scen4->GetBinContent(k);
  Fake_bin =etadue_scen4->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Fake_bin*(1.-Fake_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  etadue_scen4->SetBinError(k,err);
  MC_bin=0.;
  Fake_bin=0.;
  err=0.;
}

etadue_scen4->SetTitle("Fake rate vs #eta scen4");
etadue_scen4->SetMarkerStyle(24);
etadue_scen4->SetMarkerColor(6);
etadue_scen4->SetMarkerSize(0.9);
etadue_scen4->SetLineColor(1);
etadue_scen4->SetLineWidth(1);
etadue_scen4->Draw("P"); 

c1->Update();
c1->WaitPrimitive();

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA SCEN 1 NO ERR
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero_noErr = new TH1F("etazero_noErr","eta zero",70,0.,2.5); 
TH1F *etauno_noErr = new TH1F("etauno_noErr","eta uno",70,0.,2.5); 
TH1F *etadue_noErr = new TH1F("etadue_noErr","eta due",70,0.,2.5); 
TH1F *etatre_noErr = new TH1F("etatre_noErr","eta tre",70,0.,2.5); 

MyTree5->Project("etazero_noErr","abs(fakereceta)","fakerecnhit>=8");
MyTree5->Project("etauno_noErr","abs(fakereceta)","fakerecnhit>=8 && fake==1");

etadue_noErr->Add(etazero_noErr,etauno_noErr,1,-1);

etadue_noErr->Divide(etazero_noErr);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<71; k++){
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

etadue_noErr->SetTitle("Fake rate vs #eta noErr");
etadue_noErr->SetMarkerStyle(26);
etadue_noErr->SetMarkerColor(7);
etadue_noErr->SetMarkerSize(0.9);
etadue_noErr->SetLineColor(1);
etadue_noErr->SetLineWidth(1);
etadue_noErr->Draw("P"); 

c1->Update();
c1->WaitPrimitive();


// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS ETA STACKED
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

etadue->SetTitle("Fake rate vs #eta for ttbar events");
etadue->SetXTitle("#eta");
etadue->SetYTitle("Fake Rate");
etadue->SetAxisRange(-0.005,0.35,"Y");
etadue->Draw();
etadue_noErr->Draw("same");
etadue_scen1->Draw("same");
etadue_scen2->Draw("same");
etadue_scen3->Draw("same");
etadue_scen4->Draw("same");

TLegend *leg1 = new TLegend(0.105,0.65,0.45,0.895);
leg1->SetTextAlign(32);
leg1->SetTextColor(1);
leg1->SetTextSize(0.033);
leg1->SetFillColor(0);


leg1->AddEntry(etadue,"perfect ", "P");
leg1->AddEntry(etadue_scen1,"SurveyLAS ", "P");
leg1->AddEntry(etadue_scen2,"SurveyLASCosmics ", "P");
leg1->AddEntry(etadue_scen3,"10 pb^{-1} ", "P");
leg1->AddEntry(etadue_scen4,"100 pb^{-1} ", "P");
leg1->AddEntry(etadue_noErr,"10 pb^{-1}; APE not used", "P");


leg1->Draw();

c1->Update();
c1->SaveAs("Fake_eta_ttbar.eps");
c1->SaveAs("Fake_eta_ttbar.gif");
c1->WaitPrimitive();

gROOT->Reset();
gROOT->Clear();

delete c1;
}
