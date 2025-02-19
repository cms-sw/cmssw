{
gROOT->Reset();
gROOT->Clear();

gROOT->Reset();
gROOT->Clear();

gStyle->SetNdivisions(10);
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
gStyle->SetOptStat(kFALSE);
gStyle->SetOptFit(0111);
gStyle->SetStatH(0.1); 

TCanvas *c1 = new TCanvas("c1", "c1",129,17,926,703);
// c1->SetBorderSize(2);
// c1->SetFrameFillColor(0);
// c1->SetFillColor(0);
c1->SetGrid(1,1);
//c1->SetLogx(1);

TFile f0("/gpfs/defilippis/ValidationMisalignedTracker_ttbar.root");  
TTree *MyTree=FakeTracks;

TFile f1("/gpfs/defilippis/ValidationMisalignedTracker_ttbar_SurveyLAS.root");  
TTree *MyTree1=FakeTracks;

TFile f2("/gpfs/defilippis/ValidationMisalignedTracker_ttbar_SurveyLASCosmics.root");  
TTree *MyTree2=FakeTracks;

TFile f3("/gpfs/defilippis/ValidationMisalignedTracker_ttbar_10pb.root");  
TTree *MyTree3=FakeTracks;

TFile f4("/gpfs/defilippis/ValidationMisalignedTracker_ttbar_100pb.root");  
TTree *MyTree4=FakeTracks;

TFile f5("/gpfs/defilippis/ValidationMisalignedTracker_ttbar_10pb_NOAPE.root");  
TTree *MyTree5=FakeTracks;


//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS NHIT ALIGNED
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *nhitzero = new TH1F("nhitzero","nhit zero",11,7.5,18.5); 
TH1F *nhituno = new TH1F("nhituno","nhit uno",11,7.5,18.5); 
TH1F *nhitdue = new TH1F("nhitdue","nhit due",11,7.5,18.5); 
TH1F *nhittre = new TH1F("nhittre","nhit tre",11,7.5,18.5); 

MyTree->Project("nhitzero","fakerecnhit");
MyTree->Project("nhituno","fakerecnhit","fake==1");

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

TH1F *nhitzero_scen1 = new TH1F("nhitzero_scen1","nhit zero",11,7.5,18.5); 
TH1F *nhituno_scen1 = new TH1F("nhituno_scen1","nhit uno",11,7.5,18.5); 
TH1F *nhitdue_scen1 = new TH1F("nhitdue_scen1","nhit due",11,7.5,18.5); 

MyTree1->Project("nhitzero_scen1","fakerecnhit");
MyTree1->Project("nhituno_scen1","fakerecnhit","fake==1");

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


// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS NHIT SCEN 1
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *nhitzero_scen2 = new TH1F("nhitzero_scen2","nhit zero",11,7.5,18.5); 
TH1F *nhituno_scen2 = new TH1F("nhituno_scen2","nhit uno",11,7.5,18.5); 
TH1F *nhitdue_scen2 = new TH1F("nhitdue_scen2","nhit due",11,7.5,18.5); 

MyTree2->Project("nhitzero_scen2","fakerecnhit");
MyTree2->Project("nhituno_scen2","fakerecnhit","fake==1");

nhitdue_scen2->Add(nhitzero_scen2,nhituno_scen2,1,-1);

nhitdue_scen2->Divide(nhitzero_scen2);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = nhitzero_scen2->GetBinContent(k);
  Fake_bin =nhitdue_scen2->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Fake_bin*(1.-Fake_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  nhitdue_scen2->SetBinError(k,err);
  MC_bin=0.;
  Fake_bin=0.;
  err=0.;
}

nhitdue_scen2->SetTitle("Fake rate vs #nhit");
nhitdue_scen2->SetMarkerStyle(22);
nhitdue_scen2->SetMarkerColor(4);
nhitdue_scen2->SetMarkerSize(0.9);
nhitdue_scen2->SetLineColor(1);
nhitdue_scen2->SetLineWidth(1);
nhitdue_scen2->Draw("P"); 

c1->Update();
//c1->WaitPrimitive();


// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS NHIT SCEN 1
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *nhitzero_scen3 = new TH1F("nhitzero_scen3","nhit zero",11,7.5,18.5); 
TH1F *nhituno_scen3 = new TH1F("nhituno_scen3","nhit uno",11,7.5,18.5); 
TH1F *nhitdue_scen3 = new TH1F("nhitdue_scen3","nhit due",11,7.5,18.5); 

MyTree3->Project("nhitzero_scen3","fakerecnhit");
MyTree3->Project("nhituno_scen3","fakerecnhit","fake==1");

nhitdue_scen3->Add(nhitzero_scen3,nhituno_scen3,1,-1);

nhitdue_scen3->Divide(nhitzero_scen3);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = nhitzero_scen3->GetBinContent(k);
  Fake_bin =nhitdue_scen3->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Fake_bin*(1.-Fake_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  nhitdue_scen3->SetBinError(k,err);
  MC_bin=0.;
  Fake_bin=0.;
  err=0.;
}

nhitdue_scen3->SetTitle("Fake rate vs #nhit");
nhitdue_scen3->SetMarkerStyle(23);
nhitdue_scen3->SetMarkerColor(5);
nhitdue_scen3->SetMarkerSize(0.9);
nhitdue_scen3->SetLineColor(1);
nhitdue_scen3->SetLineWidth(1);
nhitdue_scen3->Draw("P"); 

c1->Update();
//c1->WaitPrimitive();


// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS NHIT SCEN 1
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *nhitzero_scen4 = new TH1F("nhitzero_scen4","nhit zero",11,7.5,18.5); 
TH1F *nhituno_scen4 = new TH1F("nhituno_scen4","nhit uno",11,7.5,18.5); 
TH1F *nhitdue_scen4 = new TH1F("nhitdue_scen4","nhit due",11,7.5,18.5); 

MyTree4->Project("nhitzero_scen4","fakerecnhit");
MyTree4->Project("nhituno_scen4","fakerecnhit","fake==1");

nhitdue_scen4->Add(nhitzero_scen4,nhituno_scen4,1,-1);

nhitdue_scen4->Divide(nhitzero_scen4);


float MC_bin=0.,Fake_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = nhitzero_scen4->GetBinContent(k);
  Fake_bin =nhitdue_scen4->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Fake_bin*(1.-Fake_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  nhitdue_scen4->SetBinError(k,err);
  MC_bin=0.;
  Fake_bin=0.;
  err=0.;
}

nhitdue_scen4->SetTitle("Fake rate vs #nhit");
nhitdue_scen4->SetMarkerStyle(24);
nhitdue_scen4->SetMarkerColor(6);
nhitdue_scen4->SetMarkerSize(0.9);
nhitdue_scen4->SetLineColor(1);
nhitdue_scen4->SetLineWidth(1);
nhitdue_scen4->Draw("P"); 

c1->Update();
//c1->WaitPrimitive();



//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS NHIT SCEN 1 NO ERR
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *nhitzero_noErr = new TH1F("nhitzero_noErr","nhit zero",11,7.5,18.5); 
TH1F *nhituno_noErr = new TH1F("nhituno_noErr","nhit uno",11,7.5,18.5); 
TH1F *nhitdue_noErr = new TH1F("nhitdue_noErr","nhit due",11,7.5,18.5); 

MyTree5->Project("nhitzero_noErr","fakerecnhit");
MyTree5->Project("nhituno_noErr","fakerecnhit","fake==1");

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
nhitdue_noErr->SetMarkerStyle(25);
nhitdue_noErr->SetMarkerColor(7);
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
nhitdue->SetAxisRange(0.001,0.95,"Y");
nhitdue->Draw();
nhitdue_noErr->Draw("same");
nhitdue_scen1->Draw("same");
nhitdue_scen2->Draw("same");
nhitdue_scen3->Draw("same");
nhitdue_scen4->Draw("same");


TLegend *leg1 = new TLegend(0.1,0.75,0.45,0.9);
leg1->SetTextAlign(32);
leg1->SetTextColor(1);
leg1->SetTextSize(0.02);

leg1->AddEntry(nhitdue,"perfect alignment", "P");
leg1->AddEntry(nhitdue_scen1,"SurveyLAS alignment", "P");
leg1->AddEntry(nhitdue_scen2,"SurveyLASCosmics alignment", "P");
leg1->AddEntry(nhitdue_scen3,"10 pb-1 alignment", "P");
leg1->AddEntry(nhitdue_scen4,"100 pb-1 alignment", "P");
leg1->AddEntry(nhitdue_noErr,"10 pb-1 alignment; APE not used", "P");

leg1->Draw();

c1->Update();
c1->SaveAs("Fake_nhit_ttbar.eps");
c1->WaitPrimitive();

gROOT->Reset();
gROOT->Clear();

delete c1;
}
