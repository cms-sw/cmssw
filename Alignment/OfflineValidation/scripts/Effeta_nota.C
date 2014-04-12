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
gStyle->SetTitleSize(0.046,"xy");
gStyle->SetLabelSize(0.052,"xy");
// gStyle->SetTitleFillColor(0);
gStyle->SetHistFillStyle(1001);
gStyle->SetHistFillColor(0);
gStyle->SetHistLineStyle(1);
gStyle->SetHistLineWidth(2);
gStyle->SetHistLineColor(2);
gStyle->SetTitleXOffset(1.1);
gStyle->SetTitleYOffset(1.15);
gStyle->SetOptStat(1110);
gStyle->SetOptStat(kFALSE);
gStyle->SetOptFit(0111);
gStyle->SetStatH(0.1); 


TCanvas *c1 = new TCanvas("c1","c1",129,17,926,703);
c1->SetBorderSize(2);
c1->SetFrameFillColor(0);
c1->SetLogy(0);
c1->SetGrid(1); 

c1->cd(); 

//TFile f0("../../singlemu_310607/MisalignmentIdeal131.root");  
//TFile f0("../../Misalignment_scenarioIdeal_singlemu131.root");
//TTree *MyTree=Tracks;

TFile f0("ValidationMisalignedTracker_singlemu100_merged.root");
TTree *MyTree=EffTracks;

TFile f1("../../SurveyLAS/singlemu/Misalignment_SurveyLASOnlyScenario_refitter_singlemu.root");
TTree *MyTree2=Tracks;

TFile f2("Misalignment_SurveyLASOnlyScenario_refitter_zmumu_singlemuSurveyLASCosmics.root");
TTree *MyTree3=Tracks;

TFile f3("../../singlemu_310607/Misalignment10.root");
TTree *MyTree4=Tracks;

TFile f4("../../singlemu_310607/Misalignment100.root");
TTree *MyTree5=Tracks;

TFile f5("../../singlemu_310607/Misalignment_scenario10_refitter_singlemu_noape.root");
TTree *MyTree6=Tracks;

////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA ALIGNED
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero = new TH1F("etazero","eta zero",80,0.,2.5); 
TH1F *etauno = new TH1F("etauno","eta uno",80,0.,2.5); 

MyTree->Project("etazero","abs(eta)");
MyTree->Project("etauno","abs(eta)","eff==1");
TH1F *Effeta = etazero->Clone("Efficiency vs #eta");

Effeta->Reset();
Effeta->Divide(etauno,etazero,1,1); 
Effeta->Sumw2();

float MC_bin=0.,Eff_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = etazero->GetBinContent(k);
  Eff_bin = Effeta->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Eff_bin*(1.-Eff_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  Effeta->SetBinError(k,err);
  MC_bin=0.;
  Eff_bin=0.;
  err=0.;
}

Effeta->SetMarkerStyle(20);
Effeta->SetMarkerColor(2);
Effeta->SetMarkerSize(0.9);
Effeta->SetLineColor(1);
Effeta->SetLineWidth(1);
Effeta->Draw("P"); 

c1->Update();
c1->WaitPrimitive();

////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA SCEN 1
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero_scen1 = new TH1F("etazero_scen1","eta zero",80,0.,2.5); 
TH1F *etauno_scen1 = new TH1F("etauno_scen1","eta uno",80,0.,2.5); 

MyTree2->Project("etazero_scen1","abs(eta)");
MyTree2->Project("etauno_scen1","abs(eta)","eff==1");
TH1F *Effeta_scen1 = etazero_scen1->Clone("Efficiency vs #eta");

Effeta_scen1->Reset();
Effeta_scen1->Divide(etauno_scen1,etazero_scen1,1,1); 
Effeta_scen1->Sumw2();

float MCeta_bin=0.,Eff_bin=0.,err=0.;

for (int k=1; k<81; k++){
  MC_bin = etazero_scen1->GetBinContent(k);
  Eff_bin = Effeta_scen1->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Eff_bin*(1.-Eff_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
  }
  Effeta_scen1->SetBinError(k,err);
  
  MC_bin=0.;
  Eff_bin=0.;
  err=0.;
}

Effeta_scen1->SetMarkerStyle(21);
Effeta_scen1->SetMarkerColor(3);
Effeta_scen1->SetMarkerSize(0.9);
Effeta_scen1->SetLineColor(1);
Effeta_scen1->SetLineWidth(1);
Effeta_scen1->Draw("P"); 
c1->Update();
c1->WaitPrimitive();

// // ////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

// // //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// // /// EFFICIENCIES VS ETA SCEN 2
// // //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero_scen2 = new TH1F("etazero_scen2","eta zero",80,0.,2.5); 
TH1F *etauno_scen2 = new TH1F("etauno_scen2","eta uno",80,0.,2.5); 

MyTree3->Project("etazero_scen2","abs(eta)");
MyTree3->Project("etauno_scen2","abs(eta)","eff==1");
TH1F *Effeta_scen2 = etazero_scen2->Clone("Efficiency vs #eta");

Effeta_scen2->Reset();
Effeta_scen2->Divide(etauno_scen2,etazero_scen2,1,1); 
Effeta_scen2->Sumw2();

float MC_bin=0.,Eff_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = etazero_scen2->GetBinContent(k);
  Eff_bin = Effeta_scen2->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Eff_bin*(1.-Eff_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
  }
  Effeta_scen2->SetBinError(k,err);
  
  MC_bin=0.;
  Eff_bin=0.;
  err=0.;
}

Effeta_scen2->SetMarkerStyle(22);
Effeta_scen2->SetMarkerColor(4);
Effeta_scen2->SetMarkerSize(0.9);
Effeta_scen2->SetLineColor(1);
Effeta_scen2->SetLineWidth(1);
Effeta_scen2->Draw("P"); 
c1->Update();
//c1->WaitPrimitive();

// // ////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

// // //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// // /// EFFICIENCIES VS ETA SCEN 2
// // //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero_scen3 = new TH1F("etazero_scen3","eta zero",80,0.,2.5); 
TH1F *etauno_scen3 = new TH1F("etauno_scen3","eta uno",80,0.,2.5); 

MyTree4->Project("etazero_scen3","abs(eta)");
MyTree4->Project("etauno_scen3","abs(eta)","eff==1");
TH1F *Effeta_scen3 = etazero_scen3->Clone("Efficiency vs #eta");

Effeta_scen3->Reset();
Effeta_scen3->Divide(etauno_scen3,etazero_scen3,1,1); 
Effeta_scen3->Sumw2();

float MC_bin=0.,Eff_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = etazero_scen3->GetBinContent(k);
  Eff_bin = Effeta_scen3->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Eff_bin*(1.-Eff_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
  }
  Effeta_scen3->SetBinError(k,err);
  
  MC_bin=0.;
  Eff_bin=0.;
  err=0.;
}

Effeta_scen3->SetMarkerStyle(23);
Effeta_scen3->SetMarkerColor(5);
Effeta_scen3->SetMarkerSize(0.9);
Effeta_scen3->SetLineColor(1);
Effeta_scen3->SetLineWidth(1);
Effeta_scen3->Draw("P"); 
c1->Update();
//c1->WaitPrimitive();


// // //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// // /// EFFICIENCIES VS ETA SCEN 2
// // //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *etazero_scen4 = new TH1F("etazero_scen4","eta zero",80,0.,2.5); 
TH1F *etauno_scen4 = new TH1F("etauno_scen4","eta uno",80,0.,2.5); 

MyTree5->Project("etazero_scen4","abs(eta)");
MyTree5->Project("etauno_scen4","abs(eta)","eff==1");
TH1F *Effeta_scen4 = etazero_scen4->Clone("Efficiency vs #eta");

Effeta_scen4->Reset();
Effeta_scen4->Divide(etauno_scen4,etazero_scen4,1,1); 
Effeta_scen4->Sumw2();

float MC_bin=0.,Eff_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = etazero_scen4->GetBinContent(k);
  Eff_bin = Effeta_scen4->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Eff_bin*(1.-Eff_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
  }
  Effeta_scen4->SetBinError(k,err);
  
  MC_bin=0.;
  Eff_bin=0.;
  err=0.;
}

Effeta_scen4->SetMarkerStyle(24);
Effeta_scen4->SetMarkerColor(6);
Effeta_scen4->SetMarkerSize(0.9);
Effeta_scen4->SetLineColor(1);
Effeta_scen4->SetLineWidth(1);
Effeta_scen4->Draw("P"); 
c1->Update();
//c1->WaitPrimitive();

// /////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS ETA SCEN 1 APE = 0
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

 TH1F *etazero_scen1_noErr = new TH1F("etazero_scen1_noErr","eta zero",80,0.,2.5); 
 TH1F *etauno_scen1_noErr = new TH1F("etauno_scen1_noErr","eta uno",80,0.,2.5); 

 MyTree6->Project("etazero_scen1_noErr","abs(eta)");
 MyTree6->Project("etauno_scen1_noErr","abs(eta)","eff==1");
 TH1F *Effeta_scen1_noErr = etazero_scen1_noErr->Clone("Efficiency vs #eta");

 Effeta_scen1_noErr->Reset();
 Effeta_scen1_noErr->Divide(etauno_scen1_noErr,etazero_scen1_noErr,1,1); 
 Effeta_scen1_noErr->Sumw2();

 float MC_bin=0.,Eff_bin=0.,err=0.;
 for (int k=1; k<81; k++){
   MC_bin = etazero_scen1_noErr->GetBinContent(k);
   Eff_bin = Effeta_scen1_noErr->GetBinContent(k);
   if (MC_bin != 0.) {
     err=Eff_bin*(1.-Eff_bin)/MC_bin;
     if (err >0) {
       err=sqrt(err);
     }      
     else {
       err=0.0001;  
     }
   }
   Effeta_scen1_noErr->SetBinError(k,err);
  
   MC_bin=0.;
   Eff_bin=0.;
   err=0.;
 }

 Effeta_scen1_noErr->SetMarkerStyle(26);
 Effeta_scen1_noErr->SetMarkerColor(7);
 Effeta_scen1_noErr->SetMarkerSize(0.9);
 Effeta_scen1_noErr->SetLineColor(1);
 Effeta_scen1_noErr->SetLineWidth(1);
 Effeta_scen1_noErr->Draw("P"); 
 c1->Update();

// /////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// c1->SetGrid(1,1);

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA STACKED
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
hframe = new TH2F("hframe","Global Efficiency vs #eta",80,0.,2.5,00,0,1.05);
hframe->SetXTitle("#eta");
hframe->SetYTitle("Global Efficiency");
hframe->Draw();
Effeta_scen1_noErr->Draw("same");
Effeta->Draw("same");
Effeta_scen1_noErr->SetTitle("Global Efficiency vs #eta");
Effeta_scen1_noErr->SetXTitle("#eta");
Effeta_scen1_noErr->SetYTitle("Global Efficiency");
Effeta_scen1->Draw("same");
Effeta_scen2->Draw("same");
Effeta_scen3->Draw("same");
Effeta_scen4->Draw("same");

//Effeta_scen1_noErr->Draw();
//Effeta->Draw("same");
//Effeta_scen1_noErr->SetTitle("Global Efficiency vs #eta");
//Effeta_scen1_noErr->SetXTitle("#eta");
//Effeta_scen1_noErr->SetYTitle("Global Efficiency");
//Effeta_scen1->Draw("same");
//Effeta_scen2->Draw("same");
//Effeta_scen3->Draw("same");
//Effeta_scen4->Draw("same");


TLegend *leg1 = new TLegend(0.52,0.11,0.87,0.36); 
leg1->SetTextAlign(32);
leg1->SetTextColor(1);
leg1->SetTextSize(0.033);
leg1->SetFillColor(0);

leg1->AddEntry(Effeta,"perfect ", "P");
leg1->AddEntry(Effeta_scen1,"SurveyLAS ", "P");
leg1->AddEntry(Effeta_scen2,"SurveyLASCosmics ", "P");
leg1->AddEntry(Effeta_scen3,"10 pb^{-1} ", "P");
leg1->AddEntry(Effeta_scen4,"100 pb^{-1} ", "P");
leg1->AddEntry(Effeta_scen1_noErr,"10 pb^{-1}; APE not used", "P");

leg1->Draw();

c1->Update();
//c1->WaitPrimitive();

c1->SaveAs("Eff_eta_nota.eps");
c1->SaveAs("Eff_eta_nota.gif");
c1->SaveAs("Eff_eta_nota.pdf");

gROOT->Reset();
gROOT->Clear();

delete c1;
}
