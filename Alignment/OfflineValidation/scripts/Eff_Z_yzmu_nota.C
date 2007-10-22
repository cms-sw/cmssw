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

TFile f0("../../Z/MisalignmentIdeal.root");  
TTree *MyTree=Tracks;

TFile f1("../../SurveyLAS/zmumu/Misalignment_SurveyLASOnlyScenario_refitter_zmumu.root");
TTree *MyTree2=Tracks;

TFile f2("Misalignment_SurveyLASOnlyScenario_refitter_zmumuSurveyLASCosmics.root");
TTree *MyTree3=Tracks;

TFile f3("../../Z/Misalignment10.root");
TTree *MyTree4=Tracks;

TFile f4("../../Z/Misalignment100.root");
TTree *MyTree5=Tracks;
 
TFile f5("../../Z/Misalignment_NOAPE_2.root");
//TFile f5("../../SurveyLAS/zmumu65To100/Misalignment_SurveyLASOnlyScenario_refitter_zmumu_NOAPE.root");
TTree *MyTree6=Tracks;


////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA ALIGNED
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *yzmuzero = new TH1F("yzmuzero","yzmu zero",80,0.,2.7); 
TH1F *yzmuuno = new TH1F("yzmuuno","yzmu uno",80,0.,2.7); 

MyTree->Project("yzmuzero","yzmu","yzmu>-101.");
MyTree->Project("yzmuuno","yzmu","eff==1 && recyzmu>-101.");
TH1F *Effyzmu = yzmuzero->Clone("Z selection efficiency vs yzmu");

Effyzmu->Reset();
Effyzmu->Divide(yzmuuno,yzmuzero,1,1); 
Effyzmu->Sumw2();

float MC_bin=0.,Eff_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = yzmuzero->GetBinContent(k);
  Eff_bin = Effyzmu->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Eff_bin*(1.-Eff_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
    }
  Effyzmu->SetBinError(k,err);
  MC_bin=0.;
  Eff_bin=0.;
  err=0.;
}

Effyzmu->SetMarkerStyle(20);
Effyzmu->SetMarkerColor(2);
Effyzmu->SetMarkerSize(0.9);
Effyzmu->SetLineColor(1);
Effyzmu->SetLineWidth(1);
Effyzmu->Draw("P"); 

c1->Update();
//c1->WaitPrimitive();

////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS ETA SCEN 1
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *yzmuzero_scen1 = new TH1F("yzmuzero_scen1","yzmu zero",80,0.,2.7); 
TH1F *yzmuuno_scen1 = new TH1F("yzmuuno_scen1","yzmu uno",80,0.,2.7); 

MyTree2->Project("yzmuzero_scen1","yzmu","yzmu>-101.");
MyTree2->Project("yzmuuno_scen1","yzmu","eff==1 && recyzmu>-101.");
TH1F *Effyzmu_scen1 = yzmuzero_scen1->Clone("Z selection efficiency vs yzmu");

Effyzmu_scen1->Reset();
Effyzmu_scen1->Divide(yzmuuno_scen1,yzmuzero_scen1,1,1); 
Effyzmu_scen1->Sumw2();

float MCyzmu_bin=0.,Eff_bin=0.,err=0.;

for (int k=1; k<81; k++){
  MC_bin = yzmuzero_scen1->GetBinContent(k);
  Eff_bin = Effyzmu_scen1->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Eff_bin*(1.-Eff_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
  }
  Effyzmu_scen1->SetBinError(k,err);
  
  MC_bin=0.;
  Eff_bin=0.;
  err=0.;
}

Effyzmu_scen1->SetMarkerStyle(21);
Effyzmu_scen1->SetMarkerColor(3);
Effyzmu_scen1->SetMarkerSize(0.9);
Effyzmu_scen1->SetLineColor(1);
Effyzmu_scen1->SetLineWidth(1);
Effyzmu_scen1->Draw("P"); 
c1->Update();
//c1->WaitPrimitive();

// // ////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS YZMU SCEN 2
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *yzmuzero_scen2 = new TH1F("yzmuzero_scen2","yzmu zero",80,0.,2.7); 
TH1F *yzmuuno_scen2 = new TH1F("yzmuuno_scen2","yzmu uno",80,0.,2.7); 

MyTree3->Project("yzmuzero_scen2","yzmu","yzmu>-101.");
MyTree3->Project("yzmuuno_scen2","yzmu","eff==1 && recyzmu>-101.");
TH1F *Effyzmu_scen2 = yzmuzero_scen2->Clone("Z selection efficiency vs yzmu");

Effyzmu_scen2->Reset();
Effyzmu_scen2->Divide(yzmuuno_scen2,yzmuzero_scen2,1,1); 
Effyzmu_scen2->Sumw2();

float MC_bin=0.,Eff_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = yzmuzero_scen2->GetBinContent(k);
  Eff_bin = Effyzmu_scen2->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Eff_bin*(1.-Eff_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
  }
  Effyzmu_scen2->SetBinError(k,err);
  
  MC_bin=0.;
  Eff_bin=0.;
  err=0.;
}

Effyzmu_scen2->SetMarkerStyle(22);
Effyzmu_scen2->SetMarkerColor(4);
Effyzmu_scen2->SetMarkerSize(0.9);
Effyzmu_scen2->SetLineColor(1);
Effyzmu_scen2->SetLineWidth(1);
Effyzmu_scen2->Draw("P"); 
c1->Update();
//c1->WaitPrimitive();

// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS YZMU SCEN 2
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *yzmuzero_scen3 = new TH1F("yzmuzero_scen3","yzmu zero",80,0.,2.7); 
TH1F *yzmuuno_scen3 = new TH1F("yzmuuno_scen3","yzmu uno",80,0.,2.7); 

MyTree4->Project("yzmuzero_scen3","yzmu","yzmu>-101.");
MyTree4->Project("yzmuuno_scen3","yzmu","eff==1 && recyzmu>-101.");
TH1F *Effyzmu_scen3 = yzmuzero_scen3->Clone("Z selection efficiency vs yzmu");

Effyzmu_scen3->Reset();
Effyzmu_scen3->Divide(yzmuuno_scen3,yzmuzero_scen3,1,1); 
Effyzmu_scen3->Sumw2();

float MC_bin=0.,Eff_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = yzmuzero_scen3->GetBinContent(k);
  Eff_bin = Effyzmu_scen3->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Eff_bin*(1.-Eff_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
  }
  Effyzmu_scen3->SetBinError(k,err);
  
  MC_bin=0.;
  Eff_bin=0.;
  err=0.;
}

Effyzmu_scen3->SetMarkerStyle(23);
Effyzmu_scen3->SetMarkerColor(5);
Effyzmu_scen3->SetMarkerSize(0.9);
Effyzmu_scen3->SetLineColor(1);
Effyzmu_scen3->SetLineWidth(1);
Effyzmu_scen3->Draw("P"); 
c1->Update();
//c1->WaitPrimitive();

// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS YZMU SCEN 2
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

TH1F *yzmuzero_scen4 = new TH1F("yzmuzero_scen4","yzmu zero",80,0.,2.7); 
TH1F *yzmuuno_scen4 = new TH1F("yzmuuno_scen4","yzmu uno",80,0.,2.7); 

MyTree5->Project("yzmuzero_scen4","yzmu","yzmu>-101.");
MyTree5->Project("yzmuuno_scen4","yzmu","eff==1 && recyzmu>-101.");
TH1F *Effyzmu_scen4 = yzmuzero_scen4->Clone("Z selection efficiency vs yzmu");

Effyzmu_scen4->Reset();
Effyzmu_scen4->Divide(yzmuuno_scen4,yzmuzero_scen4,1,1); 
Effyzmu_scen4->Sumw2();

float MC_bin=0.,Eff_bin=0.,err=0.;
for (int k=1; k<81; k++){
  MC_bin = yzmuzero_scen4->GetBinContent(k);
  Eff_bin = Effyzmu_scen4->GetBinContent(k);
  if (MC_bin != 0.) {
    err=Eff_bin*(1.-Eff_bin)/MC_bin;
    if (err >0) {
      err=sqrt(err);
    }      
    else {
      err=0.0001;  
    }
  }
  Effyzmu_scen4->SetBinError(k,err);
  
  MC_bin=0.;
  Eff_bin=0.;
  err=0.;
}

Effyzmu_scen4->SetMarkerStyle(24);
Effyzmu_scen4->SetMarkerColor(6);
Effyzmu_scen4->SetMarkerSize(0.9);
Effyzmu_scen4->SetLineColor(1);
Effyzmu_scen4->SetLineWidth(1);
Effyzmu_scen4->Draw("P"); 
c1->Update();
//c1->WaitPrimitive();


// /////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// /// EFFICIENCIES VS YZMU SCEN 1 APE = 0
// //&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

 TH1F *yzmuzero_scen1_noErr = new TH1F("yzmuzero_scen1_noErr","yzmu zero",80,0.,2.7); 
 TH1F *yzmuuno_scen1_noErr = new TH1F("yzmuuno_scen1_noErr","yzmu uno",80,0.,2.7); 

 MyTree6->Project("yzmuzero_scen1_noErr","yzmu","yzmu>-101.");
 MyTree6->Project("yzmuuno_scen1_noErr","yzmu","eff==1 && recyzmu>-101.");
 TH1F *Effyzmu_scen1_noErr = yzmuzero_scen1_noErr->Clone("Efficiency vs #yzmu");

 Effyzmu_scen1_noErr->Reset();
 Effyzmu_scen1_noErr->Divide(yzmuuno_scen1_noErr,yzmuzero_scen1_noErr,1,1); 
 Effyzmu_scen1_noErr->Sumw2();

 float MC_bin=0.,Eff_bin=0.,err=0.;
 for (int k=1; k<81; k++){
   MC_bin = yzmuzero_scen1_noErr->GetBinContent(k);
   Eff_bin = Effyzmu_scen1_noErr->GetBinContent(k);
   if (MC_bin != 0.) {
     err=Eff_bin*(1.-Eff_bin)/MC_bin;
     if (err >0) {
       err=sqrt(err);
     }      
     else {
       err=0.0001;  
     }
   }
   Effyzmu_scen1_noErr->SetBinError(k,err);
  
   MC_bin=0.;
   Eff_bin=0.;
   err=0.;
 }

 Effyzmu_scen1_noErr->SetMarkerStyle(26);
 Effyzmu_scen1_noErr->SetMarkerColor(7);
 Effyzmu_scen1_noErr->SetMarkerSize(0.9);
 Effyzmu_scen1_noErr->SetLineColor(1);
 Effyzmu_scen1_noErr->SetLineWidth(1);
 Effyzmu_scen1_noErr->Draw("P"); 
 c1->Update();

// /////&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// c1->SetGrid(1,1);

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
/// EFFICIENCIES VS YZMU STACKED
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

Effyzmu->SetTitle("Z selection Efficiency vs y");
Effyzmu->SetXTitle("y");
Effyzmu->SetYTitle("Z selection Efficiency");
Effyzmu->Draw("");
Effyzmu_scen1->Draw("same");
Effyzmu_scen2->Draw("same");
Effyzmu_scen3->Draw("same");
Effyzmu_scen4->Draw("same");
Effyzmu_scen1_noErr->Draw("same");

TLegend *leg1 = new TLegend(0.45,0.11,0.8,0.22); 
leg1->SetTextAlign(32);
leg1->SetTextColor(1);
leg1->SetTextSize(0.015);

leg1->AddEntry(Effyzmu,"perfect alignment", "P");
leg1->AddEntry(Effyzmu_scen1,"SurveyLAS alignment", "P");
leg1->AddEntry(Effyzmu_scen2,"SurveyLASCosmics alignment", "P");
leg1->AddEntry(Effyzmu_scen3,"10 pb-1 alignment", "P");
leg1->AddEntry(Effyzmu_scen4,"100 pb-1 alignment", "P");
leg1->AddEntry(Effyzmu_scen1_noErr,"10 pb-1 alignment; APE not used", "P");

leg1->Draw();

c1->Update();
//c1->WaitPrimitive();

c1->SaveAs("Eff_Z_yzmu.gif");
c1->SaveAs("Eff_Z_yzmu.eps");
c1->SaveAs("Eff_Z_yzmu.pdf");

gROOT->Reset();
gROOT->Clear();

delete c1;
}
