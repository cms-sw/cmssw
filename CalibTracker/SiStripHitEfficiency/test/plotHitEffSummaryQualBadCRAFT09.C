////////////////////////////
// Script to plot sensor efficiency layer by layer
// The input files are the output from CalibTracker/SiStripHitEfficiency/src/HitRes.cc
// Usage in root: root> .x plotHitEffSummaryQualBadCRAFT09.C
// Original Author:  Keith Ulmer  University of Colorado 
//                   keith.ulmer@colorado.edu
////////////////////////////

{
#include <vector>
#include <tdrstyle.C>
  
  gROOT->Reset();
  setTDRStyle();
  
  int nLayers = 21;
  float RatioNoBad[nLayers+1];
  float errRatio[nLayers+1];
  float RatioAll[nLayers+1];
  float errRatioAll[nLayers+1];
  float Mod[nLayers+1];
  float ModErr[nLayers+1];
  int cont[nLayers+1];

  TH1F * found = new TH1F("found","found",nLayers+1,0,nLayers+1);
  TH1F * all = new TH1F("all","all",nLayers+1,0,nLayers+1);

  float SNoBad[nLayers+1],BNoBad[nLayers+1], AllNoBad[nLayers+1];
  float SAll[nLayers+1],BAll[nLayers+1];
  
  SNoBad[0] = 0;
  BNoBad[0] = 6;
  SAll[0] = 0;
  BAll[0] = 6;
  AllNoBad[0] = 6;
  
  for (Long_t i=1; i< nLayers+1; ++i) {
    if (i==10) i++;
    TString file = TString("HitEff_V11_69912_layer") + i;
    TFile* f = TFile::Open(file+".root");
    f->cd("anEff");
    traj->Draw("ModIsBad","SiStripQualBad==0&(Id<402674821||Id>470078964||(Id<470048885&Id>402674977))","goff");
    //traj->Draw("ModIsBad","SiStripQualBad==0","goff");
    //exclude bad modules from TID 2: 402674821-402674977
    // and TEC 3: 470048885-470078964
    SNoBad[i] =  htemp->GetBinContent(1);
    if (i==13) SNoBad[i] = htemp->GetBinContent(3); //not sure why the entries are in bin 3 for layer 13?
    BNoBad[i] =  htemp->GetBinContent(2);
    cout << "filling for layer " << i << " s = " << htemp->GetBinContent(1) << " b = " << htemp->GetBinContent(2) << endl;
    if ((SNoBad[i]+BNoBad[i]) > 5) {
      AllNoBad[i] = (SNoBad[i]+BNoBad[i]);
      float ratio = (SNoBad[i]*1. / (SNoBad[i]+BNoBad[i])*1.);
      RatioNoBad[i] = ratio;
      float deno = (SNoBad[i] + BNoBad[i]) * (SNoBad[i] + BNoBad[i]) * (SNoBad[i] + BNoBad[i]);
      errRatio[i] = sqrt( (SNoBad[i]*BNoBad[i]) / deno*1.);
    } else {
      RatioNoBad[i] = -1.0;
      errRatio[i] = 0.0;
    }
    
    cout << i << " SNoBad " << SNoBad[i] << " BNoBad " << BNoBad[i] << " ratio "  << ratio*100. <<  " +o- " << errRatio[i]*100. << endl;
    Mod[i] = i;
    ModErr[i] = 0.;
    
    bool isTEC = false;
    if (i>13) isTEC = true;
    TString cut;
    if (isTEC) cut = TString("abs(TrajLocY)>2");
    else  cut = TString("");
    traj->Draw("ModIsBad>>htemp2",cut,"goff");

    //traj->Draw("ModIsBad>>htemp2","","goff");
    SAll[i] =  htemp2->GetBinContent(1);
    BAll[i] =  htemp2->GetBinContent(2);
    if ((SAll[i]+BAll[i]) > 5) {
      float ratio = (SAll[i]*1. / (SAll[i]+BAll[i])*1.);
      RatioAll[i] = ratio;
      float deno = (SAll[i] + BAll[i]) * (SAll[i] + BAll[i]) * (SAll[i] + BAll[i]);
      errRatioAll[i] = sqrt( (SAll[i]*BAll[i]) / deno*1.);
      found->SetBinContent(i,SAll[i]);
      all->SetBinContent(i,SAll[i]+BAll[i]);
    } else {
      RatioAll[i] = -1.0;
      errRatioAll[i] = 0.0;
      found->SetBinContent(i,0);
      all->SetBinContent(i,10);
    }

    cout << i << " SAll " << SAll[i] << " BAll " << BAll[i] << " ratio "  << ratio*100. <<  " +o- " << errRatioAll[i]*100. << endl;

  }

  TCanvas *c7 =new TCanvas("c7"," test ",10,10,800,600);
  c7->SetFillColor(0);
  c7->SetGrid();

  found->Sumw2();
  all->Sumw2();

  gr = new TGraphAsymmErrors(nLayers+1);
  gr->BayesDivide(found,all); 

  for(int j = 0; j<nLayers+1; j++){
    gr->SetPointError(j, 0., 0., gr->GetErrorYlow(j),gr->GetErrorYhigh(j) );
  }
  
  gr->GetXaxis()->SetLimits(0,nLayers);
  gr->SetMarkerColor(1);
  gr->SetMarkerSize(1.2);
  gr->SetLineColor(1);
  gr->SetLineWidth(4);
  gr->SetMarkerStyle(21);
  gr->SetMinimum(0.89);
  gr->SetMaximum(1.005);
  gr->GetYaxis()->SetTitle("Uncorrected efficiency");
  gr->GetXaxis()->SetTitle("");


  for ( int j=1; j<nLayers+1; j++) {
    if (j==10) j++;
    TString label;
    if (j<5) {
      label = TString("TIB ")+j;
    } else if (j>4&&j<11) {
      label = TString("TOB ")+(j-4);
    } else if (j>10&&j<14) {
      label = TString("TID ")+(j-10);
    } else if (j>13) {
      label = TString("TEC ")+(j-13);
    }
    gr->GetXaxis()->SetBinLabel((j*100)/(nLayers)-2,label);
  }
  gr->Draw("AP");

  TPaveText *pt = new TPaveText(0.3,0.35,0.5,0.45,"blNDC");
  pt->SetBorderSize(0);
  pt->SetFillColor(0);
  TText *text = pt->AddText("CMS 2008");
  pt->Draw("same");
  

  c7->SaveAs("HitEffSummary69912QualBad_TECcut.png");
  c7->SaveAs("HitEffSummary69912QualBad_TECcut.eps");
  c7->SaveAs("HitEffSummary69912QualBad_TECcut.pdf");
}
