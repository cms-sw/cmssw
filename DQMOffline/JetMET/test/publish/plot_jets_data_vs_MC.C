//-------------------------------
// Usage: .L plot_jets_data_vs_MC.C
//        plot_METDQM("DQM_V0001_R000123575__JetMET__CMSSW_3_3_4__Harvesting.root",
//                    "DQM_reference.root",
//                    123575,
//                    "path-to-web-space",
//                    "png");
//-------------------------------
#include <stdio.h>
#include <stdlib.h>

int plot_jets_data_vs_MC(std::string filename, std::string reffile, int run, std::string outdir, std::string imgformat="png") {

  //-------------------------------

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetFrameLineWidth(2);
  gStyle->SetPadColor(0);
  gStyle->SetTitleFillColor(0);
  gStyle->SetStatColor(0);
  
  gStyle->SetOptStat(111110);
  gStyle->SetOptFit(1100);
  
  gStyle->SetStatX(0.92);
  gStyle->SetStatY(0.86);
  gStyle->SetStatW(0.60);
  gStyle->SetStatH(0.20);

  gStyle->SetTitleX(0.15);
  gStyle->SetTitleY(0.98);
  gStyle->SetTitleW(0.5);
  gStyle->SetTitleH(0.06);

  //-------------------------------

  TFile *f1, *f2, *h;

  TCanvas* Pt = new TCanvas("Pt", "Pt", 1200, 900);
  TCanvas* Eta = new TCanvas("Eta", "Eta", 1200, 900);
  TCanvas* Phi = new TCanvas("Phi", "Phi", 1200, 900);
  TCanvas* Mass = new TCanvas("Mass", "Mass", 1200, 900);
  TCanvas* resEMF = new TCanvas("resEMF", "resEMF", 1200, 900);
  TCanvas* fHPD = new TCanvas("fHPD", "fHPD", 1200, 900);
  TCanvas* N90Hits = new TCanvas("N90Hits", "N90Hits", 1200, 900);
  TCanvas* Constituents = new TCanvas("Constituents", "Constituents", 1200, 900);

  //-----
  char cprefix[3000];
  sprintf(cprefix,"DQMData/Run %d/JetMET/Run summary",run);
  printf("%s\n",cprefix);
  char cprefixRef[3000];
  sprintf(cprefixRef,"DQMData/Run 1/JetMET/Run summary");
  printf("%s\n",cprefixRef);
  char ctitle[3000];
  char ctitleRef[3000];
  char cjpgname[3000];
  //-----

  Pt->Divide(1,1);
  Eta->Divide(1,1);
  Phi->Divide(1,1);
  Mass->Divide(1,1);
  resEMF->Divide(1,1);
  fHPD->Divide(1,1);
  N90Hits->Divide(1,1);
  Constituents->Divide(1,1);

  std::cout << filename << std::endl;
  f1 = new TFile(filename.c_str());
  f2 = new TFile(reffile.c_str());
  
  h = new TFile( "result.root", "RECREATE");

  //
  sprintf(ctitle,"%s/Jet/CleanedAntiKtJets/Pt",cprefix);
  sprintf(ctitleRef,"%s/Jet/AntiKtJets/Pt",cprefixRef);
  TH1D *hDpT = f1->Get(ctitle);
  TH1D *hMpT = f2->Get(ctitleRef);
  //
  sprintf(ctitle,"%s/Jet/CleanedAntiKtJets/Eta",cprefix);
  sprintf(ctitleRef,"%s/Jet/AntiKtJets/Eta",cprefixRef);
  TH1D *hDEta = f1->Get(ctitle);
  TH1D *hMEta = f2->Get(ctitleRef);
  //
  sprintf(ctitle,"%s/Jet/CleanedAntiKtJets/Phi",cprefix);
  sprintf(ctitleRef,"%s/Jet/AntiKtJets/Phi",cprefixRef);
  TH1D *hDPhi = f1->Get(ctitle);
  TH1D *hMPhi = f2->Get(ctitleRef);
  //
  sprintf(ctitle,"%s/Jet/CleanedAntiKtJets/Mass",cprefix);
  sprintf(ctitleRef,"%s/Jet/AntiKtJets/Mass",cprefixRef);
  TH1D *hDMass = f1->Get(ctitle);
  TH1D *hMMass = f2->Get(ctitleRef);
  //
  sprintf(ctitle,"%s/Jet/CleanedAntiKtJets/resEMF",cprefix);
  sprintf(ctitleRef,"%s/Jet/AntiKtJets/resEMF",cprefixRef);
  TH1D *hDresEMF = f1->Get(ctitle);
  TH1D *hMresEMF = f2->Get(ctitleRef);
  //
  sprintf(ctitle,"%s/Jet/CleanedAntiKtJets/fHPD",cprefix);
  sprintf(ctitleRef,"%s/Jet/AntiKtJets/fHPD",cprefixRef);
  TH1D *hDfHPD = f1->Get(ctitle);
  TH1D *hMfHPD = f2->Get(ctitleRef);
  //
  sprintf(ctitle,"%s/Jet/CleanedAntiKtJets/N90Hits",cprefix);
  sprintf(ctitleRef,"%s/Jet/AntiKtJets/N90Hits",cprefixRef);
  TH1D *hDN90Hits = f1->Get(ctitle);
  TH1D *hMN90Hits = f2->Get(ctitleRef);
  //
  sprintf(ctitle,"%s/Jet/CleanedAntiKtJets/Constituents",cprefix);
  sprintf(ctitleRef,"%s/Jet/AntiKtJets/Constituents",cprefixRef);
  TH1D *hDConstituents = f1->Get(ctitle);
  TH1D *hMConstituents = f2->Get(ctitleRef);
  //

  h->Write();

  //---------------
  Pt->cd(1);
  hMpT->SetLineColor(2);
  hMpT->SetFillColor(2);
  hMpT->SetLineWidth(3);
  hDpT->SetLineWidth(3);
  hDpT->SetMarkerStyle(20);
  if (hMpT->GetEntries()>0) 
    hMpT->Scale((hDpT->GetEntries())/(hMpT->GetEntries()));
  else 
    hMpT->Scale(1);
  hMpT->SetStats(kFALSE);

  THStack * hpT = new THStack( "hs", "jet pT");

  hpT->Add( hMpT, "hist");
  hpT->Add( hDpT, "e");

  hpT->Draw("nostack");
//  if (hDpT->GetMaximum()>hMpT->GetMaximum()) { hDpT->Draw(""); hMpT->Draw("Sames");}
//  else if { hMpT->Draw(""); hDpT->Draw("sames");}

  //---------------
  Eta->cd(1);
  hMEta->SetLineColor(2);
  hMEta->SetFillColor(2);
  hMEta->SetLineWidth(3);
  hDEta->SetLineWidth(3);
  hDEta->SetMarkerStyle(20);
  if (hMEta->GetEntries()>0) 
    hMEta->Scale((hDEta->GetEntries())/(hMEta->GetEntries()));
  else 
    hMEta->Scale(1)
  hMEta->SetStats(kFALSE);

  THStack * hEta = new THStack( "hs", "jet eta");

  hEta->Add( hMEta, "hist");
  hEta->Add( hDEta, "e"); 
  hEta->Draw("nostack");

  //---------------
  Phi->cd(1);
  hMPhi->SetLineColor(2);
  hMPhi->SetFillColor(2);
  hMPhi->SetLineWidth(3);
  hDPhi->SetLineWidth(3);
  hDPhi->SetMarkerStyle(20);
  if (hMPhi->GetEntries()>0) 
    hMPhi->Scale((hDPhi->GetEntries())/(hMPhi->GetEntries()));
  else 
    hMPhi->Scale(1)
  hMPhi->SetStats(kFALSE);

  THStack * hPhi = new THStack( "hs", "jet phi");

  hPhi->Add( hMPhi, "hist");
  hPhi->Add( hDPhi, "e");

  hPhi->Draw("nostack");

  //---------------
  Mass->cd(1);
  hMMass->SetLineColor(2);
  hMMass->SetFillColor(2);
  hMMass->SetLineWidth(3);
  hDMass->SetLineWidth(3);
  hDMass->SetMarkerStyle(20);
  if (hMMass->GetEntries()>0) 
    hMMass->Scale((hDMass->GetEntries())/(hMMass->GetEntries()));
  else 
    hMMass->Scale(1)
  hMMass->SetStats(kFALSE);

  THStack * hMass = new THStack( "hs", "jet mass");

  hMass->Add( hMMass, "hist");
  hMass->Add( hDMass, "e");

  hMass->Draw("nostack");

  //---------------
  resEMF->cd(1);
  hMresEMF->SetLineColor(2);
  hMresEMF->SetFillColor(2);
  hMresEMF->SetLineWidth(3);
  hDresEMF->SetLineWidth(3);
  hDresEMF->SetMarkerStyle(20);
  if (hMresEMF->GetEntries()>0) 
    hMresEMF->Scale((hDresEMF->GetEntries())/(hMresEMF->GetEntries()));
  else 
    hMresEMF->Scale(1)
  hMresEMF->SetStats(kFALSE);

  THStack * hresEMF = new THStack( "hs", "restricted EMF");

  hresEMF->Add( hMresEMF, "hist");
  hresEMF->Add( hDresEMF, "e");

  hresEMF->Draw("nostack");

  //---------------
  fHPD->cd(1);
  hMfHPD->SetLineColor(2);
  hMfHPD->SetFillColor(2);
  hMfHPD->SetLineWidth(3);
  hDfHPD->SetLineWidth(3);
  hDfHPD->SetMarkerStyle(20);
  if (hMfHPD->GetEntries()>0) 
    hMfHPD->Scale((hDfHPD->GetEntries())/(hMfHPD->GetEntries()));
  else 
    hMfHPD->Scale(1)
  hMfHPD->SetStats(kFALSE);

  THStack * hfHPD = new THStack( "hs", "fHPD");

  hfHPD->Add( hMfHPD, "hist");
  hfHPD->Add( hDfHPD, "e");

  hfHPD->Draw("nostack");

  //---------------
  N90Hits->cd(1);
  hMN90Hits->SetLineColor(2);
  hMN90Hits->SetFillColor(2);
  hMN90Hits->SetLineWidth(3);
  hDN90Hits->SetLineWidth(3);
  hDN90Hits->SetMarkerStyle(20);
  if (hMN90Hits->GetEntries()>0) 
    hMN90Hits->Scale((hDN90Hits->GetEntries())/(hMN90Hits->GetEntries()));
  else 
    hMN90Hits->Scale(1)
  hMN90Hits->SetStats(kFALSE);

  THStack * hN90Hits = new THStack( "hs", "N90Hits");

  hN90Hits->Add( hMN90Hits, "hist");
  hN90Hits->Add( hDN90Hits, "e");

  hN90Hits->Draw("nostack");

  //---------------
  Constituents->cd(1);
  hMConstituents->SetLineColor(2);
  hMConstituents->SetFillColor(2);
  hMConstituents->SetLineWidth(3);
  hDConstituents->SetLineWidth(3);
  hDConstituents->SetMarkerStyle(20);
  if (hMConstituents->GetEntries()>0) 
    hMConstituents->Scale((hDConstituents->GetEntries())/(hMConstituents->GetEntries()));
  else 
    hMConstituents->Scale(1)
  hMConstituents->SetStats(kFALSE);

  THStack * hConstituents = new THStack( "hs", "# of Constituents");

  hConstituents->Add( hMConstituents, "hist");
  hConstituents->Add( hDConstituents, "e");

  hConstituents->Draw("nostack");

  //---------------
  Pt->Update();
  Eta->Update();
  Phi->Update();
  Mass->Update();
  resEMF->Update();
  fHPD->Update();
  N90Hits->Update();
  Constituents->Update();

  //---------------
  sprintf(cjpgname,"%s%d/JetDQM/CaloJetAntiKt/pt.%s",outdir.c_str(),run,imgformat.c_str());
  Pt->SaveAs(cjpgname);
  sprintf(cjpgname,"%s%d/JetDQM/CaloJetAntiKt/Eta.%s",outdir.c_str(),run,imgformat.c_str());
  Eta->SaveAs(cjpgname);
  sprintf(cjpgname,"%s%d/JetDQM/CaloJetAntiKt/Phi.%s",outdir.c_str(),run,imgformat.c_str());
  Phi->SaveAs(cjpgname);
  sprintf(cjpgname,"%s%d/JetDQM/CaloJetAntiKt/Mass.%s",outdir.c_str(),run,imgformat.c_str());
  Mass->SaveAs(cjpgname);
  sprintf(cjpgname,"%s%d/JetDQM/CaloJetAntiKt/resEMF.%s",outdir.c_str(),run,imgformat.c_str());
  resEMF->SaveAs(cjpgname);
  sprintf(cjpgname,"%s%d/JetDQM/CaloJetAntiKt/fHPD.%s",outdir.c_str(),run,imgformat.c_str());
  fHPD->SaveAs(cjpgname);
  sprintf(cjpgname,"%s%d/JetDQM/CaloJetAntiKt/N90Hits.%s",outdir.c_str(),run,imgformat.c_str());
  N90Hits->SaveAs(cjpgname);
  sprintf(cjpgname,"%s%d/JetDQM/CaloJetAntiKt/Constituents.%s",outdir.c_str(),run,imgformat.c_str());
  Constituents->SaveAs(cjpgname);

  h->Close();

}

