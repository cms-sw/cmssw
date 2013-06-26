void Plot_mass() {

  gStyle->SetOptStat ("111111");
  gStyle->SetOptFit (1);
  
  TFile * A = new TFile ("./1_testYJM_SM_f4r4.root");
  A->cd();
  TH1F * Mu = dynamic_cast<TH1F*> (A->Get("hRecBestZ_Mass_fine")); 
  TH1F * MuZ = dynamic_cast<TH1F*> (A->Get("hRecBestZ_Mass")); 
  TH1F * Mup = dynamic_cast<TH1F*> (A->Get("Mass_fine_P"));
  TH1F * MuZp = dynamic_cast<TH1F*> (A->Get("Mass_P"));
  TProfile * UL_pt = dynamic_cast<TProfile*> (A->Get("hLikeVSMu_LikelihoodVSPt_prof"));
  TProfile * UL_phi = dynamic_cast<TProfile*> (A->Get("hLikeVSMu_LikelihoodVSPhi_prof"));
  TProfile * UL_eta = dynamic_cast<TProfile*> (A->Get("hLikeVSMu_LikelihoodVSEta_prof"));
  TProfile * UR_pt = dynamic_cast<TProfile*> (A->Get("hResVSMu_ResolutionVSPt_prof"));
  TProfile * UR_phi = dynamic_cast<TProfile*> (A->Get("hResVSMu_ResolutionVSPhi_prof"));
  TProfile * UR_eta = dynamic_cast<TProfile*> (A->Get("hResVSMu_ResolutionVSEta_prof"));

  TFile * B = new TFile ("./4_testYJM_SM_f4r4.root");
  B->cd();
  TH1F * Mc = dynamic_cast<TH1F*> (B->Get("hRecBestZ_Mass_fine")); 
  TH1F * McZ = dynamic_cast<TH1F*> (B->Get("hRecBestZ_Mass")); 
  TH1F * Mcp = dynamic_cast<TH1F*> (B->Get("Mass_fine_P"));
  TH1F * McZp = dynamic_cast<TH1F*> (B->Get("Mass_P"));
  TProfile * CL_pt = dynamic_cast<TProfile*> (B->Get("hLikeVSMu_LikelihoodVSPt_prof"));
  TProfile * CL_phi = dynamic_cast<TProfile*> (B->Get("hLikeVSMu_LikelihoodVSPhi_prof"));
  TProfile * CL_eta = dynamic_cast<TProfile*> (B->Get("hLikeVSMu_LikelihoodVSEta_prof"));
  TProfile * CR_pt = dynamic_cast<TProfile*> (B->Get("hResVSMu_ResolutionVSPt_prof"));
  TProfile * CR_phi = dynamic_cast<TProfile*> (B->Get("hResVSMu_ResolutionVSPhi_prof"));
  TProfile * CR_eta = dynamic_cast<TProfile*> (B->Get("hResVSMu_ResolutionVSEta_prof"));

  double ResHalfWidth[6] = {20., 0.5, 0.5, 0.5, 0.2, 0.2};
  double ResMass[6] = {90.986, 10.3552, 10.0233, 9.4603, 3.68609, 3.0969};

  TCanvas * Allres = new TCanvas ("Allres", "All resonances", 600, 600);
  Allres->Divide (2,3);
  
  for (int ires=0; ires<6; ires++) {
    Allres->cd(ires+1);
    if (ires==0) {
      McZ->SetAxisRange(ResMass[ires]-ResHalfWidth[ires], ResMass[ires]+ResHalfWidth[ires]);
      McZ->SetLineColor(kRed);
      McZp->SetNormFactor(Mc->Integral());
      McZp->SetAxisRange(ResMass[ires]-ResHalfWidth[ires], ResMass[ires]+ResHalfWidth[ires]);
      McZp->SetLineColor(kBlue);
      McZ->SetMarkerColor(kRed);
      McZ->DrawCopy("PE");
      McZp->DrawCopy("SAMEHISTO");
    } else {
      Mc->SetAxisRange(ResMass[ires]-ResHalfWidth[ires], ResMass[ires]+ResHalfWidth[ires]);
      Mc->SetLineColor(kRed);
      Mc->SetMarkerColor(kRed);
      Mcp->SetNormFactor(Mc->Integral());
      Mcp->SetAxisRange(ResMass[ires]-ResHalfWidth[ires], ResMass[ires]+ResHalfWidth[ires]);
      Mcp->SetLineColor(kBlue);
      Mc->DrawCopy("PE");
      Mcp->DrawCopy("SAMEHISTO");
    }
  }
  Allres->Print("Plot_mass_fitYJM_SM_f4r4.ps");

  TCanvas * Allres2 = new TCanvas ("Allres2", "All resonances", 600, 600);
  Allres2->Divide (2,3);
  
  for (int ires=0; ires<6; ires++) {
    Allres2->cd(ires+1);
    if (ires==0) {
      MuZ->SetAxisRange(ResMass[ires]-ResHalfWidth[ires], ResMass[ires]+ResHalfWidth[ires]);
      MuZ->SetLineColor(kBlack);
      MuZ->SetMarkerColor(kBlack);
      MuZp->SetNormFactor(Mu->Integral());
      MuZp->SetAxisRange(ResMass[ires]-ResHalfWidth[ires], ResMass[ires]+ResHalfWidth[ires]);
      MuZp->SetLineColor(kBlue);
      MuZ->DrawCopy("PE");
      MuZp->DrawCopy("SAMEHISTO");
    } else {
      Mu->SetAxisRange(ResMass[ires]-ResHalfWidth[ires], ResMass[ires]+ResHalfWidth[ires]);
      Mu->SetLineColor(kBlack);
      Mu->SetMarkerColor(kBlack);
      Mup->SetNormFactor(Mu->Integral());
      Mup->SetAxisRange(ResMass[ires]-ResHalfWidth[ires], ResMass[ires]+ResHalfWidth[ires]);
      Mup->SetLineColor(kBlue);
      Mu->DrawCopy("PE");
      Mup->DrawCopy("SAMEHISTO");
    }
  }
  Allres2->Print("Plot_mass_bef_fitYJM_SM_f4r4.ps");


  TCanvas * LR = new TCanvas ("LR", "Likelihood and resolution before and after corrections", 600, 600 );
  LR->Divide (2,3);

  LR->cd (1);
  UL_pt->SetMinimum(-8);
  UL_pt->SetMaximum(-4);
  UL_pt->Draw("");
  CL_pt->SetLineColor(kRed);
  CL_pt->Draw("SAME");
  LR->cd (2);
  UR_pt->SetMaximum(0.15);
  UR_pt->Draw("");
  CR_pt->SetLineColor(kRed);
  CR_pt->Draw("SAME");
  LR->cd (3);
  UL_eta->SetMinimum(-8);
  UL_eta->SetMaximum(-4);
  UL_eta->Draw("");
  CL_eta->SetLineColor(kRed);
  CL_eta->Draw("SAME");
  LR->cd (4);
  UR_eta->SetMaximum(0.15);
  UR_eta->Draw("");
  CR_eta->SetLineColor(kRed);
  CR_eta->Draw("SAME");
  LR->cd (5);
  UL_phi->SetMinimum(-8);
  UL_phi->SetMaximum(-4);
  UL_phi->Draw("");
  CL_phi->SetLineColor(kRed);
  CL_phi->Draw("SAME");
  LR->cd (6);
  UR_phi->SetMaximum(0.15);
  UR_phi->Draw("");
  CR_phi->SetLineColor(kRed);
  CR_phi->Draw("SAME");
  LR->Print("Plot_mass_Lik_ResYJM_SM_f4r4.ps");


  TCanvas * G = new TCanvas ("G", "Mass before and after corrections", 600, 600 );
  G->Divide (3,3);

  G->cd (1);
  Mu->SetAxisRange (2.8, 3.4);
  Mu->SetLineColor (kBlue);
  Mc->SetLineColor (kRed);
  
  Mu->DrawCopy ("HISTO");
  Mc->DrawCopy ("SAME");

  G->cd (2);
  Mu->SetAxisRange (9., 10.);
  Mc->SetAxisRange (9., 10.);
  Mu->DrawCopy ("HISTO");
  Mc->DrawCopy ("SAME"); 

  G->cd (3);
  MuZ->SetAxisRange (70., 110.);
  McZ->SetAxisRange (70., 110.);

  MuZ->DrawCopy ("HISTO");
  McZ->DrawCopy ("SAME"); 

  G->cd (4);
  Mu->SetAxisRange (2.8, 3.4);
  Mu->Fit ("gaus", "", "", 3., 3.2);
  Mu->DrawCopy();
  G->cd (7);
  Mc->SetAxisRange (2.8, 3.4);
  Mc->Fit ("gaus", "", "", 3., 3.2);
  Mc->DrawCopy();
  G->cd (5);
  Mu->SetAxisRange (9., 10.);
  Mu->Fit ("gaus", "", "", 9., 10.);
  Mu->DrawCopy();
  G->cd (8);
  Mc->SetAxisRange (9., 10.);
  Mc->Fit ("gaus", "", "", 9., 10..);
  Mc->DrawCopy();
  G->cd (6);
  MuZ->SetAxisRange (70., 110.);
  MuZ->Fit ("gaus", "", "", 70., 110.);
  MuZ->DrawCopy();
  G->cd (9);
  McZ->SetAxisRange (70., 110.);
  McZ->Fit ("gaus", "", "", 70., 110.);
  McZ->DrawCopy();
  G->Print("Plot_mass_Resonances_YJM_SM_f4r4.ps");
 
  TCanvas * Spectrum = new TCanvas ("Spectrum", "Mass before and after corrections", 600, 600 );
  //Spectrum->Divide(1,2);
  Spectrum->cd(1);
  MuZ->SetAxisRange(1.,15.);
  McZ->SetAxisRange(1.,15.);
  McZ->SetLineColor(kRed);
  McZ->DrawCopy("HISTO");
  Spectrum->cd(2);
  MuZ->SetLineColor(kBlack);
  MuZ->DrawCopy("SAMEHISTO");
  Spectrum->Print("Spectrum_YJM_SM_f4r4.ps");

}
