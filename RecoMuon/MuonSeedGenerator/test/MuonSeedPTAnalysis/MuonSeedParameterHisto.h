#ifndef RecoMuon_MuonSeedParameterHisto_H
#define RecoMuon_MuonSeedParameterHisto_H

/** \class SeedParameterHisto
 *  Collection of histograms for SeedGenerator test.
 *
 * Author: S.C. Kao  - UC Riverside
 */

#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TString.h"
#include <string>
#include <iostream>


class H2DRecHit1 {
public:
 
 H2DRecHit1(std::string name_) {
    TString N1 = name_.c_str();
    name=N1;
    hcsc_dt  = new TH2F(N1+"_hcsc_dt", " CSC Seg vs DT Seg ", 20, -0.25, 9.75, 20, -0.25, 9.75);
    heta_csc = new TH2F(N1+"_heta_csc", " CSC Seg vs eta ", 180, 0.7, 2.5, 20, -0.25, 9.75);
    heta_dt  = new TH2F(N1+"_heta_dt",  " DT  Seg vs eta ", 130, 0.0, 1.3, 20, -0.25, 9.75);
    heta_mu  = new TH2F(N1+"_heta_mu",  " All Seg vs eta ", 300, 0.0, 3.0, 20, -0.25, 9.75);
    heta_mu1 = new TH2F(N1+"_heta_mu1", " All Seg vs eta from track", 300, 0.0, 3.0, 20, -0.25, 9.75);
    heta_mu2 = new TH2F(N1+"_heta_mu2", " All Seg vs eta from simseg", 300, 0.0, 3.0, 20, -0.25, 9.75);

    heta_eta = new TH2F(N1+"_heta_eta", " eta_simhits vs eta_track", 300, 0.0, 3.0, 300, 0.0, 3.0);

    heta_rh  = new TH2F(N1+"_heta_rh",  " All rechits vs eta ", 250, 0.0, 2.5, 20, -0.25, 9.75);
    heta_cscrh = new TH2F(N1+"_heta_cscrh", " csc rechits vs eta ", 250, 0.0, 2.5, 20, -0.25, 9.75);
    heta_dtrh  = new TH2F(N1+"_heta_dtrh",  " dt  rechits vs eta ", 250, 0.0, 2.5, 20, -0.25, 9.75);
    heta_trk = new TH1F(N1+"_heta_trk"," eta from track",300, 0.0, 3.0);

    heta_nSeg_seed = new TH2F(N1+"_heta_nSeg_seed",  " eta vs nSeg from rec seed", 300, 0.0, 3., 20, -0.25, 9.75);

    heta_ptLossC1 = new TH2F(N1+"_heta_ptLossC1","CSC eta vs ptLoss at layer1", 18, 0.7, 2.5, 60, 0.5, 1.1);
    heta_ptLossC2 = new TH2F(N1+"_heta_ptLossC2","CSC eta vs ptLoss at layer2", 18, 0.7, 2.5, 60, 0.5, 1.1);
    heta_ptLossC3 = new TH2F(N1+"_heta_ptLossC3","CSC eta vs ptLoss at layer3", 18, 0.7, 2.5, 60, 0.5, 1.1);
    heta_ptLossC4 = new TH2F(N1+"_heta_ptLossC4","CSC eta vs ptLoss at layer4", 18, 0.7, 2.5, 60, 0.5, 1.1);

    heta_ptLossD1 = new TH2F(N1+"_heta_ptLossD1","DT eta vs ptLoss at layer1", 13, 0., 1.3, 60, 0.5, 1.1);
    heta_ptLossD2 = new TH2F(N1+"_heta_ptLossD2","DT eta vs ptLoss at layer2", 13, 0., 1.3, 60, 0.5, 1.1);
    heta_ptLossD3 = new TH2F(N1+"_heta_ptLossD3","DT eta vs ptLoss at layer3", 13, 0., 1.3, 60, 0.5, 1.1);
    heta_ptLossD4 = new TH2F(N1+"_heta_ptLossD4","DT eta vs ptLoss at layer4", 13, 0., 1.3, 60, 0.5, 1.1);

    hpt_ptLossC2 = new TH2F(N1+"_hpt_ptLossC2","CSC pt vs ptLoss at layer2", 250, 0., 250., 80, 0.3, 1.1);
    hpt_ptLossD1 = new TH2F(N1+"_hpt_ptLossD1","DT pt vs ptLoss at layer1", 250, 0., 250., 80, 0.3, 1.1);
 } 

 H2DRecHit1(TString name_, TFile* file) {
    name=name_;
    hcsc_dt  = (TH2F *) file->Get(name+"_hcsc_dt");
    heta_csc = (TH2F *) file->Get(name+"_heta_csc");
    heta_dt  = (TH2F *) file->Get(name+"_heta_dt");
    heta_mu  = (TH2F *) file->Get(name+"_heta_mu");
    heta_mu1 = (TH2F *) file->Get(name+"_heta_mu1");
    heta_mu2 = (TH2F *) file->Get(name+"_heta_mu2");

    heta_eta = (TH2F *) file->Get(name+"_heta_eta");

    heta_rh  = (TH2F *) file->Get(name+"_heta_rh");
    heta_cscrh = (TH2F *) file->Get(name+"_heta_cscrh");
    heta_dtrh  = (TH2F *) file->Get(name+"_heta_dtrh");
    heta_trk = (TH1F *) file->Get(name+"_heta_trk");

    heta_nSeg_seed = (TH2F *) file->Get(name+"_heta_nSeg_seed"); 
 
    heta_ptLossC1 = (TH2F *) file->Get(name+"_heta_ptLossC1");
    heta_ptLossC2 = (TH2F *) file->Get(name+"_heta_ptLossC2");
    heta_ptLossC3 = (TH2F *) file->Get(name+"_heta_ptLossC3");
    heta_ptLossC4 = (TH2F *) file->Get(name+"_heta_ptLossC4");

    heta_ptLossD1 = (TH2F *) file->Get(name+"_heta_ptLossD1");
    heta_ptLossD2 = (TH2F *) file->Get(name+"_heta_ptLossD2");
    heta_ptLossD3 = (TH2F *) file->Get(name+"_heta_ptLossD3");
    heta_ptLossD4 = (TH2F *) file->Get(name+"_heta_ptLossD4");

    hpt_ptLossC2 = (TH2F *) file->Get(name+"_hpt_ptLossC2");
    hpt_ptLossD1 = (TH2F *) file->Get(name+"_hpt_ptLossD1");
 }

 /// Destructor
 virtual ~H2DRecHit1() {
    delete hcsc_dt;
    delete heta_csc;
    delete heta_dt;
    delete heta_mu;
    delete heta_mu1;
    delete heta_mu2;

    delete heta_eta;

    delete heta_rh;
    delete heta_cscrh;
    delete heta_dtrh;
    delete heta_trk;

    delete heta_nSeg_seed;

    delete heta_ptLossC1;
    delete heta_ptLossC2;
    delete heta_ptLossC3;
    delete heta_ptLossC4;

    delete heta_ptLossD1;
    delete heta_ptLossD2;
    delete heta_ptLossD3;
    delete heta_ptLossD4;
  
    delete hpt_ptLossC2;
    delete hpt_ptLossD1;
 }

 void Fill1(int csc_nu,int dt_nu,int all_nu,double eta_c, double eta_d,double eta_a, double eta_trk) {
      hcsc_dt->Fill(csc_nu,dt_nu);
      heta_csc->Fill(eta_c,csc_nu);
      heta_dt->Fill(eta_d,dt_nu);
      heta_mu->Fill(eta_a,all_nu);
      heta_mu1->Fill(eta_trk,all_nu);
      heta_trk->Fill(eta_trk);
      heta_eta->Fill(eta_a,eta_trk);
 }
 void Fill1a(double eta_a, int rh_nu, int cscrh_nu, int dtrh_nu) {
      heta_rh->Fill(eta_a,rh_nu);
      heta_cscrh->Fill(eta_a,cscrh_nu);
      heta_dtrh->Fill(eta_a,dtrh_nu);
 }
 void Fill1b(int sim_nu, double eta_sim ) {
      heta_mu2->Fill(eta_sim,sim_nu);
 }

 void Fill1c1(double eta, double ptloss) {
      heta_ptLossC1->Fill(eta,ptloss);
 }
 void Fill1c2(double eta, double ptloss, double pt) {
      heta_ptLossC2->Fill(eta,ptloss);
      hpt_ptLossC2->Fill(pt,ptloss);
 }
 void Fill1c3(double eta, double ptloss) {
      heta_ptLossC3->Fill(eta,ptloss);
 }
 void Fill1c4(double eta, double ptloss) {
      heta_ptLossC4->Fill(eta,ptloss);
 }
 void Fill1d1(double eta, double ptloss, double pt) {
      heta_ptLossD1->Fill(eta,ptloss);
      hpt_ptLossD1->Fill(pt,ptloss);
 }
 void Fill1d2(double eta, double ptloss) {
      heta_ptLossD2->Fill(eta,ptloss);
 }
 void Fill1d3(double eta, double ptloss) {
      heta_ptLossD3->Fill(eta,ptloss);
 }
 void Fill1d4(double eta, double ptloss) {
      heta_ptLossD4->Fill(eta,ptloss);
 }

 void Fill1h(int nSeg_seed, float eta_seed) {
      heta_nSeg_seed->Fill(eta_seed,nSeg_seed);
 }
 
 void Write() {
      hcsc_dt->Write();
      heta_csc->Write();
      heta_dt->Write();
      heta_mu->Write();
      heta_mu1->Write();
      heta_mu2->Write();

      heta_eta->Write();

      heta_rh->Write();
      heta_cscrh->Write();
      heta_dtrh->Write();
      heta_trk->Write();

      heta_nSeg_seed->Write();

      heta_ptLossC1->Write();
      heta_ptLossC2->Write();
      heta_ptLossC3->Write();
      heta_ptLossC4->Write();

      heta_ptLossD1->Write();
      heta_ptLossD2->Write();
      heta_ptLossD3->Write();
      heta_ptLossD4->Write();

      hpt_ptLossC2->Write();
      hpt_ptLossD1->Write();
 }

 TH2F *hcsc_dt;
 TH2F *heta_csc;
 TH2F *heta_dt;
 TH2F *heta_mu;
 TH2F *heta_mu1;
 TH2F *heta_mu2;

 TH2F *heta_eta;

 TH2F *heta_rh;
 TH2F *heta_cscrh;
 TH2F *heta_dtrh;
 TH1F *heta_trk;

 TH2F *heta_nSeg_seed;

 TH2F *heta_ptLossC1;
 TH2F *heta_ptLossC2;
 TH2F *heta_ptLossC3;
 TH2F *heta_ptLossC4;

 TH2F *heta_ptLossD1;
 TH2F *heta_ptLossD2;
 TH2F *heta_ptLossD3;
 TH2F *heta_ptLossD4;

 TH2F *hpt_ptLossC2;
 TH2F *hpt_ptLossD1;

 TString name;

};

// For all CSC
class H2DRecHit2 {
public:

 H2DRecHit2(std::string name_) {
    TString N2 = name_.c_str();
    name=N2;

    hPt = new TH1F(N2+"_hPt", " Pt of Tracks ", 100,5.,205.);
    hPa_Pt     = new TH2F(N2+"_hPa_Pt", "P vs Pt", 50, 5., 205., 100, 0., 1000.);
    hPt_CSCSeg = new TH2F(N2+"_hPt_CSCSeg", "Pt vs total CSC segments number", 40, 5., 205., 40, -0.25, 19.75);
    hst_csc_nu = new TH2F(N2+"_hst_csc_nu", "station vs csc seg number", 6, 0, 6, 20, -0.25, 9.75);
    hst_csc_nu1= new TH2F(N2+"_hst_csc_nu1","station vs csc seg number w/ > 5 hits seg", 6, 0, 6, 20, -0.25, 9.75);
    hChi2_d_Dof= new TH1F(N2+"_hChi2_d_Dof"," chi2/dof ",5000,0.,50. ); 

    eta_dphi01 = new TH2F(N2+"01_eta_dphi", "eta vs dphi01", 180, 0.9, 2.7, 2000, -0.2, 0.2);
    eta_dphi12 = new TH2F(N2+"12_eta_dphi", "eta vs dphi12", 180, 0.9, 2.7, 2000, -0.2, 0.2);
    eta_dphi13 = new TH2F(N2+"13_eta_dphi", "eta vs dphi13", 180, 0.9, 2.7, 2000, -0.2, 0.2);
    eta_dphi14 = new TH2F(N2+"14_eta_dphi", "eta vs dphi14", 180, 0.9, 2.7, 2000, -0.2, 0.2);
    eta_dphi23 = new TH2F(N2+"23_eta_dphi", "eta vs dphi23", 180, 0.9, 2.7, 2000, -0.2, 0.2);
    eta_dphi24 = new TH2F(N2+"24_eta_dphi", "eta vs dphi24", 180, 0.9, 2.7, 2000, -0.2, 0.2);
    eta_dphi34 = new TH2F(N2+"34_eta_dphi", "eta vs dphi34", 180, 0.9, 2.7, 2000, -0.2, 0.2);

    eta_rdphi01 = new TH2F(N2+"01_eta_rdphi", "eta vs rdphi01", 180, 0.9, 2.7, 2000, -0.2, 0.2);
    eta_rdphi12 = new TH2F(N2+"12_eta_rdphi", "eta vs rdphi12", 180, 0.9, 2.7, 2000, -0.2, 0.2);
    eta_rdphi13 = new TH2F(N2+"13_eta_rdphi", "eta vs rdphi13", 180, 0.9, 2.7, 2000, -0.2, 0.2);
    eta_rdphi14 = new TH2F(N2+"14_eta_rdphi", "eta vs rdphi14", 180, 0.9, 2.7, 2000, -0.2, 0.2);
    eta_rdphi23 = new TH2F(N2+"23_eta_rdphi", "eta vs rdphi23", 180, 0.9, 2.7, 2000, -0.2, 0.2);
    eta_rdphi24 = new TH2F(N2+"24_eta_rdphi", "eta vs rdphi24", 180, 0.9, 2.7, 2000, -0.2, 0.2);
    eta_rdphi34 = new TH2F(N2+"34_eta_rdphi", "eta vs rdphi34", 180, 0.9, 2.7, 2000, -0.2, 0.2);

    eta_dphiPt01 = new TH2F(N2+"01_eta_dphiPt", "eta vs dphi*Pt01", 180, 0.9, 2.7, 600, -3., 3.);
    eta_dphiPt12 = new TH2F(N2+"12_eta_dphiPt", "eta vs dphi*Pt12", 180, 0.9, 2.7, 600, -3., 3.);
    eta_dphiPt13 = new TH2F(N2+"13_eta_dphiPt", "eta vs dphi*Pt13", 180, 0.9, 2.7, 600, -3., 3.);
    eta_dphiPt14 = new TH2F(N2+"14_eta_dphiPt", "eta vs dphi*Pt14", 180, 0.9, 2.7, 600, -3., 3.);
    eta_dphiPt23 = new TH2F(N2+"23_eta_dphiPt", "eta vs dphi*Pt23", 180, 0.9, 2.7, 600, -3., 3.);
    eta_dphiPt24 = new TH2F(N2+"24_eta_dphiPt", "eta vs dphi*Pt24", 180, 0.9, 2.7, 600, -3., 3.);
    eta_dphiPt34 = new TH2F(N2+"34_eta_dphiPt", "eta vs dphi*Pt34", 180, 0.9, 2.7, 600, -3., 3.);

    eta_rdphiPt01 = new TH2F(N2+"01_eta_rdphiPt", "eta vs rdphi*Pt01", 180, 0.9, 2.7, 600, -3., 3.);
    eta_rdphiPt12 = new TH2F(N2+"12_eta_rdphiPt", "eta vs rdphi*Pt12", 180, 0.9, 2.7, 600, -3., 3.);
    eta_rdphiPt13 = new TH2F(N2+"13_eta_rdphiPt", "eta vs rdphi*Pt13", 180, 0.9, 2.7, 600, -3., 3.);
    eta_rdphiPt14 = new TH2F(N2+"14_eta_rdphiPt", "eta vs rdphi*Pt14", 180, 0.9, 2.7, 600, -3., 3.);
    eta_rdphiPt23 = new TH2F(N2+"23_eta_rdphiPt", "eta vs rdphi*Pt23", 180, 0.9, 2.7, 600, -3., 3.);
    eta_rdphiPt24 = new TH2F(N2+"24_eta_rdphiPt", "eta vs rdphi*Pt24", 180, 0.9, 2.7, 600, -3., 3.);
    eta_rdphiPt34 = new TH2F(N2+"34_eta_rdphiPt", "eta vs rdphi*Pt34", 180, 0.9, 2.7, 600, -3., 3.);

  }

 H2DRecHit2(TString name_, TFile* file) {
    name=name_;

    hPt = (TH1F *) file->Get(name+"_hPt");
    hPa_Pt     = (TH2F *) file->Get(name+"_hPa_Pt");
    hPt_CSCSeg = (TH2F *) file->Get(name+"_hPt_CSCSeg");
    hst_csc_nu = (TH2F *) file->Get(name+"_hst_csc_nu");
    hst_csc_nu1= (TH2F *) file->Get(name+"_hst_csc_nu1");
    hChi2_d_Dof= (TH1F *) file->Get(name+"_hChi2_d_Dof");

    eta_dphi01 = (TH2F *) file->Get(name+"01_eta_dphi");
    eta_dphi12 = (TH2F *) file->Get(name+"12_eta_dphi");
    eta_dphi13 = (TH2F *) file->Get(name+"13_eta_dphi");
    eta_dphi14 = (TH2F *) file->Get(name+"14_eta_dphi");
    eta_dphi23 = (TH2F *) file->Get(name+"23_eta_dphi");
    eta_dphi24 = (TH2F *) file->Get(name+"24_eta_dphi");
    eta_dphi34 = (TH2F *) file->Get(name+"34_eta_dphi");

    eta_rdphi01 = (TH2F *) file->Get(name+"01_eta_rdphi");
    eta_rdphi12 = (TH2F *) file->Get(name+"12_eta_rdphi");
    eta_rdphi13 = (TH2F *) file->Get(name+"13_eta_rdphi");
    eta_rdphi14 = (TH2F *) file->Get(name+"14_eta_rdphi");
    eta_rdphi23 = (TH2F *) file->Get(name+"23_eta_rdphi");
    eta_rdphi24 = (TH2F *) file->Get(name+"24_eta_rdphi");
    eta_rdphi34 = (TH2F *) file->Get(name+"34_eta_rdphi");
   
    eta_dphiPt01 = (TH2F *) file->Get(name+"01_eta_dphiPt");
    eta_dphiPt12 = (TH2F *) file->Get(name+"12_eta_dphiPt");
    eta_dphiPt13 = (TH2F *) file->Get(name+"13_eta_dphiPt");
    eta_dphiPt14 = (TH2F *) file->Get(name+"14_eta_dphiPt");
    eta_dphiPt23 = (TH2F *) file->Get(name+"23_eta_dphiPt");
    eta_dphiPt24 = (TH2F *) file->Get(name+"24_eta_dphiPt");
    eta_dphiPt34 = (TH2F *) file->Get(name+"34_eta_dphiPt");

    eta_rdphiPt01 = (TH2F *) file->Get(name+"01_eta_rdphiPt");
    eta_rdphiPt12 = (TH2F *) file->Get(name+"12_eta_rdphiPt");
    eta_rdphiPt13 = (TH2F *) file->Get(name+"13_eta_rdphiPt");
    eta_rdphiPt14 = (TH2F *) file->Get(name+"14_eta_rdphiPt");
    eta_rdphiPt23 = (TH2F *) file->Get(name+"23_eta_rdphiPt");
    eta_rdphiPt24 = (TH2F *) file->Get(name+"24_eta_rdphiPt");
    eta_rdphiPt34 = (TH2F *) file->Get(name+"43_eta_rdphiPt");

  } 

  /// Destructor
  virtual ~H2DRecHit2() {

    delete hPt;
    delete hPa_Pt;
    delete hPt_CSCSeg;
    delete hst_csc_nu;
    delete hst_csc_nu1;
    delete hChi2_d_Dof;

    delete eta_dphi01;
    delete eta_dphi12;
    delete eta_dphi13;
    delete eta_dphi14;
    delete eta_dphi23;
    delete eta_dphi24;
    delete eta_dphi34;

    delete eta_rdphi01;
    delete eta_rdphi12;
    delete eta_rdphi13;
    delete eta_rdphi14;
    delete eta_rdphi23;
    delete eta_rdphi24;
    delete eta_rdphi34;

    delete eta_dphiPt01;
    delete eta_dphiPt12;
    delete eta_dphiPt13;
    delete eta_dphiPt14;
    delete eta_dphiPt23;
    delete eta_dphiPt24;
    delete eta_dphiPt34;

    delete eta_rdphiPt01;
    delete eta_rdphiPt12;
    delete eta_rdphiPt13;
    delete eta_rdphiPt14;
    delete eta_rdphiPt23;
    delete eta_rdphiPt24;
    delete eta_rdphiPt34;
 
  }

  void Fill3(float Pt, float Pa, int csc_nu)
  {
       hPt->Fill(Pt);
       hPa_Pt->Fill(Pt,Pa);
       hPt_CSCSeg->Fill(Pt,csc_nu);
  }

  void Fill3b(float chi2)
  {
       hChi2_d_Dof->Fill(chi2);
  }

  void Fill4(int k, int csc_nu, int csc_nu1 )
  {
       hst_csc_nu->Fill(k,csc_nu);
       hst_csc_nu1->Fill(k,csc_nu1);
  }

  void Fill5_0(double dphi,double dphir,double ptxdphi,double ptxdphir,double eta,double eta_r)
  {
    eta_dphi01->Fill(eta,dphi);
    eta_rdphi01->Fill(eta_r,dphir);
    eta_dphiPt01->Fill(eta,ptxdphi);
    eta_rdphiPt01->Fill(eta_r,ptxdphir);
  }
  void Fill5_1(double dphi,double dphir,double ptxdphi,double ptxdphir,double eta,double eta_r)
  {
    eta_dphi12->Fill(eta,dphi);
    eta_rdphi12->Fill(eta_r,dphir);
    eta_dphiPt12->Fill(eta,ptxdphi);
    eta_rdphiPt12->Fill(eta_r,ptxdphir);
  }
  void Fill5_2(double dphi,double dphir,double ptxdphi,double ptxdphir,double eta,double eta_r)
  {
    eta_dphi13->Fill(eta,dphi);
    eta_rdphi13->Fill(eta_r,dphir);
    eta_dphiPt13->Fill(eta,ptxdphi);
    eta_rdphiPt13->Fill(eta_r,ptxdphir);
  }
  void Fill5_3(double dphi,double dphir,double ptxdphi,double ptxdphir,double eta,double eta_r)
  {
    eta_dphi14->Fill(eta,dphi);
    eta_rdphi14->Fill(eta_r,dphir);
    eta_dphiPt14->Fill(eta,ptxdphi);
    eta_rdphiPt14->Fill(eta_r,ptxdphir);
  }
  void Fill5_4(double dphi,double dphir,double ptxdphi,double ptxdphir,double eta,double eta_r)
  {
    eta_dphi23->Fill(eta,dphi);
    eta_rdphi23->Fill(eta_r,dphir);
    eta_dphiPt23->Fill(eta,ptxdphi);
    eta_rdphiPt23->Fill(eta_r,ptxdphir);
  }
  void Fill5_5(double dphi,double dphir,double ptxdphi,double ptxdphir,double eta,double eta_r)
  {
    eta_dphi24->Fill(eta,dphi);
    eta_rdphi24->Fill(eta_r,dphir);
    eta_dphiPt24->Fill(eta,ptxdphi);
    eta_rdphiPt24->Fill(eta_r,ptxdphir);
  }
  void Fill5_6(double dphi,double dphir,double ptxdphi,double ptxdphir,double eta,double eta_r)
  {
    eta_dphi34->Fill(eta,dphi);
    eta_rdphi34->Fill(eta_r,dphir);
    eta_dphiPt34->Fill(eta,ptxdphi);
    eta_rdphiPt34->Fill(eta_r,ptxdphir);
  }

  void Write() {

       hPt->Write();
       hPa_Pt->Write();
       hPt_CSCSeg->Write();
       hst_csc_nu->Write();
       hst_csc_nu1->Write();
       hChi2_d_Dof->Write();
 
       eta_dphi01->Write();
       eta_dphi12->Write();
       eta_dphi13->Write();
       eta_dphi14->Write();
       eta_dphi23->Write();
       eta_dphi24->Write();
       eta_dphi34->Write();

       eta_rdphi01->Write();
       eta_rdphi12->Write();
       eta_rdphi13->Write();
       eta_rdphi14->Write();
       eta_rdphi23->Write();
       eta_rdphi24->Write();
       eta_rdphi34->Write();

       eta_dphiPt01->Write();
       eta_dphiPt12->Write();
       eta_dphiPt13->Write();
       eta_dphiPt14->Write();
       eta_dphiPt23->Write();
       eta_dphiPt24->Write();
       eta_dphiPt34->Write();

       eta_rdphiPt01->Write();
       eta_rdphiPt12->Write();
       eta_rdphiPt13->Write();
       eta_rdphiPt14->Write();
       eta_rdphiPt23->Write();
       eta_rdphiPt24->Write();
       eta_rdphiPt34->Write();
  
  }

  TH1F *hPt;
  TH2F *hPa_Pt;
  TH2F *hPt_CSCSeg;
  TH2F *hst_csc_nu;
  TH2F *hst_csc_nu1;
  TH1F *hChi2_d_Dof;

  TH2F *eta_dphi01;
  TH2F *eta_dphi12;
  TH2F *eta_dphi13;
  TH2F *eta_dphi14;
  TH2F *eta_dphi23;
  TH2F *eta_dphi24;
  TH2F *eta_dphi34;

  TH2F *eta_rdphi01;
  TH2F *eta_rdphi12;
  TH2F *eta_rdphi13;
  TH2F *eta_rdphi14;
  TH2F *eta_rdphi23;
  TH2F *eta_rdphi24;
  TH2F *eta_rdphi34;

  TH2F *eta_dphiPt01;
  TH2F *eta_dphiPt12;
  TH2F *eta_dphiPt13;
  TH2F *eta_dphiPt14;
  TH2F *eta_dphiPt23;
  TH2F *eta_dphiPt24;
  TH2F *eta_dphiPt34;

  TH2F *eta_rdphiPt01;
  TH2F *eta_rdphiPt12;
  TH2F *eta_rdphiPt13;
  TH2F *eta_rdphiPt14;
  TH2F *eta_rdphiPt23;
  TH2F *eta_rdphiPt24;
  TH2F *eta_rdphiPt34;

  TString name;
};


class H2DRecHit3 {
public:

 H2DRecHit3(std::string name_) {
    TString N3 = name_.c_str();
    name=N3;

    hPt = new TH1F(N3+"_hPt", " Pt of Tracks ", 100,5.,205.);
    hPa_Pt     = new TH2F(N3+"_hPa_Pt", "P vs Pt", 50, 5., 205., 100, 0., 1000.);
    hPt_DTSeg  = new TH2F(N3+"_hPt_DTSeg",  "Pt vs total DT segments number", 40, 5., 205., 40, -0.25, 19.75);
    hst_dt_nu  = new TH2F(N3+"_hst_dt_nu", "station vs dt seg number", 6, 0, 6, 20, -0.25, 9.75);
    hst_dt_nu1 = new TH2F(N3+"_hst_dt_nu1","station vs dt seg number w/ > 5 hits seg", 6, 0, 6, 20, -0.25, 9.75);
    hChi2_d_Dof= new TH1F(N3+"_hChi2_d_Dof"," chi2/dof ",5000,0.,50. ); 

    hrdphi_Pt   = new TH2F(N3+"_hrdphi_Pt", "rec dphi vs Pt", 50, 5., 205., 200, -0.06, 0.06);
    hrdeta_Pt   = new TH2F(N3+"_hrdeta_Pt", "rec deta vs Pt", 50, 5., 205., 202, -0.01, 0.01);

    hbeta_dphi12 = new TH2F(N3+"12_eta_dphi", "eta vs dphi12", 120, 0., 1.2, 2000, -0.1, 0.1);
    hbeta_dphi13 = new TH2F(N3+"13_eta_dphi", "eta vs dphi13", 120, 0., 1.2, 2000, -0.1, 0.1);
    hbeta_dphi14 = new TH2F(N3+"14_eta_dphi", "eta vs dphi14", 120, 0., 1.2, 2000, -0.1, 0.1);
    hbeta_dphi23 = new TH2F(N3+"23_eta_dphi", "eta vs dphi23", 120, 0., 1.2, 2000, -0.1, 0.1);
    hbeta_dphi24 = new TH2F(N3+"24_eta_dphi", "eta vs dphi24", 120, 0., 1.2, 2000, -0.1, 0.1);
    hbeta_dphi34 = new TH2F(N3+"34_eta_dphi", "eta vs dphi34", 120, 0., 1.2, 2000, -0.1, 0.1);

    hbeta_rdphi12 = new TH2F(N3+"12_eta_rdphi", "eta vs rdphi12", 120, 0., 1.2, 2000, -0.1, 0.1);
    hbeta_rdphi13 = new TH2F(N3+"13_eta_rdphi", "eta vs rdphi13", 120, 0., 1.2, 2000, -0.1, 0.1);
    hbeta_rdphi14 = new TH2F(N3+"14_eta_rdphi", "eta vs rdphi14", 120, 0., 1.2, 2000, -0.1, 0.1);
    hbeta_rdphi23 = new TH2F(N3+"23_eta_rdphi", "eta vs rdphi23", 120, 0., 1.2, 2000, -0.1, 0.1);
    hbeta_rdphi24 = new TH2F(N3+"24_eta_rdphi", "eta vs rdphi24", 120, 0., 1.2, 2000, -0.1, 0.1);
    hbeta_rdphi34 = new TH2F(N3+"34_eta_rdphi", "eta vs rdphi34", 120, 0., 1.2, 2000, -0.1, 0.1);

    hbeta_dphiPt12 = new TH2F(N3+"12_eta_dphiPt", "eta vs dphi*Pt12", 120, 0., 1.2, 200, -1.5, 1.5);
    hbeta_dphiPt13 = new TH2F(N3+"13_eta_dphiPt", "eta vs dphi*Pt13", 120, 0., 1.2, 200, -1.5, 1.5);
    hbeta_dphiPt14 = new TH2F(N3+"14_eta_dphiPt", "eta vs dphi*Pt14", 120, 0., 1.2, 200, -1.5, 1.5);
    hbeta_dphiPt23 = new TH2F(N3+"23_eta_dphiPt", "eta vs dphi*Pt23", 120, 0., 1.2, 200, -1.5, 1.5);
    hbeta_dphiPt24 = new TH2F(N3+"24_eta_dphiPt", "eta vs dphi*Pt24", 120, 0., 1.2, 200, -1.5, 1.5);
    hbeta_dphiPt34 = new TH2F(N3+"34_eta_dphiPt", "eta vs dphi*Pt34", 120, 0., 1.2, 200, -1.5, 1.5);

    hbeta_rdphiPt12 = new TH2F(N3+"12_eta_rdphiPt", "eta vs rdphi*Pt12", 120, 0., 1.2, 200, -1.5, 1.5);
    hbeta_rdphiPt13 = new TH2F(N3+"13_eta_rdphiPt", "eta vs rdphi*Pt13", 120, 0., 1.2, 200, -1.5, 1.5);
    hbeta_rdphiPt14 = new TH2F(N3+"14_eta_rdphiPt", "eta vs rdphi*Pt14", 120, 0., 1.2, 200, -1.5, 1.5);
    hbeta_rdphiPt23 = new TH2F(N3+"23_eta_rdphiPt", "eta vs rdphi*Pt23", 120, 0., 1.2, 200, -1.5, 1.5);
    hbeta_rdphiPt24 = new TH2F(N3+"24_eta_rdphiPt", "eta vs rdphi*Pt24", 120, 0., 1.2, 200, -1.5, 1.5);
    hbeta_rdphiPt34 = new TH2F(N3+"34_eta_rdphiPt", "eta vs rdphi*Pt34", 120, 0., 1.2, 200, -1.5, 1.5);

  }

 H2DRecHit3(TString name_, TFile* file) {
    name=name_;

    hPt = (TH1F *) file->Get(name+"_hPt");
    hPa_Pt     = (TH2F *) file->Get(name+"_hPa_Pt");
    hPt_DTSeg  = (TH2F *) file->Get(name+"_hPt_DTSeg");
    hst_dt_nu  = (TH2F *) file->Get(name+"_hst_dt_nu");
    hst_dt_nu1 = (TH2F *) file->Get(name+"_hst_dt_nu1");
    hChi2_d_Dof= (TH1F *) file->Get(name+"_hChi2_d_Dof");

    hrdeta_Pt   = (TH2F *) file->Get(name+"_hrdeta_Pt");
    hrdphi_Pt   = (TH2F *) file->Get(name+"_hrdphi_Pt");

    hbeta_dphi12 = (TH2F *) file->Get(name+"12_eta_dphi");
    hbeta_dphi13 = (TH2F *) file->Get(name+"13_eta_dphi");
    hbeta_dphi14 = (TH2F *) file->Get(name+"14_eta_dphi");
    hbeta_dphi23 = (TH2F *) file->Get(name+"23_eta_dphi");
    hbeta_dphi24 = (TH2F *) file->Get(name+"24_eta_dphi");
    hbeta_dphi34 = (TH2F *) file->Get(name+"34_eta_dphi");

    hbeta_rdphi12 = (TH2F *) file->Get(name+"12_eta_rdphi");
    hbeta_rdphi13 = (TH2F *) file->Get(name+"13_eta_rdphi");
    hbeta_rdphi14 = (TH2F *) file->Get(name+"14_eta_rdphi");
    hbeta_rdphi23 = (TH2F *) file->Get(name+"23_eta_rdphi");
    hbeta_rdphi24 = (TH2F *) file->Get(name+"24_eta_rdphi");
    hbeta_rdphi34 = (TH2F *) file->Get(name+"34_eta_rdphi");

    hbeta_dphiPt12 = (TH2F *) file->Get(name+"12_eta_dphiPt");
    hbeta_dphiPt13 = (TH2F *) file->Get(name+"13_eta_dphiPt");
    hbeta_dphiPt14 = (TH2F *) file->Get(name+"14_eta_dphiPt");
    hbeta_dphiPt23 = (TH2F *) file->Get(name+"23_eta_dphiPt");
    hbeta_dphiPt24 = (TH2F *) file->Get(name+"24_eta_dphiPt");
    hbeta_dphiPt34 = (TH2F *) file->Get(name+"34_eta_dphiPt");

    hbeta_rdphiPt12 = (TH2F *) file->Get(name+"12_eta_rdphiPt");
    hbeta_rdphiPt13 = (TH2F *) file->Get(name+"13_eta_rdphiPt");
    hbeta_rdphiPt14 = (TH2F *) file->Get(name+"14_eta_rdphiPt");
    hbeta_rdphiPt23 = (TH2F *) file->Get(name+"23_eta_rdphiPt");
    hbeta_rdphiPt24 = (TH2F *) file->Get(name+"24_eta_rdphiPt");
    hbeta_rdphiPt34 = (TH2F *) file->Get(name+"34_eta_rdphiPt");

  } 

  /// Destructor
  virtual ~H2DRecHit3() {

    delete hPt;
    delete hPa_Pt;
    delete hPt_DTSeg;
    delete hst_dt_nu;
    delete hst_dt_nu1;
    delete hChi2_d_Dof;

    delete hrdphi_Pt;
    delete hrdeta_Pt;

    delete hbeta_dphi12;
    delete hbeta_dphi13;
    delete hbeta_dphi14;
    delete hbeta_dphi23;
    delete hbeta_dphi24;
    delete hbeta_dphi34;

    delete hbeta_rdphi12;
    delete hbeta_rdphi13;
    delete hbeta_rdphi14;
    delete hbeta_rdphi23;
    delete hbeta_rdphi24;
    delete hbeta_rdphi34;

    delete hbeta_dphiPt12;
    delete hbeta_dphiPt13;
    delete hbeta_dphiPt14;
    delete hbeta_dphiPt23;
    delete hbeta_dphiPt24;
    delete hbeta_dphiPt34;

    delete hbeta_rdphiPt12;
    delete hbeta_rdphiPt13;
    delete hbeta_rdphiPt14;
    delete hbeta_rdphiPt23;
    delete hbeta_rdphiPt24;
    delete hbeta_rdphiPt34;

  }

  void Fill3a(float Pt, float Pa, int dt_nu)
  {
       hPt->Fill(Pt);
       hPa_Pt->Fill(Pt,Pa);
       hPt_DTSeg->Fill(Pt,dt_nu);
  }

  void Fill3c(float chi2)
  {
       hChi2_d_Dof->Fill(chi2);
  }

  void Fill4a(int k, int dt_nu, int dt_nu1)
  {
       hst_dt_nu->Fill(k,dt_nu);
       hst_dt_nu1->Fill(k,dt_nu1);
  }

  void Fill6_1(double dphi,double dphir,double ptxdphi,double ptxdphir,double eta,double eta_r)
  {
    hbeta_dphi12->Fill(eta,dphi);
    hbeta_rdphi12->Fill(eta_r,dphir);
    hbeta_dphiPt12->Fill(eta,ptxdphi);
    hbeta_rdphiPt12->Fill(eta_r,ptxdphir);
  }
  void Fill6_2(double dphi,double dphir,double ptxdphi,double ptxdphir,double eta,double eta_r)
  {
    hbeta_dphi13->Fill(eta,dphi);
    hbeta_rdphi13->Fill(eta_r,dphir);
    hbeta_dphiPt13->Fill(eta,ptxdphi);
    hbeta_rdphiPt13->Fill(eta_r,ptxdphir);
  }
  void Fill6_3(double dphi,double dphir,double ptxdphi,double ptxdphir,double eta,double eta_r)
  {
    hbeta_dphi14->Fill(eta,dphi);
    hbeta_rdphi14->Fill(eta_r,dphir);
    hbeta_dphiPt14->Fill(eta,ptxdphi);
    hbeta_rdphiPt14->Fill(eta_r,ptxdphir);
  }
  void Fill6_4(double dphi,double dphir,double ptxdphi,double ptxdphir,double eta,double eta_r)
  {
    hbeta_dphi23->Fill(eta,dphi);
    hbeta_rdphi23->Fill(eta_r,dphir);
    hbeta_dphiPt23->Fill(eta,ptxdphi);
    hbeta_rdphiPt23->Fill(eta_r,ptxdphir);
  }
  void Fill6_5(double dphi,double dphir,double ptxdphi,double ptxdphir,double eta,double eta_r)
  {
    hbeta_dphi24->Fill(eta,dphi);
    hbeta_rdphi24->Fill(eta_r,dphir);
    hbeta_dphiPt24->Fill(eta,ptxdphi);
    hbeta_rdphiPt24->Fill(eta_r,ptxdphir);
  }
  void Fill6_6(double dphi,double dphir,double ptxdphi,double ptxdphir,double eta,double eta_r)
  {
    hbeta_dphi34->Fill(eta,dphi);
    hbeta_rdphi34->Fill(eta_r,dphir);
    hbeta_dphiPt34->Fill(eta,ptxdphi);
    hbeta_rdphiPt34->Fill(eta_r,ptxdphir);
  }

  void Write() {

       hPt->Write();
       hPa_Pt->Write();
       hPt_DTSeg->Write();
       hst_dt_nu->Write();
       hst_dt_nu1->Write();
       hChi2_d_Dof->Write();

       hrdphi_Pt->Write();
       hrdeta_Pt->Write();

       hbeta_dphi12->Write();
       hbeta_dphi13->Write();
       hbeta_dphi14->Write();
       hbeta_dphi23->Write();
       hbeta_dphi24->Write();
       hbeta_dphi34->Write();

       hbeta_rdphi12->Write();
       hbeta_rdphi13->Write();
       hbeta_rdphi14->Write();
       hbeta_rdphi23->Write();
       hbeta_rdphi24->Write();
       hbeta_rdphi34->Write();

       hbeta_dphiPt12->Write();
       hbeta_dphiPt13->Write();
       hbeta_dphiPt14->Write();
       hbeta_dphiPt23->Write();
       hbeta_dphiPt24->Write();
       hbeta_dphiPt34->Write();

       hbeta_rdphiPt12->Write();
       hbeta_rdphiPt13->Write();
       hbeta_rdphiPt14->Write();
       hbeta_rdphiPt23->Write();
       hbeta_rdphiPt24->Write();
       hbeta_rdphiPt34->Write();

  }

  TH1F *hPt;
  TH2F *hPa_Pt;
  TH2F *hPt_DTSeg;
  TH2F *hst_dt_nu;
  TH2F *hst_dt_nu1;
  TH1F *hChi2_d_Dof;

  TH2F *hrdphi_Pt;
  TH2F *hrdeta_Pt;

  TH2F *hbeta_dphi12;
  TH2F *hbeta_dphi13;
  TH2F *hbeta_dphi14;
  TH2F *hbeta_dphi23;
  TH2F *hbeta_dphi24;
  TH2F *hbeta_dphi34;

  TH2F *hbeta_rdphi12;
  TH2F *hbeta_rdphi13;
  TH2F *hbeta_rdphi14;
  TH2F *hbeta_rdphi23;
  TH2F *hbeta_rdphi24;
  TH2F *hbeta_rdphi34;

  TH2F *hbeta_dphiPt12;
  TH2F *hbeta_dphiPt13;
  TH2F *hbeta_dphiPt14;
  TH2F *hbeta_dphiPt23;
  TH2F *hbeta_dphiPt24;
  TH2F *hbeta_dphiPt34;

  TH2F *hbeta_rdphiPt12;
  TH2F *hbeta_rdphiPt13;
  TH2F *hbeta_rdphiPt14;
  TH2F *hbeta_rdphiPt23;
  TH2F *hbeta_rdphiPt24;
  TH2F *hbeta_rdphiPt34;

  TString name;
};

class H2DRecHit4 {
public:

 H2DRecHit4(std::string name_) {
    TString N4 = name_.c_str();
    name=N4;

    eta_dphiA  = new TH2F(N4+"_eta_dphiA", "eta vs dphit @ME", 180, 0.9, 2.7, 200, -0.03, 0.03);
    eta_rdphiA = new TH2F(N4+"_eta_rdphiA", "eta vs rdphi @ME", 180, 0.9, 2.7, 200, -0.03, 0.03);
    eta_detaA  = new TH2F(N4+"_eta_detaA", "eta vs deta  @ME", 180, 0.9, 2.7, 101, -0.01, 0.01);
    eta_rdetaA  = new TH2F(N4+"_eta_rdetaA", "eta vs rdeta  @ME", 180, 0.9, 2.7, 101, -0.01, 0.01);

    eta_dphiPtA = new TH2F(N4+"_eta_dphiPtA", "eta vs dphi*Pt @ME", 180, 0.9, 2.7, 200, -1.0, 1.0);
    eta_rdphiPtA = new TH2F(N4+"_eta_rdphiPtA", "eta vs rdphi*Pt @ME", 180, 0.9, 2.7, 200, -1.0, 1.0);

    eta_Pt    = new TH2F(N4+"_eta_Pt", "eta vs Pt @ME", 180, 0.9, 2.7, 50, 5.0, 205.0);
    eta_dphir = new TH2F(N4+"_eta_dphir", "eta vs dphi resolution @ME", 180, 0.9, 2.7, 101, -0.0005, 0.0005);
    hpt_dphiA  = new TH2F(N4+"_hpt_dphiA", "pt vs dphi @ME", 50, 5., 205, 200, 0.0, 0.1);
    hpt_rdphiA = new TH2F(N4+"_hpt_rdphiA", "pt vs rdphi @ME", 50, 5., 205, 200, 0.0, 0.1);
  }

 H2DRecHit4(TString name_, TFile* file) {
    name=name_;

    eta_dphiA = (TH2F *) file->Get(name+"_eta_dphiA");
    eta_rdphiA = (TH2F *) file->Get(name+"_eta_rdphiA");
    eta_detaA = (TH2F *) file->Get(name+"_eta_detaA");
    eta_rdetaA = (TH2F *) file->Get(name+"_eta_rdetaA");

    eta_dphiPtA = (TH2F *) file->Get(name+"_eta_dphiPtA");
    eta_rdphiPtA = (TH2F *) file->Get(name+"_eta_rdphiPtA");

    eta_Pt    = (TH2F *) file->Get(name+"_eta_Pt");
    eta_dphir = (TH2F *) file->Get(name+"_eta_dphir");
    hpt_dphiA  = (TH2F *) file->Get(name+"_hpt_dphiA");
    hpt_rdphiA = (TH2F *) file->Get(name+"_hpt_rdphiA");
  } 

  /// Destructor
  virtual ~H2DRecHit4() {

    delete eta_dphiA;
    delete eta_rdphiA;
    delete eta_detaA;
    delete eta_rdetaA;

    delete eta_dphiPtA;
    delete eta_rdphiPtA;

    delete eta_dphir;
    delete eta_Pt;
    delete hpt_dphiA;
    delete hpt_rdphiA;
  }

  void Fill8(double ptxdphi, double dphi, double deta, double eta, double pt)
  {
     eta_dphiA->Fill(eta,dphi);
     eta_detaA->Fill(eta,deta);
     eta_dphiPtA->Fill(eta,ptxdphi);
     hpt_dphiA->Fill(pt,dphi);
  }
  void Fill8a(double ptxrdphi, double rdphi, double rdeta, double reta, double pt, double dphir)
  {
     eta_rdphiA->Fill(reta,rdphi);
     eta_rdetaA->Fill(reta,rdeta);
     eta_rdphiPtA->Fill(reta,ptxrdphi);
     hpt_rdphiA->Fill(pt,rdphi);
     eta_Pt->Fill(reta,pt);
     eta_dphir->Fill(reta,dphir);
  }

  void Write() {
       eta_dphiA->Write();
       eta_rdphiA->Write();
       eta_detaA->Write();
       eta_rdetaA->Write();

       eta_dphiPtA->Write();
       eta_rdphiPtA->Write();

       hpt_dphiA->Write();
       hpt_rdphiA->Write();
       eta_Pt->Write();
       eta_dphir->Write();
  }

  TH2F *eta_dphiA;
  TH2F *eta_rdphiA;
  TH2F *eta_detaA;
  TH2F *eta_rdetaA;

  TH2F *eta_dphiPtA;
  TH2F *eta_rdphiPtA;

  TH2F *hpt_dphiA;
  TH2F *hpt_rdphiA;
  TH2F *eta_Pt;
  TH2F *eta_dphir;

  TString name;
};

class H2DRecHit5 {
public:

 H2DRecHit5(std::string name_) {
    TString N5 = name_.c_str();
    name=N5;

    eta_dphiA  = new TH2F(N5+"_eta_dphiA", "eta vs dphi @MB", 120, 0., 1.2, 200, -0.03, 0.03);
    eta_rdphiA = new TH2F(N5+"_eta_rdphiA", "eta vs rdphi @MB", 120, 0., 1.2, 200, -0.03, 0.03);
    eta_detaA  = new TH2F(N5+"_eta_detaA", "eta vs deta  @MB", 120, 0., 1.2, 101, -0.01, 0.01);
    eta_rdetaA  = new TH2F(N5+"_eta_rdetaA", "eta vs rdeta  @MB", 120, 0., 1.2, 101, -0.01, 0.01);

    eta_dphiPtA  = new TH2F(N5+"_eta_dphiPtA", "eta vs dphi*Pt @MB", 120, 0., 1.2, 200, -1.0, 1.0);
    eta_rdphiPtA = new TH2F(N5+"_eta_rdphiPtA", "eta vs rdphi*Pt @MB", 120, 0., 1.2, 200, -1.0, 1.0);

    eta_Pt    = new TH2F(N5+"_eta_Pt", "eta vs Pt @MB", 120, 0., 1.2, 50, 5.0, 205.0);
    eta_dphir = new TH2F(N5+"_eta_dphir", "eta vs dphi resolution @MB", 120, 0., 1.2, 101, -0.0005, 0.0005);
    hpt_dphiA  = new TH2F(N5+"_hpt_dphiA", "pt vs dphi @MB", 50, 5., 205.0, 200, 0.0, 0.1);
    hpt_rdphiA = new TH2F(N5+"_hpt_rdphiA", "pt vs rdphi @MB", 50, 5., 205.0, 200, 0.0, 0.1);
  }

 H2DRecHit5(TString name_, TFile* file) {
    name=name_;

    eta_dphiA = (TH2F *) file->Get(name+"_eta_dphiA");
    eta_rdphiA = (TH2F *) file->Get(name+"_eta_rdphiA");
    eta_detaA = (TH2F *) file->Get(name+"_eta_detaA");
    eta_rdetaA = (TH2F *) file->Get(name+"_eta_rdetaA");

    eta_dphiPtA = (TH2F *) file->Get(name+"_eta_dphiPtA");
    eta_rdphiPtA = (TH2F *) file->Get(name+"_eta_rdphiPtA");

    eta_Pt    = (TH2F *) file->Get(name+"_eta_Pt");
    eta_dphir = (TH2F *) file->Get(name+"_eta_dphir");
    hpt_dphiA  = (TH2F *) file->Get(name+"_hpt_dphiA");
    hpt_rdphiA = (TH2F *) file->Get(name+"_hpt_rdphiA");
  } 

  /// Destructor
  virtual ~H2DRecHit5() {

    delete eta_dphiA;
    delete eta_rdphiA;
    delete eta_detaA;
    delete eta_rdetaA;

    delete eta_dphiPtA;
    delete eta_rdphiPtA;

    delete eta_Pt;
    delete eta_dphir;
    delete hpt_dphiA;
    delete hpt_rdphiA;
  }

  void Fill9(double ptxdphi,double dphi, double deta, double eta, double pt)
  {
     eta_dphiA->Fill(eta,dphi);
     eta_detaA->Fill(eta,deta);
     eta_dphiPtA->Fill(eta,ptxdphi);
     hpt_dphiA->Fill(pt,dphi);
  }
  void Fill9a(double ptxrdphi, double rdphi, double rdeta, double reta, double pt, double dphir)
  {
     eta_rdphiA->Fill(reta,rdphi);
     eta_rdetaA->Fill(reta,rdeta);
     eta_rdphiPtA->Fill(reta,ptxrdphi);
     hpt_rdphiA->Fill(pt,rdphi);
     eta_Pt->Fill(reta,pt);
     eta_dphir->Fill(reta,dphir);
  }

  void Write() {

       eta_dphiA->Write();
       eta_rdphiA->Write();
       eta_detaA->Write();
       eta_rdetaA->Write();

       eta_dphiPtA->Write();
       eta_rdphiPtA->Write();

       hpt_dphiA->Write();
       hpt_rdphiA->Write();
       eta_Pt->Write();
       eta_dphir->Write();
  }

  TH2F *eta_dphiA;
  TH2F *eta_rdphiA;
  TH2F *eta_detaA;
  TH2F *eta_rdetaA;

  TH2F *eta_dphiPtA;
  TH2F *eta_rdphiPtA;

  TH2F *hpt_dphiA;
  TH2F *hpt_rdphiA;
  TH2F *eta_Pt;
  TH2F *eta_dphir;

  TString name;
};

class H2DRecHit6 {
public:

 H2DRecHit6(std::string name_) {
    TString N6 = name_.c_str();
    name=N6;

    eta_dphi  = new TH2F(N6+"_eta_dphi", "eta vs dphit @ME", 180, 0.9, 2.7, 800, -0.2, 0.2);
    eta_rdphi = new TH2F(N6+"_eta_rdphi", "eta vs rdphi @ME", 180, 0.9, 2.7, 800, -0.2, 0.2);

    eta_phir  = new TH2F(N6+"_eta_phir", "eta vs phi resolution  @ME", 180, 0.9, 2.7, 101, -0.01, 0.01);
    eta_etar  = new TH2F(N6+"_eta_etar", "eta vs eta resolution  @ME", 180, 0.9, 2.7, 101, -0.01, 0.01);

    eta_dphiPt = new TH2F(N6+"_eta_dphiPt", "eta vs dphi*Pt 1", 180, 0.9, 2.7, 400, -3., 3.);
    eta_rdphiPt = new TH2F(N6+"_eta_rdphiPt", "eta vs rdphi*Pt 1", 180, 0.9, 2.7, 400, -3., 3.);

    eta_Pt1   = new TH2F(N6+"_eta_Pt1", "eta vs Pt @ME", 180, 0.9, 2.7, 50, 5.0, 205.0);
    hpt_dphi = new TH2F(N6+"_hpt_dphi", "pt vs dphi @ME", 50, 5., 205, 200, 0.0, 0.1);
    hpt_rdphi = new TH2F(N6+"_hpt_rdphi", "pt vs rdphi @ME", 50, 5., 205, 200, 0.0, 0.1);
  }

 H2DRecHit6(TString name_, TFile* file) {
    name=name_;

    eta_dphi  = (TH2F *) file->Get(name+"_eta_dphi");
    eta_rdphi = (TH2F *) file->Get(name+"_eta_rdphi");

    eta_phir = (TH2F *) file->Get(name+"_eta_phir");
    eta_etar = (TH2F *) file->Get(name+"_eta_etar");

    eta_Pt1 = (TH2F *) file->Get(name+"_eta_Pt1");
    eta_dphiPt = (TH2F *) file->Get(name+"_eta_dphiPt");
    eta_rdphiPt = (TH2F *) file->Get(name+"_eta_rdphiPt");

    hpt_dphi = (TH2F *) file->Get(name+"_hpt_dphiA");
    hpt_rdphi = (TH2F *) file->Get(name+"_hpt_rdphiA");
  } 

  /// Destructor
  virtual ~H2DRecHit6() {

    delete  eta_dphi;
    delete  eta_rdphi;

    delete  eta_phir;
    delete  eta_etar;

    delete eta_dphiPt;
    delete eta_rdphiPt;

    delete eta_Pt1;
    delete hpt_dphi;
    delete hpt_rdphi;
  }

  void Fill8b(double ptxdphi, double dphi, double eta, double pt) 
  {
     eta_dphiPt->Fill(eta,ptxdphi);
     eta_dphi->Fill(eta,dphi);
     hpt_dphi->Fill(pt,dphi);
  }
  void Fill8c(double ptxrdphi, double rdphi, double phir, double etar, double reta, double pt) 
  {
     eta_rdphiPt->Fill(reta,ptxrdphi);
     eta_rdphi->Fill(reta,rdphi);

     eta_phir->Fill(reta,phir);
     eta_etar->Fill(reta,etar);
     hpt_rdphi->Fill(pt,rdphi);
     eta_Pt1->Fill(reta,pt);
  }

  void Write() {
 
       eta_dphi->Write();
       eta_rdphi->Write();

       eta_phir->Write();
       eta_etar->Write();

       eta_dphiPt->Write();
       eta_rdphiPt->Write();

       hpt_dphi->Write();
       hpt_rdphi->Write();
       eta_Pt1->Write();
  }
 
  TH2F *eta_dphi;
  TH2F *eta_rdphi;
 
  TH2F *eta_phir;
  TH2F *eta_etar;

  TH2F *eta_dphiPt;
  TH2F *eta_rdphiPt;

  TH2F *hpt_dphi;
  TH2F *hpt_rdphi;
  TH2F *eta_Pt1;

  TString name;
};

class H2DRecHit7 {
public:

 H2DRecHit7(std::string name_) {
    TString N7 = name_.c_str();
    name=N7;

    eta_dphi  = new TH2F(N7+"_eta_dphi", "eta vs dphit @MB", 120, 0., 1.2, 800, -0.2, 0.2);
    eta_rdphi = new TH2F(N7+"_eta_rdphi", "eta vs rdphi @MB", 120, 0., 1.2, 800, -0.2, 0.2);

    eta_phir  = new TH2F(N7+"_eta_phir", "eta vs phi resolution  @MB", 120, 0., 1.2, 101, -0.01, 0.01);
    eta_etar  = new TH2F(N7+"_eta_etar", "eta vs eta resolution  @MB", 120, 0., 1.2, 101, -0.01, 0.01);

    eta_dphiPt  = new TH2F(N7+"_eta_dphiPt", "eta vs dphi*Pt @MB", 120, 0., 1.2, 400, -3., 3.);
    eta_rdphiPt = new TH2F(N7+"_eta_rdphiPt", "eta vs rdphi*Pt @MB", 120, 0., 1.2, 400, -3., 3.);

    eta_Pt1   = new TH2F(N7+"_eta_Pt1", "eta vs Pt @MB", 180, 0.9, 2.7, 50, 5.0, 205.0);
    hpt_dphi = new TH2F(N7+"_hpt_dphi", "pt vs dphi @MB", 50, 5., 205, 200, 0.0, 0.1);
    hpt_rdphi = new TH2F(N7+"_hpt_rdphi", "pt vs rdphi @MB", 50, 5., 205, 200, 0.0, 0.1);
  }

 H2DRecHit7(TString name_, TFile* file) {
    name=name_;

    eta_dphi  = (TH2F *) file->Get(name+"_eta_dphi");
    eta_rdphi = (TH2F *) file->Get(name+"_eta_rdphi");

    eta_phir = (TH2F *) file->Get(name+"_eta_phir");
    eta_etar = (TH2F *) file->Get(name+"_eta_etar");

    eta_dphiPt = (TH2F *) file->Get(name+"_eta_dphiPt");
    eta_rdphiPt = (TH2F *) file->Get(name+"_eta_rdphiPt");

    eta_Pt1 = (TH2F *) file->Get(name+"_eta_Pt1");
    hpt_dphi = (TH2F *) file->Get(name+"_hpt_dphiA");
    hpt_rdphi = (TH2F *) file->Get(name+"_hpt_rdphiA");
  } 

  /// Destructor
  virtual ~H2DRecHit7() {

    delete  eta_dphi;
    delete  eta_rdphi;

    delete  eta_phir;
    delete  eta_etar;

    delete eta_dphiPt;
    delete eta_rdphiPt;

    delete eta_Pt1;
    delete hpt_dphi;
    delete hpt_rdphi;
  }

  void Fill9b(double ptxdphi, double dphi, double eta, double pt)
  {
     eta_dphiPt->Fill(eta,ptxdphi);
     eta_dphi->Fill(eta,dphi);
     hpt_dphi->Fill(pt,dphi);
  }
  void Fill9c(double ptxrdphi, double rdphi, double phir, double etar, double reta, double pt) 
  {
     eta_rdphiPt->Fill(reta,ptxrdphi);
     eta_rdphi->Fill(reta,rdphi);

     eta_phir->Fill(reta,phir);
     eta_etar->Fill(reta,etar);
     hpt_rdphi->Fill(pt,rdphi);
     eta_Pt1->Fill(reta,pt);
  }

  void Write() {

       eta_dphi->Write();
       eta_rdphi->Write();

       eta_phir->Write();
       eta_etar->Write();

       eta_dphiPt->Write();
       eta_rdphiPt->Write();

       hpt_dphi->Write();
       hpt_rdphi->Write();
       eta_Pt1->Write();
  }

  TH2F *eta_dphi;
  TH2F *eta_rdphi;
 
  TH2F *eta_phir;
  TH2F *eta_etar;

  TH2F *eta_dphiPt;
  TH2F *eta_rdphiPt;

  TH2F *hpt_dphi;
  TH2F *hpt_rdphi;
  TH2F *eta_Pt1;

  TString name;
};

class H2DRecHit10 {
public:

 H2DRecHit10(std::string name_) {
    TString N10 = name_.c_str();
    name=N10;

    eta_dphiA  = new TH2F(N10+"_eta_dphiA", "eta vs dphi @OL", 60, 0.7,1.3, 200, -0.08, 0.08);
    eta_rdphiA = new TH2F(N10+"_eta_rdphiA", "eta vs rdphi @OL", 60, 0.7,1.3, 200, -0.08, 0.08);
    eta_detaA  = new TH2F(N10+"_eta_detaA", "eta vs deta  @OL", 60, 0.7,1.3, 101, -0.01, 0.01);
    eta_rdetaA  = new TH2F(N10+"_eta_rdetaA", "eta vs rdeta  @OL", 60, 0.7,1.3, 101, -0.01, 0.01);

    eta_dphiPtA  = new TH2F(N10+"_eta_dphiPtA", "eta vs dphi*Pt @OL", 60, 0.7,1.3, 200, -1.0, 1.0);
    eta_rdphiPtA = new TH2F(N10+"_eta_rdphiPtA", "eta vs rdphi*Pt @OL", 60, 0.7,1.3, 200, -1.0, 1.0);

    eta_Pt    = new TH2F(N10+"_eta_Pt", "eta vs Pt @OL", 60, 0.7,1.3, 50, 5.0, 205.0);
    eta_dphir = new TH2F(N10+"_eta_dphir", "eta vs dphi resolution @OL", 60, 0.7,1.3, 101, -0.0005, 0.0005);
    hpt_dphiA  = new TH2F(N10+"_hpt_dphiA", "pt vs dphi @OL", 50, 5., 205.0, 200, 0.0, 0.1);
    hpt_rdphiA = new TH2F(N10+"_hpt_rdphiA", "pt vs rdphi @OL", 50, 5., 205.0, 200, 0.0, 0.1);
  }

 H2DRecHit10(TString name_, TFile* file) {
    name=name_;

    eta_dphiA = (TH2F *) file->Get(name+"_eta_dphiA");
    eta_rdphiA = (TH2F *) file->Get(name+"_eta_rdphiA");
    eta_detaA = (TH2F *) file->Get(name+"_eta_detaA");
    eta_rdetaA = (TH2F *) file->Get(name+"_eta_rdetaA");

    eta_dphiPtA = (TH2F *) file->Get(name+"_eta_dphiPtA");
    eta_rdphiPtA = (TH2F *) file->Get(name+"_eta_rdphiPtA");

    eta_Pt    = (TH2F *) file->Get(name+"_eta_Pt");
    eta_dphir = (TH2F *) file->Get(name+"_eta_dphir");
    hpt_dphiA  = (TH2F *) file->Get(name+"_hpt_dphiA");
    hpt_rdphiA = (TH2F *) file->Get(name+"_hpt_rdphiA");
  } 

  /// Destructor
  virtual ~H2DRecHit10() {

    delete eta_dphiA;
    delete eta_rdphiA;
    delete eta_detaA;
    delete eta_rdetaA;

    delete eta_dphiPtA;
    delete eta_rdphiPtA;

    delete eta_Pt;
    delete eta_dphir;
    delete hpt_dphiA;
    delete hpt_rdphiA;
  }

  void Fill12(double ptxdphi,double dphi, double deta, double eta, double pt)
  {
     eta_dphiA->Fill(eta,dphi);
     eta_detaA->Fill(eta,deta);
     eta_dphiPtA->Fill(eta,ptxdphi);
     hpt_dphiA->Fill(pt,dphi);
  }
  void Fill12a(double ptxrdphi, double rdphi, double rdeta, double reta, double pt, double dphir)
  {
     eta_rdphiA->Fill(reta,rdphi);
     eta_rdetaA->Fill(reta,rdeta);
     eta_rdphiPtA->Fill(reta,ptxrdphi);
     hpt_rdphiA->Fill(pt,rdphi);
     eta_Pt->Fill(reta,pt);
     eta_dphir->Fill(reta,dphir);
  }

  void Write() {

       eta_dphiA->Write();
       eta_rdphiA->Write();
       eta_detaA->Write();
       eta_rdetaA->Write();

       eta_dphiPtA->Write();
       eta_rdphiPtA->Write();

       hpt_dphiA->Write();
       hpt_rdphiA->Write();
       eta_Pt->Write();
       eta_dphir->Write();
  }

  TH2F *eta_dphiA;
  TH2F *eta_rdphiA;
  TH2F *eta_detaA;
  TH2F *eta_rdetaA;

  TH2F *eta_dphiPtA;
  TH2F *eta_rdphiPtA;

  TH2F *hpt_dphiA;
  TH2F *hpt_rdphiA;
  TH2F *eta_Pt;
  TH2F *eta_dphir;

  TString name;
};

#endif
