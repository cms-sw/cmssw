#ifndef RecoMuon_MuonSeedValidatorHisto_H
#define RecoMuon_MuonSeedValidatorHisto_H

/** \class SeedValidatorHisto
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

    heta_mu4 = new TH2F(N1+"_heta_mu4", " Seg multiplicity vs eta_of_track", 59, -2.95, 2.95, 40, -0.25, 19.75);
    heta_mu3 = new TH2F(N1+"_heta_mu3", " good Rec Seg vs eta_of_track", 59, -2.95, 2.95, 20, -0.25, 9.75);
    heta_mu2 = new TH2F(N1+"_heta_mu2", " Rec Seg vs eta_of_track", 59, -2.95, 2.95, 20, -0.25, 9.75);
    heta_mu1 = new TH2F(N1+"_heta_mu1", " Sim Seg vs eta_of_track", 59, -2.95, 2.95, 20, -0.25, 9.75);

    heta_rh  = new TH2F(N1+"_heta_rh",  " All rechits vs eta ", 250, 0.0, 2.5, 20, -0.25, 9.75);

    heta_NSeed  = new TH2F(N1+"_heta_NSeed", " eta vs N of Seed", 59, -2.95, 2.95, 40, -0.25, 19.75);
    heta_NSta   = new TH2F(N1+"_heta_NSta", " eta vs N of Sta ", 59, -2.95, 2.95, 20, -0.25, 9.75);
    heta_Sim    = new TH1F(N1+"_heta_Sim", " eta of SimTrack ", 59, -2.95, 2.95);
    heta_simPt  = new TH2F(N1+"_heta_simPt",  " eta vs sim pt ", 59, -2.95, 2.95, 2500, 0.0, 2500.0);
    heta_simQPt = new  TH2F(N1+"_heta_simQPt",  " eta vs sim  Q/Pt ", 59, -2.95, 2.95, 800, -0.2, 0.2);
    heta_simQPa = new  TH2F(N1+"_heta_simQPa",  " eta vs sim  Q/Pa ", 59, -2.95, 2.95, 800, -0.2, 0.2);

    heta_pt  = new  TH2F(N1+"_heta_pt",  " eta vs seed pt ", 59, -2.95, 2.95, 2500, 0.0, 2500.0);
    heta_pa  = new  TH2F(N1+"_heta_pa",  " eta vs seed p  ", 59, -2.95, 2.95, 2500, 0.0, 2500.0);
    heta_bestPt  = new  TH2F(N1+"_heta_bestPt"," eta vs best seed pt ", 59, -2.95, 2.95, 5000, -2500.0, 2500.0);

    heta_pullQP = new  TH2F(N1+"_heta_pullQP",  "eta vs best pull of Q/P ", 59, -2.95, 2.95, 1000, -10.0, 10.0);
    heta_pullQPt= new  TH2F(N1+"_heta_pullQPt", "eta vs best pull of Q/Pt ", 59, -2.95, 2.95,1000, -10.0, 10.0);
    heta_QPt    = new  TH2F(N1+"_heta_QPt",     "eta vs best Q/Pt ", 59, -2.95, 2.95, 1000, -0.4, 0.4);
    heta_errQP  = new  TH2F(N1+"_heta_errQP",   "eta vs best Q/P  error", 59, -2.95, 2.95, 500, -0.01, 0.24);
    heta_errQPt = new  TH2F(N1+"_heta_errQPt",  "eta vs best Q/Pt error", 59, -2.95, 2.95, 500, -0.01, 0.24);
    heta_resolSd= new  TH2F(N1+"_heta_resolSd", "eta vs best Q/Pt resol", 59, -2.95, 2.95, 500, -1., 1.);

    Tpt_ptLoss  = new  TH2F(N1+"_Tpt_ptLoss",  " trk pt vs ptLoss ratio ", 250, 0., 250., 80, 0.3, 1.1);
    Mpt_ptLoss  = new  TH2F(N1+"_Mpt_ptLoss",  " seed pt vs ptLoss ratio ", 250, 0., 250., 80, 0.3, 1.1);

    heta_statrk = new TH1F(N1+"_heta_statrk"," eta from sta tracks",59, -2.95, 2.95);
    heta_staQPt = new TH2F(N1+"_heta_staQPt", " eta vs sta  Q/Pt ", 59, -2.95, 2.95, 800, -0.2, 0.2);
    heta_staQPa = new TH2F(N1+"_heta_staQPa", " eta vs sta  Q/Pa ", 59, -2.95, 2.95, 800, -0.2, 0.2);
    heta_staPt  = new TH2F(N1+"_heta_staPt",  " eta vs sta  Pt ", 59, -2.95, 2.95, 2500, 0.0, 2500.0);
    heta_staPa  = new TH2F(N1+"_heta_staPa",  " eta vs sta  P  ", 59, -2.95, 2.95, 2500, 0.0, 2500.0);
    heta_simPt1 = new TH2F(N1+"_heta_simPt1",  " eta vs sim pt ", 59, -2.95, 2.95, 2500, 0.0, 2500.0);
    heta_resolSA= new  TH2F(N1+"_heta_resolSA", " eta vs Q/Pt resol", 59, -2.95, 2.95, 500, -1., 1.);

    heta_errdx = new  TH2F(N1+"_heta_errdx", "eta vs dx/dz error", 59, -2.95, 2.95, 500, 0., 0.01);
    heta_errdy = new  TH2F(N1+"_heta_errdy", "eta vs dy/dz error", 59, -2.95, 2.95, 500, 0., 0.15);
    heta_errx  = new  TH2F(N1+"_heta_errx", "eta vs x error", 59, -2.95, 2.95, 1000, 0., 0.1);
    heta_erry  = new  TH2F(N1+"_heta_erry", "eta vs y error", 59, -2.95, 2.95, 1000, 0., 1.);

    h_dh_st = new TH2F(N1+"_h_dh_st", " dEta from vector vs station", 19, -4.75, 4.75,400,-1.0,1.0);
    h_df_st = new TH2F(N1+"_h_df_st", " dPhi from vector vs station", 19, -4.75, 4.75,800,-0.1,0.1);
    h_dx_st = new TH2F(N1+"_h_dx_st", " dx from position vs station", 19, -4.75, 4.75,400,-4.0,4.0);
    h_dy_st = new TH2F(N1+"_h_dy_st", " dy from position vs station", 19, -4.75, 4.75,400,-4.0,4.0);

    h_Nsta_Nseed = new TH2F(N1+"_h_Nsta_Nseed", " N_STA vs N_Seed ", 30, -0.25, 14.75, 30, -0.25, 14.75);

    hPt = new TH1F(N1+"_hPt", " Pt of Tracks ", 100,5.,205.);
    hPa_Pt     = new TH2F(N1+"_hPa_Pt", "P vs Pt", 50, 5., 205., 100, 0., 1000.);
    hPa_nSeg = new TH2F(N1+"_hPa_nSeg", "Pt vs total CSC segments number", 40, 5., 205., 40, -0.25, 19.75);

    hP_Showers1 = new TH2F(N1+"_hP_Showers1", "Pt vs # of Showering segments 1", 600, 0., 1200., 50, -0.5, 49.5);
    hP_Showers2 = new TH2F(N1+"_hP_Showers2", "Pt vs # of Showering segments 2", 600, 0., 1200., 50, -0.5, 49.5);
    hP_Showers3 = new TH2F(N1+"_hP_Showers3", "Pt vs # of Showering segments 3", 600, 0., 1200., 50, -0.5, 49.5);

    hP_Showers1a = new TH2F(N1+"_hP_Showers1a", "Pt vs # of Showering segs w/o st1 1", 600, 0., 1200., 50, -0.5, 49.5);
    hP_Showers2a = new TH2F(N1+"_hP_Showers2a", "Pt vs # of Showering segs w/o st1 2", 600, 0., 1200., 50, -0.5, 49.5);
    hP_Showers3a = new TH2F(N1+"_hP_Showers3a", "Pt vs # of Showering segs w/o st1 3", 600, 0., 1200., 50, -0.5, 49.5);

    hP_Showers1b = new TH2F(N1+"_hP_Showers1b", "Pt vs # of Showering  1", 600, 0., 1200., 5, -0.5, 4.5);
    hP_Showers2b = new TH2F(N1+"_hP_Showers2b", "Pt vs # of Showering  2", 600, 0., 1200., 5, -0.5, 4.5);
    hP_Showers3b = new TH2F(N1+"_hP_Showers3b", "Pt vs # of Showering  3", 600, 0., 1200., 5, -0.5, 4.5);

    hP_avShower1 = new TH2F(N1+"_hP_avShower1", "Pt vs ave. Showering segments 1", 600, 0., 1200., 500, -0.5, 999.5);
    hP_avShower2 = new TH2F(N1+"_hP_avShower2", "Pt vs ave. Showering segments 2", 600, 0., 1200., 500, -0.5, 999.5);
    hP_avShower3 = new TH2F(N1+"_hP_avShower3", "Pt vs ave. Showering segments 3", 600, 0., 1200., 500, -0.5, 999.5);

    hP_maxR1 = new TH2F(N1+"_hP_maxR1", "Pt vs max dR of the cone 1", 600, 0., 1200., 300, -0.5, 1.);
    hP_maxR2 = new TH2F(N1+"_hP_maxR2", "Pt vs max dR of the cone 2", 600, 0., 1200., 300, -0.5, 1.);
    hP_maxR3 = new TH2F(N1+"_hP_maxR3", "Pt vs max dR of the cone 3", 600, 0., 1200., 300, -0.5, 1.);
 } 

 H2DRecHit1(TString name_, TFile* file) {
    name=name_;
    heta_mu4 = (TH2F *) file->Get(name+"_heta_mu4");
    heta_mu3 = (TH2F *) file->Get(name+"_heta_mu3");
    heta_mu2 = (TH2F *) file->Get(name+"_heta_mu2");
    heta_mu1 = (TH2F *) file->Get(name+"_heta_mu1");

    heta_rh  = (TH2F *) file->Get(name+"_heta_rh");

    heta_NSeed  = (TH2F *) file->Get(name+"_heta_NSeed");
    heta_NSta   = (TH2F *) file->Get(name+"_heta_NSta");
    heta_Sim    = (TH1F *) file->Get(name+"_heta_Sim");
    heta_simPt   = (TH2F *) file->Get(name+"_heta_simPt");
    heta_simQPt  = (TH2F *) file->Get(name+"_heta_simQPt");
    heta_simQPa  = (TH2F *) file->Get(name+"_heta_simQPa");

    heta_pt      = (TH2F *) file->Get(name+"_heta_pt");
    heta_pa      = (TH2F *) file->Get(name+"_heta_pa");
    heta_bestPt  = (TH2F *) file->Get(name+"_heta_bestPt");

    heta_pullQP  = (TH2F *) file->Get(name+"_heta_pullQP");
    heta_pullQPt = (TH2F *) file->Get(name+"_heta_pullQPt");
    heta_QPt     = (TH2F *) file->Get(name+"_heta_QPt");
    heta_errQP   = (TH2F *) file->Get(name+"_heta_errQP");
    heta_errQPt  = (TH2F *) file->Get(name+"_heta_errQPt");
    heta_resolSd = (TH2F *) file->Get(name+"_heta_resolSd");
    Tpt_ptLoss   = (TH2F *) file->Get(name+"_Tpt_ptLoss");
    Mpt_ptLoss   = (TH2F *) file->Get(name+"_Mpt_ptLoss");

    heta_statrk = (TH1F *) file->Get(name+"_heta_statrk");
    heta_staQPt = (TH2F *) file->Get(name+"_heta_staQPt");
    heta_staQPa = (TH2F *) file->Get(name+"_heta_staQPa");
    heta_staPt  = (TH2F *) file->Get(name+"_heta_staPt");
    heta_staPa  = (TH2F *) file->Get(name+"_heta_staPa");
    heta_simPt1 = (TH2F *) file->Get(name+"_heta_simPt1");
    heta_resolSA= (TH2F *) file->Get(name+"_heta_resolSA");

    heta_errdx  = (TH2F *) file->Get(name+"_heta_errdx");
    heta_errdy  = (TH2F *) file->Get(name+"_heta_errdy");
    heta_errx   = (TH2F *) file->Get(name+"_heta_errx");
    heta_erry   = (TH2F *) file->Get(name+"_heta_erry");

 
    h_dh_st = (TH2F *) file->Get(name+"_h_dh_st");
    h_df_st = (TH2F *) file->Get(name+"_h_df_st");
    h_dx_st = (TH2F *) file->Get(name+"_h_dx_st");
    h_dy_st = (TH2F *) file->Get(name+"_h_dy_st");

    h_Nsta_Nseed = (TH2F *) file->Get(name+"_h_Nsta_Nseed");

    hPt = (TH1F *) file->Get(name+"_hPt");
    hPa_Pt     = (TH2F *) file->Get(name+"_hPa_Pt");
    hPa_nSeg = (TH2F *) file->Get(name+"_hPa_nSeg");

    hP_Showers1 = (TH2F *) file->Get(name+"_hP_Showers1");
    hP_Showers2 = (TH2F *) file->Get(name+"_hP_Showers2");
    hP_Showers3 = (TH2F *) file->Get(name+"_hP_Showers3");

    hP_Showers1a = (TH2F *) file->Get(name+"_hP_Showers1a");
    hP_Showers2a = (TH2F *) file->Get(name+"_hP_Showers2a");
    hP_Showers3a = (TH2F *) file->Get(name+"_hP_Showers3a");

    hP_Showers1b = (TH2F *) file->Get(name+"_hP_Showers1b");
    hP_Showers2b = (TH2F *) file->Get(name+"_hP_Showers2b");
    hP_Showers3b = (TH2F *) file->Get(name+"_hP_Showers3b");

    hP_avShower1 = (TH2F *) file->Get(name+"_hP_avShower1");
    hP_avShower2 = (TH2F *) file->Get(name+"_hP_avShower2");
    hP_avShower3 = (TH2F *) file->Get(name+"_hP_avShower3");

    hP_maxR1 = (TH2F *) file->Get(name+"_hP_maxR1");
    hP_maxR2 = (TH2F *) file->Get(name+"_hP_maxR2");
    hP_maxR3 = (TH2F *) file->Get(name+"_hP_maxR3");
 }

 /// Destructor
 virtual ~H2DRecHit1() {
    delete heta_mu4;
    delete heta_mu3;
    delete heta_mu2;
    delete heta_mu1;

    delete heta_rh;

    delete heta_NSeed;
    delete heta_NSta;
    delete heta_Sim;
    delete heta_simPt;
    delete heta_simQPt;
    delete heta_simQPa;

    delete hPt;
    delete hPa_Pt;
    delete hPa_nSeg;

    delete hP_Showers1;
    delete hP_Showers2;
    delete hP_Showers3;

    delete hP_Showers1a;
    delete hP_Showers2a;
    delete hP_Showers3a;

    delete hP_Showers1b;
    delete hP_Showers2b;
    delete hP_Showers3b;

    delete hP_avShower1;
    delete hP_avShower2;
    delete hP_avShower3;

    delete hP_maxR1;
    delete hP_maxR2;
    delete hP_maxR3;

    delete heta_pt;
    delete heta_pa;
    delete heta_bestPt;

    delete heta_pullQP;
    delete heta_pullQPt;
    delete heta_QPt;
    delete heta_errQP;
    delete heta_errQPt;
    delete heta_resolSd;
    delete Tpt_ptLoss;
    delete Mpt_ptLoss;

    delete heta_statrk;
    delete heta_staQPt;
    delete heta_staQPa;
    delete heta_staPt;
    delete heta_staPa;
    delete heta_simPt1;
    delete heta_resolSA;

    delete heta_errdx;
    delete heta_errdy;
    delete heta_errx;
    delete heta_erry;
   
    delete h_dh_st;
    delete h_df_st;
    delete h_dx_st;
    delete h_dy_st;

    delete h_Nsta_Nseed;

 }

 void Fill1(int layer_nu, int goodlayer_nu, int seg_sum, int sim_nu, double eta_simtrk) {
      heta_mu4->Fill(eta_simtrk,seg_sum);
      heta_mu3->Fill(eta_simtrk,goodlayer_nu);
      heta_mu2->Fill(eta_simtrk,layer_nu);
      heta_mu1->Fill(eta_simtrk,sim_nu);
 }
 void Fill1a(int rh_nu, double eta_a) {
      heta_rh->Fill(eta_a,rh_nu);
 }
 void Fill1b(double eta_sim, int nSeed, int nSta, double simPt, double simQPt, double simQPa) {
      heta_NSeed->Fill(eta_sim,nSeed);
      heta_NSta ->Fill(eta_sim,nSta);
      heta_Sim ->Fill(eta_sim);
      heta_simPt->Fill(eta_sim, simPt);
      heta_simQPt->Fill(eta_sim, simQPt);
      heta_simQPa->Fill(eta_sim, simQPa);
 }
 void Fill1c(float Pt, float Pa, int nu_seg)
 {
      hPt->Fill(Pt);
      hPa_Pt->Fill(Pt,Pa);
      hPa_nSeg->Fill(Pa,nu_seg);
 }
 void Fill1d1(double P, int showers, int showersa, int showersb, double ave, double maxR )
 {
      hP_Showers1->Fill(P, showers);
      hP_Showers1a->Fill(P, showersa);
      hP_Showers1b->Fill(P, showersb);
      hP_avShower1->Fill(P, ave);
      hP_maxR1->Fill(P, maxR);
 }
 void Fill1d2(double P, int showers, int showersa, int showersb, double ave, double maxR )
 {
      hP_Showers2->Fill(P, showers);
      hP_Showers2a->Fill(P, showersa);
      hP_Showers2b->Fill(P, showersb);
      hP_avShower2->Fill(P, ave);
      hP_maxR2->Fill(P, maxR);
 }
 void Fill1d3(double P, int showers, int showersa, int showersb, double ave, double maxR )
 {
      hP_Showers3->Fill(P, showers);
      hP_Showers3a->Fill(P, showersa);
      hP_Showers3b->Fill(P, showersb);
      hP_avShower3->Fill(P, ave);
      hP_maxR3->Fill(P, maxR);
 }
 void Fill1f(double eta, double err_dx, double err_dy, double err_x, double err_y) {
      heta_errdx->Fill(eta,err_dx);
      heta_errdy->Fill(eta,err_dy);
      heta_errx->Fill(eta,err_x);
      heta_erry->Fill(eta,err_y);
 }
 void Fill1g(double seed_mT, double seed_mA, double bestSeed_mT, float eta_seed, double ptLoss, double ptTrk) {
      heta_pt->Fill(eta_seed,seed_mT);
      heta_pa->Fill(eta_seed,seed_mA);
      heta_bestPt->Fill(eta_seed,bestSeed_mT);
      Tpt_ptLoss->Fill(ptTrk, ptLoss);
      Mpt_ptLoss->Fill(seed_mT, ptLoss);
 }
 void Fill1i(double pull_qbp, float eta_seed, double QPt, double pull_qbpt, double errqbp, double errqbpt, double resol_qbpt){
      heta_pullQP->Fill(eta_seed, pull_qbp);
      heta_pullQPt->Fill(eta_seed, pull_qbpt);
      heta_QPt->Fill(eta_seed, QPt);
      heta_errQP->Fill(eta_seed, errqbp);
      heta_errQPt->Fill(eta_seed, errqbpt);
      heta_resolSd->Fill(eta_seed, resol_qbpt);
 }
 void Fill1j(double eta_sta, double sta_qbp, double sta_qbpt, double sta_pt, double sta_pa, double sim_pt, double resol_qbpt) {
      heta_staQPt->Fill(eta_sta, sta_qbpt);
      heta_staQPa->Fill(eta_sta, sta_qbp);
      heta_staPt ->Fill(eta_sta, sta_pt);
      heta_simPt1->Fill(eta_sta, sim_pt);
      heta_staPa ->Fill(eta_sta, sta_pa);
      heta_resolSA ->Fill(eta_sta, resol_qbpt);
 }
 void Fill1e(int station, double dh_v, double df_v, double dx_p, double dy_p) {
      h_dh_st->Fill(station, dh_v);    
      h_df_st->Fill(station, df_v);    
      h_dx_st->Fill(station, dx_p);    
      h_dy_st->Fill(station, dy_p);    
 }

 void Fill1o(int Nsta, int Nseed ) {
      h_Nsta_Nseed->Fill(Nseed, Nsta);
 }

 void Write() {
      heta_mu4->Write();
      heta_mu3->Write();
      heta_mu2->Write();
      heta_mu1->Write();

      heta_rh->Write();

      heta_NSeed->Write();
      heta_NSta ->Write();
      heta_Sim ->Write();
      heta_simPt->Write();
      heta_simQPt->Write();
      heta_simQPa->Write();

      heta_pt->Write();
      heta_pa->Write();
      heta_bestPt->Write();

      heta_pullQP->Write();
      heta_pullQPt->Write();
      heta_QPt->Write();
      heta_errQP->Write();
      heta_errQPt->Write();
      heta_resolSd->Write();
      Tpt_ptLoss->Write();
      Mpt_ptLoss->Write();

      heta_statrk->Write();
      heta_staQPt->Write();
      heta_staQPa->Write();
      heta_staPt->Write();
      heta_staPa->Write();
      heta_simPt1->Write();
      heta_resolSA->Write();

      heta_errdx->Write();
      heta_errdy->Write();
      heta_errx->Write();
      heta_erry->Write();

      h_dh_st->Write();
      h_df_st->Write();
      h_dx_st->Write();
      h_dy_st->Write();

      h_Nsta_Nseed->Write();

      hPt->Write();
      hPa_Pt->Write();
      hPa_nSeg->Write();

      hP_Showers1->Write();
      hP_Showers2->Write();
      hP_Showers3->Write();

      hP_Showers1a->Write();
      hP_Showers2a->Write();
      hP_Showers3a->Write();

      hP_Showers1b->Write();
      hP_Showers2b->Write();
      hP_Showers3b->Write();

      hP_avShower1->Write();
      hP_avShower2->Write();
      hP_avShower3->Write();

      hP_maxR1->Write();
      hP_maxR2->Write();
      hP_maxR3->Write();
 }

 TH2F *heta_mu4;
 TH2F *heta_mu3;
 TH2F *heta_mu2;
 TH2F *heta_mu1;

 TH2F *heta_rh;

 TH2F *heta_NSeed;
 TH2F *heta_NSta;
 TH1F *heta_Sim;
 TH2F *heta_simPt;
 TH2F *heta_simQPt;
 TH2F *heta_simQPa;

 TH2F *heta_pt;
 TH2F *heta_pa;
 TH2F *heta_bestPt;

 TH2F *heta_pullQP;
 TH2F *heta_pullQPt;
 TH2F *heta_QPt;
 TH2F *heta_errQP;
 TH2F *heta_errQPt;
 TH2F *heta_resolSd;
 TH2F *Tpt_ptLoss;
 TH2F *Mpt_ptLoss;

 TH1F *heta_statrk;
 TH2F *heta_staQPt;
 TH2F *heta_staQPa;
 TH2F *heta_staPt;
 TH2F *heta_staPa;
 TH2F *heta_simPt1;
 TH2F *heta_resolSA;

 TH2F *heta_errdx;
 TH2F *heta_errdy;
 TH2F *heta_errx;
 TH2F *heta_erry;

 TH2F *h_dh_st;
 TH2F *h_df_st;
 TH2F *h_dx_st;
 TH2F *h_dy_st;

 TH2F *h_Nsta_Nseed;

 TH1F *hPt;
 TH2F *hPa_Pt;
 TH2F *hPa_nSeg;

 TH2F *hP_Showers1;
 TH2F *hP_Showers2;
 TH2F *hP_Showers3;

 TH2F *hP_Showers1a;
 TH2F *hP_Showers2a;
 TH2F *hP_Showers3a;

 TH2F *hP_Showers1b;
 TH2F *hP_Showers2b;
 TH2F *hP_Showers3b;

 TH2F *hP_avShower1;
 TH2F *hP_avShower2;
 TH2F *hP_avShower3;

 TH2F *hP_maxR1;
 TH2F *hP_maxR2;
 TH2F *hP_maxR3;

 TString name;

};


class H2DRecHit2 {
public:

 H2DRecHit2(std::string name_) {
    TString N2 = name_.c_str();
    name=N2;

    heta_nSimSegs= new TH2F(N2+"_heta_nSimSegs", "# Segs vs eta from simtrack", 59, -2.95, 2.95, 30, -0.25, 14.75);
    heta_nSegs   = new TH2F(N2+"_heta_nSegs", "# Segs vs eta from simtrack", 59, -2.95, 2.95, 30, -0.25, 14.75);
    heta_nlayers = new TH2F(N2+"_heta_nlayers", "# layers vs eta from simtrack", 59, -2.95, 2.95, 20, -0.25, 9.75);
    heta_dNSegs  = new TH2F(N2+"_heta_dNSegs", "# simseg - recseg vs eta from simtrack", 59, -2.95, 2.95, 39, -9.75, 9.75);
    heta_dNlayers= new TH2F(N2+"_heta_dNlayers", "# simseg - layers vs eta from simtrack", 59, -2.95, 2.95, 39, -9.75, 9.75);

    heta_cscrh = new TH2F(N2+"_heta_cscrh", " csc rechits vs eta ", 59, -2.95, 2.95, 40, -0.25, 19.75);
    heta_dtrh  = new TH2F(N2+"_heta_dtrh",  " dt  rechits vs eta ", 59, -2.95, 2.95, 40, -0.25, 19.75);

  }

 H2DRecHit2(TString name_, TFile* file) {
    name=name_;

    heta_nSimSegs= (TH2F *) file->Get(name+"_heta_nSimSegs");
    heta_nSegs   = (TH2F *) file->Get(name+"_heta_nSegs");
    heta_nlayers = (TH2F *) file->Get(name+"_heta_nlayers");
 
    heta_dNSegs  = (TH2F *) file->Get(name+"_heta_dNSegs");
    heta_dNlayers= (TH2F *) file->Get(name+"_heta_dNlayers");

    heta_cscrh = (TH2F *) file->Get(name+"_heta_cscrh");
    heta_dtrh  = (TH2F *) file->Get(name+"_heta_dtrh");

  } 

  /// Destructor
  virtual ~H2DRecHit2() {

    delete heta_nSimSegs;
    delete heta_nSegs;
    delete heta_nlayers;
    delete heta_dNSegs;
    delete heta_dNlayers;

    delete heta_cscrh;
    delete heta_dtrh;
  }

  void Fill2a(double eta_simTrk,int nu_layer, int nu_seg, int nu_simseg, int dNSegs, int dNlayers) 
  {
       heta_nSimSegs->Fill(eta_simTrk,nu_simseg);
       heta_nSegs->Fill(eta_simTrk,nu_seg);
       heta_nlayers->Fill(eta_simTrk,nu_layer);
       heta_dNSegs->Fill(eta_simTrk, dNSegs);
       heta_dNlayers->Fill(eta_simTrk, dNlayers);
  }
  void Fill2b(double eta_c, int cscrh_nu)
  {
       heta_cscrh->Fill(eta_c,cscrh_nu);
  }
  void Fill2c(double eta_d, int dtrh_nu)
  {
       heta_dtrh->Fill(eta_d,dtrh_nu);
  }


  void Write() {

       heta_nSimSegs->Write();
       heta_nSegs->Write();
       heta_nlayers->Write();
       heta_dNSegs->Write();
       heta_dNlayers->Write();

       heta_cscrh->Write();
       heta_dtrh->Write();

  }

  TH2F *heta_nSimSegs;
  TH2F *heta_nSegs;
  TH2F *heta_nlayers;

  TH2F *heta_dNSegs;
  TH2F *heta_dNlayers;

  TH2F *heta_cscrh;
  TH2F *heta_dtrh;

  TString name;
};


class H2DRecHit3 {
public:

 H2DRecHit3(std::string name_) {
    TString N3 = name_.c_str();
    name=N3;

    // for fail sta tracking!
    hsimEta_NSeed   = new TH2F(N3+"_hsimEta_NSeed", "No sta, sim_eta vs NSeed", 59, -2.95, 2.95, 20, -0.25, 9.75);

    hsimEta_seedEta = new TH2F(N3+"_hsimEta_seedEta", "No sta, sim_eta vs seed_eta", 59, -2.95, 2.95, 59, -2.95, 2.95);
    hsimEta_seedPt = new  TH2F(N3+"_hsimEta_seedPt",  "NO sta, sim_eta vs seed pt ", 59, -2.95, 2.95, 500, 0.0, 500.0);
    hsimEta_errPt  = new  TH2F(N3+"_hsimEta_errPt", "No sta, sim_eta vs Pt error", 59, -2.95, 2.95, 500, 0.0, 500.0);
    hsimEta_pullQPt  = new  TH2F(N3+"_hsimEta_pullQPt", "No sta, sim_eta vs q/pt pull", 59, -2.95, 2.95,1000, -10.0, 10.0);
    hsimEta_NStation = new TH2F(N3+"_hsimEta_NStation", "No sta, sim_eta vs N Seg", 59, -2.95, 2.95, 20, -0.25, 9.75);
    hsimEta_NSeg1  = new TH2F(N3+"_hsimEta_NSeg1","No sta, sim_eta vs N good Seg", 59, -2.95, 2.95, 20, -0.25, 9.75);

    hsimEta_dh = new TH2F(N3+"_hsimEta_dh","No sta, sim_eta vs h_seed - h_Seg", 59, -2.95, 2.95, 400, -1., 1.);
    hsimEta_df = new TH2F(N3+"_hsimEta_df","No sta, sim_eta vs f_seed - f_Seg", 59, -2.95, 2.95, 400, -0.1, 0.1);

    hPhi_resid = new TH1F(N3+"_hPhi_resid"," phi residual for fail sta case ",500, -0.25, 0.25);
    hEta_resid = new TH1F(N3+"_hEta_resid"," eta residual for fail sta case ",500, -0.25, 0.25);

  }

 H2DRecHit3(TString name_, TFile* file) {
    name=name_;

    hsimEta_NSeed   = (TH2F *) file->Get(name+"_hsimEta_NSeed");

    hsimEta_seedEta = (TH2F *) file->Get(name+"_hsimEta_seedEta");
    hsimEta_seedPt  = (TH2F *) file->Get(name+"_hsimEta_seedPt");
    hsimEta_errPt   = (TH2F *) file->Get(name+"_hsimEta_errPt");
    hsimEta_pullQPt = (TH2F *) file->Get(name+"_hsimEta_pullQPt");
    hsimEta_NStation = (TH2F *) file->Get(name+"_hsimEta_NStation");
    hsimEta_NSeg1   = (TH2F *) file->Get(name+"_hsimEta_NSeg1");
    hsimEta_dh      = (TH2F *) file->Get(name+"_hsimEta_dh");
    hsimEta_df      = (TH2F *) file->Get(name+"_hsimEta_df");
    hPhi_resid = (TH1F *) file->Get(name+"_hPhi_resid");
    hEta_resid = (TH1F *) file->Get(name+"_hEta_resid");
  } 

  /// Destructor
  virtual ~H2DRecHit3() {

    delete hsimEta_NSeed;
    delete hsimEta_seedEta; 
    delete hsimEta_seedPt;
    delete hsimEta_errPt;
    delete hsimEta_pullQPt;
    delete hsimEta_NStation;
    delete hsimEta_NSeg1;
    delete hsimEta_dh;
    delete hsimEta_df;
    delete hPhi_resid;
    delete hEta_resid;
  }

  void Fill3a(double simEta, double seedEta, double seed_mT, double ptErr, double pullQPt) {
       hsimEta_seedEta->Fill(simEta, seedEta); 
       hsimEta_seedPt->Fill(simEta, seed_mT);
       hsimEta_errPt->Fill(simEta, ptErr);
       hsimEta_pullQPt->Fill(simEta, pullQPt);
  }
  void Fill3b(double simEta, int NStation, int NGoodSeg) {
       hsimEta_NStation->Fill(simEta, NStation);
       hsimEta_NSeg1->Fill(simEta, NGoodSeg);
  }
  void Fill3c(double phi_resid, double eta_resid) {
       hPhi_resid->Fill(phi_resid);
       hEta_resid->Fill(eta_resid);
  }
  void Fill3d(double sim_eta, double dEta, double dPhi) {
       hsimEta_dh->Fill(sim_eta, dEta);
       hsimEta_df->Fill(sim_eta, dPhi);
  }
  void Fill3f(double simEta, int nu_seed) {
       hsimEta_NSeed->Fill(simEta, nu_seed);
  }

  void Write() {

       hsimEta_NSeed->Write();
       hsimEta_seedEta->Write(); 
       hsimEta_seedPt->Write();
       hsimEta_errPt->Write();
       hsimEta_pullQPt->Write();
       hsimEta_NStation->Write();
       hsimEta_NSeg1->Write();
       hsimEta_dh->Write();
       hsimEta_df->Write();
       hPhi_resid->Write();
       hEta_resid->Write();

  }

  TH2F *hsimEta_NSeed;
  TH2F *hsimEta_seedEta; 
  TH2F *hsimEta_seedPt;
  TH2F *hsimEta_errPt;
  TH2F *hsimEta_pullQPt;
  TH2F *hsimEta_NStation;
  TH2F *hsimEta_NSeg1;
  TH2F *hsimEta_dh;
  TH2F *hsimEta_df;
  TH1F *hPhi_resid;
  TH1F *hEta_resid;

  TString name;
};


class H2DRecHit4 {
public:

 H2DRecHit4() {

    // open a scope to look at a certain case!

    hSeedPhi    = new TH1F("hSeedPhi"," seed phi distribution ",160, -3.1415, 3.1415);
    hStaPhi     = new TH1F("hStaPhi"," sta phi distribution ",160, -3.1415, 3.1415);

    hSeedSimEta = new TH2F("hSeedSimEta"," seed - sim eta distribution ",59, -2.95, 2.95, 59, -2.95, 2.95);
    hStaSimEta  = new TH2F("hStaSimEta"," sta - sim eta distribution "  ,59, -2.95, 2.95, 59, -2.95, 2.95);

    h_dh_st1 = new TH2F("h_dh_st1", " dEta from vector vs station bad pt", 19, -4.75, 4.75,400,-1.0,1.0);
    h_df_st1 = new TH2F("h_df_st1", " dPhi from vector vs station bad pt", 19, -4.75, 4.75,800,-0.1,0.1);
    h_dx_st1 = new TH2F("h_dx_st1", " dx from position vs station bad pt", 19, -4.75, 4.75,400,-4.0,4.0);
    h_dy_st1 = new TH2F("h_dy_st1", " dy from position vs station bad pt", 19, -4.75, 4.75,400,-4.0,4.0);

    //  pt vs. nHits and chi2
    h_pt_nHits = new TH2F("h_pt_nHits", " pt vs nHits ", 104, -0.25, 51.75, 250, 0.0,250.0);
    h_pt_chi2  = new TH2F("h_pt_chi2" , " pt vs chi2  ", 500, 0., 50., 250, 0.0,250.0);

 }
 
 H2DRecHit4( TFile* file ) {

    hSeedPhi = (TH1F *) file->Get("hSeedPhi");
    hStaPhi  = (TH1F *) file->Get("hStaPhi");
    hSeedSimEta = (TH2F *) file->Get("hSeedSimEta");
    hStaSimEta  = (TH2F *) file->Get("hStaSimEta");

    h_dh_st1 = (TH2F *) file->Get("h_dh_st1");
    h_df_st1 = (TH2F *) file->Get("h_df_st1");
    h_dx_st1 = (TH2F *) file->Get("h_dx_st1");
    h_dy_st1 = (TH2F *) file->Get("h_dy_st1");

    h_pt_nHits = (TH2F *) file->Get("h_pt_nHits");
    h_pt_chi2  = (TH2F *) file->Get("h_pt_chi2");

 } 
 
  /// Destructor
 virtual ~H2DRecHit4() {

    delete hSeedPhi;
    delete hStaPhi;

    delete hSeedSimEta;
    delete hStaSimEta;

    delete h_dh_st1;
    delete h_df_st1;
    delete h_dx_st1;
    delete h_dy_st1;

    delete h_pt_nHits;
    delete h_pt_chi2;
 }

 void Fill4a( double sta_phi, double sta_eta, double sim_eta) {
       hStaPhi->Fill( sta_phi );
       hStaSimEta->Fill( sta_eta, sim_eta );
 }
 void Fill4b(double seed_phi, double seed_eta, double sim_eta) {
       hSeedPhi->Fill(seed_phi);
       hSeedSimEta->Fill( seed_eta, sim_eta );
 }
 void Fill4c(int station, double dh_v, double df_v, double dx_p, double dy_p) {
       h_dh_st1->Fill(station, dh_v);    
       h_df_st1->Fill(station, df_v);    
       h_dx_st1->Fill(station, dx_p);    
       h_dy_st1->Fill(station, dy_p);    
 }
 void Fill4d(double pt, int nhits, double chi2) {
       h_pt_nHits->Fill(nhits, pt);
       h_pt_chi2->Fill(chi2, pt);
 }


 void Write() {

       hSeedPhi->Write();
       hStaPhi->Write();

       hSeedSimEta->Write();
       hStaSimEta->Write();

       h_dh_st1->Write();
       h_df_st1->Write();
       h_dx_st1->Write();
       h_dy_st1->Write();

       h_pt_nHits->Write();
       h_pt_chi2->Write();

 }

  TH1F *hSeedPhi;
  TH1F *hStaPhi;

  TH2F *hSeedSimEta;
  TH2F *hStaSimEta;

  TH2F *h_dh_st1;
  TH2F *h_df_st1;
  TH2F *h_dx_st1;
  TH2F *h_dy_st1;

  TH2F *h_pt_nHits;
  TH2F *h_pt_chi2;

};

class H2DRecHit5 {
public:

 H2DRecHit5() {

    h_UnRelatedSeed = new TH2F("h_UnRelatedSeed", " eta of unrelated seeds ", 59, -2.95, 2.95, 2500,0.,2500.);
    h_UnRelatedSta  = new TH2F("h_UnRelatedSta", " eta of unrelated sta ", 59, -2.95, 2.95, 2500,0.,2500.);
    h_OrphanSeed = new TH2F("h_OrphanSeed", "sim eta vs orphan seed eta", 59, -2.95, 2.95, 59, -2.95, 2.95);
    h_OrphanStaEta = new TH2F("h_OrphanStaEta", "sim eta vs orphanSta eta", 59, -2.95, 2.95, 59, -2.95, 2.95);
    h_OrphanStaPhi = new TH2F("h_OrphanStaPhi", "sim phi vs orphanSta phi", 100, -3.1416, 3.1416, 100, -3.1416, 3.1416);
    h_OrphanSeedPt = new TH2F("h_OrphanSeedPt", "sim eta vs orphan seed Pt", 59, -2.95, 2.95, 2500, 0., 2500);
    h_OrphanStaPt  = new TH2F("h_OrphanStaPt",  "sim eta vs orphan sta  Pt", 59, -2.95, 2.95, 2500, 0., 2500);

 } 
 
 H2DRecHit5( TFile* file ) {

    h_UnRelatedSeed = (TH2F *) file->Get("h_UnRelatedSeed");
    h_UnRelatedSta  = (TH2F *) file->Get("h_UnRelatedSta");
    h_OrphanSeed    = (TH2F *) file->Get("h_OrphanSeed");
    h_OrphanStaEta  = (TH2F *) file->Get("h_OrphanStaEta");
    h_OrphanStaPhi  = (TH2F *) file->Get("h_OrphanStaPhi");

    h_OrphanSeedPt  = (TH2F *) file->Get("h_OrphanSeedPt");
    h_OrphanStaPt   = (TH2F *) file->Get("h_OrphanStaPt");

 } 
 
  /// Destructor
 virtual ~H2DRecHit5() {

    delete h_UnRelatedSeed;
    delete h_UnRelatedSta;
    delete h_OrphanSeed;
    delete h_OrphanStaEta;
    delete h_OrphanStaPhi;
    delete h_OrphanSeedPt;
    delete h_OrphanStaPt;

 }

 void Fill5a(double seed_eta, double seed_pt) {
      h_UnRelatedSeed->Fill(seed_eta, seed_pt);
 } 
 void Fill5b(double seed_eta, double sim_eta, double seed_pt) {
      h_OrphanSeed->Fill(sim_eta, seed_eta );
      h_OrphanSeedPt->Fill(sim_eta, seed_pt );
 } 
 void Fill5c(double sta_eta, double sta_pt) {
      h_UnRelatedSta->Fill(sta_eta, sta_pt);
 } 
 void Fill5d(double sta_eta, double sim_eta, double sta_pt, double sta_phi, double sim_phi) {
      h_OrphanStaEta->Fill(sim_eta, sta_eta);
      h_OrphanStaPhi->Fill(sim_phi, sta_phi);
      h_OrphanStaPt->Fill(sim_eta, sta_pt);
 }
 

 void Write() {

      h_UnRelatedSeed->Write();
      h_UnRelatedSta->Write();
      h_OrphanSeed->Write();
      h_OrphanStaEta->Write();
      h_OrphanStaPhi->Write();
      h_OrphanSeedPt->Write();
      h_OrphanStaPt->Write();
 }

 TH2F *h_UnRelatedSeed;
 TH2F *h_UnRelatedSta;
 TH2F *h_OrphanSeed;
 TH2F *h_OrphanStaEta;
 TH2F *h_OrphanStaPhi;
 TH2F *h_OrphanSeedPt;
 TH2F *h_OrphanStaPt;

};
#endif
