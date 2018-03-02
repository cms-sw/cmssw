// ----------------------------------------------------------------------------------------------------------------
// Basic example script for making tracking performance plots using the ntuples produced by L1TrackNtupleMaker.cc
// By Louise Skinnari, June 2013  
// ----------------------------------------------------------------------------------------------------------------

#include "TROOT.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TMath.h"
#include <TError.h>


#include <iostream>
#include <string>
#include <vector>

using namespace std;

void SetPlotStyle();
void mySmallText(Double_t x,Double_t y,Color_t color,char *text);

double getIntervalContainingFractionOfEntries( TH1* histogram, double interval );
void makeResidualIntervalPlot( TString type, TString dir, TString variable, TH1F* h_68, TH1F* h_90, TH1F* h_99, double minY, double maxY );


// ----------------------------------------------------------------------------------------------------------------
// Main script
// ----------------------------------------------------------------------------------------------------------------


void L1TrackNtuplePlot(TString type, int TP_select_injet=0, int TP_select_pdgid=0, int TP_select_eventid=0, float TP_minPt=2.0, float TP_maxPt=100.0, float TP_maxEta=2.4) {

  // type:              this is the input file you want to process (minus ".root" extension)
  // TP_select_pdgid:   if non-zero, only select TPs with a given PDG ID
  // TP_select_eventid: if zero, only look at TPs from primary interaction, else, include TPs from pileup
  // TP_minPt:          only look at TPs with pt > X GeV
  // TP_maxPt:          only look at TPs with pt < X GeV
  // TP_maxEta:         only look at TPs with |eta| < X

  // TP_select_injet: only look at TPs that are within a jet with pt > 30 GeV (==1) or within a jet with pt > 100 GeV (==2) or all TPs (==0)
 

  gROOT->SetBatch();
  gErrorIgnoreLevel = kWarning;
  
  SetPlotStyle();
  
  
  // ----------------------------------------------------------------------------------------------------------------
  // define input options

  int L1Tk_minNstub = 4;  
  float L1Tk_maxChi2 = 999999;  
  float L1Tk_maxChi2dof = 999999.;  
  
  bool doDetailedPlots = false; //turn on to make full set of plots
  bool doGausFit = false;       //do gaussian fit for resolution vs eta/pt plots
  bool doLooseMatch = false;    //looser MC truth matching

  //some counters for integrated efficiencies
  int n_all_eta2p5 = 0;
  int n_all_eta1p75 = 0;
  int n_all_eta1p0 = 0;
  int n_match_eta2p5 = 0;
  int n_match_eta1p75 = 0;
  int n_match_eta1p0 = 0;

  // counters for total track rates 
  int ntrk_pt2 = 0;
  int ntrk_pt3 = 0;
  int ntrk_pt10 = 0;
  int ntp_pt2 = 0;
  int ntp_pt3 = 0;
  int ntp_pt10 = 0;

  //int ntrk_4stub_pt3 = 0;
  //int ntrk_4stub_pt10 = 0;
  //int ntrk_4stubchi2_pt3 = 0;
  //int ntrk_4stubchi2_pt10 = 0;


  // ----------------------------------------------------------------------------------------------------------------
  // read ntuples
  TChain* tree = new TChain("L1TrackNtuple/eventTree");
  tree->Add(type+".root");
  
  if (tree->GetEntries() == 0) {
    cout << "File doesn't exist or is empty, returning..." << endl;
    return;
  }
  

  // ----------------------------------------------------------------------------------------------------------------
  // define leafs & branches

  // tracking particles
  vector<float>* tp_pt;
  vector<float>* tp_eta;
  vector<float>* tp_phi;
  vector<float>* tp_dxy;
  vector<float>* tp_z0;
  vector<float>* tp_d0;
  vector<int>*   tp_pdgid;
  vector<int>*   tp_nmatch;
  vector<int>*   tp_nstub;
  //vector<int>*   tp_nstublayer;
  vector<int>*   tp_eventid;
  //vector<int>*   tp_injet;
  //vector<int>*   tp_injet_highpt;
  
  // *L1 track* properties, for tracking particles matched to a L1 track
  vector<float>* matchtrk_pt;
  vector<float>* matchtrk_eta;
  vector<float>* matchtrk_phi;
  vector<float>* matchtrk_d0;
  vector<float>* matchtrk_z0;
  vector<float>* matchtrk_chi2;
  vector<int>*   matchtrk_nstub;
  //vector<int>*   matchtrk_injet;
  //vector<int>*   matchtrk_injet_highpt;

  // all L1 tracks
  vector<float>* trk_pt;
  vector<float>* trk_eta;
  vector<float>* trk_phi;
  vector<float>* trk_chi2;
  vector<int>*   trk_nstub;
  //vector<int>*   trk_injet;
  //vector<int>*   trk_injet_highpt;

  //vector<float>* jet_eta;
  //vector<float>* jet_pt;
  //vector<float>* jet_tp_sumpt;
  //vector<float>* jet_matchtrk_sumpt;
  //vector<float>* jet_trk_sumpt;


  TBranch* b_tp_pt;
  TBranch* b_tp_eta;
  TBranch* b_tp_phi;
  TBranch* b_tp_dxy;
  TBranch* b_tp_z0;
  TBranch* b_tp_d0;
  TBranch* b_tp_pdgid;
  TBranch* b_tp_nmatch;
  TBranch* b_tp_nstub;
  //TBranch* b_tp_nstublayer;
  TBranch* b_tp_eventid;
  //TBranch* b_tp_injet;
  //TBranch* b_tp_injet_highpt;

  TBranch* b_matchtrk_pt;
  TBranch* b_matchtrk_eta;
  TBranch* b_matchtrk_phi;
  TBranch* b_matchtrk_d0;
  TBranch* b_matchtrk_z0;
  TBranch* b_matchtrk_chi2; 
  TBranch* b_matchtrk_nstub;
  //TBranch* b_matchtrk_injet;
  //TBranch* b_matchtrk_injet_highpt;

  TBranch* b_trk_pt; 
  TBranch* b_trk_eta; 
  TBranch* b_trk_phi; 
  TBranch* b_trk_chi2; 
  TBranch* b_trk_nstub; 
  //TBranch* b_trk_injet;
  //TBranch* b_trk_injet_highpt;

  //TBranch* b_jet_eta;
  //TBranch* b_jet_pt;
  //TBranch* b_jet_tp_sumpt;
  //TBranch* b_jet_matchtrk_sumpt;
  //TBranch* b_jet_trk_sumpt;

  tp_pt  = 0;
  tp_eta = 0;
  tp_phi = 0;
  tp_dxy = 0;
  tp_z0  = 0;
  tp_d0  = 0;
  tp_pdgid = 0;
  tp_nmatch = 0;
  tp_nstub = 0;
  //tp_nstublayer = 0;
  tp_eventid = 0;
  //tp_injet = 0;
  //tp_injet_highpt = 0;

  matchtrk_pt  = 0;
  matchtrk_eta = 0;
  matchtrk_phi = 0;
  matchtrk_d0  = 0;
  matchtrk_z0  = 0;
  matchtrk_chi2  = 0; 
  matchtrk_nstub = 0;
  //matchtrk_injet = 0;
  //matchtrk_injet_highpt = 0;

  trk_pt = 0; 
  trk_eta = 0; 
  trk_phi = 0; 
  trk_chi2 = 0; 
  trk_nstub = 0; 
  //trk_injet = 0;
  //trk_injet_highpt = 0;

  //jet_eta = 0;
  //jet_pt = 0;
  //jet_tp_sumpt = 0;
  //jet_matchtrk_sumpt = 0;
  //jet_trk_sumpt = 0;


  tree->SetBranchAddress("tp_pt",     &tp_pt,     &b_tp_pt);
  tree->SetBranchAddress("tp_eta",    &tp_eta,    &b_tp_eta);
  tree->SetBranchAddress("tp_phi",    &tp_phi,    &b_tp_phi);
  tree->SetBranchAddress("tp_dxy",    &tp_dxy,    &b_tp_dxy);
  tree->SetBranchAddress("tp_z0",     &tp_z0,     &b_tp_z0);
  tree->SetBranchAddress("tp_d0",     &tp_d0,     &b_tp_d0);
  tree->SetBranchAddress("tp_pdgid",  &tp_pdgid,  &b_tp_pdgid);
  if (doLooseMatch) tree->SetBranchAddress("tp_nloosematch", &tp_nmatch, &b_tp_nmatch);
  else tree->SetBranchAddress("tp_nmatch", &tp_nmatch, &b_tp_nmatch);
  tree->SetBranchAddress("tp_nstub",      &tp_nstub,      &b_tp_nstub);
  //tree->SetBranchAddress("tp_nstublayer", &tp_nstublayer, &b_tp_nstublayer);
  tree->SetBranchAddress("tp_eventid",    &tp_eventid,    &b_tp_eventid);
  /*
  if (TP_select_injet > 0) {
    tree->SetBranchAddress("tp_injet",   &tp_injet,   &b_tp_injet);
    tree->SetBranchAddress("tp_injet_highpt",   &tp_injet_highpt,   &b_tp_injet_highpt);
  }
  */

  if (doLooseMatch) {
    tree->SetBranchAddress("loosematchtrk_pt",    &matchtrk_pt,    &b_matchtrk_pt);
    tree->SetBranchAddress("loosematchtrk_eta",   &matchtrk_eta,   &b_matchtrk_eta);
    tree->SetBranchAddress("loosematchtrk_phi",   &matchtrk_phi,   &b_matchtrk_phi);
    tree->SetBranchAddress("loosematchtrk_d0",    &matchtrk_d0,    &b_matchtrk_d0);
    tree->SetBranchAddress("loosematchtrk_z0",    &matchtrk_z0,    &b_matchtrk_z0);
    tree->SetBranchAddress("loosematchtrk_chi2",  &matchtrk_chi2,  &b_matchtrk_chi2);
    tree->SetBranchAddress("loosematchtrk_nstub", &matchtrk_nstub, &b_matchtrk_nstub);
    /*
    if (TP_select_injet > 0) {
      tree->SetBranchAddress("loosematchtrk_injet",   &matchtrk_injet,   &b_matchtrk_injet);
      tree->SetBranchAddress("loosematchtrk_injet_highpt",   &matchtrk_injet_highpt,   &b_matchtrk_injet_highpt);
    }
    */
  }
  else {
    tree->SetBranchAddress("matchtrk_pt",    &matchtrk_pt,    &b_matchtrk_pt);
    tree->SetBranchAddress("matchtrk_eta",   &matchtrk_eta,   &b_matchtrk_eta);
    tree->SetBranchAddress("matchtrk_phi",   &matchtrk_phi,   &b_matchtrk_phi);
    tree->SetBranchAddress("matchtrk_d0",    &matchtrk_d0,    &b_matchtrk_d0);
    tree->SetBranchAddress("matchtrk_z0",    &matchtrk_z0,    &b_matchtrk_z0);
    tree->SetBranchAddress("matchtrk_chi2",  &matchtrk_chi2,  &b_matchtrk_chi2);
    tree->SetBranchAddress("matchtrk_nstub", &matchtrk_nstub, &b_matchtrk_nstub);
    /*
    if (TP_select_injet > 0) {
      tree->SetBranchAddress("matchtrk_injet",   &matchtrk_injet,   &b_matchtrk_injet);
      tree->SetBranchAddress("matchtrk_injet_highpt",   &matchtrk_injet_highpt,   &b_matchtrk_injet_highpt);
    }
    */
  }

  tree->SetBranchAddress("trk_pt",   &trk_pt,   &b_trk_pt);
  tree->SetBranchAddress("trk_eta",  &trk_eta,  &b_trk_eta);
  tree->SetBranchAddress("trk_phi",  &trk_phi,  &b_trk_phi);
  tree->SetBranchAddress("trk_chi2", &trk_chi2, &b_trk_chi2);
  tree->SetBranchAddress("trk_nstub", &trk_nstub, &b_trk_nstub);
  /*
  if (TP_select_injet > 0) {
    tree->SetBranchAddress("trk_injet",   &trk_injet,   &b_trk_injet);
    tree->SetBranchAddress("trk_injet_highpt",   &trk_injet_highpt,   &b_trk_injet_highpt);
  }
  */

  /*
  if (TP_select_injet > 0) {
    tree->SetBranchAddress("jet_eta", &jet_eta, &b_jet_eta);
    tree->SetBranchAddress("jet_pt",  &jet_pt,  &b_jet_pt);
    tree->SetBranchAddress("jet_tp_sumpt",   &jet_tp_sumpt,   &b_jet_tp_sumpt);
    tree->SetBranchAddress("jet_trk_sumpt",  &jet_trk_sumpt, &b_jet_trk_sumpt);
    if (doLooseMatch) tree->SetBranchAddress("jet_loosematchtrk_sumpt",  &jet_matchtrk_sumpt, &b_jet_matchtrk_sumpt);
    else tree->SetBranchAddress("jet_matchtrk_sumpt",  &jet_matchtrk_sumpt, &b_jet_matchtrk_sumpt);
  }
  */


  // ----------------------------------------------------------------------------------------------------------------
  // histograms
  // ----------------------------------------------------------------------------------------------------------------

  /////////////////////////////////////////////////
  // NOTATION:                                   //
  // 'C' - Central eta range, |eta|<0.8          //
  // 'I' - Intermediate eta range, 0.8<|eta|<1.6 //
  // 'F' - Forward eta range, |eta|>1.6          //
  //                                             //
  // 'L' - Low pt range,  pt<8 GeV               //
  // 'H' - High pt range, pt>8 GeV               //
  /////////////////////////////////////////////////



  // ----------------------------------------------------------------------------------------------------------------
  // for efficiencies

  TH1F* h_tp_pt    = new TH1F("tp_pt",   ";Tracking particle p_{T} [GeV]; Tracking particles / 1.0 GeV", 100,  0,   100.0);
  TH1F* h_tp_pt_L  = new TH1F("tp_pt_L", ";Tracking particle p_{T} [GeV]; Tracking particles / 0.1 GeV",  80,  0,     8.0);
  TH1F* h_tp_pt_LC = new TH1F("tp_pt_LC",";Tracking particle p_{T} [GeV]; Tracking particles / 0.1 GeV",  80,  0,     8.0);
  TH1F* h_tp_pt_H  = new TH1F("tp_pt_H", ";Tracking particle p_{T} [GeV]; Tracking particles / 1.0 GeV",  92,  8.0, 100.0);
  TH1F* h_tp_eta   = new TH1F("tp_eta",  ";Tracking particle #eta; Tracking particles / 0.1",             50, -2.5,   2.5);
  TH1F* h_tp_eta_L = new TH1F("tp_eta_L",";Tracking particle #eta; Tracking particles / 0.1",             50, -2.5,   2.5);
  TH1F* h_tp_eta_H = new TH1F("tp_eta_H",";Tracking particle #eta; Tracking particles / 0.1",             50, -2.5,   2.5);


  TH1F* h_match_tp_pt    = new TH1F("match_tp_pt",   ";Tracking particle p_{T} [GeV]; Tracking particles / 1.0 GeV", 100,  0,   100.0);
  TH1F* h_match_tp_pt_L  = new TH1F("match_tp_pt_L", ";Tracking particle p_{T} [GeV]; Tracking particles / 0.1 GeV",  80,  0,     8.0);
  TH1F* h_match_tp_pt_LC = new TH1F("match_tp_pt_LC",";Tracking particle p_{T} [GeV]; Tracking particles / 0.1 GeV",  80,  0,     8.0);
  TH1F* h_match_tp_pt_H  = new TH1F("match_tp_pt_H", ";Tracking particle p_{T} [GeV]; Tracking particles / 0.1 GeV",  92,  8.0, 100.0);
  TH1F* h_match_tp_eta   = new TH1F("match_tp_eta",  ";Tracking particle #eta; Tracking particles / 0.1",             50, -2.5,   2.5);
  TH1F* h_match_tp_eta_L = new TH1F("match_tp_eta_L",";Tracking particle #eta; Tracking particles / 0.1",             50, -2.5,   2.5);
  TH1F* h_match_tp_eta_H = new TH1F("match_tp_eta_H",";Tracking particle #eta; Tracking particles / 0.1",             50, -2.5,   2.5);


  // ----------------------------------------------------------------------------------------------------------------
  // resolution vs. pt histograms

  // ----------------------------------------------
  // for new versions of resolution vs pt/eta plots
  unsigned int nBinsPtRes = 500;
  double maxPtRes = 30.;

  unsigned int nBinsPtRelRes = 1000;
  double maxPtRelRes = 10.;

  unsigned int nBinsEtaRes = 500;
  double maxEtaRes = 0.1;

  unsigned int nBinsPhiRes = 500;
  double maxPhiRes = 0.2;

  unsigned int nBinsZ0Res = 100;
  double maxZ0Res = 4.0;
  // ----------------------------------------------
  
  const int nRANGE = 20;
  TString ptrange[nRANGE] = {"0-5","5-10", "10-15","15-20","20-25","25-30","30-35","35-40","40-45","45-50","50-55",
  			     "55-60","60-65","65-70","70-75","75-80","80-85","85-90","90-95","95-100"};

  TH1F* h_absResVsPt_pt[nRANGE];
  TH1F* h_absResVsPt_ptRel[nRANGE];
  TH1F* h_absResVsPt_z0[nRANGE];
  TH1F* h_absResVsPt_phi[nRANGE];
  TH1F* h_absResVsPt_eta[nRANGE];
  TH1F* h_absResVsPt_d0[nRANGE];

  TH1F* h_absResVsPt_pt_L[nRANGE];
  TH1F* h_absResVsPt_ptRel_L[nRANGE];
  TH1F* h_absResVsPt_z0_L[nRANGE];
  TH1F* h_absResVsPt_phi_L[nRANGE];
  TH1F* h_absResVsPt_eta_L[nRANGE];
  TH1F* h_absResVsPt_d0_L[nRANGE];

  for (int i=0; i<nRANGE; i++) {
    h_absResVsPt_pt[i]    = new TH1F("absResVsPt_pt_"+ptrange[i],   ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1",   nBinsPtRes,    0, maxPtRes);
    h_absResVsPt_ptRel[i] = new TH1F("absResVsPt_ptRel_"+ptrange[i],";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02",nBinsPtRelRes, 0, maxPtRelRes);
    h_absResVsPt_z0[i]    = new TH1F("absResVsPt_z0_"+ptrange[i],   ";z_{0} residual (L1 - sim) [GeV]; L1 tracks / 0.1",   nBinsZ0Res,    0, maxZ0Res);
    h_absResVsPt_phi[i]   = new TH1F("absResVsPt_phi_"+ptrange[i],  ";#phi residual (L1 - sim) [GeV]; L1 tracks / 0.1",    nBinsPhiRes,   0, maxPhiRes);
    h_absResVsPt_eta[i]   = new TH1F("absResVsPt_eta_"+ptrange[i],  ";#eta residual (L1 - sim) [GeV]; L1 tracks / 0.1",    nBinsEtaRes,   0, maxEtaRes);
    h_absResVsPt_d0[i]    = new TH1F("absResVsPt_d0_"+ptrange[i],   ";d_{0}residual (L1 - sim) [GeV]; L1 tracks / 0.1",    100,           0, 0.02);
  }

  const int nRANGE_L = 10;
  TString ptrange_L[nRANGE] = {"3-3.5", "3.5-4", "4-4.5", "4.5-5", "5-5.5", "5.5-6", "6-6.5", "6.5-7", "7-7.5", "7.5-8" };

  for (int i=0; i<nRANGE; i++) {
    h_absResVsPt_pt_L[i]    = new TH1F("absResVsPt_L_pt_"+ptrange_L[i],   ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1",   nBinsPtRes,    0, maxPtRes);
    h_absResVsPt_ptRel_L[i] = new TH1F("absResVsPt_L_ptRel_"+ptrange_L[i],";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02",nBinsPtRelRes, 0, maxPtRelRes);
    h_absResVsPt_z0_L[i]    = new TH1F("absResVsPt_L_z0_"+ptrange_L[i],   ";z_{0} residual (L1 - sim) [GeV]; L1 tracks / 0.1",   nBinsZ0Res,    0, maxZ0Res);
    h_absResVsPt_phi_L[i]   = new TH1F("absResVsPt_L_phi_"+ptrange_L[i],  ";#phi residual (L1 - sim) [GeV]; L1 tracks / 0.1",    nBinsPhiRes,   0, maxPhiRes);
    h_absResVsPt_eta_L[i]   = new TH1F("absResVsPt_L_eta_"+ptrange_L[i],  ";#eta residual (L1 - sim) [GeV]; L1 tracks / 0.1",    nBinsEtaRes,   0, maxEtaRes);
    h_absResVsPt_d0_L[i]    = new TH1F("absResVsPt_L_d0_"+ptrange_L[i],   ";d_{0}residual (L1 - sim) [GeV]; L1 tracks / 0.1",    100,           0, 0.02);
  }

  // resolution vs. eta histograms
  const int nETARANGE = 24;
  TString etarange[nETARANGE] = {"0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0",
				 "1.1","1.2","1.3","1.4","1.5","1.6","1.7","1.8","1.9","2.0",
				 "2.1","2.2","2.3","2.4"};

  TH1F* h_absResVsEta_eta[nETARANGE];
  TH1F* h_absResVsEta_z0[nETARANGE];
  TH1F* h_absResVsEta_phi[nETARANGE];
  TH1F* h_absResVsEta_ptRel[nETARANGE];
  TH1F* h_absResVsEta_d0[nETARANGE];

  TH1F* h_absResVsEta_eta_L[nETARANGE];
  TH1F* h_absResVsEta_z0_L[nETARANGE];
  TH1F* h_absResVsEta_phi_L[nETARANGE];
  TH1F* h_absResVsEta_ptRel_L[nETARANGE];
  TH1F* h_absResVsEta_d0_L[nETARANGE];

  TH1F* h_absResVsEta_eta_H[nETARANGE];
  TH1F* h_absResVsEta_z0_H[nETARANGE];
  TH1F* h_absResVsEta_phi_H[nETARANGE];
  TH1F* h_absResVsEta_ptRel_H[nETARANGE];
  TH1F* h_absResVsEta_d0_H[nETARANGE];

  for (int i=0; i<nETARANGE; i++) {
    h_absResVsEta_eta[i]   = new TH1F("absResVsEta_eta_"+etarange[i],  ";#eta residual (L1 - sim) [GeV]; L1 tracks / 0.1",     nBinsEtaRes,   0, maxEtaRes);
    h_absResVsEta_z0[i]    = new TH1F("absResVsEta_z0_"+etarange[i],   ";|z_{0} residual (L1 - sim)| [cm]; L1 tracks / 0.01",  nBinsZ0Res,    0, maxZ0Res);
    h_absResVsEta_phi[i]   = new TH1F("absResVsEta_phi_"+etarange[i],  ";#phi residual (L1 - sim) [GeV]; L1 tracks / 0.1",     nBinsPhiRes,   0, maxPhiRes);    
    h_absResVsEta_ptRel[i] = new TH1F("absResVsEta_ptRel_"+etarange[i],";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", nBinsPtRelRes, 0, maxPtRelRes);
    h_absResVsEta_d0[i]    = new TH1F("absResVsEta_d0_"+etarange[i],   ";d_{0}residual (L1 - sim) [GeV]; L1 tracks / 0.1",     100,           0, 0.02);

    h_absResVsEta_eta_L[i]   = new TH1F("absResVsEta_eta_L_"+etarange[i],  ";#eta residual (L1 - sim) [GeV]; L1 tracks / 0.1",     nBinsEtaRes,   0, maxEtaRes);
    h_absResVsEta_z0_L[i]    = new TH1F("absResVsEta_z0_L_"+etarange[i],   ";|z_{0} residual (L1 - sim)| [cm]; L1 tracks / 0.01",  nBinsZ0Res,    0, maxZ0Res);
    h_absResVsEta_phi_L[i]   = new TH1F("absResVsEta_phi_L_"+etarange[i],  ";#phi residual (L1 - sim) [GeV]; L1 tracks / 0.1",     nBinsPhiRes,   0, maxPhiRes);    
    h_absResVsEta_ptRel_L[i] = new TH1F("absResVsEta_ptRel_L_"+etarange[i],";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", nBinsPtRelRes, 0, maxPtRelRes);
    h_absResVsEta_d0_L[i]    = new TH1F("absResVsEta_d0_L_"+etarange[i],   ";d_{0}residual (L1 - sim) [GeV]; L1 tracks / 0.1",     100,           0, 0.02);

    h_absResVsEta_eta_H[i]   = new TH1F("absResVsEta_eta_H_"+etarange[i],  ";#eta residual (L1 - sim) [GeV]; L1 tracks / 0.1",     nBinsEtaRes,   0, maxEtaRes);
    h_absResVsEta_z0_H[i]    = new TH1F("absResVsEta_z0_H_"+etarange[i],   ";|z_{0} residual (L1 - sim)| [cm]; L1 tracks / 0.01",  nBinsZ0Res,    0, maxZ0Res);
    h_absResVsEta_phi_H[i]   = new TH1F("absResVsEta_phi_H_"+etarange[i],  ";#phi residual (L1 - sim) [GeV]; L1 tracks / 0.1",     nBinsPhiRes,   0, maxPhiRes);    
    h_absResVsEta_ptRel_H[i] = new TH1F("absResVsEta_ptRel_H_"+etarange[i],";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", nBinsPtRelRes, 0, maxPtRelRes);
    h_absResVsEta_d0_H[i]    = new TH1F("absResVsEta_d0_H_"+etarange[i],   ";d_{0}residual (L1 - sim) [GeV]; L1 tracks / 0.1",     100,           0, 0.02);
  }

  // resolution vs phi 

  const int nPHIRANGE = 32;
  TString phirange[nPHIRANGE] = {"-3.0","-2.8","-2.6","-2.4","-2.2",
			     "-2.0","-1.8","-1.6","-1.4","-1.2",
			     "-1.0","-0.8","-0.6","-0.4","-0.2",
			     "0.0","0.2","0.4","0.6","0.8",
			     "1.0","1.2","1.4","1.6","1.8",
			     "2.0","2.2","2.4","2.6","2.8",
			     "3.0","3.2"};

  TH1F* h_absResVsPhi_pt[nPHIRANGE];
  TH1F* h_absResVsPhi_ptRel[nPHIRANGE];

  for (int i=0; i<nPHIRANGE; i++) {
    h_absResVsPhi_pt[i]    = new TH1F("absResVsPt_pt_"+phirange[i],   ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1",   nBinsPtRes,    0, maxPtRes);
    h_absResVsPhi_ptRel[i] = new TH1F("absResVsPt_ptRel_"+phirange[i],";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02",nBinsPtRelRes, 0, maxPtRelRes);
  }


  // total track rates 
  TH1F* h_trk_vspt = new TH1F("trk_vspt", ";Track p_{T} [GeV]; ",40,0,20);
  TH1F* h_tp_vspt  = new TH1F("tp_vspt", ";TP p_{T} [GeV]; ",40,0,20);


  TH1F* h_tp_z0    = new TH1F("tp_z0",   ";Tracking particle z_{0} [cm]; Tracking particles / 1.0 cm",    50, -25.0, 25.0);
  TH1F* h_tp_z0_L    = new TH1F("tp_z0_L",   ";Tracking particle z_{0} [cm]; Tracking particles / 1.0 cm",    50, -25.0, 25.0);
  TH1F* h_tp_z0_H    = new TH1F("tp_z0_H",   ";Tracking particle z_{0} [cm]; Tracking particles / 1.0 cm",    50, -25.0, 25.0);

  TH1F* h_match_tp_z0    = new TH1F("match_tp_z0",   ";Tracking particle z_{0} [cm]; Tracking particles / 1.0 cm",    50, -25.0, 25.0);
  TH1F* h_match_tp_z0_L    = new TH1F("match_tp_z0_L",   ";Tracking particle z_{0} [cm]; Tracking particles / 1.0 cm",    50, -25.0, 25.0);
  TH1F* h_match_tp_z0_H    = new TH1F("match_tp_z0_H",   ";Tracking particle z_{0} [cm]; Tracking particles / 1.0 cm",    50, -25.0, 25.0);


  // ----------------------------------------------------------------------------------------------------------------
  //
  //       ******************   additional histograms drawn if using 'detailed' option   ******************
  //
  // ----------------------------------------------------------------------------------------------------------------

  TH1F* h_tp_phi   = new TH1F("tp_phi",  ";Tracking particle #phi [rad]; Tracking particles / 0.1",       64, -3.2,   3.2);
  TH1F* h_tp_d0    = new TH1F("tp_d0",   ";Tracking particle d_{0} [cm]; Tracking particles / 0.0004 cm",100, -0.02, 0.02);

  TH1F* h_match_tp_phi   = new TH1F("match_tp_phi",  ";Tracking particle #phi [rad]; Tracking particles / 0.1",       64, -3.2,   3.2);
  TH1F* h_match_tp_d0    = new TH1F("match_tp_d0",   ";Tracking particle d_{0} [cm]; Tracking particles / 0.0004 cm",100, -0.02,   0.02);

  TH1F* h_match_trk_nstub   = new TH1F("match_trk_nstub",   ";Number of stubs; L1 tracks / 1.0", 15, 0, 15);
  TH1F* h_match_trk_nstub_C = new TH1F("match_trk_nstub_C", ";Number of stubs; L1 tracks / 1.0", 15, 0, 15);
  TH1F* h_match_trk_nstub_I = new TH1F("match_trk_nstub_I", ";Number of stubs; L1 tracks / 1.0", 15, 0, 15);
  TH1F* h_match_trk_nstub_F = new TH1F("match_trk_nstub_F", ";Number of stubs; L1 tracks / 1.0", 15, 0, 15);

  // chi2 histograms
  // note: last bin is an overflow bin
  TH1F* h_match_trk_chi2     = new TH1F("match_trk_chi2",     ";#chi^{2}; L1 tracks / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_C_L = new TH1F("match_trk_chi2_C_L", ";#chi^{2}; L1 tracks / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_I_L = new TH1F("match_trk_chi2_I_L", ";#chi^{2}; L1 tracks / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_F_L = new TH1F("match_trk_chi2_F_L", ";#chi^{2}; L1 tracks / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_C_H = new TH1F("match_trk_chi2_C_H", ";#chi^{2}; L1 tracks / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_I_H = new TH1F("match_trk_chi2_I_H", ";#chi^{2}; L1 tracks / 1.0", 100, 0, 100);
  TH1F* h_match_trk_chi2_F_H = new TH1F("match_trk_chi2_F_H", ";#chi^{2}; L1 tracks / 1.0", 100, 0, 100);

  // chi2/dof histograms
  // note: lastbin is an overflow bin
  TH1F* h_match_trk_chi2_dof     = new TH1F("match_trk_chi2_dof",     ";#chi^{2} / D.O.F.; L1 tracks / 0.2", 100, 0, 20);
  TH1F* h_match_trk_chi2_dof_C_L = new TH1F("match_trk_chi2_dof_C_L", ";#chi^{2} / D.O.F.; L1 tracks / 0.2", 100, 0, 20);
  TH1F* h_match_trk_chi2_dof_I_L = new TH1F("match_trk_chi2_dof_I_L", ";#chi^{2} / D.O.F.; L1 tracks / 0.2", 100, 0, 20);
  TH1F* h_match_trk_chi2_dof_F_L = new TH1F("match_trk_chi2_dof_F_L", ";#chi^{2} / D.O.F.; L1 tracks / 0.2", 100, 0, 20);
  TH1F* h_match_trk_chi2_dof_C_H = new TH1F("match_trk_chi2_dof_C_H", ";#chi^{2} / D.O.F.; L1 tracks / 0.2", 100, 0, 20);
  TH1F* h_match_trk_chi2_dof_I_H = new TH1F("match_trk_chi2_dof_I_H", ";#chi^{2} / D.O.F.; L1 tracks / 0.2", 100, 0, 20);
  TH1F* h_match_trk_chi2_dof_F_H = new TH1F("match_trk_chi2_dof_F_H", ";#chi^{2} / D.O.F.; L1 tracks / 0.2", 100, 0, 20);


  // ----------------------------------------------------------------------------------------------------------------
  // resolution histograms
  TH1F* h_res_pt    = new TH1F("res_pt",    ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.05",   200,-5.0,   5.0);
  TH1F* h_res_ptRel = new TH1F("res_ptRel", ";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.01", 200,-1.0,   1.0);
  TH1F* h_res_eta   = new TH1F("res_eta",   ";#eta residual (L1 - sim); L1 tracks / 0.0002",        100,-0.01,  0.01);
  TH1F* h_res_phi   = new TH1F("res_phi",   ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001",  100,-0.005, 0.005);

  TH1F* h_res_z0    = new TH1F("res_z0",    ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1.0, 1.0);
  TH1F* h_res_z0_C  = new TH1F("res_z0_C",  ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1.0, 1.0);
  TH1F* h_res_z0_I  = new TH1F("res_z0_I",  ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1.0, 1.0);
  TH1F* h_res_z0_F  = new TH1F("res_z0_F",  ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1.0, 1.0);
  TH1F* h_res_z0_L  = new TH1F("res_z0_L",  ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1.0, 1.0);
  TH1F* h_res_z0_H  = new TH1F("res_z0_H",  ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1.0, 1.0);
  
  TH1F* h_res_z0_C_L = new TH1F("res_z0_C_L", ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100,(-1)*1.0, 1.0);
  TH1F* h_res_z0_I_L = new TH1F("res_z0_I_L", ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100,(-1)*1.0, 1.0);
  TH1F* h_res_z0_F_L = new TH1F("res_z0_F_L", ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100,(-1)*1.0, 1.0);
  TH1F* h_res_z0_C_H = new TH1F("res_z0_C_H", ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100,(-1)*1.0, 1.0);
  TH1F* h_res_z0_I_H = new TH1F("res_z0_I_H", ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100,(-1)*1.0, 1.0);
  TH1F* h_res_z0_F_H = new TH1F("res_z0_F_H", ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100,(-1)*1.0, 1.0);

  TH1F* h_res_d0 = new TH1F("res_d0", ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0002 cm", 200,-0.02,0.02);
  TH1F* h_res_d0_C = new TH1F("res_d0_C", ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res_d0_I = new TH1F("res_d0_I", ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res_d0_F = new TH1F("res_d0_F", ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res_d0_L = new TH1F("res_d0_L", ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res_d0_H = new TH1F("res_d0_H", ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0001 cm", 200,-0.05,0.05);

  TH1F* h_res_d0_C_L = new TH1F("res_d0_C_L", ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res_d0_I_L = new TH1F("res_d0_I_L", ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res_d0_F_L = new TH1F("res_d0_F_L", ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res_d0_C_H = new TH1F("res_d0_C_H", ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res_d0_I_H = new TH1F("res_d0_I_H", ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0001 cm", 200,-0.05,0.05);
  TH1F* h_res_d0_F_H = new TH1F("res_d0_F_H", ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0001 cm", 200,-0.05,0.05);

  // ----------------------------------------------------------------------------------------------------------------
  // more resolution vs pt 

  TH1F* h_resVsPt_pt[nRANGE];
  TH1F* h_resVsPt_pt_C[nRANGE];
  TH1F* h_resVsPt_pt_I[nRANGE];
  TH1F* h_resVsPt_pt_F[nRANGE];

  TH1F* h_resVsPt_ptRel[nRANGE];
  TH1F* h_resVsPt_ptRel_C[nRANGE];
  TH1F* h_resVsPt_ptRel_I[nRANGE];
  TH1F* h_resVsPt_ptRel_F[nRANGE];

  TH1F* h_resVsPt_z0[nRANGE];
  TH1F* h_resVsPt_z0_C[nRANGE];
  TH1F* h_resVsPt_z0_I[nRANGE];
  TH1F* h_resVsPt_z0_F[nRANGE];

  TH1F* h_resVsPt_phi[nRANGE];
  TH1F* h_resVsPt_phi_C[nRANGE];
  TH1F* h_resVsPt_phi_I[nRANGE];
  TH1F* h_resVsPt_phi_F[nRANGE];

  TH1F* h_resVsPt_eta[nRANGE];
  TH1F* h_resVsPt_d0[nRANGE];

  for (int i=0; i<nRANGE; i++) {
    h_resVsPt_pt[i]   = new TH1F("resVsPt_pt_"+ptrange[i],   ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1", 100, -5.0, 5.0);
    h_resVsPt_pt_C[i] = new TH1F("resVsPt_pt_C_"+ptrange[i], ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1", 100, -5.0, 5.0);
    h_resVsPt_pt_I[i] = new TH1F("resVsPt_pt_I_"+ptrange[i], ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1", 100, -5.0, 5.0);
    h_resVsPt_pt_F[i] = new TH1F("resVsPt_pt_F_"+ptrange[i], ";p_{T} residual (L1 - sim) [GeV]; L1 tracks / 0.1", 100, -5.0, 5.0);

    h_resVsPt_ptRel[i]   = new TH1F("resVsPt_ptRel_"+ptrange[i],   ";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 300, -0.15, 0.15);
    h_resVsPt_ptRel_C[i] = new TH1F("resVsPt_ptRel_c_"+ptrange[i], ";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 300, -0.15, 0.15);
    h_resVsPt_ptRel_I[i] = new TH1F("resVsPt_ptRel_I_"+ptrange[i], ";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 300, -0.15, 0.15);
    h_resVsPt_ptRel_F[i] = new TH1F("resVsPt_ptRel_F_"+ptrange[i], ";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 300, -0.15, 0.15);

    h_resVsPt_z0[i]   = new TH1F("resVsPt_z0_"+ptrange[i],   ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1, 1);
    h_resVsPt_z0_C[i] = new TH1F("resVsPt_z0_C_"+ptrange[i], ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1, 1);
    h_resVsPt_z0_I[i] = new TH1F("resVsPt_z0_I_"+ptrange[i], ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1, 1);
    h_resVsPt_z0_F[i] = new TH1F("resVsPt_z0_F_"+ptrange[i], ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.02", 100, -1, 1);

    h_resVsPt_phi[i]   = new TH1F("resVsPt_phi_"+ptrange[i],   ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100, -0.005, 0.005);
    h_resVsPt_phi_C[i] = new TH1F("resVsPt_phi_C_"+ptrange[i], ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100, -0.005, 0.005);
    h_resVsPt_phi_I[i] = new TH1F("resVsPt_phi_I_"+ptrange[i], ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100, -0.005, 0.005);
    h_resVsPt_phi_F[i] = new TH1F("resVsPt_phi_F_"+ptrange[i], ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100, -0.005, 0.005);

    h_resVsPt_eta[i] = new TH1F("resVsPt_eta_"+ptrange[i], ";#eta residual (L1 - sim); L1 tracks / 0.0002", 100, -0.01, 0.01);

    h_resVsPt_d0[i]  = new TH1F("resVsPt_d0_"+ptrange[i],   ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.0004",   100, -0.02, 0.02);
  }

  // ----------------------------------------------------------------------------------------------------------------
  // more resolution vs eta

  TH1F* h_resVsEta_eta[nETARANGE];
  TH1F* h_resVsEta_eta_L[nETARANGE];
  TH1F* h_resVsEta_eta_H[nETARANGE];

  TH1F* h_resVsEta_z0[nETARANGE];
  TH1F* h_resVsEta_z0_L[nETARANGE];
  TH1F* h_resVsEta_z0_H[nETARANGE];

  TH1F* h_resVsEta_phi[nETARANGE];
  TH1F* h_resVsEta_phi_L[nETARANGE];
  TH1F* h_resVsEta_phi_H[nETARANGE];

  TH1F* h_resVsEta_ptRel[nETARANGE];
  TH1F* h_resVsEta_ptRel_L[nETARANGE];
  TH1F* h_resVsEta_ptRel_H[nETARANGE];

  TH1F* h_resVsEta_d0[nETARANGE];
  TH1F* h_resVsEta_d0_L[nETARANGE];
  TH1F* h_resVsEta_d0_H[nETARANGE];

  for (int i=0; i<nETARANGE; i++) {
    h_resVsEta_eta[i]   = new TH1F("resVsEta_eta_"+etarange[i],   ";#eta residual (L1 - sim); L1 tracks / 0.0002", 100,-0.01,0.01);
    h_resVsEta_eta_L[i] = new TH1F("resVsEta_eta_L_"+etarange[i], ";#eta residual (L1 - sim); L1 tracks / 0.0002", 100,-0.01,0.01);
    h_resVsEta_eta_H[i] = new TH1F("resVsEta_eta_H_"+etarange[i], ";#eta residual (L1 - sim); L1 tracks / 0.0002", 100,-0.01,0.01);

    h_resVsEta_z0[i]   = new TH1F("resVsEta2_z0_"+etarange[i],   ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.01",   100,-1, 1);
    h_resVsEta_z0_L[i] = new TH1F("resVsEta2_z0_L_"+etarange[i], ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.01",   100,-1, 1);
    h_resVsEta_z0_H[i] = new TH1F("resVsEta2_z0_H_"+etarange[i], ";z_{0} residual (L1 - sim) [cm]; L1 tracks / 0.01",   100,-1, 1);

    h_resVsEta_phi[i]   = new TH1F("resVsEta2_phi_"+etarange[i],   ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100,-0.005,0.005);
    h_resVsEta_phi_L[i] = new TH1F("resVsEta2_phi_L_"+etarange[i], ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100,-0.005,0.005);
    h_resVsEta_phi_H[i] = new TH1F("resVsEta2_phi_H_"+etarange[i], ";#phi residual (L1 - sim) [rad]; L1 tracks / 0.0001", 100,-0.005,0.005);

    h_resVsEta_ptRel[i]   = new TH1F("resVsEta2_ptRel_"+etarange[i],  ";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.01", 100,-0.5,0.5);
    h_resVsEta_ptRel_L[i] = new TH1F("resVsEta2_ptRel_L_"+etarange[i],";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 100,-0.1,0.1);
    h_resVsEta_ptRel_H[i] = new TH1F("resVsEta2_ptRel_H_"+etarange[i],";p_{T} residual (L1 - sim) / p_{T}; L1 tracks / 0.02", 100,-0.25,0.25);

    h_resVsEta_d0[i]    = new TH1F("resVsEta_d0_"+etarange[i],   ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.004",    100,-0.02, 0.02);
    h_resVsEta_d0_L[i]  = new TH1F("resVsEta_d0_L_"+etarange[i], ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.004",    100,-0.02, 0.02);
    h_resVsEta_d0_H[i]  = new TH1F("resVsEta_d0_H_"+etarange[i], ";d_{0} residual (L1 - sim) [cm]; L1 tracks / 0.004",    100,-0.02, 0.02);

  }
  // ----------------------------------------------------------------------------------------------------------------



  // ----------------------------------------------------------------------------------------------------------------
  // additional ones for sum pt in jets
  /*
  TH1F* h_jet_tp_sumpt_vspt  = new TH1F("jet_tp_sumpt_vspt",  ";sum(TP p_{T}) [GeV]; Gen jets / 5.0 GeV", 20, 0, 200.0);
  TH1F* h_jet_trk_sumpt_vspt = new TH1F("jet_trk_sumpt_vspt", ";sum(TP p_{T}) [GeV]; Gen jets / 5.0 GeV", 20, 0, 200.0);
  TH1F* h_jet_matchtrk_sumpt_vspt = new TH1F("jet_matchtrk_sumpt_vspt", ";sum(TP p_{T}) [GeV]; Gen jets / 5.0 GeV", 20, 0, 200.0);

  TH1F* h_jet_tp_sumpt_vseta  = new TH1F("jet_tp_sumpt_vseta",  ";Gen jet #eta; Gen jets / 0.2", 24, -2.4, 2.4);
  TH1F* h_jet_trk_sumpt_vseta = new TH1F("jet_trk_sumpt_vseta", ";Gen jet #eta; Gen jets / 0.2", 24, -2.4, 2.4);
  TH1F* h_jet_matchtrk_sumpt_vseta = new TH1F("jet_matchtrk_sumpt_vseta", ";Gen jet #eta; Gen jets / 0.2", 24, -2.4, 2.4);

  h_jet_tp_sumpt_vseta->Sumw2();
  h_jet_tp_sumpt_vspt->Sumw2();
  h_jet_trk_sumpt_vseta->Sumw2();
  h_jet_trk_sumpt_vspt->Sumw2();
  h_jet_matchtrk_sumpt_vseta->Sumw2();
  h_jet_matchtrk_sumpt_vspt->Sumw2();
  */


  // ----------------------------------------------------------------------------------------------------------------
  // number of tracks per event 

  TH1F* h_ntrk_pt2  = new TH1F("ntrk_pt2", ";# tracks (p_{T} > 2 GeV) / event; Events", 300, 0, 300.0);
  TH1F* h_ntrk_pt3  = new TH1F("ntrk_pt3", ";# tracks (p_{T} > 3 GeV) / event; Events", 300, 0, 300.0);
  TH1F* h_ntrk_pt10 = new TH1F("ntrk_pt10", ";# tracks (p_{T} > 10 GeV) / event; Events", 100, 0, 100.0);



  // ----------------------------------------------------------------------------------------------------------------
  //        * * * * *     S T A R T   O F   A C T U A L   R U N N I N G   O N   E V E N T S     * * * * *
  // ----------------------------------------------------------------------------------------------------------------
  
  int nevt = tree->GetEntries();
  cout << "number of events = " << nevt << endl;


  // ----------------------------------------------------------------------------------------------------------------
  // event loop
  for (int i=0; i<nevt; i++) {

    tree->GetEntry(i,0);
  
    /*
    // ----------------------------------------------------------------------------------------------------------------
    // sumpt in jets
    if (TP_select_injet > 0) {
      for (int ij=0; ij<(int)jet_tp_sumpt->size(); ij++) {
	
	float fraction = 0;
	float fractionMatch = 0;
	if (jet_tp_sumpt->at(ij) > 0) {
	  fraction = jet_trk_sumpt->at(ij)/jet_tp_sumpt->at(ij);
	  fractionMatch = jet_matchtrk_sumpt->at(ij)/jet_tp_sumpt->at(ij);
	}
	
	h_jet_tp_sumpt_vspt->Fill(jet_tp_sumpt->at(ij),1.0);
	h_jet_trk_sumpt_vspt->Fill(jet_tp_sumpt->at(ij),fraction);
	h_jet_matchtrk_sumpt_vspt->Fill(jet_tp_sumpt->at(ij),fractionMatch);
	
	h_jet_tp_sumpt_vseta->Fill(jet_eta->at(ij),1.0);
	h_jet_trk_sumpt_vseta->Fill(jet_eta->at(ij),fraction);
	h_jet_matchtrk_sumpt_vseta->Fill(jet_eta->at(ij),fractionMatch);
      }
    }
    */


    // ----------------------------------------------------------------------------------------------------------------
    // track loop for total rates

    int ntrkevt_pt2 = 0;
    int ntrkevt_pt3 = 0;
    int ntrkevt_pt10 = 0;

    for (int it=0; it<(int)trk_pt->size(); it++) {

      /*           
      // only look at tracks in (ttbar) jets ?
      if (TP_select_injet > 0) {
	if (TP_select_injet == 1 && trk_injet->at(it) == 0) continue;       
	if (TP_select_injet == 2 && trk_injet_highpt->at(it) == 0) continue;       
      }
      */

      if (trk_pt->at(it) > 2.0 && fabs(trk_eta->at(it))<2.5) {
	ntrk_pt2++;
	ntrkevt_pt2++;
	h_trk_vspt->Fill(trk_pt->at(it));
	//if (trk_nstub->at(it) > 3) ntrk_4stub_pt3++;
	//if (trk_nstub->at(it) > 3 && trk_chi2->at(it) < 100.) ntrk_4stubchi2_pt3++;
      }
      if (trk_pt->at(it) > 3.0 && fabs(trk_eta->at(it))<2.5) {
	ntrk_pt3++;
	ntrkevt_pt3++;
	//if (trk_nstub->at(it) > 3) ntrk_4stub_pt3++;
	//if (trk_nstub->at(it) > 3 && trk_chi2->at(it) < 100.) ntrk_4stubchi2_pt3++;
      }
      if (trk_pt->at(it) > 10.0 && fabs(trk_eta->at(it))<2.5) {
	ntrk_pt10++;
	ntrkevt_pt10++;
	//if (trk_nstub->at(it) > 3) ntrk_4stub_pt10++;
	//if (trk_nstub->at(it) > 3 && trk_chi2->at(it) < 100.) ntrk_4stubchi2_pt10++;
      }
    }

    h_ntrk_pt2->Fill(ntrkevt_pt2);
    h_ntrk_pt3->Fill(ntrkevt_pt3);
    h_ntrk_pt10->Fill(ntrkevt_pt10);


    // ----------------------------------------------------------------------------------------------------------------
    // tracking particle loop
    for (int it=0; it<(int)tp_pt->size(); it++) {

      /*
      // only look at TPs in (ttbar) jets ?
      if (TP_select_injet > 0) {
	if (TP_select_injet == 1 && tp_injet->at(it) == 0) continue;       
	if (TP_select_injet == 2 && tp_injet_highpt->at(it) == 0) continue;       
      }
      */
      
      // total track rates
      //if (tp_nstub->at(it) >= 4 && tp_nstublayer->at(it) >= 4 && fabs(tp_dxy->at(it)) < 1 && fabs(tp_eta->at(it)) < TP_maxEta) {
      if (fabs(tp_dxy->at(it)) < 1 && fabs(tp_eta->at(it)) < TP_maxEta) { //the stub requirements are applied when making the ntuples
	if (tp_pt->at(it) > 2.0) {
	  ntp_pt2++;
	  h_tp_vspt->Fill(tp_pt->at(it));
	}
	if (tp_pt->at(it) > 3.0) ntp_pt3++;
	if (tp_pt->at(it) > 10.0) ntp_pt10++;
      }

      // cut on PDG ID at plot stage?
      if (TP_select_pdgid != 0) {
	if (abs(tp_pdgid->at(it)) != abs(TP_select_pdgid)) continue;
      }
      
      // cut on event ID (eventid=0 means the TP is from the primary interaction, so *not* selecting only eventid=0 means including stuff from pileup)
      if (TP_select_eventid == 0 && tp_eventid->at(it) != 0) continue;


      if (tp_dxy->at(it) > 1) continue;
      if (tp_pt->at(it) < 0.2) continue;
      if (tp_pt->at(it) > TP_maxPt) continue;
      if (fabs(tp_eta->at(it)) > TP_maxEta) continue;

      h_tp_pt->Fill(tp_pt->at(it));
      if (tp_pt->at(it) < 8.0) h_tp_pt_L->Fill(tp_pt->at(it));
      else h_tp_pt_H->Fill(tp_pt->at(it));
      if (tp_pt->at(it) < 8.0 && fabs(tp_eta->at(it))<1.0) h_tp_pt_LC->Fill(tp_pt->at(it));
      
      if (tp_pt->at(it) > TP_minPt) {
	
	if (fabs(tp_eta->at(it)) < 1.0) n_all_eta1p0++;
	else if (fabs(tp_eta->at(it)) < 1.75) n_all_eta1p75++;
	else n_all_eta2p5++;
	
	h_tp_eta->Fill(tp_eta->at(it));
	h_tp_phi->Fill(tp_phi->at(it));
	h_tp_z0->Fill(tp_z0->at(it));
	h_tp_d0->Fill(tp_d0->at(it));
	
	if (tp_pt->at(it) < 8.0) {
    h_tp_eta_L->Fill(tp_eta->at(it));
    h_tp_z0_L->Fill(tp_z0->at(it));
  }
	else {
    h_tp_eta_H->Fill(tp_eta->at(it));
    h_tp_z0_H->Fill(tp_z0->at(it));
  }
	
      }
      
      
      // ----------------------------------------------------------------------------------------------------------------
      // was the tracking particle matched to a L1 track?
      if (tp_nmatch->at(it) < 1) continue;
      
      
      // ----------------------------------------------------------------------------------------------------------------
      // use only tracks with min X stubs
      if (matchtrk_nstub->at(it) < L1Tk_minNstub) continue;


      // ----------------------------------------------------------------------------------------------------------------
      // fill chi2 & chi2/dof histograms before making chi2 cut
      
      float chi2 = matchtrk_chi2->at(it);
      int ndof = 2*matchtrk_nstub->at(it)-4;
      float chi2dof = (float)chi2/ndof;
      if (chi2 > 100) chi2 = 99.9; //for overflow bin
      if (chi2dof > 20) chi2dof = 19.99; //for overflow bin
      
      
      if (tp_pt->at(it) > TP_minPt) { //TP pt > TP_minPt
	
	h_match_trk_chi2->Fill(chi2);
	h_match_trk_chi2_dof->Fill(chi2dof);
	
	// central eta
	if (fabs(tp_eta->at(it)) < 0.8) {
	  if (tp_pt->at(it) < 8.0) {
	    h_match_trk_chi2_C_L->Fill(chi2);
	    h_match_trk_chi2_dof_C_L->Fill(chi2dof);
	  } 
	  else {
	    h_match_trk_chi2_C_H->Fill(chi2);
	    h_match_trk_chi2_dof_C_H->Fill(chi2dof);
	  }
	}
	// intermediate eta
	else if (fabs(tp_eta->at(it)) < 1.6 && fabs(tp_eta->at(it)) >= 0.8) {
	  if (tp_pt->at(it) < 8.0) {
	    h_match_trk_chi2_I_L->Fill(chi2);
	    h_match_trk_chi2_dof_I_L->Fill(chi2dof);
	  }
	  else {
	    h_match_trk_chi2_I_H->Fill(chi2);
	    h_match_trk_chi2_dof_I_H->Fill(chi2dof);
	  }
	}
	// forward eta
	else if (fabs(tp_eta->at(it)) >= 1.6) {
	  if (tp_pt->at(it) < 8.0) {
	    h_match_trk_chi2_F_L->Fill(chi2);
	    h_match_trk_chi2_dof_F_L->Fill(chi2dof);
	  } 
	  else {
	    h_match_trk_chi2_F_H->Fill(chi2);
	    h_match_trk_chi2_dof_F_H->Fill(chi2dof);
	  }
	}
      }//end TP pt > TP_minPt
      
      
      // ----------------------------------------------------------------------------------------------------------------
      // cut on chi2?
      if (matchtrk_chi2->at(it) > L1Tk_maxChi2) continue;
      if (matchtrk_chi2->at(it)/ndof > L1Tk_maxChi2dof) continue;
      
      
      // use tight quality cut selection?
      /*
      if (matchtrk_nstub->at(it)==4) {
	if (fabs(matchtrk_eta->at(it))<2.2 && matchtrk_consistency->at(it)>10) continue;
	else if (fabs(matchtrk_eta->at(it))>2.2 && chi2dof>5.0) continue;
      }
      if (matchtrk_pt->at(it)>10.0 && chi2dof>5.0) continue;
      */
      
      // ----------------------------------------------------------------------------------------------------------------
      // more plots
      
      // fill matched track histograms
      h_match_tp_pt->Fill(tp_pt->at(it));
      if (tp_pt->at(it) < 8.0) h_match_tp_pt_L->Fill(tp_pt->at(it));
      else h_match_tp_pt_H->Fill(tp_pt->at(it));
      if (tp_pt->at(it) < 8.0 && fabs(tp_eta->at(it))<1.0) h_match_tp_pt_LC->Fill(tp_pt->at(it));
      
      if (tp_pt->at(it) > TP_minPt) {
	h_match_tp_eta->Fill(tp_eta->at(it));
	h_match_tp_phi->Fill(tp_phi->at(it));
	h_match_tp_z0->Fill(tp_z0->at(it));
	h_match_tp_d0->Fill(tp_d0->at(it));
	
	if (fabs(tp_eta->at(it)) < 1.0) n_match_eta1p0++;
	else if (fabs(tp_eta->at(it)) < 1.75) n_match_eta1p75++;
	else n_match_eta2p5++;
	
	if (tp_pt->at(it) < 8.0) {
    h_match_tp_eta_L->Fill(tp_eta->at(it));
    h_match_tp_z0_L->Fill(tp_z0->at(it));
  }
	else {
    h_match_tp_z0_H->Fill(tp_z0->at(it));
    h_match_tp_eta_H->Fill(tp_eta->at(it));
  }
      }
      
      
      // for the following, only consider TPs with pt > TP_minPt
      if (tp_pt->at(it) < TP_minPt) continue;
      
      
      // fill nstub histograms
      h_match_trk_nstub->Fill(matchtrk_nstub->at(it));
      if (fabs(tp_eta->at(it)) < 0.8) h_match_trk_nstub_C->Fill(matchtrk_nstub->at(it));
      else if (fabs(tp_eta->at(it)) < 1.6 && fabs(tp_eta->at(it)) >= 0.8) h_match_trk_nstub_I->Fill(matchtrk_nstub->at(it));
      else if (fabs(tp_eta->at(it)) >= 1.6) h_match_trk_nstub_F->Fill(matchtrk_nstub->at(it));
      
      
      // ----------------------------------------------------------------------------------------------------------------
      // fill resolution histograms
      
      h_res_pt   ->Fill(matchtrk_pt->at(it)  - tp_pt->at(it));
      h_res_ptRel->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
      h_res_eta  ->Fill(matchtrk_eta->at(it) - tp_eta->at(it));
      h_res_phi  ->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
      h_res_z0   ->Fill(matchtrk_z0->at(it)  - tp_z0->at(it));
      if (matchtrk_d0->at(it) < 999.) h_res_d0->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
      
      if (fabs(tp_eta->at(it)) < 0.8) h_res_z0_C->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
      else if (fabs(tp_eta->at(it)) < 1.6 && fabs(tp_eta->at(it)) >= 0.8) h_res_z0_I->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
      else if (fabs(tp_eta->at(it)) >= 1.6) h_res_z0_F->Fill(tp_z0->at(it) - tp_z0->at(it));
      
      
      if (tp_pt->at(it) < 8.0) {
	h_res_z0_L->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
	if (fabs(tp_eta->at(it)) < 1.0) h_res_z0_C_L->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
	else  h_res_z0_F_L->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
      }
      else {
	h_res_z0_H->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
	if (fabs(tp_eta->at(it)) < 1.0) h_res_z0_C_H->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
	else h_res_z0_F_H->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
      }
      
      
      if (matchtrk_d0->at(it) < 999.) {	
	if (fabs(tp_eta->at(it)) < 0.8) h_res_d0_C->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
	else if (fabs(tp_eta->at(it)) < 1.6 && fabs(tp_eta->at(it)) >= 0.8) h_res_d0_I->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
	else if (fabs(tp_eta->at(it)) >= 1.6) h_res_d0_F->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
	
	if (tp_pt->at(it) < 8.0) {
	  h_res_d0_L->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
	  if (fabs(tp_eta->at(it)) < 1.0) h_res_d0_C_L->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
	  else h_res_d0_F_L->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
	}
	else {
	  h_res_d0_H->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
	  if (fabs(tp_eta->at(it)) < 1.0) h_res_d0_C_H->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
	  else h_res_d0_F_H->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
	}
	
      }
      
      
      // ----------------------------------------------------------------------------------------------------------------
      // fill resolution vs. pt histograms    
      for (int im=0; im<nRANGE; im++) {
	if ( (tp_pt->at(it) > (float)im*5.0) && (tp_pt->at(it) < (float)im*5.0+5.0) ) {
	  h_resVsPt_pt[im]   ->Fill(matchtrk_pt->at(it)  - tp_pt->at(it));
	  h_resVsPt_ptRel[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	  h_resVsPt_eta[im]  ->Fill(matchtrk_eta->at(it) - tp_eta->at(it));
	  h_resVsPt_phi[im]  ->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
	  h_resVsPt_z0[im]   ->Fill(matchtrk_z0->at(it)  - tp_z0->at(it));

          h_absResVsPt_pt[im]   ->Fill( fabs( matchtrk_pt->at(it)  - tp_pt->at(it) ));
          h_absResVsPt_ptRel[im]->Fill( fabs( (matchtrk_pt->at(it) - tp_pt->at(it)) )/tp_pt->at(it) );
          h_absResVsPt_z0[im]   ->Fill( fabs( matchtrk_z0->at(it)  - tp_z0->at(it) ) );
          h_absResVsPt_phi[im]  ->Fill( fabs( matchtrk_phi->at(it) - tp_phi->at(it) ) );
          h_absResVsPt_eta[im]  ->Fill( fabs( matchtrk_eta->at(it) - tp_eta->at(it) ) );

	  if (fabs(tp_eta->at(it)) < 0.8) {
	    h_resVsPt_pt_C[im]   ->Fill(matchtrk_pt->at(it)  - tp_pt->at(it));
	    h_resVsPt_ptRel_C[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	    h_resVsPt_z0_C[im]   ->Fill(matchtrk_z0->at(it)  - tp_z0->at(it));
	    h_resVsPt_phi_C[im]  ->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
	  }
	  else if (fabs(tp_eta->at(it)) < 1.6 && fabs(tp_eta->at(it)) >= 0.8) {
	    h_resVsPt_pt_I[im]   ->Fill(matchtrk_pt->at(it)  - tp_pt->at(it));
	    h_resVsPt_ptRel_I[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	    h_resVsPt_z0_I[im]   ->Fill(matchtrk_z0->at(it)  - tp_z0->at(it));
	    h_resVsPt_phi_I[im]  ->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
	  }
	  else if (fabs(tp_eta->at(it)) >= 1.6) {
	    h_resVsPt_pt_F[im]   ->Fill(matchtrk_pt->at(it)  - tp_pt->at(it));
	    h_resVsPt_ptRel_F[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	    h_resVsPt_z0_F[im]   ->Fill(matchtrk_z0->at(it)  - tp_z0->at(it));
	    h_resVsPt_phi_F[im]  ->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
	  }
	  
	  if (matchtrk_d0->at(it) < 999) {
	    h_resVsPt_d0[im]->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
	    h_absResVsPt_d0[im]->Fill( fabs( matchtrk_d0->at(it) - tp_d0->at(it) ) );
	  }
	  
	}
      }
      
      for (int im=6; im<nRANGE_L+6; im++) {
	if ( (tp_pt->at(it) > (float)im*0.5 ) && (tp_pt->at(it) <= (float)im*0.5+0.5) ) {
	  h_absResVsPt_pt_L[im-6]   ->Fill( fabs( matchtrk_pt->at(it)  - tp_pt->at(it) ));
	  h_absResVsPt_ptRel_L[im-6]->Fill( fabs( (matchtrk_pt->at(it) - tp_pt->at(it)) )/tp_pt->at(it) );
	  h_absResVsPt_z0_L[im-6]   ->Fill( fabs( matchtrk_z0->at(it)  - tp_z0->at(it) ) );
	  h_absResVsPt_phi_L[im-6]  ->Fill( fabs( matchtrk_phi->at(it) - tp_phi->at(it) ) );
	  h_absResVsPt_eta_L[im-6]  ->Fill( fabs( matchtrk_eta->at(it) - tp_eta->at(it) ) );
	}
      }
      
      // ----------------------------------------------------------------------------------------------------------------
      // fill resolution vs. eta histograms
      for (int im=0; im<nETARANGE; im++) {
       if ( (fabs(tp_eta->at(it)) > (float)im*0.1) && (fabs(tp_eta->at(it)) < (float)im*0.1+0.1) ) {
	 h_resVsEta_ptRel[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	 h_resVsEta_eta[im]  ->Fill(matchtrk_eta->at(it) - tp_eta->at(it));
	 h_resVsEta_phi[im]  ->Fill(matchtrk_phi->at(it) - tp_phi->at(it));
	 h_resVsEta_z0[im]   ->Fill(matchtrk_z0->at(it)  - tp_z0->at(it));

         h_absResVsEta_ptRel[im]->Fill( fabs( (matchtrk_pt->at(it) - tp_pt->at(it)) )/tp_pt->at(it));
         h_absResVsEta_eta[im]  ->Fill( fabs( matchtrk_eta->at(it) - tp_eta->at(it) ) );
         h_absResVsEta_phi[im]  ->Fill( fabs( matchtrk_phi->at(it) - tp_phi->at(it) ) );
         h_absResVsEta_z0[im]   ->Fill( fabs( matchtrk_z0->at(it)  - tp_z0->at(it) ) );

	 if (tp_pt->at(it)<8.0) {
	   h_resVsEta_ptRel_L[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	   h_resVsEta_eta_L[im]->Fill(matchtrk_eta->at(it) - tp_eta->at(it));
	   h_resVsEta_z0_L[im]->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
	   h_resVsEta_phi_L[im]->Fill(matchtrk_phi->at(it) - tp_phi->at(it));

	   h_absResVsEta_ptRel_L[im]->Fill( fabs( (matchtrk_pt->at(it) - tp_pt->at(it)) )/tp_pt->at(it));
	   h_absResVsEta_eta_L[im]  ->Fill( fabs( matchtrk_eta->at(it) - tp_eta->at(it) ) );
	   h_absResVsEta_phi_L[im]  ->Fill( fabs( matchtrk_phi->at(it) - tp_phi->at(it) ) );
	   h_absResVsEta_z0_L[im]   ->Fill( fabs( matchtrk_z0->at(it)  - tp_z0->at(it) ) );

	   if (matchtrk_d0->at(it) < 999) {
	     h_resVsEta_d0_L[im]->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
	     h_absResVsEta_d0_L[im]->Fill( fabs( matchtrk_d0->at(it) - tp_d0->at(it) ) );
	   }
	 }
	 else {
	   h_resVsEta_ptRel_H[im]->Fill((matchtrk_pt->at(it) - tp_pt->at(it))/tp_pt->at(it));
	   h_resVsEta_eta_H[im]->Fill(matchtrk_eta->at(it) - tp_eta->at(it));
	   h_resVsEta_z0_H[im]->Fill(matchtrk_z0->at(it) - tp_z0->at(it));
	   h_resVsEta_phi_H[im]->Fill(matchtrk_phi->at(it) - tp_phi->at(it));

	   h_absResVsEta_ptRel_H[im]->Fill( fabs( (matchtrk_pt->at(it) - tp_pt->at(it)) )/tp_pt->at(it));
	   h_absResVsEta_eta_H[im]  ->Fill( fabs( matchtrk_eta->at(it) - tp_eta->at(it) ) );
	   h_absResVsEta_phi_H[im]  ->Fill( fabs( matchtrk_phi->at(it) - tp_phi->at(it) ) );
	   h_absResVsEta_z0_H[im]   ->Fill( fabs( matchtrk_z0->at(it)  - tp_z0->at(it) ) );

	   if (matchtrk_d0->at(it) < 999) {
	     h_resVsEta_d0_H[im]->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
	     h_absResVsEta_d0_H[im]->Fill( fabs( matchtrk_d0->at(it) - tp_d0->at(it) ) );
	   }
	 }
	 
	 if (matchtrk_d0->at(it) < 999) {
	   h_resVsEta_d0[im]->Fill(matchtrk_d0->at(it) - tp_d0->at(it));
	   h_absResVsEta_d0[im]->Fill( fabs( matchtrk_d0->at(it) - tp_d0->at(it) ) );
	 }
	 
       }
      }

      // ----------------------------------------------------------------------------------------------------------------
      // fill resolution vs. phi histograms    
      for (int im=0; im<nPHIRANGE; im++) {
	if ( (tp_phi->at(it) > (float)im*0.2-3.2) && (tp_phi->at(it) < (float)im*0.2-3.0) ) {
	  h_absResVsPhi_pt[im]   ->Fill( fabs(matchtrk_pt->at(it)  - tp_pt->at(it)) );
	  h_absResVsPhi_ptRel[im]->Fill( fabs((matchtrk_pt->at(it) - tp_pt->at(it)) )/tp_pt->at(it));
	}
      }

      
    } // end of matched track loop
    
  } // end of event loop
  // ----------------------------------------------------------------------------------------------------------------
  

  // ----------------------------------------------------------------------------------------------------------------
  // 2D plots  
  // ----------------------------------------------------------------------------------------------------------------

  TH1F* h2_resVsPt_pt   = new TH1F("resVsPt2_pt",   ";Tracking particle p_{T} [GeV]; p_{T} resolution", 20,0,100);
  TH1F* h2_resVsPt_pt_C = new TH1F("resVsPt2_pt_C", ";Tracking particle p_{T} [GeV]; p_{T} resolution", 20,0,100);
  TH1F* h2_resVsPt_pt_I = new TH1F("resVsPt2_pt_I", ";Tracking particle p_{T} [GeV]; p_{T} resolution", 20,0,100);
  TH1F* h2_resVsPt_pt_F = new TH1F("resVsPt2_pt_F", ";Tracking particle p_{T} [GeV]; p_{T} resolution", 20,0,100);

  TH1F* h2_resVsPt_ptRel = new TH1F("resVsPt2_ptRel",     ";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 20,0,100);
  TH1F* h2_resVsPt_ptRel_C = new TH1F("resVsPt2_ptRel_C", ";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 20,0,100);
  TH1F* h2_resVsPt_ptRel_I = new TH1F("resVsPt2_ptRel_I", ";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 20,0,100);
  TH1F* h2_resVsPt_ptRel_F = new TH1F("resVsPt2_ptRel_F", ";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 20,0,100);

  TH1F* h2_mresVsPt_pt   = new TH1F("mresVsPt2_pt",   ";Tracking particle p_{T} [GeV]; Mean(p_{T} residual) [GeV]", 20,0,100);
  TH1F* h2_mresVsPt_pt_C = new TH1F("mresVsPt2_pt_C", ";Tracking particle p_{T} [GeV]; Mean(p_{T} residual) [GeV]", 20,0,100);
  TH1F* h2_mresVsPt_pt_I = new TH1F("mresVsPt2_pt_I", ";Tracking particle p_{T} [GeV]; Mean(p_{T} residual) [GeV]", 20,0,100);
  TH1F* h2_mresVsPt_pt_F = new TH1F("mresVsPt2_pt_F", ";Tracking particle p_{T} [GeV]; Mean(p_{T} residual) [GeV]", 20,0,100);

  TH1F* h2_resVsPt_z0   = new TH1F("resVsPt2_z0",   ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]", 20,0,100);
  TH1F* h2_resVsPt_z0_C = new TH1F("resVsPt2_z0_C", ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]", 20,0,100);
  TH1F* h2_resVsPt_z0_I = new TH1F("resVsPt2_z0_I", ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]", 20,0,100);
  TH1F* h2_resVsPt_z0_F = new TH1F("resVsPt2_z0_F", ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]", 20,0,100);

  TH1F* h2_resVsPt_phi   = new TH1F("resVsPt2_phi",   ";Tracking particle p_{T} [GeV]; #phi resolution [rad]", 20,0,100);
  TH1F* h2_resVsPt_phi_C = new TH1F("resVsPt2_phi_C", ";Tracking particle p_{T} [GeV]; #phi resolution [rad]", 20,0,100);
  TH1F* h2_resVsPt_phi_I = new TH1F("resVsPt2_phi_I", ";Tracking particle p_{T} [GeV]; #phi resolution [rad]", 20,0,100);
  TH1F* h2_resVsPt_phi_F = new TH1F("resVsPt2_phi_F", ";Tracking particle p_{T} [GeV]; #phi resolution [rad]", 20,0,100);

  TH1F* h2_resVsPt_eta   = new TH1F("resVsPt2_eta",  ";Tracking particle p_{T} [GeV]; #eta resolution", 20,0,100);

  TH1F* h2_resVsPt_d0 = new TH1F("resVsPt2_d0",   ";Tracking particle p_{T} [GeV]; d_{0} resolution [cm]", 20,0,100);

  TH1F* h2_resVsPt_pt_68    = new TH1F("resVsPt_pt_68",   ";Tracking particle p_{T} [GeV]; p_{T} resolution",         20,0,100);
  TH1F* h2_resVsPt_ptRel_68 = new TH1F("resVsPt_ptRel_68",";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 20,0,100);
  TH1F* h2_resVsPt_z0_68    = new TH1F("resVsPt_z0_68",   ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]",    20,0,100);
  TH1F* h2_resVsPt_phi_68   = new TH1F("resVsPt_phi_68",  ";Tracking particle p_{T} [GeV]; #phi resolution [rad]",    20,0,100);
  TH1F* h2_resVsPt_eta_68   = new TH1F("resVsPt_eta_68",  ";Tracking particle p_{T} [GeV]; #eta resolution [rad]",    20,0,100);
  TH1F* h2_resVsPt_d0_68    = new TH1F("resVsPt_d0_68",   ";Tracking particle p_{T} [GeV]; d_{0} resolution [rad]",   20,0,100);

  TH1F* h2_resVsPt_pt_90    = new TH1F("resVsPt_pt_90",   ";Tracking particle p_{T} [GeV]; p_{T} resolution",         20,0,100);
  TH1F* h2_resVsPt_ptRel_90 = new TH1F("resVsPt_ptRel_90",";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 20,0,100);
  TH1F* h2_resVsPt_z0_90    = new TH1F("resVsPt_z0_90",   ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]",    20,0,100);
  TH1F* h2_resVsPt_phi_90   = new TH1F("resVsPt_phi_90",  ";Tracking particle p_{T} [GeV]; #phi resolution [rad]",    20,0,100);
  TH1F* h2_resVsPt_eta_90   = new TH1F("resVsPt_eta_90",  ";Tracking particle p_{T} [GeV]; #eta resolution [rad]",    20,0,100);
  TH1F* h2_resVsPt_d0_90    = new TH1F("resVsPt_d0_90",   ";Tracking particle p_{T} [GeV]; d_{0} resolution [rad]",   20,0,100);

  TH1F* h2_resVsPt_pt_99    = new TH1F("resVsPt_pt_99",   ";Tracking particle p_{T} [GeV]; p_{T} resolution",         20,0,100);
  TH1F* h2_resVsPt_ptRel_99 = new TH1F("resVsPt_ptRel_99",";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 20,0,100);
  TH1F* h2_resVsPt_z0_99    = new TH1F("resVsPt_z0_99",   ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]",    20,0,100);
  TH1F* h2_resVsPt_phi_99   = new TH1F("resVsPt_phi_99",  ";Tracking particle p_{T} [GeV]; #phi resolution [rad]",    20,0,100);
  TH1F* h2_resVsPt_eta_99   = new TH1F("resVsPt_eta_99",  ";Tracking particle p_{T} [GeV]; #eta resolution [rad]",    20,0,100);
  TH1F* h2_resVsPt_d0_99    = new TH1F("resVsPt_d0_99",   ";Tracking particle p_{T} [GeV]; d_{0} resolution [rad]",   20,0,100);

  TH1F* h2_resVsPt_pt_L_68    = new TH1F("resVsPt_pt_L_68",   ";Tracking particle p_{T} [GeV]; p_{T} resolution",         10,3,8);
  TH1F* h2_resVsPt_ptRel_L_68 = new TH1F("resVsPt_ptRel_L_68",";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 10,3,8);
  TH1F* h2_resVsPt_z0_L_68    = new TH1F("resVsPt_z0_L_68",   ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]",    10,3,8);
  TH1F* h2_resVsPt_phi_L_68   = new TH1F("resVsPt_phi_L_68",  ";Tracking particle p_{T} [GeV]; #phi resolution [rad]",    10,3,8);
  TH1F* h2_resVsPt_eta_L_68   = new TH1F("resVsPt_eta_L_68",  ";Tracking particle p_{T} [GeV]; #eta resolution [rad]",    10,3,8);
  TH1F* h2_resVsPt_d0_L_68    = new TH1F("resVsPt_d0_L_68",   ";Tracking particle p_{T} [GeV]; d_{0} resolution [rad]",   10,3,8);

  TH1F* h2_resVsPt_pt_L_90    = new TH1F("resVsPt_pt_L_90",   ";Tracking particle p_{T} [GeV]; p_{T} resolution",         10,3,8);
  TH1F* h2_resVsPt_ptRel_L_90 = new TH1F("resVsPt_ptRel_L_90",";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 10,3,8);
  TH1F* h2_resVsPt_z0_L_90    = new TH1F("resVsPt_z0_L_90",   ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]",    10,3,8);
  TH1F* h2_resVsPt_phi_L_90   = new TH1F("resVsPt_phi_L_90",  ";Tracking particle p_{T} [GeV]; #phi resolution [rad]",    10,3,8);
  TH1F* h2_resVsPt_eta_L_90   = new TH1F("resVsPt_eta_L_90",  ";Tracking particle p_{T} [GeV]; #eta resolution [rad]",    10,3,8);
  TH1F* h2_resVsPt_d0_L_90    = new TH1F("resVsPt_d0_L_90",   ";Tracking particle p_{T} [GeV]; d_{0} resolution [rad]",   10,3,8);

  TH1F* h2_resVsPt_pt_L_99    = new TH1F("resVsPt_pt_L_99",   ";Tracking particle p_{T} [GeV]; p_{T} resolution",         10,3,8);
  TH1F* h2_resVsPt_ptRel_L_99 = new TH1F("resVsPt_ptRel_L_99",";Tracking particle p_{T} [GeV]; p_{T} resolution / p_{T}", 10,3,8);
  TH1F* h2_resVsPt_z0_L_99    = new TH1F("resVsPt_z0_L_99",   ";Tracking particle p_{T} [GeV]; z_{0} resolution [cm]",    10,3,8);
  TH1F* h2_resVsPt_phi_L_99   = new TH1F("resVsPt_phi_L_99",  ";Tracking particle p_{T} [GeV]; #phi resolution [rad]",    10,3,8);
  TH1F* h2_resVsPt_eta_L_99   = new TH1F("resVsPt_eta_L_99",  ";Tracking particle p_{T} [GeV]; #eta resolution [rad]",    10,3,8);
  TH1F* h2_resVsPt_d0_L_99    = new TH1F("resVsPt_d0_L_99",   ";Tracking particle p_{T} [GeV]; d_{0} resolution [rad]",   10,3,8);

  for (int i=0; i<nRANGE; i++) {
    // set bin content and error
    h2_resVsPt_pt  ->SetBinContent(i+1, h_resVsPt_pt[i]  ->GetRMS());
    h2_resVsPt_pt  ->SetBinError(  i+1, h_resVsPt_pt[i]  ->GetRMSError());
    h2_resVsPt_pt_C->SetBinContent(i+1, h_resVsPt_pt_C[i]->GetRMS());
    h2_resVsPt_pt_C->SetBinError(  i+1, h_resVsPt_pt_C[i]->GetRMSError());
    h2_resVsPt_pt_I->SetBinContent(i+1, h_resVsPt_pt_I[i]->GetRMS());
    h2_resVsPt_pt_I->SetBinError(  i+1, h_resVsPt_pt_I[i]->GetRMSError());
    h2_resVsPt_pt_F->SetBinContent(i+1, h_resVsPt_pt_F[i]->GetRMS());
    h2_resVsPt_pt_F->SetBinError(  i+1, h_resVsPt_pt_F[i]->GetRMSError());

    h2_resVsPt_ptRel  ->SetBinContent(i+1, h_resVsPt_ptRel[i]  ->GetRMS());
    h2_resVsPt_ptRel  ->SetBinError(  i+1, h_resVsPt_ptRel[i]  ->GetRMSError());
    h2_resVsPt_ptRel_C->SetBinContent(i+1, h_resVsPt_ptRel_C[i]->GetRMS());
    h2_resVsPt_ptRel_C->SetBinError(  i+1, h_resVsPt_ptRel_C[i]->GetRMSError());
    h2_resVsPt_ptRel_I->SetBinContent(i+1, h_resVsPt_ptRel_I[i]->GetRMS());
    h2_resVsPt_ptRel_I->SetBinError(  i+1, h_resVsPt_ptRel_I[i]->GetRMSError());
    h2_resVsPt_ptRel_F->SetBinContent(i+1, h_resVsPt_ptRel_F[i]->GetRMS());
    h2_resVsPt_ptRel_F->SetBinError(  i+1, h_resVsPt_ptRel_F[i]->GetRMSError());

    h2_mresVsPt_pt  ->SetBinContent(i+1, h_resVsPt_pt[i]  ->GetMean());
    h2_mresVsPt_pt  ->SetBinError(  i+1, h_resVsPt_pt[i]  ->GetMeanError());
    h2_mresVsPt_pt_C->SetBinContent(i+1, h_resVsPt_pt_C[i]->GetMean());
    h2_mresVsPt_pt_C->SetBinError(  i+1, h_resVsPt_pt_C[i]->GetMeanError());
    h2_mresVsPt_pt_I->SetBinContent(i+1, h_resVsPt_pt_I[i]->GetMean());
    h2_mresVsPt_pt_I->SetBinError(  i+1, h_resVsPt_pt_I[i]->GetMeanError());
    h2_mresVsPt_pt_F->SetBinContent(i+1, h_resVsPt_pt_F[i]->GetMean());
    h2_mresVsPt_pt_F->SetBinError(  i+1, h_resVsPt_pt_F[i]->GetMeanError());

    h2_resVsPt_z0  ->SetBinContent(i+1, h_resVsPt_z0[i]  ->GetRMS());
    h2_resVsPt_z0  ->SetBinError(  i+1, h_resVsPt_z0[i]  ->GetRMSError());
    h2_resVsPt_z0_C->SetBinContent(i+1, h_resVsPt_z0_C[i]->GetRMS());
    h2_resVsPt_z0_C->SetBinError(  i+1, h_resVsPt_z0_C[i]->GetRMSError());
    h2_resVsPt_z0_I->SetBinContent(i+1, h_resVsPt_z0_I[i]->GetRMS());
    h2_resVsPt_z0_I->SetBinError(  i+1, h_resVsPt_z0_I[i]->GetRMSError());
    h2_resVsPt_z0_F->SetBinContent(i+1, h_resVsPt_z0_F[i]->GetRMS());
    h2_resVsPt_z0_F->SetBinError(  i+1, h_resVsPt_z0_F[i]->GetRMSError());

    h2_resVsPt_phi  ->SetBinContent(i+1, h_resVsPt_phi[i]  ->GetRMS());
    h2_resVsPt_phi  ->SetBinError(  i+1, h_resVsPt_phi[i]  ->GetRMSError());
    h2_resVsPt_phi_C->SetBinContent(i+1, h_resVsPt_phi_C[i]->GetRMS());
    h2_resVsPt_phi_C->SetBinError(  i+1, h_resVsPt_phi_C[i]->GetRMSError());
    h2_resVsPt_phi_I->SetBinContent(i+1, h_resVsPt_phi_I[i]->GetRMS());
    h2_resVsPt_phi_I->SetBinError(  i+1, h_resVsPt_phi_I[i]->GetRMSError());
    h2_resVsPt_phi_F->SetBinContent(i+1, h_resVsPt_phi_F[i]->GetRMS());
    h2_resVsPt_phi_F->SetBinError(  i+1, h_resVsPt_phi_F[i]->GetRMSError());

    h2_resVsPt_eta->SetBinContent(i+1, h_resVsPt_eta[i]->GetRMS());
    h2_resVsPt_eta->SetBinError(  i+1, h_resVsPt_eta[i]->GetRMSError());

    h2_resVsPt_d0->SetBinContent(i+1, h_resVsPt_d0[i]->GetRMS());
    h2_resVsPt_d0->SetBinError(  i+1, h_resVsPt_d0[i]->GetRMSError());

    h2_resVsPt_pt_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_pt[i], 0.68 ));
    h2_resVsPt_pt_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_pt[i], 0.90 ));
    h2_resVsPt_pt_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_pt[i], 0.99 ));

    h2_resVsPt_ptRel_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_ptRel[i], 0.68 ));
    h2_resVsPt_ptRel_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_ptRel[i], 0.90 ));
    h2_resVsPt_ptRel_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_ptRel[i], 0.99 ));

    h2_resVsPt_eta_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_eta[i], 0.68 ));
    h2_resVsPt_eta_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_eta[i], 0.90 ));
    h2_resVsPt_eta_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_eta[i], 0.99 ));

    h2_resVsPt_z0_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_z0[i], 0.68 ));
    h2_resVsPt_z0_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_z0[i], 0.90 ));
    h2_resVsPt_z0_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_z0[i], 0.99 ));

    h2_resVsPt_phi_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_phi[i], 0.68 ));
    h2_resVsPt_phi_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_phi[i], 0.90 ));
    h2_resVsPt_phi_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_phi[i], 0.99 ));

    h2_resVsPt_d0_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_d0[i], 0.68 ));
    h2_resVsPt_d0_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_d0[i], 0.90 ));
    h2_resVsPt_d0_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_d0[i], 0.99 ));
  }
  
  for (int i=0; i<nRANGE_L; i++) {
    h2_resVsPt_pt_L_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_pt_L[i], 0.68 ));
    h2_resVsPt_pt_L_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_pt_L[i], 0.90 ));
    h2_resVsPt_pt_L_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_pt_L[i], 0.99 ));

    h2_resVsPt_ptRel_L_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_ptRel_L[i], 0.68 ));
    h2_resVsPt_ptRel_L_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_ptRel_L[i], 0.90 ));
    h2_resVsPt_ptRel_L_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_ptRel_L[i], 0.99 ));

    h2_resVsPt_eta_L_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_eta_L[i], 0.68 ));
    h2_resVsPt_eta_L_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_eta_L[i], 0.90 ));
    h2_resVsPt_eta_L_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_eta_L[i], 0.99 ));

    h2_resVsPt_z0_L_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_z0_L[i], 0.68 ));
    h2_resVsPt_z0_L_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_z0_L[i], 0.90 ));
    h2_resVsPt_z0_L_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_z0_L[i], 0.99 ));

    h2_resVsPt_phi_L_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_phi_L[i], 0.68 ));
    h2_resVsPt_phi_L_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_phi_L[i], 0.90 ));
    h2_resVsPt_phi_L_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_phi_L[i], 0.99 ));

    h2_resVsPt_d0_L_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_d0_L[i], 0.68 ));
    h2_resVsPt_d0_L_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_d0_L[i], 0.90 ));
    h2_resVsPt_d0_L_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPt_d0_L[i], 0.99 ));
  }

  // resolution vs. eta histograms
  TH1F* h2_resVsEta_eta   = new TH1F("resVsEta2_eta",   ";Tracking particle |#eta|; #eta resolution", 24,0,2.4);
  TH1F* h2_resVsEta_eta_L = new TH1F("resVsEta2_eta_L", ";Tracking particle |#eta|; #eta resolution", 24,0,2.4);
  TH1F* h2_resVsEta_eta_H = new TH1F("resVsEta2_eta_H", ";Tracking particle |#eta|; #eta resolution", 24,0,2.4);

  TH1F* h2_mresVsEta_eta   = new TH1F("mresVsEta2_eta",   ";Tracking particle |#eta|; Mean(#eta residual)", 24,0,2.4);
  TH1F* h2_mresVsEta_eta_L = new TH1F("mresVsEta2_eta_L", ";Tracking particle |#eta|; Mean(#eta residual)", 24,0,2.4);
  TH1F* h2_mresVsEta_eta_H = new TH1F("mresVsEta2_eta_H", ";Tracking particle |#eta|; Mean(#eta residual)", 24,0,2.4);

  TH1F* h2_resVsEta_z0   = new TH1F("resVsEta_z0",   ";Tracking particle |#eta|; z_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_z0_L = new TH1F("resVsEta_z0_L", ";Tracking particle |#eta|; z_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_z0_H = new TH1F("resVsEta_z0_H", ";Tracking particle |#eta|; z_{0} resolution [cm]", 24,0,2.4);

  TH1F* h2_resVsEta_phi   = new TH1F("resVsEta_phi",   ";Tracking particle |#eta|; #phi resolution [rad]", 24,0,2.4);
  TH1F* h2_resVsEta_phi_L = new TH1F("resVsEta_phi_L", ";Tracking particle |#eta|; #phi resolution [rad]", 24,0,2.4);
  TH1F* h2_resVsEta_phi_H = new TH1F("resVsEta_phi_H", ";Tracking particle |#eta|; #phi resolution [rad]", 24,0,2.4);

  TH1F* h2_resVsEta_ptRel   = new TH1F("resVsEta_ptRel", ";Tracking particle |#eta|; p_{T} resolution / p_{T}", 24,0,2.4);
  TH1F* h2_resVsEta_ptRel_L = new TH1F("resVsEta_ptRel_L", ";Tracking particle |#eta|; p_{T} resolution / p_{T}", 24,0,2.4);
  TH1F* h2_resVsEta_ptRel_H = new TH1F("resVsEta_ptRel_H", ";Tracking particle |#eta|; p_{T} resolution / p_{T}", 24,0,2.4);

  TH1F* h2_resVsEta_d0  = new TH1F("resVsEta2_d0",  ";Tracking particle |#eta|; d_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_d0_L  = new TH1F("resVsEta2_d0_L",  ";Tracking particle |#eta|; d_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_d0_H  = new TH1F("resVsEta2_d0_H",  ";Tracking particle |#eta|; d_{0} resolution [cm]", 24,0,2.4);


  TH1F* h2_resVsEta_eta_68  = new TH1F("resVsEta_eta_68",   ";Tracking particle |#eta|; #eta resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_eta_90   = new TH1F("resVsEta_eta_90",   ";Tracking particle |#eta|; #eta resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_eta_99   = new TH1F("resVsEta_eta_99",   ";Tracking particle |#eta|; #eta resolution [cm]", 24,0,2.4);

  TH1F* h2_resVsEta_z0_68   = new TH1F("resVsEta_z0_68",   ";Tracking particle |#eta|; z_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_z0_90   = new TH1F("resVsEta_z0_90",   ";Tracking particle |#eta|; z_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_z0_99   = new TH1F("resVsEta_z0_99",   ";Tracking particle |#eta|; z_{0} resolution [cm]", 24,0,2.4);

  TH1F* h2_resVsEta_phi_68   = new TH1F("resVsEta_phi_68",   ";Tracking particle |#eta|; #phi resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_phi_90   = new TH1F("resVsEta_phi_90",   ";Tracking particle |#eta|; #phi resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_phi_99   = new TH1F("resVsEta_phi_99",   ";Tracking particle |#eta|; #phi resolution [cm]", 24,0,2.4);

  TH1F* h2_resVsEta_ptRel_68   = new TH1F("resVsEta_ptRel_68",   ";Tracking particle |#eta|; p_{T} resolution / p_{T} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_ptRel_90   = new TH1F("resVsEta_ptRel_90",   ";Tracking particle |#eta|; p_{T} resolution / p_{T} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_ptRel_99   = new TH1F("resVsEta_ptRel_99",   ";Tracking particle |#eta|; p_{T} resolution / p_{T} resolution [cm]", 24,0,2.4);
  
  TH1F* h2_resVsEta_d0_68   = new TH1F("resVsEta_d0_68",   ";Tracking particle |#eta|; d_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_d0_90   = new TH1F("resVsEta_d0_90",   ";Tracking particle |#eta|; d_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_d0_99   = new TH1F("resVsEta_d0_99",   ";Tracking particle |#eta|; d_{0} resolution [cm]", 24,0,2.4);

  TH1F* h2_resVsEta_eta_L_68   = new TH1F("resVsEta_eta_L_68",  ";Tracking particle |#eta|; #eta resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_z0_L_68    = new TH1F("resVsEta_z0_L_68",   ";Tracking particle |#eta|; z_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_phi_L_68   = new TH1F("resVsEta_phi_L_68",  ";Tracking particle |#eta|; #phi resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_ptRel_L_68 = new TH1F("resVsEta_ptRel_L_68",";Tracking particle |#eta|; p_{T} resolution / p_{T} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_d0_L_68    = new TH1F("resVsEta_d0_L_68",   ";Tracking particle |#eta|; d_{0} resolution [cm]", 24,0,2.4);

  TH1F* h2_resVsEta_eta_L_90   = new TH1F("resVsEta_eta_L_90",  ";Tracking particle |#eta|; #eta resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_z0_L_90    = new TH1F("resVsEta_z0_L_90",   ";Tracking particle |#eta|; z_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_phi_L_90   = new TH1F("resVsEta_phi_L_90",  ";Tracking particle |#eta|; #phi resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_ptRel_L_90 = new TH1F("resVsEta_ptRel_L_90",";Tracking particle |#eta|; p_{T} resolution / p_{T} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_d0_L_90    = new TH1F("resVsEta_d0_L_90",   ";Tracking particle |#eta|; d_{0} resolution [cm]", 24,0,2.4);

  TH1F* h2_resVsEta_eta_L_99   = new TH1F("resVsEta_eta_L_99",  ";Tracking particle |#eta|; #eta resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_z0_L_99    = new TH1F("resVsEta_z0_L_99",   ";Tracking particle |#eta|; z_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_phi_L_99   = new TH1F("resVsEta_phi_L_99",  ";Tracking particle |#eta|; #phi resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_ptRel_L_99 = new TH1F("resVsEta_ptRel_L_99",";Tracking particle |#eta|; p_{T} resolution / p_{T} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_d0_L_99    = new TH1F("resVsEta_d0_L_99",   ";Tracking particle |#eta|; d_{0} resolution [cm]", 24,0,2.4);

  TH1F* h2_resVsEta_eta_H_68   = new TH1F("resVsEta_eta_H_68",  ";Tracking particle |#eta|; #eta resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_z0_H_68    = new TH1F("resVsEta_z0_H_68",   ";Tracking particle |#eta|; z_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_phi_H_68   = new TH1F("resVsEta_phi_H_68",  ";Tracking particle |#eta|; #phi resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_ptRel_H_68 = new TH1F("resVsEta_ptRel_H_68",";Tracking particle |#eta|; p_{T} resolution / p_{T} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_d0_H_68    = new TH1F("resVsEta_d0_H_68",   ";Tracking particle |#eta|; d_{0} resolution [cm]", 24,0,2.4);

  TH1F* h2_resVsEta_eta_H_90   = new TH1F("resVsEta_eta_H_90",  ";Tracking particle |#eta|; #eta resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_z0_H_90    = new TH1F("resVsEta_z0_H_90",   ";Tracking particle |#eta|; z_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_phi_H_90   = new TH1F("resVsEta_phi_H_90",  ";Tracking particle |#eta|; #phi resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_ptRel_H_90 = new TH1F("resVsEta_ptRel_H_90",";Tracking particle |#eta|; p_{T} resolution / p_{T} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_d0_H_90    = new TH1F("resVsEta_d0_H_90",   ";Tracking particle |#eta|; d_{0} resolution [cm]", 24,0,2.4);

  TH1F* h2_resVsEta_eta_H_99   = new TH1F("resVsEta_eta_H_99",  ";Tracking particle |#eta|; #eta resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_z0_H_99    = new TH1F("resVsEta_z0_H_99",   ";Tracking particle |#eta|; z_{0} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_phi_H_99   = new TH1F("resVsEta_phi_H_99",  ";Tracking particle |#eta|; #phi resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_ptRel_H_99 = new TH1F("resVsEta_ptRel_H_99",";Tracking particle |#eta|; p_{T} resolution / p_{T} resolution [cm]", 24,0,2.4);
  TH1F* h2_resVsEta_d0_H_99    = new TH1F("resVsEta_d0_H_99",   ";Tracking particle |#eta|; d_{0} resolution [cm]", 24,0,2.4);



  // resolution vs. eta histograms (gaussian fit)
  TH1F* h3_resVsEta_eta_L = new TH1F("resVsEta_eta_L_gaus", ";|#eta|; #sigma(#eta)", 24,0,2.4);
  TH1F* h3_resVsEta_eta_H = new TH1F("resVsEta_eta_H_gaus", ";|#eta|; #sigma(#eta)", 24,0,2.4);

  TH1F* h3_resVsEta_z0_L = new TH1F("resVsEta_z0_L_gaus", ";|#eta|; #sigma(z_{0}) [cm]", 24,0,2.4);
  TH1F* h3_resVsEta_z0_H = new TH1F("resVsEta_z0_H_gaus", ";|#eta|; #sigma(z_{0}) [cm]", 24,0,2.4);

  TH1F* h3_resVsEta_phi_L = new TH1F("resVsEta_phi_L_gaus", ";|#eta|; #sigma(#phi) [rad]", 24,0,2.4);
  TH1F* h3_resVsEta_phi_H = new TH1F("resVsEta_phi_H_gaus", ";|#eta|; #sigma(#phi) [rad]", 24,0,2.4);

  TH1F* h3_resVsEta_ptRel_L = new TH1F("resVsEta_ptRel_L_gaus", ";|#eta|; #sigma(p_{T}) / p_{T}", 24,0,2.4);
  TH1F* h3_resVsEta_ptRel_H = new TH1F("resVsEta_ptRel_H_gaus", ";|#eta|; #sigma(p_{T}) / p_{T}", 24,0,2.4);

  TString fitdir = "FitResults/";


  for (int i=0; i<nETARANGE; i++) {
    // set bin content and error
    h2_resVsEta_eta  ->SetBinContent(i+1, h_resVsEta_eta[i]  ->GetRMS());
    h2_resVsEta_eta  ->SetBinError(  i+1, h_resVsEta_eta[i]  ->GetRMSError());
    h2_resVsEta_eta_L->SetBinContent(i+1, h_resVsEta_eta_L[i]->GetRMS());
    h2_resVsEta_eta_L->SetBinError(  i+1, h_resVsEta_eta_L[i]->GetRMSError());
    h2_resVsEta_eta_H->SetBinContent(i+1, h_resVsEta_eta_H[i]->GetRMS());
    h2_resVsEta_eta_H->SetBinError(  i+1, h_resVsEta_eta_H[i]->GetRMSError());

    h2_mresVsEta_eta  ->SetBinContent(i+1, h_resVsEta_eta[i]  ->GetMean());
    h2_mresVsEta_eta  ->SetBinError(  i+1, h_resVsEta_eta[i]  ->GetMeanError());
    h2_mresVsEta_eta_L->SetBinContent(i+1, h_resVsEta_eta_L[i]->GetMean());
    h2_mresVsEta_eta_L->SetBinError(  i+1, h_resVsEta_eta_L[i]->GetMeanError());
    h2_mresVsEta_eta_H->SetBinContent(i+1, h_resVsEta_eta_H[i]->GetMean());
    h2_mresVsEta_eta_H->SetBinError(  i+1, h_resVsEta_eta_H[i]->GetMeanError());

    h2_resVsEta_z0 ->SetBinContent(i+1, h_resVsEta_z0[i] ->GetRMS());
    h2_resVsEta_z0 ->SetBinError(  i+1, h_resVsEta_z0[i] ->GetRMSError());
    h2_resVsEta_z0_L ->SetBinContent(i+1, h_resVsEta_z0_L[i] ->GetRMS());
    h2_resVsEta_z0_L ->SetBinError(  i+1, h_resVsEta_z0_L[i] ->GetRMSError());
    h2_resVsEta_z0_H ->SetBinContent(i+1, h_resVsEta_z0_H[i] ->GetRMS());
    h2_resVsEta_z0_H ->SetBinError(  i+1, h_resVsEta_z0_H[i] ->GetRMSError());

    h2_resVsEta_phi->SetBinContent(i+1, h_resVsEta_phi[i]->GetRMS());
    h2_resVsEta_phi->SetBinError(  i+1, h_resVsEta_phi[i]->GetRMSError());
    h2_resVsEta_phi_L->SetBinContent(i+1, h_resVsEta_phi_L[i]->GetRMS());
    h2_resVsEta_phi_L->SetBinError(  i+1, h_resVsEta_phi_L[i]->GetRMSError());
    h2_resVsEta_phi_H->SetBinContent(i+1, h_resVsEta_phi_H[i]->GetRMS());
    h2_resVsEta_phi_H->SetBinError(  i+1, h_resVsEta_phi_H[i]->GetRMSError());

    h2_resVsEta_ptRel->SetBinContent(i+1, h_resVsEta_ptRel[i]->GetRMS());
    h2_resVsEta_ptRel->SetBinError(  i+1, h_resVsEta_ptRel[i]->GetRMSError());
    h2_resVsEta_ptRel_L->SetBinContent(i+1, h_resVsEta_ptRel_L[i]->GetRMS());
    h2_resVsEta_ptRel_L->SetBinError(  i+1, h_resVsEta_ptRel_L[i]->GetRMSError());
    h2_resVsEta_ptRel_H->SetBinContent(i+1, h_resVsEta_ptRel_H[i]->GetRMS());
    h2_resVsEta_ptRel_H->SetBinError(  i+1, h_resVsEta_ptRel_H[i]->GetRMSError());

    h2_resVsEta_d0->SetBinContent(i+1, h_resVsEta_d0[i] ->GetRMS());
    h2_resVsEta_d0->SetBinError(  i+1, h_resVsEta_d0[i] ->GetRMSError());
    h2_resVsEta_d0_L->SetBinContent(i+1, h_resVsEta_d0_L[i] ->GetRMS());
    h2_resVsEta_d0_L->SetBinError(  i+1, h_resVsEta_d0_L[i] ->GetRMSError());
    h2_resVsEta_d0_H->SetBinContent(i+1, h_resVsEta_d0_H[i] ->GetRMS());
    h2_resVsEta_d0_H->SetBinError(  i+1, h_resVsEta_d0_H[i] ->GetRMSError());

    h2_resVsEta_eta_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_eta[i], 0.68 ));
    h2_resVsEta_eta_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_eta[i], 0.90 ));
    h2_resVsEta_eta_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_eta[i], 0.99 ));

    h2_resVsEta_z0_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_z0[i], 0.68 ));
    h2_resVsEta_z0_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_z0[i], 0.90 ));
    h2_resVsEta_z0_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_z0[i], 0.99 ));

    h2_resVsEta_phi_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_phi[i], 0.68 ));
    h2_resVsEta_phi_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_phi[i], 0.90 ));
    h2_resVsEta_phi_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_phi[i], 0.99 ));

    h2_resVsEta_ptRel_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_ptRel[i], 0.68 ));
    h2_resVsEta_ptRel_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_ptRel[i], 0.90 ));
    h2_resVsEta_ptRel_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_ptRel[i], 0.99 ));

    h2_resVsEta_d0_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_d0[i], 0.68 ));
    h2_resVsEta_d0_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_d0[i], 0.90 ));
    h2_resVsEta_d0_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_d0[i], 0.99 ));


    h2_resVsEta_eta_L_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_eta_L[i], 0.68 ));
    h2_resVsEta_z0_L_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_z0_L[i], 0.68 ));
    h2_resVsEta_phi_L_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_phi_L[i], 0.68 ));
    h2_resVsEta_ptRel_L_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_ptRel_L[i], 0.68 ));
    h2_resVsEta_d0_L_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_d0_L[i], 0.68 ));

    h2_resVsEta_eta_L_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_eta_L[i], 0.90 ));
    h2_resVsEta_z0_L_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_z0_L[i], 0.90 ));
    h2_resVsEta_phi_L_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_phi_L[i], 0.90 ));
    h2_resVsEta_ptRel_L_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_ptRel_L[i], 0.90 ));
    h2_resVsEta_d0_L_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_d0_L[i], 0.90 ));

    h2_resVsEta_eta_L_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_eta_L[i], 0.99 ));
    h2_resVsEta_z0_L_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_z0_L[i], 0.99 ));
    h2_resVsEta_phi_L_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_phi_L[i], 0.99 ));
    h2_resVsEta_ptRel_L_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_ptRel_L[i], 0.99 ));
    h2_resVsEta_d0_L_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_d0_L[i], 0.99 ));

    h2_resVsEta_eta_H_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_eta_H[i], 0.68 ));
    h2_resVsEta_z0_H_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_z0_H[i], 0.68 ));
    h2_resVsEta_phi_H_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_phi_H[i], 0.68 ));
    h2_resVsEta_ptRel_H_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_ptRel_H[i], 0.68 ));
    h2_resVsEta_d0_H_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_d0_H[i], 0.68 ));

    h2_resVsEta_eta_H_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_eta_H[i], 0.90 ));
    h2_resVsEta_z0_H_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_z0_H[i], 0.90 ));
    h2_resVsEta_phi_H_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_phi_H[i], 0.90 ));
    h2_resVsEta_ptRel_H_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_ptRel_H[i], 0.90 ));
    h2_resVsEta_d0_H_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_d0_H[i], 0.90 ));

    h2_resVsEta_eta_H_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_eta_H[i], 0.99 ));
    h2_resVsEta_z0_H_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_z0_H[i], 0.99 ));
    h2_resVsEta_phi_H_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_phi_H[i], 0.99 ));
    h2_resVsEta_ptRel_H_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_ptRel_H[i], 0.99 ));
    h2_resVsEta_d0_H_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsEta_d0_H[i], 0.99 ));



    // ---------------------------------------------------------------------------------------------------
    // gaussian fit instead
    // ---------------------------------------------------------------------------------------------------
  
    if (doGausFit) {

      TCanvas cfit;
      char text[500];
      
      float sigma = 0;
      float esigma = 0;
      TF1* fit;
      
      float rms = 0;
      float erms = 0;
      
      fit = new TF1("fit", "gaus", -0.01,0.01);
      h_resVsEta_eta_L[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_eta_L[i]->GetRMS();
      erms = h_resVsEta_eta_L[i]->GetRMSError();
      h3_resVsEta_eta_L->SetBinContent(i+1, sigma);   
      h3_resVsEta_eta_L->SetBinError(i+1, esigma);   
      h_resVsEta_eta_L[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      sprintf(text,"p_{T} < 5 GeV");
      mySmallText(0.2,0.7,1,text);
      cfit.SaveAs(fitdir+"resVsEta_eta_L_"+etarange[i]+".png");
      delete fit;
      
      fit = new TF1("fit", "gaus", -0.01,0.01);
      h_resVsEta_eta_H[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_eta_H[i]->GetRMS();
      erms = h_resVsEta_eta_H[i]->GetRMSError();
      h3_resVsEta_eta_H->SetBinContent(i+1, sigma);   
      h3_resVsEta_eta_H->SetBinError(i+1, esigma);   
      h_resVsEta_eta_H[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      sprintf(text,"p_{T} > 15 GeV");
      mySmallText(0.2,0.7,1,text);
      cfit.SaveAs(fitdir+"resVsEta_eta_H_"+etarange[i]+".png");
      delete fit;
      
      fit = new TF1("fit", "gaus", -1,1);
      h_resVsEta_z0_L[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_z0_L[i]->GetRMS();
      erms = h_resVsEta_z0_L[i]->GetRMSError();
      h3_resVsEta_z0_L->SetBinContent(i+1, sigma);   
      h3_resVsEta_z0_L->SetBinError(i+1, esigma);   
      h_resVsEta_z0_L[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      sprintf(text,"p_{T} < 5 GeV");
      mySmallText(0.2,0.7,1,text);
      cfit.SaveAs(fitdir+"resVsEta_z0_L_"+etarange[i]+".png");
      delete fit;
      
      fit = new TF1("fit", "gaus", -1,1);
      h_resVsEta_z0_H[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_z0_H[i]->GetRMS();
      erms = h_resVsEta_z0_H[i]->GetRMSError();
      h3_resVsEta_z0_H->SetBinContent(i+1, sigma);   
      h3_resVsEta_z0_H->SetBinError(i+1, esigma);   
      h_resVsEta_z0_H[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      sprintf(text,"p_{T} > 15 GeV");
      mySmallText(0.2,0.7,1,text);
      cfit.SaveAs(fitdir+"resVsEta_z0_H_"+etarange[i]+".png");
      delete fit;
      
      fit = new TF1("fit", "gaus", -0.005,0.005);
      h_resVsEta_phi_L[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_phi_L[i]->GetRMS();
      erms = h_resVsEta_phi_L[i]->GetRMSError();
      h3_resVsEta_phi_L->SetBinContent(i+1, sigma);   
      h3_resVsEta_phi_L->SetBinError(i+1, esigma);   
      h_resVsEta_phi_L[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      sprintf(text,"p_{T} < 5 GeV");
      mySmallText(0.2,0.7,1,text);
      cfit.SaveAs(fitdir+"resVsEta_phi_L_"+etarange[i]+".png");
      delete fit;
      
      fit = new TF1("fit", "gaus", -0.005,0.005);
      h_resVsEta_phi_H[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_phi_H[i]->GetRMS();
      erms = h_resVsEta_phi_H[i]->GetRMSError();
      h3_resVsEta_phi_H->SetBinContent(i+1, sigma);   
      h3_resVsEta_phi_H->SetBinError(i+1, esigma);   
      h_resVsEta_phi_H[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      sprintf(text,"p_{T} > 15 GeV");
      mySmallText(0.2,0.7,1,text);
      cfit.SaveAs(fitdir+"resVsEta_phi_H_"+etarange[i]+".png");
      delete fit;
      
      
      fit = new TF1("fit", "gaus", -0.5,0.5);
      h_resVsEta_ptRel_L[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_ptRel_L[i]->GetRMS();
      erms = h_resVsEta_ptRel_L[i]->GetRMSError();
      h3_resVsEta_ptRel_L->SetBinContent(i+1, sigma);   
      h3_resVsEta_ptRel_L->SetBinError(i+1, esigma);   
      h_resVsEta_ptRel_L[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      sprintf(text,"p_{T} < 5 GeV");
      mySmallText(0.2,0.7,1,text);
      cfit.SaveAs(fitdir+"resVsEta_ptRel_L_"+etarange[i]+".png");
      delete fit;
      
      fit = new TF1("fit", "gaus", -0.5,0.5);
      h_resVsEta_ptRel_H[i]->Fit("fit","R");
      sigma  = fit->GetParameter(2);
      esigma = fit->GetParError(2);
      rms = h_resVsEta_ptRel_H[i]->GetRMS();
      erms = h_resVsEta_ptRel_H[i]->GetRMSError();
      h3_resVsEta_ptRel_H->SetBinContent(i+1, sigma);   
      h3_resVsEta_ptRel_H->SetBinError(i+1, esigma);   
      h_resVsEta_ptRel_H[i]->Draw();
      sprintf(text,"RMS: %.4f +/- %.4f",rms,erms);
      mySmallText(0.2,0.86,1,text);
      sprintf(text,"Fit: %.4f +/- %.4f",sigma,esigma);
      mySmallText(0.2,0.8,2,text);
      sprintf(text,"p_{T} > 15 GeV");
      mySmallText(0.2,0.7,1,text);
      cfit.SaveAs(fitdir+"resVsEta_ptRel_H_"+etarange[i]+".png");
      delete fit;

    }//end doGausFit
    
  }


  TH1F* h2_resVsPhi_pt_68    = new TH1F("resVsPhi2_pt_68",   ";Tracking particle #phi; p_{T} resolution",         32,-3.2,3.2);
  TH1F* h2_resVsPhi_pt_90    = new TH1F("resVsPhi2_pt_90",   ";Tracking particle #phi; p_{T} resolution",         32,-3.2,3.2);
  TH1F* h2_resVsPhi_pt_99    = new TH1F("resVsPhi2_pt_99",   ";Tracking particle #phi; p_{T} resolution",         32,-3.2,3.2);
  TH1F* h2_resVsPhi_ptRel_68 = new TH1F("resVsPhi2_ptRel_68",";Tracking particle #phi; p_{T} resolution / p_{T}", 32,-3.2,3.2);
  TH1F* h2_resVsPhi_ptRel_90 = new TH1F("resVsPhi2_ptRel_90",";Tracking particle #phi; p_{T} resolution / p_{T}", 32,-3.2,3.2);
  TH1F* h2_resVsPhi_ptRel_99 = new TH1F("resVsPhi2_ptRel_99",";Tracking particle #phi; p_{T} resolution / p_{T}", 32,-3.2,3.2);

  for (int i=0; i<nPHIRANGE; i++) {
    h2_resVsPhi_pt_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPhi_pt[i], 0.68 ));
    h2_resVsPhi_pt_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPhi_pt[i], 0.90 ));
    h2_resVsPhi_pt_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPhi_pt[i], 0.99 ));

    h2_resVsPhi_ptRel_68->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPhi_ptRel[i], 0.68 ));
    h2_resVsPhi_ptRel_90->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPhi_ptRel[i], 0.90 ));
    h2_resVsPhi_ptRel_99->SetBinContent(i+1, getIntervalContainingFractionOfEntries( h_absResVsPhi_ptRel[i], 0.99 ));
  }

  
  // set minimum to zero
  h2_resVsPt_pt  ->SetMinimum(0);
  h2_resVsPt_pt_C->SetMinimum(0);
  h2_resVsPt_pt_I->SetMinimum(0);
  h2_resVsPt_pt_F->SetMinimum(0);

  h2_resVsPt_ptRel  ->SetMinimum(0);
  h2_resVsPt_ptRel_C->SetMinimum(0);
  h2_resVsPt_ptRel_I->SetMinimum(0);
  h2_resVsPt_ptRel_F->SetMinimum(0);

  h2_resVsPt_z0  ->SetMinimum(0);
  h2_resVsPt_z0_C->SetMinimum(0);
  h2_resVsPt_z0_I->SetMinimum(0);
  h2_resVsPt_z0_F->SetMinimum(0);

  h2_resVsPt_phi  ->SetMinimum(0);
  h2_resVsPt_phi_C->SetMinimum(0);
  h2_resVsPt_phi_I->SetMinimum(0);
  h2_resVsPt_phi_F->SetMinimum(0);

  h2_resVsPt_eta->SetMinimum(0);

  h2_resVsEta_eta  ->SetMinimum(0);
  h2_resVsEta_eta_L->SetMinimum(0);
  h2_resVsEta_eta_H->SetMinimum(0);

  h2_resVsEta_z0 ->SetMinimum(0);
  h2_resVsEta_z0_L ->SetMinimum(0);
  h2_resVsEta_z0_H ->SetMinimum(0);

  h2_resVsEta_phi->SetMinimum(0);
  h2_resVsEta_phi_L->SetMinimum(0);
  h2_resVsEta_phi_H->SetMinimum(0);

  h2_resVsEta_ptRel->SetMinimum(0);
  h2_resVsEta_ptRel_L->SetMinimum(0);
  h2_resVsEta_ptRel_H->SetMinimum(0);

  h2_resVsPt_d0  ->SetMinimum(0);
  h2_resVsEta_d0 ->SetMinimum(0);
  h2_resVsEta_d0_L ->SetMinimum(0);
  h2_resVsEta_d0_H ->SetMinimum(0);


  // -------------------------------------------------------------------------------------------
  // output file for histograms
  // -------------------------------------------------------------------------------------------
 
  if (TP_select_pdgid != 0) {
    char pdgidtxt[500];
    sprintf(pdgidtxt,"_pdgid%i",TP_select_pdgid);
    type = type+pdgidtxt;
  }
  //else if (TP_select_injet == 1) type = type+"_injet";
  //else if (TP_select_injet == 2) type = type+"_injet_highpt";

  TFile* fout = new TFile("output_"+type+".root","recreate");
  
  
  // -------------------------------------------------------------------------------------------
  // draw and save plots
  // -------------------------------------------------------------------------------------------

  char ctxt[500];
  TCanvas c;

  TString DIR = "TrkPlots/";

  // plots overlaying 68, 90, 99% confidence levels]
  // if (doDetailedPlots) {
    // makeResidualIntervalPlot will save the individual plots to the root file
    makeResidualIntervalPlot( type, DIR, "resVsPt_ptRel", h2_resVsPt_ptRel_68, h2_resVsPt_ptRel_90, h2_resVsPt_ptRel_99, 0, 0.5 );
    makeResidualIntervalPlot( type, DIR, "resVsPt_pt", h2_resVsPt_pt_68, h2_resVsPt_pt_90, h2_resVsPt_pt_99, 0, 20 );
    makeResidualIntervalPlot( type, DIR, "resVsPt_z0", h2_resVsPt_z0_68, h2_resVsPt_z0_90, h2_resVsPt_z0_99, 0, 2.0 );
    makeResidualIntervalPlot( type, DIR, "resVsPt_phi", h2_resVsPt_phi_68, h2_resVsPt_phi_90, h2_resVsPt_phi_99, 0, 0.006 );
    makeResidualIntervalPlot( type, DIR, "resVsPt_eta", h2_resVsPt_eta_68, h2_resVsPt_eta_90, h2_resVsPt_eta_99, 0, 0.05 );
    //makeResidualIntervalPlot( type, DIR, "resVsPt_d0", h2_resVsPt_d0_68, h2_resVsPt_d0_90, h2_resVsPt_d0_99, 0, 0.02 );

  if (doDetailedPlots) {
    makeResidualIntervalPlot( type, DIR, "resVsPt_L_ptRel", h2_resVsPt_ptRel_L_68, h2_resVsPt_ptRel_L_90, h2_resVsPt_ptRel_L_99, 0, 0.5 );
    makeResidualIntervalPlot( type, DIR, "resVsPt_L_pt", h2_resVsPt_pt_L_68, h2_resVsPt_pt_L_90, h2_resVsPt_pt_L_99, 0, 20 );
    makeResidualIntervalPlot( type, DIR, "resVsPt_L_z0", h2_resVsPt_z0_L_68, h2_resVsPt_z0_L_90, h2_resVsPt_z0_L_99, 0, 2.0 );
    makeResidualIntervalPlot( type, DIR, "resVsPt_L_phi", h2_resVsPt_phi_L_68, h2_resVsPt_phi_L_90, h2_resVsPt_phi_L_99, 0, 0.1 );
    makeResidualIntervalPlot( type, DIR, "resVsPt_L_eta", h2_resVsPt_eta_L_68, h2_resVsPt_eta_L_90, h2_resVsPt_eta_L_99, 0, 0.05 );
    //makeResidualIntervalPlot( type, DIR, "resVsPt_L_d0", h2_resVsPt_d0_L_68, h2_resVsPt_d0_L_90, h2_resVsPt_d0_L_99, 0, 0.02 );
  }
    makeResidualIntervalPlot( type, DIR, "resVsEta_eta", h2_resVsEta_eta_68, h2_resVsEta_eta_90, h2_resVsEta_eta_99, 0, 0.05 );
    makeResidualIntervalPlot( type, DIR, "resVsEta_z0", h2_resVsEta_z0_68, h2_resVsEta_z0_90, h2_resVsEta_z0_99, 0, 2.0 );
    makeResidualIntervalPlot( type, DIR, "resVsEta_phi", h2_resVsEta_phi_68, h2_resVsEta_phi_90, h2_resVsEta_phi_99, 0, 0.01 );
    makeResidualIntervalPlot( type, DIR, "resVsEta_ptRel", h2_resVsEta_ptRel_68, h2_resVsEta_ptRel_90, h2_resVsEta_ptRel_99, 0, 0.4 );
    //makeResidualIntervalPlot( type, DIR, "resVsEta_d0", h2_resVsEta_d0_68, h2_resVsEta_d0_90, h2_resVsEta_d0_99, 0, 0.02 );

  if (doDetailedPlots) {
    makeResidualIntervalPlot( type, DIR, "resVsEta_L_eta", h2_resVsEta_eta_L_68, h2_resVsEta_eta_L_90, h2_resVsEta_eta_L_99, 0, 0.05 );
    makeResidualIntervalPlot( type, DIR, "resVsEta_L_z0", h2_resVsEta_z0_L_68, h2_resVsEta_z0_L_90, h2_resVsEta_z0_L_99, 0, 2.0 );
    makeResidualIntervalPlot( type, DIR, "resVsEta_L_phi", h2_resVsEta_phi_L_68, h2_resVsEta_phi_L_90, h2_resVsEta_phi_L_99, 0, 0.01 );
    makeResidualIntervalPlot( type, DIR, "resVsEta_L_ptRel", h2_resVsEta_ptRel_L_68, h2_resVsEta_ptRel_L_90, h2_resVsEta_ptRel_L_99, 0, 0.4 );
    //makeResidualIntervalPlot( type, DIR, "resVsEta_L_d0", h2_resVsEta_d0_L_68, h2_resVsEta_d0_L_90, h2_resVsEta_d0_L_99, 0, 0.02 );
  
    makeResidualIntervalPlot( type, DIR, "resVsEta_H_eta", h2_resVsEta_eta_H_68, h2_resVsEta_eta_H_90, h2_resVsEta_eta_H_99, 0, 0.05 );
    makeResidualIntervalPlot( type, DIR, "resVsEta_H_z0", h2_resVsEta_z0_H_68, h2_resVsEta_z0_H_90, h2_resVsEta_z0_H_99, 0, 2.0 );
    makeResidualIntervalPlot( type, DIR, "resVsEta_H_phi", h2_resVsEta_phi_H_68, h2_resVsEta_phi_H_90, h2_resVsEta_phi_H_99, 0, 0.01 );
    makeResidualIntervalPlot( type, DIR, "resVsEta_H_ptRel", h2_resVsEta_ptRel_H_68, h2_resVsEta_ptRel_H_90, h2_resVsEta_ptRel_H_99, 0, 0.4 );
    //makeResidualIntervalPlot( type, DIR, "resVsEta_H_d0", h2_resVsEta_d0_H_68, h2_resVsEta_d0_H_90, h2_resVsEta_d0_H_99, 0, 0.02 );

    makeResidualIntervalPlot( type, DIR, "resVsPhi_ptRel", h2_resVsPhi_ptRel_68, h2_resVsPhi_ptRel_90, h2_resVsPhi_ptRel_99, 0, 0.5 );
    makeResidualIntervalPlot( type, DIR, "resVsPhi_pt", h2_resVsPhi_pt_68, h2_resVsPhi_pt_90, h2_resVsPhi_pt_99, 0, 20 );
  }
  // }


  // ----------------------------------------------------------------------------------------------------------
  // resoultion vs pt
  // ----------------------------------------------------------------------------------------------------------

  h2_resVsPt_pt_90->SetMinimum(0);
  h2_resVsPt_pt_90->SetMarkerStyle(20);
  h2_resVsPt_pt_90->Draw("p");
  h2_resVsPt_pt_90->Write();
  c.SaveAs(DIR+type+"_resVsPt_pt_90.eps");
  c.SaveAs(DIR+type+"_resVsPt_pt_90.png");

  h2_resVsPt_ptRel_90->SetMinimum(0);
  h2_resVsPt_ptRel_90->SetMarkerStyle(20);
  h2_resVsPt_ptRel_90->Draw("p");
  h2_resVsPt_ptRel_90->Write();
  c.SaveAs(DIR+type+"_resVsPt_ptRel_90.eps");
  c.SaveAs(DIR+type+"_resVsPt_ptRel_90.png");

  h2_resVsPt_z0_90->SetMinimum(0);
  h2_resVsPt_z0_90->SetMarkerStyle(20);
  h2_resVsPt_z0_90->Draw("p");
  h2_resVsPt_z0_90->Write();
  c.SaveAs(DIR+type+"_resVsPt_z0_90.eps");
  c.SaveAs(DIR+type+"_resVsPt_z0_90.png");

  h2_resVsPt_phi_90->SetMinimum(0);
  h2_resVsPt_phi_90->SetMarkerStyle(20);
  h2_resVsPt_phi_90->Draw("p");
  h2_resVsPt_phi_90->Write();
  c.SaveAs(DIR+type+"_resVsPt_phi_90.eps");
  c.SaveAs(DIR+type+"_resVsPt_phi_90.png");

  h2_resVsPt_eta_90->SetMinimum(0);
  h2_resVsPt_eta_90->SetMarkerStyle(20);
  h2_resVsPt_eta_90->Draw("p");
  h2_resVsPt_eta_90->Write();
  c.SaveAs(DIR+type+"_resVsPt_eta_90.eps");
  c.SaveAs(DIR+type+"_resVsPt_eta_90.png");

  /*
  h2_resVsPt_phi_90->SetMinimum(0);
  h2_resVsPt_d0_90->SetMarkerStyle(20);
  h2_resVsPt_d0_90->Draw("p");
  c.SaveAs(DIR+type+"_resVsPt_d0_90.eps");
  c.SaveAs(DIR+type+"_resVsPt_d0_90.png");
  */

  if (doDetailedPlots) {
    h2_resVsPt_eta->Write();

    h2_resVsPt_pt->Write();
    h2_resVsPt_pt_C->Write();    
    h2_resVsPt_pt_I->Write();
    h2_resVsPt_pt_F->Write();

    h2_resVsPt_ptRel->Write();
    h2_resVsPt_ptRel_C->Write();
    h2_resVsPt_ptRel_I->Write();
    h2_resVsPt_ptRel_F->Write();

    h2_mresVsPt_pt->Write();
    h2_mresVsPt_pt_C->Write();
    h2_mresVsPt_pt_I->Write();
    h2_mresVsPt_pt_F->Write();

    h2_resVsPt_d0->Write();

    h2_resVsPt_z0_C->Write();
    h2_resVsPt_z0_I->Write();
    h2_resVsPt_z0_F->Write();

    h2_resVsPt_phi->Write();
    h2_resVsPt_phi_C->Write();
    h2_resVsPt_phi_I->Write();
    h2_resVsPt_phi_F->Write();
  }
    

  // ----------------------------------------------------------------------------------------------------------
  // resolution vs eta
  // ----------------------------------------------------------------------------------------------------------

  h2_resVsEta_eta_90->SetMinimum(0);
  h2_resVsEta_eta_90->SetMarkerStyle(20);
  h2_resVsEta_eta_90->Draw("p");
  c.SaveAs(DIR+type+"_resVsEta_eta_90.eps");
  c.SaveAs(DIR+type+"_resVsEta_eta_90.png");

  if (doDetailedPlots) {
    h2_resVsEta_eta_L_90->Draw("p");
    sprintf(ctxt,"p_{T} < 8 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_eta_L_90.eps");
    c.SaveAs(DIR+type+"_resVsEta_eta_L_90.png");
    
    h2_resVsEta_eta_H_90->Draw("p");
    sprintf(ctxt,"p_{T} > 8 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_eta_H_90.eps");
    c.SaveAs(DIR+type+"_resVsEta_eta_H_90.png");
  }

  h2_resVsEta_z0_90->SetMinimum(0);
  h2_resVsEta_z0_90->SetMarkerStyle(20);
  h2_resVsEta_z0_90->Draw("p");
  c.SaveAs(DIR+type+"_resVsEta_z0_90.eps");
  c.SaveAs(DIR+type+"_resVsEta_z0_90.png");

  if (doDetailedPlots) {
    h2_resVsEta_z0_L_90->Draw();
    sprintf(ctxt,"p_{T} < 8 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_z0_L_90.eps");
    c.SaveAs(DIR+type+"_resVsEta_z0_L_90.png");
    
    h2_resVsEta_z0_H_90->Draw();
    sprintf(ctxt,"p_{T} > 8 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_z0_H_90.eps");
    c.SaveAs(DIR+type+"_resVsEta_z0_H_90.png");
  }

  /*
  h2_resVsEta_d0_90->Draw();
  c.SaveAs(DIR+type+"_resVsEta_d0_90.eps");
  c.SaveAs(DIR+type+"_resVsEta_d0_90.png");

  h2_resVsEta_d0_L_90->Draw();
  sprintf(ctxt,"p_{T} < 8 GeV");
  mySmallText(0.22,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_resVsEta_d0_L_90.eps");
  c.SaveAs(DIR+type+"_resVsEta_d0_L_90.png");

  h2_resVsEta_d0_H_90->Draw();
  sprintf(ctxt,"p_{T} > 8 GeV");
  mySmallText(0.22,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_resVsEta_d0_H_90.eps");
  c.SaveAs(DIR+type+"_resVsEta_d0_H_90.png");
  */

  h2_resVsEta_phi_90->SetMinimum(0);
  h2_resVsEta_phi_90->SetMarkerStyle(20);
  h2_resVsEta_phi_90->Draw("p");
  c.SaveAs(DIR+type+"_resVsEta_phi_90.eps");
  c.SaveAs(DIR+type+"_resVsEta_phi_90.png");

  if (doDetailedPlots) {
    h2_resVsEta_phi_L_90->Draw();
    sprintf(ctxt,"p_{T} < 8 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_phi_L_90.eps");
    c.SaveAs(DIR+type+"_resVsEta_phi_L_90.png");
    
    h2_resVsEta_phi_H_90->Draw();
    sprintf(ctxt,"p_{T} > 8 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_phi_H_90.eps");
    c.SaveAs(DIR+type+"_resVsEta_phi_H_90.png");
  }

  h2_resVsEta_ptRel_90->SetMinimum(0);
  h2_resVsEta_ptRel_90->SetMarkerStyle(20);
  h2_resVsEta_ptRel_90->Draw("p");
  c.SaveAs(DIR+type+"_resVsEta_ptRel_90.eps");
  c.SaveAs(DIR+type+"_resVsEta_ptRel_90.png");

  if (doDetailedPlots) {
    h2_resVsEta_ptRel_L_90->Draw();
    sprintf(ctxt,"p_{T} < 8 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_ptRel_L_90.eps");
    c.SaveAs(DIR+type+"_resVsEta_ptRel_L_90.png");
    
    h2_resVsEta_ptRel_H_90->Draw();
    sprintf(ctxt,"p_{T} > 8 GeV");
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_resVsEta_ptRel_H_90.eps");
    c.SaveAs(DIR+type+"_resVsEta_ptRel_H_90.png");

    h2_resVsEta_eta->Write();
    h2_resVsEta_eta_L->Write();
    h2_resVsEta_eta_H->Write();

    h2_mresVsEta_eta->Write();
    h2_mresVsEta_eta_L->Write();
    h2_mresVsEta_eta_H->Write();

    h2_resVsEta_z0->Write();
    h2_resVsEta_z0_L->Write();
    h2_resVsEta_z0_H->Write();

    h2_resVsEta_d0->Write();
    h2_resVsEta_d0_L->Write();
    h2_resVsEta_d0_H->Write();

    h2_resVsEta_phi->Write();
    h2_resVsEta_phi_L->Write();
    h2_resVsEta_phi_H->Write();
    
    h2_resVsEta_ptRel->Write();
    h2_resVsEta_ptRel_L->Write();
    h2_resVsEta_ptRel_H->Write();
  }
  
  if (doGausFit) {
    h3_resVsEta_eta_L->Write();
    h3_resVsEta_z0_L->Write();
    h3_resVsEta_phi_L->Write();
    h3_resVsEta_ptRel_L->Write();
    
    h3_resVsEta_eta_H->Write();
    h3_resVsEta_z0_H->Write();
    h3_resVsEta_phi_H->Write();
    h3_resVsEta_ptRel_H->Write();
  }

  // resolution vs phi
  if (doDetailedPlots) {
    h2_resVsPhi_pt_90->SetMinimum(0);
    h2_resVsPhi_pt_90->SetMarkerStyle(20);
    h2_resVsPhi_pt_90->Draw("p");
    c.SaveAs(DIR+type+"_resVsPhi_pt_90.eps");
    c.SaveAs(DIR+type+"_resVsPhi_pt_90.png");
    
    h2_resVsPhi_ptRel_90->SetMinimum(0);
    h2_resVsPhi_ptRel_90->SetMarkerStyle(20);
    h2_resVsPhi_ptRel_90->Draw("p");
    c.SaveAs(DIR+type+"_resVsPhi_ptRel_90.eps");
    c.SaveAs(DIR+type+"_resVsPhi_ptRel_90.png");
  }


  // ----------------------------------------------------------------------------------------------------------------
  // track quality plots
  // ----------------------------------------------------------------------------------------------------------------

  if (doDetailedPlots) {
    h_match_trk_nstub->Write();
    h_match_trk_nstub_C->Write();
    h_match_trk_nstub_I->Write();
    h_match_trk_nstub_F->Write();
  }

  h_match_trk_chi2->Draw();
  h_match_trk_chi2->Write();
  sprintf(ctxt,"|eta| < 2.4");
  mySmallText(0.52,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_match_trk_chi2.png");
  c.SaveAs(DIR+type+"_match_trk_chi2.eps");
    
  h_match_trk_chi2_dof->Draw();
  h_match_trk_chi2_dof->Write();
  sprintf(ctxt,"|eta| < 2.4");
  mySmallText(0.52,0.82,1,ctxt);
  c.SaveAs(DIR+type+"_match_trk_chi2_dof.png");
  c.SaveAs(DIR+type+"_match_trk_chi2_dof.eps");

  if (doDetailedPlots) {
    h_match_trk_chi2_C_L->Write();
    h_match_trk_chi2_I_L->Write();
    h_match_trk_chi2_F_L->Write();
    h_match_trk_chi2_C_H->Write();
    h_match_trk_chi2_I_H->Write();
    h_match_trk_chi2_F_H->Write();

    h_match_trk_chi2_dof_C_L->Write();
    h_match_trk_chi2_dof_I_L->Write();
    h_match_trk_chi2_dof_F_L->Write();
    h_match_trk_chi2_dof_C_H->Write();
    h_match_trk_chi2_dof_I_H->Write();
    h_match_trk_chi2_dof_F_H->Write();
  }
   
 
  // ----------------------------------------------------------------------------------------------------------------
  // efficiency plots  
  // ----------------------------------------------------------------------------------------------------------------

  // rebin pt/phi plots
  h_tp_pt->Rebin(2);
  h_tp_phi->Rebin(2);
  h_match_tp_pt->Rebin(2);
  h_match_tp_phi->Rebin(2);

  // calculate the effeciency
  h_match_tp_pt->Sumw2();
  h_tp_pt->Sumw2();
  TH1F* h_eff_pt = (TH1F*) h_match_tp_pt->Clone();
  h_eff_pt->SetName("eff_pt");
  h_eff_pt->GetYaxis()->SetTitle("Efficiency");
  h_eff_pt->Divide(h_match_tp_pt, h_tp_pt, 1.0, 1.0, "B");

  h_match_tp_pt_L->Sumw2();
  h_tp_pt_L->Sumw2();
  TH1F* h_eff_pt_L = (TH1F*) h_match_tp_pt_L->Clone();
  h_eff_pt_L->SetName("eff_pt_L");
  h_eff_pt_L->GetYaxis()->SetTitle("Efficiency");
  h_eff_pt_L->Divide(h_match_tp_pt_L, h_tp_pt_L, 1.0, 1.0, "B");

  h_match_tp_pt_LC->Sumw2();
  h_tp_pt_LC->Sumw2();
  TH1F* h_eff_pt_LC = (TH1F*) h_match_tp_pt_LC->Clone();
  h_eff_pt_LC->SetName("eff_pt_LC");
  h_eff_pt_LC->GetYaxis()->SetTitle("Efficiency");
  h_eff_pt_LC->Divide(h_match_tp_pt_LC, h_tp_pt_LC, 1.0, 1.0, "B");

  h_match_tp_pt_H->Sumw2();
  h_tp_pt_H->Sumw2();
  TH1F* h_eff_pt_H = (TH1F*) h_match_tp_pt_H->Clone();
  h_eff_pt_H->SetName("eff_pt_H");
  h_eff_pt_H->GetYaxis()->SetTitle("Efficiency");
  h_eff_pt_H->Divide(h_match_tp_pt_H, h_tp_pt_H, 1.0, 1.0, "B");

  h_match_tp_eta->Sumw2();
  h_tp_eta->Sumw2();
  TH1F* h_eff_eta = (TH1F*) h_match_tp_eta->Clone();
  h_eff_eta->SetName("eff_eta");
  h_eff_eta->GetYaxis()->SetTitle("Efficiency");
  h_eff_eta->Divide(h_match_tp_eta, h_tp_eta, 1.0, 1.0, "B");
  
  h_match_tp_eta_L->Sumw2();
  h_tp_eta_L->Sumw2();
  TH1F* h_eff_eta_L = (TH1F*) h_match_tp_eta_L->Clone();
  h_eff_eta_L->SetName("eff_eta_L");
  h_eff_eta_L->GetYaxis()->SetTitle("Efficiency");
  h_eff_eta_L->Divide(h_match_tp_eta_L, h_tp_eta_L, 1.0, 1.0, "B");

  h_match_tp_eta_H->Sumw2();
  h_tp_eta_H->Sumw2();
  TH1F* h_eff_eta_H = (TH1F*) h_match_tp_eta_H->Clone();
  h_eff_eta_H->SetName("eff_eta_H");
  h_eff_eta_H->GetYaxis()->SetTitle("Efficiency");
  h_eff_eta_H->Divide(h_match_tp_eta_H, h_tp_eta_H, 1.0, 1.0, "B");

  h_match_tp_phi->Sumw2();
  h_tp_phi->Sumw2();
  TH1F* h_eff_phi = (TH1F*) h_match_tp_phi->Clone();
  h_eff_phi->SetName("eff_phi");
  h_eff_phi->GetYaxis()->SetTitle("Efficiency");
  h_eff_phi->Divide(h_match_tp_phi, h_tp_phi, 1.0, 1.0, "B");

  h_match_tp_z0->Sumw2();
  h_tp_z0->Sumw2();
  TH1F* h_eff_z0 = (TH1F*) h_match_tp_z0->Clone();
  h_eff_z0->SetName("eff_z0");
  h_eff_z0->GetYaxis()->SetTitle("Efficiency");
  h_eff_z0->Divide(h_match_tp_z0, h_tp_z0, 1.0, 1.0, "B");

  h_match_tp_z0_L->Sumw2();
  h_tp_z0_L->Sumw2();
  TH1F* h_eff_z0_L = (TH1F*) h_match_tp_z0_L->Clone();
  h_eff_z0_L->SetName("eff_z0_L");
  h_eff_z0_L->GetYaxis()->SetTitle("Efficiency");
  h_eff_z0_L->Divide(h_match_tp_z0_L, h_tp_z0_L, 1.0, 1.0, "B");

  h_match_tp_z0_H->Sumw2();
  h_tp_z0_H->Sumw2();
  TH1F* h_eff_z0_H = (TH1F*) h_match_tp_z0_H->Clone();
  h_eff_z0_H->SetName("eff_z0_H");
  h_eff_z0_H->GetYaxis()->SetTitle("Efficiency");
  h_eff_z0_H->Divide(h_match_tp_z0_H, h_tp_z0_H, 1.0, 1.0, "B");

  h_match_tp_d0->Sumw2();
  h_tp_d0->Sumw2();
  TH1F* h_eff_d0 = (TH1F*) h_match_tp_d0->Clone();
  h_eff_d0->SetName("eff_d0");
  h_eff_d0->GetYaxis()->SetTitle("Efficiency");
  h_eff_d0->Divide(h_match_tp_d0, h_tp_d0, 1.0, 1.0, "B");


  // set the axis range
  h_eff_pt  ->SetAxisRange(0,1.1,"Y");
  h_eff_pt_L->SetAxisRange(0,1.1,"Y");
  h_eff_pt_LC->SetAxisRange(0,1.1,"Y");
  h_eff_pt_H->SetAxisRange(0,1.1,"Y");
  h_eff_eta ->SetAxisRange(0,1.1,"Y");
  h_eff_eta_L ->SetAxisRange(0,1.1,"Y");
  h_eff_eta_H ->SetAxisRange(0,1.1,"Y");
  h_eff_phi ->SetAxisRange(0,1.1,"Y");
  h_eff_z0  ->SetAxisRange(0,1.1,"Y");
  h_eff_z0_L  ->SetAxisRange(0,1.1,"Y");
  h_eff_z0_H  ->SetAxisRange(0,1.1,"Y");
  h_eff_d0  ->SetAxisRange(0,1.1,"Y");

  if (type.Contains("Electron") || type.Contains("Pion")) h_eff_pt->SetAxisRange(0,49,"X");

  gPad->SetGridx();
  gPad->SetGridy();

  // draw and save plots
  h_eff_pt->Draw();
  h_eff_pt->Write();
  c.SaveAs(DIR+type+"_eff_pt.eps");
  c.SaveAs(DIR+type+"_eff_pt.png");

  if (type.Contains("Mu")) {
    h_eff_pt->GetYaxis()->SetRangeUser(0.8,1.01); // zoomed-in plot
    c.SaveAs(DIR+type+"_eff_pt_zoom.eps");
    c.SaveAs(DIR+type+"_eff_pt_zoom.png");
  }

  //  if (doDetailedPlots) {
    h_eff_pt_L->Draw();
    h_eff_pt_L->Write();
    sprintf(ctxt,"p_{T} < 8 GeV");
    mySmallText(0.45,0.5,1,ctxt);
    c.SaveAs(DIR+type+"_eff_pt_L.eps");
    c.SaveAs(DIR+type+"_eff_pt_L.png");

    if (type.Contains("Mu")) {
      h_eff_pt_L->GetYaxis()->SetRangeUser(0.8,1.01); // zoomed-in plot
      c.SaveAs(DIR+type+"_eff_pt_L_zoom.eps");
      c.SaveAs(DIR+type+"_eff_pt_L_zoom.png");
    }
    
    h_eff_pt_LC->Draw();
    h_eff_pt_LC->Write();
    sprintf(ctxt,"p_{T} < 8 GeV, |#eta|<1.0");
    mySmallText(0.45,0.5,1,ctxt);
    c.SaveAs(DIR+type+"_eff_pt_LC.eps");
    c.SaveAs(DIR+type+"_eff_pt_LC.png");
    
    h_eff_pt_H->Draw();
    h_eff_pt_H->Write();
    sprintf(ctxt,"p_{T} > 8 GeV");
    mySmallText(0.45,0.5,1,ctxt);
    c.SaveAs(DIR+type+"_eff_pt_H.eps");
    c.SaveAs(DIR+type+"_eff_pt_H.png");
    //  }

  h_eff_eta->Draw();
  h_eff_eta->Write();
  c.SaveAs(DIR+type+"_eff_eta.eps");
  c.SaveAs(DIR+type+"_eff_eta.png");
  
  if (type.Contains("Mu")) {
    h_eff_eta->GetYaxis()->SetRangeUser(0.8,1.01); // zoomed-in plot
    c.SaveAs(DIR+type+"_eff_eta_zoom.eps");
    c.SaveAs(DIR+type+"_eff_eta_zoom.png");
  }

  if (doDetailedPlots) {
    h_eff_eta_L->Draw();
    h_eff_eta_L->Write();
    sprintf(ctxt,"p_{T} < 8 GeV");
    mySmallText(0.45,0.5,1,ctxt);
    c.SaveAs(DIR+type+"_eff_eta_L.eps");
    c.SaveAs(DIR+type+"_eff_eta_L.png");
    
    h_eff_eta_H->Draw();
    h_eff_eta_H->Write();
    sprintf(ctxt,"p_{T} > 8 GeV");
    mySmallText(0.45,0.5,1,ctxt);
    c.SaveAs(DIR+type+"_eff_eta_H.eps");
    c.SaveAs(DIR+type+"_eff_eta_H.png");
    
    h_eff_z0->Draw();
    h_eff_z0->Write();
    c.SaveAs(DIR+type+"_eff_z0.eps");
    c.SaveAs(DIR+type+"_eff_z0.png");

    h_eff_z0_L->Write();
    h_eff_z0_H->Write();

    h_eff_phi->Draw();
    h_eff_phi->Write();
    c.SaveAs(DIR+type+"_eff_phi.eps");
    c.SaveAs(DIR+type+"_eff_phi.png");
    
    if (type.Contains("Mu")) {
      h_eff_phi->GetYaxis()->SetRangeUser(0.8,1.01); // zoomed-in plot
      c.SaveAs(DIR+type+"_eff_phi_zoom.eps");
      c.SaveAs(DIR+type+"_eff_phi_zoom.png");
    }
    
    h_eff_d0->Draw();
    h_eff_d0->Write();
    c.SaveAs(DIR+type+"_eff_d0.eps");
    c.SaveAs(DIR+type+"_eff_d0.png");
  }
  
  gPad->SetGridx(0);
  gPad->SetGridy(0);


  // ----------------------------------------------------------------------------------------------------------------
  // more resolution plots
  // ----------------------------------------------------------------------------------------------------------------

  float rms = 0;

  if (doDetailedPlots) {

    // draw and save plots
    h_res_pt->Draw();
    rms = h_res_pt->GetRMS();
    sprintf(ctxt,"RMS = %.4f",rms);
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res_pt.eps");
    c.SaveAs(DIR+type+"_res_pt.png");
    
    h_res_ptRel->Draw();
    rms = h_res_ptRel->GetRMS();
    sprintf(ctxt,"RMS = %.4f",rms);	
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res_ptRel.eps");
    c.SaveAs(DIR+type+"_res_ptRel.png");

    h_res_eta->Draw();
    rms = h_res_eta->GetRMS();
    sprintf(ctxt,"RMS = %.3e",rms);	
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res_eta.eps");
    c.SaveAs(DIR+type+"_res_eta.png");
    
    h_res_phi->Draw();
    rms = h_res_phi->GetRMS();
    sprintf(ctxt,"RMS = %.3e",rms);	
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res_phi.eps");
    c.SaveAs(DIR+type+"_res_phi.png");
    
    h_res_z0->Draw();
    rms = h_res_z0->GetRMS();
    sprintf(ctxt,"RMS = %.4f",rms);	
    mySmallText(0.22,0.82,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0.eps");
    c.SaveAs(DIR+type+"_res_z0.png");
    
    h_res_z0_C->Draw();
    rms = h_res_z0_C->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_C.eps");
    c.SaveAs(DIR+type+"_res_z0_C.png");
    
    h_res_z0_I->Draw();
    rms = h_res_z0_I->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_I.eps");
    c.SaveAs(DIR+type+"_res_z0_I.png");
    
    h_res_z0_F->Draw();
    rms = h_res_z0_F->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_F.eps");
    c.SaveAs(DIR+type+"_res_z0_F.png");

    h_res_z0_C_L->Draw();
    h_res_z0_C_L->Write();
    rms = h_res_z0_C_L->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_C_L.eps");
    c.SaveAs(DIR+type+"_res_z0_C_L.png");
    
    h_res_z0_I_L->Draw();
    h_res_z0_I_L->Write();
    rms = h_res_z0_I_L->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_I_L.eps");
    c.SaveAs(DIR+type+"_res_z0_I_L.png");
    
    h_res_z0_F_L->Draw();
    h_res_z0_F_L->Write();
    rms = h_res_z0_F_L->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_F_L.eps");
    c.SaveAs(DIR+type+"_res_z0_F_L.png");

    h_res_z0_C_H->Draw();
    h_res_z0_C_H->Write();
    rms = h_res_z0_C_H->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| < 0.8");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_C_H.eps");
    c.SaveAs(DIR+type+"_res_z0_C_H.png");
    
    h_res_z0_I_H->Draw();
    h_res_z0_I_H->Write();
    rms = h_res_z0_I_H->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"0.8 < |eta| < 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_I_H.eps");
    c.SaveAs(DIR+type+"_res_z0_I_H.png");
    
    h_res_z0_F_H->Draw();
    h_res_z0_F_H->Write();
    rms = h_res_z0_F_H->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"|eta| > 1.6");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_F_H.eps");
    c.SaveAs(DIR+type+"_res_z0_F_H.png");

    h_res_z0_L->Draw();
    h_res_z0_L->Write();
    rms = h_res_z0_L->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"p_{T} < 5 GeV");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_L.eps");
    c.SaveAs(DIR+type+"_res_z0_L.png");
    
    h_res_z0_H->Draw();
    h_res_z0_H->Write();
    rms = h_res_z0_H->GetRMS();
    sprintf(ctxt,"RMS = %.4f;",rms);
    mySmallText(0.22,0.82,1,ctxt);
    sprintf(ctxt,"p_{T} > 15 GeV");
    mySmallText(0.22,0.76,1,ctxt);
    c.SaveAs(DIR+type+"_res_z0_H.eps");
    c.SaveAs(DIR+type+"_res_z0_H.png");

    if (h_res_d0->GetEntries()>0) {
      h_res_d0->Draw();
      rms = h_res_d0->GetRMS();
      sprintf(ctxt,"RMS = %.4f",rms);	
      mySmallText(0.22,0.82,1,ctxt);
      c.SaveAs(DIR+type+"_res_d0.eps");
      c.SaveAs(DIR+type+"_res_d0.png");

      h_res_d0_C->Draw();
      h_res_d0_C->Write();
      rms = h_res_d0_C->GetRMS();
      sprintf(ctxt,"RMS = %.4f;",rms);
      mySmallText(0.22,0.82,1,ctxt);
      sprintf(ctxt,"|eta| < 0.8");
      mySmallText(0.22,0.76,1,ctxt);
      c.SaveAs(DIR+type+"_res_d0_C.eps");
      c.SaveAs(DIR+type+"_res_d0_C.png");
    
      h_res_d0_I->Draw();
      h_res_d0_I->Write();
      rms = h_res_d0_I->GetRMS();
      sprintf(ctxt,"RMS = %.4f;",rms);
      mySmallText(0.22,0.82,1,ctxt);
      sprintf(ctxt,"0.8 < |eta| < 1.6");
      mySmallText(0.22,0.76,1,ctxt);
      c.SaveAs(DIR+type+"_res_d0_I.eps");
      c.SaveAs(DIR+type+"_res_d0_I.png");
    
      h_res_d0_F->Draw();
      h_res_d0_F->Write();
      rms = h_res_d0_F->GetRMS();
      sprintf(ctxt,"RMS = %.4f;",rms);
      mySmallText(0.22,0.82,1,ctxt);
      sprintf(ctxt,"|eta| > 1.6");
      mySmallText(0.22,0.76,1,ctxt);
      c.SaveAs(DIR+type+"_res_d0_F.eps");
      c.SaveAs(DIR+type+"_res_d0_F.png");
    
      h_res_d0_C_L->Draw();
      h_res_d0_C_L->Write();
      rms = h_res_d0_C_L->GetRMS();
      sprintf(ctxt,"RMS = %.4f;",rms);
      mySmallText(0.22,0.82,1,ctxt);
      sprintf(ctxt,"|eta| < 0.8");
      mySmallText(0.22,0.76,1,ctxt);
      c.SaveAs(DIR+type+"_res_d0_C_L.eps");
      c.SaveAs(DIR+type+"_res_d0_C_L.png");
    
      h_res_d0_I_L->Draw();
      h_res_d0_I_L->Write();
      rms = h_res_d0_I_L->GetRMS();
      sprintf(ctxt,"RMS = %.4f;",rms);
      mySmallText(0.22,0.82,1,ctxt);
      sprintf(ctxt,"0.8 < |eta| < 1.6");
      mySmallText(0.22,0.76,1,ctxt);
      c.SaveAs(DIR+type+"_res_d0_I_L.eps");
      c.SaveAs(DIR+type+"_res_d0_I_L.png");
    
      h_res_d0_F_L->Draw();
      h_res_d0_F_L->Write();
      rms = h_res_d0_F_L->GetRMS();
      sprintf(ctxt,"RMS = %.4f;",rms);
      mySmallText(0.22,0.82,1,ctxt);
      sprintf(ctxt,"|eta| > 1.6");
      mySmallText(0.22,0.76,1,ctxt);
      c.SaveAs(DIR+type+"_res_d0_F_L.eps");
      c.SaveAs(DIR+type+"_res_d0_F_L.png");
      
      h_res_d0_C_H->Draw();
      h_res_d0_C_H->Write();
      rms = h_res_d0_C_H->GetRMS();
      sprintf(ctxt,"RMS = %.4f;",rms);
      mySmallText(0.22,0.82,1,ctxt);
      sprintf(ctxt,"|eta| < 0.8");
      mySmallText(0.22,0.76,1,ctxt);
      c.SaveAs(DIR+type+"_res_d0_C_H.eps");
      c.SaveAs(DIR+type+"_res_d0_C_H.png");
    
      h_res_d0_I_H->Draw();
      h_res_d0_I_H->Write();
      rms = h_res_d0_I_H->GetRMS();
      sprintf(ctxt,"RMS = %.4f;",rms);
      mySmallText(0.22,0.82,1,ctxt);
      sprintf(ctxt,"0.8 < |eta| < 1.6");
      mySmallText(0.22,0.76,1,ctxt);
      c.SaveAs(DIR+type+"_res_d0_I_H.eps");
      c.SaveAs(DIR+type+"_res_d0_I_H.png");
      
      h_res_d0_F_H->Draw();
      h_res_d0_F_H->Write();
      rms = h_res_d0_F_H->GetRMS();
      sprintf(ctxt,"RMS = %.4f;",rms);
      mySmallText(0.22,0.82,1,ctxt);
      sprintf(ctxt,"|eta| > 1.6");
      mySmallText(0.22,0.76,1,ctxt);
      c.SaveAs(DIR+type+"_res_d0_F_H.eps");
      c.SaveAs(DIR+type+"_res_d0_F_H.png");
      
      h_res_d0_L->Draw();
      h_res_d0_L->Write();
      rms = h_res_d0_L->GetRMS();
      sprintf(ctxt,"RMS = %.4f;",rms);
      mySmallText(0.22,0.82,1,ctxt);
      sprintf(ctxt,"p_{T} < 5 GeV");
      mySmallText(0.22,0.76,1,ctxt);
      c.SaveAs(DIR+type+"_res_d0_L.eps");
      c.SaveAs(DIR+type+"_res_d0_L.png");
      
      h_res_d0_H->Draw();
      h_res_d0_H->Write();
      rms = h_res_d0_H->GetRMS();
      sprintf(ctxt,"RMS = %.4f;",rms);
      mySmallText(0.22,0.82,1,ctxt);
      sprintf(ctxt,"p_{T} > 15 GeV");
      mySmallText(0.22,0.76,1,ctxt);
      c.SaveAs(DIR+type+"_res_d0_H.eps");
      c.SaveAs(DIR+type+"_res_d0_H.png");
    }
  }
  

  // ---------------------------------------------------------------------------------------------------------
  // total track rates vs pt 

  h_trk_vspt->Scale(1.0/nevt);
  h_tp_vspt->Scale(1.0/nevt);

  h_tp_vspt->GetYaxis()->SetTitle("Tracks / event");
  h_tp_vspt->GetXaxis()->SetTitle("Track p_{T} [GeV]");
  h_tp_vspt->SetLineColor(4);
  h_tp_vspt->SetLineStyle(2);

  float max = h_tp_vspt->GetMaximum();
  if (h_trk_vspt->GetMaximum() > max) max = h_trk_vspt->GetMaximum();
  h_tp_vspt->SetAxisRange(0,max*1.05,"Y");  

  h_tp_vspt->Draw();
  h_trk_vspt->Draw("same");
  h_tp_vspt->Draw("same");

  h_trk_vspt->Write();
  h_tp_vspt->Write();

  char txt[500];
  sprintf(txt,"average # tracks/event = %.1f",h_trk_vspt->GetSum());
  mySmallText(0.5,0.85,1,txt);
  char txt3[500];
  sprintf(txt3,"average # TPs(stubs in #geq 4 layers)/");
  char txt2[500];
  sprintf(txt2,"event = %.1f",h_tp_vspt->GetSum());
  mySmallText(0.5,0.79,4,txt3);
  mySmallText(0.5,0.74,4,txt2);

  if (doDetailedPlots) {
    c.SaveAs(DIR+type+"_trackrate_vspt.png");
    c.SaveAs(DIR+type+"_trackrate_vspt.eps");
  }

  // ---------------------------------------------------------------------------------------------------------
  // sum track/ TP pt in jets
  /*
  if (TP_select_injet > 0) {
    
    TH1F* h_frac_sumpt_vspt = (TH1F*) h_jet_trk_sumpt_vspt->Clone();
    h_frac_sumpt_vspt->SetName("frac_sumpt_vspt");
    h_frac_sumpt_vspt->GetYaxis()->SetTitle("L1 sum(p_{T}) / TP sum(p_{T})");
    h_frac_sumpt_vspt->Divide(h_jet_trk_sumpt_vspt, h_jet_tp_sumpt_vspt, 1.0, 1.0, "B");
    
    TH1F* h_frac_sumpt_vseta = (TH1F*) h_jet_trk_sumpt_vseta->Clone();
    h_frac_sumpt_vseta->SetName("frac_sumpt_vseta");
    h_frac_sumpt_vseta->GetYaxis()->SetTitle("L1 sum(p_{T}) / TP sum(p_{T})");
    h_frac_sumpt_vseta->Divide(h_jet_trk_sumpt_vseta, h_jet_tp_sumpt_vseta, 1.0, 1.0, "B");
    
    
    TH1F* h_matchfrac_sumpt_vspt = (TH1F*) h_jet_matchtrk_sumpt_vspt->Clone();
    h_matchfrac_sumpt_vspt->SetName("matchfrac_sumpt_vspt");
    h_matchfrac_sumpt_vspt->GetYaxis()->SetTitle("Matched L1 sum(p_{T}) / TP sum(p_{T})");
    h_matchfrac_sumpt_vspt->Divide(h_jet_matchtrk_sumpt_vspt, h_jet_tp_sumpt_vspt, 1.0, 1.0, "B");
    
    TH1F* h_matchfrac_sumpt_vseta = (TH1F*) h_jet_matchtrk_sumpt_vseta->Clone();
    h_matchfrac_sumpt_vseta->SetName("matchfrac_sumpt_vseta");
    h_matchfrac_sumpt_vseta->GetYaxis()->SetTitle("Matched L1 sum(p_{T}) / TP sum(p_{T})");
    h_matchfrac_sumpt_vseta->Divide(h_jet_matchtrk_sumpt_vseta, h_jet_tp_sumpt_vseta, 1.0, 1.0, "B");

    
    h_frac_sumpt_vspt->Draw();
    c.SaveAs(DIR+type+"_sumpt_vspt.png"); 
    c.SaveAs(DIR+type+"_sumpt_vspt.eps"); 
    
    h_frac_sumpt_vseta->Draw();
    c.SaveAs(DIR+type+"_sumpt_vseta.png"); 
    c.SaveAs(DIR+type+"_sumpt_vseta.eps"); 
    
    h_matchfrac_sumpt_vspt->Draw();
    c.SaveAs(DIR+type+"_sumpt_match_vspt.png"); 
    c.SaveAs(DIR+type+"_sumpt_match_vspt.eps"); 
    
    h_matchfrac_sumpt_vseta->Draw();
    c.SaveAs(DIR+type+"_sumpt_match_vseta.png"); 
    c.SaveAs(DIR+type+"_sumpt_match_vseta.eps"); 
  }
  */


  // nbr tracks per event 

  if (doDetailedPlots) {
    h_ntrk_pt3->Draw();
    h_ntrk_pt3->Write();
    c.SaveAs(DIR+type+"_trackrate_pt3_perevt.png");
    c.SaveAs(DIR+type+"_trackrate_pt3_perevt.eps");
    
    h_ntrk_pt10->Draw();
    h_ntrk_pt10->Write();
    c.SaveAs(DIR+type+"_trackrate_pt10_perevt.png");
    c.SaveAs(DIR+type+"_trackrate_pt10_perevt.eps");
  }

  fout->Close();



  // ---------------------------------------------------------------------------------------------------------
  //some printouts

  float k = (float)n_match_eta1p0;
  float N = (float)n_all_eta1p0;
  if (fabs(N)>0) cout << endl << "efficiency for |eta| < 1.0 = " << k/N*100.0 << " +- " << 1.0/N*sqrt(k*(1.0 - k/N))*100.0 << endl;
  k = (float)n_match_eta1p75;
  N = (float)n_all_eta1p75;
  if (fabs(N)>0) cout << "efficiency for 1.0 < |eta| < 1.75 = " << k/N*100.0 << " +- " << 1.0/N*sqrt(k*(1.0 - k/N))*100.0 << endl;
  k = (float)n_match_eta2p5;
  N = (float)n_all_eta2p5;
  if (fabs(N)>0) cout << "efficiency for 1.75 < |eta| < 2.5 = " << k/N*100.0 << " +- " << 1.0/N*sqrt(k*(1.0 - k/N))*100.0 << endl;
  N = (float) n_all_eta1p0 + n_all_eta1p75 + n_all_eta2p5;
  k = (float) n_match_eta1p0 + n_match_eta1p75 + n_match_eta2p5;
  if (fabs(N)>0) cout << "combined efficiency for |eta| < 2.5 = " << k/N*100.0 << " +- " << 1.0/N*sqrt(k*(1.0 - k/N))*100.0 << endl << endl;

  // track rates
  cout << "# TP/event (pt > 2.0) = " << (float)ntp_pt2/nevt << endl;
  cout << "# TP/event (pt > 3.0) = " << (float)ntp_pt3/nevt << endl;
  cout << "# TP/event (pt > 10.0) = " << (float)ntp_pt10/nevt << endl;

  cout << "# tracks/event (pt > 2.0) = " << (float)ntrk_pt2/nevt << endl;
  cout << "# tracks/event (pt > 3.0) = " << (float)ntrk_pt3/nevt << endl;
  cout << "# tracks/event (pt > 10.0) = " << (float)ntrk_pt10/nevt << endl;

  //cout << "# tracks/event (pt > 3.0), 4stubs = " << (float)ntrk_4stub_pt3/nevt << endl;
  //cout << "# tracks/event (pt > 10.0), 4stubs = " << (float)ntrk_4stub_pt10/nevt << endl;
  //cout << "# tracks/event (pt > 3.0), 4stubs+chi2 = " << (float)ntrk_4stubchi2_pt3/nevt << endl;
  //cout << "# tracks/event (pt > 10.0), 4stubs+chi2 = " << (float)ntrk_4stubchi2_pt10/nevt << endl;


}


void SetPlotStyle() {

  // from ATLAS plot style macro

  // use plain black on white colors
  gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameFillColor(0);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(0);
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(0);
  gStyle->SetStatColor(0);
  gStyle->SetHistLineColor(1);

  gStyle->SetPalette(1);

  // set the paper & margin sizes
  gStyle->SetPaperSize(20,26);
  gStyle->SetPadTopMargin(0.05);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadBottomMargin(0.16);
  gStyle->SetPadLeftMargin(0.16);

  // set title offsets (for axis label)
  gStyle->SetTitleXOffset(1.4);
  gStyle->SetTitleYOffset(1.4);

  // use large fonts
  gStyle->SetTextFont(42);
  gStyle->SetTextSize(0.05);
  gStyle->SetLabelFont(42,"x");
  gStyle->SetTitleFont(42,"x");
  gStyle->SetLabelFont(42,"y");
  gStyle->SetTitleFont(42,"y");
  gStyle->SetLabelFont(42,"z");
  gStyle->SetTitleFont(42,"z");
  gStyle->SetLabelSize(0.05,"x");
  gStyle->SetTitleSize(0.05,"x");
  gStyle->SetLabelSize(0.05,"y");
  gStyle->SetTitleSize(0.05,"y");
  gStyle->SetLabelSize(0.05,"z");
  gStyle->SetTitleSize(0.05,"z");

  // use bold lines and markers
  gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(1.2);
  gStyle->SetHistLineWidth(2.);
  gStyle->SetLineStyleString(2,"[12 12]");

  // get rid of error bar caps
  gStyle->SetEndErrorSize(0.);

  // do not display any of the standard histogram decorations
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);

  // put tick marks on top and RHS of plots
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);

}


void mySmallText(Double_t x,Double_t y,Color_t color,char *text) {
  Double_t tsize=0.044;
  TLatex l;
  l.SetTextSize(tsize); 
  l.SetNDC();
  l.SetTextColor(color);
  l.DrawLatex(x,y,text);
}

double getIntervalContainingFractionOfEntries( TH1* absResidualHistogram, double quantileToCalculate ) {
  
  double totalIntegral = absResidualHistogram->Integral( 0, absResidualHistogram->GetNbinsX() + 1 );

  // Check that the interval is not somewhere in the overflow bin
  double maxAllowedEntriesInOverflow = totalIntegral * ( 1 - quantileToCalculate );
  double nEntriesInOverflow = absResidualHistogram->GetBinContent( absResidualHistogram->GetNbinsX() + 1 );
  if ( nEntriesInOverflow > maxAllowedEntriesInOverflow ) {
        // cout << "WARNING : Cannot compute range corresponding to interval, as it is in the overflow bin" << endl;
        return absResidualHistogram->GetXaxis()->GetXmax()  * 1.2;
  }

  // Calculate quantile for given interval
  double interval[1];
  double quantile[1] = { quantileToCalculate };
  absResidualHistogram->GetQuantiles( 1, interval, quantile);

  return interval[0];
}

void makeResidualIntervalPlot( TString type, TString dir, TString variable, TH1F* h_68, TH1F* h_90, TH1F* h_99, double minY, double maxY ) {

  TCanvas c;

  h_68->SetMinimum( minY );
  h_90->SetMinimum( minY );
  h_99->SetMinimum( minY );

  h_68->SetMaximum( maxY );
  h_90->SetMaximum( maxY );
  h_99->SetMaximum( maxY );

  h_68->SetMarkerStyle(20);
  h_90->SetMarkerStyle(26);
  h_99->SetMarkerStyle(24);

  h_68->Draw("P");
  h_68->Write();
  h_90->Draw("P same");
  h_90->Write();
  h_99->Draw("P same");
  h_99->Write();

  TLegend* l = new TLegend(0.65,0.65,0.85,0.85);
  l->SetFillStyle(0);
  l->SetBorderSize(0);
  l->SetTextSize(0.04);
  l->AddEntry(h_99,"99%","p");
  l->AddEntry(h_90,"90%","p");
  l->AddEntry(h_68,"68%","p");
  l->SetTextFont(42);
  l->Draw();  

  c.SaveAs(dir+type+"_"+variable+"_interval.png");

  delete l;
}


