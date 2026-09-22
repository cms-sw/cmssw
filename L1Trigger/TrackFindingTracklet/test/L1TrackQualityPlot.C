// ----------------------------------------------------------------------------------------------------------------
// Basic example ROOT script for making tracking performance plots using the ntuples produced by L1TrackNtupleMaker.cc
//
// e.g. in ROOT do: [0] .L L1TrackQualityPlot.C++
//                  [1] L1TrackQualityPlot("TTbar_PU200_D76")
//
// By Claire Savard, July 2020
// Based off of L1TrackNtuplePlot.C
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
#include "TGraph.h"
#include "TError.h"
#include "TGraphErrors.h"
#include "TGraphPainter.h"
#include "TSystem.h"
#include "TMultiGraph.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

void SetPlotStyle();

// ----------------------------------------------------------------------------------------------------------------
// Main script
// ----------------------------------------------------------------------------------------------------------------

void L1TrackQualityPlot(TString type, TString type_dir = "", TString treeName = "") {
  // type:              this is the name of the input file you want to process (minus ".root" extension)
  // type_dir:          this is the directory containing the input file you want to process. Note that this must end with a "/", as in "EventSets/"

  gROOT->SetBatch();
  gErrorIgnoreLevel = kWarning;

  SetPlotStyle();
  gSystem->mkdir("MVA_plots");

  // ----------------------------------------------------------------------------------------------------------------
  // define input options
  // ----------------------------------------------------------------------------------------------------------------

  // TP selection cuts (from Python script)
  float TP_minPt = 2.0;
  float TP_maxEta = 2.5;
  float TP_maxLxy = 1.0;
  float TP_maxLz = 30.0;
  float TP_maxD0 = 1.0;
  int TP_select_eventid = 0;
  int TP_select_pdgid = 0;

  // Track quality cuts (from Python script)
  int L1Tk_minNstub = 4;
  float L1Tk_maxChi2 = 999999;
  float L1Tk_maxChi2dof = 999999;

  // ----------------------------------------------------------------------------------------------------------------
  // read ntuples
  TChain* tree = new TChain("L1TrackNtuple" + treeName + "/eventTree");
  tree->Add(type_dir + type + ".root");

  if (tree->GetEntries() == 0) {
    cout << "File doesn't exist or is empty, returning..."
         << endl;  //cout's kept in this file as it is an example standalone plotting script, not running in central CMSSW
    return;
  }

  // ----------------------------------------------------------------------------------------------------------------
  // define leafs & branches
  // all L1 tracks
  vector<float>* trk_pt;
  vector<float>* trk_eta;
  vector<float>* trk_phi;
  vector<float>* trk_chi2;
  vector<float>* trk_chi2rphi;
  vector<float>* trk_chi2rz;
  vector<int>* trk_nstub;
  vector<int>* trk_lhits;
  vector<int>* trk_dhits;
  vector<int>* trk_seed;
  vector<int>* trk_hitpattern;
  vector<unsigned int>* trk_phiSector;
  vector<int>* trk_genuine;
  vector<int>* trk_loose;
  vector<float>* trk_MVA1;
  vector<int>* trk_matchtp_eventtype;
  vector<float>* trk_matchtp_pdgid;

  // TP branches (for efficiency calculation)
  vector<float>* tp_pt;
  vector<float>* tp_eta;
  vector<float>* tp_phi;
  vector<float>* tp_z0;
  vector<float>* tp_d0;
  vector<float>* tp_lxy;
  vector<float>* tp_lz;
  vector<int>* tp_nmatch;
  vector<int>* tp_eventid;
  vector<int>* tp_pdgid;

  // Matched track branches (aligned with TPs)
  vector<int>* matchtrk_nstub;
  vector<float>* matchtrk_chi2;
  vector<float>* matchtrk_chi2_dof;
  vector<float>* matchtrk_MVA1;

  TBranch* b_trk_pt;
  TBranch* b_trk_eta;
  TBranch* b_trk_phi;
  TBranch* b_trk_chi2;
  TBranch* b_trk_chi2rphi;
  TBranch* b_trk_chi2rz;
  TBranch* b_trk_nstub;
  TBranch* b_trk_lhits;
  TBranch* b_trk_dhits;
  TBranch* b_trk_phiSector;
  TBranch* b_trk_seed;
  TBranch* b_trk_hitpattern;
  TBranch* b_trk_genuine;
  TBranch* b_trk_loose;
  TBranch* b_trk_MVA1;
  TBranch* b_trk_matchtp_eventtype;
  TBranch* b_trk_matchtp_pdgid;

  TBranch* b_tp_pt;
  TBranch* b_tp_eta;
  TBranch* b_tp_phi;
  TBranch* b_tp_z0;
  TBranch* b_tp_d0;
  TBranch* b_tp_lxy;
  TBranch* b_tp_lz;
  TBranch* b_tp_nmatch;
  TBranch* b_tp_eventid;
  TBranch* b_tp_pdgid;

  TBranch* b_matchtrk_nstub;
  TBranch* b_matchtrk_chi2;
  TBranch* b_matchtrk_chi2_dof;
  TBranch* b_matchtrk_MVA1;

  // Initialize track pointers
  trk_pt = 0;
  trk_eta = 0;
  trk_phi = 0;
  trk_chi2 = 0;
  trk_chi2rphi = 0;
  trk_chi2rz = 0;
  trk_nstub = 0;
  trk_lhits = 0;
  trk_dhits = 0;
  trk_phiSector = 0;
  trk_seed = 0;
  trk_hitpattern = 0;
  trk_matchtp_eventtype = 0;
  trk_genuine = 0;
  trk_loose = 0;
  trk_MVA1 = 0;
  trk_matchtp_pdgid = 0;

  // Initialize TP pointers
  tp_pt = 0;
  tp_eta = 0;
  tp_phi = 0;
  tp_z0 = 0;
  tp_d0 = 0;
  tp_lxy = 0;
  tp_lz = 0;
  tp_nmatch = 0;
  tp_eventid = 0;
  tp_pdgid = 0;

  // Initialize matched track pointers
  matchtrk_nstub = 0;
  matchtrk_chi2 = 0;
  matchtrk_chi2_dof = 0;
  matchtrk_MVA1 = 0;

  // Set track branch addresses
  tree->SetBranchAddress("trk_pt", &trk_pt, &b_trk_pt);
  tree->SetBranchAddress("trk_eta", &trk_eta, &b_trk_eta);
  tree->SetBranchAddress("trk_phi", &trk_phi, &b_trk_phi);
  tree->SetBranchAddress("trk_chi2", &trk_chi2, &b_trk_chi2);
  tree->SetBranchAddress("trk_chi2rphi", &trk_chi2rphi, &b_trk_chi2rphi);
  tree->SetBranchAddress("trk_chi2rz", &trk_chi2rz, &b_trk_chi2rz);
  tree->SetBranchAddress("trk_nstub", &trk_nstub, &b_trk_nstub);
  tree->SetBranchAddress("trk_lhits", &trk_lhits, &b_trk_lhits);
  tree->SetBranchAddress("trk_dhits", &trk_dhits, &b_trk_dhits);
  tree->SetBranchAddress("trk_phiSector", &trk_phiSector, &b_trk_phiSector);
  tree->SetBranchAddress("trk_seed", &trk_seed, &b_trk_seed);
  tree->SetBranchAddress("trk_hitpattern", &trk_hitpattern, &b_trk_hitpattern);
  tree->SetBranchAddress("trk_genuine", &trk_genuine, &b_trk_genuine);
  tree->SetBranchAddress("trk_loose", &trk_loose, &b_trk_loose);
  tree->SetBranchAddress("trk_MVA1", &trk_MVA1, &b_trk_MVA1);
  tree->SetBranchAddress("trk_matchtp_eventtype", &trk_matchtp_eventtype, &b_trk_matchtp_eventtype);
  tree->SetBranchAddress("trk_matchtp_pdgid", &trk_matchtp_pdgid, &b_trk_matchtp_pdgid);

  // Set TP branch addresses
  tree->SetBranchAddress("tp_pt", &tp_pt, &b_tp_pt);
  tree->SetBranchAddress("tp_eta", &tp_eta, &b_tp_eta);
  tree->SetBranchAddress("tp_phi", &tp_phi, &b_tp_phi);
  tree->SetBranchAddress("tp_z0", &tp_z0, &b_tp_z0);
  tree->SetBranchAddress("tp_d0", &tp_d0, &b_tp_d0);
  tree->SetBranchAddress("tp_lxy", &tp_lxy, &b_tp_lxy);
  tree->SetBranchAddress("tp_lz", &tp_lz, &b_tp_lz);
  tree->SetBranchAddress("tp_nmatch", &tp_nmatch, &b_tp_nmatch);
  tree->SetBranchAddress("tp_eventid", &tp_eventid, &b_tp_eventid);
  tree->SetBranchAddress("tp_pdgid", &tp_pdgid, &b_tp_pdgid);

  // Set matched track branch addresses
  tree->SetBranchAddress("matchtrk_nstub", &matchtrk_nstub, &b_matchtrk_nstub);
  tree->SetBranchAddress("matchtrk_chi2", &matchtrk_chi2, &b_matchtrk_chi2);
  tree->SetBranchAddress("matchtrk_chi2_dof", &matchtrk_chi2_dof, &b_matchtrk_chi2_dof);
  tree->SetBranchAddress("matchtrk_MVA1", &matchtrk_MVA1, &b_matchtrk_MVA1);

  // ----------------------------------------------------------------------------------------------------------------
  // histograms
  // ----------------------------------------------------------------------------------------------------------------

  TH1F* h_trk_MVA1 = new TH1F("trk_MVA1", "; MVA1; L1 tracks", 50, 0, 1);

  TH1F* h_trk_MVA1_real = new TH1F("trk_MVA1_real", ";MVA1; L1 tracks", 50, 0, 1);
  h_trk_MVA1_real->SetLineColor(3);
  TH1F* h_trk_MVA1_fake = new TH1F("trk_MVA1_fake", ";MVA1; L1 tracks", 50, 0, 1);
  h_trk_MVA1_fake->SetLineColor(4);

  // ----------------------------------------------------------------------------------------------------------------
  //        * * * * *     S T A R T   O F   A C T U A L   R U N N I N G   O N   E V E N T S     * * * * *
  // ----------------------------------------------------------------------------------------------------------------

  int nevt = tree->GetEntries();
  cout << "number of events = " << nevt << endl;
  cout << "TP selection cuts:" << endl;
  cout << "  TP_minPt = " << TP_minPt << endl;
  cout << "  TP_maxEta = " << TP_maxEta << endl;
  cout << "  TP_maxLxy = " << TP_maxLxy << endl;
  cout << "  TP_maxLz = " << TP_maxLz << endl;
  cout << "  TP_maxD0 = " << TP_maxD0 << endl;
  cout << "  L1Tk_minNstub = " << L1Tk_minNstub << endl;
  cout << endl;

  // ----------------------------------------------------------------------------------------------------------------
  // event loop
  vector<float> MVA1s;
  vector<int> genuines;
  vector<float> etas;
  vector<float> pts;
  vector<int> pdgids;

  // For Efficiency vs Purity (TP-based)
  vector<float> all_tp_pt, all_tp_eta, all_tp_phi, all_tp_z0, all_tp_d0, all_tp_lxy, all_tp_lz;
  vector<int> all_tp_nmatch, all_tp_eventid, all_tp_pdgid;
  vector<float> all_matchtrk_nstub, all_matchtrk_chi2, all_matchtrk_chi2_dof, all_matchtrk_MVA1;

  // Track data for purity calculation
  vector<float> trk_MVA1s;
  vector<int> trk_eventtypes;

  // Event loop - collect all data
  for (int i = 0; i < nevt; i++) {
    tree->GetEntry(i, 0);

    for (int it = 0; it < (int)trk_pt->size(); it++) {
      // ----------------------------------------------------------------------------------------------------------------
      // track properties

      float MVA1 = trk_MVA1->at(it);
      int genuine = trk_genuine->at(it);  // Track matches truth particle
      int eventtype = trk_matchtp_eventtype->at(it);
      float eta = trk_eta->at(it);
      float pt = trk_pt->at(it);
      float pdgid = trk_matchtp_pdgid->at(it);

      MVA1s.push_back(MVA1);
      genuines.push_back(genuine);
      etas.push_back(eta);
      pts.push_back(pt);
      pdgids.push_back(pdgid);

      h_trk_MVA1->Fill(MVA1);
      if (genuine) {
        if (eventtype == 1)
          h_trk_MVA1_real->Fill(MVA1);  // Genuine track in signal pp
      } else {                          // Fake track
        h_trk_MVA1_fake->Fill(MVA1);
      }

      // Store for purity calculation
      trk_MVA1s.push_back(MVA1);
      trk_eventtypes.push_back(eventtype);
    }

    // ===== EFFICIENCY VS PURITY: Store TP data =====
    for (int it = 0; it < (int)tp_pt->size(); it++) {
      all_tp_pt.push_back(tp_pt->at(it));
      all_tp_eta.push_back(tp_eta->at(it));
      all_tp_phi.push_back(tp_phi->at(it));
      all_tp_z0.push_back(tp_z0->at(it));
      all_tp_d0.push_back(tp_d0->at(it));
      all_tp_lxy.push_back(tp_lxy->at(it));
      all_tp_lz.push_back(tp_lz->at(it));
      all_tp_nmatch.push_back(tp_nmatch->at(it));
      all_tp_eventid.push_back(tp_eventid->at(it));
      all_tp_pdgid.push_back(tp_pdgid->at(it));

      // Store matched track info (if available)
      if (tp_nmatch->at(it) > 0 && it < (int)matchtrk_nstub->size()) {
        all_matchtrk_nstub.push_back(matchtrk_nstub->at(it));
        all_matchtrk_chi2.push_back(matchtrk_chi2->at(it));
        all_matchtrk_chi2_dof.push_back(matchtrk_chi2_dof->at(it));
        all_matchtrk_MVA1.push_back(matchtrk_MVA1->at(it));
      } else {
        all_matchtrk_nstub.push_back(0);
        all_matchtrk_chi2.push_back(999999);
        all_matchtrk_chi2_dof.push_back(999999);
        all_matchtrk_MVA1.push_back(-1.0);
      }
    }
  }

  // ========================================================================
  // PART 1: ORIGINAL PLOTS (ROC curves, TPR vs eta, TPR vs pt, etc.)
  // ========================================================================

  // -------------------------------------------------------------------------------------------
  // create ROC curve
  // ROC = Receiver Operating Characteristic Curve, a plot of True Positive Rate vs False Positive Rate
  // TPR = True Positive Rate or Identification efficiency, fraction of real tracks correctly identified as real
  // FPR = False Positive Rate or Fake Rate, fraction of fake tracks incorrectly identified as real
  // dt = Decision Threshold or cut on the MVA output, below this identify track as fake, above identify as real
  // -------------------------------------------------------------------------------------------

  vector<float> TPR, TPR_mu, TPR_el, TPR_had;
  vector<float> FPR;
  vector<float> dec_thresh;
  int n = 100;  //num of entries on ROC curve
  for (int i = 0; i < n; i++) {
    float dt = (float)i / (n - 1);                   //make sure it starts at (0,0) and ends at (1,1)
    float TP = 0, TP_mu = 0, TP_el = 0, TP_had = 0;  //True Positives
    float FP = 0;                                    //False Positives
    float P = 0, P_mu = 0, P_el = 0, P_had = 0;      //Total Positives
    float N = 0;                                     //Total Negatives
    for (int k = 0; k < (int)MVA1s.size(); k++) {
      if (genuines.at(k)) {
        P++;
        if (MVA1s.at(k) > dt)
          TP++;
        if (abs(pdgids.at(k)) == 13) {  //muons
          P_mu++;
          if (MVA1s.at(k) > dt)
            TP_mu++;
        } else if (abs(pdgids.at(k)) == 11) {  //electrons
          P_el++;
          if (MVA1s.at(k) > dt)
            TP_el++;
        } else if (abs(pdgids.at(k)) > 37 && abs(pdgids.at(k)) != 999) {  //hadrons
          P_had++;
          if (MVA1s.at(k) > dt)
            TP_had++;
        }
      } else {
        N++;
        if (MVA1s.at(k) > dt)
          FP++;
      }
    }
    TPR.push_back((float)TP / P);
    TPR_mu.push_back((float)TP_mu / P_mu);
    TPR_el.push_back((float)TP_el / P_el);
    TPR_had.push_back((float)TP_had / P_had);
    FPR.push_back((float)FP / N);
    dec_thresh.push_back(dt);
  }

  // calculate AUC (Area under the ROC curve)
  float AUC = 0., AUC_mu = 0., AUC_el = 0., AUC_had = 0.;
  for (int i = 0; i < n - 1; i++) {
    AUC += (TPR[i] + TPR[i + 1]) / 2 * (FPR[i] - FPR[i + 1]);
    AUC_mu += (TPR_mu[i] + TPR_mu[i + 1]) / 2 * (FPR[i] - FPR[i + 1]);
    AUC_el += (TPR_el[i] + TPR_el[i + 1]) / 2 * (FPR[i] - FPR[i + 1]);
    AUC_had += (TPR_had[i] + TPR_had[i + 1]) / 2 * (FPR[i] - FPR[i + 1]);
  }

  TGraph* ROC = new TGraph(n, FPR.data(), TPR.data());
  ROC->SetName("ROC");
  ROC->SetTitle(("ROC curve (AUC = " + to_string(AUC) + "); FPR; TPR").c_str());
  ROC->SetLineWidth(4);

  TGraph* ROC_mu = new TGraph(n, FPR.data(), TPR_mu.data());
  ROC_mu->SetName("ROC_mu");
  ROC_mu->SetTitle(("ROC curve (muons, AUC = " + to_string(AUC_mu) + "); FPR; TPR").c_str());
  ROC_mu->SetLineWidth(4);

  TGraph* ROC_el = new TGraph(n, FPR.data(), TPR_el.data());
  ROC_el->SetName("ROC_el");
  ROC_el->SetTitle(("ROC curve (electrons, AUC = " + to_string(AUC_el) + "); FPR; TPR").c_str());
  ROC_el->SetLineWidth(4);

  TGraph* ROC_had = new TGraph(n, FPR.data(), TPR_had.data());
  ROC_had->SetName("ROC_had");
  ROC_had->SetTitle(("ROC curve (hadrons, AUC = " + to_string(AUC_had) + "); FPR; TPR").c_str());
  ROC_had->SetLineWidth(4);

  TGraph* TPR_vs_dt = new TGraph(n, dec_thresh.data(), TPR.data());
  TPR_vs_dt->SetName("TPR_vs_dt");
  TPR_vs_dt->SetTitle("TPR vs decision threshold; decision thresh.; TPR");
  TPR_vs_dt->SetLineColor(3);
  TPR_vs_dt->SetLineWidth(4);

  TGraph* FPR_vs_dt = new TGraph(n, dec_thresh.data(), FPR.data());
  FPR_vs_dt->SetName("FPR_vs_dt");
  FPR_vs_dt->SetTitle("FPR vs decision threshold; decision thresh.; FPR");
  FPR_vs_dt->SetLineColor(4);
  FPR_vs_dt->SetLineWidth(4);

  TGraph* TPR_vs_dt_mu = new TGraph(n, dec_thresh.data(), TPR_mu.data());
  TPR_vs_dt_mu->SetName("TPR_vs_dt_mu");
  TPR_vs_dt_mu->SetTitle("TPR vs decision threshold (muons); decision thresh.; TPR");
  TPR_vs_dt_mu->SetLineColor(3);
  TPR_vs_dt_mu->SetLineWidth(4);

  TGraph* TPR_vs_dt_el = new TGraph(n, dec_thresh.data(), TPR_el.data());
  TPR_vs_dt_el->SetName("TPR_vs_dt_el");
  TPR_vs_dt_el->SetTitle("TPR vs decision threshold (electrons); decision thresh.; TPR");
  TPR_vs_dt_el->SetLineColor(3);
  TPR_vs_dt_el->SetLineWidth(4);

  TGraph* TPR_vs_dt_had = new TGraph(n, dec_thresh.data(), TPR_had.data());
  TPR_vs_dt_had->SetName("TPR_vs_dt_had");
  TPR_vs_dt_had->SetTitle("TPR vs decision threshold (hadrons); decision thresh.; TPR");
  TPR_vs_dt_had->SetLineColor(3);
  TPR_vs_dt_had->SetLineWidth(4);

  // -------------------------------------------------------------------------------------------
  // create TPR vs. eta and FPR vs. eta
  // -------------------------------------------------------------------------------------------

  vector<float> TPR_eta, TPR_eta_mu, TPR_eta_el, TPR_eta_had;
  vector<float> TPR_eta_err, TPR_eta_err_mu, TPR_eta_err_el, TPR_eta_err_had;
  vector<float> FPR_eta, FPR_eta_err;
  vector<float> eta_range, eta_range_err;
  n = 20;
  float eta_low = -2.4;
  float eta_high = 2.4;
  float eta_temp = eta_low;
  float eta_step = (eta_high - eta_low) / n;
  float dt = .5;
  for (int ct = 0; ct < n; ct++) {
    float TP = 0, TP_mu = 0, TP_el = 0, TP_had = 0;
    float FP = 0;
    float P = 0, P_mu = 0, P_el = 0, P_had = 0;
    float N = 0;
    for (int k = 0; k < (int)etas.size(); k++) {
      if (etas.at(k) > eta_temp && etas.at(k) <= (eta_temp + eta_step)) {
        if (genuines.at(k)) {
          P++;
          if (MVA1s.at(k) > dt)
            TP++;
          if (abs(pdgids.at(k)) == 13) {  //muons
            P_mu++;
            if (MVA1s.at(k) > dt)
              TP_mu++;
          } else if (abs(pdgids.at(k)) == 11) {  //electrons
            P_el++;
            if (MVA1s.at(k) > dt)
              TP_el++;
          } else if (abs(pdgids.at(k)) > 37 && abs(pdgids.at(k)) != 999) {  //hadrons
            P_had++;
            if (MVA1s.at(k) > dt)
              TP_had++;
          }
        } else {
          N++;
          if (MVA1s.at(k) > dt)
            FP++;
        }
      }
    }

    //use min function to return 0 if no data filled
    TPR_eta.push_back(min(TP / P, P));
    TPR_eta_mu.push_back(min(TP_mu / P_mu, P_mu));
    TPR_eta_el.push_back(min(TP_el / P_el, P_el));
    TPR_eta_had.push_back(min(TP_had / P_had, P_had));
    TPR_eta_err.push_back(min((float)sqrt(TP * (P - TP) / pow(P, 3)), P));
    TPR_eta_err_mu.push_back(min((float)sqrt(TP_mu * (P_mu - TP_mu) / pow(P_mu, 3)), P_mu));
    TPR_eta_err_el.push_back(min((float)sqrt(TP_mu * (P_el - TP_el) / pow(P_el, 3)), P_el));
    TPR_eta_err_had.push_back(min((float)sqrt(TP_had * (P_had - TP_had) / pow(P_had, 3)), P_had));

    FPR_eta.push_back(min(FP / N, N));
    FPR_eta_err.push_back(min((float)sqrt(FP * (N - FP) / pow(N, 3)), N));

    //fill eta range
    eta_range.push_back(eta_temp + eta_step / 2);
    eta_range_err.push_back(eta_step / 2);

    eta_temp += eta_step;
  }

  TGraphErrors* TPR_vs_eta =
      new TGraphErrors(n, eta_range.data(), TPR_eta.data(), eta_range_err.data(), TPR_eta_err.data());
  TPR_vs_eta->SetName("TPR_vs_eta");
  TPR_vs_eta->SetTitle("TPR vs. #eta; #eta; TPR");

  TGraphErrors* FPR_vs_eta =
      new TGraphErrors(n, eta_range.data(), FPR_eta.data(), eta_range_err.data(), FPR_eta_err.data());
  FPR_vs_eta->SetName("FPR_vs_eta");
  FPR_vs_eta->SetTitle("FPR vs. #eta; #eta; FPR");

  TGraphErrors* TPR_vs_eta_mu =
      new TGraphErrors(n, eta_range.data(), TPR_eta_mu.data(), eta_range_err.data(), TPR_eta_err_mu.data());
  TPR_vs_eta_mu->SetName("TPR_vs_eta_mu");
  TPR_vs_eta_mu->SetTitle("TPR vs. #eta (muons); #eta; TPR");

  TGraphErrors* TPR_vs_eta_el =
      new TGraphErrors(n, eta_range.data(), TPR_eta_el.data(), eta_range_err.data(), TPR_eta_err_el.data());
  TPR_vs_eta_el->SetName("TPR_vs_eta_el");
  TPR_vs_eta_el->SetTitle("TPR vs. #eta (electrons); #eta; TPR");

  TGraphErrors* TPR_vs_eta_had =
      new TGraphErrors(n, eta_range.data(), TPR_eta_had.data(), eta_range_err.data(), TPR_eta_err_had.data());
  TPR_vs_eta_had->SetName("TPR_vs_eta_had");
  TPR_vs_eta_had->SetTitle("TPR vs. #eta (hadrons); #eta; TPR");

  // -------------------------------------------------------------------------------------------
  // create TPR vs. pt and FPR vs. pt
  // -------------------------------------------------------------------------------------------

  vector<float> TPR_pt, TPR_pt_mu, TPR_pt_el, TPR_pt_had;
  vector<float> TPR_pt_err, TPR_pt_err_mu, TPR_pt_err_el, TPR_pt_err_had;
  vector<float> FPR_pt, FPR_pt_err;
  vector<float> pt_range, pt_range_err;
  n = 10;
  float logpt_low = log10(2);     //set low pt in log
  float logpt_high = log10(100);  //set high pt in log
  float logpt_temp = logpt_low;
  float logpt_step = (logpt_high - logpt_low) / n;
  dt = .5;
  for (int ct = 0; ct < n; ct++) {
    float TP = 0, TP_mu = 0, TP_el = 0, TP_had = 0;
    float FP = 0;
    float P = 0, P_mu = 0, P_el = 0, P_had = 0;
    float N = 0;
    for (int k = 0; k < (int)pts.size(); k++) {
      if (pts.at(k) > pow(10, logpt_temp) && pts.at(k) <= (pow(10, logpt_temp + logpt_step))) {
        if (genuines.at(k)) {
          P++;
          if (MVA1s.at(k) > dt)
            TP++;
          if (abs(pdgids.at(k)) == 13) {  //muons
            P_mu++;
            if (MVA1s.at(k) > dt)
              TP_mu++;
          } else if (abs(pdgids.at(k)) == 11) {  //electrons
            P_el++;
            if (MVA1s.at(k) > dt)
              TP_el++;
          } else if (abs(pdgids.at(k)) > 37 && abs(pdgids.at(k)) != 999) {  //hadrons
            P_had++;
            if (MVA1s.at(k) > dt)
              TP_had++;
          }
        } else {
          N++;
          if (MVA1s.at(k) > dt)
            FP++;
        }
      }
    }

    //use min function to return 0 if no data filled
    TPR_pt.push_back(min(TP / P, P));
    TPR_pt_mu.push_back(min(TP_mu / P_mu, P_mu));
    TPR_pt_el.push_back(min(TP_el / P_el, P_el));
    TPR_pt_had.push_back(min(TP_had / P_had, P_had));
    TPR_pt_err.push_back(min((float)sqrt(TP * (P - TP) / pow(P, 3)), P));
    TPR_pt_err_mu.push_back(min((float)sqrt(TP_mu * (P_mu - TP_mu) / pow(P_mu, 3)), P_mu));
    TPR_pt_err_el.push_back(min((float)sqrt(TP_el * (P_el - TP_el) / pow(P_el, 3)), P_el));
    TPR_pt_err_had.push_back(min((float)sqrt(TP_had * (P_had - TP_had) / pow(P_had, 3)), P_had));

    FPR_pt.push_back(min(FP / N, N));
    FPR_pt_err.push_back(min((float)sqrt(FP * (N - FP) / pow(N, 3)), N));

    //fill pt range
    pt_range.push_back((pow(10, logpt_temp) + pow(10, logpt_temp + logpt_step)) / 2);  //halfway in bin
    pt_range_err.push_back((pow(10, logpt_temp + logpt_step) - pow(10, logpt_temp)) / 2);

    logpt_temp += logpt_step;
  }

  TGraphErrors* TPR_vs_pt = new TGraphErrors(n, pt_range.data(), TPR_pt.data(), pt_range_err.data(), TPR_pt_err.data());
  TPR_vs_pt->SetName("TPR_vs_pt");
  TPR_vs_pt->SetTitle("TPR vs. p_{T}; p_{T}; TPR");

  TGraphErrors* FPR_vs_pt = new TGraphErrors(n, pt_range.data(), FPR_pt.data(), pt_range_err.data(), FPR_pt_err.data());
  FPR_vs_pt->SetName("FPR_vs_pt");
  FPR_vs_pt->SetTitle("FPR vs. p_{T}; p_{T}; FPR");

  TGraphErrors* TPR_vs_pt_mu =
      new TGraphErrors(n, pt_range.data(), TPR_pt_mu.data(), pt_range_err.data(), TPR_pt_err_mu.data());
  TPR_vs_pt_mu->SetName("TPR_vs_pt_mu");
  TPR_vs_pt_mu->SetTitle("TPR vs. p_{T} (muons); p_{T}; TPR");

  TGraphErrors* TPR_vs_pt_el =
      new TGraphErrors(n, pt_range.data(), TPR_pt_el.data(), pt_range_err.data(), TPR_pt_err_el.data());
  TPR_vs_pt_el->SetName("TPR_vs_pt_el");
  TPR_vs_pt_el->SetTitle("TPR vs. p_{T} (electrons); p_{T}; TPR");

  TGraphErrors* TPR_vs_pt_had =
      new TGraphErrors(n, pt_range.data(), TPR_pt_had.data(), pt_range_err.data(), TPR_pt_err_had.data());
  TPR_vs_pt_had->SetName("TPR_vs_pt_had");
  TPR_vs_pt_had->SetTitle("TPR vs. p_{T} (hadrons); p_{T}; TPR");

  // ========================================================================
  // EFFICIENCY VS PURITY
  // ========================================================================

  vector<float> efficiency, purity;
  int n_points = 100;

  float base_efficiency = 0.0;
  float base_purity = 0.0;

  cout << "\nCalculating Efficiency vs Purity for " << n_points << " MVA cuts..." << endl;

  for (int i = 0; i < n_points; i++) {
    float mva_cut = (float)i / (n_points - 1);

    // Calculate EFFICIENCY
    int tp_passing_cuts = 0;
    int tp_matched = 0;
    int n_total_tps = all_tp_pt.size();

    for (int it = 0; it < n_total_tps; it++) {
      bool passes_cuts =
          (all_tp_pt[it] >= 0.2 && fabs(all_tp_eta[it]) <= TP_maxEta && fabs(all_tp_lxy[it]) <= TP_maxLxy &&
           fabs(all_tp_lz[it]) <= TP_maxLz && fabs(all_tp_d0[it]) <= TP_maxD0 &&
           all_tp_eventid[it] == TP_select_eventid && all_tp_pt[it] >= TP_minPt);

      if (TP_select_pdgid != 0) {
        passes_cuts = passes_cuts && (abs(all_tp_pdgid[it]) == abs(TP_select_pdgid));
      }

      if (passes_cuts) {
        tp_passing_cuts++;

        bool is_matched = (all_tp_nmatch[it] > 0 && all_matchtrk_nstub[it] >= L1Tk_minNstub &&
                           all_matchtrk_chi2[it] <= L1Tk_maxChi2 && all_matchtrk_chi2_dof[it] <= L1Tk_maxChi2dof &&
                           all_matchtrk_MVA1[it] >= mva_cut);

        if (is_matched) {
          tp_matched++;
        }
      }
    }

    float eff = (tp_passing_cuts > 0) ? (float)tp_matched / tp_passing_cuts : 0.0;

    // Calculate PURITY
    int n_total_tracks = trk_MVA1s.size();
    int n_fakes = 0;
    int n_good = 0;

    for (int it = 0; it < n_total_tracks; it++) {
      if (mva_cut > 0) {
        if (trk_MVA1s[it] >= mva_cut) {
          if (trk_eventtypes[it] == -999) {
            n_fakes++;
          } else {
            n_good++;
          }
        }
      } else {
        if (trk_eventtypes[it] == -999) {
          n_fakes++;
        } else {
          n_good++;
        }
      }
    }

    int n_total_selected = n_fakes + n_good;
    float pur = (n_total_selected > 0) ? (float)n_good / n_total_selected : 0.0;

    efficiency.push_back(eff);
    purity.push_back(pur);

    if (i == 0) {
      base_efficiency = eff;
      base_purity = pur;
      cout << "At MVA = 0: Efficiency = " << eff << ", Purity = " << pur << endl;
      cout << "TPs passing cuts = " << tp_passing_cuts << ", TPs matched = " << tp_matched << endl;
    }
  }

  TGraph* graph_eff_vs_purity = new TGraph(n_points, purity.data(), efficiency.data());
  graph_eff_vs_purity->SetName("EffVsPurity");
  graph_eff_vs_purity->SetTitle("Efficiency vs Purity;L1 Track Purity;L1 Track Finding Efficiency");
  graph_eff_vs_purity->SetLineColor(kBlue);
  graph_eff_vs_purity->SetLineWidth(5);
  graph_eff_vs_purity->SetMarkerStyle(21);
  graph_eff_vs_purity->SetMarkerSize(1.2);
  graph_eff_vs_purity->SetMarkerColor(kBlue);

  TGraph* fill_graph = new TGraph(n_points);
  for (int i = 0; i < n_points; i++) {
    fill_graph->SetPoint(i, purity[i], efficiency[i]);
  }

  TLine* base_line = new TLine(0.75, base_efficiency, 1.10, base_efficiency);
  base_line->SetLineStyle(kDashed);
  base_line->SetLineColor(kBlack);
  base_line->SetLineWidth(2);

  // ========================================================================
  // PART 3: OUTPUT AND SAVE ALL PLOTS
  // ========================================================================

  TFile* fout = new TFile(type_dir + "MVAoutput_" + type + treeName + ".root", "recreate");
  TCanvas c;

  // -------------------------------------------------------------------------------------------
  // draw and save plots
  // -------------------------------------------------------------------------------------------

  h_trk_MVA1->Draw();
  h_trk_MVA1->Write();
  c.SaveAs("MVA_plots/trk_MVA.pdf");

  h_trk_MVA1_real->Draw();
  h_trk_MVA1_fake->Draw("same");
  h_trk_MVA1_fake->SetTitle("Performance vs. decision threshold; decision thresh.; performance measure");
  TLegend* leg1 = new TLegend();
  leg1->AddEntry(h_trk_MVA1_real, "real", "l");
  leg1->AddEntry(h_trk_MVA1_fake, "fake", "l");
  leg1->Draw("same");
  c.Write("trk_MVA_rf");
  c.SaveAs("MVA_plots/trk_MVA_rf.pdf");

  ROC->Draw("AL");
  ROC->Write();
  c.SaveAs("MVA_plots/ROC.pdf");

  ROC_mu->Draw("AL");
  ROC_mu->Write();
  c.SaveAs("MVA_plots/ROC_mu.pdf");

  ROC_el->Draw("AL");
  ROC_el->Write();
  c.SaveAs("MVA_plots/ROC_el.pdf");

  ROC_had->Draw("AL");
  ROC_had->Write();
  c.SaveAs("MVA_plots/ROC_had.pdf");
  c.Clear();

  TPR_vs_dt->Draw();
  FPR_vs_dt->Draw("same");
  TPR_vs_dt->SetTitle("Performance vs. decision threshold; decision thresh.; performance measure");
  TLegend* leg2 = new TLegend();
  leg2->AddEntry(TPR_vs_dt, "TPR", "l");
  leg2->AddEntry(FPR_vs_dt, "FPR", "l");
  leg2->Draw("same");
  c.Write("TPR_FPR_vs_dt");
  c.SaveAs("MVA_plots/TPR_FPR_vs_dt.pdf");
  c.Clear();

  TPR_vs_dt_mu->Draw();
  FPR_vs_dt->Draw("same");
  TPR_vs_dt_mu->SetTitle("Performance vs. decision threshold (muons); decision thresh.; performance measure");
  TLegend* leg3 = new TLegend();
  leg3->AddEntry(TPR_vs_dt_mu, "TPR", "l");
  leg3->AddEntry(FPR_vs_dt, "FPR", "l");
  leg3->Draw("same");
  c.Write("TPR_FPR_vs_dt_mu");
  c.SaveAs("MVA_plots/TPR_FPR_vs_dt_mu.pdf");
  c.Clear();

  TPR_vs_dt_el->Draw();
  FPR_vs_dt->Draw("same");
  TPR_vs_dt_el->SetTitle("Performance vs. decision threshold (electrons); decision thresh.; performance measure");
  TLegend* leg4 = new TLegend();
  leg4->AddEntry(TPR_vs_dt_el, "TPR", "l");
  leg4->AddEntry(FPR_vs_dt, "FPR", "l");
  leg4->Draw("same");
  c.Write("TPR_FPR_vs_dt_el");
  c.SaveAs("MVA_plots/TPR_FPR_vs_dt_el.pdf");
  c.Clear();

  TPR_vs_dt_had->Draw();
  FPR_vs_dt->Draw("same");
  TPR_vs_dt_had->SetTitle("Performance vs. decision threshold (hadrons); decision thresh.; performance measure");
  TLegend* leg5 = new TLegend();
  leg5->AddEntry(TPR_vs_dt_had, "TPR", "l");
  leg5->AddEntry(FPR_vs_dt, "FPR", "l");
  leg5->Draw("same");
  c.Write("TPR_FPR_vs_dt_had");
  c.SaveAs("MVA_plots/TPR_FPR_vs_dt_had.pdf");
  c.Clear();

  TPR_vs_eta->Draw("ap");
  TPR_vs_eta->Write();
  c.SaveAs("MVA_plots/TPR_vs_eta.pdf");

  TPR_vs_eta_mu->Draw("ap");
  TPR_vs_eta_mu->Write();
  c.SaveAs("MVA_plots/TPR_vs_eta_mu.pdf");

  TPR_vs_eta_el->Draw("ap");
  TPR_vs_eta_el->Write();
  c.SaveAs("MVA_plots/TPR_vs_eta_el.pdf");

  TPR_vs_eta_had->Draw("ap");
  TPR_vs_eta_had->Write();
  c.SaveAs("MVA_plots/TPR_vs_eta_had.pdf");

  FPR_vs_eta->Draw("ap");
  FPR_vs_eta->Write();
  c.SaveAs("MVA_plots/FPR_vs_eta.pdf");

  TPR_vs_pt->Draw("ap");
  TPR_vs_pt->Write();
  c.SaveAs("MVA_plots/TPR_vs_pt.pdf");

  TPR_vs_pt_mu->Draw("ap");
  TPR_vs_pt_mu->Write();
  c.SaveAs("MVA_plots/TPR_vs_pt_mu.pdf");

  TPR_vs_pt_el->Draw("ap");
  TPR_vs_pt_el->Write();
  c.SaveAs("MVA_plots/TPR_vs_pt_el.pdf");

  TPR_vs_pt_had->Draw("ap");
  TPR_vs_pt_had->Write();
  c.SaveAs("MVA_plots/TPR_vs_pt_had.pdf");

  FPR_vs_pt->Draw("ap");
  FPR_vs_pt->Write();
  c.SaveAs("MVA_plots/FPR_vs_pt.pdf");

  // ---- EFFICIENCY VS PURITY PLOT ----

  c.Clear();
  c.SetGrid();

  graph_eff_vs_purity->Draw("APL");
  graph_eff_vs_purity->GetXaxis()->SetLimits(0.75, 1.10);
  graph_eff_vs_purity->SetMinimum(0.75);
  graph_eff_vs_purity->SetMaximum(1.00);

  base_line->Draw("same");

  TLegend* leg_eff = new TLegend(0.12, 0.12, 0.60, 0.32);
  leg_eff->SetBorderSize(1);
  leg_eff->SetFillColor(kWhite);
  leg_eff->SetFillStyle(1001);
  leg_eff->SetTextSize(0.030);
  leg_eff->SetTextFont(42);
  leg_eff->AddEntry(graph_eff_vs_purity, Form("%s", type.Data()), "lp");
  leg_eff->AddEntry(base_line, Form("Base Efficiency (No MVA Cuts) = %.1f%%", base_efficiency * 100), "l");
  leg_eff->Draw("same");

  c.Write("EffVsPurity");
  c.SaveAs("MVA_plots/EffVsPurity.pdf");

  graph_eff_vs_purity->Write();
  fill_graph->Write();

  fout->Close();

  cout << "\nAll plots saved to MVA_plots/ directory" << endl;
  cout << "Efficiency vs Purity plot saved to MVA_plots/EffVsPurity.pdf" << endl;
  cout << "Base Efficiency at MVA=0: " << base_efficiency << endl;
  cout << "Base Purity at MVA=0: " << base_purity << endl;
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
  gStyle->SetPaperSize(20, 26);
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
  gStyle->SetLabelFont(42, "x");
  gStyle->SetTitleFont(42, "x");
  gStyle->SetLabelFont(42, "y");
  gStyle->SetTitleFont(42, "y");
  gStyle->SetLabelFont(42, "z");
  gStyle->SetTitleFont(42, "z");
  gStyle->SetLabelSize(0.05, "x");
  gStyle->SetTitleSize(0.05, "x");
  gStyle->SetLabelSize(0.05, "y");
  gStyle->SetTitleSize(0.05, "y");
  gStyle->SetLabelSize(0.05, "z");
  gStyle->SetTitleSize(0.05, "z");

  // use bold lines and markers
  //gStyle->SetMarkerStyle(20);
  gStyle->SetMarkerSize(1.2);
  gStyle->SetHistLineWidth(4.);
  gStyle->SetLineStyleString(4, "[12 12]");

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
