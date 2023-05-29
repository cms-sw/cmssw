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

  // these are the LOOSE cuts, baseline scenario for efficiency and rate plots ==> configure as appropriate
  int L1Tk_minNstub = 4;
  float L1Tk_maxChi2 = 999999;
  float L1Tk_maxChi2dof = 999999.;

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
  vector<int>* trk_fake;
  vector<int>* trk_genuine;
  vector<int>* trk_loose;
  vector<float>* trk_MVA1;
  vector<float>* trk_matchtp_pdgid;

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
  TBranch* b_trk_fake;
  TBranch* b_trk_genuine;
  TBranch* b_trk_loose;
  TBranch* b_trk_MVA1;
  TBranch* b_trk_matchtp_pdgid;

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
  trk_fake = 0;
  trk_genuine = 0;
  trk_loose = 0;
  trk_MVA1 = 0;
  trk_matchtp_pdgid = 0;

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
  tree->SetBranchAddress("trk_fake", &trk_fake, &b_trk_fake);
  tree->SetBranchAddress("trk_genuine", &trk_genuine, &b_trk_genuine);
  tree->SetBranchAddress("trk_loose", &trk_loose, &b_trk_loose);
  tree->SetBranchAddress("trk_MVA1", &trk_MVA1, &b_trk_MVA1);
  tree->SetBranchAddress("trk_matchtp_pdgid", &trk_matchtp_pdgid, &b_trk_matchtp_pdgid);

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

  // ----------------------------------------------------------------------------------------------------------------
  // event loop
  vector<float> MVA1s;
  vector<float> fakes;
  vector<float> etas;
  vector<float> pts;
  vector<int> pdgids;
  for (int i = 0; i < nevt; i++) {
    tree->GetEntry(i, 0);

    for (int it = 0; it < (int)trk_pt->size(); it++) {
      // ----------------------------------------------------------------------------------------------------------------
      // track properties

      float MVA1 = trk_MVA1->at(it);
      float fake = trk_fake->at(it);
      float eta = trk_eta->at(it);
      float pt = trk_pt->at(it);
      float pdgid = trk_matchtp_pdgid->at(it);

      MVA1s.push_back(MVA1);
      fakes.push_back(fake);
      etas.push_back(eta);
      pts.push_back(pt);
      pdgids.push_back(pdgid);

      h_trk_MVA1->Fill(MVA1);
      if (fake == 1.)
        h_trk_MVA1_real->Fill(MVA1);
      else if (fake == 0.)
        h_trk_MVA1_fake->Fill(MVA1);
    }
  }

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
      if (fakes.at(k)) {
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
        if (fakes.at(k)) {
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
        if (fakes.at(k)) {
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

  // -------------------------------------------------------------------------------------------
  // output file for histograms and graphs
  // -------------------------------------------------------------------------------------------

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

  fout->Close();
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
