//To find the rate from a (very very large) min bias sample,
//Runs off of the track object ntuplizer

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
TH1D* GetCumulative(TH1D* plot);

void trkMET_rate() {
  TChain* tree = new TChain("L1TrackNtuple/eventTree");
  // tree->Add("/uscms_data/d3/csavard/working_dir/for_Emily/CMSSW_11_1_3/src/L1Trigger/TrackFindingTracklet/test/crab_output/crab_nu_gun/results/NuGun_PU200_TDR.root");
  tree->Add("CheckingJets_CMSSW11_CMS.root");

  if (tree->GetEntries() == 0) {
    cout << "File doesn't exist or is empty, returning..." << endl;
    return;
  }

  // define leafs & branches
  float trueMET = 0;
  float trueTkMET = 0;
  float trkMET = 0;
  vector<float>* pv_L1reco;
  //gen particles
  vector<float>* gen_pt;
  vector<float>* gen_phi;
  vector<int>* gen_pdgid;

  // tracking particles
  vector<float>* tp_pt;
  vector<float>* tp_eta;
  vector<float>* tp_phi;
  vector<float>* tp_dxy;
  vector<float>* tp_z0;
  vector<float>* tp_d0;
  vector<int>* tp_pdgid;
  vector<int>* tp_nmatch;
  vector<int>* tp_nstub;
  // vector<int>*   tp_nLayers;
  vector<int>* tp_eventid;
  // vector<int>*   tp_signal;
  vector<int>* tp_charge;

  // *L1 track* properties, for tracking particles matched to a L1 track
  vector<float>* matchtrk_pt;
  vector<float>* matchtrk_eta;
  vector<float>* matchtrk_phi;
  vector<float>* matchtrk_d0;
  vector<float>* matchtrk_z0;
  vector<float>* matchtrk_chi2;
  vector<float>* matchtrk_MVA1;
  vector<float>* matchtrk_bendchi2;
  vector<int>* matchtrk_nstub;
  vector<int>* matchtrk_seed;

  // all L1 tracks
  vector<float>* trk_pt;
  vector<float>* trk_eta;
  vector<float>* trk_phi;
  vector<float>* trk_z0;
  vector<float>* trk_chi2;
  vector<float>* trk_MVA1;
  vector<float>* trk_bendchi2;
  vector<int>* trk_nstub;
  vector<int>* trk_seed;
  vector<int>* trk_fake;

  TBranch* b_gen_pt;
  TBranch* b_gen_phi;
  TBranch* b_gen_pdgid;

  TBranch* b_tp_pt;
  TBranch* b_tp_eta;
  TBranch* b_tp_phi;
  TBranch* b_tp_dxy;
  TBranch* b_tp_z0;
  TBranch* b_tp_d0;
  TBranch* b_tp_pdgid;
  TBranch* b_tp_nmatch;
  // TBranch* b_tp_nLayers;
  TBranch* b_tp_nstub;
  TBranch* b_tp_eventid;
  // TBranch* b_tp_signal;
  TBranch* b_tp_charge;

  TBranch* b_matchtrk_pt;
  TBranch* b_matchtrk_eta;
  TBranch* b_matchtrk_phi;
  TBranch* b_matchtrk_d0;
  TBranch* b_matchtrk_z0;
  TBranch* b_matchtrk_chi2;
  TBranch* b_matchtrk_MVA1;
  TBranch* b_matchtrk_bendchi2;
  TBranch* b_matchtrk_nstub;
  TBranch* b_matchtrk_seed;

  TBranch* b_trk_pt;
  TBranch* b_trk_eta;
  TBranch* b_trk_phi;
  TBranch* b_trk_z0;
  TBranch* b_trk_chi2;
  TBranch* b_trk_MVA1;
  TBranch* b_trk_bendchi2;
  TBranch* b_trk_nstub;
  TBranch* b_trk_seed;
  TBranch* b_trk_fake;

  TBranch* b_pv_L1reco;
  TBranch* b_trueMET;
  TBranch* b_trueTkMET;
  TBranch* b_trkMET;

  trueMET = 0;
  trueTkMET = 0;
  trkMET = 0;

  gen_pt = 0;
  gen_phi = 0;
  gen_pdgid = 0;

  tp_pt = 0;
  tp_eta = 0;
  tp_phi = 0;
  tp_dxy = 0;
  tp_z0 = 0;
  tp_d0 = 0;
  tp_pdgid = 0;
  tp_nmatch = 0;
  // tp_nLayers = 0;
  tp_nstub = 0;
  tp_eventid = 0;
  // tp_signal = 0;
  tp_charge = 0;

  matchtrk_pt = 0;
  matchtrk_eta = 0;
  matchtrk_phi = 0;
  matchtrk_d0 = 0;
  matchtrk_z0 = 0;
  matchtrk_chi2 = 0;
  matchtrk_MVA1 = 0;
  matchtrk_bendchi2 = 0;
  matchtrk_nstub = 0;
  matchtrk_seed = 0;

  trk_pt = 0;
  trk_eta = 0;
  trk_phi = 0;
  trk_z0 = 0;
  trk_chi2 = 0;
  trk_MVA1 = 0;
  trk_bendchi2 = 0;
  trk_nstub = 0;
  trk_seed = 0;
  trk_fake = 0;

  pv_L1reco = 0;

  tree->SetBranchAddress("pv_L1reco", &pv_L1reco, &b_pv_L1reco);
  tree->SetBranchAddress("trueMET", &trueMET, &b_trueMET);
  tree->SetBranchAddress("trueTkMET", &trueTkMET, &b_trueTkMET);
  tree->SetBranchAddress("trkMET", &trkMET, &b_trkMET);
  tree->SetBranchAddress("gen_pt", &gen_pt, &b_gen_pt);
  tree->SetBranchAddress("gen_phi", &gen_phi, &b_gen_phi);
  tree->SetBranchAddress("gen_pdgid", &gen_pdgid, &b_gen_pdgid);

  tree->SetBranchAddress("tp_pt", &tp_pt, &b_tp_pt);
  tree->SetBranchAddress("tp_eta", &tp_eta, &b_tp_eta);
  tree->SetBranchAddress("tp_phi", &tp_phi, &b_tp_phi);
  tree->SetBranchAddress("tp_dxy", &tp_dxy, &b_tp_dxy);
  tree->SetBranchAddress("tp_z0", &tp_z0, &b_tp_z0);
  tree->SetBranchAddress("tp_d0", &tp_d0, &b_tp_d0);
  tree->SetBranchAddress("tp_pdgid", &tp_pdgid, &b_tp_pdgid);
  tree->SetBranchAddress("tp_nmatch", &tp_nmatch, &b_tp_nmatch);
  // tree->SetBranchAddress("tp_nLayers", &tp_nLayers, &b_tp_nLayers);
  tree->SetBranchAddress("tp_nstub", &tp_nstub, &b_tp_nstub);
  tree->SetBranchAddress("tp_eventid", &tp_eventid, &b_tp_eventid);
  // tree->SetBranchAddress("tp_signal",    &tp_signal,    &b_tp_signal);
  tree->SetBranchAddress("tp_charge", &tp_charge, &b_tp_charge);

  tree->SetBranchAddress("matchtrk_pt", &matchtrk_pt, &b_matchtrk_pt);
  tree->SetBranchAddress("matchtrk_eta", &matchtrk_eta, &b_matchtrk_eta);
  tree->SetBranchAddress("matchtrk_phi", &matchtrk_phi, &b_matchtrk_phi);
  tree->SetBranchAddress("matchtrk_d0", &matchtrk_d0, &b_matchtrk_d0);
  tree->SetBranchAddress("matchtrk_z0", &matchtrk_z0, &b_matchtrk_z0);
  tree->SetBranchAddress("matchtrk_chi2", &matchtrk_chi2, &b_matchtrk_chi2);
  tree->SetBranchAddress("matchtrk_MVA1", &matchtrk_MVA1, &b_matchtrk_MVA1);
  tree->SetBranchAddress("matchtrk_bendchi2", &matchtrk_bendchi2, &b_matchtrk_bendchi2);
  tree->SetBranchAddress("matchtrk_nstub", &matchtrk_nstub, &b_matchtrk_nstub);

  tree->SetBranchAddress("trk_pt", &trk_pt, &b_trk_pt);
  tree->SetBranchAddress("trk_eta", &trk_eta, &b_trk_eta);
  tree->SetBranchAddress("trk_phi", &trk_phi, &b_trk_phi);
  tree->SetBranchAddress("trk_z0", &trk_z0, &b_trk_z0);
  tree->SetBranchAddress("trk_chi2", &trk_chi2, &b_trk_chi2);
  tree->SetBranchAddress("trk_MVA1", &trk_MVA1, &b_trk_MVA1);
  tree->SetBranchAddress("trk_bendchi2", &trk_bendchi2, &b_trk_bendchi2);
  tree->SetBranchAddress("trk_nstub", &trk_nstub, &b_trk_nstub);
  tree->SetBranchAddress("trk_fake", &trk_fake, &b_trk_fake);

  //Need trkMET from tracking particles, and trkMET from tracks

  float numBins = 100.0;
  float cutoff = 200.0 - 0.1;
  float TP_minPt = 2.0;
  float TP_maxEta = 2.4;
  float chi2dof_cut = 10.0;
  float bendchi2_cut = 2.2;

  TH1D* trueTkMET_thresh = new TH1D("trueTkMET_thresh",
                                    "trueTkMET_thresh;trueTkMET threshold (GeV);Occurrences",
                                    numBins,
                                    0,
                                    numBins);  //using tracking particles
  TH1D* recoTkMET_thresh = new TH1D(
      "recoTkMET_thresh", "recoTkMET_thresh;recoTkMET threshold (GeV);Occurrences", numBins, 0, numBins);  //using tracks

  bool rerun_trueTkMET = false;
  bool rerun_recoTkMET = false;

  int nevt = tree->GetEntries();
  cout << "number of events = " << nevt << endl;

  // event loop
  for (int i = 0; i < nevt; i++) {
    tree->GetEntry(i);
    if (i % 10000 == 0)
      std::cout << i << "/" << nevt << std::endl;

    float trueTkMET_ntuple = 0;  //grab these from ntuple
    float recoTkMET_ntuple = 0;  //grab these from ntuple

    if (!rerun_trueTkMET)
      trueTkMET_ntuple = trueTkMET;
    if (!rerun_recoTkMET)
      recoTkMET_ntuple = trkMET;

    if (rerun_trueTkMET) {
      float trueTkMET_calc = 0;
      float trueTkMETx = 0;
      float trueTkMETy = 0;
      // tracking particle loop
      for (int it = 0; it < (int)tp_pt->size(); it++) {
        float this_tp_pt = tp_pt->at(it);
        float this_tp_eta = tp_eta->at(it);
        float this_tp_phi = tp_phi->at(it);
        int this_tp_signal = tp_eventid->at(it);
        float deltaZ = fabs(tp_z0->at(it) - pv_L1reco->at(0));

        // kinematic cuts
        if (tp_pt->at(it) < TP_minPt)
          continue;
        if (fabs(this_tp_eta) > TP_maxEta)
          continue;
        if (tp_nstub->at(it) < 4)
          continue;
        if (fabs(tp_z0->at(it)) > 15)
          continue;
        // if (this_tp_signal!=0) continue;
        // if (tp_dxy->at(it) > 1) continue;
        if (tp_charge->at(it) == 0)
          continue;

        float deltaZ_cut = 3.0;  // cuts out PU
        if (fabs(this_tp_eta) >= 0 && fabs(this_tp_eta) < 0.7)
          deltaZ_cut = 0.4;
        else if (fabs(this_tp_eta) >= 0.7 && fabs(this_tp_eta) < 1.0)
          deltaZ_cut = 0.6;
        else if (fabs(this_tp_eta) >= 1.0 && fabs(this_tp_eta) < 1.2)
          deltaZ_cut = 0.76;
        else if (fabs(this_tp_eta) >= 1.2 && fabs(this_tp_eta) < 1.6)
          deltaZ_cut = 1.0;
        else if (fabs(this_tp_eta) >= 1.6 && fabs(this_tp_eta) < 2.0)
          deltaZ_cut = 1.7;
        else if (fabs(this_tp_eta) >= 2.0 && fabs(this_tp_eta) <= 2.4)
          deltaZ_cut = 2.20;

        if (deltaZ > deltaZ_cut)
          continue;

        trueTkMETx += this_tp_pt * cos(this_tp_phi);
        trueTkMETy += this_tp_pt * sin(this_tp_phi);
      }  // end of tracking particle loop

      trueTkMET_calc = sqrt(trueTkMETx * trueTkMETx + trueTkMETy * trueTkMETy);
      trueTkMET_ntuple = trueTkMET_calc;
    }  //re-run trueTkMET

    if (rerun_recoTkMET) {  //also useful for checking different track quality cuts
      float recoTkMET_calc = 0;
      float recoTkMETx = 0;
      float recoTkMETy = 0;

      for (int it = 0; it < (int)trk_pt->size(); it++) {
        float thisTrk_pt = trk_pt->at(it);
        float thisTrk_eta = trk_eta->at(it);
        int thisTrk_nstub = trk_nstub->at(it);
        float thisTrk_chi2dof = trk_chi2->at(it) / (2 * thisTrk_nstub - 4);
        float thisTrk_bendchi2 = trk_bendchi2->at(it);
        float thisTrk_phi = trk_phi->at(it);
        float thisTrk_MVA = trk_MVA1->at(it);
        int thisTrk_fake = trk_fake->at(it);
        float thisTrk_z0 = trk_z0->at(it);
        float deltaZ = thisTrk_z0 - pv_L1reco->at(0);

        float deltaZ_cut = 3.0;  // cuts out PU
        if (fabs(thisTrk_eta) >= 0 && fabs(thisTrk_eta) < 0.7)
          deltaZ_cut = 0.4;
        else if (fabs(thisTrk_eta) >= 0.7 && fabs(thisTrk_eta) < 1.0)
          deltaZ_cut = 0.6;
        else if (fabs(thisTrk_eta) >= 1.0 && fabs(thisTrk_eta) < 1.2)
          deltaZ_cut = 0.76;
        else if (fabs(thisTrk_eta) >= 1.2 && fabs(thisTrk_eta) < 1.6)
          deltaZ_cut = 1.0;
        else if (fabs(thisTrk_eta) >= 1.6 && fabs(thisTrk_eta) < 2.0)
          deltaZ_cut = 1.7;
        else if (fabs(thisTrk_eta) >= 2.0 && fabs(thisTrk_eta) <= 2.4)
          deltaZ_cut = 2.20;

        if (thisTrk_pt < TP_minPt || fabs(thisTrk_eta) > TP_maxEta || thisTrk_nstub < 4 || fabs(thisTrk_z0) > 15)
          continue;
        if (fabs(deltaZ) > deltaZ_cut)
          continue;

        if (thisTrk_chi2dof < chi2dof_cut && thisTrk_bendchi2 < bendchi2_cut) {
          recoTkMETx += thisTrk_pt * cos(thisTrk_phi);
          recoTkMETy += thisTrk_pt * sin(thisTrk_phi);
        }
      }  //end loop over tracks

      recoTkMET_calc = sqrt(recoTkMETx * recoTkMETx + recoTkMETy * recoTkMETy);
      recoTkMET_ntuple = recoTkMET_calc;
    }  //re-run reco tkMET

    trueTkMET_thresh->Fill(trueTkMET_ntuple);
    recoTkMET_thresh->Fill(recoTkMET_ntuple);
  }  // end event loop

  // -------------------------------------------------------------------------------------------
  trueTkMET_thresh->SetStats(0);
  recoTkMET_thresh->SetStats(0);
  trueTkMET_thresh->SetLineColor(kBlack);
  recoTkMET_thresh->SetLineColor(kRed);
  trueTkMET_thresh->SetTitle("");
  recoTkMET_thresh->SetTitle("");

  TLegend* leg = new TLegend(0.48, 0.68, 0.88, 0.88);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.04);
  leg->SetTextFont(42);
  leg->AddEntry(trueTkMET_thresh, "trueTkMET", "l");
  leg->AddEntry(recoTkMET_thresh, "recoTkMET", "l");

  TH1D* fCumulative_true = GetCumulative(trueTkMET_thresh);
  TH1D* fCumulative_reco = GetCumulative(recoTkMET_thresh);
  fCumulative_true->Scale(4.0e3 / fCumulative_true->GetBinContent(1));
  fCumulative_reco->Scale(4.0e3 / fCumulative_reco->GetBinContent(1));

  fCumulative_true->SetTitle("MinBias Events PU 200; Track MET [GeV]; L1 Rate [kHz]");
  fCumulative_true->SetMarkerSize(0.7);
  fCumulative_true->SetMarkerStyle(20);
  fCumulative_true->SetMarkerColor(fCumulative_true->GetLineColor());

  fCumulative_reco->SetMarkerSize(0.7);
  fCumulative_reco->SetMarkerStyle(20);
  fCumulative_reco->SetMarkerColor(fCumulative_reco->GetLineColor());

  // fCumulative_true->Rebin(2); fCumulative_reco->Rebin(2);

  TCanvas* can = new TCanvas("can", "can");
  can->SetGridx();
  can->SetGridy();
  can->SetLogy();

  fCumulative_true->GetYaxis()->SetRangeUser(1E-01, 1E05);

  fCumulative_true->Draw();
  fCumulative_reco->Draw("same");
  leg->Draw("same");

  TLine* line_rate = new TLine(0, 35, numBins, 35);
  line_rate->SetLineColor(kBlack);
  line_rate->SetLineWidth(2);
  line_rate->Draw("SAME");

  std::cout << "Track MET Threshold at 35kHz "
            << fCumulative_reco->GetBinLowEdge(fCumulative_reco->FindLastBinAbove(35)) << std::endl;

  can->SaveAs("trkMET_minBias_RatePlot.root");
}

TH1D* GetCumulative(TH1D* plot) {
  std::string newName = Form("cumulative_%s", plot->GetName());
  TH1D* temp = (TH1D*)plot->Clone(newName.c_str());
  temp->SetDirectory(0);
  for (int i = 0; i < plot->GetNbinsX() + 1; i++) {
    double content(0.0), error2(0.0);
    for (int j = i; j < plot->GetNbinsX() + 1; j++) {
      content += plot->GetBinContent(j);
      error2 += plot->GetBinError(j) * plot->GetBinError(j);
    }
    temp->SetBinContent(i, content);
    temp->SetBinError(i, TMath::Sqrt(error2));
  }
  return temp;
}
