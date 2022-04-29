//To make basic turn-on curve for trkMET
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

void trkMET_turnOn() {
  TChain* tree = new TChain("L1TrackNtuple/eventTree");
  // tree->Add("/uscms_data/d3/csavard/working_dir/for_Emily/CMSSW_11_1_3/src/L1Trigger/TrackFindingTracklet/test/crab_output/crab_ttbar_pu200/results/TTbar_PU200_D49.root");
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
  float cutoff = 300.0 - 0.1;
  float TP_minPt = 2.0;
  float TP_maxEta = 2.4;
  float chi2dof_cut = 10.0;
  float bendchi2_cut = 2.2;

  TH1F* h_trueMET = new TH1F("trueMET", ";trueMET [GeV]; Events", numBins, 0, 500.0);
  TH1F* h_trueTkMET = new TH1F("trueTkMET", ";trueTkMET [GeV]; Events", numBins, 0, 500.0);
  TH1F* h_recoTkMET = new TH1F("recoTkMET", ";recoTkMET [GeV]; Events", numBins, 0, 500.0);

  TH1F* h_trueTkMET_num = new TH1F("trueTkMET_turnon", ";trueMET [GeV]; Events", numBins, 0, 500.0);
  TH1F* h_recoTkMET_num = new TH1F("recoTkMET_turnon", ";trueMET [GeV]; Events", numBins, 0, 500.0);

  //From rate code
  float trueTkMET_rate = 49.0;
  float recoTkMET_rate = 49.0;

  bool rerun_trueMET = true;
  bool rerun_trueTkMET = true;
  bool rerun_recoTkMET = true;

  int nevt = tree->GetEntries();
  cout << "number of events = " << nevt << endl;

  // event loop
  for (int i = 0; i < nevt; i++) {
    tree->GetEntry(i);
    if (i % 10000 == 0)
      std::cout << i << "/" << nevt << std::endl;

    float trueMET_ntuple = trueMET;      //grab these from ntuple
    float trueTkMET_ntuple = trueTkMET;  //grab these from ntuple
    float recoTkMET_ntuple = trkMET;     //grab these from ntuple

    if (rerun_trueMET) {
      float trueMETx = 0.0;
      float trueMETy = 0.0;
      float trueMET_calc = 0.0;
      for (size_t i = 0; i < gen_pt->size(); ++i) {
        int id = gen_pdgid->at(i);
        float pt = gen_pt->at(i);
        float phi = gen_phi->at(i);
        bool isNeutrino = false;
        if ((fabs(id) == 12 || fabs(id) == 14 || fabs(id) == 16))
          isNeutrino = true;
        if ((isNeutrino || id == 1000022)) {  //only gen parts saved are status==1
          trueMETx += pt * cos(phi);
          trueMETy += pt * sin(phi);
        }
      }
      trueMET_calc = sqrt(trueMETx * trueMETx + trueMETy * trueMETy);
      trueMET_ntuple = trueMET_calc;
    }  //re-run true MET (gen particles)

    if (rerun_trueTkMET) {
      float trueTkMET_calc = 0;
      float trueTkMETx = 0;
      float trueTkMETy = 0;
      // tracking particle loop
      for (int it = 0; it < (int)tp_pt->size(); it++) {
        float this_tp_pt = tp_pt->at(it);
        float this_tp_eta = tp_eta->at(it);
        float this_tp_phi = tp_phi->at(it);
        int this_tp_eventID = tp_eventid->at(it);

        // kinematic cuts
        if (tp_pt->at(it) < TP_minPt)
          continue;
        if (fabs(this_tp_eta) > TP_maxEta)
          continue;
        if (tp_nstub->at(it) < 4)
          continue;
        if (fabs(tp_z0->at(it)) > 15)
          continue;
        if (this_tp_eventID != 0)
          continue;  // hard interaction only
        // if (tp_dxy->at(it) > 1) continue;
        if (tp_charge->at(it) == 0)
          continue;

        trueTkMETx += this_tp_pt * cos(this_tp_phi);
        trueTkMETy += this_tp_pt * sin(this_tp_phi);
      }  // end of tracking particle loop
      trueTkMET_calc = sqrt(trueTkMETx * trueTkMETx + trueTkMETy * trueTkMETy);
      trueTkMET_ntuple = trueTkMET_calc;
    }  // re-run true trk MET

    if (rerun_recoTkMET) {  //also useful for checking performance of new track quality cuts
      float recoTkMET_calc = 0;
      float recoTkMETx = 0;
      float recoTkMETy = 0;
      // loop over all tracks
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

        // if (thisTrk_pt>200.0) thisTrk_pt = 200.0;

        if (thisTrk_chi2dof < chi2dof_cut && thisTrk_bendchi2 < bendchi2_cut) {
          recoTkMETx += thisTrk_pt * cos(thisTrk_phi);
          recoTkMETy += thisTrk_pt * sin(thisTrk_phi);
        }
      }  //end loop over tracks
      recoTkMET_calc = sqrt(recoTkMETx * recoTkMETx + recoTkMETy * recoTkMETy);
      if (recoTkMET_calc != recoTkMET_ntuple)
        std::cout << "Ugh: " << recoTkMET_calc << ", " << recoTkMET_ntuple << std::endl;

      recoTkMET_ntuple = recoTkMET_calc;
    }  //re-run reco trk MET

    h_trueMET->Fill(trueMET_ntuple);
    h_trueTkMET->Fill(trueTkMET_ntuple);
    h_recoTkMET->Fill(recoTkMET_ntuple);

    //get rate cuts from Rate Plot script
    //if reco MET passes cut, fill with trueMET
    if (trueTkMET_ntuple > 49)
      h_trueTkMET_num->Fill(trueMET_ntuple);
    if (recoTkMET_ntuple > 49)
      h_recoTkMET_num->Fill(trueMET_ntuple);
  }  // end event loop

  //Draw
  h_trueMET->SetLineColor(kBlack);
  h_trueMET->SetLineStyle(2);
  h_trueTkMET->SetLineColor(kBlack);
  h_recoTkMET->SetLineColor(kRed);

  TLegend* leg = new TLegend(0.55, 0.15, 0.85, 0.35);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.04);
  leg->SetTextFont(42);
  leg->AddEntry(h_trueTkMET, "trueTkMET", "l");
  leg->AddEntry(h_recoTkMET, "recoTkMET", "l");

  // calculate the recoTkMET trigger turn on curve
  h_trueMET->Rebin(2);
  h_trueTkMET_num->Rebin(2);
  h_recoTkMET_num->Rebin(2);
  h_trueMET->Sumw2();
  h_trueTkMET->Sumw2();
  h_trueTkMET_num->Sumw2();

  TH1F* h_TkMET_turnon = (TH1F*)h_trueTkMET_num->Clone();
  h_TkMET_turnon->SetName("trueTkMET_turnon");
  h_TkMET_turnon->SetStats(0);
  h_TkMET_turnon->SetTitle("; Gen-level MET [GeV]; Efficiency");
  h_TkMET_turnon->Divide(h_trueTkMET_num, h_trueMET, 1.0, 1.0, "B");
  h_TkMET_turnon->SetMarkerStyle(20);
  h_TkMET_turnon->SetMarkerSize(0.7);
  h_TkMET_turnon->SetMarkerColor(kBlack);
  h_TkMET_turnon->SetLineColor(kBlack);

  h_recoTkMET_num->Sumw2();
  TH1F* h_recoTkMET_turnon = (TH1F*)h_recoTkMET_num->Clone();
  h_recoTkMET_turnon->SetName("recoTkMET_turnon");
  h_recoTkMET_turnon->SetStats(0);
  h_recoTkMET_turnon->SetTitle("; Gen-level MET [GeV]; Efficiency");
  h_recoTkMET_turnon->Divide(h_recoTkMET_num, h_trueMET, 1.0, 1.0, "B");
  h_recoTkMET_turnon->SetMarkerStyle(20);
  h_recoTkMET_turnon->SetMarkerSize(0.7);
  h_recoTkMET_turnon->SetMarkerColor(kRed);
  h_recoTkMET_turnon->SetLineColor(kRed);

  TCanvas* can = new TCanvas("can", "can", 800, 600);
  can->SetGridx();
  can->SetGridy();
  h_TkMET_turnon->SetMinimum(0);
  h_TkMET_turnon->SetMaximum(1.1);
  h_TkMET_turnon->Draw();
  h_recoTkMET_turnon->Draw("same");
  leg->Draw("same");

  float trueTkMET_95eff = h_TkMET_turnon->GetBinLowEdge(h_TkMET_turnon->FindFirstBinAbove(0.95));
  float recoTkMET_95eff = h_recoTkMET_turnon->GetBinLowEdge(h_recoTkMET_turnon->FindFirstBinAbove(0.95));

  std::cout << "Online (true) Track MET at " << trueTkMET_rate << " GeV = " << trueTkMET_95eff << " GeV (offline)"
            << std::endl;
  std::cout << "Online (reco) Track MET at " << recoTkMET_rate << " GeV = " << recoTkMET_95eff << " GeV (offline)"
            << std::endl;

  can->SaveAs("trkMET_ttbarPU200_TurnOnPlot.root");
}
