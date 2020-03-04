/*
 *  See header file for a description of this class.
 *
 *  \author Bibhuprasad Mahakud (Purdue University, West Lafayette, USA)
 */
#include "DQMOffline/Muon/interface/TriggerMatchMonitor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "TLorentzVector.h"

#include <string>
#include <TMath.h>
using namespace std;
using namespace edm;

//#define DEBUG

TriggerMatchMonitor::TriggerMatchMonitor(const edm::ParameterSet& pSet) {
  LogTrace(metname) << "[TriggerMatchMonitor] Parameters initialization";

  parameters = pSet;

  beamSpotToken_ = consumes<reco::BeamSpot>(parameters.getUntrackedParameter<edm::InputTag>("offlineBeamSpot")),
  primaryVerticesToken_ =
      consumes<std::vector<reco::Vertex>>(parameters.getUntrackedParameter<edm::InputTag>("offlinePrimaryVertices")),
  theMuonCollectionLabel_ = consumes<edm::View<reco::Muon>>(parameters.getParameter<edm::InputTag>("MuonCollection"));
  thePATMuonCollectionLabel_ =
      consumes<edm::View<pat::Muon>>(parameters.getParameter<edm::InputTag>("patMuonCollection"));
  theVertexLabel_ = consumes<reco::VertexCollection>(parameters.getParameter<edm::InputTag>("VertexLabel"));
  theBeamSpotLabel_ = mayConsume<reco::BeamSpot>(parameters.getParameter<edm::InputTag>("BeamSpotLabel"));
  triggerResultsToken_ =
      consumes<edm::TriggerResults>(parameters.getUntrackedParameter<edm::InputTag>("triggerResults"));
  triggerObjects_ =
      consumes<std::vector<pat::TriggerObjectStandAlone>>(parameters.getParameter<edm::InputTag>("triggerObjects"));

  triggerPathName1_ = parameters.getParameter<string>("triggerPathName1");
  triggerHistName1_ = parameters.getParameter<string>("triggerHistName1");
  triggerPtThresholdPath1_ = parameters.getParameter<double>("triggerPtThresholdPath1");
  triggerPathName2_ = parameters.getParameter<string>("triggerPathName2");
  triggerHistName2_ = parameters.getParameter<string>("triggerHistName2");
  triggerPtThresholdPath2_ = parameters.getParameter<double>("triggerPtThresholdPath2");
  theFolder = parameters.getParameter<string>("folder");
}
TriggerMatchMonitor::~TriggerMatchMonitor() {}

void TriggerMatchMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                         edm::Run const& /*iRun*/,
                                         edm::EventSetup const& /*iSetup*/) {
  ibooker.cd();
  ibooker.setCurrentFolder(theFolder);

  // monitoring of trigger match parameter

  matchHists.push_back(ibooker.book1D("DelR_HLT_" + triggerHistName1_ + "_v1",
                                      "DeltaR_(offline,HLT)_triggerPass(" + triggerHistName1_ + ")",
                                      500,
                                      0.0,
                                      0.5));
  matchHists.push_back(ibooker.book1D("DelR_HLT_" + triggerHistName1_ + "_v2",
                                      "DeltaR_(offline,HLT)_triggerPass(" + triggerHistName1_ + ")",
                                      100,
                                      0.5,
                                      1.5));
  matchHists.push_back(ibooker.book1D(
      "PtRatio_HLT_" + triggerHistName1_, "PtRatio_(HLTPt/OfflinePt)_" + triggerHistName1_, 200, -5., 5.0));

  matchHists.push_back(ibooker.book1D("DelR_L1_" + triggerHistName1_ + "_v1",
                                      "DeltaR_(offline, L1)_triggerPass(" + triggerHistName1_ + ")",
                                      500,
                                      0.0,
                                      1.0));
  matchHists.push_back(ibooker.book1D("DelR_L1_" + triggerHistName1_ + "_v2",
                                      "DeltaR_(offline, L1)_triggerPass(" + triggerHistName1_ + ")",
                                      500,
                                      0.0,
                                      2.0));
  matchHists.push_back(ibooker.book1D(
      "PtRatio_L1_" + triggerHistName1_, "PtRatio_(HLTPt/OfflinePt)_" + triggerHistName1_, 200, -5., 5.0));

  matchHists.push_back(ibooker.book1D("DelR_HLT_" + triggerHistName2_ + "_v1",
                                      "DeltaR_(offline,HLT)_triggerPass(" + triggerHistName2_ + ")",
                                      500,
                                      0.0,
                                      0.5));
  matchHists.push_back(ibooker.book1D("DelR_HLT_" + triggerHistName2_ + "_v2",
                                      "DeltaR_(offline,HLT)_triggerPass(" + triggerHistName2_ + ")",
                                      100,
                                      0.5,
                                      1.5));
  matchHists.push_back(ibooker.book1D(
      "PtRatio_HLT_" + triggerHistName2_, "PtRatio_(HLTPt/OfflinePt)_" + triggerHistName2_, 200, -5., 5.0));

  matchHists.push_back(ibooker.book1D("DelR_L1_" + triggerHistName2_ + "_v1",
                                      "DeltaR_(offline, L1)_triggerPass(" + triggerHistName2_ + ")",
                                      250,
                                      0.0,
                                      0.5));
  matchHists.push_back(ibooker.book1D("DelR_L1_" + triggerHistName2_ + "_v2",
                                      "DeltaR_(offline, L1)_triggerPass(" + triggerHistName2_ + ")",
                                      100,
                                      0.5,
                                      1.5));
  matchHists.push_back(ibooker.book1D(
      "PtRatio_L1_" + triggerHistName2_, "PtRatio_(HLTPt/OfflinePt)_" + triggerHistName2_, 200, -5., 5.0));

  ibooker.cd();
  ibooker.setCurrentFolder(theFolder + "/EfficiencyInput");

  h_passHLTPath1_eta_Tight = ibooker.book1D(
      "passHLT" + triggerHistName1_ + "_eta_Tight", " HLT(" + triggerHistName1_ + ") pass #eta", 8, -2.5, 2.5);
  h_passHLTPath1_pt_Tight = ibooker.book1D(
      "passHLT" + triggerHistName1_ + "_pt_Tight", " HLT(" + triggerHistName1_ + ") pass pt", 10, 20, 220);
  h_passHLTPath1_phi_Tight = ibooker.book1D(
      "passHLT" + triggerHistName1_ + "_phi_Tight", " HLT(" + triggerHistName1_ + ") pass phi", 8, -3.0, 3.0);
  h_totalHLTPath1_eta_Tight = ibooker.book1D(
      "totalHLT" + triggerHistName1_ + "_eta_Tight", " HLT(" + triggerHistName1_ + ") total #eta", 8, -2.5, 2.5);
  h_totalHLTPath1_pt_Tight = ibooker.book1D(
      "totalHLT" + triggerHistName1_ + "_pt_Tight", " HLT(" + triggerHistName1_ + ") total pt", 10, 20., 220);
  h_totalHLTPath1_phi_Tight = ibooker.book1D(
      "totalHLT" + triggerHistName1_ + "_phi_Tight", " HLT(" + triggerHistName1_ + ") total phi", 8, -3.0, 3.0);

  h_passHLTPath2_eta_Tight = ibooker.book1D(
      "passHLT" + triggerHistName2_ + "_eta_Tight", " HLT(" + triggerHistName2_ + ") pass #eta", 8, -2.5, 2.5);
  h_passHLTPath2_pt_Tight = ibooker.book1D(
      "passHLT" + triggerHistName2_ + "_pt_Tight", " HLT(" + triggerHistName2_ + ") pass pt", 10, 20., 220);
  h_passHLTPath2_phi_Tight = ibooker.book1D(
      "passHLT" + triggerHistName2_ + "_phi_Tight", " HLT(" + triggerHistName2_ + ") pass phi", 8, -3.0, 3.0);
  h_totalHLTPath2_eta_Tight = ibooker.book1D(
      "totalHLT" + triggerHistName2_ + "_eta_Tight", " HLT(" + triggerHistName2_ + ") total #eta", 8, -2.5, 2.5);
  h_totalHLTPath2_pt_Tight = ibooker.book1D(
      "totalHLT" + triggerHistName2_ + "_pt_Tight", " HLT(" + triggerHistName2_ + ") total pt", 10, 20, 220);
  h_totalHLTPath2_phi_Tight = ibooker.book1D(
      "totalHLT" + triggerHistName2_ + "_phi_Tight", " HLT(" + triggerHistName2_ + ") total phi", 8, -3.0, 3.0);
}
void TriggerMatchMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LogTrace(metname) << "[TriggerMatchMonitor] Analyze the mu in different eta regions";

  edm::Handle<edm::View<reco::Muon>> muons;
  iEvent.getByToken(theMuonCollectionLabel_, muons);

  edm::Handle<edm::View<pat::Muon>> PATmuons;
  iEvent.getByToken(thePATMuonCollectionLabel_, PATmuons);

  edm::Handle<std::vector<pat::TriggerObjectStandAlone>> triggerObjects;
  iEvent.getByToken(triggerObjects_, triggerObjects);

  Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);

  reco::Vertex::Point posVtx;
  reco::Vertex::Error errVtx;
  Handle<std::vector<reco::Vertex>> recVtxs;
  iEvent.getByToken(primaryVerticesToken_, recVtxs);
  unsigned int theIndexOfThePrimaryVertex = 999.;
  for (unsigned int ind = 0; ind < recVtxs->size(); ++ind) {
    if ((*recVtxs)[ind].isValid() && !((*recVtxs)[ind].isFake())) {
      theIndexOfThePrimaryVertex = ind;
      break;
    }
  }
  if (theIndexOfThePrimaryVertex < 100) {
    posVtx = ((*recVtxs)[theIndexOfThePrimaryVertex]).position();
    errVtx = ((*recVtxs)[theIndexOfThePrimaryVertex]).error();
  } else {
    LogInfo("RecoMuonValidator") << "reco::PrimaryVertex not found, use BeamSpot position instead\n";
    Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByToken(beamSpotToken_, recoBeamSpotHandle);
    reco::BeamSpot bs = *recoBeamSpotHandle;
    posVtx = bs.position();
    errVtx(0, 0) = bs.BeamWidthX();
    errVtx(1, 1) = bs.BeamWidthY();
    errVtx(2, 2) = bs.sigmaZ();
  }
  const reco::Vertex thePrimaryVertex(posVtx, errVtx);

  if (PATmuons.isValid()) {                  //valid pat Muon
    for (const auto& patMuon : *PATmuons) {  //pat muon loop
      bool Isolated =
          patMuon.pfIsolationR04().sumChargedHadronPt +
              TMath::Max(0.,
                         patMuon.pfIsolationR04().sumNeutralHadronEt + patMuon.pfIsolationR04().sumPhotonEt -
                             0.5 * patMuon.pfIsolationR04().sumPUPt) /
                  patMuon.pt() <
          0.25;

      if (patMuon.isGlobalMuon() && Isolated && patMuon.isTightMuon(thePrimaryVertex)) {  //isolated tight muon

        TLorentzVector offlineMuon;
        offlineMuon.SetPtEtaPhiM(patMuon.pt(), patMuon.eta(), patMuon.phi(), 0.0);

        const char* ptrmuPath1 = triggerPathName1_.c_str();  //
        const char* ptrmuPath2 = triggerPathName2_.c_str();  //
        if (patMuon.pt() > triggerPtThresholdPath1_) {
          h_totalHLTPath1_eta_Tight->Fill(patMuon.eta());
          h_totalHLTPath1_pt_Tight->Fill(patMuon.pt());
          h_totalHLTPath1_phi_Tight->Fill(patMuon.phi());
        }
        if (patMuon.pt() > triggerPtThresholdPath2_) {
          h_totalHLTPath2_eta_Tight->Fill(patMuon.eta());
          h_totalHLTPath2_pt_Tight->Fill(patMuon.pt());
          h_totalHLTPath2_phi_Tight->Fill(patMuon.phi());
        }
        if (patMuon.triggered(ptrmuPath1) && patMuon.hltObject() != nullptr) {
          TLorentzVector hltMuon;
          hltMuon.SetPtEtaPhiM(patMuon.hltObject()->pt(), patMuon.hltObject()->eta(), patMuon.hltObject()->phi(), 0.0);
          double DelRrecoHLT = offlineMuon.DeltaR(hltMuon);

          matchHists[0]->Fill(DelRrecoHLT);
          matchHists[1]->Fill(DelRrecoHLT);
          matchHists[2]->Fill(patMuon.hltObject()->pt() / patMuon.pt());
          if (DelRrecoHLT < 0.2 && patMuon.pt() > triggerPtThresholdPath1_) {
            h_passHLTPath1_eta_Tight->Fill(patMuon.eta());
            h_passHLTPath1_pt_Tight->Fill(patMuon.pt());
            h_passHLTPath1_phi_Tight->Fill(patMuon.phi());
          }
          if (patMuon.l1Object() != nullptr) {
            TLorentzVector L1Muon;
            L1Muon.SetPtEtaPhiM(patMuon.l1Object()->pt(), patMuon.l1Object()->eta(), patMuon.l1Object()->phi(), 0.0);
            double DelRrecoL1 = offlineMuon.DeltaR(L1Muon);
            matchHists[3]->Fill(DelRrecoL1);
            matchHists[4]->Fill(DelRrecoL1);
            matchHists[5]->Fill(patMuon.l1Object()->pt() / patMuon.pt());
          }
        }

        /// Mu50 test
        if (patMuon.triggered(ptrmuPath2)) {
          TLorentzVector hltMuon50;
          hltMuon50.SetPtEtaPhiM(
              patMuon.hltObject()->pt(), patMuon.hltObject()->eta(), patMuon.hltObject()->phi(), 0.0);
          double DelRrecoHLT50 = offlineMuon.DeltaR(hltMuon50);

          matchHists[6]->Fill(DelRrecoHLT50);
          matchHists[7]->Fill(DelRrecoHLT50);
          matchHists[8]->Fill(patMuon.hltObject()->pt() / patMuon.pt());
          if (DelRrecoHLT50 < 0.2 && patMuon.pt() > triggerPtThresholdPath2_) {
            h_passHLTPath2_eta_Tight->Fill(patMuon.eta());
            h_passHLTPath2_pt_Tight->Fill(patMuon.pt());
            h_passHLTPath2_phi_Tight->Fill(patMuon.phi());
          }

          if (patMuon.l1Object() != nullptr) {
            TLorentzVector L1Muon50;
            L1Muon50.SetPtEtaPhiM(patMuon.l1Object()->pt(), patMuon.l1Object()->eta(), patMuon.l1Object()->phi(), 0.0);
            double DelRrecoL150 = offlineMuon.DeltaR(L1Muon50);
            matchHists[9]->Fill(DelRrecoL150);
            matchHists[10]->Fill(DelRrecoL150);
            matchHists[11]->Fill(patMuon.l1Object()->pt() / patMuon.pt());
          }
        }
      }  //isolated tight muon
    }    //pat muon loop
  }      //valid pat muon
}
