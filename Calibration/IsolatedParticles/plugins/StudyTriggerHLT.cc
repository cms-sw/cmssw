// -*- C++ -*-
//
// Package:    IsolatedParticles
// Class:      StudyTriggerHLT
//
/**\class StudyTriggerHLT StudyTriggerHLT.cc Calibration/IsolatedParticles/plugins/StudyTriggerHLT.cc

 Description: Studies single particle response measurements in data/MC

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Thu Mar  4 18:52:02 CST 2011
//
//

// system include files
#include <memory>
#include <string>

// Root objects
#include "TH1.h"
#include "TH2.h"

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

class StudyTriggerHLT : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::SharedResources> {
public:
  explicit StudyTriggerHLT(const edm::ParameterSet&);
  ~StudyTriggerHLT() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) {}

  std::string truncate_str(const std::string&);

  // ----------member data ---------------------------
  HLTConfigProvider hltConfig_;
  edm::Service<TFileService> fs_;
  const int verbosity_;
  const std::vector<std::string> trigNames_, newNames_;
  const edm::InputTag triggerEvent_, theTriggerResultsLabel_;
  const edm::InputTag labelMuon_, labelGenTrack_;
  const std::string theTrackQuality_;
  std::vector<std::string> HLTNames_;
  bool changed_, firstEvent_;
  reco::TrackBase::TrackQuality trackQuality_;

  edm::EDGetTokenT<trigger::TriggerEvent> tok_trigEvt;
  edm::EDGetTokenT<edm::TriggerResults> tok_trigRes;
  edm::EDGetTokenT<reco::MuonCollection> tok_Muon_;
  edm::EDGetTokenT<reco::TrackCollection> tok_genTrack_;
  std::vector<bool> mediumMuon_;
  TH1I *h_nHLT, *h_HLTAccept, *h_HLTCorr;
  TH2I* h_nHLTvsRN;
  std::vector<TH1I*> h_HLTAccepts;
  TH1D *h_pt, *h_eta, *h_phi, *h_dr1, *h_dr2, *h_dr3;
  int nRun_;
};

StudyTriggerHLT::StudyTriggerHLT(const edm::ParameterSet& iConfig)
    : verbosity_(iConfig.getParameter<int>("verbosity")),
      triggerEvent_(edm::InputTag("hltTriggerSummaryAOD", "", "HLT")),
      theTriggerResultsLabel_(edm::InputTag("TriggerResults", "", "HLT")),
      labelMuon_(iConfig.getParameter<edm::InputTag>("labelMuon")),
      labelGenTrack_(iConfig.getParameter<edm::InputTag>("labelTrack")),
      theTrackQuality_(iConfig.getParameter<std::string>("trackQuality")),
      nRun_(0) {
  usesResource(TFileService::kSharedResource);
  trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality_);

  // define tokens for access
  tok_trigEvt = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes = consumes<edm::TriggerResults>(theTriggerResultsLabel_);
  tok_Muon_ = consumes<reco::MuonCollection>(labelMuon_);
  tok_genTrack_ = consumes<reco::TrackCollection>(labelGenTrack_);

  edm::LogVerbatim("StudyHLT") << "Verbosity " << verbosity_ << " Trigger labels " << triggerEvent_ << " and "
                               << theTriggerResultsLabel_ << " Labels used: Track " << labelGenTrack_ << " Muon "
                               << labelMuon_ << " Track Quality " << theTrackQuality_;

  firstEvent_ = true;
  changed_ = false;
}

void StudyTriggerHLT::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("verbosity", 0);
  desc.add<edm::InputTag>("labelMuon", edm::InputTag("muons", "", "RECO"));
  desc.add<edm::InputTag>("labelTrack", edm::InputTag("generalTracks", "", "RECO"));
  desc.add<std::string>("trackQuality", "highPurity");
  descriptions.add("studyTriggerHLT", desc);
}

void StudyTriggerHLT::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  int RunNo = iEvent.id().run();
  int EvtNo = iEvent.id().event();

  if (verbosity_ > 0)
    edm::LogVerbatim("StudyHLT") << "RunNo " << RunNo << " EvtNo " << EvtNo << " Lumi " << iEvent.luminosityBlock()
                                 << " Bunch " << iEvent.bunchCrossing();

  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt, triggerEventHandle);

  if (!triggerEventHandle.isValid()) {
    edm::LogWarning("StudyHLT") << "Error! Can't get the product " << triggerEvent_.label();
  } else {
    triggerEvent = *(triggerEventHandle.product());

    /////////////////////////////TriggerResults
    edm::Handle<edm::TriggerResults> triggerResults;
    iEvent.getByToken(tok_trigRes, triggerResults);

    if (triggerResults.isValid()) {
      h_nHLT->Fill(triggerResults->size());
      h_nHLTvsRN->Fill(RunNo, triggerResults->size());

      const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);
      const std::vector<std::string>& triggerNames_ = triggerNames.triggerNames();
      for (unsigned int iHLT = 0; iHLT < triggerResults->size(); iHLT++) {
        int ipos = -1;
        std::string newtriggerName = truncate_str(triggerNames_[iHLT]);
        for (unsigned int i = 0; i < HLTNames_.size(); ++i) {
          if (newtriggerName == HLTNames_[i]) {
            ipos = i + 1;
            break;
          }
        }
        if (ipos < 0) {
          HLTNames_.push_back(newtriggerName);
          ipos = (int)(HLTNames_.size());
          if (ipos <= h_HLTAccept->GetNbinsX())
            h_HLTAccept->GetXaxis()->SetBinLabel(ipos, newtriggerName.c_str());
        }
        if ((int)(iHLT + 1) > h_HLTAccepts[nRun_]->GetNbinsX()) {
          edm::LogVerbatim("StudyHLT") << "Wrong trigger " << RunNo << " Event " << EvtNo << " Hlt " << iHLT;
        } else {
          if (firstEvent_)
            h_HLTAccepts[nRun_]->GetXaxis()->SetBinLabel(iHLT + 1, newtriggerName.c_str());
        }
        int hlt = triggerResults->accept(iHLT);
        if (hlt) {
          h_HLTAccepts[nRun_]->Fill(iHLT + 1);
          h_HLTAccept->Fill(ipos);
        }
      }
    }
  }
  firstEvent_ = false;

  double globalMin = 1000;
  edm::Handle<reco::MuonCollection> muonEventHandle;
  iEvent.getByToken(tok_Muon_, muonEventHandle);
  edm::Handle<reco::TrackCollection> trackEventHandle;
  iEvent.getByToken(tok_genTrack_, trackEventHandle);
  edm::LogVerbatim("StudyHLT") << "Muon Handle " << muonEventHandle.isValid() << " Track Handle "
                               << trackEventHandle.isValid();
  for (reco::TrackCollection::const_iterator track1 = trackEventHandle->begin(); track1 != trackEventHandle->end();
       ++track1) {
    double localMin = 1000;
    if (muonEventHandle.isValid()) {
      for (reco::MuonCollection::const_iterator recMuon = muonEventHandle->begin(); recMuon != muonEventHandle->end();
           ++recMuon) {
        if (((recMuon->isPFMuon()) && (recMuon->isGlobalMuon() || recMuon->isTrackerMuon())) &&
            (recMuon->innerTrack()->validFraction() > 0.49)) {
          double chiGlobal = ((recMuon->globalTrack().isNonnull()) ? recMuon->globalTrack()->normalizedChi2() : 999);
          bool goodGlob =
              (recMuon->isGlobalMuon() && chiGlobal < 3 && recMuon->combinedQuality().chi2LocalPosition < 12 &&
               recMuon->combinedQuality().trkKink < 20);
          if (muon::segmentCompatibility(*recMuon) > (goodGlob ? 0.303 : 0.451)) {
            double dr = reco::deltaR(track1->eta(), track1->phi(), recMuon->eta(), recMuon->phi());
            if (dr < localMin) {
              localMin = dr;
              if (localMin < globalMin)
                globalMin = localMin;
            }
          }
        }
      }
    }
    h_pt->Fill(track1->pt());
    h_eta->Fill(track1->eta());
    h_phi->Fill(track1->phi());
    h_dr1->Fill(localMin);
    if (track1->quality(trackQuality_))
      h_dr3->Fill(localMin);
    edm::LogVerbatim("StudyHLT") << "Track pT " << track1->pt() << " eta " << track1->eta() << " phi " << track1->phi()
                                 << " minimum distance " << localMin;
  }
  edm::LogVerbatim("StudyHLT") << "GlobalMinimum  = " << globalMin;
  h_dr2->Fill(globalMin);
}

void StudyTriggerHLT::beginJob() {
  // Book histograms
  h_nHLT = fs_->make<TH1I>("h_nHLT", "size of trigger Names", 1000, 0, 1000);
  h_HLTAccept = fs_->make<TH1I>("h_HLTAccept", "HLT Accepts for all runs", 500, 0, 500);
  for (int i = 1; i <= 500; ++i)
    h_HLTAccept->GetXaxis()->SetBinLabel(i, " ");
  h_nHLTvsRN = fs_->make<TH2I>("h_nHLTvsRN", "size of trigger Names vs RunNo", 300, 319200, 319500, 100, 400, 500);
  h_pt = fs_->make<TH1D>("h_pt", "p_{t}", 50, 0, 20);
  h_pt->Sumw2();
  h_eta = fs_->make<TH1D>("h_eta", "#eta", 50, -3, 3);
  h_eta->Sumw2();
  h_phi = fs_->make<TH1D>("h_phi", "#phi", 50, -10, 10);
  h_phi->Sumw2();
  h_dr1 = fs_->make<TH1D>("dR1", "#Delta R (Track)", 1000, 0, 0.1);
  h_dr1->Sumw2();
  h_dr2 = fs_->make<TH1D>("dR2", "#Delta R (Global)", 3000, 0, 0.000003);
  h_dr2->Sumw2();
  h_dr3 = fs_->make<TH1D>("dR3", "#Delta R (Good Track)", 1000, 0, 0.1);
  h_dr3->Sumw2();
}

// ------------ method called when starting to processes a run  ------------
void StudyTriggerHLT::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  char hname[100], htit[400];
  edm::LogVerbatim("StudyHLT") << "Run[" << nRun_ << "] " << iRun.run() << " hltconfig.init "
                               << hltConfig_.init(iRun, iSetup, "HLT", changed_);
  sprintf(hname, "h_HLTAccepts_%i", iRun.run());
  sprintf(htit, "HLT Accepts for Run No %i", iRun.run());
  TH1I* hnew = fs_->make<TH1I>(hname, htit, 500, 0, 500);
  for (int i = 1; i <= 500; ++i)
    hnew->GetXaxis()->SetBinLabel(i, " ");
  h_HLTAccepts.push_back(hnew);
  edm::LogVerbatim("StudyHLT") << "beginRun " << iRun.run();
  firstEvent_ = true;
  changed_ = false;
}

// ------------ method called when ending the processing of a run  ------------
void StudyTriggerHLT::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  ++nRun_;
  edm::LogVerbatim("StudyHLT") << "endRun[" << nRun_ << "] " << iRun.run();
}

std::string StudyTriggerHLT::truncate_str(const std::string& str) {
  std::string truncated_str(str);
  int length = str.length();
  for (int i = 0; i < length - 2; i++) {
    if (str[i] == '_' && str[i + 1] == 'v' && isdigit(str.at(i + 2))) {
      int z = i + 1;
      truncated_str = str.substr(0, z);
    }
  }
  return (truncated_str);
}

DEFINE_FWK_MODULE(StudyTriggerHLT);
