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

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

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
  std::vector<std::string> HLTNames_;
  bool changed_, firstEvent_;

  edm::EDGetTokenT<trigger::TriggerEvent> tok_trigEvt;
  edm::EDGetTokenT<edm::TriggerResults> tok_trigRes;

  TH1I *h_nHLT, *h_HLTAccept, *h_HLTCorr;
  TH2I* h_nHLTvsRN;
  std::vector<TH1I*> h_HLTAccepts;
  int nRun_;
};

StudyTriggerHLT::StudyTriggerHLT(const edm::ParameterSet& iConfig)
    : verbosity_(iConfig.getUntrackedParameter<int>("verbosity", 0)),
      triggerEvent_(edm::InputTag("hltTriggerSummaryAOD", "", "HLT")),
      theTriggerResultsLabel_(edm::InputTag("TriggerResults", "", "HLT")),
      nRun_(0) {
  usesResource(TFileService::kSharedResource);

  // define tokens for access
  tok_trigEvt = consumes<trigger::TriggerEvent>(triggerEvent_);
  tok_trigRes = consumes<edm::TriggerResults>(theTriggerResultsLabel_);

  edm::LogInfo("IsoTrack") << "Verbosity " << verbosity_ << " Trigger labels " << triggerEvent_ << " and "
                           << theTriggerResultsLabel_;

  firstEvent_ = true;
  changed_ = false;
}

void StudyTriggerHLT::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<int>("verbosity", 0);
  descriptions.add("studyTriggerHLT", desc);
}

void StudyTriggerHLT::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  int RunNo = iEvent.id().run();
  int EvtNo = iEvent.id().event();

  if (verbosity_ > 0)
    edm::LogInfo("IsoTrack") << "RunNo " << RunNo << " EvtNo " << EvtNo << " Lumi " << iEvent.luminosityBlock()
                             << " Bunch " << iEvent.bunchCrossing();

  trigger::TriggerEvent triggerEvent;
  edm::Handle<trigger::TriggerEvent> triggerEventHandle;
  iEvent.getByToken(tok_trigEvt, triggerEventHandle);

  if (!triggerEventHandle.isValid()) {
    edm::LogWarning("IsoTrack") << "Error! Can't get the product " << triggerEvent_.label();
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
          edm::LogInfo("IsoTrack") << "Wrong trigger " << RunNo << " Event " << EvtNo << " Hlt " << iHLT;
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
}

void StudyTriggerHLT::beginJob() {
  // Book histograms
  h_nHLT = fs_->make<TH1I>("h_nHLT", "size of trigger Names", 1000, 0, 1000);
  h_HLTAccept = fs_->make<TH1I>("h_HLTAccept", "HLT Accepts for all runs", 500, 0, 500);
  for (int i = 1; i <= 500; ++i)
    h_HLTAccept->GetXaxis()->SetBinLabel(i, " ");
  h_nHLTvsRN = fs_->make<TH2I>("h_nHLTvsRN", "size of trigger Names vs RunNo", 300, 319200, 319500, 100, 400, 500);
}

// ------------ method called when starting to processes a run  ------------
void StudyTriggerHLT::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  char hname[100], htit[400];
  edm::LogInfo("IsoTrack") << "Run[" << nRun_ << "] " << iRun.run() << " hltconfig.init "
                           << hltConfig_.init(iRun, iSetup, "HLT", changed_);
  sprintf(hname, "h_HLTAccepts_%i", iRun.run());
  sprintf(htit, "HLT Accepts for Run No %i", iRun.run());
  TH1I* hnew = fs_->make<TH1I>(hname, htit, 500, 0, 500);
  for (int i = 1; i <= 500; ++i)
    hnew->GetXaxis()->SetBinLabel(i, " ");
  h_HLTAccepts.push_back(hnew);
  edm::LogInfo("IsoTrack") << "beginrun " << iRun.run();
  firstEvent_ = true;
  changed_ = false;
}

// ------------ method called when ending the processing of a run  ------------
void StudyTriggerHLT::endRun(edm::Run const& iRun, edm::EventSetup const&) {
  ++nRun_;
  edm::LogInfo("IsoTrack") << "endrun[" << nRun_ << "] " << iRun.run();
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
