/** \class HLTEventAnalyzerRAW
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HLTrigger/HLTcore/interface/HLTEventAnalyzerRAW.h"

// need access to class objects being referenced to get their content!
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

#include "DataFormats/L1Trigger/interface/L1HFRings.h"         // deprecate
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"      // deprecate
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"     // deprecate
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"    // deprecate
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"  // deprecate

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/MuonShower.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
#include "DataFormats/L1TParticleFlow/interface/HPSPFTau.h"
#include "DataFormats/L1TParticleFlow/interface/PFTrack.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include <cassert>

//
// constructor
//
HLTEventAnalyzerRAW::HLTEventAnalyzerRAW(const edm::ParameterSet& ps)
    : processName_(ps.getParameter<std::string>("processName")),
      triggerName_(ps.getParameter<std::string>("triggerName")),
      triggerResultsTag_(ps.getParameter<edm::InputTag>("triggerResults")),
      triggerResultsToken_(consumes<edm::TriggerResults>(triggerResultsTag_)),
      triggerEventWithRefsTag_(ps.getParameter<edm::InputTag>("triggerEventWithRefs")),
      triggerEventWithRefsToken_(consumes<trigger::TriggerEventWithRefs>(triggerEventWithRefsTag_)),
      verbose_(ps.getParameter<bool>("verbose")),
      permissive_(ps.getParameter<bool>("permissive")) {
  LOG(logMsgType_) << logMsgType_ << " configuration:\n"
                   << "   ProcessName = " << processName_ << "\n"
                   << "   TriggerName = " << triggerName_ << "\n"
                   << "   TriggerResultsTag = " << triggerResultsTag_.encode() << "\n"
                   << "   TriggerEventWithRefsTag = " << triggerEventWithRefsTag_.encode();
}

//
// member functions
//
void HLTEventAnalyzerRAW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("processName", "HLT");
  desc.add<std::string>("triggerName", "@")
      ->setComment("name of trigger Path to consider (use \"@\" to consider all Paths)");
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("triggerEventWithRefs", edm::InputTag("hltTriggerSummaryRAW", "", "HLT"));
  desc.add<bool>("verbose", false)->setComment("enable verbose mode");
  desc.add<bool>("permissive", false)
      ->setComment("if true, exceptions due to Refs pointing to unavailable collections are bypassed");
  descriptions.add("hltEventAnalyzerRAW", desc);
}

void HLTEventAnalyzerRAW::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed(true);
  if (hltConfig_.init(iRun, iSetup, processName_, changed)) {
    if (changed) {
      // check if trigger name in (new) config
      if (triggerName_ != "@") {  // "@" means: analyze all triggers in config
        const unsigned int n(hltConfig_.size());
        const unsigned int triggerIndex(hltConfig_.triggerIndex(triggerName_));
        if (triggerIndex >= n) {
          LOG(logMsgType_) << logMsgType_ << "::beginRun: TriggerName " << triggerName_
                           << " not available in (new) config!";
          LOG(logMsgType_) << "Available TriggerNames are: ";
          hltConfig_.dump("Triggers");
        }
      }
      // in verbose mode, print process info to stdout
      if (verbose_) {
        hltConfig_.dump("ProcessName");
        hltConfig_.dump("GlobalTag");
        hltConfig_.dump("TableName");
        hltConfig_.dump("Streams");
        hltConfig_.dump("Datasets");
        hltConfig_.dump("PrescaleTable");
        hltConfig_.dump("ProcessPSet");
      }
    }
  } else {
    LOG(logMsgType_) << logMsgType_ << "::beginRun: config extraction failure with process name " << processName_;
  }
}

void HLTEventAnalyzerRAW::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get event products
  triggerResultsHandle_ = iEvent.getHandle(triggerResultsToken_);
  if (not triggerResultsHandle_.isValid()) {
    LOG(logMsgType_) << logMsgType_ << "::analyze: Error in getting TriggerResults product from Event!";
    return;
  }
  triggerEventWithRefsHandle_ = iEvent.getHandle(triggerEventWithRefsToken_);
  if (not triggerEventWithRefsHandle_.isValid()) {
    LOG(logMsgType_) << logMsgType_ << "::analyze: Error in getting TriggerEventWithRefs product from Event!";
    return;
  }

  // sanity check
  assert(triggerResultsHandle_->size() == hltConfig_.size());

  // analyze this event for the triggers requested
  if (triggerName_ == "@") {
    const unsigned int n(hltConfig_.size());
    for (unsigned int i = 0; i != n; ++i) {
      analyzeTrigger(iEvent, iSetup, hltConfig_.triggerName(i));
    }
  } else {
    analyzeTrigger(iEvent, iSetup, triggerName_);
  }

  return;
}

void HLTEventAnalyzerRAW::analyzeTrigger(const edm::Event& iEvent,
                                         const edm::EventSetup& iSetup,
                                         const std::string& triggerName) {
  const unsigned int n(hltConfig_.size());
  const unsigned int triggerIndex(hltConfig_.triggerIndex(triggerName));
  assert(triggerIndex == iEvent.triggerNames(*triggerResultsHandle_).triggerIndex(triggerName));

  // abort on invalid trigger name
  if (triggerIndex >= n) {
    LOG(logMsgType_) << logMsgType_ << "::analyzeTrigger: path " << triggerName << " - not found!";
    return;
  }

  LOG(logMsgType_) << logMsgType_ << "::analyzeTrigger: path " << triggerName << " [" << triggerIndex << "]";

  // results from TriggerResults product
  LOG(logMsgType_) << " Trigger path status:"
                   << " WasRun=" << triggerResultsHandle_->wasrun(triggerIndex)
                   << " Accept=" << triggerResultsHandle_->accept(triggerIndex)
                   << " Error=" << triggerResultsHandle_->error(triggerIndex);

  // modules on this trigger path
  const unsigned int m(hltConfig_.size(triggerIndex));
  const std::vector<std::string>& moduleLabels(hltConfig_.moduleLabels(triggerIndex));
  assert(m == moduleLabels.size());

  // skip empty Path
  if (m == 0) {
    LOG(logMsgType_) << logMsgType_ << "::analyzeTrigger: path " << triggerName << " [" << triggerIndex
                     << "] is empty!";
    return;
  }

  // index of last module executed in this Path
  const unsigned int moduleIndex(triggerResultsHandle_->index(triggerIndex));
  assert(moduleIndex < m);

  LOG(logMsgType_) << " Last active module - label/type: " << moduleLabels[moduleIndex] << "/"
                   << hltConfig_.moduleType(moduleLabels[moduleIndex]) << " [" << moduleIndex << " out of 0-" << (m - 1)
                   << " on this path]";

  // results from TriggerEventWithRefs product
  photonIds_.clear();
  photonRefs_.clear();
  electronIds_.clear();
  electronRefs_.clear();
  muonIds_.clear();
  muonRefs_.clear();
  jetIds_.clear();
  jetRefs_.clear();
  compositeIds_.clear();
  compositeRefs_.clear();
  basemetIds_.clear();
  basemetRefs_.clear();
  calometIds_.clear();
  calometRefs_.clear();
  pixtrackIds_.clear();
  pixtrackRefs_.clear();

  l1emIds_.clear();
  l1emRefs_.clear();
  l1muonIds_.clear();
  l1muonRefs_.clear();
  l1jetIds_.clear();
  l1jetRefs_.clear();
  l1etmissIds_.clear();
  l1etmissRefs_.clear();
  l1hfringsIds_.clear();
  l1hfringsRefs_.clear();

  l1tmuonIds_.clear();
  l1tmuonRefs_.clear();
  l1tmuonShowerIds_.clear();
  l1tmuonShowerRefs_.clear();
  l1tegammaIds_.clear();
  l1tegammaRefs_.clear();
  l1tjetIds_.clear();
  l1tjetRefs_.clear();
  l1ttauIds_.clear();
  l1ttauRefs_.clear();
  l1tetsumIds_.clear();
  l1tetsumRefs_.clear();

  l1ttkmuIds_.clear();
  l1ttkmuRefs_.clear();
  l1ttkeleIds_.clear();
  l1ttkeleRefs_.clear();
  l1ttkemIds_.clear();
  l1ttkemRefs_.clear();
  l1tpfjetIds_.clear();
  l1tpfjetRefs_.clear();
  l1tpftauIds_.clear();
  l1tpftauRefs_.clear();
  l1thpspftauIds_.clear();
  l1thpspftauRefs_.clear();
  l1tpftrackIds_.clear();
  l1tpftrackRefs_.clear();

  pfjetIds_.clear();
  pfjetRefs_.clear();
  pftauIds_.clear();
  pftauRefs_.clear();
  pfmetIds_.clear();
  pfmetRefs_.clear();

  // Attention: must look only for modules actually run in this path for this event!
  for (unsigned int j = 0; j <= moduleIndex; ++j) {
    const std::string& moduleLabel(moduleLabels[j]);
    const std::string moduleType(hltConfig_.moduleType(moduleLabel));
    // check whether the module is packed up in TriggerEventWithRefs product
    const unsigned int filterIndex(
        triggerEventWithRefsHandle_->filterIndex(edm::InputTag(moduleLabel, "", processName_)));
    if (filterIndex < triggerEventWithRefsHandle_->size()) {
      LOG(logMsgType_) << " Filter in slot " << j << " - label/type " << moduleLabel << "/" << moduleType;
      LOG(logMsgType_) << " Filter packed up at: " << filterIndex;
      LOG(logMsgType_) << "  Accepted objects:";

      // Photons
      triggerEventWithRefsHandle_->getObjects(filterIndex, photonIds_, photonRefs_);
      showObjects(photonIds_, photonRefs_, "Photons");

      // Electrons
      triggerEventWithRefsHandle_->getObjects(filterIndex, electronIds_, electronRefs_);
      showObjects(electronIds_, electronRefs_, "Electrons");

      // Muons
      triggerEventWithRefsHandle_->getObjects(filterIndex, muonIds_, muonRefs_);
      showObjects(muonIds_, muonRefs_, "Muons");

      // Jets
      triggerEventWithRefsHandle_->getObjects(filterIndex, jetIds_, jetRefs_);
      showObjects(jetIds_, jetRefs_, "Jets");

      // Composites
      triggerEventWithRefsHandle_->getObjects(filterIndex, compositeIds_, compositeRefs_);
      showObjects(compositeIds_, compositeRefs_, "Composites");

      // BaseMETs
      triggerEventWithRefsHandle_->getObjects(filterIndex, basemetIds_, basemetRefs_);
      showObjects(basemetIds_, basemetRefs_, "BaseMETs");

      // CaloMETs
      triggerEventWithRefsHandle_->getObjects(filterIndex, calometIds_, calometRefs_);
      showObjects(calometIds_, calometRefs_, "CaloMETs");

      // PixTracks
      triggerEventWithRefsHandle_->getObjects(filterIndex, pixtrackIds_, pixtrackRefs_);
      showObjects(pixtrackIds_, pixtrackRefs_, "PixTracks");

      // L1EMs
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1emIds_, l1emRefs_);
      showObjects(l1emIds_, l1emRefs_, "L1EMs");

      // L1Muons
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1muonIds_, l1muonRefs_);
      showObjects(l1muonIds_, l1muonRefs_, "L1Muons");

      // L1Jets
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1jetIds_, l1jetRefs_);
      showObjects(l1jetIds_, l1jetRefs_, "L1Jets");

      // L1EtMiss
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1etmissIds_, l1etmissRefs_);
      showObjects(l1etmissIds_, l1etmissRefs_, "L1EtMiss");

      // L1HFRings
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1hfringsIds_, l1hfringsRefs_);
      showObjects(l1hfringsIds_, l1hfringsRefs_, "L1HFRings");

      // L1TMuons
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1tmuonIds_, l1tmuonRefs_);
      showObjects(l1tmuonIds_, l1tmuonRefs_, "L1TMuons");

      // L1TMuonShowers
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1tmuonShowerIds_, l1tmuonShowerRefs_);
      showObjects(l1tmuonShowerIds_, l1tmuonShowerRefs_, "L1TMuonShowers");

      // L1TEGammas
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1tegammaIds_, l1tegammaRefs_);
      showObjects(l1tegammaIds_, l1tegammaRefs_, "L1TEGammas");

      // L1TJets
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1tjetIds_, l1tjetRefs_);
      showObjects(l1tjetIds_, l1tjetRefs_, "L1TJets");

      // L1TTaus
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1ttauIds_, l1ttauRefs_);
      showObjects(l1ttauIds_, l1ttauRefs_, "L1TTaus");

      // L1TEtSums
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1tetsumIds_, l1tetsumRefs_);
      showObjects(l1tetsumIds_, l1tetsumRefs_, "L1TEtSum");

      /// Phase 2

      // L1TTkMuons
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1ttkmuIds_, l1ttkmuRefs_);
      showObjects(l1ttkmuIds_, l1ttkmuRefs_, "L1TTkMuons");

      // L1TTkElectrons
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1ttkeleIds_, l1ttkeleRefs_);
      showObjects(l1ttkeleIds_, l1ttkeleRefs_, "L1TTkElectrons");

      // L1TTkEMs
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1ttkemIds_, l1ttkemRefs_);
      showObjects(l1ttkemIds_, l1ttkemRefs_, "L1TTkEMs");

      // L1TPFJets
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1tpfjetIds_, l1tpfjetRefs_);
      showObjects(l1tpfjetIds_, l1tpfjetRefs_, "L1TPFJets");

      // L1TPFTaus
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1tpftauIds_, l1tpftauRefs_);
      showObjects(l1tpftauIds_, l1tpftauRefs_, "L1TPFTaus");

      // L1THPSPFTaus
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1thpspftauIds_, l1thpspftauRefs_);
      showObjects(l1thpspftauIds_, l1thpspftauRefs_, "L1THPSPFTaus");

      // L1TPFTracks
      triggerEventWithRefsHandle_->getObjects(filterIndex, l1tpftrackIds_, l1tpftrackRefs_);
      showObjects(l1tpftrackIds_, l1tpftrackRefs_, "L1TPFTracks");

      // PFJets
      triggerEventWithRefsHandle_->getObjects(filterIndex, pfjetIds_, pfjetRefs_);
      showObjects(pfjetIds_, pfjetRefs_, "PFJets");

      // PFTaus
      triggerEventWithRefsHandle_->getObjects(filterIndex, pftauIds_, pftauRefs_);
      showObjects(pftauIds_, pftauRefs_, "PFTaus");

      // PFMETs
      triggerEventWithRefsHandle_->getObjects(filterIndex, pfmetIds_, pfmetRefs_);
      showObjects(pfmetIds_, pfmetRefs_, "PFMETs");
    }
  }

  return;
}

template <>
void HLTEventAnalyzerRAW::showObject(LOG& log, trigger::VRl1hfrings::value_type const& ref) const {
  log << "hfEtSum(ring1PosEta)=" << ref->hfEtSum(l1extra::L1HFRings::kRing1PosEta)
      << " hfEtSum(ring1NegEta)=" << ref->hfEtSum(l1extra::L1HFRings::kRing1NegEta)
      << " hfEtSum(ring2PosEta)=" << ref->hfEtSum(l1extra::L1HFRings::kRing2PosEta)
      << " hfEtSum(ring2NegEta)=" << ref->hfEtSum(l1extra::L1HFRings::kRing2NegEta)
      << " hfBitCount(ring1PosEta)=" << ref->hfBitCount(l1extra::L1HFRings::kRing1PosEta)
      << " hfBitCount(ring1NegEta)=" << ref->hfBitCount(l1extra::L1HFRings::kRing1NegEta)
      << " hfBitCount(ring2PosEta)=" << ref->hfBitCount(l1extra::L1HFRings::kRing2PosEta)
      << " hfBitCount(ring2NegEta)=" << ref->hfBitCount(l1extra::L1HFRings::kRing2NegEta);
}
