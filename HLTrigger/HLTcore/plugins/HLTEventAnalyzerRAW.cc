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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
#include "DataFormats/L1Trigger/interface/L1HFRings.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
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
// constructors and destructor
//
HLTEventAnalyzerRAW::HLTEventAnalyzerRAW(const edm::ParameterSet& ps)
    : processName_(ps.getParameter<std::string>("processName")),
      triggerName_(ps.getParameter<std::string>("triggerName")),
      triggerResultsTag_(ps.getParameter<edm::InputTag>("triggerResults")),
      triggerResultsToken_(consumes<edm::TriggerResults>(triggerResultsTag_)),
      triggerEventWithRefsTag_(ps.getParameter<edm::InputTag>("triggerEventWithRefs")),
      triggerEventWithRefsToken_(consumes<trigger::TriggerEventWithRefs>(triggerEventWithRefsTag_)) {
  using namespace std;
  using namespace edm;

  LogVerbatim("HLTEventAnalyzerRAW") << "HLTEventAnalyzerRAW configuration: " << endl
                                     << "   ProcessName = " << processName_ << endl
                                     << "   TriggerName = " << triggerName_ << endl
                                     << "   TriggerResultsTag = " << triggerResultsTag_.encode() << endl
                                     << "   TriggerEventWithRefsTag = " << triggerEventWithRefsTag_.encode() << endl;
}

HLTEventAnalyzerRAW::~HLTEventAnalyzerRAW() = default;

//
// member functions
//
void HLTEventAnalyzerRAW::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("processName", "HLT");
  desc.add<std::string>("triggerName", "@");
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("triggerEventWithRefs", edm::InputTag("hltTriggerSummaryRAW", "", "HLT"));
  descriptions.add("hltEventAnalyzerRAW", desc);
}

void HLTEventAnalyzerRAW::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}

void HLTEventAnalyzerRAW::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  using namespace std;
  using namespace edm;

  bool changed(true);
  if (hltConfig_.init(iRun, iSetup, processName_, changed)) {
    if (changed) {
      // check if trigger name in (new) config
      if (triggerName_ != "@") {  // "@" means: analyze all triggers in config
        const unsigned int n(hltConfig_.size());
        const unsigned int triggerIndex(hltConfig_.triggerIndex(triggerName_));
        if (triggerIndex >= n) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "HLTEventAnalyzerRAW::analyze:"
              << " TriggerName " << triggerName_ << " not available in (new) config!" << endl;
          LogVerbatim("HLTEventAnalyzerRAW") << "Available TriggerNames are: " << endl;
          hltConfig_.dump("Triggers");
        }
      }
    }
  } else {
    LogVerbatim("HLTEventAnalyzerRAW") << "HLTEventAnalyzerRAW::analyze:"
                                       << " config extraction failure with process name " << processName_ << endl;
  }
}

// ------------ method called to produce the data  ------------
void HLTEventAnalyzerRAW::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace std;
  using namespace edm;

  LogVerbatim("HLTEventAnalyzerRAW") << endl;

  // get event products
  iEvent.getByToken(triggerResultsToken_, triggerResultsHandle_);
  if (!triggerResultsHandle_.isValid()) {
    LogVerbatim("HLTEventAnalyzerRAW")
        << "HLTEventAnalyzerRAW::analyze: Error in getting TriggerResults product from Event!" << endl;
    return;
  }
  iEvent.getByToken(triggerEventWithRefsToken_, triggerEventWithRefsHandle_);
  if (!triggerEventWithRefsHandle_.isValid()) {
    LogVerbatim("HLTEventAnalyzerRAW")
        << "HLTEventAnalyzerRAW::analyze: Error in getting TriggerEventWithRefs product from Event!" << endl;
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
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  LogVerbatim("HLTEventAnalyzerRAW") << endl;

  const unsigned int n(hltConfig_.size());
  const unsigned int triggerIndex(hltConfig_.triggerIndex(triggerName));
  assert(triggerIndex == iEvent.triggerNames(*triggerResultsHandle_).triggerIndex(triggerName));

  // abort on invalid trigger name
  if (triggerIndex >= n) {
    LogVerbatim("HLTEventAnalyzerRAW") << "HLTEventAnalyzerRAW::analyzeTrigger: path " << triggerName << " - not found!"
                                       << endl;
    return;
  }

  LogVerbatim("HLTEventAnalyzerRAW") << "HLTEventAnalyzerRAW::analyzeTrigger: path " << triggerName << " ["
                                     << triggerIndex << "]" << endl;
  // modules on this trigger path
  const unsigned int m(hltConfig_.size(triggerIndex));
  const vector<string>& moduleLabels(hltConfig_.moduleLabels(triggerIndex));

  // Results from TriggerResults product
  LogVerbatim("HLTEventAnalyzerRAW") << " Trigger path status:"
                                     << " WasRun=" << triggerResultsHandle_->wasrun(triggerIndex)
                                     << " Accept=" << triggerResultsHandle_->accept(triggerIndex)
                                     << " Error =" << triggerResultsHandle_->error(triggerIndex) << endl;
  const unsigned int moduleIndex(triggerResultsHandle_->index(triggerIndex));
  LogVerbatim("HLTEventAnalyzerRAW") << " Last active module - label/type: " << moduleLabels[moduleIndex] << "/"
                                     << hltConfig_.moduleType(moduleLabels[moduleIndex]) << " [" << moduleIndex
                                     << " out of 0-" << (m - 1) << " on this path]" << endl;
  assert(moduleIndex < m);

  // Results from TriggerEventWithRefs product
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

  // Attention: must look only for modules actually run in this path
  // for this event!
  for (unsigned int j = 0; j <= moduleIndex; ++j) {
    const string& moduleLabel(moduleLabels[j]);
    const string moduleType(hltConfig_.moduleType(moduleLabel));
    // check whether the module is packed up in TriggerEventWithRef product
    const unsigned int filterIndex(triggerEventWithRefsHandle_->filterIndex(InputTag(moduleLabel, "", processName_)));
    if (filterIndex < triggerEventWithRefsHandle_->size()) {
      LogVerbatim("HLTEventAnalyzerRAW") << " Filter in slot " << j << " - label/type " << moduleLabel << "/"
                                         << moduleType << endl;
      LogVerbatim("HLTEventAnalyzerRAW") << " Filter packed up at: " << filterIndex << endl;
      LogVerbatim("HLTEventAnalyzerRAW") << "  Accepted objects:" << endl;

      triggerEventWithRefsHandle_->getObjects(filterIndex, photonIds_, photonRefs_);
      const unsigned int nPhotons(photonIds_.size());
      if (nPhotons > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   Photons: " << nPhotons << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nPhotons; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << photonIds_[i] << " " << photonRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, electronIds_, electronRefs_);
      const unsigned int nElectrons(electronIds_.size());
      if (nElectrons > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   Electrons: " << nElectrons << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nElectrons; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << electronIds_[i] << " " << electronRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, muonIds_, muonRefs_);
      const unsigned int nMuons(muonIds_.size());
      if (nMuons > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   Muons: " << nMuons << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nMuons; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW") << "   " << i << " " << muonIds_[i] << " " << muonRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, jetIds_, jetRefs_);
      const unsigned int nJets(jetIds_.size());
      if (nJets > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   Jets: " << nJets << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nJets; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW") << "   " << i << " " << jetIds_[i] << " " << jetRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, compositeIds_, compositeRefs_);
      const unsigned int nComposites(compositeIds_.size());
      if (nComposites > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   Composites: " << nComposites << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nComposites; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << compositeIds_[i] << " " << compositeRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, basemetIds_, basemetRefs_);
      const unsigned int nBaseMETs(basemetIds_.size());
      if (nBaseMETs > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   BaseMETs: " << nBaseMETs << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nBaseMETs; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << basemetIds_[i] << " " << basemetRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, calometIds_, calometRefs_);
      const unsigned int nCaloMETs(calometIds_.size());
      if (nCaloMETs > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   CaloMETs: " << nCaloMETs << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nCaloMETs; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << calometIds_[i] << " " << calometRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, pixtrackIds_, pixtrackRefs_);
      const unsigned int nPixTracks(pixtrackIds_.size());
      if (nPixTracks > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   PixTracks: " << nPixTracks << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nPixTracks; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << pixtrackIds_[i] << " " << pixtrackRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, l1emIds_, l1emRefs_);
      const unsigned int nL1EM(l1emIds_.size());
      if (nL1EM > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   L1EM: " << nL1EM << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nL1EM; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW") << "   " << i << " " << l1emIds_[i] << " " << l1emRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, l1muonIds_, l1muonRefs_);
      const unsigned int nL1Muon(l1muonIds_.size());
      if (nL1Muon > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   L1Muon: " << nL1Muon << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nL1Muon; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << l1muonIds_[i] << " " << l1muonRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, l1jetIds_, l1jetRefs_);
      const unsigned int nL1Jet(l1jetIds_.size());
      if (nL1Jet > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   L1Jet: " << nL1Jet << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nL1Jet; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW") << "   " << i << " " << l1jetIds_[i] << " " << l1jetRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, l1etmissIds_, l1etmissRefs_);
      const unsigned int nL1EtMiss(l1etmissIds_.size());
      if (nL1EtMiss > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   L1EtMiss: " << nL1EtMiss << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nL1EtMiss; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << l1etmissIds_[i] << " " << l1etmissRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, l1hfringsIds_, l1hfringsRefs_);
      const unsigned int nL1HfRings(l1hfringsIds_.size());
      if (nL1HfRings > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   L1HfRings: " << nL1HfRings << "  - the objects: # id 4 4" << endl;
        for (unsigned int i = 0; i != nL1HfRings; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW") << "   " << i << " " << l1hfringsIds_[i] << " "
                                             << l1hfringsRefs_[i]->hfEtSum(l1extra::L1HFRings::kRing1PosEta) << " "
                                             << l1hfringsRefs_[i]->hfEtSum(l1extra::L1HFRings::kRing1NegEta) << " "
                                             << l1hfringsRefs_[i]->hfEtSum(l1extra::L1HFRings::kRing2PosEta) << " "
                                             << l1hfringsRefs_[i]->hfEtSum(l1extra::L1HFRings::kRing2NegEta) << " "
                                             << l1hfringsRefs_[i]->hfBitCount(l1extra::L1HFRings::kRing1PosEta) << " "
                                             << l1hfringsRefs_[i]->hfBitCount(l1extra::L1HFRings::kRing1NegEta) << " "
                                             << l1hfringsRefs_[i]->hfBitCount(l1extra::L1HFRings::kRing2PosEta) << " "
                                             << l1hfringsRefs_[i]->hfBitCount(l1extra::L1HFRings::kRing2NegEta) << endl;
        }
      }

      /* Phase-2 */

      triggerEventWithRefsHandle_->getObjects(filterIndex, l1ttkmuIds_, l1ttkmuRefs_);
      const unsigned int nL1TTkMuons(l1ttkmuIds_.size());
      if (nL1TTkMuons > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   L1TTkMuons: " << nL1TTkMuons << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nL1TTkMuons; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << l1ttkmuIds_[i] << " " << l1ttkmuRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, l1ttkeleIds_, l1ttkeleRefs_);
      const unsigned int nL1TTkElectrons(l1ttkeleIds_.size());
      if (nL1TTkElectrons > 0) {
        LogVerbatim("HLTEventAnalyzerRAW")
            << "   L1TTkElectrons: " << nL1TTkElectrons << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nL1TTkElectrons; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << l1ttkeleIds_[i] << " " << l1ttkeleRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, l1ttkemIds_, l1ttkemRefs_);
      const unsigned int nL1TTkEMs(l1ttkemIds_.size());
      if (nL1TTkEMs > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   L1TTkEMs: " << nL1TTkEMs << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nL1TTkEMs; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << l1ttkemIds_[i] << " " << l1ttkemRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, l1tpfjetIds_, l1tpfjetRefs_);
      const unsigned int nL1TPFJets(l1tpfjetIds_.size());
      if (nL1TPFJets > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   L1TPFJets: " << nL1TPFJets << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nL1TPFJets; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << l1tpfjetIds_[i] << " " << l1tpfjetRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, l1tpftauIds_, l1tpftauRefs_);
      const unsigned int nL1TPFTaus(l1tpftauIds_.size());
      if (nL1TPFTaus > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   L1TPFTaus: " << nL1TPFTaus << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nL1TPFTaus; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << l1tpftauIds_[i] << " " << l1tpftauRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, l1thpspftauIds_, l1thpspftauRefs_);
      const unsigned int nL1THPSPFTaus(l1thpspftauIds_.size());
      if (nL1THPSPFTaus > 0) {
        LogVerbatim("HLTEventAnalyzerRAW")
            << "   L1THPSPFTaus: " << nL1THPSPFTaus << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nL1THPSPFTaus; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << l1thpspftauIds_[i] << " " << l1thpspftauRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, l1tpftrackIds_, l1tpftrackRefs_);
      const unsigned int nL1TPFTracks(l1tpftrackIds_.size());
      if (nL1TPFTracks > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   L1TPFTracks: " << nL1TPFTracks << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nL1TPFTracks; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW")
              << "   " << i << " " << l1tpftrackIds_[i] << " " << l1tpftrackRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, pfjetIds_, pfjetRefs_);
      const unsigned int nPFJets(pfjetIds_.size());
      if (nPFJets > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   PFJets: " << nPFJets << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nPFJets; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW") << "   " << i << " " << pfjetIds_[i] << " " << pfjetRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, pftauIds_, pftauRefs_);
      const unsigned int nPFTaus(pftauIds_.size());
      if (nPFTaus > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   PFTaus: " << nPFTaus << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nPFTaus; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW") << "   " << i << " " << pftauIds_[i] << " " << pftauRefs_[i]->pt() << endl;
        }
      }

      triggerEventWithRefsHandle_->getObjects(filterIndex, pfmetIds_, pfmetRefs_);
      const unsigned int nPfMETs(pfmetIds_.size());
      if (nPfMETs > 0) {
        LogVerbatim("HLTEventAnalyzerRAW") << "   PfMETs: " << nPfMETs << "  - the objects: # id pt" << endl;
        for (unsigned int i = 0; i != nPfMETs; ++i) {
          LogVerbatim("HLTEventAnalyzerRAW") << "   " << i << " " << pfmetIds_[i] << " " << pfmetRefs_[i]->pt() << endl;
        }
      }
    }
  }

  return;
}
