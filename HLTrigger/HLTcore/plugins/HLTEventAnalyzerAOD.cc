/** \class HLTEventAnalyzerAOD
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
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTrigger/HLTcore/interface/HLTEventAnalyzerAOD.h"

#include <cassert>

//
// constructor
//
HLTEventAnalyzerAOD::HLTEventAnalyzerAOD(const edm::ParameterSet& ps)
    : processName_(ps.getParameter<std::string>("processName")),
      triggerName_(ps.getParameter<std::string>("triggerName")),
      triggerResultsTag_(ps.getParameter<edm::InputTag>("triggerResults")),
      triggerResultsToken_(consumes<edm::TriggerResults>(triggerResultsTag_)),
      triggerEventTag_(ps.getParameter<edm::InputTag>("triggerEvent")),
      triggerEventToken_(consumes<trigger::TriggerEvent>(triggerEventTag_)),
      verbose_(ps.getParameter<bool>("verbose")),
      hltPrescaleProvider_(ps, consumesCollector(), *this) {
  LOG(logMsgType_) << logMsgType_ << " configuration:\n"
                   << "   ProcessName = " << processName_ << "\n"
                   << "   TriggerName = " << triggerName_ << "\n"
                   << "   TriggerResultsTag = " << triggerResultsTag_.encode() << "\n"
                   << "   TriggerEventTag = " << triggerEventTag_.encode();
}

//
// member functions
//
void HLTEventAnalyzerAOD::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("processName", "HLT");
  desc.add<std::string>("triggerName", "@")
      ->setComment("name of trigger Path to consider (use \"@\" to consider all Paths)");
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("triggerEvent", edm::InputTag("hltTriggerSummaryAOD", "", "HLT"));
  desc.add<unsigned int>("stageL1Trigger", 1);
  desc.add<bool>("verbose", true)->setComment("enable verbose mode");
  descriptions.add("hltEventAnalyzerAODDefault", desc);
}

void HLTEventAnalyzerAOD::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed(true);
  if (hltPrescaleProvider_.init(iRun, iSetup, processName_, changed)) {
    HLTConfigProvider const& hltConfig = hltPrescaleProvider_.hltConfigProvider();

    if (changed) {
      // check if trigger name in (new) config
      if (triggerName_ != "@") {  // "@" means: analyze all triggers in config
        const unsigned int n(hltConfig.size());
        const unsigned int triggerIndex(hltConfig.triggerIndex(triggerName_));
        if (triggerIndex >= n) {
          LOG(logMsgType_) << logMsgType_ << "::beginRun: TriggerName " << triggerName_
                           << " not available in (new) config!";
          LOG(logMsgType_) << "Available TriggerNames are:";
          hltConfig.dump("Triggers");
        }
      }
      // in verbose mode, print process info to stdout
      if (verbose_) {
        hltConfig.dump("ProcessName");
        hltConfig.dump("GlobalTag");
        hltConfig.dump("TableName");
        hltConfig.dump("Streams");
        hltConfig.dump("Datasets");
        hltConfig.dump("PrescaleTable");
        hltConfig.dump("ProcessPSet");
      }
    }
  } else {
    LOG(logMsgType_) << logMsgType_ << "::beginRun: config extraction failure with process name " << processName_;
  }
}

void HLTEventAnalyzerAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get event products
  iEvent.getByToken(triggerResultsToken_, triggerResultsHandle_);
  if (!triggerResultsHandle_.isValid()) {
    LOG(logMsgType_) << logMsgType_ << "::analyze: Error in getting TriggerResults product from Event!";
    return;
  }
  iEvent.getByToken(triggerEventToken_, triggerEventHandle_);
  if (!triggerEventHandle_.isValid()) {
    LOG(logMsgType_) << logMsgType_ << "::analyze: Error in getting TriggerEvent product from Event!";
    return;
  }

  HLTConfigProvider const& hltConfig = hltPrescaleProvider_.hltConfigProvider();

  // sanity check
  assert(triggerResultsHandle_->size() == hltConfig.size());

  // analyze this event for the triggers requested
  if (triggerName_ == "@") {
    const unsigned int n(hltConfig.size());
    for (unsigned int i = 0; i != n; ++i) {
      analyzeTrigger(iEvent, iSetup, hltConfig.triggerName(i));
    }
  } else {
    analyzeTrigger(iEvent, iSetup, triggerName_);
  }

  return;
}

void HLTEventAnalyzerAOD::analyzeTrigger(const edm::Event& iEvent,
                                         const edm::EventSetup& iSetup,
                                         const std::string& triggerName) {
  HLTConfigProvider const& hltConfig = hltPrescaleProvider_.hltConfigProvider();

  const unsigned int n(hltConfig.size());
  const unsigned int triggerIndex(hltConfig.triggerIndex(triggerName));
  assert(triggerIndex == iEvent.triggerNames(*triggerResultsHandle_).triggerIndex(triggerName));

  // abort on invalid trigger name
  if (triggerIndex >= n) {
    LOG(logMsgType_) << logMsgType_ << "::analyzeTrigger: path " << triggerName << " - not found!";
    return;
  }

  auto const prescales = hltPrescaleProvider_.prescaleValues<double>(iEvent, iSetup, triggerName);

  LOG(logMsgType_) << logMsgType_ << "::analyzeTrigger: path " << triggerName << " [" << triggerIndex
                   << "] prescales L1T,HLT: " << prescales.first << "," << prescales.second;

  auto const prescalesInDetail = hltPrescaleProvider_.prescaleValuesInDetail<double>(iEvent, iSetup, triggerName);
  {
    LOG logtmp(logMsgType_);
    logtmp << logMsgType_ << "::analyzeTrigger: path " << triggerName << " [" << triggerIndex
           << "]\n prescales L1T: " << prescalesInDetail.first.size();
    for (size_t idx = 0; idx < prescalesInDetail.first.size(); ++idx) {
      logtmp << " " << idx << ":" << prescalesInDetail.first[idx].first << "/" << prescalesInDetail.first[idx].second;
    }
    logtmp << "\n prescale HLT: " << prescalesInDetail.second;
  }

  // results from TriggerResults product
  LOG(logMsgType_) << " Trigger path status:"
                   << " WasRun=" << triggerResultsHandle_->wasrun(triggerIndex)
                   << " Accept=" << triggerResultsHandle_->accept(triggerIndex)
                   << " Error=" << triggerResultsHandle_->error(triggerIndex);

  // modules on this trigger path
  const unsigned int m(hltConfig.size(triggerIndex));
  const std::vector<std::string>& moduleLabels(hltConfig.moduleLabels(triggerIndex));
  assert(m == moduleLabels.size());

  // skip empty Paths
  if (m == 0) {
    LOG(logMsgType_) << logMsgType_ << "::analyzeTrigger: path " << triggerName << " [" << triggerIndex
                     << "] is empty!";
    return;
  }

  // index of last module executed in this Path
  const unsigned int moduleIndex(triggerResultsHandle_->index(triggerIndex));
  assert(moduleIndex < m);

  LOG(logMsgType_) << " Last active module - label/type: " << moduleLabels[moduleIndex] << "/"
                   << hltConfig.moduleType(moduleLabels[moduleIndex]) << " [" << moduleIndex << " out of 0-" << (m - 1)
                   << " on this path]";

  // results from TriggerEvent product
  // Attention: must look only for modules actually run in this path for this event!
  for (unsigned int j = 0; j <= moduleIndex; ++j) {
    const std::string& moduleLabel(moduleLabels[j]);
    const std::string moduleType(hltConfig.moduleType(moduleLabel));

    // check whether the module is packed up in TriggerEvent product
    const unsigned int filterIndex(triggerEventHandle_->filterIndex(edm::InputTag(moduleLabel, "", processName_)));
    if (filterIndex < triggerEventHandle_->sizeFilters()) {
      LOG(logMsgType_) << " 'L3' filter in slot " << j << " - label/type " << moduleLabel << "/" << moduleType;

      const trigger::Vids& VIDS(triggerEventHandle_->filterIds(filterIndex));
      const trigger::Keys& KEYS(triggerEventHandle_->filterKeys(filterIndex));
      const trigger::size_type nI(VIDS.size());
      const trigger::size_type nK(KEYS.size());
      assert(nI == nK);

      LOG(logMsgType_) << "   " << nI << " accepted 'L3' objects found: ";
      const trigger::TriggerObjectCollection& TOC(triggerEventHandle_->getObjects());
      for (trigger::size_type idx = 0; idx < nI; ++idx) {
        const trigger::TriggerObject& TO(TOC[KEYS[idx]]);
        LOG(logMsgType_) << "   " << idx << " " << VIDS[idx] << "/" << KEYS[idx] << ": " << TO.id() << " " << TO.pt()
                         << " " << TO.eta() << " " << TO.phi() << " " << TO.mass();
      }
    }
  }

  return;
}
