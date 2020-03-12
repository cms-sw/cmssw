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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTrigger/HLTcore/interface/HLTEventAnalyzerAOD.h"

#include <cassert>

//
// constructors and destructor
//
HLTEventAnalyzerAOD::HLTEventAnalyzerAOD(const edm::ParameterSet& ps)
    : processName_(ps.getParameter<std::string>("processName")),
      triggerName_(ps.getParameter<std::string>("triggerName")),
      triggerResultsTag_(ps.getParameter<edm::InputTag>("triggerResults")),
      triggerResultsToken_(consumes<edm::TriggerResults>(triggerResultsTag_)),
      triggerEventTag_(ps.getParameter<edm::InputTag>("triggerEvent")),
      triggerEventToken_(consumes<trigger::TriggerEvent>(triggerEventTag_)),
      hltPrescaleProvider_(ps, consumesCollector(), *this) {
  using namespace std;
  using namespace edm;

  LogVerbatim("HLTEventAnalyzerAOD") << "HLTEventAnalyzerAOD configuration: " << endl
                                     << "   ProcessName = " << processName_ << endl
                                     << "   TriggerName = " << triggerName_ << endl
                                     << "   TriggerResultsTag = " << triggerResultsTag_.encode() << endl
                                     << "   TriggerEventTag = " << triggerEventTag_.encode() << endl;
}

HLTEventAnalyzerAOD::~HLTEventAnalyzerAOD() = default;

//
// member functions
//
void HLTEventAnalyzerAOD::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("processName", "HLT");
  desc.add<std::string>("triggerName", "@");
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<edm::InputTag>("triggerEvent", edm::InputTag("hltTriggerSummaryAOD", "", "HLT"));
  desc.add<unsigned int>("stageL1Trigger", 1);
  descriptions.add("hltEventAnalyzerAODDefault", desc);
}

void HLTEventAnalyzerAOD::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}

void HLTEventAnalyzerAOD::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  using namespace std;
  using namespace edm;

  bool changed(true);
  if (hltPrescaleProvider_.init(iRun, iSetup, processName_, changed)) {
    HLTConfigProvider const& hltConfig = hltPrescaleProvider_.hltConfigProvider();

    if (changed) {
      // check if trigger name in (new) config
      if (triggerName_ != "@") {  // "@" means: analyze all triggers in config
        const unsigned int n(hltConfig.size());
        const unsigned int triggerIndex(hltConfig.triggerIndex(triggerName_));
        if (triggerIndex >= n) {
          LogVerbatim("HLTEventAnalyzerAOD")
              << "HLTEventAnalyzerAOD::analyze:"
              << " TriggerName " << triggerName_ << " not available in (new) config!" << endl;
          LogVerbatim("HLTEventAnalyzerAOD") << "Available TriggerNames are: " << endl;
          hltConfig.dump("Triggers");
        }
      }
      hltConfig.dump("ProcessName");
      hltConfig.dump("GlobalTag");
      hltConfig.dump("TableName");
      hltConfig.dump("Streams");
      hltConfig.dump("Datasets");
      hltConfig.dump("PrescaleTable");
      hltConfig.dump("ProcessPSet");
    }
  } else {
    LogVerbatim("HLTEventAnalyzerAOD") << "HLTEventAnalyzerAOD::analyze:"
                                       << " config extraction failure with process name " << processName_ << endl;
  }
}

// ------------ method called to produce the data  ------------
void HLTEventAnalyzerAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace std;
  using namespace edm;

  LogVerbatim("HLTEventAnalyzerAOD") << endl;

  // get event products
  iEvent.getByToken(triggerResultsToken_, triggerResultsHandle_);
  if (!triggerResultsHandle_.isValid()) {
    LogVerbatim("HLTEventAnalyzerAOD")
        << "HLTEventAnalyzerAOD::analyze: Error in getting TriggerResults product from Event!" << endl;
    return;
  }
  iEvent.getByToken(triggerEventToken_, triggerEventHandle_);
  if (!triggerEventHandle_.isValid()) {
    LogVerbatim("HLTEventAnalyzerAOD")
        << "HLTEventAnalyzerAOD::analyze: Error in getting TriggerEvent product from Event!" << endl;
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
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  LogVerbatim("HLTEventAnalyzerAOD") << endl;

  HLTConfigProvider const& hltConfig = hltPrescaleProvider_.hltConfigProvider();

  const unsigned int n(hltConfig.size());
  const unsigned int triggerIndex(hltConfig.triggerIndex(triggerName));
  assert(triggerIndex == iEvent.triggerNames(*triggerResultsHandle_).triggerIndex(triggerName));

  // abort on invalid trigger name
  if (triggerIndex >= n) {
    LogVerbatim("HLTEventAnalyzerAOD") << "HLTEventAnalyzerAOD::analyzeTrigger: path " << triggerName << " - not found!"
                                       << endl;
    return;
  }

  const std::pair<int, int> prescales(hltPrescaleProvider_.prescaleValues(iEvent, iSetup, triggerName));
  LogVerbatim("HLTEventAnalyzerAOD") << "HLTEventAnalyzerAOD::analyzeTrigger: path " << triggerName << " ["
                                     << triggerIndex << "] "
                                     << "prescales L1T,HLT: " << prescales.first << "," << prescales.second << endl;
  const std::pair<std::vector<std::pair<std::string, int> >, int> prescalesInDetail(
      hltPrescaleProvider_.prescaleValuesInDetail(iEvent, iSetup, triggerName));
  std::ostringstream message;
  for (unsigned int i = 0; i < prescalesInDetail.first.size(); ++i) {
    message << " " << i << ":" << prescalesInDetail.first[i].first << "/" << prescalesInDetail.first[i].second;
  }
  LogVerbatim("HLTEventAnalyzerAOD") << "HLTEventAnalyzerAOD::analyzeTrigger: path " << triggerName << " ["
                                     << triggerIndex << "] " << endl
                                     << "prescales L1T: " << prescalesInDetail.first.size() << message.str() << endl
                                     << " prescale HLT: " << prescalesInDetail.second << endl;

  // modules on this trigger path
  const unsigned int m(hltConfig.size(triggerIndex));
  const vector<string>& moduleLabels(hltConfig.moduleLabels(triggerIndex));

  // Results from TriggerResults product
  LogVerbatim("HLTEventAnalyzerAOD") << " Trigger path status:"
                                     << " WasRun=" << triggerResultsHandle_->wasrun(triggerIndex)
                                     << " Accept=" << triggerResultsHandle_->accept(triggerIndex)
                                     << " Error =" << triggerResultsHandle_->error(triggerIndex) << endl;
  const unsigned int moduleIndex(triggerResultsHandle_->index(triggerIndex));
  LogVerbatim("HLTEventAnalyzerAOD") << " Last active module - label/type: " << moduleLabels[moduleIndex] << "/"
                                     << hltConfig.moduleType(moduleLabels[moduleIndex]) << " [" << moduleIndex
                                     << " out of 0-" << (m - 1) << " on this path]" << endl;
  assert(moduleIndex < m);

  // Results from TriggerEvent product - Attention: must look only for
  // modules actually run in this path for this event!
  for (unsigned int j = 0; j <= moduleIndex; ++j) {
    const string& moduleLabel(moduleLabels[j]);
    const string moduleType(hltConfig.moduleType(moduleLabel));
    // check whether the module is packed up in TriggerEvent product
    const unsigned int filterIndex(triggerEventHandle_->filterIndex(InputTag(moduleLabel, "", processName_)));
    if (filterIndex < triggerEventHandle_->sizeFilters()) {
      LogVerbatim("HLTEventAnalyzerAOD") << " 'L3' filter in slot " << j << " - label/type " << moduleLabel << "/"
                                         << moduleType << endl;
      const Vids& VIDS(triggerEventHandle_->filterIds(filterIndex));
      const Keys& KEYS(triggerEventHandle_->filterKeys(filterIndex));
      const size_type nI(VIDS.size());
      const size_type nK(KEYS.size());
      assert(nI == nK);
      const size_type n(max(nI, nK));
      LogVerbatim("HLTEventAnalyzerAOD") << "   " << n << " accepted 'L3' objects found: " << endl;
      const TriggerObjectCollection& TOC(triggerEventHandle_->getObjects());
      for (size_type i = 0; i != n; ++i) {
        const TriggerObject& TO(TOC[KEYS[i]]);
        LogVerbatim("HLTEventAnalyzerAOD") << "   " << i << " " << VIDS[i] << "/" << KEYS[i] << ": " << TO.id() << " "
                                           << TO.pt() << " " << TO.eta() << " " << TO.phi() << " " << TO.mass() << endl;
      }
    }
  }

  return;
}
