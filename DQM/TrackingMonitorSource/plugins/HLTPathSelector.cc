// system includes
#include <cassert>
#include <vector>
#include <string>
#include <map>

// user includes
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Common/interface/TriggerResultsByName.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

// ROOT includes
#include "TPRegexp.h"

class HLTPathSelector : public edm::stream::EDFilter<> {
public:
  explicit HLTPathSelector(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  bool filter(edm::Event&, edm::EventSetup const&) override;
  void endJob();

private:
  // module config parameters
  const bool verbose_;
  const std::string processName_;
  const std::vector<std::string> hltPathsOfInterest_;
  const edm::InputTag triggerResultsTag_;
  const edm::InputTag triggerEventTag_;
  const edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  const edm::EDGetTokenT<trigger::TriggerEvent> triggerEventToken_;

  HLTConfigProvider hltConfig_;

  std::map<std::string, unsigned int> hltPathsMap_;
  std::map<std::string, int> tmap_;
};

using namespace std;
using namespace edm;

void HLTPathSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("verbose", false);
  desc.add<std::string>("processName", std::string(""));
  desc.add<std::vector<std::string> >("hltPathsOfInterest", {});
  desc.addUntracked<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.addUntracked<edm::InputTag>("triggerEvent", edm::InputTag("hltTriggerSummaryAOD", "", "HLT"));
  descriptions.addWithDefaultLabel(desc);
}

HLTPathSelector::HLTPathSelector(const edm::ParameterSet& ps)
    : verbose_(ps.getUntrackedParameter<bool>("verbose", false)),
      processName_(ps.getParameter<std::string>("processName")),
      hltPathsOfInterest_(ps.getParameter<std::vector<std::string> >("hltPathsOfInterest")),
      triggerResultsTag_(
          ps.getUntrackedParameter<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "HLT"))),
      triggerEventTag_(
          ps.getUntrackedParameter<edm::InputTag>("triggerEvent", edm::InputTag("hltTriggerSummaryAOD", "", "HLT"))),
      triggerResultsToken_(consumes<edm::TriggerResults>(triggerResultsTag_)),
      triggerEventToken_(consumes<trigger::TriggerEvent>(triggerEventTag_)) {}

void HLTPathSelector::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed(true);
  if (hltConfig_.init(iRun, iSetup, processName_, changed)) {
    if (changed) {
      edm::LogInfo("HLTPathSelector") << "HLT initialised";
      hltConfig_.dump("PrescaleTable");
    }
    hltPathsMap_.clear();
    const unsigned int n(hltConfig_.size());
    const std::vector<std::string>& pathList = hltConfig_.triggerNames();
    for (const auto& path : pathList) {
      if (!hltPathsOfInterest_.empty()) {
        int nmatch = 0;
        for (const auto& kt : hltPathsOfInterest_)
          nmatch += TPRegexp(kt).Match(path);
        if (!nmatch)
          continue;
      }
      const unsigned int triggerIndex(hltConfig_.triggerIndex(path));
      // abort on invalid trigger name
      if (triggerIndex >= n) {
        edm::LogError("HLTPathSelector") << "path: " << path << " - not found!";
        continue;
      }
      hltPathsMap_[path] = triggerIndex;
    }
  } else
    edm::LogError("HLTPathSelector") << " config extraction failure with process name " << processName_;
}

bool HLTPathSelector::filter(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // get event products
  edm::Handle<edm::TriggerResults> triggerResultsHandle_;
  iEvent.getByToken(triggerResultsToken_, triggerResultsHandle_);
  if (!triggerResultsHandle_.isValid()) {
    edm::LogError("HLTPathSelector") << "Error in getting TriggerResults product from Event!";
    return false;
  }

  edm::Handle<trigger::TriggerEvent> triggerEventHandle_;
  iEvent.getByToken(triggerEventToken_, triggerEventHandle_);
  if (!triggerEventHandle_.isValid()) {
    edm::LogError("HLTPathSelector") << "Error in getting TriggerEvent product from Event!";
    return false;
  }
  // sanity check
  assert(triggerResultsHandle_->size() == hltConfig_.size());

  int flag = 0;
  for (auto const& it : hltPathsMap_) {
    const std::string path(it.first);
    const unsigned int triggerIndex(it.second);
    assert(triggerIndex == iEvent.triggerNames(*triggerResultsHandle_).triggerIndex(path));

    // Results from TriggerResults product
    if (verbose_)
      edm::LogInfo("HLTPathSelector") << " Trigger path <" << path << "> status:"
                                      << " WasRun=" << triggerResultsHandle_->wasrun(triggerIndex)
                                      << " Accept=" << triggerResultsHandle_->accept(triggerIndex)
                                      << " Error=" << triggerResultsHandle_->error(triggerIndex);

    if (triggerResultsHandle_->wasrun(triggerIndex) && triggerResultsHandle_->accept(triggerIndex)) {
      ++flag;
      if (tmap_.find(path) == tmap_.end())
        tmap_[path] = 1;
      else
        tmap_[path]++;
    }
  }
  if (flag > 0)
    return true;
  return false;
}

void HLTPathSelector::endJob() {
  edm::LogInfo("HLTPathSelector") << setw(32) << "HLT Path" << setw(9) << "ACCEPT";
  for (auto const& jt : tmap_)
    edm::LogInfo("HLTPathSelector") << setw(9) << jt.second;
}
// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPathSelector);
