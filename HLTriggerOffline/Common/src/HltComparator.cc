// Originally written by James Jackson
// modified by Peter Wittich

// user include files
#include "FWCore/Common/interface/TriggerNames.h"
#include "HLTriggerOffline/Common/interface/HltComparator.h"
//#include "FWCore/Utilities/interface/Exception.h"

//#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <TH1.h>
#include <iostream>
#include <string>
#include <vector>

typedef std::vector<std::string> StringCollection;

// types of outcomes possible.
// only some are errors
enum {
  kOnOffPass = 0,
  kOnOffFail,
  kOnPassOffFail,
  kOnFailOffPass,
  kOnOffError,
  kOnRunOffError,
  kOnErrorOffRun,
  kOnRunOffNot,
  kOnNotOffRun,
  kOnOffNot
};

// Analyser constructor
HltComparator::HltComparator(const edm::ParameterSet &iConfig)
    : hltOnlineResults_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("OnlineResults"))),
      hltOfflineResults_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("OfflineResults"))),
      init_(false),
      verbose_(iConfig.getUntrackedParameter<bool>("verbose")),
      skipPathList_(iConfig.getUntrackedParameter<std::vector<std::string>>("skipPaths")),
      usePathList_(iConfig.getUntrackedParameter<std::vector<std::string>>("usePaths")) {
  // std::cout << " HERE I AM " << std::endl;
  produces<StringCollection>("failedTriggerDescription");
  // std::cout << " HERE I GO " << std::endl;
  usesResource(TFileService::kSharedResource);
}

HltComparator::~HltComparator() {}

// Initialises online --> offline trigger bit mappings and histograms
void HltComparator::initialise(const edm::TriggerResults &onlineResults,
                               const edm::TriggerResults &offlineResults,
                               edm::Event &e) {
  init_ = true;

  // Get trigger names
  const edm::TriggerNames &onlineTriggerNames = e.triggerNames(onlineResults);
  const edm::TriggerNames &offlineTriggerNames = e.triggerNames(offlineResults);
  onlineActualNames_ = onlineTriggerNames.triggerNames();
  offlineActualNames_ = offlineTriggerNames.triggerNames();
  numTriggers_ = onlineActualNames_.size();

  // do we need to throw? I guess the whole job is crap if this happens.
  // sort of assumes we're the only game in town.
  if (numTriggers_ != offlineActualNames_.size()) {
    throw cms::Exception("IncorrectTriggers") << "Online had " << numTriggers_ << "triggers, "
                                              << "Offline had " << offlineActualNames_.size() << "triggers";
  }

  // Create bit mappings
  std::map<std::string, unsigned int> offlineNameBitMap;
  for (unsigned int i = 0; i < numTriggers_; ++i) {
    offlineNameBitMap[offlineActualNames_[i]] = i;
  }
  for (unsigned int i = 0; i < numTriggers_; ++i) {
    // Find offline position for fixed online bit
    std::map<std::string, unsigned int>::iterator it = offlineNameBitMap.find(onlineActualNames_[i]);
    if (it != offlineNameBitMap.end()) {
      onlineToOfflineBitMappings_.push_back(it->second);
    } else {
      throw cms::Exception("IncorrectTriggers") << "Online trigger path " << onlineActualNames_[i]
                                                << " not found in Offline "
                                                   "processing";
    }
  }

  // Create histograms
  edm::Service<TFileService> fs;
  for (std::vector<std::string>::iterator it = onlineActualNames_.begin(); it != onlineActualNames_.end(); ++it) {
    // Bin descriptions: OnOfPass, OnOffFail, OnPassOffFail, OnFailOffPass,
    // OnOffError, OnRunOffError, OnErrorOffRun, OnRunOffNot OnNotOffRun
    // OnNotOffNot
    TH1F *h = fs->make<TH1F>(it->c_str(), it->c_str(), 10, 0, 10);
    TAxis *a = h->GetXaxis();
    a->SetBinLabel(1, "OnPass_OffPass");
    a->SetBinLabel(2, "OnFail_OffFail");
    a->SetBinLabel(3, "OnPass_OffFail");
    a->SetBinLabel(4, "OnFail_OffPass");
    a->SetBinLabel(5, "OnError_OffError");
    a->SetBinLabel(6, "OnRun_OffError");
    a->SetBinLabel(7, "OnError_OffRun");
    a->SetBinLabel(8, "OnRun_OffNotRun");
    a->SetBinLabel(9, "OnNotRun_OffRun");
    a->SetBinLabel(10, "OnNotRun_OffNotRun");
    comparisonHists_.push_back(h);
  }
}

// Format a comparison result
std::string HltComparator::formatResult(const unsigned int i) {
  switch (i) {
    case 0:
      return std::string("OnPass_OffPass");
      break;
    case 1:
      return std::string("OnFail_OffFail");
      break;
    case 2:
      return std::string("OnPass_OffFail");
      break;
    case 3:
      return std::string("OnFail_OffPass");
      break;
    case 4:
      return std::string("OnError_OffError");
      break;
    case 5:
      return std::string("OnRun_OffError");
      break;
    case 6:
      return std::string("OnError_OffRun");
      break;
    case 7:
      return std::string("OnRun_OffNotRun");
      break;
    case 8:
      return std::string("OnNotRun_OffRun");
      break;
    case 9:
      return std::string("OnNotRun_OffNotRun");
      break;
  }
  return std::string("CODE NOT KNOWN");
}

bool HltComparator::filter(edm::Event &event, const edm::EventSetup &iSetup) {
  // std::cout << "top of the filter " << std::endl;
  // Get trigger results
  edm::Handle<edm::TriggerResults> onlineResults;
  edm::Handle<edm::TriggerResults> offlineResults;
  event.getByToken(hltOnlineResults_, onlineResults);
  event.getByToken(hltOfflineResults_, offlineResults);

  std::unique_ptr<StringCollection> resultDescription(new StringCollection);

  // Initialise comparator if required
  if (!init_) {
    initialise(*onlineResults, *offlineResults, event);
  }

  // Perform trigger checks
  bool hasDisagreement = false;
  for (unsigned int i = 0; i < numTriggers_; ++i) {
    unsigned int offlineTriggerBit = onlineToOfflineBitMappings_[i];

    bool onRun = onlineResults->wasrun(i);
    bool offRun = offlineResults->wasrun(offlineTriggerBit);
    bool onAccept = onlineResults->accept(i);
    bool offAccept = offlineResults->accept(offlineTriggerBit);
    bool onError = onlineResults->error(i);
    bool offError = offlineResults->error(offlineTriggerBit);

    int result = -1;
    if (onError || offError) {
      if (onError && offError) {
        result = 4;
      } else if (onError) {
        result = 6;
      } else {
        result = 5;
      }
    } else if ((!onRun) || (!offRun)) {
      if ((!onRun) && (!offRun)) {
        result = 9;
      } else if (!onRun) {
        result = 8;
      } else {
        result = 7;
      }
    } else {
      if (onAccept && offAccept) {
        result = 0;
      } else if ((!onAccept) && (!offAccept)) {
        result = 1;
      } else if (onAccept) {
        result = 2;
      } else {
        result = 3;
      }
    }

    // Fill the results histogram
    comparisonHists_[i]->Fill(result);

    // if the online-offline comparison results in a failure, we
    // want to send the result to a special stream. Hence we _pass_ the filter.
    // If it all worked as expected the filter fails and the event doesn't go
    // to the output stream.
    if ((result == kOnPassOffFail) || (result == kOnFailOffPass) || (result == kOnRunOffError) ||
        (result == kOnErrorOffRun) || (result == kOnRunOffNot) || (result == kOnNotOffRun)) {
      // is this one we should ignore? check the skip list
      if (verbose()) {
        std::cout << "Found disagreemenet " << result << ", name is " << onlineActualNames_[i] << std::endl;
      }
      std::ostringstream desc;
      desc << onlineActualNames_[i] << ":" << formatResult(result);
      resultDescription->push_back(desc.str());
      if (std::find(skipPathList_.begin(), skipPathList_.end(), onlineActualNames_[i]) == skipPathList_.end()) {
        if (!usePathList_.empty()) {
          // only use specified paths to debug
          if (std::find(usePathList_.begin(), usePathList_.end(), onlineActualNames_[i]) != usePathList_.end())
            hasDisagreement = true;
        } else
          hasDisagreement = true;
      }
    }

    // Record the trigger error code
    // I think this should be result > 2? (pw)
    if (verbose() && (result > 1)) {
      std::cout << "HLT-Compare: Event " << event.id().event() << " Path " << onlineActualNames_[i] << " "
                << formatResult(result) << std::endl;
#ifdef NOTDEF
      triggerComparisonErrors_[event.id().event()][onlineActualNames_[i]] = result;
#endif  // NOTDEF
    }
  }

  // std::cout << " HERE I STAY " << std::endl;
  event.put(std::move(resultDescription), "failedTriggerDescription");
  // std::cout << " HERE I WENT " << std::endl;

  if (hasDisagreement)
    return true;
  else
    return false;
}

void HltComparator::beginJob() {}

// Print the trigger results
void HltComparator::endJob() {
#ifdef NOTDEF
  std::cout << "HLT-Compare ---------- Trigger Comparison Summary ----------" << std::endl;
  std::cout << "HLT-Compare  The following events had trigger mismatches:" << std::endl;
  std::map<unsigned int, std::map<std::string, unsigned int>>::iterator it;
  for (it = triggerComparisonErrors_.begin(); it != triggerComparisonErrors_.end(); ++it) {
    std::cout << "HLT-Compare  Event: " << it->first << std::endl;
    std::map<std::string, unsigned int>::iterator jt;
    for (jt = it->second.begin(); jt != it->second.end(); ++jt) {
      std::cout << "HLT-Compare    Path: " << jt->first << " : " << formatResult(jt->second) << std::endl;
    }
  }
  std::cout << "HLT-Compare ------------ End Trigger Comparison ------------" << std::endl;
#endif  // NOTDEF
}
