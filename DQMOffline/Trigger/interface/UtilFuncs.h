#ifndef DQMOffline_Trigger_UtilFuncs_h
#define DQMOffline_Trigger_UtilFuncs_h

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Math/interface/deltaR.h"

namespace hltdqm {
  inline bool passTrig(const float objEta,
                       float objPhi,
                       const trigger::TriggerEvent& trigEvt,
                       const std::string& filterName,
                       const std::string& processName) {
    constexpr float kMaxDR2 = 0.1 * 0.1;

    edm::InputTag filterTag(filterName, "", processName);
    trigger::size_type filterIndex = trigEvt.filterIndex(filterTag);
    if (filterIndex < trigEvt.sizeFilters()) {  //check that filter is in triggerEvent
      const trigger::Keys& trigKeys = trigEvt.filterKeys(filterIndex);
      const trigger::TriggerObjectCollection& trigObjColl(trigEvt.getObjects());
      for (unsigned short trigKey : trigKeys) {
        const trigger::TriggerObject& trigObj = trigObjColl[trigKey];
        if (reco::deltaR2(trigObj.eta(), trigObj.phi(), objEta, objPhi) < kMaxDR2)
          return true;
      }
    }
    return false;
  }

  //empty filters is auto pass
  inline bool passTrig(const float objEta,
                       float objPhi,
                       const trigger::TriggerEvent& trigEvt,
                       const std::vector<std::string>& filterNames,
                       bool orFilters,
                       const std::string& processName) {
    if (orFilters) {
      if (filterNames.empty())
        return true;  //auto pass if empty filters
      for (auto& filterName : filterNames) {
        if (passTrig(objEta, objPhi, trigEvt, filterName, processName) == true)
          return true;
      }
      return false;
    } else {
      for (auto& filterName : filterNames) {
        if (passTrig(objEta, objPhi, trigEvt, filterName, processName) == false)
          return false;
      }
      return true;
    }
  }

  //inspired by https://github.com/cms-sw/cmssw/blob/fc4f8bbe1258790e46e2d554aacea15c3e5d9afa/HLTrigger/HLTfilters/src/HLTHighLevel.cc#L124-L165
  //triggers are ORed together
  //empty pattern is auto pass
  inline bool passTrig(const std::string& trigPattern,
                       const edm::TriggerNames& trigNames,
                       const edm::TriggerResults& trigResults) {
    if (trigPattern.empty())
      return true;

    std::vector<std::string> trigNamesToPass;
    if (edm::is_glob(trigPattern)) {
      //matches is vector of string iterators
      const auto& matches = edm::regexMatch(trigNames.triggerNames(), trigPattern);
      for (auto& name : matches)
        trigNamesToPass.push_back(*name);
    } else {
      trigNamesToPass.push_back(trigPattern);  //not a pattern, much be a path
    }
    for (auto& trigName : trigNamesToPass) {
      size_t pathIndex = trigNames.triggerIndex(trigName);
      if (pathIndex < trigResults.size() && trigResults.accept(pathIndex))
        return true;
    }

    return false;
  }

}  // namespace hltdqm

#endif
