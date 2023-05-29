#include "DQMOffline/Lumi/interface/TriggerTools.h"

#include "FWCore/Utilities/interface/RegexMatch.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <algorithm>

//--------------------------------------------------------------------------------------------------
void TriggerTools::initPathNames(const std::vector<std::string>& triggerNames) {
  /*
        init HLT path every run (e.g. versions can change)
    */
  edm::LogVerbatim("TriggerTools") << "TriggerTools::initPathNames initHLT";
  for (auto& iRec : records) {
    iRec.hltPathName = "";
    iRec.hltPathIndex = (unsigned int)-1;
    const std::string pattern = iRec.hltPattern;
    if (edm::is_glob(pattern)) {  // handle pattern with wildcards (*,?)
      std::vector<std::vector<std::string>::const_iterator> matches = edm::regexMatch(triggerNames, pattern);
      if (matches.empty()) {
        edm::LogWarning("ZCounting") << "requested pattern [" << pattern << "] does not match any HLT paths";
      } else {
        for (auto const& match : matches) {
          iRec.hltPathName = *match;
        }
      }
    } else {  // take full HLT path name given
      iRec.hltPathName = pattern;
    }
  }
}

//--------------------------------------------------------------------------------------------------
void TriggerTools::initHLTObjects(const HLTConfigProvider& hltConfigProvider_) {
  /*
        execture each run to initialize the last filter of each trigger corresponding to the corresponding object that has fired the trigger
    */
  edm::LogVerbatim("TriggerTools") << "TriggerTools::initHLTObjects initHLTObjects";
  const std::vector<std::string>& triggerNames(hltConfigProvider_.triggerNames());

  initPathNames(triggerNames);

  for (auto& iRec : records) {
    std::vector<std::string> hltFiltersWithTags_;

    for (auto const& iPathName : triggerNames) {
      if (iPathName != iRec.hltPathName) {
        continue;
      }
      edm::LogVerbatim("TriggerTools") << "TriggerTools::initHLTObjects trigger name: " << iPathName;

      iRec.hltPathIndex = hltConfigProvider_.triggerIndex(iPathName);

      auto const& moduleLabels(hltConfigProvider_.moduleLabels(iRec.hltPathIndex));

      for (int idx = moduleLabels.size() - 1; idx >= 0; --idx) {
        auto const& moduleLabel(moduleLabels.at(idx));

        auto const& moduleEDMType(hltConfigProvider_.moduleEDMType(moduleLabel));
        if (moduleEDMType != "EDFilter") {
          continue;
        }

        auto const& moduleType(hltConfigProvider_.moduleType(moduleLabel));
        if ((moduleType == "HLTTriggerTypeFilter") or (moduleType == "HLTBool") or (moduleType == "HLTPrescaler")) {
          continue;
        }

        if (!hltConfigProvider_.saveTags(moduleLabel)) {
          continue;
        }
        edm::LogVerbatim("TriggerTools") << "TriggerTools::initHLTObjects new hlt object name: " << moduleLabel;

        iRec.hltObjName = moduleLabel;
        break;
      }
      break;
    }

    if (iRec.hltPathIndex == (unsigned int)-1) {
      edm::LogWarning("TriggerTools") << "TriggerTools::initHLTObjects hltPathIndex has not been found for: "
                                      << iRec.hltPattern << std::endl;
      continue;
    }
  }
}

//--------------------------------------------------------------------------------------------------
void TriggerTools::readEvent(const edm::Event& iEvent) {
  /*
        execture each event to load trigger objects
    */

  LogDebug("TriggerTools") << "TriggerTools::readEvent";

  iEvent.getByToken(fHLTTag_token, hTrgRes);
  if (!hTrgRes.isValid()) {
    edm::LogWarning("TriggerTools") << "TriggerTools::readEvent No valid trigger result product found";
  }

  iEvent.getByToken(fHLTObjTag_token, hTrgEvt);
  if (!hTrgEvt.isValid()) {
    edm::LogWarning("TriggerTools") << "TriggerTools::readEvent No valid trigger event product found";
  }

  triggerBits.reset();
  for (unsigned int i = 0; i < records.size(); i++) {
    if (records.at(i).hltPathIndex == (unsigned int)-1) {
      LogDebug("TriggerTools") << "TriggerTools::readEvent hltPathIndex has not been set" << std::endl;
      continue;
    }
    if (hTrgRes->accept(records.at(i).hltPathIndex)) {
      triggerBits[i] = true;
    }
  }
  LogDebug("TriggerTools") << "TriggerTools::readEvent bitset = " << triggerBits[1] << triggerBits[0];
}

//--------------------------------------------------------------------------------------------------
bool TriggerTools::pass() const {
  /*
        check if the event passed any of the initialized triggers
    */

  return triggerBits != 0;
}

//--------------------------------------------------------------------------------------------------
bool TriggerTools::passObj(const double eta, const double phi) const {
  /*
        check if the object is matched to any trigger of the initialized triggers, and that this trigger is passed
    */

  for (unsigned int i = 0; i < records.size(); i++) {
    const std::string& filterName = records.at(i).hltObjName;

    edm::InputTag filterTag(filterName, "", "HLT");
    // filterIndex must be less than the size of trgEvent or you get a CMSException: _M_range_check
    if (hTrgEvt->filterIndex(filterTag) < hTrgEvt->sizeFilters()) {
      const trigger::TriggerObjectCollection& toc(hTrgEvt->getObjects());
      const trigger::Keys& keys(hTrgEvt->filterKeys(hTrgEvt->filterIndex(filterTag)));

      for (unsigned int hlto = 0; hlto < keys.size(); hlto++) {
        trigger::size_type hltf = keys[hlto];
        const trigger::TriggerObject& tobj(toc[hltf]);
        if (reco::deltaR2(eta, phi, tobj.eta(), tobj.phi()) < DRMAX) {
          return true;
        }
      }
    }
  }
  return false;
}