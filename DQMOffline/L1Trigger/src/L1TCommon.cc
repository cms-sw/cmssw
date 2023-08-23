#include "DQMOffline/L1Trigger/interface/L1TCommon.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

namespace dqmoffline {
  namespace l1t {

    std::vector<unsigned int> getTriggerIndices(const std::vector<std::string> &requestedTriggers,
                                                const std::vector<std::string> &triggersInEvent) {
      std::vector<unsigned int> triggerIndices;

      for (const auto &requestedTriggerName : requestedTriggers) {
        std::string name(requestedTriggerName);
        std::size_t wildcarPosition = name.find('*');
        if (wildcarPosition != std::string::npos) {
          // take everything up to the wildcard
          name = name.substr(0, wildcarPosition - 1);
        }

        unsigned int triggerIndex = 0;
        for (const auto &triggerName : triggersInEvent) {
          if (triggerName.find(name) != std::string::npos) {
            triggerIndices.push_back(triggerIndex);
            break;
          }
          ++triggerIndex;
        }
      }
      return triggerIndices;
    }

    std::vector<bool> getTriggerResults(const std::vector<unsigned int> &triggers,
                                        const edm::TriggerResults &triggerResults) {
      std::vector<bool> results;
      results.resize(triggers.size());

      for (unsigned int index = 0; index < triggers.size(); ++index) {
        if (triggers[index] >= triggerResults.size()) {
          results[index] = false;
          continue;
        }

        if (triggerResults.accept(triggers[index])) {
          results[index] = true;
        } else {
          results[index] = false;
        }
      }
      return results;
    }

    std::vector<unsigned int> getFiredTriggerIndices(const std::vector<unsigned int> &triggers,
                                                     const std::vector<bool> &triggerResults) {
      std::vector<unsigned int> results;
      // std::copy_if instead?
      for (unsigned int i = 0; i < triggerResults.size(); ++i) {
        if (triggerResults[i]) {
          results.push_back(triggers[i]);
        }
      }
      return results;
    }

    bool passesAnyTriggerFromList(const std::vector<unsigned int> &triggers,
                                  const edm::TriggerResults &triggerResults) {
      std::vector<bool> results = dqmoffline::l1t::getTriggerResults(triggers, triggerResults);
      if (std::count(results.begin(), results.end(), true) == 0) {
        return false;
      }
      return true;
    }

    trigger::TriggerObjectCollection getTriggerObjects(const std::vector<edm::InputTag> &hltFilters,
                                                       const trigger::TriggerEvent &triggerEvent) {
      trigger::TriggerObjectCollection triggerObjects = triggerEvent.getObjects();
      trigger::TriggerObjectCollection results;

      for (const auto &filter : hltFilters) {
        const unsigned filterIndex = triggerEvent.filterIndex(filter);

        if (filterIndex < triggerEvent.sizeFilters()) {
          const trigger::Keys &triggerKeys(triggerEvent.filterKeys(filterIndex));
          const size_t nTriggers = triggerEvent.filterIds(filterIndex).size();
          for (size_t i = 0; i < nTriggers; ++i) {
            results.push_back(triggerObjects[triggerKeys[i]]);
          }
        }
      }
      // sort by ET
      typedef trigger::TriggerObject trigObj;
      std::sort(results.begin(), results.end(), [](const trigObj &obj1, const trigObj &obj2) {
        return obj1.et() > obj2.et();
      });
      return results;
    }

    std::vector<edm::InputTag> getHLTFilters(const std::vector<unsigned int> &triggers,
                                             const HLTConfigProvider &hltConfig,
                                             const std::string triggerProcess) {
      std::vector<edm::InputTag> results;
      for (auto trigger : triggers) {
        // For some reason various modules now come *after* "hltBoolEnd"
        // Really just want module one index before "hltBoolEnd" - AWB 2022.09.28
        unsigned int moduleIndex = 999999;
        for (int ii = 0; ii < int(hltConfig.size(trigger)); ii++) {
          if (hltConfig.moduleLabels(trigger)[ii] == "hltBoolEnd") {
            moduleIndex = ii - 1;
            break;
          }
        }
        if (moduleIndex == 999999) {
          edm::LogError("L1TCommon") << " Found no module label in trigger " << trigger << std::endl;
          continue;
        }
        const std::vector<std::string> &modules(hltConfig.moduleLabels(trigger));
        const std::string &module(modules[moduleIndex]);
        edm::InputTag filterInputTag = edm::InputTag(module, "", triggerProcess);
        results.push_back(filterInputTag);
      }
      return results;
    }

    trigger::TriggerObjectCollection getMatchedTriggerObjects(double eta,
                                                              double phi,
                                                              double maxDeltaR,
                                                              const trigger::TriggerObjectCollection triggerObjects) {
      trigger::TriggerObjectCollection results;
      typedef trigger::TriggerObject trigObj;
      std::copy_if(
          triggerObjects.begin(),
          triggerObjects.end(),
          std::back_inserter(results),
          [eta, phi, maxDeltaR](const trigObj &obj) { return deltaR(obj.eta(), obj.phi(), eta, phi) < maxDeltaR; });
      return results;
    }

  }  // namespace l1t
}  // namespace dqmoffline
