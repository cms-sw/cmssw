#include <string>

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

namespace dqmoffline {
  namespace l1t {

    std::vector<unsigned int> getTriggerIndices(const std::vector<std::string> &requestedTriggers,
                                                const std::vector<std::string> &triggersInEvent);

    std::vector<bool> getTriggerResults(const std::vector<unsigned int> &triggers,
                                        const edm::TriggerResults &triggerResults);

    std::vector<unsigned int> getFiredTriggerIndices(const std::vector<unsigned int> &triggers,
                                                     const std::vector<bool> &triggerResults);

    bool passesAnyTriggerFromList(const std::vector<unsigned int> &triggers, const edm::TriggerResults &triggerResults);

    trigger::TriggerObjectCollection getTriggerObjects(const std::vector<edm::InputTag> &hltFilters,
                                                       const trigger::TriggerEvent &triggerEvent);

    std::vector<edm::InputTag> getHLTFilters(const std::vector<unsigned int> &triggers,
                                             const HLTConfigProvider &hltConfig,
                                             const std::string triggerProcess);

    trigger::TriggerObjectCollection getMatchedTriggerObjects(double eta,
                                                              double phi,
                                                              double maxDeltaR,
                                                              const trigger::TriggerObjectCollection triggerObjects);
  }  // namespace l1t
}  // namespace dqmoffline
